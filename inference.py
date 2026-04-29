import sys
import os
import yaml
import time
import traceback
import gc
gc.enable()
import librosa
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from extra_utils import hf_spaces_gpu, dw_file, extra_clear_torch_cache, nuclear_clear_model, emergency_ram_clear
import torch
import rich
nn = torch.nn
import json
from tqdm import tqdm
import numpy as np
from typing import Literal, Optional, List, Tuple, Any, Dict
from ml_collections import ConfigDict
from omegaconf import OmegaConf
import gradio as gr
from audio import read, write, output_formats, subtractor, check, easy_resampler, ensemble_types, ensemble, multiread, get_audio_files_from_list, stereo_to_mono
from args_parser import parse_separator_args, tobool
from namer import Namer
from i18n import _i18n
import contextlib

class PathNotExist(Exception): pass
class PathsNotExist(Exception): pass
class PathNotSpecified(Exception): pass
class PathsNotSpecified(Exception): pass
class FileIsNotAudio(Exception): pass
class FilesIsNotAudio(Exception): pass
class MixNotFound(Exception): pass
class MixIsEmpty(Exception): pass
class UnknownModelType(Exception): pass
class DemixError(Exception): pass
class ConfigNotLoaded(Exception): pass
class ModelNotLoaded(Exception): pass
class ModelStateDictError(Exception): pass

HAS_OLD_AMP = False
if hasattr(torch, "cuda"):
    if hasattr(torch.cuda, "amp"):
        if hasattr(torch.cuda.amp, "autocast"):
            HAS_OLD_AMP = True
HAS_NEW_AMP = False
if hasattr(torch, "amp"):
    if hasattr(torch.amp, "autocast"):
        HAS_NEW_AMP = True

def get_autocast_context(device_type="cuda", enabled=True):
    if HAS_NEW_AMP:
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    elif HAS_OLD_AMP:
        return torch.cuda.amp.autocast(enabled=enabled)
    else:
        # Если AMP не поддерживается вообще
        return contextlib.nullcontext() # Или пустой контекст

def _getWindowingArray(window_size: int, fade_size: int) -> torch.Tensor:
    """
    Создать массив окна для плавного склеивания
    
    Args:
        window_size: Размер окна
        fade_size: Размер зоны затухания
    
    Returns:
        Массив окна
    """
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)

    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window

base_params = {
    "sec": {
        "type": "float",
        "component": "number",
        "minimum": 1,
        "maximum": 30,
        "step": 0.1,
        "default": 7,
        "info": "separation_segment_size_info"
    },
    "size": {
        "type": "int",
        "component": "slider",
        "minimum": 128,
        "maximum": 1024,
        "step": 128,
        "default": 256,
        "info": "separation_segment_size_info"
    },
    "wsize": {
        "type": "int",
        "component": "slider",
        "minimum": 320,
        "maximum": 1024,
        "step": 64,
        "default": 512,
        "info": "separation_window_size_info"
    },
    "hop": {
        "type": "int",
        "component": "slider",
        "minimum": 512,
        "maximum": 2048,
        "step": 512,
        "default": 1024,
        "info": "separation_hop_info"
    },
    "overlap": {
        "type": "int",
        "component": "slider",
        "minimum": 1,
        "maximum": 16,
        "step": 1,
        "default": 2,
        "info": "separation_overlap_info"
    },
    "batch": {
        "type": "int",
        "component": "slider",
        "minimum": 1,
        "maximum": 16,
        "step": 1,
        "default": 1,
        "info": "separation_batch_size_info"
    },
    "threshold" : {
        "type": "float",
        "component": "slider",
        "minimum": 0.1,
        "maximum": 0.3,
        "step": 0.1,
        "default": 0.2,
    },
    "aggression": {
        "type": "int",
        "component": "slider",
        "minimum": 0,
        "maximum": 100,
        "step": 1,
        "default": 5,
        "info": "separation_aggresion_info"
    },
    "enable": {
        "type": "bool",
        "component": "checkbox",
        "default": True
    },
    "disable": {
        "type": "bool",
        "component": "checkbox",
        "default": False
    },
}

add_params = {
    "mdxc": {
        "mdxc_segment_size": base_params["size"],
        "mdxc_batch_size": base_params["batch"],
        "mdxc_overlap": base_params["overlap"],
        "mdxc_denoise": base_params["disable"],
        "mdxc_override_segment": base_params["disable"]
    },
    "demucs": {
        "demucs_segment": base_params["sec"],
        "demucs_batch_size": base_params["batch"],
        "demucs_overlap": base_params["overlap"], 
        "demucs_denoise": base_params["disable"],
        "demucs_override_segment": base_params["disable"]
    },
    "mdx": {
        "mdx_hop_length": base_params["hop"],
        "mdx_segment_size": base_params["size"],
        "mdx_batch_size": base_params["batch"],
        "mdx_overlap": base_params["overlap"],
        "mdx_denoise": base_params["disable"],
        "mdx_override_segment": base_params["disable"]
    },
    "vr": {
        "vr_window_size": base_params["wsize"],
        "vr_batch_size": base_params["batch"],
        "vr_aggression": base_params["aggression"],
        "vr_post_process": base_params["disable"],
        "vr_post_process_threshold": base_params["threshold"],
        "vr_high_end_process": {**base_params["disable"], "info": "separation_hi-end_process_info"}
    },
    "mvox": {
        "mvox_segment": base_params["sec"],
        "mvox_overlap": base_params["overlap"],
        "mvox_override_segment": base_params["disable"]
    }
}
add_params_list = []
add_params_group = []
add_params_args = {}
for t_tab, t_components in add_params.items():
    add_params_group.append(t_tab)
    for t_component, t_settings in t_components.items():
        add_params_list.append(t_component)
        add_params_args[t_component] = {"default": t_settings["default"], "type": t_settings["type"]}

def get_add_params(args):
    """Безопасно получает add_params из args"""
    if hasattr(args, 'add_params') and args.add_params is not None:
        return vars(args.add_params)
    return {}

class MSSI: # Music Source Separation Inference
    def __init__(self, 
        output_dir=".", 
        output_format=output_formats[0],
        use_spec_invert=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_types = (
            "mel_band_roformer",
            "bs_roformer",
            "mdx23c",
            "scnet",
            "scnet_masked",
            "scnet_tran",
            "htdemucs",
            "bandit",
            "bandit_v2",
            "mdxnet",
            "vr",
            "medley_vox"
        )
        self.custom_model_types = self.model_types[:9]

        self.output_format = output_format
        self.device = torch.device(device)
        self.use_spec_invert = use_spec_invert
        self.model = None
        self.model_module = None
        self.state_dict = {}
        self.model_loaded = False
        self.model_type = None
        self.ckpt_path = None
        self.conf_path = None
        self.config = None
        self.target_instrument = None
        self.instruments = []
        self.input_mix = None
        self.input_file_name = None
        self.sample_rate = None
        self.selected_instruments = []
        self.output_files_list = []
        self.add_params = {}
        self.output_arrays: dict[str, np.ndarray] = {}

    def settings(self, 
        output_dir=".", 
        output_format=output_formats[0],
        use_spec_invert=False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_format = output_format
        self.use_spec_invert = use_spec_invert

    def set_add_params(self, **kwargs):
        self.add_params = kwargs

    def load_config(self, model_type: str, conf: str | Path):
        if not conf:
            raise PathNotSpecified(_i18n("path_not_specified"))
        self.conf_path = Path(conf)
        if not self.conf_path.exists():
            self.conf_path = None
            raise PathNotExist(_i18n("path_not_exist"))
        if model_type not in self.model_types:
            raise UnknownModelType(_i18n("unknown_model_type", model_type=model_type))
        self.model_type = model_type
        try:
            if self.model_type == "htdemucs":
                self.config = OmegaConf.load(self.conf_path)
                self.sample_rate = self.config.training.samplerate
            else:
                with self.conf_path.open("r", encoding="utf-8") as f:
                    self.config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
                    self.sample_rate = self.config.audio.sample_rate
            self.target_instrument = self.config.training.target_instrument
            self.instruments = self.config.training.instruments
            print(_i18n("config_loaded")+": "+self.conf_path.name)
        except FileNotFoundError:
            self.config = None
            self.conf_path = None
            self.model_type = None
            self.target_instrument = None
            self.instruments = []
            self.sample_rate = None
            raise FileNotFoundError(_i18n("config_not_found", path=conf)) from e
        except Exception as e:
            self.config = None
            self.conf_path = None
            self.model_type = None
            self.target_instrument = None
            self.instruments = []
            self.sample_rate = None
            raise ValueError(_i18n("config_load_error", error=str(e))) from e

    def prefer_target_instrument(self):
        if self.target_instrument:
            return [self.target_instrument]
        else:
            return self.instruments

    def print_instruments(self):
        print(_i18n("stems")+": "+",".join(self.instruments))
        print(_i18n("target_instrument")+": "+(self.target_instrument if self.target_instrument else _i18n("no")))

    def load_model_instance(self):
        if self.config is None or self.model_type is None:
            raise ConfigNotLoaded(_i18n("config_is_not_loaded"))
        if self.model_type == "mdx23c":
            from models import mdx23c_tfc_tdf_v3 as module
            self.model_module = module.TFC_TDF_net
            self.model = self.model_module(self.config)
            del module

        elif self.model_type == "mdxnet":
            from models import mdx_net as module
            self.model_module = module.MDXNet
            self.model = self.model_module(**dict(self.config.model))
            del module

        elif self.model_type == "vr":
            from models import vr_arch as module
            self.model_module = module.get_model
            self.model = self.model_module(self.config)
            del module

        elif self.model_type == "htdemucs":
            models_path = BASE_DIR / 'models'
            sys.path.append(str(models_path))
            from demucs import get_model as module
            self.model_module = module
            self.model = self.model_module(self.config)
            del module

        elif self.model_type == "mel_band_roformer":
            if hasattr(self.config, "windowed"):
                from models.windowed_roformer import model as module
                self.model_module = module.MelBandRoformerWSA
                self.model = self.model_module(**dict(self.config.model))
                del module
    
            elif hasattr(self.config, "conformer"):
                from models import bs_roformer as module
                self.model_module = module.MelBandConformer
                self.model = self.model_module(**dict(self.config.model))
                del module
    
            else:
                from models import bs_roformer as module
                self.model_module = module.MelBandRoformer
                self.model = self.model_module(**dict(self.config.model))
                del module
    
        elif self.model_type == "bs_roformer":
            if hasattr(self.config, "sw"):
                from models import bs_roformer as module
                self.model_module = module.BSRoformer_SW
                self.model = self.model_module(**dict(self.config.model))
                del module
    
            elif hasattr(self.config, "fno"):
                from models import bs_roformer as module
                self.model_module = module.BSRoformer_FNO
                self.model = self.model_module(**dict(self.config.model))
                del module
    
            elif hasattr(self.config, "hyperace"):
                from models import bs_roformer as module
                self.model_module = module.BSRoformerHyperACE
                self.model = self.model_module(**dict(self.config.model))
                del module
    
            elif hasattr(self.config, "hyperace2"):
                from models import bs_roformer as module
                self.model_module = module.BSRoformerHyperACE_2
                self.model = self.model_module(**dict(self.config.model))
                del module
    
            elif hasattr(self.config, "conformer"):
                from models import bs_roformer as module
                self.model_module = module.BSConformer
                self.model = self.model_module(**dict(self.config.model))
                del module
    
            elif hasattr(self.config, "conditional"):
                from models import bs_roformer as module
                self.model_module = module.BSRoformer_Conditional
                self.model = self.model_module(**dict(self.config.model))
                del module
    
            elif hasattr(self.config, "unwa_inst_large_2"):
                from models import bs_roformer as module
                self.model_module = module.BSRoformer_2
                self.model = self.model_module(**dict(self.config.model))
                del module

            elif hasattr(self.config, "siamese"):
                from models import bs_roformer as module
                self.model_module = module.BSSiameseRoformer
                self.model = self.model_module(**dict(self.config.model))
                del module

            else:
                from models import bs_roformer as module
                self.model_module = module.BSRoformer
                self.model = self.model_module(**dict(self.config.model))
                del module
    
        elif self.model_type == "bandit":
            from models.bandit.core import model as module
            self.model_module = module.MultiMaskMultiSourceBandSplitRNNSimple
            self.model = self.model_module(**self.config.model)
            del module

        elif self.model_type == "bandit_v2":
            from models.bandit_v2 import bandit as module
            self.model_module = module.Bandit
            self.model = self.model_module(**self.config.kwargs)
            del module

        elif self.model_type == "scnet_unofficial":
            from models import scnet_unofficial as module
            self.model_module = module.SCNet
            self.model = self.model_module(**self.config.model)
            del module

        elif self.model_type == "scnet":
            from models import scnet as module
            self.model_module = module.SCNet
            self.model = self.model_module(**self.config.model)
            del module

        elif self.model_type == 'scnet_masked':
            from models.scnet import scnet_masked as module
            self.model_module = module.SCNet
            self.model = self.model_module(**self.config.model)
            del module

        elif self.model_type == 'scnet_tran':
            from models.scnet import scnet_tran as module
            self.model_module = module.SCNet_Tran
            self.model = self.model_module(**self.config.model)
            del module

        elif self.model_type == 'medley_vox':
            from models import medley_vox as module
            self.model_module = module.load_model_with_args
            self.model = self.model_module(self.config.model)
            del module

        else:
            raise UnknownModelType(_i18n("unknown_model_type", model_type=self.model_type))

    def clear_model(self):
        """Стандартная очистка (сохранена для совместимости)"""
        # Принудительно переносим модель на CPU перед удалением
        if self.model is not None:
            if hasattr(self.model, "cpu"):
                self.model = self.model.cpu()
            del self.model

        self.model = None
        del self.model_module
        self.model_module = None
        self.model_loaded = False
        
        self.state_dict.clear() if hasattr(self.state_dict, 'clear') else None
        self.state_dict = {}
        
        self.config = None
        self.target_instrument = None
        self.instruments = []
        self.output_arrays.clear()
        
        self.ckpt_path = None
        self.conf_path = None
        self.model_type = None

        gc.collect()
        gc.collect()

        extra_clear_torch_cache()
        nuclear_clear_model()
        emergency_ram_clear()
        self.clear_gpu_cache()

    def clear_gpu_cache(self):
        gc.collect()
        torch.clear_autocast_cache()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

    def load_checkpoint(self, ckpt: str | Path):
        if not ckpt:
            raise PathNotSpecified(_i18n("path_not_specified"))
        self.ckpt_path = Path(ckpt)
        if not self.ckpt_path.exists():
            self.ckpt_path = None
            raise PathNotExist(_i18n("path_not_exist"))
        if not self.model:
            self.ckpt_path = None
            raise ModelNotLoaded(_i18n("model_not_loaded"))

        if self.model_type == "mdxnet":
            try:
                self.model.init_onnx_session(self.ckpt_path, self.device, 0)
                self.model_loaded = True
                print(_i18n("checkpoint_loaded") + ": " + self.ckpt_path.name)
            except Exception as e:
                self.model_loaded = False
                self.ckpt_path = None
                self.clear_model()
                print(_i18n("load_checkpoint_error", error=e))
                return
        else:
            try:
                try:
                    self.state_dict = torch.load(
                        self.ckpt_path, map_location=self.device, weights_only=True
                    )
                except torch.serialization.pickle.UnpicklingError:
                    self.state_dict = torch.load(
                        self.ckpt_path, map_location=self.device, weights_only=False
                    )

            except Exception as e:
                self.model_loaded = False
                self.ckpt_path = None
                self.clear_model()
                print(_i18n("load_checkpoint_error", error=e))
                return

            if "state" in self.state_dict:
                self.state_dict = self.state_dict["state"]
            if "state_dict" in self.state_dict:
                self.state_dict = self.state_dict["state_dict"]
            if "model_state_dict" in self.state_dict:
                self.state_dict = self.state_dict["model_state_dict"]

            if self.model_type == "medley_vox":
                has_ema_keys = any(k.startswith("ema_model") for k in self.state_dict.keys())
                
                if has_ema_keys:
                    self.state_dict = {k: v for k, v in self.state_dict.items() if k.startswith("ema_model")}
                
                new_state_dict = {}
                for k, v in self.state_dict.items():
                    if k.startswith("ema_model.module."): new_key = k.replace("ema_model.module.", "")
                    elif k.startswith("ema_model."): new_key = k.replace("ema_model.", "")
                    elif k.startswith("online_model.module."): new_key = k.replace("online_model.module.", "")
                    elif k.startswith("online_model."): new_key = k.replace("online_model.", "")
                    elif k.startswith("module."): new_key = k.replace("module.", "")
                    else: new_key = k
                    
                    if new_key not in ["initted", "step"]:
                        new_state_dict[new_key] = v
                
                self.state_dict = new_state_dict
                del new_state_dict

            try:
                self.model.load_state_dict(self.state_dict)
                self.state_dict = {}
                self.model_loaded = True
                self.model.to(self.device)
                self.model.eval()
                print(_i18n("checkpoint_loaded") + ": " + self.ckpt_path.name)
                
            except RuntimeError as e:
                try:
                    self.model.load_state_dict(self.state_dict, strict=False)
                    self.state_dict = {}
                    self.model_loaded = True
                    self.model.to(self.device)
                    self.model.eval()
                    print(_i18n("load_state_dict_error", error=e))
                    print(_i18n("checkpoint_loaded") + ": " + self.ckpt_path.name)
                except RuntimeError as e_2:
                    self.state_dict = {}
                    self.model_loaded = False
                    self.ckpt_path = None
                    self.clear_model()
                    print(_i18n("load_state_dict_error", error=e_2))
                    return

    def load_mix(self, path: str):
        self.input_file_name = None
        self.input_mix = None
        if self.config is None:
            raise ConfigNotLoaded(_i18n("config_is_not_loaded"))
        mono_bool = False
        if hasattr(self.config, "model"):
            if hasattr(self.config.model, "stereo"):
                mono_bool = False if self.config.model.stereo else True
        if not path:
            raise PathNotSpecified(_i18n("path_not_specified"))
        input_file = Path(path)
        if not input_file.exists():
            raise PathNotExist(_i18n("path_not_exist"))
        if check(path):
            self.input_file_name = input_file.stem
            self.input_mix, _ = read(path=input_file, sr=self.sample_rate, mono=mono_bool)
            self.input_mix = self.input_mix.copy()
            print(_i18n("loaded_mix")+": "+input_file.name)
            print(_i18n("array_shape")+": "+str(self.input_mix.shape))
        else:
            raise FileIsNotAudio(_i18n("file_is_not_audio", path=path))

    def load_array(self, array: np.ndarray, orig_sr: int):
        self.input_file_name = "temp_array"
        if self.config is None:
            raise ConfigNotLoaded(_i18n("config_is_not_loaded"))
        mono_bool = False
        if hasattr(self.config, "model"):
            if hasattr(self.config.model, "stereo"):
                mono_bool = not self.config.model.stereo
        self.input_mix = easy_resampler(array.copy(), orig_sr, self.sample_rate) if orig_sr != self.sample_rate else array.copy()
        if mono_bool:
            self.input_mix = stereo_to_mono(self.input_mix)

        print(_i18n("loaded_mix")+": "+_i18n("from_array"))
        print(_i18n("array_shape")+": "+str(self.input_mix.shape))

    def demix(self, add_text: str = ""):
        if self.input_mix is None:
            raise MixNotFound(_i18n("mix_not_found"))
        if self.input_mix.size == 0:
            raise MixIsEmpty(_i18n("mix_is_empty"))
        if not self.model_loaded:
            raise ModelNotLoaded(_i18n("model_not_loaded"))
        if self.model_type == "mdxnet":
            mix_tensor = torch.tensor(self.input_mix, dtype=torch.float32).to(self.device)
            batch_size = 1
            dim_t = 256
            hop_length: int = self.add_params.get("mdx_hop_length", 1024)
            batch_size: int = self.add_params.get("mdx_batch_size", 1)
            num_overlap: int = self.add_params.get("mdx_overlap", 2)
            denoise: bool = self.add_params.get("mdx_denoise", False)
            if self.add_params.get("mdx_override_segment", False):
                segment_size: int = self.add_params.get("mdx_segment_size", dim_t)
            else:
                segment_size: int = dim_t
            segment_size = round(segment_size / 128) * 128
            stem_name = self.target_instrument
            chunk_size = hop_length * (segment_size - 1)
            fade_size = chunk_size // 10
            step = chunk_size // num_overlap
            border = chunk_size - step
            self.model.post_init(segment_size, self.device)
            length_init = mix_tensor.shape[-1]

            if length_init > 2 * border and border > 0:
                wave = nn.functional.pad(mix_tensor, (border, border), mode="reflect")

            window = _getWindowingArray(chunk_size, fade_size).to(self.device)

            with torch.no_grad():
                result = torch.zeros_like(wave, device=self.device)
                counter = torch.zeros_like(wave, device=self.device)

                i = 0
                batch_data = []
                batch_locations = []
                denoise_str = " "+_i18n("denoise") if denoise else ""
                with tqdm(total=wave.shape[1], desc=_i18n("processing") + denoise_str + str(add_text), unit=_i18n("samples")) as progress_bar:
                    while i < wave.shape[1]:
                        part = wave[:, i : i + chunk_size]
                        chunk_len = part.shape[-1]

                        if chunk_len < chunk_size:
                            pad_mode = "reflect" if chunk_len > chunk_size // 2 else "constant"
                            part = nn.functional.pad(
                                part, (0, chunk_size - chunk_len), mode=pad_mode, value=0
                            )

                        batch_data.append(part)
                        batch_locations.append((i, chunk_len))
                        i += step

                        if len(batch_data) >= batch_size or i >= wave.shape[1]:
                            arr = torch.stack(batch_data, dim=0)

                            for j, (start, seg_len) in enumerate(batch_locations):
                                if denoise:
                                    processed_spec1 = self.model.forward(self.model.stft(arr[j : j + 1], chunk_size, hop_length, segment_size))
                                    processed_spec2 = self.model.forward(self.model.stft(-(arr[j : j + 1]), chunk_size, hop_length, segment_size))
                                    processed_wav = (self.model.istft(processed_spec1, chunk_size, hop_length, segment_size) + -self.model.istft(processed_spec2, chunk_size, hop_length, segment_size)) * 0.5
                                else:
                                    processed_spec = self.model.forward(self.model.stft(arr[j : j + 1], chunk_size, hop_length, segment_size))
                                    processed_wav = self.model.istft(processed_spec, chunk_size, hop_length, segment_size)

                                window_segment = window[..., :seg_len]
                                result[:, start : start + seg_len] += (
                                    processed_wav[0, :, :seg_len] * window_segment
                                )
                                counter[:, start : start + seg_len] += window_segment

                            batch_data.clear()
                            batch_locations.clear()

                        progress_bar.update(step)

                    estimated_sources = result / counter

                    if length_init > 2 * border and border > 0:
                        estimated_sources = estimated_sources[..., border:-border]
            result_separation = estimated_sources.detach().cpu().numpy()
            result_separation = np.nan_to_num(
                result_separation, nan=0.0, posinf=0.0, neginf=0.0
            )
            self.output_arrays = {stem_name: result_separation}
            del mix_tensor, window, result, counter, batch_data, batch_locations
            del estimated_sources, result_separation
            if denoise:
                del processed_spec1, processed_spec2, processed_wav
            else:
                del processed_spec, processed_wav
        elif self.model_type == "vr":
            from models.vr_arch import spec_utils, NON_ACCOM_STEMS
            aggression: int = self.add_params.get("vr_aggression", 5)
            enable_post_process: bool = self.add_params.get("vr_post_process", False)
            high_end_process: bool = self.add_params.get("vr_high_end_process", False)
            post_process_threshold: float = self.add_params.get("vr_post_process_threshold", 0.2)
            batch_size: int = self.add_params.get("vr_batch_size", 1)
            window_size: int = self.add_params.get("vr_window_size", 512)
            sr = self.sample_rate
            model_sample_rate = self.model.model_params.param["sr"]
            primary_stem, secondary_stem = self.instruments[0], self.instruments[1]
            aggr = float(int(aggression) / 100)
            aggressiveness = {
                "value": aggr,
                "split_bin": self.model.model_params.param["band"][1]["crop_stop"],
                "aggr_correction": self.model.model_params.param.get("aggr_correction"),
            }
            
            input_high_end_h = None
            input_high_end = None
            X_wave, X_spec_s = {}, {}

            bands_n = len(self.model.model_params.param["band"])

            for d in tqdm(range(bands_n, 0, -1), desc=_i18n("processing") + str(add_text), unit=_i18n("bands")):
                bp = self.model.model_params.param["band"][d]

                wav_resolution = bp["res_type"]

                if self.device.type == "mps":
                    wav_resolution = "polyphase"

                if d == bands_n:
                    X_wave[d], _ = librosa.resample(
                        y=self.input_mix,
                        orig_sr=self.sample_rate,
                        target_sr=bp["sr"],
                        res_type=wav_resolution,
                    )
                    X_spec_s[d] = spec_utils.wave_to_spectrogram(
                        X_wave[d],
                        bp["hl"],
                        bp["n_fft"],
                        self.model.model_params,
                        band=d,
                        is_v51_model=self.config.model.is_vr5,
                    )

                    if X_wave[d].ndim == 1:
                        X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
                else:
                    X_wave[d] = librosa.resample(
                        X_wave[d + 1],
                        orig_sr=self.model.model_params.param["band"][d + 1]["sr"],
                        target_sr=bp["sr"],
                        res_type=wav_resolution,
                    )
                    X_spec_s[d] = spec_utils.wave_to_spectrogram(
                        X_wave[d],
                        bp["hl"],
                        bp["n_fft"],
                        self.model.model_params,
                        band=d,
                        is_v51_model=self.config.model.is_vr5,
                    )

                if d == bands_n and high_end_process:
                    input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (
                        self.model.model_params.param["pre_filter_stop"]
                        - self.model.model_params.param["pre_filter_start"]
                    )
                    input_high_end = X_spec_s[d][
                        :, bp["n_fft"] // 2 - input_high_end_h : bp["n_fft"] // 2, :
                    ]

            X_spec = spec_utils.combine_spectrograms(
                X_spec_s, self.model.model_params, is_v51_model=self.config.model.is_vr5
            )
            del X_wave, X_spec_s

            def spec_to_wav(spec, high_end_process, input_high_end, input_high_end_h):
                if (
                    high_end_process
                    and isinstance(input_high_end, np.ndarray)
                    and input_high_end_h is not None  # Check if it's not None
                ):
                    input_high_end_ = spec_utils.mirroring(
                        "mirroring", spec, input_high_end, self.model.model_params
                    )
                    wav = spec_utils.cmb_spectrogram_to_wave(
                        spec,
                        self.model.model_params,
                        input_high_end_h,
                        input_high_end_,
                        is_v51_model=self.config.model.is_vr5,
                    )
                else:
                    wav = spec_utils.cmb_spectrogram_to_wave(
                        spec, self.model.model_params, is_v51_model=self.config.model.is_vr5
                    )

                return wav

            def _execute(X_mag_pad: np.ndarray, roi_size: int) -> np.ndarray:
                X_dataset = []
                patches = (X_mag_pad.shape[2] - 2 * self.model.offset) // roi_size
                for i in tqdm(range(patches), desc=_i18n("processing") + str(add_text), unit=_i18n("patches")):
                    start = i * roi_size
                    X_mag_window = X_mag_pad[:, :, start : start + window_size]
                    X_dataset.append(X_mag_window)


                X_dataset = np.asarray(X_dataset)
                self.model.eval()
                with torch.no_grad():
                    mask = []

                    for i in tqdm(range(0, patches, batch_size), desc=_i18n("processing") + str(add_text), unit=_i18n("chunks")):
                        X_batch = X_dataset[i : i + batch_size]
                        X_batch = torch.from_numpy(X_batch).to(self.device)
                        pred = self.model.predict_mask(X_batch)
                        if not pred.size()[3] > 0:
                            raise ValueError(
                                _i18n("window_size_error")
                            )
                        pred = pred.detach().cpu().numpy()
                        pred = np.concatenate(pred, axis=2)
                        mask.append(pred)
                    if len(mask) == 0:
                        raise ValueError(
                            _i18n("window_size_error")
                        )

                    mask = np.concatenate(mask, axis=2)
                return mask

            def postprocess(
                mask: np.ndarray, 
                X_mag: np.ndarray, 
                X_phase: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray]:
                is_non_accom_stem = False
                for stem in NON_ACCOM_STEMS:
                    if stem == primary_stem.lower():
                        is_non_accom_stem = True

                mask = spec_utils.adjust_aggr(mask, is_non_accom_stem, aggressiveness)

                if enable_post_process:
                    mask = spec_utils.merge_artifacts(
                        mask, thres=post_process_threshold
                    )

                y_spec = mask * X_mag * np.exp(1.0j * X_phase)
                v_spec = (1 - mask) * X_mag * np.exp(1.0j * X_phase)

                return y_spec, v_spec

            X_mag, X_phase = spec_utils.preprocess(X_spec)
            n_frame = X_mag.shape[2]
            pad_l, pad_r, roi_size = spec_utils.make_padding(
                n_frame, window_size, self.model.offset
            )
            X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")
            X_mag_pad /= X_mag_pad.max()
            mask = _execute(X_mag_pad, roi_size)

            mask = mask[:, :, :n_frame]

            y_spec, v_spec = postprocess(mask, X_mag, X_phase)

            y_spec = np.nan_to_num(y_spec, nan=0.0, posinf=0.0, neginf=0.0)
            v_spec = np.nan_to_num(v_spec, nan=0.0, posinf=0.0, neginf=0.0)
            primary_stem_array = spec_to_wav(y_spec, high_end_process, input_high_end, input_high_end_h)
            primary_stem_array = librosa.resample(
                primary_stem_array,
                orig_sr=model_sample_rate,
                target_sr=sr,
            ).T
            secondary_stem_array = spec_to_wav(v_spec, high_end_process, input_high_end, input_high_end_h)
            secondary_stem_array = librosa.resample(
                secondary_stem_array,
                orig_sr=model_sample_rate,
                target_sr=sr,
            ).T
            self.output_arrays = {
                primary_stem: primary_stem_array,
                secondary_stem: secondary_stem_array,
            }
            del X_spec, X_mag, X_phase, X_mag_pad, mask
            del y_spec, v_spec, primary_stem_array, secondary_stem_array
        elif self.model_type == "htdemucs":
            mix = torch.tensor(self.input_mix, dtype=torch.float32)
            segment_sec: int = self.add_params.get("demucs_segment", 10)
            denoise: bool = self.add_params.get("demucs_denoise", False)
            num_overlap = self.add_params.get("demucs_overlap", 2)
            batch_size: int = self.add_params.get("demucs_batch_size", 1)
            if self.add_params.get("demucs_override_segment", False):
                chunk_size = self.config.training.samplerate * segment_sec
            else:
                chunk_size = getattr(self.config.training, "segment", 10) * self.config.training.samplerate
            num_instruments = len(self.instruments)
            step = chunk_size // num_overlap
            fade_size = chunk_size // 10
            windowing_array = _getWindowingArray(chunk_size, fade_size)
            use_amp = getattr(self.config.training, "use_amp", True)

            with torch.inference_mode():
                req_shape = (num_instruments,) + mix.shape
                result = torch.zeros(req_shape, dtype=torch.float32)
                counter = torch.zeros(req_shape, dtype=torch.float32)

                i = 0
                batch_data = []
                batch_locations = []
                denoise_str = " "+_i18n("denoise") if denoise else ""
                with tqdm(total=mix.shape[1], desc=_i18n("processing") + denoise_str + str(add_text), unit=_i18n("samples")) as progress_bar:
                    while i < mix.shape[1]:
                        part = mix[:, i : i + chunk_size].to(self.device)
                        chunk_len = part.shape[-1]
                        pad_mode = "reflect" if chunk_len > chunk_size // 2 else "constant"
                        part = nn.functional.pad(
                            part, (0, chunk_size - chunk_len), mode=pad_mode, value=0
                        )

                        batch_data.append(part)
                        batch_locations.append((i, chunk_len))
                        i += step

                        if len(batch_data) >= batch_size or i >= mix.shape[1]:
                            arr = torch.stack(batch_data, dim=0)
                            if denoise:
                                x1 = self.model(arr)
                                x2 = self.model(-arr)
                                x = (x1 + -x2) * 0.5
                            else:
                                x = self.model(arr)
                            window = windowing_array.clone()
                            if i - step == 0:
                                window[:fade_size] = 1
                            elif i >= mix.shape[1]:
                                window[-fade_size:] = 1

                            for j, (start, seg_len) in enumerate(batch_locations):
                                result[..., start : start + seg_len] += (
                                    x[j, ..., :seg_len].cpu() * window[..., :seg_len]
                                )
                                counter[..., start : start + seg_len] += window[..., :seg_len]               

                            batch_data.clear()
                            batch_locations.clear()
                        progress_bar.update(step)
                    estimated_sources = result / counter
                    estimated_sources = estimated_sources.detach().cpu().numpy()
                    np.nan_to_num(estimated_sources, copy=False, nan=0.0)
            
            if num_instruments <= 1:
                self.output_arrays = estimated_sources
            else:
                instruments = self.instruments
                self.output_arrays = {k: v for k, v in zip(instruments, estimated_sources)}
            del mix, result, counter, batch_data, batch_locations
            if denoise:
                del estimated_sources, x, x1, x2
            else:
                del estimated_sources, x
        elif self.model_type == "medley_vox":
            import pyloudnorm as pyln
            from models.medley_vox.loudness_utils import loudnorm, db2linear
            if self.add_params.get("mvox_override_segment", False):
                segment_sec: int = self.add_params.get("mvox_segment", self.config.model.seq_dur)
            else:
                segment_sec = self.config.model.seq_dur
            overlap: int = self.add_params.get("mvox_overlap", 2)
            stems: List[str] = self.instruments
            
            if self.input_mix.ndim == 1:
                self.input_mix = np.expand_dims(self.input_mix, axis=0)
                num_channels = 1
            elif self.input_mix.ndim == 2:
                if self.input_mix.shape[0] <= self.input_mix.shape[1]:
                    num_channels = self.input_mix.shape[0]
                else:
                    self.input_mix = self.input_mix.T
                    num_channels = self.input_mix.shape[0]
            
            samplerate = self.config.model.sample_rate
            chunk_size = int(samplerate * segment_sec)
            step = chunk_size // overlap
            fade_size = chunk_size // 10
            
            meter = pyln.Meter(samplerate)
            try:
                if num_channels > 1:
                    mix_for_loudnorm = self.input_mix.T
                else:
                    mix_for_loudnorm = self.input_mix[0]
                
                mixture_d, adjusted_gain = loudnorm(mix_for_loudnorm, -24.0, meter)
                
                if num_channels > 1:
                    if isinstance(mixture_d, np.ndarray) and mixture_d.ndim == 2:
                        mixture_d = mixture_d.T
                    else:
                        mixture_d = np.tile(mixture_d, (num_channels, 1))
                else:
                    if mixture_d.ndim == 1:
                        mixture_d = mixture_d.reshape(1, -1)
                        
            except Exception as e:
                print(_i18n("loudnorm_error", error=str(e)))
                mixture_d = mix.copy()
                rms = np.sqrt(np.mean(mix**2))
                target_rms = 0.1
                if rms > 0:
                    adjusted_gain = 20 * np.log10(target_rms / rms)
                    mixture_d = mix * (target_rms / rms)
                else:
                    adjusted_gain = 0
            
            length_init = mixture_d.shape[1]
            
            windowing_array = _getWindowingArray(chunk_size, fade_size).to(self.device)
            
            result_stems = {stem: np.zeros((num_channels, length_init), dtype=np.float32) 
                          for stem in stems}
            
            mix_tensor = torch.tensor(mixture_d, dtype=torch.float32).to(self.device)
            
            counters = {stem: torch.zeros((num_channels, length_init), dtype=torch.float32, device=self.device) 
                        for stem in stems}
            
            i = 0
            with tqdm(total=length_init, desc=_i18n("processing") + str(add_text), unit=_i18n("samples")) as progress_bar:
                while i < length_init:
                    end_idx = min(i + chunk_size, length_init)
                    chunk = mix_tensor[:, i:end_idx]
                    cur_chunk_len = chunk.shape[1]
                    
                    chunk_results = torch.zeros((num_channels, 2, cur_chunk_len), dtype=torch.float32, device=self.device)
                    
                    for ch in range(num_channels):
                        channel_chunk = chunk[ch:ch+1, :]
                        
                        if cur_chunk_len < chunk_size:
                            pad_len = chunk_size - cur_chunk_len
                            channel_chunk = torch.nn.functional.pad(
                                channel_chunk, (0, pad_len), mode='constant', value=0
                            )
                        
                        channel_chunk = channel_chunk.unsqueeze(0)
                        
                        with torch.no_grad():
                            out_chunk = self.model.separate(channel_chunk)
                        
                        chunk_results[ch, :, :cur_chunk_len] = out_chunk[0, :, :cur_chunk_len].cpu()

                    window = windowing_array[:cur_chunk_len].clone()
                    if i == 0:
                        window[:fade_size] = 1
                    if end_idx >= length_init:
                        window[-fade_size:] = 1
                    
                    for stem_idx, stem in enumerate(stems):
                        result_stems[stem][:, i:end_idx] += chunk_results[:, stem_idx, :].cpu().numpy() * window.cpu().numpy()
                        counters[stem][:, i:end_idx] += window
                    
                    i += step
                    progress_bar.update(step)
                
            for stem in stems:
                counters_np = counters[stem].detach().cpu().numpy()
                mask = counters_np > 0
                result_stems[stem][mask] /= counters_np[mask]
                
                result_stems[stem] = result_stems[stem] * db2linear(-adjusted_gain)

            self.output_arrays = result_stems
            del mix_tensor, mixture_d, counters
            del result_stems, chunk_results
            del meter
        else:
            mix = torch.tensor(self.input_mix, dtype=torch.float32).to(self.device)
            segment: int = self.add_params.get("mdxc_segment_size", 256)
            if hasattr(self.config, "model"):
                if hasattr(self.config.model, "stft_hop_length"):
                    hop_length = self.config.model.stft_hop_length
                elif hasattr(self.config.model, "hop_size"):
                    hop_length = self.config.model.hop_size
                elif hasattr(self.config.model, "hop_length"):
                    hop_length = self.config.model.hop_length
            if hasattr(self.config, "audio"):
                if hasattr(self.config.audio, "hop_length"):
                    hop_length = self.config.audio.hop_length
            if hasattr(self.config, "kwargs"):
                if hasattr(self.config.kwargs, "hop_length"):
                    hop_length = self.config.kwargs.hop_length

            if self.add_params.get("mdxc_override_segment", False):
                chunk_size = int(hop_length) * (int(segment) - 1)
            else:
                chunk_size = self.config.audio.chunk_size
            instruments = self.prefer_target_instrument()
            num_instruments = len(instruments)
            denoise: bool = self.add_params.get("mdxc_denoise", False)
            num_overlap: int = self.add_params.get("mdxc_overlap", 2)

            fade_size = chunk_size // 10
            step = chunk_size // num_overlap
            border = chunk_size - step
            length_init = mix.shape[-1]
            windowing_array = _getWindowingArray(chunk_size, fade_size)

            if length_init > 2 * border and border > 0:
                mix = nn.functional.pad(mix, (border, border), mode="reflect")

            batch_size: int = self.add_params.get("mdxc_batch_size", 1)
            use_amp = getattr(self.config.training, "use_amp", True)

            with torch.inference_mode(), get_autocast_context(self.device.type, use_amp):
                req_shape = (num_instruments,) + mix.shape
                result = torch.zeros(req_shape, dtype=torch.float32)
                counter = torch.zeros(req_shape, dtype=torch.float32)

                i = 0
                batch_data = []
                batch_locations = []
                denoise_str = " "+_i18n("denoise") if denoise else ""
                with tqdm(total=mix.shape[1], desc=_i18n("processing") + denoise_str + str(add_text), unit=_i18n("samples")) as progress_bar:

                    while i < mix.shape[1]:
                        part = mix[:, i : i + chunk_size].to(self.device)
                        chunk_len = part.shape[-1]

                        pad_mode = "reflect" if chunk_len > chunk_size // 2 else "constant"
                        part = nn.functional.pad(
                            part, (0, chunk_size - chunk_len), mode=pad_mode, value=0
                        )

                        batch_data.append(part)
                        batch_locations.append((i, chunk_len))
                        i += step

                        if len(batch_data) >= batch_size or i >= mix.shape[1]:
                            arr = torch.stack(batch_data, dim=0)
                            if denoise:
                                x1 = self.model(arr)
                                x2 = self.model(-arr)
                                x = (x1 + -x2) * 0.5
                            else:
                                x = self.model(arr)

                            window = windowing_array.clone()
                            if i - step == 0:
                                window[:fade_size] = 1
                            elif i >= mix.shape[1]:
                                window[-fade_size:] = 1

                            for j, (start, seg_len) in enumerate(batch_locations):
                                result[..., start : start + seg_len] += (
                                    x[j, ..., :seg_len].cpu() * window[..., :seg_len]
                                )
                                counter[..., start : start + seg_len] += window[..., :seg_len]                

                            batch_data.clear()
                            batch_locations.clear()
                        progress_bar.update(step)

                    estimated_sources = result / counter
                    estimated_sources = estimated_sources.detach().cpu().numpy()
                    np.nan_to_num(estimated_sources, copy=False, nan=0.0)

                    if length_init > 2 * border and border > 0:
                        estimated_sources = estimated_sources[..., border:-border]

            self.output_arrays = {k: v for k, v in zip(instruments, estimated_sources)}
            del mix, result, counter, batch_data, batch_locations
            if denoise:
                del estimated_sources, x, x1, x2
            else:
                del estimated_sources, x
        self.add_second_stem()
        return

    def add_second_stem(self):
        if self.target_instrument:
            second_stem = [instrument for instrument in self.instruments if instrument != self.target_instrument][0]
            self.output_arrays[second_stem] = subtractor(self.input_mix, self.output_arrays[self.target_instrument], self.sample_rate, self.sample_rate, spectrogram=self.use_spec_invert)[0]
            print(_i18n("added_second_stem") + ": " + second_stem)
        else:
            return

    def delete_unselected_stems(self, selected_stems: list):
        if selected_stems:
            output_keys = list(self.output_arrays.keys())
            deleted_keys = []
            for stem in output_keys:
                if stem not in selected_stems:
                    self.output_arrays[stem] = None
                    del self.output_arrays[stem]
                    deleted_keys.append(stem)
            print(_i18n("deleted_stems") + f": " + ",".join(deleted_keys))
        else:
            return

    def extract_instrumental(self, extract_instrumental: bool, return_: bool = False):
        if extract_instrumental:
            if self.output_arrays:
                output_keys = [key_ for key_ in self.output_arrays]
                self.output_arrays["invert"] = self.input_mix.copy()
                for stem in output_keys:
                    self.output_arrays["invert"] = subtractor(self.output_arrays["invert"], self.output_arrays[stem], self.sample_rate, self.sample_rate, spectrogram=self.use_spec_invert)[0]
        if return_:
            return self.output_arrays["invert"]

    def write(self, template: str, format_return: str = "name_stems_list"):
        results = []
        writed_stems = []
        model_name = self.ckpt_path.stem
        print(_i18n("format_return") + ": " + _i18n(format_return))
        for stem, array in tqdm(self.output_arrays.items(), desc=_i18n("writing"), unit=_i18n('files')):        
            custom_name = Namer.template(
                template,
                STEM=stem,
                MODEL=model_name,
                NAME=Namer.short_input_name_template(template, STEM=stem, MODEL=model_name, NAME=self.input_file_name)
            )
            writed_stems.append([stem, write(Namer.iter(self.output_dir / f"{custom_name}.{self.output_format}"), array, self.sample_rate)])
        if writed_stems:
            match format_return:
                case "name_stems_list":
                    results = [self.input_file_name, writed_stems]
                case "stems_list":
                    results = writed_stems
                case "stems_list_append_self":
                    self.output_files_list.append(writed_stems)
                case "name_stems_list_append_self":
                    self.output_files_list.append([self.input_file_name, writed_stems])
        return results
    
    def clear_mix(self):
        self.input_file_name = None
        self.input_mix = None
        self.output_arrays.clear()

    def clear_outputs(self):
        self.output_files_list.clear()

    def get_outputs(self):
        return self.output_files_list
    
    def _process(self, i: int, total: int, path: str, template: str, selected_stems: list = [], extract_instrumental: bool = True):
        template = Namer.sanitize(template)
        template = Namer.dedup_template(template, keys=["NAME", "MODEL", "STEM"])
        template = Namer.short(template, length=40)
        self.clear_mix()
        self.load_mix(path)
        try:
            self.demix(f" | {i}/{total} {_i18n('files')}")
        except Exception as e:
            self.clear_mix()
            raise DemixError(_i18n("demix_error", error=e)) from e
        self.delete_unselected_stems(selected_stems)
        self.extract_instrumental(extract_instrumental)
        self.write(template, "name_stems_list_append_self")
        self.clear_mix()

    def _process_array_ensemble(self, i: int, total: int, array: np.ndarray, sr: int, primary_stem: str | None = None, invert: bool = False):
        self.clear_mix()
        self.load_array(array, sr)
        try:
            self.demix(f" | {i}/{total} {_i18n('models')} | {self.ckpt_path.stem}")
        except Exception as e:
            self.clear_mix()
            raise DemixError(_i18n("demix_error", error=e)) from e
        self.delete_unselected_stems([primary_stem])
        if invert:
            result = self.extract_instrumental(True, return_=True)
        else:
            result = self.output_arrays[primary_stem]
        return result, self.sample_rate

    def load_model(self, model_type: str, ckpt: str | Path, conf: str | Path):
        self.clear_model()
        self.load_config(model_type=model_type, conf=conf)
        self.load_model_instance()
        self.load_checkpoint(ckpt=ckpt)

    def inference(self, input: str | list, /, *inputs, template: str = "NAME_MDOEL_STEM", selected_stems: list = [], extract_instrumental: bool = False):
        self.clear_outputs()
        all_inputs = []
        if isinstance(input, list):
            all_inputs.extend(input)
        else:
            all_inputs.append(input)
        if inputs:
            all_inputs.extend(inputs)
        total = len(all_inputs)
        for i, input_file in enumerate(all_inputs, start=1):
            try:
                self._process(i, total, input_file, template=template, selected_stems=selected_stems, extract_instrumental=extract_instrumental)
            except Exception as e:
                traceback.print_exc()
        return self.get_outputs()
    
class ModelManager:
    def __init__(self):
        self.info = {}
        self.info_url = "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/models.json?download=true"
        self.info_path = Path(BASE_DIR) / "models.json"
        self.load_info()
        self.cache_dir = Path(BASE_DIR) / "separation_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Убираем lambda-функции, заменяем на обычные методы
        
    def get_all_models(self):
        return [mn for mn in self.info]
    
    def get_stems(self, model_name):
        return [stem for stem in self.info.get(model_name, {}).get("stems", [])]
    
    def get_target_instrument(self, model_name):
        return self.info.get(model_name, {}).get("target_instrument", None)
    
    def get_model_type(self, model_name):
        return self.info.get(model_name, {}).get("model_type", "")
    
    def get_links(self, model_name):
        return (self.info.get(model_name, {}).get("checkpoint_url", None), 
                self.info.get(model_name, {}).get("config_url", None))
    
    def generate_local_paths(self, model_name):
        return (self.cache_dir / f"{model_name}.ckpt", 
                self.cache_dir / f"{model_name}_config.yaml")
    
    def check_installed(self, model_name):
        return [path.exists() for path in self.generate_local_paths(model_name)]
    
    def check_installed2(self, model_name):
        return all(self.check_installed(model_name))

    def load_info(self):
        self.info = json.loads(self.info_path.read_text("utf-8"))

    def show_info(self, limit: int = None, stem: str = None, only_installed: bool = False):
        models = []
        if stem:
            models = [
                model for model in self.get_all_models() 
                if (stem in self.get_stems(model) or 
                    stem.lower() in self.get_stems(model) or 
                    stem.upper() in self.get_stems(model) or 
                    stem.capitalize() in self.get_stems(model) or 
                    stem.title() in self.get_stems(model))
            ]
        else:
            models = self.get_all_models()
        if only_installed:
            models = [model for model in models if self.check_installed2(model)]
        if limit:
            models = models[:limit]

        console = rich.console.Console()
        table = rich.table.Table(title=_i18n("model_info"), show_lines=True)
        table.add_column(_i18n("model_name"), no_wrap=True)
        table.add_column(_i18n("output_stems"))
        table.add_section()
        table.add_row(_i18n("table_model_info_installed_legend"), _i18n("table_model_info_target_instrument_legend"))
        table.add_section()
        if models:
            for model_ in models:
                target_instrument = self.get_target_instrument(model_)
                stems = self.get_stems(model_)
                if target_instrument:
                    for i, stem in enumerate(stems):
                        if stem == target_instrument:
                            stems[i] = f"[green]{stem}[/]"
                            
                stems_str = ", ".join(stems)
                table.add_row(f"[green]{model_}[/]" if self.check_installed2(model_) else model_, stems_str)
        else:
            table.add_row(_i18n("na"), _i18n("na"))

        console.print(table)

    def update_info(self):
        dw_file(self.info_url, self.info_path)
        print(_i18n("model_info_updated"))

    def download(self, model_name: str):
        status = ""
        urls = self.get_links(model_name)
        local_paths = self.generate_local_paths(model_name)
        local_exists = self.check_installed(model_name)
        for url, local_path, exists in zip(urls, local_paths, local_exists):
            if not exists:
                dw_file(url, local_path)
        if all(local_exists):
            status = _i18n("model_already_downloaded")
        else:
            status = _i18n("model_downloaded")
        print(status)
        return status

class Ensembler:
    def __init__(self):
        self.arrays = []
        self.srs = []

    def add_array(self, y: np.ndarray, sr: int):
        self.arrays.append(y)
        self.srs.append(sr)

    def get_arrays(self):
        return self.arrays
    
    def get_srs(self):
        return self.srs
    
    def clear(self): 
        self.arrays.clear()

class Separator(ModelManager):
    def __init__(self):
        super().__init__()
        self.mssi = MSSI()

    def unload_model(self):
        self.mssi.clear_model()

    @hf_spaces_gpu # (duration=120) Для спейса LongQuota / длинная квота на HuggingFace ZeroGPU (по умолчанию 60 секунд)
    def separate_base(
        self, 
        input_valid_files: list[str | Path], 
        model_name: str, 
        template: str, 
        checkpoint: str | Path, 
        config: str | Path, 
        selected_stems: list, 
        extract_instrumental: bool
    ):
        self.mssi.clear_model() 
        self.mssi.load_model(self.get_model_type(model_name), checkpoint, config)
        self.mssi.print_instruments()
        results = self.mssi.inference(input_valid_files, template=template, selected_stems=selected_stems, extract_instrumental=extract_instrumental)
        self.mssi.clear_model()
        return results

    def separate(
        self,
        input_files: list[str | Path],
        output_dir: str | Path = Path("."),
        output_format: str = output_formats[0],
        template: str = "NAME_(STEM)_MODEL",
        model_name: str = "bs_6stem",
        extract_instrumental: bool = False,
        use_spec_invert: bool = False,
        selected_stems: list = [],
        add_params: dict = {}
    ):
        if not output_dir:
            output_dir = ""
        input_valid_files = get_audio_files_from_list(input_files, only_files=False)
        if not input_valid_files:
            raise PathsNotSpecified(_i18n("paths_not_specified"))
        self.mssi.settings(output_dir=output_dir, output_format=output_format, use_spec_invert=use_spec_invert)
        self.mssi.set_add_params(**add_params)
        self.download(model_name)
        checkpoint, config = self.generate_local_paths(model_name)
        results = self.separate_base(input_valid_files, model_name, template, checkpoint, config, selected_stems, extract_instrumental)
        return results

    @hf_spaces_gpu # (duration=120) Для спейса LongQuota / длинная квота на HuggingFace ZeroGPU (по умолчанию 60 секунд)
    def custom_separate(
        self,
        input_files: list,
        output_dir: str | Path = Path("."),
        output_format: str = output_formats[0],
        template: str = "NAME_(STEM)_MODEL",
        model_type: str = "bs_roformer",
        ckpt: str = "model.ckpt",
        conf: str = "conf.ckpt",
        extract_instrumental: bool = False,
        use_spec_invert: bool = False,
        selected_stems: list = [],
        add_params: dict = {}
    ):
        if not output_dir:
            output_dir = ""
        input_valid_files = get_audio_files_from_list(input_files, only_files=False)
        if not input_valid_files:
            raise PathsNotSpecified(_i18n("paths_not_specified"))
        checkpoint, config = Path(ckpt), Path(conf)
        model_name = checkpoint.stem
        self.mssi.settings(output_dir=output_dir, output_format=output_format, use_spec_invert=use_spec_invert)
        self.mssi.set_add_params(**add_params)
        self.mssi.clear_model()
        self.mssi.load_model(model_type, checkpoint, config)
        self.previous_model_name = model_name
        self.mssi.print_instruments()
        results = self.mssi.inference(input_valid_files, template=template, selected_stems=selected_stems, extract_instrumental=extract_instrumental)
        self.mssi.clear_model()
        return results

    def print_flow(self, flow):
        """Print current ensemble flow in a formatted table (like show_info)"""
        if not flow:
            return
        
        console = rich.console.Console()
        table = rich.table.Table(title="", show_lines=True)
        table.add_column("#", style="cyan", no_wrap=True)
        table.add_column(_i18n("model_name"))
        table.add_column(_i18n("primary_stem"))
        table.add_column(_i18n("invert"))
        table.add_column(_i18n("weights"), justify="right")
        
        for idx, (model_name, primary_stem, invert, weight) in enumerate(flow, start=1):
            invert_str = _i18n("yes") if invert else _i18n("no")
            table.add_row(
                str(idx),
                model_name,
                primary_stem,
                invert_str,
                f"{weight:.2f}" if isinstance(weight, (int, float)) else str(weight)
            )
        
        console.print(table)

    @hf_spaces_gpu # (duration=120) Для спейса LongQuota / длинная квота на HuggingFace ZeroGPU (по умолчанию 60 секунд)
    def auto_ensemble_base(
            self, 
            model_name: str, 
            checkpoint: str | Path, 
            config: str | Path,
            i: int,
            model_count: int,
            mix: np.ndarray,
            orig_sr: int,
            primary_stem: str,
            invert: bool
        ):
        self.mssi.clear_model()
        self.mssi.load_model(self.get_model_type(model_name), checkpoint, config)
        self.mssi.print_instruments()
        output, model_sr = self.mssi._process_array_ensemble(i, model_count, mix, orig_sr, primary_stem, invert)
        self.mssi.clear_model()
        return output, model_sr

    def auto_ensemble(
        self, 
        input_file: str | Path,
        output_dir: str | Path = Path("."),
        flow: list[list[str | bool | int | float]] = [],
        template: str = "NAME_TYPE_COUNT",
        etype: str = ensemble_types[0],
        output_format: str = output_formats[0],
        use_spec_invert: bool = False,
        save_primary_stems: bool = False
    ) -> tuple[str, str, list[str]]:
        if not output_dir:
            output_dir = ""
        if not input_file:
            raise PathNotSpecified(_i18n("path_not_specified"))
        input_file = Path(input_file)
        if not input_file.exists():
            raise PathNotExist(_i18n("path_not_exist"))
        if not check(input_file):
            raise FileIsNotAudio(_i18n("file_is_not_audio", path=input_file))
        self.print_flow(flow)
        if not flow:
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_count = len(flow)
        print(_i18n("ensemble_type")+": "+etype)
        print(_i18n("ensemble_models_count")+": "+str(model_count))
        mix, orig_sr = read(input_file, sr=44100)
        template = Namer.sanitize(template)
        template = Namer.dedup_template(template, keys=["NAME", "TYPE", "COUNT"])
        template = Namer.short(template, length=40)
        invert_key = "_invert"
        custom_name = Namer.template(
            template,
            TYPE=etype,
            COUNT=model_count,
            NAME=Namer.short_input_name_template(template, TYPE=etype, COUNT=model_count, NAME=input_file.stem)
        )
        auto_ensembler = Ensembler()
        weights = []
        saved_primary_stems = []
        self.mssi.set_add_params(**{"demucs_denoise": True, "mdx_denoise": True})
        for i, (model_name, primary_stem, invert, weight) in enumerate(flow, start=1):
            try:
                self.download(model_name)
                checkpoint, config = self.generate_local_paths(model_name)
                output, model_sr = self.auto_ensemble_base(model_name, checkpoint, config, i, model_count, mix, orig_sr, primary_stem, invert)
                auto_ensembler.add_array(output, model_sr)
                weights.append(weight)
                if save_primary_stems:
                    primary_stem_file_name = primary_stem + (invert_key if invert else "")
                    saved_primary_stems.append(write(Namer.iter(output_dir / model_name / f"{primary_stem_file_name}.flac"), output, model_sr))
            except Exception as e:
                print(_i18n("error_occured_separation")+": "+str(e))
                gr.Warning(message="<b>"+f'{_i18n("error_occured_separation")}'.replace("\n", "<br>")+": "+str(e)+"</b>", title="")
                continue
        extracted_primary_stems = auto_ensembler.get_arrays()
        srs = auto_ensembler.get_srs()
        output_array, sr_ = ensemble(extracted_primary_stems, srs, etype, weights)
        extracted_primary_stems = None
        auto_ensembler.clear()
        auto_ensembler, output = None, None
        del auto_ensembler, output
        inverted_array, i_sr = subtractor(mix, output_array, orig_sr, sr_, spectrogram=use_spec_invert)
        return write(Namer.iter(output_dir / f"{custom_name}.{output_format}"), output_array, sr_), write(Namer.iter(output_dir / f"{Namer.short(custom_name+invert_key)}.{output_format}"), inverted_array, i_sr), saved_primary_stems
  
    def manual_ensemble(
        self,
        input_files: list[str | Path],
        output_dir: str | Path = Path("."),
        weights: list[float] | None = None,
        template: str = "ensembled_TYPE_COUNT",
        etype: str = ensemble_types[0],
        output_format: str = output_formats[0],
    ) -> str:
        if not output_dir:
            output_dir = ""
        input_valid_files = get_audio_files_from_list(input_files, only_files=True)
        if not input_valid_files:
            raise PathsNotSpecified(_i18n("paths_not_specified"))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays, srs = multiread(input_valid_files)
        model_count = len(srs)
        results, max_sr = ensemble(arrays, srs, etype, weights)
        template = Namer.sanitize(template)
        template = Namer.dedup_template(template, keys=["TYPE", "COUNT"])
        template = Namer.short(template, length=40)
        custom_name = Namer.template(
            template,
            TYPE=etype,
            COUNT=model_count
        )
        return write(Namer.iter(output_dir / f"{custom_name}.{output_format}"), results, max_sr)
    
    def subtract(self, audio1: str | Path, audio2: str | Path, output_dir: str | Path = Path("."), output_format: str = output_formats[0], use_spec_invert: bool = False, template: str = "invert_TYPE_NAME"):
        if not output_dir:
            output_dir = ""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not audio1 or not audio2:
            raise PathsNotSpecified(_i18n("paths_not_specified"))
        audio1, audio2 = Path(audio1), Path(audio2)
        if not audio1.exists() or not audio2.exists():
            raise PathsNotExist(_i18n("paths_not_exist"))
        if not check(audio1) or not check(audio2):
            raise FilesIsNotAudio(_i18n("files_is_not_audio"))
        template = Namer.sanitize(template)
        template = Namer.dedup_template(template, keys=["NAME", "TYPE"])
        template = Namer.short(template, length=40)
        invert_type_key = ("spectrogram" if use_spec_invert else "waveform")
        custom_name = Namer.template(
            template,
            TYPE=invert_type_key,
            NAME=Namer.short_input_name_template(template, TYPE=invert_type_key, NAME=audio1.stem)
        )
        y1, sr1 = read(audio1)
        y2, sr2 = read(audio2)
        inverted, min_sr = subtractor(y1, y2, sr1, sr2, spectrogram=use_spec_invert)
        return write(Namer.iter(output_dir / f"{custom_name}.{output_format}"), inverted, min_sr)
    
if __name__ == "__main__":
    separator = Separator()
    args = parse_separator_args(add_params_args)
    if args.mode == "separate":
        separator.separate(
            input_files=args.input,
            output_dir=args.output_dir,
            output_format=args.output_format,
            template=args.template,
            model_name=args.model_name,
            extract_instrumental=args.extract_instrumental,
            use_spec_invert=args.use_spec_invert,
            selected_stems=args.selected_stems,
            add_params=get_add_params(args)
        )
    elif args.mode == "custom_separate":
        separator.custom_separate(
            input_files=args.input,
            output_dir=args.output_dir,
            output_format=args.output_format,
            template=args.template,
            model_type=args.model_type,
            ckpt=args.checkpoint_path,
            conf=args.config_path,
            extract_instrumental=args.extract_instrumental,
            use_spec_invert=args.use_spec_invert,
            selected_stems=args.selected_stems,
            add_params=get_add_params(args)
        )
    elif args.mode == "auto_ensemble":
        if args.preset:
            flow = json.loads(Path(args.preset).read_text("utf-8"))
        elif args.flow:
            flow = []
            for params in args.flow:
                list_values_param = params.split(":")
                if len(list_values_param) == 4:
                    flow.append([str(list_values_param[0]), str(list_values_param[1]), tobool(list_values_param[2]), float(list_values_param[3])])
                else:
                    raise ValueError()
        separator.auto_ensemble(
            input_file=args.input,
            output_dir=args.output_dir, 
            flow=flow,
            template=args.template,
            etype=args.ensemble_type,
            output_format=args.output_format,
            use_spec_invert=args.use_spec_invert, 
            save_primary_stems=args.save_primary_stems
        )
    elif args.mode == "manual_ensemble":
        separator.manual_ensemble(
            input_files=args.input,
            output_dir=args.output_dir,
            weights=args.weights,
            template=args.template,
            etype=args.ensemble_type,
            output_format=args.output_format
        )
    elif args.mode == "subtract":
        separator.subtract(
            audio1=args.input_1,
            audio2=args.input_2,
            output_dir=args.output_dir,
            output_format=args.output_format,
            use_spec_invert=args.spec_invert,
            template=args.template
        )
    elif args.mode == "info":
        if args.update:
            separator.update_info()
        elif args.download:
            separator.download(args.model_name)
        elif args.clear_cache:
            separator.cache_dir.unlink(missing_ok=True)
        else:
            separator.show_info(args.limit, args.stem, args.only_installed)
    
