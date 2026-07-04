from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))
from extra_utils import hf_spaces_gpu, extra_clear_torch_cache, nuclear_clear_model, emergency_ram_clear, print_current_device
if __package__:
    from .hubert_manager import get_hubert, download_hubert, huberts_fairseq
    from .pipeline import VC
    from .config import Config
else:
    from vbach_lib.hubert_manager import get_hubert, download_hubert, huberts_fairseq
    from vbach_lib.pipeline import VC
    from vbach_lib.config import Config

def lazy_synthesizer_import():
    if __package__:
        from .algorithm.synthesizers import Synthesizer as module
    else:
        from vbach_lib.algorithm.synthesizers import Synthesizer as module
    return module

def lazy_hubert_transformers_import():
    from transformers import HubertModel as module
    return module

def lazy_hubert_fairseq_import():
    if __package__:
        from .fairseq import load_model as module
    else:
        from vbach_lib.fairseq import load_model as module
    return module

from pathlib import Path
import traceback
from audio import read, write, split_channels, split_mid_side, multi_channel_array_from_arrays, output_formats, stereo_to_mono, reshape, mix_arrays, get_audio_files_from_list, check, get_metadata, check_taglib_not_installed
from inference import PathsNotSpecified, PathNotExist, PathNotSpecified, FileIsNotAudio
from i18n import _i18n
from namer import Namer
from args_parser import parse_vbach_args
import numpy as np
import torch
from torch import nn
import gc
from copy import deepcopy

class VbachModelNotFound(Exception): pass

stereo_modes = ("mono", "left/right", "sim/dif")

def load_audio(path: str | Path, sr: int, stereo_mode: str = stereo_modes[0]):
    mixtures = []
    add_text = []
    if stereo_mode == "mono":
        mix, _sr = read(path, sr, mono=True, flatten=True)
        mixtures.append(mix)
        add_text.append(None)
    elif stereo_mode == "left/right":
        mix, _sr = read(path, sr, mono=False)
        mixtures.extend(split_channels(mix))
        add_text.extend(["[L]", "[R]"])
    elif stereo_mode == "sim/dif":
        mix, _sr = read(path, sr, mono=False)
        center, stereo_base = split_mid_side(mix, var=3, sr=sr)
        phantom_center = stereo_to_mono(center, to_flatten=True)
        stereo_base_left, stereo_base_right = split_channels(stereo_base)
        mixtures.extend([phantom_center, stereo_base_left, stereo_base_right])
        add_text.extend(["[Sim]", "[Dif-L]", "[Dif-R]"])
    return mixtures, add_text

def post_process_audio(mixtures: list[np.ndarray], target_sr: int, stereo_mode: str = stereo_modes[0]):
    if stereo_mode == "mono":
        return reshape(mixtures[0], ("channels", "samples"))
    elif stereo_mode == "left/right":
        dtype = mixtures[0].dtype
        return multi_channel_array_from_arrays(*mixtures, index=1, dtype=dtype)
    elif stereo_mode == "sim/dif":
        sim, dif_l, dif_r = mixtures
        dtype = sim.dtype
        sim_channel = multi_channel_array_from_arrays(sim, sim, index=1, dtype=dtype)
        dif_channel = multi_channel_array_from_arrays(dif_l, dif_r, index=1, dtype=dtype)
        return mix_arrays([sim_channel, dif_channel], [target_sr, target_sr], target_sr, index=1, dtype=dtype)[0]

class VbachConverter:
    def __init__(self):
        self.config = Config()
        self.hubert_model = None 
        self.cpt = self.version = self.net_g = self.tgt_sr = self.vc = self.use_f0 = self.vocoder = self.emb_weight_shape = self.required_keys = self.missing_keys = self.text_enc_hidden_dim = None

    def load_hubert(self, name: str, use_transformers: bool):
        if use_transformers:
            HubertModel = lazy_hubert_transformers_import()
            class HubertModelWithFinalProj(HubertModel):
                """Hubert модель с финальной проекцией"""
                
                def __init__(self, config):
                    super().__init__(config)
                    self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)
            model_path = get_hubert(name, True)
            self.hubert_model = HubertModelWithFinalProj.from_pretrained(model_path)
            self.hubert_model = self.hubert_model.to(self.config.device)
        else:
            load_model = lazy_hubert_fairseq_import()
            model_path = get_hubert(name, False)
            self.hubert_model = load_model(model_path)
            self.hubert_model = self.hubert_model.to(self.config.device)
            self.hubert_model = self.hubert_model.half() if self.config.is_half else self.hubert_model.float()
            self.hubert_model.eval()
        print(_i18n("hubert_checkpoint_loaded")+": "+name)

    def unload_hubert(self):
        self.hubert_model = self.hubert_model.cpu()
        self.hubert_model = None
        gc.collect()
        extra_clear_torch_cache()
        nuclear_clear_model()
        emergency_ram_clear()

    def unload_model(self):
        self.net_g = self.net_g.cpu()
        del self.cpt, self.version, self.net_g, self.tgt_sr, self.vc, self.use_f0, self.vocoder, self.emb_weight_shape, self.required_keys, self.missing_keys, self.text_enc_hidden_dim
        self.cpt = self.version = self.net_g = self.tgt_sr = self.vc = self.use_f0 = self.vocoder = self.emb_weight_shape = self.required_keys = self.missing_keys = self.text_enc_hidden_dim = None
        extra_clear_torch_cache()
        nuclear_clear_model()
        emergency_ram_clear()

    def clear_gpu_cache(self):
        gc.collect()
        torch.clear_autocast_cache()
        if self.config.device.type == "mps":
            torch.mps.empty_cache()
        if self.config.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

    def get_vc(self, model_path: str | Path, use_transformers: bool):
        Synthesizer = lazy_synthesizer_import()
        self.cpt = torch.load(model_path, map_location="cpu", weights_only=True)
        self.required_keys = ["config", "weight"]
        self.missing_keys = [key for key in self.required_keys if key not in self.cpt]

        self.tgt_sr = self.cpt["config"][-1]

        self.emb_weight_shape = self.cpt["weight"]["emb_g.weight"].shape
        self.cpt["config"][-3] = self.emb_weight_shape[0]

        self.use_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")
        self.vocoder = self.cpt.get("vocoder", "HiFi-GAN")

        self.text_enc_hidden_dim = 768 if self.version == "v2" else 256

        self.net_g = Synthesizer(
            *self.cpt["config"],
            use_f0=self.use_f0,
            text_enc_hidden_dim=self.text_enc_hidden_dim,
            vocoder=self.vocoder,
        )

        if hasattr(self.net_g, "enc_q"):
            del self.net_g.enc_q
        else:
            pass

        self.net_g.load_state_dict(
            self.cpt["weight"], strict=False
        )
        self.net_g.eval()

        self.net_g = self.net_g.to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.vc = VC(self.tgt_sr, self.config, use_transformers)
        print(_i18n("checkpoint_loaded")+": "+Path(model_path).name)

    @hf_spaces_gpu # (duration=120) Для спейса LongQuota / длинная квота на HuggingFace ZeroGPU (по умолчанию 60 секунд)
    def convert_audio(
        self,
        audio_input: str | Path | list[str | Path],
        output_dir: str | Path,
        model_path: str,
        index_path: str,
        pitch: int = 0,
        f0_method: str = "rmvpe+",
        index_rate: float = 0.75,
        volume_envelope: float = 0.25,
        protect: float = 0.33,
        hop_length: int = 128,
        embedder_model: str = "hubert_base",
        use_transformers: bool = False,
        output_format: str = output_formats[0],
        stereo_mode: str = stereo_modes[0],
        f0_min: int = 50,
        f0_max: int = 1100,
        chunk_duration: int = 7,
        template: str = "MODEL_NAME_F0METHOD_PITCH",
        **kwargs,
    ):
        print_current_device(self.config.device)
        template = Namer.sanitize(template)
        template = Namer.dedup_template(template, keys=["NAME", "F0METHOD", "MODEL", "PITCH"])
        template = Namer.short(template, length=40)

        if not model_path:
            raise VbachModelNotFound()
    
        self.get_vc(model_path, use_transformers)
        model_name = Path(model_path).stem
    
        if not self.hubert_model:
            self.load_hubert(embedder_model, use_transformers)

        if not output_dir:
            output_dir = ""

        output_dir = Path(output_dir)

        input_valid_files = get_audio_files_from_list(audio_input, only_files=False)
        if not input_valid_files:
            raise PathsNotSpecified(_i18n("paths_not_specified"))

        total = len(input_valid_files)

        print(_i18n("f0_method")+": "+f0_method)

        processed_audios = []

        for i, audio_input_path in enumerate(input_valid_files, start=1):
            try:
                input_file_name = Path(audio_input_path).stem
                mixtures, add_text = load_audio(audio_input_path, 16000, stereo_mode)
                metadata = get_metadata(audio_input_path)
                print(_i18n("loaded_mix")+": "+Path(audio_input_path).name)
                converted_mixtures = []

                for mix, add_text_progress in zip(mixtures, add_text):
                    audio_max = np.abs(mix).max() / 0.95
                    if audio_max > 1:
                        mix /= audio_max
                    audio_opt = self.vc.pipeline(
                        model=self.hubert_model,
                        net_g=self.net_g,
                        sid=0,
                        audio=mix,
                        pitch=pitch,
                        f0_method=f0_method,
                        f0_file=None,
                        hop_length=hop_length,
                        file_index=index_path,
                        index_rate=index_rate,
                        pitch_guidance=self.use_f0,
                        volume_envelope=volume_envelope,
                        version=self.version,
                        protect=protect,
                        tgt_sr=self.tgt_sr,
                        f0_min=f0_min,
                        f0_max=f0_max,
                        chunk_duration=chunk_duration,
                        add_text_channel=add_text_progress,
                        add_text_custom=f"{i}/{total} {_i18n('files')}",
                    )
                    converted_mixtures.append(audio_opt)
                custom_name = Namer.template(
                    template,
                    PITCH=pitch,
                    F0METHOD=f0_method,
                    MODEL=model_name,
                    NAME=Namer.short_input_name_template(template, PITCH=pitch, F0METHOD=f0_method, MODEL=model_name, NAME=input_file_name)
                )
                new_metadata = {}
                if metadata:
                    new_metadata = deepcopy(metadata)
                    if "TITLE" in metadata:
                        new_metadata["TITLE"] = f"[{f0_method} / {pitch} / {stereo_mode}] {metadata['TITLE']}"
                    else:
                        new_metadata["TITLE"] = f"[{f0_method} / {pitch} / {stereo_mode}] {input_file_name}"

                    if "ARTIST" in metadata:
                        new_metadata["ARTIST"] = f"{metadata['ARTIST']} [{model_name}]"
                    else:
                        new_metadata["ARTIST"] = f"{model_name}"
                else:
                    new_metadata["TITLE"] = f"[{f0_method} / {pitch} / {stereo_mode}] {input_file_name}"
                    new_metadata["ARTIST"] = f"{model_name}"

                processed_audios.append(write(Namer.iter(output_dir / f"{custom_name}.{output_format}"), post_process_audio(converted_mixtures, self.tgt_sr, stereo_mode), self.tgt_sr, 320, False, new_metadata))
            except Exception as e:
                traceback.print_exc()

        self.unload_model()
        self.unload_hubert()

        return processed_audios

    @hf_spaces_gpu # (duration=120) Для спейса LongQuota / длинная квота на HuggingFace ZeroGPU (по умолчанию 60 секунд)
    def convert_audio_custom_f0(
        self,
        audio_input: str | Path,
        output_dir: str | Path,
        model_path: str,
        index_path: str,
        pitch: int = 0,
        f0_file: str | Path = None,
        index_rate: float = 0.75,
        volume_envelope: float = 0.25,
        protect: float = 0.33,
        embedder_model: str = "hubert_base",
        use_transformers: bool = False,
        output_format: str = output_formats[0],
        f0_min: int = 50,
        f0_max: int = 1100,
        chunk_duration: int = 7,
        template: str = "MODEL_NAME_F0METHOD_PITCH",
        **kwargs,
    ):
        print_current_device(self.config.device)
        template = Namer.sanitize(template)
        template = Namer.dedup_template(template, keys=["NAME", "F0METHOD", "MODEL", "PITCH"])
        template = Namer.short(template, length=40)

        if not model_path:
            raise VbachModelNotFound()

        model_name = Path(model_path).stem
        self.get_vc(model_path, use_transformers)
    
        if not self.hubert_model:
            self.load_hubert(embedder_model, use_transformers)

        if not output_dir:
            output_dir = ""

        output_dir = Path(output_dir)
        output_path = None

        print(_i18n("f0_method")+": "+"custom")

        try:
            if not audio_input:
                raise PathNotSpecified(_i18n("path_not_specified"))
            audio_input = Path(audio_input)
            if not audio_input.exists():
                raise PathNotExist(_i18n("path_not_exist"))
            if check(audio_input):
                input_file_name = Path(audio_input).stem
                mix, sr = read(audio_input, sr=16000, mono=True, flatten=True)
                metadata = get_metadata(audio_input)
                print(_i18n("loaded_mix")+": "+Path(audio_input).name)
            else:
                raise FileIsNotAudio(_i18n("file_is_not_audio", path=audio_input))

            audio_max = np.abs(mix).max() / 0.95
            if audio_max > 1:
                mix /= audio_max
            audio_opt = self.vc.pipeline(
                model=self.hubert_model,
                net_g=self.net_g,
                sid=0,
                audio=mix,
                pitch=pitch,
                f0_method=None,
                f0_file=f0_file,
                hop_length=0,
                file_index=index_path,
                index_rate=index_rate,
                pitch_guidance=self.use_f0,
                volume_envelope=volume_envelope,
                version=self.version,
                protect=protect,
                tgt_sr=self.tgt_sr,
                f0_min=f0_min,
                f0_max=f0_max,
                chunk_duration=chunk_duration,
                add_text_channel="",
                add_text_custom=f"{_i18n('custom_f0')}",
            )
            custom_name = Namer.template(
                template,
                PITCH=pitch,
                F0METHOD="custom",
                MODEL=model_name,
                NAME=Namer.short_input_name_template(template, PITCH=pitch, F0METHOD="custom", MODEL=model_name, NAME=input_file_name)
            )
            new_metadata = {}
            if metadata:
                new_metadata = deepcopy(metadata)
                if "TITLE" in metadata:
                    new_metadata["TITLE"] = f"[custom / {pitch} / mono] {metadata['TITLE']}"
                else:
                    new_metadata["TITLE"] = f"[custom / {pitch} / mono] {input_file_name}"

                if "ARTIST" in metadata:
                    new_metadata["ARTIST"] = f"{metadata['ARTIST']} [{model_name}]"
                else:
                    new_metadata["ARTIST"] = f"{model_name}"

            else:
                new_metadata["TITLE"] = f"[custom / {pitch} / mono] {input_file_name}"
                new_metadata["ARTIST"] = f"{model_name}"

            output_path = write(Namer.iter(output_dir / f"{custom_name}.{output_format}"), audio_opt, self.tgt_sr, 320, False, new_metadata)
        except Exception as e:
            traceback.print_exc()

        self.unload_model()
        self.unload_hubert()

        return output_path

if __name__ == "__main__":
    check_taglib_not_installed()
    vbach = VbachConverter()
    args = parse_vbach_args()
    if args.mode == "infer":
        vbach.convert_audio(
            audio_input=args.input,
            output_dir=args.output_dir,
            model_path=args.checkpoint_path,
            index_path=args.index_path,
            pitch=args.pitch,
            f0_method=args.f0_method,
            index_rate=args.index_rate,
            volume_envelope=args.volume_envelope,
            protect=args.protect,
            hop_length=args.hop_length,
            embedder_model=args.embedder,
            use_transformers=args.use_transformers,
            output_format=args.output_format,
            stereo_mode=args.stereo_mode,
            f0_min=args.f0_min,
            f0_max=args.f0_max,
            chunk_duration=args.chunk_duration,
            template=args.template
        )
    elif args.mode == "infer_custom_f0":
        vbach.convert_audio_custom_f0(
            audio_input=args.input,
            output_dir=args.output_dir,
            model_path=args.checkpoint_path,
            index_path=args.index_path,
            pitch=args.pitch,
            f0_file=args.f0_file,
            index_rate=args.index_rate,
            volume_envelope=args.volume_envelope,
            protect=args.protect,
            embedder_model=args.embedder,
            use_transformers=args.use_transformers,
            output_format=args.output_format,
            stereo_mode=args.stereo_mode,
            f0_min=args.f0_min,
            f0_max=args.f0_max,
            chunk_duration=args.chunk_duration,
            template=args.template
        )
    elif args.mode == "download_hubert":
        download_hubert(args.embedder, args.use_transformers)