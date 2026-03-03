import sys
import json
import numpy as np
import torch
import torch.nn as nn
import yaml
import librosa
import torch.nn.functional as F
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from typing import Dict, List, Tuple, Any, List, Optional

def load_config(model_type: str, config_path: str) -> Any:
    try:
        with open(config_path, "r") as f:
            if model_type == "htdemucs":
                config = OmegaConf.load(config_path)
            else:
                config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
                if hasattr(config.audio, "new_chunk_size"):
                    if hasattr(config.audio, "chunk_size"):
                        config.audio.chunk_size = config.audio.new_chunk_size
                if hasattr(config.audio, "new_dim_t"):
                    if hasattr(config.audio, "dim_t"):
                        config.audio.dim_t = config.audio.new_dim_t
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")

def get_model_from_config(model_type: str, config_path: str) -> Tuple:
    config = load_config(model_type, config_path)

    if model_type == "mdx23c":
        from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == "mdxnet":
        from models.mdx_net import MDXNet
        model = MDXNet(**dict(config.model))
    elif model_type == "vr":
        from models.vr_arch import VRNet
        model = VRNet(**dict(config.model))
    elif model_type == "htdemucs":
        from models.demucs4ht import get_model
        model = get_model(config)
    elif model_type == "mel_band_roformer":
        if hasattr(config, "windowed"):
            from models.windowed_roformer.model import MelBandRoformerWSA
            model = MelBandRoformerWSA(**dict(config.model))
        elif hasattr(config, "conformer"):
            from models.bs_roformer import MelBandConformer
            model = MelBandConformer(**dict(config.model))
        else:
            from models.bs_roformer import MelBandRoformer
            model = MelBandRoformer(**dict(config.model))
    elif model_type == "bs_roformer":
        if hasattr(config, "sw"):
            from models.bs_roformer import BSRoformer_SW
            model = BSRoformer_SW(**dict(config.model))
        elif hasattr(config, "fno"):
            from models.bs_roformer import BSRoformer_FNO
            model = BSRoformer_FNO(**dict(config.model))
        elif hasattr(config, "hyperace"):
            from models.bs_roformer import BSRoformerHyperACE
            model = BSRoformerHyperACE(**dict(config.model))
        elif hasattr(config, "hyperace2"):
            from models.bs_roformer import BSRoformerHyperACE_2
            model = BSRoformerHyperACE_2(**dict(config.model))
        elif hasattr(config, "conformer"):
            from models.bs_roformer import BSConformer
            model = BSConformer(**dict(config.model))
        elif hasattr(config, "conditional"):
            from models.bs_roformer import BSRoformer_Conditional
            model = BSRoformer_Conditional(**dict(config.model))
        elif hasattr(config, "unwa_inst_large_2"):
            from models.bs_roformer import BSRoformer_2
            model = BSRoformer_2(**dict(config.model))
        else:
            from models.bs_roformer import BSRoformer
            model = BSRoformer(**dict(config.model))
    elif model_type == "bandit":
        from models.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
        model = MultiMaskMultiSourceBandSplitRNNSimple(**config.model)
    elif model_type == "bandit_v2":
        from models.bandit_v2.bandit import Bandit
        model = Bandit(**config.kwargs)
    elif model_type == "scnet_unofficial":
        from models.scnet_unofficial import SCNet
        model = SCNet(**config.model)
    elif model_type == "scnet":
        from models.scnet import SCNet
        model = SCNet(**config.model)
    elif model_type == 'scnet_masked':
        from models.scnet.scnet_masked import SCNet
        model = SCNet(**config.model)
    elif model_type == 'scnet_tran':
        from models.scnet.scnet_tran import SCNet_Tran
        model = SCNet_Tran(**config.model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model, config

def _getWindowingArray(window_size: int, fade_size: int) -> torch.Tensor:
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)

    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window

def demix_mdxnet(
    config: Any,
    model: Any,
    mix: np.ndarray,
    device: torch.device,
    pbar: bool = False,
) -> Dict[str, np.ndarray]:
    mix_tensor = torch.tensor(mix, dtype=torch.float32)
    inv_mix_tensor = torch.tensor(-mix, dtype=torch.float32)

    num_overlap = config.inference.num_overlap
    denoise = config.inference.denoise
    stem_name = model.primary_stem
    if denoise:
        processed_wav = model.process_wave(mix_tensor, device, num_overlap, pbar=pbar)
        inv_processed_wav = model.process_wave(
            inv_mix_tensor, device, num_overlap, pbar=pbar
        )
        result = processed_wav.cpu().numpy()
        inv_result = inv_processed_wav.cpu().numpy()
        result_separation = (result + -inv_result) * 0.5
    else:
        processed_wav = model.process_wave(mix_tensor, device, num_overlap, pbar=pbar)
        result_separation = processed_wav.cpu().numpy()

    result_separation = np.nan_to_num(
        result_separation, nan=0.0, posinf=0.0, neginf=0.0
    )

    return {stem_name: result_separation}


def demix_vr(
    config: Any,
    model: Any,
    mix: np.ndarray,
    device: torch.device,
    pbar: bool = False,
) -> Dict[str, np.ndarray]:
    return model.demix(
        mix, config.audio.sample_rate, device, config.inference.aggression
    )

def demix_demucs(config, model, mix, device, pbar=False):
    mix = torch.tensor(mix, dtype=torch.float32)
    chunk_size = config.training.samplerate * config.training.segment
    num_instruments = len(config.training.instruments)
    num_overlap = config.inference.num_overlap
    step = chunk_size // num_overlap
    fade_size = chunk_size // 10
    windowing_array = _getWindowingArray(chunk_size, fade_size)

    batch_size = config.inference.batch_size
    use_amp = getattr(config.training, "use_amp", True)

    with torch.cuda.amp.autocast(enabled=use_amp):
        with torch.inference_mode():
            req_shape = (num_instruments,) + mix.shape
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)

            i = 0
            batch_data = []
            batch_locations = []

            while i < mix.shape[1]:
                part = mix[:, i : i + chunk_size].to(device)
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
                    x = model(arr)

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

                    processed = min(i, mix.shape[1])
                    total = mix.shape[1]
                    sys.stdout.write(
                        json.dumps(
                            {"processing": {"processed": processed, "total": total}}
                        )
                        + "\n"
                    )
                    sys.stdout.flush()

                    batch_data.clear()
                    batch_locations.clear()

            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

    if num_instruments <= 1:
        return estimated_sources
    else:
        instruments = config.training.instruments
        return {k: v for k, v in zip(instruments, estimated_sources)}


def demix_generic(
    config: ConfigDict,
    model: torch.nn.Module,
    mix: torch.Tensor,
    device: torch.device,
    pbar: bool = False,
) -> Dict[str, np.ndarray]:
    mix = torch.tensor(mix, dtype=torch.float32)
    chunk_size = config.audio.chunk_size
    instruments = prefer_target_instrument(config)
    num_instruments = len(instruments)
    num_overlap = config.inference.num_overlap

    fade_size = chunk_size // 10
    step = chunk_size // num_overlap
    border = chunk_size - step
    length_init = mix.shape[-1]
    windowing_array = _getWindowingArray(chunk_size, fade_size)

    if length_init > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode="reflect")

    batch_size = config.inference.batch_size
    use_amp = getattr(config.training, "use_amp", True)

    with torch.cuda.amp.autocast(enabled=use_amp):
        with torch.inference_mode():
            req_shape = (num_instruments,) + mix.shape
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)

            i = 0
            batch_data = []
            batch_locations = []

            while i < mix.shape[1]:
                part = mix[:, i : i + chunk_size].to(device)
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
                    x = model(arr)

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

                    processed = min(i, mix.shape[1])
                    total = mix.shape[1]
                    sys.stdout.write(
                        json.dumps(
                            {"processing": {"processed": processed, "total": total}},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    sys.stdout.flush()

                    batch_data.clear()
                    batch_locations.clear()

            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            if length_init > 2 * border and border > 0:
                estimated_sources = estimated_sources[..., border:-border]

    return {k: v for k, v in zip(instruments, estimated_sources)}


def demix(
    config: ConfigDict,
    model: torch.nn.Module,
    mix: np.ndarray,
    device: torch.device,
    model_type: str,
    pbar: bool = False,
) -> Dict[str, np.ndarray]:
    if model_type == "vr":
        return demix_vr(config, model, mix, device, pbar)
    elif model_type == "mdxnet":
        return demix_mdxnet(config, model, mix, device, pbar)
    elif model_type == "htdemucs":
        return demix_demucs(config, model, mix, device, pbar)
    else:
        return demix_generic(config, model, mix, device, pbar)

def prefer_target_instrument(config: ConfigDict) -> List[str]:
    if config.training.get("target_instrument"):
        return [config.training.target_instrument]
    else:
        return config.training.instruments

def prefer_target_instrument_test(
    config: ConfigDict, selected_instruments: Optional[List[str]] = None
) -> List[str]:
    available_instruments = config.training.instruments

    if selected_instruments is not None:
        return [
            instr for instr in selected_instruments if instr in available_instruments
        ]
    elif config.training.get("target_instrument"):
        return [config.training.target_instrument]
    else:
        return available_instruments
