import pyworld
import numpy as np
import parselmouth
from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))
from functools import lru_cache
from scipy import signal
import librosa
import torch
import torchcrepe
from i18n import _i18n
from audio import read
from namer import Namer

import json
from extra_utils import dw_file, nuclear_clear_model, emergency_ram_clear, extra_clear_torch_cache, hf_spaces_gpu
from args_parser import parse_f0_extract
if __package__:
    from .predictors.FCPE import FCPEF0Predictor
    from .predictors.RMVPE import RMVPE0Predictor
    from .predictors.HPA_RMVPE import HPA_RMVPE
else:
    from predictors.FCPE import FCPEF0Predictor
    from predictors.RMVPE import RMVPE0Predictor
    from predictors.HPA_RMVPE import HPA_RMVPE

BASE_DIR = Path(__file__).resolve().parent / "predictors"
BASE_DIR.mkdir(parents=True, exist_ok=True)
RMVPE_PATH = BASE_DIR / "rmvpe.pt"
HPA_RMVPE_PATH = BASE_DIR / "hpa_rmvpe.pt"
FCPE_PATH = BASE_DIR / "fcpe.pt"

f0_methods = (
    "rmvpe",
    "hpa-rmvpe",
    "fcpe",
    "fcpe+unvoiced_rmvpe",
    "mangio-crepe",
    "mangio-crepe-tiny",
    "mangio-crepe+unvoiced_rmvpe", 
    "mangio-crepe-tiny+unvoiced_rmvpe",
    "harvest",
    "pm",
    "pyin",
)
crepe_like_f0_methods = (f0_methods[4], f0_methods[5], f0_methods[6], f0_methods[7], f0_methods[10])

class UnknownF0Method(Exception): pass
class F0CurveNotFound(Exception): pass

requirements = {
    "rmvpe": ["https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/predictors/rmvpe.pt?download=true", RMVPE_PATH],
    "hpa_rmvpe": ["https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/predictors/hpa_rmvpe.pt?download=true", HPA_RMVPE_PATH],
    "fcpe": ["https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/predictors/fcpe.pt?download=true", FCPE_PATH]
}

def download_requirements(f0_method: str):
    if "+unvoiced_rmvpe" in f0_method:
        url1, path1 = requirements.get("rmvpe", (None, None))
        if all([url1, path1]):
            if not path1.exists():
                dw_file(url1, path1)
    url, path = requirements.get(f0_method.replace("+unvoiced_rmvpe", ""), (None, None))
    if all([url, path]):
        if not path.exists():
            dw_file(url, path)

input_audio_path2wav = {}

@lru_cache(maxsize=128)
def get_harvest_f0(
    input_audio_path: str, 
    fs: int, 
    f0max: float, 
    f0min: float, 
    frame_period: float
) -> np.ndarray:
    """
    Получить F0 с помощью Harvest
    
    Args:
        input_audio_path: Путь к аудиофайлу
        fs: Частота дискретизации
        f0max: Максимальная частота F0
        f0min: Минимальная частота F0
        frame_period: Период кадра
    
    Returns:
        Массив F0
    """
    audio = input_audio_path2wav[input_audio_path]
    f0, t = pyworld.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0

def f0_extract(
    x,
    sample_rate,
    p_len,
    f0_method,
    crepe_hop_length,
    window,
    device,
    time_step,
    is_half,
    f0_min=50,
    f0_max=1100,
):
    global input_audio_path2wav

    download_requirements(f0_method)

    if f0_method in ["mangio-crepe", "mangio-crepe-tiny"]:
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        audio = torch.from_numpy(x).to(device, copy=True).unsqueeze(0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        pitch_ = torchcrepe.predict(
            audio,
            sample_rate,
            crepe_hop_length,
            f0_min,
            f0_max,
            "tiny" if f0_method == "mangio-crepe-tiny" else "full",
            batch_size=crepe_hop_length * 2,
            device=device,
            pad=True,
        )

        p_len = p_len or x.shape[0] // crepe_hop_length
        source = np.array(pitch_.squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source,
        )
        f0 = np.nan_to_num(target)

    if f0_method in ["mangio-crepe+unvoiced_rmvpe", "mangio-crepe-tiny+unvoiced_rmvpe"]:
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        audio = torch.from_numpy(x).to(device, copy=True).unsqueeze(0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        pitch_ = torchcrepe.predict(
            audio,
            sample_rate,
            crepe_hop_length,
            f0_min,
            f0_max,
            "tiny" if f0_method == "mangio-crepe-tiny+unvoiced_rmvpe" else "full",
            batch_size=crepe_hop_length * 2,
            device=device,
            pad=True,
        )

        p_len = p_len or x.shape[0] // crepe_hop_length
        source = np.array(pitch_.squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source,
        )
        f0_1 = np.nan_to_num(target)
        model_2 = RMVPE0Predictor(
            RMVPE_PATH, is_half=is_half, device=device
        )
        f0_2: np.ndarray = model_2.infer_from_audio_with_pitch(
            x, thred=0.03, f0_min=f0_min, f0_max=f0_max
        )
        f0_1_len = f0_1.size
        f0_2_len = f0_2.size
        max_len = max(f0_1_len, f0_2_len)
        f0_1 = np.pad(f0_1, pad_width=(0, max_len - f0_1_len), mode='constant', constant_values=0)
        f0_2 = np.pad(f0_2, pad_width=(0, max_len - f0_2_len), mode='constant', constant_values=0)
        condition = (f0_2 == 0) & (f0_1 != 0)
        f0 = np.where(condition, 0, f0_1)

    elif f0_method == "pyin":
        f0, *_ = librosa.pyin(
            x.astype(np.float32),
            sr=sample_rate,
            fmin=f0_min,
            fmax=f0_max,
            hop_length=crepe_hop_length,
        )
        source = np.array(f0)
        source[source < 0.001] = np.nan
        f0 = np.nan_to_num(
            np.interp(
                np.arange(0, len(source) * p_len, len(source)) / p_len,
                np.arange(0, len(source)),
                source,
            )
        )

    elif f0_method == "fcpe":
        model = FCPEF0Predictor(
            FCPE_PATH,
            f0_min=int(f0_min),
            f0_max=int(f0_max),
            dtype=torch.float32,
            device=device,
            sample_rate=sample_rate,
            threshold=0.03,
        )
        f0 = model.compute_f0(x, p_len=p_len or len(x) // window)
        del model

    elif f0_method == "fcpe+unvoiced_rmvpe":
        model_1 = FCPEF0Predictor(
            FCPE_PATH,
            f0_min=int(f0_min),
            f0_max=int(f0_max),
            dtype=torch.float32,
            device=device,
            sample_rate=sample_rate,
            threshold=0.03,
        )
        f0_1: np.ndarray = model_1.compute_f0(x, p_len=p_len or len(x) // window)
        model_2 = RMVPE0Predictor(
            RMVPE_PATH, is_half=is_half, device=device
        )
        f0_2: np.ndarray = model_2.infer_from_audio_with_pitch(
            x, thred=0.03, f0_min=f0_min, f0_max=f0_max
        )
        f0_1_len = f0_1.size
        f0_2_len = f0_2.size
        max_len = max(f0_1_len, f0_2_len)
        f0_1 = np.pad(f0_1, pad_width=(0, max_len - f0_1_len), mode='constant', constant_values=0)
        f0_2 = np.pad(f0_2, pad_width=(0, max_len - f0_2_len), mode='constant', constant_values=0)
        condition = (f0_2 == 0) & (f0_1 != 0)
        f0 = np.where(condition, 0, f0_1)
        del model_1, model_2

    elif f0_method == "harvest":
        input_audio_path2wav = {}
        input_audio_path2wav["test.mp3"] = x.astype(np.double)
        f0 = get_harvest_f0("test.mp3", sample_rate, f0_max, f0_min, 10)
        f0 = signal.medfilt(f0, 3)

    elif f0_method == "pm":
        f0 = (
            parselmouth.Sound(x, sample_rate)
            .to_pitch_ac(
                time_step=time_step / 1000,
                voicing_threshold=0.6,
                pitch_floor=f0_min,
                pitch_ceiling=f0_max,
            )
            .selected_array["frequency"]
        )
        pad_size: int = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            f0 = np.pad(
                f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
            )

    elif f0_method == "rmvpe":
        model = RMVPE0Predictor(
            RMVPE_PATH, is_half=is_half, device=device
        )
        f0 = model.infer_from_audio_with_pitch(
            x, thred=0.03, f0_min=f0_min, f0_max=f0_max
        )
        del model
    elif f0_method == "hpa-rmvpe":
        model = HPA_RMVPE(
            HPA_RMVPE_PATH, is_half=is_half, device=device
        )
        f0 = model.infer_from_audio_with_pitch(
            x, thred=0.03, f0_min=f0_min, f0_max=f0_max
        )
        del model
    else:
        raise UnknownF0Method(_i18n("unknown_f0_method", method=f0_method))
    
    return f0
    
@hf_spaces_gpu
def f0_extract_and_write(input_audio: str | Path, f0_method: str = f0_methods[0], f0_min: int = 50, f0_max: int = 1100, output_path: str | Path = None):
    path = Path(input_audio)
    sample_rate: int = 16000
    audio, sr_ = read(path, sr=sample_rate, mono=True, flatten=True)
    if not output_path:
        output_path = Path(Namer.iter(path.with_suffix(".json")))
    else:
        output_path = Path(output_path)
    hop_length = 128
    window: int = 160
    time_step: float = window / sample_rate * 1000
    p_len = len(audio) // window
    f0 = f0_extract(audio, sample_rate, p_len, f0_method, hop_length, window, "cuda" if torch.cuda.is_available() else "cpu", time_step, False, f0_min, f0_max)
    f0_info = {
        "method": f0_method,
        "sample_rate": sample_rate,
        "window": window,
        "p_len": p_len,
        "freqs": [freq for freq in f0.tolist()]
    }
    output_path.write_text(json.dumps(f0_info, indent=4), encoding="utf-8")
    del f0_info, f0
    extra_clear_torch_cache()
    nuclear_clear_model()
    emergency_ram_clear()
    return output_path.as_posix()

def f0_import(input_json: str | Path):
    path = Path(input_json)
    f0_info = json.loads(path.read_text("utf-8"))
    f0_list = f0_info.get("freqs")
    if f0_list:
        return np.array(f0_list, dtype=np.float32)
    else:
        raise F0CurveNotFound(_i18n("f0_curve_not_found"))
    
if __name__ == "__main__":
    args = parse_f0_extract()
    f0_extract_and_write(
        input_audio=args.input,
        f0_method=args.f0_method,
        f0_min=args.f0_min,
        f0_max=args.f0_max,
        output_path=args.output_path
    )
