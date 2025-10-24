
import torch
import numpy as np
import librosa
from multiprocessing import cpu_count
from vbach.lib.fairseq import load_model_ensemble_and_task, load_checkpoint_to_cpu

from vbach.lib.algorithm.synthesizers import Synthesizer
from .pipeline import VC

from vbach.utils.audio_utils import Audio

from vbach.utils.remove_center import remove_center

audio = Audio()

def overlay_mono_on_stereo(mono_audio, stereo_audio, gain=0.5):
    if mono_audio is None or stereo_audio is None:
        raise ValueError("Input audio arrays cannot be None")
    
    # Ensure float32 for processing
    mono_audio = mono_audio.astype(np.float32)
    stereo_audio = stereo_audio.astype(np.float32)

    # Convert mono to stereo if needed
    if mono_audio.ndim == 1:
        mono_audio = np.vstack([mono_audio, mono_audio])
    elif mono_audio.shape[0] == 1:
        mono_audio = np.vstack([mono_audio[0], mono_audio[0]])

    if mono_audio.shape[0] != 2 or stereo_audio.shape[0] != 2:
        raise ValueError("Shapes must be (2, N)")

    min_len = min(mono_audio.shape[1], stereo_audio.shape[1])
    if min_len == 0:
        raise ValueError("Audio arrays cannot be empty")

    mono_audio = mono_audio[:, :min_len]
    stereo_audio = stereo_audio[:, :min_len]
    
    result = stereo_audio + mono_audio * gain

    # Normalize to prevent clipping
    max_amp = np.max(np.abs(result))
    if max_amp > 0:
        result /= max_amp

    # Convert back to int16 for output (if needed)
    result = (result * 32767).astype(np.int16)

    return result

def load_audio(
    file_path: str,
    target_sr: int,
    stereo_mode: str
) -> np.ndarray:
    """
    Загружает аудиофайл с помощью librosa, обрабатывает и возвращает аудиосигнал
    
    Параметры:
        file_path: Путь к аудиофайлу
        target_sr: Целевая частота дискретизации
        mono: Преобразовать в моно (по умолчанию True)
        normalize: Нормализовать аудио (по умолчанию False)
        duration: Загрузить только указанную длительность (в секундах)
        offset: Начальное смещение для загрузки (в секундах)
    
    Возвращает:
        Аудиоданные в виде numpy array (моно: (samples,), стерео: (channels, samples))
    
    Исключения:
        RuntimeError: При ошибках загрузки или обработки аудио
    """
    try:
        mid, left, right = None, None, None
        
        if stereo_mode == "mono":
            # Загрузка аудио с помощью librosa
            mid_audio, sr = audio.read(
                i=file_path,
                sr=None,
                mono=True
            )
            mid_audio = librosa.resample(
                mid_audio,  # Исправлено: было audio
                orig_sr=sr, 
                target_sr=target_sr
            )
            mid = mid_audio.flatten()
            
        elif stereo_mode == "left/right" or stereo_mode == "sim/dif":
            # Загрузка аудио с помощью librosa
            stereo_audio, sr = audio.read(
                i=file_path,
                sr=None,
                mono=False
            )

            if stereo_mode == "left/right":
                left_audio = stereo_audio[0]  # Исправлено: было [:, 0]
                right_audio = stereo_audio[1] # Исправлено: было [:, 1]
                left_audio = librosa.resample(
                    left_audio, 
                    orig_sr=sr, 
                    target_sr=target_sr
                )
                right_audio = librosa.resample(
                    right_audio, 
                    orig_sr=sr, 
                    target_sr=target_sr
                )

                left = left_audio.flatten()
                right = right_audio.flatten()

            elif stereo_mode == "sim/dif":
                mid_left, mid_right, dif_left, dif_right = remove_center(input_array=stereo_audio, samplerate=sr)
                mid_audio = (mid_left + mid_right) * 0.5

                mid_audio = librosa.resample(
                    mid_audio, 
                    orig_sr=sr, 
                    target_sr=target_sr
                )
                dif_left = librosa.resample(
                    dif_left, 
                    orig_sr=sr, 
                    target_sr=target_sr
                )
                dif_right = librosa.resample(
                    dif_right, 
                    orig_sr=sr, 
                    target_sr=target_sr
                )

                mid = mid_audio.flatten()
                left = dif_left.flatten()   # Исправлено: было left_audio
                right = dif_right.flatten() # Исправлено: было right_audio

        return mid, left, right
    
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки аудио '{file_path}': {str(e)}")

class Config:
    def __init__(self):
        self.device = self.get_device()
        self.is_half = self.device == "cpu"
        self.n_cpu = cpu_count()
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def device_config(self):
        if torch.cuda.is_available():
            print("Используется устройство CUDA")
            self._configure_gpu()
        elif torch.backends.mps.is_available():
            print("Используется устройство MPS")
            self.device = "mps"
        else:
            print("Используется CPU")
            self.device = "cpu"
            self.is_half = True

        x_pad, x_query, x_center, x_max = (
            (3, 10, 60, 65) if self.is_half else (1, 6, 38, 41)
        )
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad, x_query, x_center, x_max = (1, 5, 30, 32)

        return x_pad, x_query, x_center, x_max

    def _configure_gpu(self):
        self.gpu_name = torch.cuda.get_device_name(self.device)
        low_end_gpus = ["16", "P40", "P10", "1060", "1070", "1080"]
        if (
            any(gpu in self.gpu_name for gpu in low_end_gpus)
            and "V100" not in self.gpu_name.upper()
        ):
            self.is_half = False
        self.gpu_mem = int(
            torch.cuda.get_device_properties(self.device).total_memory
            / 1024
            / 1024
            / 1024
            + 0.4
        )

# Загрузка модели Hubert
def load_hubert(device, is_half, model_path):
    models, saved_cfg, task = load_model_ensemble_and_task(
        [model_path], suffix=""
    )
    hubert = models[0].to(device)
    hubert = hubert.half() if is_half else hubert.float()
    hubert.eval()
    return hubert

# Получение голосового преобразователя
def get_vc(device, is_half, config, model_path):
    cpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if "config" not in cpt or "weight" not in cpt:
        raise ValueError(
            f"Некорректный формат для {model_path}. "
            "Используйте голосовую модель, обученную с использованием RVC v2."
        )

    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    pitch_guidance = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    input_dim = 768 if version == "v2" else 256

    net_g = Synthesizer(
        *cpt["config"],
        use_f0=pitch_guidance,
        input_dim=input_dim,
        is_half=is_half,
    )

    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(device)
    net_g = net_g.half() if is_half else net_g.float()

    vc = VC(tgt_sr, config)
    return cpt, version, net_g, tgt_sr, vc

def rvc_infer(
    index_path,
    index_rate,
    input_path,
    output_path,
    pitch,
    f0_method,
    cpt,
    version,
    net_g,
    filter_radius,
    tgt_sr,
    volume_envelope,
    protect,
    hop_length,
    vc,
    hubert_model,
    f0_min=50,
    f0_max=1100,
    format_output="wav",
    output_bitrate="320k",
    stereo_mode="mono"
):

    mid, left, right = load_audio(input_path, 16000, stereo_mode)
    pitch_guidance = cpt.get("f0", 1)
    
    if stereo_mode == "mono":
        if mid is None:
            raise ValueError("Mono audio data is None")
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            mid,
            input_path,
            pitch,
            f0_method,
            index_path,
            index_rate,
            pitch_guidance,
            filter_radius,
            tgt_sr,
            0,
            volume_envelope,
            version,
            protect,
            hop_length,
            f0_file=None,
            f0_min=f0_min,
            f0_max=f0_max,
        )
        
    elif stereo_mode == "left/right":
        if left is None or right is None:
            raise ValueError("Left or right audio channel is None")
            
        left_audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            left,
            input_path,
            pitch,
            f0_method,
            index_path,
            index_rate,
            pitch_guidance,
            filter_radius,
            tgt_sr,
            0,
            volume_envelope,
            version,
            protect,
            hop_length,
            f0_file=None,
            f0_min=f0_min,
            f0_max=f0_max,
        )
        right_audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            right,
            input_path,
            pitch,
            f0_method,
            index_path,
            index_rate,
            pitch_guidance,
            filter_radius,
            tgt_sr,
            0,
            volume_envelope,
            version,
            protect,
            hop_length,
            f0_file=None,
            f0_min=f0_min,
            f0_max=f0_max,
        )

        # Ensure both channels have the same length
        min_len = min(len(left_audio_opt), len(right_audio_opt))
        if min_len == 0:
            raise ValueError("Processed audio is empty")

        left_audio_opt = left_audio_opt[:min_len]
        right_audio_opt = right_audio_opt[:min_len]

        audio_opt = np.stack((left_audio_opt, right_audio_opt), axis=0)

    elif stereo_mode == "sim/dif":
        if mid is None or left is None or right is None:
            raise ValueError("Mid, left or right audio channel is None")
            
        mid_audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            mid,
            input_path,
            pitch,
            f0_method,
            index_path,
            index_rate,
            pitch_guidance,
            filter_radius,
            tgt_sr,
            0,
            volume_envelope,
            version,
            protect,
            hop_length,
            f0_file=None,
            f0_min=f0_min,
            f0_max=f0_max,
        )
        left_audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            left,
            input_path,
            pitch,
            f0_method,
            index_path,
            index_rate,
            pitch_guidance,
            filter_radius,
            tgt_sr,
            0,
            volume_envelope,
            version,
            protect,
            hop_length,
            f0_file=None,
            f0_min=f0_min,
            f0_max=f0_max,
        )
        right_audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            right,
            input_path,
            pitch,
            f0_method,
            index_path,
            index_rate,
            pitch_guidance,
            filter_radius,
            tgt_sr,
            0,
            volume_envelope,
            version,
            protect,
            hop_length,
            f0_file=None,
            f0_min=f0_min,
            f0_max=f0_max,
        )

        # Ensure all channels have the same length
        min_len = min(len(mid_audio_opt), len(left_audio_opt), len(right_audio_opt))
        if min_len == 0:
            raise ValueError("Processed audio is empty")

        mid_audio_opt = mid_audio_opt[:min_len]
        left_audio_opt = left_audio_opt[:min_len]
        right_audio_opt = right_audio_opt[:min_len]

        dif_audio_opt = np.stack((left_audio_opt, right_audio_opt), axis=0)
   
        audio_opt = overlay_mono_on_stereo(mid_audio_opt, dif_audio_opt)

    audio.write(o=output_path, array=audio_opt, sr=tgt_sr, of=format_output, br=output_bitrate)
    return output_path
