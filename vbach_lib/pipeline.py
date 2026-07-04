import os
import gc
import torch
from torch import nn
import torch.nn.functional as F
from scipy import signal
import librosa
import numpy as np
from typing import Tuple, Any, Dict, List, Optional, Union, Callable
from tqdm import tqdm
from pathlib import Path
import sys

def lazy_faiss_import():
    import faiss as module
    return module

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))
from i18n import _i18n
if __package__:
    from .f0_extractor import f0_extract, f0_import, f0_methods
else:
    from vbach_lib.f0_extractor import f0_extract, f0_import, f0_methods

FILTER_ORDER: int = 5
CUTOFF_FREQUENCY: int = 48
SAMPLE_RATE: int = 16000
bh, ah = signal.butter(
    N=FILTER_ORDER, Wn=CUTOFF_FREQUENCY, btype="high", fs=SAMPLE_RATE
)

class AudioProcessor:
    """Класс для обработки аудио"""
    
    @staticmethod
    def change_rms(
        sourceaudio: np.ndarray, 
        source_rate: int, 
        targetaudio: np.ndarray, 
        target_rate: int, 
        rate: float
    ) -> np.ndarray:
        """
        Изменить RMS (громкость) аудио
        
        Args:
            sourceaudio: Исходное аудио
            source_rate: Частота исходного аудио
            targetaudio: Целевое аудио
            target_rate: Частота целевого аудио
            rate: Коэффициент изменения
        
        Returns:
            Измененное аудио
        """
        rms1 = librosa.feature.rms(
            y=sourceaudio,
            frame_length=source_rate // 2 * 2,
            hop_length=source_rate // 2,
        )
        rms2 = librosa.feature.rms(
            y=targetaudio,
            frame_length=target_rate // 2 * 2,
            hop_length=target_rate // 2,
        )

        rms1 = F.interpolate(
            torch.from_numpy(rms1).float().unsqueeze(0),
            size=targetaudio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = F.interpolate(
            torch.from_numpy(rms2).float().unsqueeze(0),
            size=targetaudio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6)

        adjustedaudio: np.ndarray = (
            targetaudio
            * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1)).numpy()
        )
        return adjustedaudio

class VC:
    """Класс для голосового преобразования"""
    
    def __init__(self, tgt_sr: int, config: Any, use_transformers: bool) -> None:
        """
        Инициализация VC
        
        Args:
            tgt_sr: Целевая частота дискретизации
            config: Конфигурация
            stack: Стек ("fairseq" или "transformers")
        """
        self.x_pad: int = config.x_pad
        self.x_query: int = config.x_query
        self.x_center: int = config.x_center
        self.x_max: int = config.x_max
        self.is_half: bool = config.is_half
        self.sample_rate: int = 16000
        self.window: int = 160
        self.t_pad: int = self.sample_rate * self.x_pad
        self.t_pad_tgt: int = tgt_sr * self.x_pad
        self.t_pad2: int = self.t_pad * 2
        self.t_query: int = self.sample_rate * self.x_query
        self.t_center: int = self.sample_rate * self.x_center
        self.t_max: int = self.sample_rate * self.x_max
        self.time_step: float = self.window / self.sample_rate * 1000
        self.device: torch.device = config.device
        self.voice_conversion: Callable = self._vc_transformers if use_transformers else self._vc

    def get_f0(
        self,
        x: np.ndarray,
        p_len: int,
        pitch: float,
        f0_method: str,
        hop_length: int,
        f0_min: int = 50,
        f0_max: int = 1100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        global input_audio_path2wav
        time_step: float = self.window / self.sample_rate * 1000
        f0_mel_min: float = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max: float = 1127 * np.log(1 + f0_max / 700)

        f0 = f0_extract(x, self.sample_rate, p_len, f0_method, hop_length, self.window, self.device, time_step, self.is_half, f0_min, f0_max)

        f0 *= pow(2, pitch / 12)
        tf0: int = self.sample_rate // self.window

        f0bak: np.ndarray = f0.copy()
        f0_mel: np.ndarray = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse: np.ndarray = np.rint(f0_mel).astype(int)
        
        return f0_coarse, f0bak

    def get_f0_from_file(
        self,
        f0_file: str | Path,
        offset: int,
        p_len: int,
        pitch: float,
        f0_min: int = 50,
        f0_max: int = 1100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        f0_mel_min: float = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max: float = 1127 * np.log(1 + f0_max / 700)

        f0_imported = f0_import(f0_file)
        if f0_imported.size < p_len:
            f0_imported = np.pad(f0_imported, (0, p_len - f0_imported.size), mode='constant', constant_values=0)
        f0_imported = f0_imported[:p_len]
        offset_p_len = offset // self.window
        f0 = np.pad(f0_imported, (offset_p_len, offset_p_len), mode="reflect")

        f0 *= pow(2, pitch / 12)
        tf0: int = self.sample_rate // self.window

        f0bak: np.ndarray = f0.copy()
        f0_mel: np.ndarray = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse: np.ndarray = np.rint(f0_mel).astype(int)
        
        return f0_coarse, f0bak

    def _vc(
        self,
        model: nn.Module,
        net_g: nn.Module,
        sid: torch.Tensor,
        audio0: np.ndarray,
        pitch: Optional[torch.Tensor],
        pitchf: Optional[torch.Tensor],
        index: Any,
        big_npy: Optional[np.ndarray],
        index_rate: float,
        version: str,
        protect: float,
    ) -> np.ndarray:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        feats = torch.from_numpy(audio0)
        feats = feats.half() if self.is_half else feats.float()

        if feats.dim() == 2:
            feats = feats.mean(-1)

        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs: Dict[str, Any] = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }

        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

            if protect < 0.5 and pitch is not None and pitchf is not None:
                feats0 = feats.clone()

            if index is not None and big_npy is not None and index_rate != 0:
                npy = feats[0].cpu().numpy()
                npy = npy.astype("float32") if self.is_half else npy
                score, ix = index.search(npy, k=8)
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                npy = npy.astype("float16") if self.is_half else npy
                feats = (
                    torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                    + (1 - index_rate) * feats
                )

            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
            if protect < 0.5 and pitch is not None and pitchf is not None:
                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                    0, 2, 1
                )

            p_len: int = audio0.shape[0] // self.window
            if feats.shape[1] < p_len:
                p_len = feats.shape[1]
                if pitch is not None and pitchf is not None:
                    pitch = pitch[:, :p_len]
                    pitchf = pitchf[:, :p_len]

            if protect < 0.5 and pitch is not None and pitchf is not None:
                pitchff = pitchf.clone()
                pitchff[pitchf > 0] = 1
                pitchff[pitchf < 1] = protect
                pitchff = pitchff.unsqueeze(-1)
                feats = feats * pitchff + feats0 * (1 - pitchff)
                feats = feats.to(feats0.dtype)

            p_len_tensor = torch.tensor([p_len], device=self.device).long()

            if pitch is not None and pitchf is not None:
                audio1 = (
                    (net_g.infer(feats, p_len_tensor, pitch, pitchf, sid)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )
            else:
                audio1 = (
                    (net_g.infer(feats, p_len_tensor, sid)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )

        del feats, p_len_tensor, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio1

    def _vc_transformers(
        self,
        model: nn.Module,
        net_g: nn.Module,
        sid: torch.Tensor,
        audio0: np.ndarray,
        pitch: Optional[torch.Tensor],
        pitchf: Optional[torch.Tensor],
        index: Any,
        big_npy: Optional[np.ndarray],
        index_rate: float,
        version: str,
        protect: float,
    ) -> np.ndarray:
        """
        Внутренний метод голосового преобразования (transformers)
        
        Args:
            model: Модель Hubert
            net_g: Генератор
            sid: ID спикера
            audio0: Аудиоданные
            pitch: Высота тона
            pitchf: F0
            index: Индекс FAISS
            big_npy: Массив эмбеддингов
            index_rate: Коэффициент влияния индекса
            version: Версия модели
            protect: Защита согласных
        
        Returns:
            Преобразованные аудиоданные
        """
        with torch.no_grad():
            pitch_guidance: bool = pitch is not None and pitchf is not None
            feats = torch.from_numpy(audio0).float()
            feats = feats.mean(-1) if feats.dim() == 2 else feats
            assert feats.dim() == 1, feats.dim()
            feats = feats.view(1, -1).to(self.device)
            feats = model(feats)["last_hidden_state"]
            feats = (
                model.final_proj(feats[0]).unsqueeze(0) if version == "v1" else feats
            )
            feats0 = feats.clone() if pitch_guidance else None
            
            if index is not None and big_npy is not None and index_rate != 0:
                feats = self._retrieve_speaker_embeddings(feats, index, big_npy, index_rate)
                
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
            p_len: int = min(audio0.shape[0] // self.window, feats.shape[1])
            
            if pitch_guidance:
                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                    0, 2, 1
                )
                if pitch is not None and pitchf is not None:
                    pitch = pitch[:, :p_len]
                    pitchf = pitchf[:, :p_len]
                    
                if protect < 0.5:
                    pitchff = pitchf.clone()
                    pitchff[pitchf > 0] = 1
                    pitchff[pitchf < 1] = protect
                    feats = feats * pitchff.unsqueeze(-1) + feats0 * (
                        1 - pitchff.unsqueeze(-1)
                    )
                    feats = feats.to(feats0.dtype)
            else:
                pitch, pitchf = None, None
                
            p_len_tensor = torch.tensor([p_len], device=self.device).long()
            audio1 = (
                (net_g.infer(feats.float(), p_len_tensor, pitch, pitchf.float() if pitchf is not None else None, sid)[0][0, 0])
                .data.cpu()
                .float()
                .numpy()
            )
            
            del feats, feats0, p_len_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return audio1

    def pipeline(
        self,
        model: nn.Module,
        net_g: nn.Module,
        sid: int,
        audio: np.ndarray,
        pitch: float,
        f0_method: str | None,
        f0_file: str | None,
        file_index: Optional[str],
        index_rate: float,
        pitch_guidance: bool,
        tgt_sr: int,
        volume_envelope: float,
        version: str,
        protect: float,
        hop_length: int,
        f0_min: int = 50,
        f0_max: int = 1100,
        chunk_duration: int = 3,
        add_text_channel: str = None,
        add_text_custom: str = None,
        resample_sr: int = 0,
    ) -> np.ndarray:
        
        add_text_channel_str = ""
        if add_text_channel and add_text_channel != "":
            add_text_channel_str = f" {add_text_channel}"

        add_text_custom_str = ""
        if add_text_custom and add_text_custom != "":
            add_text_custom_str = f" | {add_text_custom}"
        device = self.device
        audio = signal.filtfilt(bh, ah, audio)
        audio_len = len(audio)

        if (
            file_index
            and file_index != ""
            and os.path.exists(file_index)
            and index_rate != 0
        ):
            faiss = lazy_faiss_import()
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as e:
                print(f"{_i18n('faiss_error')}: {e}")
                index = big_npy = None
        else:
            index = big_npy = None

        sid_tensor = torch.tensor(sid, device=device).unsqueeze(0).long()

        real_chunk_size = min(
            self.sample_rate * int(chunk_duration), 
            audio_len
        )
        offset = int(self.sample_rate // 12.5)
        extra_offset = int(tgt_sr // 0.8)
        
        audio_pad = np.pad(audio, (offset, offset), mode="reflect")

        pitch_tensor: Optional[torch.Tensor] = None
        pitchf_tensor: Optional[torch.Tensor] = None
        
        if pitch_guidance:
            p_len = len(audio_pad) // self.window
            if f0_method and f0_method in f0_methods:
                print(_i18n("extracting_f0"))
                pitch_coarse, pitchf = self.get_f0(
                    audio_pad,
                    p_len,
                    pitch,
                    f0_method,
                    hop_length,
                    f0_min,
                    f0_max,
                )
                pitch_coarse = pitch_coarse[:p_len]
                pitchf = pitchf[:p_len]
                print(_i18n("extracting_f0_success"))
                if device.type == "mps":
                    pitchf = pitchf.astype(np.float32)
                pitch_tensor = torch.tensor(pitch_coarse, device=device).unsqueeze(0).long()
                pitchf_tensor = torch.tensor(pitchf, device=device).unsqueeze(0).float()
            elif not f0_method or f0_file:
                print(_i18n("importing_f0"))
                pitch_coarse, pitchf = self.get_f0_from_file(
                    f0_file,
                    offset,
                    len(audio) // self.window,
                    pitch,
                    f0_min,
                    f0_max,
                )
                pitch_coarse = pitch_coarse[:p_len]
                pitchf = pitchf[:p_len]
                print(_i18n("importing_f0_success"))
                if device.type == "mps":
                    pitchf = pitchf.astype(np.float32)
                pitch_tensor = torch.tensor(pitch_coarse, device=device).unsqueeze(0).long()
                pitchf_tensor = torch.tensor(pitchf, device=device).unsqueeze(0).float()

        processed_chunks: List[Tuple[int, int, np.ndarray, int, int]] = []
        start = 0

        with tqdm(total=audio_len, desc=_i18n("processing") + str(add_text_channel_str) + str(add_text_custom_str), unit=_i18n("samples"), leave=False) as progress_bar:

            while start < audio_len:
                end = min(start + real_chunk_size, audio_len)

                need_left = start > 0
                need_right = end < audio_len
                pad_left = offset if need_left else 0
                pad_right = offset if need_right else 0

                chunk_start_in_pad = start - pad_left
                chunk_end_in_pad = end + pad_right

                inf_start = chunk_start_in_pad + offset - extra_offset
                inf_end = chunk_end_in_pad + offset + extra_offset
                
                actual_inf_start = max(0, inf_start)
                actual_inf_end = min(len(audio_pad), inf_end)
                
                actual_extra_left = (chunk_start_in_pad + offset) - actual_inf_start
                actual_extra_right = actual_inf_end - (chunk_end_in_pad + offset)

                chunk_audio = audio_pad[actual_inf_start : actual_inf_end]

                f0_start = actual_inf_start // self.window
                f0_end = actual_inf_end // self.window

                if pitch_guidance and pitch_tensor is not None and pitchf_tensor is not None:
                    out = self.voice_conversion(
                        model,
                        net_g,
                        sid_tensor,
                        chunk_audio,
                        pitch_tensor[:, f0_start:f0_end],
                        pitchf_tensor[:, f0_start:f0_end],
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )
                else:
                    out = self.voice_conversion(
                        model,
                        net_g,
                        sid_tensor,
                        chunk_audio,
                        None,
                        None,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )

                scale_factor = tgt_sr / self.sample_rate
                
                cut_left = int(round(actual_extra_left * scale_factor))
                cut_right = int(round(actual_extra_right * scale_factor))
                
                if cut_right > 0:
                    out = out[cut_left : -cut_right]
                else:
                    out = out[cut_left:]

                output_start = int(round((chunk_start_in_pad) / self.sample_rate * tgt_sr))
                output_end = output_start + len(out)

                processed_chunks.append(
                    (output_start, output_end, out, pad_left, pad_right)
                )
                progress_bar.update(end - start)

                start = end

        if not processed_chunks:
            raise RuntimeError(_i18n("no_chunks_error"))

        max_output_end = max(end for _c, end, _c, _c, _c in processed_chunks)
        output = np.zeros(max_output_end, dtype=np.float32)
        weight = np.zeros(max_output_end, dtype=np.float32)

        for start_idx, end_idx, chunk, pad_left, pad_right in processed_chunks:
            chunk_len = len(chunk)
            if chunk_len != (end_idx - start_idx):
                end_idx = start_idx + chunk_len

            w = np.ones(chunk_len, dtype=np.float32)
            fade_len = int(round(offset / self.sample_rate * tgt_sr))

            if pad_left > 0 and fade_len > 0 and start_idx > 0:
                actual_fade = min(fade_len, chunk_len)
                w[:actual_fade] = np.linspace(0, 1, actual_fade)
                
            if pad_right > 0 and fade_len > 0 and end_idx < max_output_end:
                actual_fade = min(fade_len, chunk_len)
                w[-actual_fade:] = np.linspace(1, 0, actual_fade)

            output_end = min(end_idx, len(output))
            chunk = chunk[: output_end - start_idx]
            w = w[: output_end - start_idx]

            output[start_idx:output_end] += chunk * w
            weight[start_idx:output_end] += w

        mask = weight > 1e-8
        output[mask] /= weight[mask]

        expected_final_len = int(round(audio_len / self.sample_rate * tgt_sr))
        audio_opt = output[:expected_final_len]

        if volume_envelope != 1:
            audio_opt = AudioProcessor.change_rms(
                audio, self.sample_rate, audio_opt, tgt_sr, volume_envelope
            )
        if resample_sr >= self.sample_rate and tgt_sr != resample_sr:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )

        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return audio_opt

    def _retrieve_speaker_embeddings(
        self, 
        feats: torch.Tensor, 
        index: Any, 
        big_npy: np.ndarray, 
        index_rate: float
    ) -> torch.Tensor:
        """
        Получить эмбеддинги спикера из индекса
        
        Args:
            feats: Эмбеддинги
            index: Индекс FAISS
            big_npy: Массив эмбеддингов
            index_rate: Коэффициент влияния индекса
        
        Returns:
            Обновленные эмбеддинги
        """
        npy = feats[0].cpu().numpy()
        score, ix = index.search(npy, k=8)
        weight = np.square(1 / score)
        weight /= weight.sum(axis=1, keepdims=True)
        npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
        feats = (
            torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
            + (1 - index_rate) * feats
        )
        return feats
