import gradio as gr
import json
import pandas as pd
import tempfile
import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from separator.audio_writer import write_audio_file

N_FFT = 2048
WIN_LENGTH = 2048
HOP_LENGTH = WIN_LENGTH // 4

class Inverter:
    def __init__(self):
        self.test = "test"
        
    def load_audio(self, filepath):
        """Загрузка аудиофайла с помощью librosa"""
        if filepath is None:
            return None, None
        try:
            return librosa.load(filepath, sr=None, mono=False)
        except Exception as e:
            print(f"Ошибка загрузки аудио: {e}")
            return None, None

    def process_channel(self, y1_ch, y2_ch, sr, method):
        """Обработка одного аудиоканала"""
        if method == "waveform":
            return y1_ch - y2_ch
        
        elif method == "spectrogram":
            # Вычисляем спектрограммы
            S1 = librosa.stft(y1_ch, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
            S2 = librosa.stft(y2_ch, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
            
            # Амплитудные спектрограммы
            mag1 = np.abs(S1)
            mag2 = np.abs(S2)
            
            # Спектральное вычитание
            mag_result = np.maximum(mag1 - mag2, 0)
            
            # Сохраняем фазовую информацию исходного сигнала
            phase = np.angle(S1)
            
            # Комбинируем амплитуду результата с фазой
            S_result = mag_result * np.exp(1j * phase)
            
            # Обратное преобразование
            return librosa.istft(
                S_result,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                win_length=WIN_LENGTH,
                length=len(y1_ch)
            )
    
    def process_audio(self, audio1_path, audio2_path, out_format, method):
        # Загрузка аудиофайлов
        y1, sr1 = self.load_audio(audio1_path)
        y2, sr2 = self.load_audio(audio2_path)
        
        if sr1 is None or sr2 is None:
            raise gr.Error()
        
        # Определяем количество каналов
        channels1 = 1 if y1.ndim == 1 else y1.shape[0]
        channels2 = 1 if y2.ndim == 1 else y2.shape[0]
        
        # Преобразование в форму (samples, channels)
        if channels1 > 1:
            y1 = y1.T  # (channels, samples) -> (samples, channels)
        else:
            y1 = y1.reshape(-1, 1)
        
        if channels2 > 1:
            y2 = y2.T  # (channels, samples) -> (samples, channels)
        else:
            y2 = y2.reshape(-1, 1)
        
        # Ресемплинг до одинаковой частоты дискретизации
        if sr1 != sr2:
            if channels2 > 1:
                # Ресемплинг для каждого канала отдельно
                y2_resampled = np.zeros((len(y2), channels2), dtype=np.float32)
                for c in range(channels2):
                    y2_resampled[:, c] = librosa.resample(
                        y2[:, c], 
                        orig_sr=sr2, 
                        target_sr=sr1
                    )
                y2 = y2_resampled
            else:
                y2 = librosa.resample(y2[:, 0], orig_sr=sr2, target_sr=sr1)
                y2 = y2.reshape(-1, 1)
            sr2 = sr1
        
        # Приводим к одинаковой длине
        min_len = min(len(y1), len(y2))
        y1 = y1[:min_len]
        y2 = y2[:min_len]
        
        # Обрабатываем каждый канал отдельно
        result_channels = []
        
        # Если основной сигнал моно, а удаляемый стерео - преобразуем удаляемый в моно
        if channels1 == 1 and channels2 > 1:
            y2 = y2.mean(axis=1, keepdims=True)
            channels2 = 1
        
        for c in range(channels1):
            # Выбираем канал для основного сигнала
            y1_ch = y1[:, c]
            
            # Выбираем канал для удаляемого сигнала
            if channels2 == 1:
                y2_ch = y2[:, 0]
            else:
                # Если каналов удаляемого сигнала больше, используем соответствующий канал
                y2_ch = y2[:, min(c, channels2-1)]
            
            # Обрабатываем канал
            result_ch = self.process_channel(y1_ch, y2_ch, sr1, method)
            result_channels.append(result_ch)
        
        # Собираем каналы в один массив
        if len(result_channels) > 1:
            result = np.column_stack(result_channels)
        else:
            result = np.array(result_channels[0])
        
        # Нормализация (предотвращение клиппинга)
        if result.ndim > 1:
            # Для многоканального аудио нормализуем каждый канал отдельно
            for c in range(result.shape[1]):
                channel = result[:, c]
                max_val = np.max(np.abs(channel))
                if max_val > 0:
                    result[:, c] = channel * 0.9 / max_val
        else:
            max_val = np.max(np.abs(result))
            if max_val > 0:
                result = result * 0.9 / max_val
    
        folder_path = os.path.dirname(audio2_path)
    
        inverted_wav = os.path.join(folder_path, "inverted.wav")
        sf.write(inverted_wav, result, sr1)
        inverted = os.path.join(folder_path, f"inverted_{os.path.splitext(os.path.basename(audio2_path))[0]}.{out_format}")
        write_audio_file(inverted, result.T, sr1, out_format, "320k")
        return inverted, inverted_wav