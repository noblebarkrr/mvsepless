import gradio as gr
import json
import pandas as pd
import tempfile
import os
import numpy as np
import librosa
from separator.audio_utils import Audio

audio = Audio()

class Inverter:
    def __init__(self, save_to_temp=False):
        self.test = "test"
        self.save_to_temp = save_to_temp
        self.w_types = [
            "boxcar",  # Прямоугольное окно
            "triang",  # Треугольное окно
            "blackman",  # Окно Блэкмана
            "hamming",  # Окно Хэмминга
            "hann",  # Окно Ханна
            "bartlett",  # Окно Бартлетта
            "flattop",  # Окно с плоской вершиной
            "parzen",  # Окно Парзена
            "bohman",  # Окно Бохмана
            "blackmanharris",  # Окно Блэкмана-Харриса
            "nuttall",  # Окно Нуттала
            "barthann",  # Окно Бартлетта-Ханна
            "cosine",  # Косинусное окно
            "exponential",  # Экспоненциальное окно
            "tukey",  # Окно Туки
            "taylor",  # Окно Тейлора
            "lanczos",  # Окно Ланцоша
        ]

    def load_audio(self, filepath):
        """Загрузка аудиофайла с помощью librosa"""
        try:
            y, sr = audio.read(i=filepath, sr=None, mono=False)
            return y, sr
        except Exception as e:
            print(f"Ошибка загрузки аудио: {e}")
            return None, None

    def process_channel(self, y1_ch, y2_ch, sr, method, w_size=2048, overlap=2, w_type="hann"):
        """Обработка одного аудиоканала"""
        HOP_LENGTH = w_size // overlap
        if method == "waveform":
            return y1_ch - y2_ch

        elif method == "spectrogram":
            # Вычисляем спектрограммы
            S1 = librosa.stft(
                y1_ch, n_fft=w_size, hop_length=HOP_LENGTH, win_length=w_size
            )
            S2 = librosa.stft(
                y2_ch, n_fft=w_size, hop_length=HOP_LENGTH, win_length=w_size
            )

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
                n_fft=w_size,
                hop_length=HOP_LENGTH,
                win_length=w_size,
                length=len(y1_ch),
            )

    def process_audio(self, audio1_path, audio2_path, out_format, method, w_size=2048, overlap=2, w_type="hann"):
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

        if sr1 != sr2:
            if channels2 > 1:
                # Ресемплинг для каждого канала отдельно
                y2_resampled_list = []
                for c in range(channels2):
                    channel_resampled = librosa.resample(
                        y2[:, c], orig_sr=sr2, target_sr=sr1
                    )
                    y2_resampled_list.append(channel_resampled)
                
                # Находим минимальную длину среди всех каналов
                min_channel_length = min(len(ch) for ch in y2_resampled_list)
                
                # Обрезаем все каналы до одинаковой длины и собираем в массив
                y2_resampled = np.zeros((min_channel_length, channels2), dtype=np.float32)
                for c, channel in enumerate(y2_resampled_list):
                    y2_resampled[:, c] = channel[:min_channel_length]
                
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
                y2_ch = y2[:, min(c, channels2 - 1)]

            # Обрабатываем канал
            result_ch = self.process_channel(y1_ch, y2_ch, sr1, method, w_size=w_size, overlap=overlap, w_type=w_type)
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

        if self.save_to_temp:
            folder_path = tempfile.mkdtemp(prefix="inverter_temp")
        else:
            folder_path = os.path.dirname(audio2_path)
        inverted_wav = os.path.join(folder_path, "inverted.wav")
        audio.write(o=inverted_wav, array=result.T, sr=sr1, of="wav")
        inverted = os.path.join(
            folder_path,
            f"inverted_{os.path.splitext(os.path.basename(audio2_path))[0]}.{out_format}",
        )
        audio.write(o=inverted, array=result.T, sr=sr1, of=out_format, br="320k")
        return inverted, inverted_wav
