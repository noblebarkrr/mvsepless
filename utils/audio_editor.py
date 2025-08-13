import os
import librosa
import soundfile as sf
import numpy as np

class AudioEditor:
    def __init__(self):
        pass

    def resample_audio(self, audio_path):
        if not audio_path or not os.path.isfile(audio_path):
            return None
            
        original_name = os.path.splitext(os.path.basename(audio_path))[0]
        folder_path = os.path.dirname(audio_path)
        resampled_path = os.path.join(folder_path, f"resampled_{original_name}.wav")
        
        target_sr = 44100

        # Загрузка аудио через librosa с сохранением оригинальной структуры каналов
        y, orig_sr = librosa.load(audio_path, sr=None, mono=False)
        
        # Определение типа аудио (моно/стерео)
        if y.ndim == 1:
            channels = 1
            y = y.reshape(-1, 1)
        else:
            channels = y.shape[0]
            y = y.T

        # Ресемплинг только если необходима смена частоты
        if orig_sr != target_sr:
            resampled_channels = []
            for channel in range(channels):
                channel_data = y[:, channel]
                resampled = librosa.resample(
                    y=channel_data,
                    orig_sr=orig_sr,
                    target_sr=target_sr,
                    res_type="kaiser_best"  # Высококачественный метод
                )
                resampled_channels.append(resampled)
            
            # Синхронизация длины каналов
            min_length = min(len(c) for c in resampled_channels)
            resampled_data = np.vstack([c[:min_length] for c in resampled_channels]).T
        else:
            resampled_data = y

        sf.write(
            resampled_path,
            resampled_data,
            target_sr,
            subtype="PCM_16"
        )
        
        return resampled_path
    
    def convert_to_mono(self, audio_path):
        if not audio_path or not os.path.isfile(audio_path):
            return None
            
        original_name = os.path.splitext(os.path.basename(audio_path))[0]
        folder_path = os.path.dirname(audio_path)
        mono_path = os.path.join(folder_path, f"mono_{original_name}.wav")
        
        # Загрузка аудио с сохранением оригинальной структуры каналов
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        
        # Преобразование в моно
        if y.ndim > 1:
            y_mono = np.mean(y, axis=0)
        else:
            y_mono = y
        
        sf.write(
            mono_path,
            y_mono,
            sr,
            subtype="PCM_16"
        )
        
        return mono_path
    def split_channels(self, audio_path):
        if not audio_path or not os.path.isfile(audio_path):
            return None
            
        original_name = os.path.splitext(os.path.basename(audio_path))[0]
        folder_path = os.path.dirname(audio_path)
        
        # Загрузка аудио с сохранением оригинальной структуры каналов
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        
        # Определение количества каналов
        if y.ndim == 1:
            channels = [y]
        else:
            channels = [y[i] for i in range(y.shape[0])]
        
        channel_paths = []
        for i, channel_data in enumerate(channels):
            channel_path = os.path.join(folder_path, f"{original_name}_channel_{i+1}.wav")
            sf.write(
                channel_path,
                channel_data,
                sr,
                subtype="PCM_16"
            )
            channel_paths.append(channel_path)
        
        return channel_paths
    
    def merge_channels(self, channel_paths):
        if not channel_paths or not all(os.path.isfile(p) for p in channel_paths):
            return None
            
        folder_path = os.path.dirname(channel_paths[0])
        merged_name = os.path.join(folder_path, "merged_audio.wav")
        
        # Загрузка всех каналов
        channels = []
        for path in channel_paths:
            y, sr = librosa.load(path, sr=None, mono=False)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            channels.append(y.T)
        
        # Объединение каналов
        merged_data = np.vstack(channels).T
        
        sf.write(
            merged_name,
            merged_data,
            sr,
            subtype="PCM_16"
        )
        
        return merged_name
    
    def normalize_audio(self, audio_path):
        if not audio_path or not os.path.isfile(audio_path):
            return None
            
        original_name = os.path.splitext(os.path.basename(audio_path))[0]
        folder_path = os.path.dirname(audio_path)
        normalized_path = os.path.join(folder_path, f"normalized_{original_name}.wav")
        
        # Загрузка аудио с сохранением оригинальной структуры каналов
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        
        # Нормализация громкости
        if y.ndim > 1:
            y = np.mean(y, axis=0)
        y = librosa.util.normalize(y)
        sf.write(
            normalized_path,
            y,
            sr,
            subtype="PCM_16"
        )
        return normalized_path
    
    def change_volume(self, audio_path, volume_factor):
        if not audio_path or not os.path.isfile(audio_path):
            return None
            
        original_name = os.path.splitext(os.path.basename(audio_path))[0]
        folder_path = os.path.dirname(audio_path)
        volume_changed_path = os.path.join(folder_path, f"volume_changed_{original_name}.wav")
        
        # Загрузка аудио с сохранением оригинальной структуры каналов
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        
        # Изменение громкости
        y *= volume_factor
        
        sf.write(
            volume_changed_path,
            y,
            sr,
            subtype="PCM_16"
        )
        
        return volume_changed_path
    
    def trim_audio(self, audio_path, start_time, end_time):
        if not audio_path or not os.path.isfile(audio_path):
            return None
            
        original_name = os.path.splitext(os.path.basename(audio_path))[0]
        folder_path = os.path.dirname(audio_path)
        trimmed_path = os.path.join(folder_path, f"trimmed_{original_name}.wav")
        
        # Загрузка аудио с сохранением оригинальной структуры каналов
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        
        # Преобразование времени в сэмплы
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Обрезка аудио
        y_trimmed = y[:, start_sample:end_sample]
        
        sf.write(
            trimmed_path,
            y_trimmed.T,
            sr,
            subtype="PCM_16"
        )
        
        return trimmed_path
    
    def apply_effects(self, audio_path, effects):
        if not audio_path or not os.path.isfile(audio_path):
            return None
            
        original_name = os.path.splitext(os.path.basename(audio_path))[0]
        folder_path = os.path.dirname(audio_path)
        effects_applied_path = os.path.join(folder_path, f"effects_{original_name}.wav")
        
        # Загрузка аудио с сохранением оригинальной структуры каналов
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        
        # Применение эффектов
        for effect in effects:
            if effect == "reverb":
                y = librosa.effects.preemphasis(y)
            elif effect == "echo":
                y = librosa.effects.harmonic(y)
            elif effect == "distortion":
                y = np.clip(y * 2, -1, 1)
            elif effect == "pitch_shift":
                y = librosa.effects.pitch_shift(y, sr, n_steps=2)
            elif effect == "time_stretch":
                y = librosa.effects.time_stretch(y, rate=1.5)
        sf.write(
            effects_applied_path,
            y.T,
            sr,
            subtype="PCM_16"
        )
        return effects_applied_path
    
    def change_sample_rate(self, audio_path, target_sr):
        if not audio_path or not os.path.isfile(audio_path):
            return None
            
        original_name = os.path.splitext(os.path.basename(audio_path))[0]
        folder_path = os.path.dirname(audio_path)
        resampled_path = os.path.join(folder_path, f"resampled_{original_name}_{target_sr}.wav")
        
        # Загрузка аудио с сохранением оригинальной структуры каналов
        y, orig_sr = librosa.load(audio_path, sr=None, mono=False)
        
        # Ресемплинг только если необходима смена частоты
        if orig_sr != target_sr:
            resampled_data = librosa.resample(y, orig_sr, target_sr)
        else:
            resampled_data = y
        
        sf.write(
            resampled_path,
            resampled_data.T,
            target_sr,
            subtype="PCM_16"
        )
        
        return resampled_path
    
    def apply_compression(self, audio_path, threshold=-20, ratio=4):
        if not audio_path or not os.path.isfile(audio_path):
            return None
            
        original_name = os.path.splitext(os.path.basename(audio_path))[0]
        folder_path = os.path.dirname(audio_path)
        compressed_path = os.path.join(folder_path, f"compressed_{original_name}.wav")
        
        # Загрузка аудио с сохранением оригинальной структуры каналов
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        
        # Применение компрессии
        y_compressed = librosa.effects.compressor(y, threshold=threshold, ratio=ratio)
        
        sf.write(
            compressed_path,
            y_compressed.T,
            sr,
            subtype="PCM_16"
        )
        
        return compressed_path
    