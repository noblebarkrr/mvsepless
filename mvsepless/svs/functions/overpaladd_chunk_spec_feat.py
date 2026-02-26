import torch
import torch.nn.functional as F
import librosa
import numpy as np
from asteroid.dsp.overlap_add import LambdaOverlapAdd
from ..utils.logging import AverageMeter
from .silence_split import magspec_vad, webrtc_vad
import gc


class LambdaOverlapAdd_Chunkwise_SpectralFeatures(LambdaOverlapAdd):
    """
    Исправленная версия с корректной обработкой типов данных.
    """

    def __init__(
        self,
        nnet,
        n_src,
        window_size,
        hop_size=None,
        window="hanning",
        reorder_chunks=True,
        enable_grad=False,
        device="cpu",
        sr=24000,
        vad_method="spec",
        spectral_features="mfcc",
        use_memory_efficient=True,
        max_chunk_duration=2.0,
        feature_compression=True,
        use_mixed_precision=False,  # По умолчанию отключаем mixed precision из-за проблем с типами
    ):
        super().__init__(
            nnet, n_src, window_size, hop_size, window, reorder_chunks, enable_grad
        )
        self.nnet = self.nnet.to(device)
        self.device = device
        self.sr = sr
        self.vad_method = vad_method
        self.spectral_features = spectral_features
        self.use_memory_efficient = use_memory_efficient
        self.max_chunk_duration = max_chunk_duration
        self.feature_compression = feature_compression
        self.use_mixed_precision = use_mixed_precision
        
        # Определяем тип данных для модели
        self.model_dtype = next(nnet.parameters()).dtype
        
        # Уменьшаем размер признаков для экономии памяти
        if spectral_features == "mfcc":
            self.n_mfcc = 13  # Уменьшаем с 20 до 13 коэффициентов
        else:
            self.n_mfcc = 1
            
        # Предварительно выделенные буферы для избежания повторных аллокаций
        self._feature_buffer = {}
        self._processed_chunks = 0
        
        # Периодическая очистка кэша
        self._cleanup_threshold = 5
        
        # Оптимизация памяти PyTorch
        if device.type == 'cuda':
            try:
                torch.cuda.set_per_process_memory_fraction(0.7)
            except:
                pass  # Не все версии поддерживают эту функцию

    def _clear_memory(self):
        """Агрессивная очистка памяти."""
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Очищаем буферы
        self._feature_buffer.clear()
        self._processed_chunks = 0

    def _ensure_dtype(self, tensor, target_dtype):
        """Гарантирует соответствие типов данных."""
        if tensor.dtype != target_dtype:
            return tensor.to(target_dtype)
        return tensor

    def _compute_spectral_features_memory_efficient(self, audio_segment):
        """
        Максимально эффективное по памяти вычисление признаков.
        Исправлена обработка типов данных.
        """
        if audio_segment is None or audio_segment.numel() == 0:
            return None
            
        # Сохраняем оригинальный тип для последующего восстановления
        original_dtype = audio_segment.dtype
            
        # Переводим на CPU для вычислений
        audio_cpu = audio_segment.cpu().float()  # Конвертируем в float32 для librosa
        
        # Освобождаем GPU память
        if self.device.type == 'cuda':
            del audio_segment
            torch.cuda.empty_cache()
        
        batch_size, n_src, length = audio_cpu.shape
        features = []
        
        try:
            for src in range(n_src):
                # Берем только один источник за раз
                audio_src = audio_cpu[0, src, :].numpy()
                
                if self.spectral_features == "mfcc":
                    # Используем минимальные параметры MFCC
                    mfcc = librosa.feature.mfcc(
                        y=audio_src,
                        sr=self.sr,
                        n_mfcc=self.n_mfcc,
                        n_fft=min(self.window_size, 2048),  # Ограничиваем FFT
                        hop_length=self.hop_size,
                        n_mels=40  # Уменьшаем количество mel фильтров
                    )
                    # Берем только среднее значение по времени
                    feature = mfcc[1:, :].mean(axis=1, keepdims=True).flatten()
                    
                elif self.spectral_features == "spectral_centroid":
                    centroid = librosa.feature.spectral_centroid(
                        y=audio_src,
                        sr=self.sr,
                        n_fft=min(self.window_size, 2048),
                        hop_length=self.hop_size
                    )
                    feature = centroid.mean(axis=1, keepdims=True).flatten()
                
                # Конвертируем в тензор с правильным типом
                feature_tensor = torch.from_numpy(feature).float()
                
                # Сжатие признаков если нужно
                if self.feature_compression and len(feature) > 10:
                    # Используем простую проекцию: берем только первые 5 компонент
                    feature_tensor = feature_tensor[:5]
                
                features.append(feature_tensor.unsqueeze(0))
                
                # Периодическая очистка
                if src % 2 == 0:
                    gc.collect()
            
            if not features:
                return None
                
            # Объединяем признаки
            result = torch.cat(features, dim=0).unsqueeze(0)
            
            # Возвращаем на устройство с правильным типом
            result = result.to(device=self.device, dtype=original_dtype)
            
            return result
            
        finally:
            # Очищаем промежуточные данные
            del audio_cpu, features
            gc.collect()

    def _process_chunk_memory_efficient(self, x, start, end):
        """
        Обработка одного чанка с минимальным использованием памяти.
        Исправлена обработка mixed precision.
        """
        # Извлекаем чанк
        chunk = x[..., start:end]
        
        # Проверяем размер чанка
        chunk_duration = (end - start) / self.sr
        result = None
        
        try:
            if chunk_duration > self.max_chunk_duration:
                # Разбиваем на подчанки
                sub_chunk_size = int(self.max_chunk_duration * self.sr)
                sub_chunks = []
                
                for sub_start in range(0, end - start, sub_chunk_size):
                    sub_end = min(sub_start + sub_chunk_size, end - start)
                    sub_chunk = chunk[..., sub_start:sub_end]
                    
                    # Обрабатываем подчанк без mixed precision для избежания проблем с типами
                    if self.use_mixed_precision and self.device.type == 'cuda':
                        # Используем autocast только если включено и доступно
                        with torch.cuda.amp.autocast(enabled=True):
                            sub_result = self.nnet(sub_chunk)
                    else:
                        sub_result = self.nnet(sub_chunk)
                    
                    # Ensure correct dtype
                    sub_result = self._ensure_dtype(sub_result, self.model_dtype)
                    
                    sub_chunks.append(sub_result.cpu())  # Сразу на CPU
                    
                    # Очищаем память
                    del sub_chunk, sub_result
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                
                # Объединяем результаты
                if sub_chunks:
                    result = torch.cat(sub_chunks, dim=-1).to(self.device)
                    del sub_chunks
            else:
                # Обрабатываем целиком
                if self.use_mixed_precision and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast(enabled=True):
                        result = self.nnet(chunk)
                else:
                    result = self.nnet(chunk)
                
                # Ensure correct dtype
                result = self._ensure_dtype(result, self.model_dtype)
            
            return result
            
        finally:
            # Освобождаем исходный чанк
            del chunk
            if self.device.type == 'cuda' and not self.use_mixed_precision:
                torch.cuda.empty_cache()

    def ola_forward(self, x):
        """
        Максимально эффективная по памяти forward функция.
        """
        self._clear_memory()
        
        assert x.ndim == 3
        batch, channels, n_frames = x.size()
        
        # Ensure input has correct dtype
        x = self._ensure_dtype(x, self.model_dtype)
        
        # Получаем границы VAD
        audio_np = x[0, 0, :].cpu().float().numpy()
        
        try:
            if self.vad_method == "spec":
                starts, ends = magspec_vad(
                    audio_np,
                    n_fft=self.window_size,
                    hop_length=self.hop_size,
                )
            elif self.vad_method == "webrtc":
                starts, ends = webrtc_vad(
                    audio_np, self.sr, vad_mode=3, frame_size=0.03
                )
        finally:
            # Освобождаем numpy массив
            del audio_np
            gc.collect()
        
        # Создаем выходной тензор на CPU для экономии GPU памяти
        out_cpu = (x.cpu().float() / self.n_src).repeat(1, self.n_src, 1)
        
        # Переносим x обратно на GPU если нужно
        x = x.to(self.device)
        
        # Буфер для предыдущих признаков
        prev_features = None
        
        for frame_idx in range(len(starts)):
            # Очищаем память перед каждым чанком
            if frame_idx % 2 == 0:
                self._clear_memory()
            
            start, end = starts[frame_idx], ends[frame_idx]
            
            # Обрабатываем чанк с минимальным использованием памяти
            frame = self._process_chunk_memory_efficient(x, start, end)
            
            if frame is None:
                continue
                
            if frame_idx == 0:
                n_src = frame.shape[1]
                
                # Вычисляем признаки для первого чанка
                prev_features = self._compute_spectral_features_memory_efficient(frame)
                
                if prev_features is not None:
                    if not hasattr(self, 'sc_avg'):
                        self.sc_avg = AverageMeter()
                    self.sc_avg.update(prev_features)
            
            elif frame_idx != 0 and self.reorder_chunks and prev_features is not None:
                # Вычисляем признаки для текущего чанка
                current_features = self._compute_spectral_features_memory_efficient(frame)
                
                if current_features is not None and prev_features is not None:
                    # Простая перестановка на основе минимального расстояния
                    with torch.no_grad():
                        # Вычисляем расстояния между признаками
                        distances = torch.cdist(
                            current_features.squeeze(0).float(),
                            prev_features.squeeze(0).float()
                        )
                        
                        # Находим наилучшее соответствие
                        perm = distances.argmin(dim=0)
                        
                        # Применяем перестановку
                        if not torch.all(perm == torch.arange(n_src, device=perm.device)):
                            frame = frame[:, perm]
                    
                    # Обновляем признаки
                    prev_features = current_features
                    
                    # Обновляем среднее
                    if hasattr(self, 'sc_avg'):
                        self.sc_avg.update(prev_features)
                    
                    # Очищаем
                    del current_features
                    
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            # Копируем результат в выходной тензор (на CPU) с правильным типом
            frame_cpu = frame.cpu().float()
            out_cpu[..., start:end] = frame_cpu.to(out_cpu.dtype)
            
            # Очищаем frame
            del frame, frame_cpu
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            self._processed_chunks += 1
            
            # Периодическая полная очистка
            if self._processed_chunks >= self._cleanup_threshold:
                self._clear_memory()
        
        # Финальная очистка
        self._clear_memory()
        
        # Возвращаем с правильным типом
        return out_cpu.to(device=self.device, dtype=self.model_dtype)

    def forward(self, x):
        """
        Forward с максимальной экономией памяти.
        """
        # Устанавливаем градиенты если нужно
        with torch.set_grad_enabled(self.enable_grad):
            
            # Если входной тензор слишком большой, обрабатываем по частям
            if x.numel() > 1e7:  # ~10M сэмплов
                chunk_size = int(15 * self.sr)  # 30 секунд
                outputs = []
                
                try:
                    for i in range(0, x.size(-1), chunk_size):
                        end = min(i + chunk_size, x.size(-1))
                        x_chunk = x[..., i:end]
                        
                        # Обрабатываем чанк
                        out_chunk = self.ola_forward(x_chunk)
                        outputs.append(out_chunk.cpu())
                        
                        # Очищаем память
                        del x_chunk, out_chunk
                        self._clear_memory()
                    
                    # Объединяем результаты
                    if outputs:
                        result = torch.cat(outputs, dim=-1)
                        return result.to(device=self.device, dtype=self.model_dtype)
                    else:
                        return torch.zeros_like(x)
                        
                finally:
                    del outputs
                    self._clear_memory()
            else:
                return self.ola_forward(x)