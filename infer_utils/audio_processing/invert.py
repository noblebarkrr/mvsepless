import soundfile as sf
import numpy as np

def invert_and_overlay_wav(audio1_path, audio2_path, output_path):
    """
    Инвертирует audio1, накладывает на audio2 и сохраняет в WAV (PCM 16-bit).
    
    Поддерживает float/int, автоматически конвертирует в 16-bit.
    """
    # Читаем данные и параметры
    data1, sr1 = sf.read(audio1_path, dtype='float32')
    data2, sr2 = sf.read(audio2_path, dtype='float32')
    
    # Проверяем совпадение частоты и числа каналов
    if sr1 != sr2:
        raise ValueError("Частота дискретизации не совпадает!")
    if data1.ndim != data2.ndim:
        raise ValueError("Число каналов не совпадает!")
    
    # Инвертируем первый сигнал
    inverted_data1 = -data1
    
    # Обрезаем до минимальной длины
    min_len = min(len(inverted_data1), len(data2))
    inverted_data1 = inverted_data1[:min_len]
    data2 = data2[:min_len]
    
    # Микшируем (инвертированный audio1 + audio2)
    mixed = data2 + inverted_data1
    
    # Нормализуем, чтобы избежать клиппинга
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    
    # Конвертируем в 16-bit PCM и сохраняем
    mixed_int16 = (mixed * 32767).astype(np.int16)
    sf.write(output_path, mixed_int16, sr1, subtype='PCM_16')

# Пример использования:
# invert_and_overlay_wav("input1.wav", "input2.wav", "output.wav")


