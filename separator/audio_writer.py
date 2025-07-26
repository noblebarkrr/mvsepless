from pydub import AudioSegment
import numpy as np

def write_audio_file(output_file_path, numpy_array, sample_rate, output_format, bitrate):
    """
    Записывает аудиофайл из numpy массива в указанном формате с помощью pydub.
    
    Параметры:
        output_file_path (str): Путь для сохранения файла (без расширения)
        numpy_array (numpy.ndarray): Аудиоданные в виде numpy массива
        sample_rate (int): Частота дискретизации (в Гц)
        output_format (str): Формат выходного файла ('mp3', 'flac', 'wav', 'aiff', 'm4a', 'aac', 'ogg', 'opus')
        encoder_settings (dict, optional): Cловарь с настройками кодировки аудио
    """
    try:
        # Проверка и нормализация входных данных
        if not isinstance(numpy_array, np.ndarray):
            raise ValueError("Input must be a numpy array")
        
        # Преобразование в правильную форму (samples, channels)
        if len(numpy_array.shape) == 1:
            numpy_array = numpy_array.reshape(-1, 1)  # Моно
        elif len(numpy_array.shape) == 2:
            if numpy_array.shape[0] == 2:  # Если (channels, samples)
                numpy_array = numpy_array.T  # Транспонируем в (samples, channels)
        else:
            raise ValueError("Input array must be 1D or 2D")
        
        # Нормализация до диапазона [-1.0, 1.0] если нужно
        if np.issubdtype(numpy_array.dtype, np.floating):
            numpy_array = np.clip(numpy_array, -1.0, 1.0)
            numpy_array = (numpy_array * 32767).astype(np.int16)
        elif numpy_array.dtype != np.int16:
            numpy_array = numpy_array.astype(np.int16)
        
        # Создание AudioSegment
        if numpy_array.shape[1] == 1:  # Моно
            audio_segment = AudioSegment(
                numpy_array.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,  # 16-bit = 2 bytes
                channels=1
            )
        else:  # Стерео
            # Для стерео нужно чередовать байты левого и правого каналов
            interleaved = np.empty((numpy_array.shape[0] * 2,), dtype=np.int16)
            interleaved[0::2] = numpy_array[:, 0]  # Левый канал
            interleaved[1::2] = numpy_array[:, 1]  # Правый канал
            audio_segment = AudioSegment(
                interleaved.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=2
            )
        
        # Формирование параметров экспорта
        
        parameters = {}            
        if bitrate:
            parameters['bitrate'] = bitrate
        
        # Поддержка различных форматов
        format_mapping = {
            'mp3': 'mp3',
            'flac': 'flac',
            'wav': 'wav',
            'aiff': 'aiff',
            'm4a': 'ipod',  # для m4a в pydub используется кодек ipod
            'aac': 'adts',  # для aac в pydub используется adts
            'ogg': 'ogg',
            'opus': 'opus'
        }
        
        if output_format not in format_mapping:
            raise ValueError(f"Unsupported format: {output_format}. Supported formats are: {list(format_mapping.keys())}")
        
        # Добавление расширения файла, если его нет
        if not output_file_path.lower().endswith(f'.{output_format}'):
            output_file_path = f"{output_file_path}.{output_format}"
        
        # Экспорт в нужный формат
        audio_segment.export(output_file_path, format=format_mapping[output_format], **parameters)
    
    except Exception as e:
        raise RuntimeError(f"Error writing audio file: {str(e)}")