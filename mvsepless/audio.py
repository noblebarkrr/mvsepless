import os
import subprocess
import numpy as np
from gradio_helper import str2bool
from scipy.signal import ShortTimeFFT, resample
from scipy.signal.windows import dpss, hann
from numpy.typing import DTypeLike
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
from i18n import _i18n

ffmpeg_path = "ffmpeg"
ffprobe_path = "ffprobe"
n_fft = 4096
hop = 1024


def average(*ints: Union[int, float]) -> float:
    """
    Вычислить среднее арифметическое
    
    Args:
        *ints: Числа для усреднения
    
    Returns:
        Среднее значение
    """
    numbers = len(ints)
    return sum(ints) / numbers


def check_installed() -> None:
    """Проверить наличие ffmpeg и ffprobe"""
    try:
        ffmpeg_version_output = subprocess.check_output(
            [ffmpeg_path, "-version"], text=True
        )
        print(_i18n("ffmpeg_found"))
    except:
        print(_i18n("ffmpeg_not_found"))

    try:
        ffprobe_version_output = subprocess.check_output(
            [ffprobe_path, "-version"], text=True
        )
        print(_i18n("ffprobe_found"))
    except:
        print(_i18n("ffprobe_not_found"))


def get_ogg_bitrate(sample_rate: int, channels: int = 2) -> int:
    """
    Определяет рекомендуемый битрейт для OGG на основе частоты дискретизации
    
    Args:
        sample_rate: Частота дискретизации
        channels: Количество каналов
    
    Returns:
        Рекомендуемый битрейт
    """
    if sample_rate >= 40000:
        per_channel = 240
    elif sample_rate >= 26000:
        per_channel = 190
    elif sample_rate >= 15000:
        per_channel = 90
    elif sample_rate >= 9000:
        per_channel = 50
    elif sample_rate >= 8000:
        per_channel = 42
    else:
        per_channel = 30
    
    return int(per_channel * channels)


SAMPLE_FORMATS_DICT: Dict[Union[str, type], str] = {
    "int16": "s16le",
    "int32": "s32le",
    "float32": "f32le",
    "float64": "f64le",
    np.int16: "s16le",
    np.int32: "s32le",
    np.float32: "f32le",
    np.float64: "f64le",
}

audio_formats: List[str] = [
    'aac', 'ac3', 'ac4', 'adts', 'aiff', 'au', 'caf', 'dts', 'eac3',
    'flac', 'm4a', 'mp3', 'mp2', 'ogg', 'oga', 'opus', 'ra', 'raw',
    'snd', 'voc', 'wav', 'wma', 'wv'
]

video_formats_with_audio: List[str] = [
    '3gp', '3g2', 'asf', 'avi', 'flv', 'f4v', 'm4v', 'mkv', 'mov',
    'mp4', 'mpeg', 'mpg', 'mts', 'mxf', 'ogv', 'rm', 'rmvb', 'ts',
    'vob', 'webm', 'wmv'
]

input_formats: List[str] = video_formats_with_audio + audio_formats

output_formats: List[str] = [
    "mp3", "wav", "flac", "ogg", "opus", "m4a", "aac", "ac3", "aiff", "wma"
]

input_extensions: List[str] = [f".{of}" for of in input_formats]

output_extensions: List[str] = [f".{of}" for of in output_formats]

codec_args: Dict[str, Dict[bool, List[str]]] = {
    ".mp3": {
        True: ["-c:a", "libmp3lame", "-sample_fmt", "fltp"],
        False: ["-c:a", "libmp3lame", "-sample_fmt", "s16p"]
    },
    ".wav": {
        True: ["-c:a", "pcm_f32le", "-sample_fmt", "flt"],
        False: ["-c:a", "pcm_s16le", "-sample_fmt", "s16"]
    },
    ".flac": {
        True: ["-c:a", "flac", "-sample_fmt", "s32"],
        False: ["-c:a", "flac", "-sample_fmt", "s16"]
    },
    ".ogg": {
        True: ["-c:a", "libvorbis", "-sample_fmt", "fltp"],
        False: ["-c:a", "libvorbis", "-sample_fmt", "fltp"]
    },
    ".opus": {
        True: ["-c:a", "libopus", "-sample_fmt", "flt"],
        False: ["-c:a", "libopus", "-sample_fmt", "s16"]
    },
    ".m4a": {
        True: ["-c:a", "aac", "-sample_fmt", "fltp"],
        False: ["-c:a", "aac", "-sample_fmt", "fltp"]
    },
    ".aac": {
        True: ["-c:a", "aac", "-sample_fmt", "fltp"],
        False: ["-c:a", "aac", "-sample_fmt", "fltp"]
    },
    ".ac3": {
        True: ["-c:a", "ac3", "-sample_fmt", "fltp"],
        False: ["-c:a", "ac3", "-sample_fmt", "fltp"]
    },
    ".aiff": {
        True: ["-c:a", "pcm_f32be", "-sample_fmt", "flt"],
        False: ["-c:a", "pcm_s16be", "-sample_fmt", "s16"]
    },
    ".wma": {
        True: ["-c:a", "wmav2", "-sample_fmt", "fltp"],
        False: ["-c:a", "wmav2", "-sample_fmt", "fltp"]
    }
}


def get_codec_args(extension: str, prefer_float: bool) -> List[str]:
    """
    Получить аргументы кодека для FFmpeg
    
    Args:
        extension: Расширение файла
        prefer_float: Предпочитать float формат
    
    Returns:
        Список аргументов FFmpeg
    """
    if extension not in codec_args:
        return []
    return codec_args[extension][prefer_float]


allowed_chars: str = r"1234567890"


def sanitize_output(output: str) -> str:
    """
    Очистить вывод от посторонних символов
    
    Args:
        output: Выходная строка
    
    Returns:
        Очищенная строка
    """
    return "".join([char for char in output if char in allowed_chars])


def get_sr(path: str, stream: int = 0) -> int:
    """
    Получить частоту дискретизации аудиофайла
    
    Args:
        path: Путь к файлу
        stream: Номер аудиопотока
    
    Returns:
        Частота дискретизации
    """
    cmd = [ffprobe_path, "-i", path, "-v", "quiet", "-hide_banner", 
           "-show_entries", "stream=sample_rate", "-select_streams", f"a:{stream}", 
           "-of", "compact=p=0:nk=1"]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    sample_rate = stdout.decode('utf-8').strip()
    sample_rate = sanitize_output(sample_rate)
    if sample_rate.isdigit():
        return int(sample_rate)
    else:
        print(_i18n("sr_read_error", path=path))
        return 0


def get_channels(path: str, stream: int = 0) -> int:
    """
    Получить количество каналов аудиофайла
    
    Args:
        path: Путь к файлу
        stream: Номер аудиопотока
    
    Returns:
        Количество каналов
    """
    cmd = [ffprobe_path, "-i", path, "-v", "quiet", "-hide_banner", 
           "-show_entries", "stream=channels", "-select_streams", f"a:{stream}", 
           "-of", "compact=p=0:nk=1"]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    channels = stdout.decode('utf-8').strip()
    channels = sanitize_output(channels)
    if channels.isdigit():
        return int(channels)
    else:
        print(_i18n("channels_read_error", path=path))
        return 0


def check(path: str) -> bool:
    """
    Проверить, является ли файл валидным аудио
    
    Args:
        path: Путь к файлу
    
    Returns:
        True если файл содержит аудио
    """
    channels = get_channels(path)
    sr = get_sr(path)
    return channels != 0 and sr != 0


def read(
    path: str, 
    sr: Optional[int] = None, 
    mono: bool = False, 
    dtype: DTypeLike = "float32", 
    multi_channel: bool = False, 
    num_channels: int = 2, 
    stream: int = 0, 
    flatten: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Прочитать аудиофайл
    
    Args:
        path: Путь к файлу
        sr: Частота дискретизации
        mono: Читать как моно
        dtype: Тип данных
        multi_channel: Многоканальный режим
        num_channels: Количество каналов
        stream: Номер аудиопотока
        flatten: Вернуть плоский массив
    
    Returns:
        Кортеж (аудиоданные, частота дискретизации)
    """
    output_format = SAMPLE_FORMATS_DICT.get(dtype, None)
    if not sr:
        sr = get_sr(path, stream)
    channels = 1 if mono else (get_channels(path, stream) if multi_channel else num_channels)
    
    if not output_format:
        output_format = "f32le"
        cmd = [ffmpeg_path, "-i", path, "-map", f"0:a:{stream}", "-vn", 
               "-f", output_format, "-ac", str(channels), "-ar", str(sr), "-"]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
        )
        stdout, stderr = process.communicate()
        y = np.frombuffer(stdout, dtype=np.float32)
        y = convert_to_dtype(y, dtype)
    else:
        cmd = [ffmpeg_path, "-i", path, "-map", f"0:a:{stream}", "-vn", 
               "-f", output_format, "-ac", str(channels), "-ar", str(sr), "-"]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
        )
        stdout, stderr = process.communicate()
        y = np.frombuffer(stdout, dtype=dtype)
    
    if mono:
        if flatten:
            y = y.flatten()
        else:
            y = y.reshape((-1, 1)).T
    else:
        y = y.reshape((-1, channels)).T
    
    return y.copy(), sr


def multiread(
    paths: Union[List[str], Tuple[str, ...]], 
    *args, 
    **kwargs
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Прочитать несколько аудиофайлов
    
    Args:
        paths: Список путей к файлам
        *args: Аргументы для read
        **kwargs: Именованные аргументы для read
    
    Returns:
        Кортеж (список аудиоданных, список частот дискретизации)
    """
    readed_files = []
    srs = []
    len_arrays = len(paths)
    for i, path in enumerate(paths, start=1):
        array, sr = read(path, *args, **kwargs)
        readed_files.append(array)
        srs.append(sr)
        print(_i18n("reading_progress", current=i, total=len_arrays), end="\r")
    print("")
    return readed_files, srs


def bitrate_to_int(a: Union[str, int, float]) -> int:
    """
    Преобразовать битрейт в целое число
    
    Args:
        a: Битрейт в виде строки или числа
    
    Returns:
        Битрейт как целое число
    """
    if isinstance(a, str):
        if a.endswith(("k", "K")):
            numeric_part = a[:-1]
            if numeric_part.isdigit():
                return int(numeric_part)
            else:
                print(_i18n("invalid_bitrate", bitrate=a))
                return 320
        else:
            if a.isdigit():
                return int(a)
            else:
                print(_i18n("invalid_bitrate", bitrate=a))
                return 320
    elif isinstance(a, (int, float)):
        return int(a)
    else:
        return 320


def get_info_array(y: np.ndarray) -> Tuple[int, int, Optional[int], bool]:
    """
    Получить информацию об аудио массиве
    
    Args:
        y: Аудио массив
    
    Returns:
        Кортеж (количество каналов, количество сэмплов, индекс оси, флаг flatten)
    """
    if y.ndim == 1:
        flatten = True
        channels = 1
        samples = len(y)
        array_index = -1
    elif y.ndim == 2:
        flatten = False
        if y.shape[0] < y.shape[1]:
            channels = y.shape[0]
            samples = y.shape[1]
            array_index = 1
        else:
            channels = y.shape[1]
            samples = y.shape[0]
            array_index = 0
    else:
        raise ValueError(_i18n("array_dim_error"))
    return channels, samples, array_index, flatten


def get_axis_from_array_index(index: int) -> int:
    """
    Получить ось для операций на основе индекса массива
    
    Args:
        index: Индекс массива
    
    Returns:
        Номер оси
    """
    if index == -1:
        return -1
    elif index == 1:
        return 0
    elif index == 0:
        return 1
    else:
        return -1


def get_duration_from_array(y: np.ndarray, sr: Optional[int] = None) -> Union[float, int]:
    """
    Получить длительность аудио из массива
    
    Args:
        y: Аудио массив
        sr: Частота дискретизации
    
    Returns:
        Длительность в секундах или количество сэмплов
    """
    len_samples: int = get_info_array(y)[1]
    if sr is not None:
        return len_samples / sr
    else:
        return len_samples


def is_float(y: np.ndarray) -> bool:
    """
    Проверить, является ли массив float типом
    
    Args:
        y: Аудио массив
    
    Returns:
        True если тип float
    """
    return np.issubdtype(y.dtype, np.floating)


def is_float_dtype(dtype: DTypeLike) -> bool:
    """
    Проверить, является ли тип данных float
    
    Args:
        dtype: Тип данных
    
    Returns:
        True если тип float
    """
    return np.issubdtype(dtype, np.floating)


def float_to_int(y: np.ndarray, dtype: DTypeLike) -> np.ndarray:
    """
    Преобразовать float массив в целочисленный
    
    Args:
        y: Float массив
        dtype: Целевой тип данных
    
    Returns:
        Целочисленный массив
    """
    info = np.iinfo(dtype)
    min_val = info.min
    max_val = info.max
    
    if min_val < 0:
        y_scaled = y * max_val
        y_rounded = np.round(y_scaled)
        y_clipped = np.clip(y_rounded, min_val, max_val)
        return y_clipped.astype(dtype)
    elif min_val == 0:
        y_normalized = (y + 1) / 2
        y_scaled = y_normalized * max_val
        y_rounded = np.round(y_scaled)
        y_clipped = np.clip(y_rounded, 0, max_val)
        return y_clipped.astype(dtype)
    else:
        raise ValueError(_i18n("unexpected_min_val", value=min_val))


def int_to_int(y: np.ndarray, dtype: DTypeLike) -> np.ndarray:
    """
    Преобразовать целочисленный массив в другой целочисленный тип
    
    Args:
        y: Целочисленный массив
        dtype: Целевой тип данных
    
    Returns:
        Преобразованный массив
    """
    info_dst = np.iinfo(dtype)
    info_src = np.iinfo(y.dtype)
    y_float = y.astype(np.float64)
    src_range = info_src.max - info_src.min
    dst_range = info_dst.max - info_dst.min
    if src_range == 0:
        return np.full_like(y, info_dst.min, dtype=dtype)
    y_scaled = (y_float - info_src.min) * (dst_range / src_range) + info_dst.min
    y_rounded = np.round(y_scaled)
    y_clipped = np.clip(y_rounded, info_dst.min, info_dst.max)
    return y_clipped.astype(dtype)


def int_to_float(y: np.ndarray, dtype: DTypeLike) -> np.ndarray:
    """
    Преобразовать целочисленный массив в float
    
    Args:
        y: Целочисленный массив
        dtype: Целевой тип данных
    
    Returns:
        Float массив
    """
    info = np.iinfo(y.dtype)
    if info.min == 0:
        y_normalized = (y.astype(np.float64) + -int(average(info.min, info.max))) / info.max
    elif info.min < 0:
        abs_max = max(abs(info.min), abs(info.max))
        y_normalized = y.astype(np.float64) / abs_max
    else:
        raise ValueError(_i18n("unexpected_min_val", value=info.min))
    return y_normalized.astype(dtype)


def float_to_float(y: np.ndarray, dtype: DTypeLike) -> np.ndarray:
    """
    Преобразовать float массив в другой float тип
    
    Args:
        y: Float массив
        dtype: Целевой тип данных
    
    Returns:
        Преобразованный массив
    """
    return y.astype(dtype)


def get_center_value_from_dtype(dtype: DTypeLike) -> int:
    """
    Получить центральное значение для типа данных
    
    Args:
        dtype: Тип данных
    
    Returns:
        Центральное значение
    """
    if is_float_dtype(dtype):
        return 0
    else:
        info = np.iinfo(dtype)
        return int(average(info.min, info.max))


def convert_to_dtype(y: np.ndarray, dtype: DTypeLike) -> np.ndarray:
    """
    Преобразовать массив в указанный тип данных
    
    Args:
        y: Входной массив
        dtype: Целевой тип данных
    
    Returns:
        Преобразованный массив
    """
    if is_float(y):
        if is_float_dtype(dtype):
            return float_to_float(y, dtype)
        else:
            return float_to_int(y, dtype)
    else:
        if is_float_dtype(dtype):
            return int_to_float(y, dtype)
        else:
            return int_to_int(y, dtype)


def dc_offset(y: np.ndarray, offset: Union[float, int]) -> np.ndarray:
    """
    Добавить смещение постоянного тока
    
    Args:
        y: Аудио массив
        offset: Смещение
    
    Returns:
        Массив со смещением
    """
    orig_dtype = y.dtype
    y = convert_to_dtype(y, np.float32)
    y = y + offset
    return convert_to_dtype(y, orig_dtype)


def gain(y: np.ndarray, gain_value: Union[float, int]) -> np.ndarray:
    """
    Применить усиление к аудио
    
    Args:
        y: Аудио массив
        gain_value: Коэффициент усиления
    
    Returns:
        Усиленный массив
    """
    orig_dtype = y.dtype
    y = convert_to_dtype(y, np.float32)
    y = y * gain_value
    return convert_to_dtype(y, orig_dtype)


def normalize(y: np.ndarray, target_peak: Union[float, int] = 1.0) -> np.ndarray:
    """
    Нормализовать аудио по пиковому значению
    
    Args:
        y: Аудио массив
        target_peak: Целевое пиковое значение
    
    Returns:
        Нормализованный массив
    """
    orig_dtype = y.dtype
    y = convert_to_dtype(y, np.float32)
    current_peak = np.max(np.abs(y))
    if current_peak > 0:
        scaling_factor = target_peak / current_peak
        y = y * scaling_factor
    return convert_to_dtype(y, orig_dtype)


def create_zero_array(samples: int, dtype: DTypeLike) -> np.ndarray:
    """
    Создать массив нулей с центром для типа данных
    
    Args:
        samples: Количество сэмплов
        dtype: Тип данных
    
    Returns:
        Массив нулей
    """
    return np.array([get_center_value_from_dtype(dtype) for _c in range(samples)], dtype=dtype)


def split_channels(y: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Разделить многоканальное аудио на отдельные каналы
    
    Args:
        y: Аудио массив
    
    Returns:
        Кортеж массивов каналов
    """
    channels, samples, array_index, flatten = get_info_array(y)
    channels_arrays = []
    if not flatten:
        if array_index == 1:
            for ch in range(channels):
                channels_arrays.append(y[ch, :])
        else:
            for ch in range(channels):
                channels_arrays.append(y[:, ch])
        return tuple(channels_arrays)
    else:
        return (y,)


from scipy.signal import windows


def get_stft_obj(sr: int, n_fft: int, hop: int) -> ShortTimeFFT:
    """
    Создает STFT с окном DPSS для сверхточного разделения частот
    
    Args:
        sr: Частота дискретизации
        n_fft: Размер FFT
        hop: Шаг
    
    Returns:
        Объект ShortTimeFFT
    """
    win_dpss = str2bool(os.environ.get("MVSEPLESS_DPSS", "False"))
    if win_dpss:
        win = dpss(n_fft, NW=3, sym=False)
    else:
        win = hann(n_fft, sym=False)
    return ShortTimeFFT(win, hop=hop, fs=sr, scale_to='magnitude', phase_shift=None)


def split_mid_side(
    y: np.ndarray, 
    var: int = 1, 
    sr: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Разделить стерео на Mid/Side
    
    Args:
        y: Аудио массив
        var: Вариант разделения (0-4)
        sr: Частота дискретизации
    
    Returns:
        Кортеж (mid, side)
    """
    channels, samples, array_index, flatten = get_info_array(y)
    axis = get_axis_from_array_index(array_index)
    if channels != 2:
        raise Exception(_i18n("stereo_required"))
    orig_dtype = y.dtype
    y = convert_to_dtype(y, np.float32)
    channels_arrays = split_channels(y)
    left_channel = channels_arrays[0]
    right_channel = channels_arrays[1]
    mid_channel_one = (left_channel * 0.5) + (right_channel * 0.5)
    
    if var == 0:
        print(_i18n("mid_side_var0"))
        side_channel = np.stack([(left_channel + -mid_channel_one), (right_channel + -mid_channel_one)], axis=axis)
        mid_channel = y + -side_channel
    elif var == 1:
        print(_i18n("mid_side_var1"))
        mid_channel = np.stack([mid_channel_one, mid_channel_one], axis=axis)
        side_channel = y + -mid_channel
    elif var == 2:
        print(_i18n("mid_side_var2"))
        same_sign = (left_channel * right_channel) > 0
        center_mono = np.where(
            same_sign,
            np.minimum(np.abs(left_channel), np.abs(right_channel)) * np.sign(left_channel),
            0.0
        )
        mid_channel = np.stack([center_mono, center_mono], axis=axis)
        stereo_L = left_channel - center_mono
        stereo_R = right_channel - center_mono
        side_channel = np.stack([stereo_L, stereo_R], axis=axis)
    elif var == 3:
        print(_i18n("mid_side_var3"))
        if not sr: 
            raise Exception(_i18n("sr_required"))
        
        sft = get_stft_obj(sr, n_fft=n_fft, hop=hop)
        y_float = convert_to_dtype(y, np.float32)
        channels = split_channels(y_float)
        
        # Получаем спектры левого и правого каналов
        Lf = sft.stft(channels[0])
        Rf = sft.stft(channels[1])
        
        # Вычисляем схожесть (когерентность)
        similarity_L = np.real(Lf * np.conj(Rf))
        similarity_R = np.real(Rf * np.conj(Lf))
        mask_l = similarity_L > 0
        mask_r = similarity_R > 0
        magL = np.abs(Lf)
        magR = np.abs(Rf)

        magC_L = np.minimum(magL, magR) * mask_l
        magC_R = np.minimum(magL, magR) * mask_r

        C_L = magC_L * np.exp(1j * np.angle(Rf))
        C_R = magC_R * np.exp(1j * np.angle(Lf))
        SL = Lf - C_L
        SR = Rf - C_R
        
        len_orig = y.shape[-1]
        center_l = sft.istft(C_L, k1=len_orig)
        center_r = sft.istft(C_R, k1=len_orig)
        side_l = sft.istft(SL, k1=len_orig)
        side_r = sft.istft(SR, k1=len_orig)
        
        mid_ch = multi_channel_array_from_arrays(center_l, center_r, index=array_index, dtype=y.dtype)
        side_ch = multi_channel_array_from_arrays(side_l, side_r, index=array_index, dtype=y.dtype)
        
        return mid_ch, side_ch
    elif var == 4:
        print(_i18n("mid_side_var4"))
        mid_channel = mid_channel_one
        side_channel = left_channel + -right_channel
    else:
        raise ValueError(_i18n("unknown_var", var=var))
    
    return convert_to_dtype(mid_channel, orig_dtype), convert_to_dtype(side_channel, orig_dtype)


def mid_side_to_stereo(
    y: np.ndarray, 
    z: np.ndarray, 
    index: int = -1, 
    dtype: DTypeLike = np.float32
) -> np.ndarray:
    """
    Преобразовать Mid/Side обратно в стерео
    
    Args:
        y: Mid канал
        z: Side канал
        index: Индекс оси
        dtype: Тип данных
    
    Returns:
        Стерео массив
    """
    y, z = convert_to_dtype(y, np.float32), convert_to_dtype(z, np.float32)
    mid = multi_channel_array_from_arrays(y, y, index=index, dtype=np.float32)
    side = multi_channel_array_from_arrays(z, -z, index=index, dtype=np.float32)
    return convert_to_dtype(mid + side, dtype)


def mono_to_stereo(
    y: np.ndarray, 
    index: int, 
    num_channels: int = 2
) -> np.ndarray:
    """
    Преобразовать моно в стерео
    
    Args:
        y: Моно массив
        index: Индекс оси
        num_channels: Количество каналов
    
    Returns:
        Стерео массив
    """
    channels, samples, array_index, flatten = get_info_array(y)
    axis = get_axis_from_array_index(array_index)
    new_axis = get_axis_from_array_index(index)
    orig_dtype = y.dtype
    if channels == 1:
        if flatten:
            return np.stack([y for _c in range(num_channels)], axis=new_axis, dtype=orig_dtype)
        else:
            return np.stack([y.flatten() for _c in range(num_channels)], axis=new_axis, dtype=orig_dtype)
    else:
        if num_channels <= channels:
            return y
        else:
            for _i in range(num_channels - channels):
                y = np.append(y, create_zero_array(samples, orig_dtype), axis=new_axis)
            return y


def stereo_to_mono(y: np.ndarray, to_flatten: bool = False) -> np.ndarray:
    """
    Преобразовать стерео в моно
    
    Args:
        y: Стерео массив
        to_flatten: Вернуть плоский массив
    
    Returns:
        Моно массив
    """
    channels, samples, array_index, flatten = get_info_array(y)
    orig_dtype = y.dtype
    y = convert_to_dtype(y, np.float32)
    if channels > 1:
        mono = create_zero_array(samples, np.float64)
        for ch in split_channels(y):
            mono = mono + gain(ch, (1 / channels))
        if not to_flatten:
            if array_index == 0:
                return mono.reshape((1, -1))
            else:
                return mono.reshape((-1, 1))
        else:
            return mono
    else:
        return y


def multi_channel_array_from_arrays(
    *arrays: np.ndarray, 
    index: int = -1, 
    dtype: DTypeLike
) -> np.ndarray:
    """
    Создать многоканальный массив из отдельных каналов
    
    Args:
        *arrays: Массивы каналов
        index: Индекс оси
        dtype: Тип данных
    
    Returns:
        Многоканальный массив
    """
    return np.stack([convert_to_dtype(array, dtype) for array in arrays], 
                    axis=get_axis_from_array_index(index), 
                    dtype=dtype)


def reshape(y: np.ndarray, shape: Tuple[str, ...] = ("channels", "samples")) -> np.ndarray:
    """
    Изменить форму аудио массива
    
    Args:
        y: Аудио массив
        shape: Целевая форма
    
    Returns:
        Измененный массив
    """
    channels, samples, array_index, flatten = get_info_array(y)
    
    if shape == ("channels", "samples"):
        if array_index == 0:
            return y.T
        elif array_index == 1:
            return y
        elif array_index is None and flatten:
            return y.reshape((-1, 1)).T
        else:
            if y.shape[0] == channels:
                return y
            else:
                return y.T
    
    elif shape == ("samples", "channels"):
        if array_index == 1:  # (channels, samples)
            return y.T
        elif array_index == 0:  # (samples, channels)
            return y
        elif array_index == -1 and flatten:
            return y.reshape((-1, 1))
        else:
            if y.shape[0] == samples:
                return y
            else:
                return y.T
    
    elif shape == ("samples",):
        if channels == 1 and not flatten:
            return y.flatten()
        elif flatten:
            return y
        else:
            return stereo_to_mono(y, to_flatten=True)
    
    else:
        raise ValueError(f"{_i18n('unknown_shape')}: {shape}")


def easy_resampler(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Простой ресемплинг аудио
    
    Args:
        y: Аудио массив
        orig_sr: Исходная частота
        target_sr: Целевая частота
    
    Returns:
        Ресемплированный массив
    """
    channels, samples, array_index, flatten = get_info_array(y)
    orig_dtype = y.dtype
    ratio = float(target_sr) / orig_sr
    n_samples = int(np.ceil(samples * ratio))
    resampled = resample(y, n_samples, axis=array_index)
    return convert_to_dtype(resampled, orig_dtype)


def add_zero_to_end(y: np.ndarray, max_samples: int) -> np.ndarray:
    """
    Добавить нули в конец массива до указанной длины
    
    Args:
        y: Аудио массив
        max_samples: Максимальное количество сэмплов
    
    Returns:
        Дополненный массив
    """
    channels, samples, array_index, flatten = get_info_array(y)
    center_value = get_center_value_from_dtype(y.dtype)
    if samples < max_samples:
        if flatten:
            pad_width = (0, max_samples - samples)
        else:
            if array_index == 1:
                pad_width = ((0, 0), (0, max_samples - samples))
            else:
                pad_width = ((0, max_samples - samples), (0, 0))
        return np.pad(y, pad_width, mode="constant", constant_values=center_value)
    else:
        return trim(y, 0, max_samples)


def fit_arrays(
    arrays: Union[Tuple[np.ndarray, ...], List[np.ndarray]], 
    srs: Union[Tuple[int, ...], List[int]], 
    max_channels: int = 2, 
    min_sr: int = 44100, 
    flatten: bool = False, 
    max_samples: int = -1, 
    extend: bool = True
) -> Tuple[np.ndarray, ...]:
    """
    Привести несколько массивов к единому формату
    
    Args:
        arrays: Список массивов
        srs: Список частот дискретизации
        max_channels: Максимальное количество каналов
        min_sr: Минимальная частота дискретизации
        flatten: Вернуть плоские массивы
        max_samples: Максимальное количество сэмплов
        extend: Дополнить до максимальной длины
    
    Returns:
        Кортеж приведенных массивов
    """
    if len(arrays) != len(srs):
        raise Exception(_i18n("arrays_srs_mismatch"))
    
    new_arrays = []
    
    arrays_with_srs = list(zip(arrays, srs))
    len_arrays = len(arrays_with_srs)
    
    durations = [get_duration_from_array(array) for array, _c in arrays_with_srs]
    max_samples = max(durations) if durations else 0

    for i, (array, sr) in enumerate(arrays_with_srs, start=1):
        channels1, samples1, array_index1, _c = get_info_array(array)
        a1 = easy_resampler(array, sr, min_sr)
        if flatten:
            a1 = stereo_to_mono(a1, to_flatten=True)
        else:
            if max_channels >= 2:
                a1 = mono_to_stereo(a1, array_index1, max_channels)
            else:
                a1 = stereo_to_mono(a1)
        a1 = reshape(a1, shape=("channels", "samples"))
        new_arrays.append(a1)
        print(_i18n("fitting_progress", current=i, total=len_arrays), end="\r")
    print("")
    
    if extend:
        for i, array_ in enumerate(new_arrays):
            new_arrays[i] = add_zero_to_end(array_, max_samples)
            print(_i18n("extending_progress", current=i, total=len_arrays), end="\r")
        print("")
    
    return tuple(new_arrays)


def subtractor(
    y: np.ndarray, 
    z: np.ndarray, 
    sr1: int, 
    sr2: int, 
    spectrogram: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Вычесть одно аудио из другого
    
    Args:
        y: Первое аудио
        z: Второе аудио
        sr1: Частота первого
        sr2: Частота второго
        spectrogram: Использовать спектрограмму
    
    Returns:
        Кортеж (результат, частота дискретизации)
    """
    channels1, _, array_index1, flatten1 = get_info_array(y)
    channels2, _, array_index2, flatten2 = get_info_array(z)
    orig_dtype1 = y.dtype
    y = convert_to_dtype(y, np.float32)
    z = convert_to_dtype(z, np.float32)
    max_channels = max(channels1, channels2)
    min_sr = min(sr1, sr2)
    yz = fit_arrays([y, z], [sr1, sr2], max_channels=max_channels, min_sr=min_sr)
    y, z = yz[0], yz[1]
    
    if spectrogram:
        print(_i18n("subtract_spectrogram"))
        sft = get_stft_obj(min_sr, n_fft=n_fft, hop=hop)
        res_channels = []
        
        # Обрабатываем каналы по одному, чтобы не забивать RAM
        for ch_y, ch_z in zip(split_channels(y), split_channels(z)):
            spec_y = sft.stft(ch_y.astype(np.float32))
            spec_z = sft.stft(ch_z.astype(np.float32))
            
            # Вычитание амплитуд: Mag_res = max(Mag_y - Mag_z, 0)
            # Сохраняем фазу сигнала 'y'
            res_spec = np.maximum(np.abs(spec_y) - np.abs(spec_z), 0) * np.exp(1j * np.angle(spec_y))
            
            del spec_y, spec_z  # Явно освобождаем память
            
            res_wav = sft.istft(res_spec, k1=ch_y.shape[-1])
            res_channels.append(res_wav)
            
        subtracted = multi_channel_array_from_arrays(*res_channels, index=1, dtype=orig_dtype1)
        return subtracted, min_sr
    else:
        print(_i18n("subtract_phase"))
        return convert_to_dtype(y - z, orig_dtype1), min_sr


def absmax(a: np.ndarray, *, axis: Optional[int] = None) -> np.ndarray:
    """
    Получить элемент с максимальным абсолютным значением
    
    Args:
        a: Входной массив
        axis: Ось
    
    Returns:
        Элемент с максимальным абсолютным значением
    """
    if axis is None:
        return a.flatten()[np.argmax(np.abs(a))]
    dims = list(a.shape)
    dims.pop(axis)
    indices = np.ogrid[tuple(slice(0, d) for d in dims)]
    argmax = np.abs(a).argmax(axis=axis)
    indices = list(indices)
    indices.insert(axis % len(a.shape), argmax)
    return a[tuple(indices)]


def absmin(a: np.ndarray, *, axis: Optional[int] = None) -> np.ndarray:
    """
    Получить элемент с минимальным абсолютным значением
    
    Args:
        a: Входной массив
        axis: Ось
    
    Returns:
        Элемент с минимальным абсолютным значением
    """
    if axis is None:
        return a.flatten()[np.argmin(np.abs(a))]
    dims = list(a.shape)
    dims.pop(axis)
    indices = np.ogrid[tuple(slice(0, d) for d in dims)]
    argmax = np.abs(a).argmin(axis=axis)
    indices.insert((len(a.shape) + axis) % len(a.shape), argmax)
    return a[tuple(indices)]


def lambda_max(
    arr: np.ndarray, 
    axis: Optional[int] = None, 
    key: Optional[Callable] = None, 
    keepdims: bool = False
) -> np.ndarray:
    """
    Применить функцию максимума с ключом
    
    Args:
        arr: Входной массив
        axis: Ось
        key: Функция ключа
        keepdims: Сохранить размерность
    
    Returns:
        Результат
    """
    if key is None:
        key = np.abs
    idxs = np.argmax(key(arr), axis)
    if axis is not None:
        idxs = np.expand_dims(idxs, axis)
        result = np.take_along_axis(arr, idxs, axis)
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result
    else:
        return arr.flatten()[idxs]


def lambda_min(
    arr: np.ndarray, 
    axis: Optional[int] = None, 
    key: Optional[Callable] = None, 
    keepdims: bool = False
) -> np.ndarray:
    """
    Применить функцию минимума с ключом
    
    Args:
        arr: Входной массив
        axis: Ось
        key: Функция ключа
        keepdims: Сохранить размерность
    
    Returns:
        Результат
    """
    if key is None:
        key = np.abs
    idxs = np.argmin(key(arr), axis)
    if axis is not None:
        idxs = np.expand_dims(idxs, axis)
        result = np.take_along_axis(arr, idxs, axis)
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result
    else:
        return arr.flatten()[idxs]


def ensemble(
    pred_tracks: List[np.ndarray], 
    srs: List[int], 
    weights: List[float], 
    algorithm: str, 
    dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, int]:
    """
    Создать ансамбль из нескольких предсказаний
    
    Args:
        pred_tracks: Список предсказаний
        srs: Список частот дискретизации
        weights: Веса
        algorithm: Алгоритм объединения
        dtype: Тип данных
    
    Returns:
        Кортеж (результат, частота дискретизации)
    """
    if algorithm == "min_fft":
        max_sr = int(min(srs))
    else:
        max_sr = int(max(srs))
    
    # Подгоняем все треки к одной длине и частоте
    pred_tracks = list(fit_arrays(pred_tracks, srs, max_channels=2, min_sr=max_sr))
    
    sft = get_stft_obj(max_sr, n_fft=2048, hop=1024)
    final_length = pred_tracks[0].shape[-1]
    ensemble_wav_channels = []

    for ch_idx in range(2):  # Для каждого канала (L и R)
        accumulator = None
        total_weight = sum(weights)
        
        for i, track in enumerate(pred_tracks):
            # Извлекаем канал и считаем STFT
            spec = sft.stft(track[ch_idx].astype(np.float32))
            
            if algorithm == "avg_fft":
                weighted_spec = spec * weights[i]
                if accumulator is None:
                    accumulator = weighted_spec
                else:
                    accumulator += weighted_spec
            elif algorithm in ["min_fft", "max_fft", "median_fft"]:
                # Для медианы и экстремумов собираем стек для одного канала
                if i == 0:
                    accumulator = [spec]
                else:
                    accumulator.append(spec)
            
            del spec

        # Финализация алгоритма
        if algorithm == "avg_fft":
            res_spec = accumulator / total_weight
        elif algorithm == "median_fft":
            res_spec = np.median(np.real(accumulator), axis=0) + 1j * np.median(np.imag(accumulator), axis=0)
        elif algorithm == "min_fft":
            res_spec = lambda_min(np.array(accumulator), axis=0, key=np.abs)
        elif algorithm == "max_fft":
            res_spec = absmax(np.array(accumulator), axis=0)
        else:
            raise ValueError(_i18n("unknown_algorithm", alg=algorithm))
        
        ensemble_wav_channels.append(sft.istft(res_spec, k1=final_length))
        del accumulator

    result = multi_channel_array_from_arrays(*ensemble_wav_channels, index=1, dtype=dtype)
    print(_i18n("ensemble_complete"))
    return result, max_sr


def concatenate(
    arrays: Union[Tuple[np.ndarray, ...], List[np.ndarray]], 
    srs: Union[Tuple[int, ...], List[int]], 
    dtype=np.float32
) -> Tuple[np.ndarray, int]:
    """
    Склеить несколько аудио массивов
    
    Args:
        arrays: Список массивов
        srs: Список частот дискретизации
        dtype: Тип данных
    
    Returns:
        Кортеж (результат, частота дискретизации)
    """
    max_sr = int(max(*srs))
    arrayss = fit_arrays([convert_to_dtype(array, np.float64) for array in arrays], 
                         srs, max_channels=2, min_sr=max_sr, extend=False)
    result = np.concatenate(arrayss, axis=1, dtype=np.float64)
    print(_i18n("concatenate_complete"))
    return convert_to_dtype(result, dtype), max_sr


def trim(y: np.ndarray, start: int = 0, end: int = -1) -> np.ndarray:
    """
    Обрезать аудио массив
    
    Args:
        y: Аудио массив
        start: Начальная позиция
        end: Конечная позиция
    
    Returns:
        Обрезанный массив
    """
    channels, samples, array_index, flatten = get_info_array(y)
    end_index = samples
    _end = end if end > 0 and end <= end_index else end_index
    if flatten:
        return y[start:_end]
    elif array_index == 0:
        return y[start:_end, :]
    elif array_index == 1:
        return y[:, start:_end]
    else:
        return y


def reverse(y: np.ndarray) -> np.ndarray:
    """
    Перевернуть аудио массив
    
    Args:
        y: Аудио массив
    
    Returns:
        Перевернутый массив
    """
    channels, samples, array_index, flatten = get_info_array(y)
    if flatten:
        return np.flip(y)
    else:
        return np.flip(y, axis=array_index)


def write(
    path: str, 
    y: np.ndarray, 
    sr: int, 
    bitrate: Union[int, str] = 320, 
    prefer_float: bool = False
) -> str:
    """
    Записать аудио в файл
    
    Args:
        path: Путь для сохранения
        y: Аудио массив
        sr: Частота дискретизации
        bitrate: Битрейт
        prefer_float: Предпочитать float формат
    
    Returns:
        Путь к сохраненному файлу
    """
    if str2bool(os.environ.get("MVSEPLESS_WRITE_ABS", "False")):
        path = os.path.abspath(path)
    
    name, ext = os.path.splitext(path)
    dir_path = os.path.dirname(path)
    if dir_path != "":
        os.makedirs(dir_path, exist_ok=True)
    
    if not sr:
        raise Exception(_i18n("sr_required"))
    
    dtype = y.dtype
    channels, *_ = get_info_array(y)
    y = reshape(y, shape=("samples", "channels"))
    
    sample_format = SAMPLE_FORMATS_DICT.get(str(dtype), None)
    if not sample_format:
        sample_format = "f32le"
        y = convert_to_dtype(y, np.float32)
    
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)
    
    bitrate_val = bitrate_to_int(bitrate)
    if ext == ".ogg":
        max_bitrate = get_ogg_bitrate(sr, channels)
        if bitrate_val > max_bitrate:
            print(_i18n("ogg_bitrate_adjusted", old=bitrate_val, new=max_bitrate))
            bitrate_val = max_bitrate
    elif ext == ".opus":
        max_bitrate = 256 * channels
        if bitrate_val > max_bitrate:
            print(_i18n("opus_bitrate_adjusted", old=bitrate_val, new=max_bitrate))
            bitrate_val = max_bitrate
    
    bitrate_fixed = 32 if bitrate_val < 32 else 320 if bitrate_val > 320 else bitrate_val
    
    cmd = [ffmpeg_path, "-y", "-f", sample_format, "-ar", str(sr), "-ac", str(channels), 
           "-i", "-", *get_codec_args(ext, prefer_float), "-ab", f"{bitrate_fixed}k", path]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=None,
        stderr=subprocess.PIPE,
        bufsize=10**8
    )
    
    try:
        stdout_data, stderr_data = process.communicate(input=y.tobytes())
        
        if process.returncode != 0:
            error_msg = stderr_data.decode('utf-8', errors='ignore')
            print(_i18n("ffmpeg_error", error=error_msg))
            raise Exception(_i18n("ffmpeg_exit_code", code=process.returncode))
            
    except Exception as e:
        print(_i18n("write_critical_error", error=str(e)))
        process.kill()
        raise e

    return path


def multiwrite(
    arrays: Union[Tuple[np.ndarray, ...], List[np.ndarray]], 
    srs: Union[Tuple[int, ...], List[int]], 
    paths: Union[Tuple[str, ...], List[str]], 
    bitrate: Union[int, str] = 320, 
    prefer_float: bool = False, 
    callable_func: Optional[Callable] = None, 
    strict: bool = False
) -> Tuple[str, ...]:
    """
    Записать несколько аудио массивов в файлы
    
    Args:
        arrays: Список массивов
        srs: Список частот дискретизации
        paths: Список путей для сохранения
        bitrate: Битрейт
        prefer_float: Предпочитать float формат
        callable_func: Функция обратного вызова
        strict: Строгий режим
    
    Returns:
        Кортеж сохраненных путей
    """
    saved_paths = []
    exceptions = []
    
    if len(arrays) == len(srs) == len(paths):
        save_arrays = list(zip(arrays, srs, paths))
        for array, sr, path in save_arrays:
            if callable_func is not None:
                callable_func(path)
            try:
                saved_paths.append(write(path, array, sr, bitrate=bitrate, prefer_float=prefer_float))
            except Exception as e:
                if strict:
                    raise Exception(str(e))
                else:
                    print(_i18n("write_error", error=str(e)))
                    exceptions.append(str(e))
    
    if not saved_paths:
        exceptions_str = '\n'.join(exceptions)
        raise Exception(_i18n("no_files_written", errors=exceptions_str))
    
    return tuple(saved_paths)