from tqdm import tqdm
import subprocess
import numpy as np
from pathlib import Path
import librosa
from scipy.signal import ShortTimeFFT, resample, hilbert
from scipy.signal.windows import dpss, hann
from numpy.typing import DTypeLike
import json
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
from i18n import _i18n
try:
    import taglib
    HAS_TAGLIB = True
except:
    HAS_TAGLIB = False

ffmpeg_path = "ffmpeg"
ffprobe_path = "ffprobe"
n_fft = 2048
hop = 1024

def print_saved(path: str | Path):
    if path:
        print(_i18n("saved_file")+": "+Path(path).as_posix())

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

def check_taglib_not_installed():
    if not HAS_TAGLIB:
        print(_i18n("no_has_taglib"))

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

ensemble_types = ("avg_fft", "min_fft", "max_fft", "median_fft")

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

mid_side_vars = (
    "mid/side (orig - side)",
    "mid/side (orig - mid)",
    "sim/dif wave",
    "sim/dif stft",
    "mid/side (left - inverted right)"
)

stereo_split_types = (
    "left/right",
    "mid/side",
    "sim/dif",
)

def sanitize_output(output: str) -> str:
    """
    Очистить вывод от посторонних символов
    
    Args:
        output: Выходная строка
    
    Returns:
        Очищенная строка
    """
    return "".join([char for char in output if char in allowed_chars])


def get_sr(path: str | Path, stream: int = 0) -> int:
    """
    Получить частоту дискретизации аудиофайла
    
    Args:
        path: Путь к файлу
        stream: Номер аудиопотока
    
    Returns:
        Частота дискретизации
    """
    path = Path(path)
    cmd = [ffprobe_path, "-i", path.as_posix(), "-v", "quiet", "-hide_banner", 
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


def get_channels(path: str | Path, stream: int = 0) -> int:
    """
    Получить количество каналов аудиофайла
    
    Args:
        path: Путь к файлу
        stream: Номер аудиопотока
    
    Returns:
        Количество каналов
    """
    path = Path(path)
    cmd = [ffprobe_path, "-i", path.as_posix(), "-v", "quiet", "-hide_banner", 
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

def get_metadata(path: str | Path) -> dict:
    """
    Получить метаданные аудиофайла
    
    Args:
        path: Путь к файлу
    
    Returns:
        Словарь с метаданными
    """
    path = Path(path)
    metadata = {}
    imported_metadata = {}
    if HAS_TAGLIB:
        try:
            with taglib.File(path) as audio_metadata:
                imported_metadata = audio_metadata.tags
        except Exception as e:
            print(e)
            pass

    if imported_metadata:
        for k, v in imported_metadata.items():
            if isinstance(v, str):
                metadata[k] = v
            elif isinstance(v, (list, tuple)):
                metadata[k] = ", ".join(v)
            else:
                pass
            
    return metadata

def check(path: str | Path) -> bool:
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
    path: str | Path, 
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
    path = Path(path)
    output_format = SAMPLE_FORMATS_DICT.get(dtype, None)
    if not sr:
        sr = get_sr(path, stream)
    channels = 1 if mono else (get_channels(path, stream) if multi_channel else num_channels)
    
    if not output_format:
        output_format = "f32le"
        cmd = [ffmpeg_path, "-i", path.as_posix(), "-map", f"0:a:{stream}", "-vn", 
               "-f", output_format, "-ac", str(channels), "-ar", str(sr), "-"]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
        )
        stdout, stderr = process.communicate()
        y = np.frombuffer(stdout, dtype=np.float32)
        y = convert_to_dtype(y, dtype)
    else:
        cmd = [ffmpeg_path, "-i", path.as_posix(), "-map", f"0:a:{stream}", "-vn", 
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
    paths: Union[List[str | Path], Tuple[str | Path, ...]], 
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
    for path in tqdm(paths, desc=_i18n("multi_reading"), unit=_i18n("files")):
        array, sr = read(path, *args, **kwargs)
        readed_files.append(array)
        srs.append(sr)
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
        raise ValueError(_i18n("array_dim_error", axis=y.ndim))
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


def normalizer(y: np.ndarray, target_peak: Union[float, int] = 1.0) -> np.ndarray:
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
    return np.full((samples,), get_center_value_from_dtype(dtype), dtype=dtype)


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
    win = hann(n_fft, sym=False)
    return ShortTimeFFT(win, hop=hop, fs=sr, scale_to='magnitude', phase_shift=None)


def split_mid_side(
    y: np.ndarray, 
    var: int | str = 1, 
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
    
    if var == 0 or var == "mid/side (orig - side)":
        print(_i18n("mid_side_var0"))
        side_channel = np.stack([(left_channel + -mid_channel_one), (right_channel + -mid_channel_one)], axis=axis)
        mid_channel = y + -side_channel
    elif var == 1 or var == "mid/side (orig - mid)":
        print(_i18n("mid_side_var1"))
        mid_channel = np.stack([mid_channel_one, mid_channel_one], axis=axis)
        side_channel = y + -mid_channel
    elif var == 2 or var == "sim/dif wave" or var == "center/wide wave":
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
    elif var == 3 or var == "sim/dif stft" or var == "center/wide stft":
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
    elif var == 4 or var == "mid/side (left - inverted right)":
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
    channels, samples, array_index, flatten = get_info_array(y)
    orig_dtype = y.dtype
    y = convert_to_dtype(y, np.float32)
    
    if channels > 1:
        mono = create_zero_array(samples, np.float64)
        for ch in split_channels(y):
            mono = mono + gain(ch, (1 / channels))
        
        if not to_flatten:
            # Сохраняем ту же ориентацию, что и входной массив, но с 1 каналом
            if array_index == 0:  # вход был (samples, channels)
                return convert_to_dtype(mono.reshape(-1, 1), orig_dtype)
            else:  # array_index == 1 или flatten, вход был (channels, samples)
                return convert_to_dtype(mono.reshape(1, -1), orig_dtype)
        else:
            return convert_to_dtype(mono, orig_dtype)
    else:
        if to_flatten and not flatten:
            return convert_to_dtype(y.flatten(), orig_dtype)
        elif not to_flatten and flatten:
            if array_index == 0:
                return convert_to_dtype(y.reshape(-1, 1), orig_dtype)
            else:
                return convert_to_dtype(y.reshape(1, -1), orig_dtype)
        else:
            return convert_to_dtype(y, orig_dtype)


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
    y = convert_to_dtype(y, np.float32)
    resampled = librosa.resample(
        y,
        orig_sr=orig_sr,
        target_sr=target_sr,
    )
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
        extend: Дополнить до максимальной длины
    
    Returns:
        Кортеж приведенных массивов
    """
    if len(arrays) != len(srs):
        raise Exception(_i18n("arrays_srs_mismatch"))
    
    new_arrays = []
    
    arrays_with_srs = list(zip(arrays, srs))
    len_arrays = len(arrays_with_srs)
    same_sr = len(set(srs)) <= 1

    for (array, sr) in tqdm(arrays_with_srs, desc=_i18n("fitting_progress"), unit=_i18n("arrays")):
        channels1, samples1, array_index1, _c = get_info_array(array)
        if same_sr and sr == min_sr:
            a1 = array
        else:
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

    durations = [get_duration_from_array(array) for array in new_arrays]
    max_samples = max(durations) if durations else 0

    if extend:
        for i, array_ in tqdm(enumerate(new_arrays), desc=_i18n("extending_progress"), unit=_i18n("arrays")):
            new_arrays[i] = add_zero_to_end(array_, max_samples)
    
    return tuple(new_arrays)


def subtractor(
    y: np.ndarray, 
    z: np.ndarray, 
    sr1: int, 
    sr2: int, 
    spectrogram: bool = False,
    max_sr: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Вычесть одно аудио из другого
    
    Args:
        y: Первое аудио
        z: Второе аудио
        sr1: Частота первого
        sr2: Частота второго
        spectrogram: Использовать спектрограмму
        max_sr: Использовать максимальную частоту вместо минимальной
    
    Returns:
        Кортеж (результат, частота дискретизации)
    """
    channels1, _, array_index1, flatten1 = get_info_array(y)
    channels2, _, array_index2, flatten2 = get_info_array(z)
    orig_dtype1 = y.dtype
    y = convert_to_dtype(y, np.float32)
    z = convert_to_dtype(z, np.float32)
    max_channels = max(channels1, channels2)
    min_sr = max(sr1, sr2) if max_sr else min(sr1, sr2)
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
    ensemble_type: str = ensemble_types[0],
    weights: List[float] = [],  
    dtype: np.dtype = np.float32,
    disable_progress: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Создать ансамбль из нескольких предсказаний
    
    Args:
        pred_tracks: Список предсказаний (ожидается форма [channels, samples])
        srs: Список частот дискретизации
        ensemble_type: Алгоритм объединения ('avg_fft', 'min_fft', 'max_fft', 'median_fft')
        weights: Веса для avg_fft
        dtype: Тип данных
        disable_progress: Отключить отображение прогресса
    
    Returns:
        Кортеж (результат, частота дискретизации)
    """
    if ensemble_type == "min_fft":
        result_sr = int(min(srs))
    else:
        result_sr = int(max(srs))

    if ensemble_type == "avg_fft":
        if weights:
            if len(weights) > len(pred_tracks):
                weights = weights[:len(pred_tracks)]
            elif len(weights) < len(pred_tracks):
                weights = weights + [1.0] * (len(pred_tracks) - len(weights))
        else:
            weights = [1.0] * len(pred_tracks)
        total_weight = sum(weights)
    
    pred_tracks = list(fit_arrays(pred_tracks, srs, max_channels=2, min_sr=result_sr))
    
    sft = get_stft_obj(result_sr, n_fft=2048, hop=1024)
    final_length = pred_tracks[0].shape[-1]
    
    if ensemble_type == "avg_fft":
        left_accumulator = None
        right_accumulator = None
    elif ensemble_type in ["min_fft", "max_fft", "median_fft"]:
        left_accumulator = []
        right_accumulator = []
    
    with tqdm(
        total=len(pred_tracks),
        desc=_i18n("ensemble_processing"),
        unit=_i18n("tracks"),
        disable=disable_progress,
        leave=False
    ) as pbar:
        
        for i, track in enumerate(pred_tracks):
            spec_left = sft.stft(convert_to_dtype(track[0], np.float32))
            spec_right = sft.stft(convert_to_dtype(track[1], np.float32))
            
            if ensemble_type == "avg_fft":
                weighted_left = spec_left * weights[i]
                weighted_right = spec_right * weights[i]
                
                if left_accumulator is None:
                    left_accumulator = weighted_left
                    right_accumulator = weighted_right
                else:
                    left_accumulator += weighted_left
                    right_accumulator += weighted_right
                    
            elif ensemble_type in ["min_fft", "max_fft", "median_fft"]:
                left_accumulator.append(spec_left)
                right_accumulator.append(spec_right)
            
            del spec_left, spec_right
            pbar.update(1)
    
    if ensemble_type == "avg_fft":
        left_res_spec = left_accumulator / total_weight
        right_res_spec = right_accumulator / total_weight
        
    elif ensemble_type == "median_fft":

        left_real = np.real(left_accumulator)
        left_imag = np.imag(left_accumulator)
        right_real = np.real(right_accumulator)
        right_imag = np.imag(right_accumulator)
        
        left_res_spec = np.median(left_real, axis=0) + 1j * np.median(left_imag, axis=0)
        right_res_spec = np.median(right_real, axis=0) + 1j * np.median(right_imag, axis=0)
        
    elif ensemble_type == "min_fft":
        left_res_spec = lambda_min(np.array(left_accumulator), axis=0, key=np.abs)
        right_res_spec = lambda_min(np.array(right_accumulator), axis=0, key=np.abs)
        
    elif ensemble_type == "max_fft":
        left_res_spec = absmax(np.array(left_accumulator), axis=0)
        right_res_spec = absmax(np.array(right_accumulator), axis=0)
        
    else:
        raise ValueError(_i18n("unknown_etype", alg=ensemble_type))
    
    left_channel = sft.istft(left_res_spec, k1=final_length)
    right_channel = sft.istft(right_res_spec, k1=final_length)
    
    # Собираем многоканальный массив
    result = multi_channel_array_from_arrays(left_channel, right_channel, index=1, dtype=dtype)
    
    return result, result_sr


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


def mix_arrays(
    arrays: list[np.ndarray], 
    srs: list[int], 
    target_sr: int, 
    index: int = -1, 
    dtype: DTypeLike = np.float32,
    normalize: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Смешать несколько аудио массивов (сложение с нормализацией)
    
    Args:
        arrays: Список массивов для смешивания
        srs: Список частот дискретизации
        target_sr: Целевая частота дискретизации
        index: Индекс оси для выходного массива
        dtype: Тип данных
    
    Returns:
        Кортеж (смешанный массив, частота дискретизации)
    """
    if len(arrays) != len(srs):
        raise Exception(_i18n("arrays_srs_mismatch"))
    
    if len(arrays) == 0:
        raise Exception(_i18n("no_arrays_to_mix"))
    
    # Конвертируем все массивы в float32 для смешивания
    arrays_float = [convert_to_dtype(array, np.float32) for array in arrays]
    
    # Приводим все массивы к единому формату (одинаковая частота, длина, каналы)
    # Определяем максимальное количество каналов среди всех массивов
    max_channels = 2  # По умолчанию стерео
    for array in arrays_float:
        channels, _, _, _ = get_info_array(array)
        if channels > max_channels:
            max_channels = channels
    
    # Подгоняем все массивы к target_sr и max_channels
    fitted_arrays = list(fit_arrays(
        arrays_float, 
        srs, 
        max_channels=max_channels, 
        min_sr=target_sr,
        extend=True  # Дополняем до максимальной длины
    ))
    
    # Получаем форму для смешивания
    mixed = None
    num_arrays = len(fitted_arrays)
    
    for array in fitted_arrays:
        if mixed is None:
            mixed = array.copy()
        else:
            mixed = mixed + array
    
    # Нормализуем, чтобы избежать клиппинга
    # Делим на количество массивов для усреднения
    if normalize:
        mixed = mixed / num_arrays
        
        mixed = normalizer(mixed, 1)
    
    # Преобразуем в целевой тип данных и нужную форму
    result = convert_to_dtype(mixed, dtype)
    
    # Изменяем форму согласно индексу
    if index != -1:
        channels, samples, _, flatten = get_info_array(result)
        if not flatten:
            if index == 0:  # (samples, channels)
                result = result.T
            elif index == 1:  # (channels, samples) - уже в этом формате
                pass
    else:
        # По умолчанию возвращаем в формате (channels, samples)
        result = reshape(result, shape=("channels", "samples"))
    
    print(_i18n("mix_complete", count=num_arrays))
    return result, target_sr

def frequency_blend_phases(phase1, phase2, freq_bins, low_cutoff=500, high_cutoff=5000):
    """Blend two phase arrays with different weights depending on frequency."""
    blended_phase = np.zeros_like(phase1)
    for i, freq in enumerate(freq_bins):
        if freq < low_cutoff:
            blend_factor = 0.1  # Mostly keep original phase for low frequencies
        elif freq > high_cutoff:
            blend_factor = 0.9  # Strongly blend with new phase for high frequencies
        else:
            blend_factor = 0.1 + 0.8 * ((freq - low_cutoff) / (high_cutoff - low_cutoff))
        blended_phase[i, :] = (1 - blend_factor) * phase1[i, :] + blend_factor * phase2[i, :]
    return blended_phase

def phase_corrector(
    target: np.ndarray,
    source: np.ndarray,
    target_sr: int,
    source_sr: int,
    freq_blend_phases: bool = False,
    transfer_magnitude: bool = False, transfer_phase: bool = True, 
    low_cutoff: int = 500, high_cutoff: int = 5000,
) -> Tuple[np.ndarray, int]:

    orig_dtype = target.dtype

    target = convert_to_dtype(target, np.float32)
    source = convert_to_dtype(source, np.float32)

    ch_target = get_info_array(target)[0]
    ch_source = get_info_array(source)[0]
    max_channels = max(ch_target, ch_source, 1)

    target, source = fit_arrays(
        [target, source],
        [target_sr, source_sr],
        max_channels=max_channels,
        min_sr=target_sr,
        flatten=False,
        extend=True
    )

    sft = get_stft_obj(target_sr, n_fft=n_fft, hop=hop)
    final_length = target.shape[-1]

    res_channels = []
    for target_channel, source_channel in zip(split_channels(target), split_channels(source)):
        target_spec = sft.stft(target_channel)
        source_spec = sft.stft(source_channel)
        target_magnitude, target_phase = np.abs(target_spec), np.angle(target_spec)
        source_magnitude, source_phase = np.abs(source_spec), np.angle(source_spec)
        freqs = librosa.fft_frequencies(sr=source_sr, n_fft=source_spec.shape[0] * 2 - 1)
        modified_target_spec = target_spec.copy()

        if transfer_magnitude:
            modified_target_spec = source_magnitude * np.exp(1j * np.angle(modified_target_spec))

        if transfer_phase:
            if freq_blend_phases:
                new_phase = frequency_blend_phases(target_phase, source_phase, freqs, low_cutoff, high_cutoff)
            else:
                new_phase = source_phase
            
            modified_target_spec = np.abs(modified_target_spec) * np.exp(1j * new_phase)

        res_wav = sft.istft(modified_target_spec, k1=final_length)
        res_channels.append(res_wav)
        del target_spec, source_spec, modified_target_spec


    result = multi_channel_array_from_arrays(
        *res_channels, index=1, dtype=orig_dtype
    )
    return result, target_sr


def phase_shift(
    y: np.ndarray,
    degrees: Union[float, int]
) -> np.ndarray:
    """
    Константный фазовый сдвиг всех частот на заданный угол.

    Реализован через преобразование Гильберта (всепроходный фазовый сдвиг),
    поэтому не требует частоты дискретизации и не вносит оконных артефактов.
    DC-составляющая (среднее) сохраняется неизменной.

    Проверочные точки:
        0   град -> сигнал не меняется
        90  град -> -Hilbert(y)  (квадратурный сигнал)
        180 град -> -y           (инверсия полярности)

    Args:
        y: Аудио массив (любая ориентация: 1D / (ch, s) / (s, ch))
        degrees: Угол сдвига фазы в градусах

    Returns:
        Массив той же формы и dtype со сдвинутой фазой
    """

    orig_dtype = y.dtype
    y_f = convert_to_dtype(y, np.float64)  # float64 для точности FFT

    _, _, array_index, _ = get_info_array(y_f)
    axis = array_index  # ось сэмплов (-1 для 1D, 0 или 1 для 2D)

    theta = np.deg2rad(float(degrees))
    factor = np.exp(1j * theta)

    # Сохраняем DC отдельно: умножение аналитического сигнала на exp(j*theta)
    # иначе повернуло бы и постоянную составляющую (при 90° среднее -> 0).
    dc = np.mean(y_f, axis=axis, keepdims=True)
    analytic = hilbert(y_f - dc, axis=axis)
    shifted = np.real(analytic * factor) + dc

    return convert_to_dtype(shifted, orig_dtype)

def _iir_filter(
    y: np.ndarray,
    sr: int,
    hz: Union[float, int],
    btype: str,        # "low" | "high"
    order: int = 8
) -> np.ndarray:
    """
    Внутренний IIR-фильтр Баттерворта с нулевой фазой (sosfiltfilt).

    Граничные частоты обработаны явно, чтобы butter не падал на Wn вне (0, 1):
        lowpass:  hz <= 0      -> тишина (весь спектр срезан)
                  hz >= sr/2   -> passthrough (нечего резать)
        highpass: hz <= 0      -> passthrough
                  hz >= sr/2   -> тишина

    Args:
        y: Аудио массив любой ориентации
        sr: Частота дискретизации
        hz: Частота среза в Гц
        btype: "low" или "high"
        order: Порядок фильтра (крутизна; 8 по умолчанию — хороший баланс)

    Returns:
        Массив той же формы и dtype
    """
    from scipy.signal import butter, sosfiltfilt  # локальный импорт

    orig_dtype = y.dtype
    _, samples, array_index, _ = get_info_array(y)

    # Пустой сигнал — нечего фильтровать.
    if samples == 0:
        return y.copy()

    nyq = float(sr) * 0.5
    wn = float(hz) / nyq  # нормированная частота (доля от Найквиста)

    # --- Граничные случаи (вне допустимого для butter диапазона (0, 1)) ---
    if btype == "low":
        if wn >= 1.0:
            return y.copy()                 # passthrough
        if wn <= 0.0:
            return np.zeros_like(y)         # всё срезано -> тишина
    else:  # "high"
        if wn <= 0.0:
            return y.copy()                 # passthrough
        if wn >= 1.0:
            return np.zeros_like(y)         # всё срезано -> тишина

    # Работаем в float64 для устойчивости IIR.
    y_f = convert_to_dtype(y, np.float64)

    # wn уже строго внутри (0, 1) после проверок выше.
    # Небольшой clamp на всякий случай от float-ошибок на границах.
    wn = float(np.clip(wn, 1e-7, 1.0 - 1e-7))

    sos = butter(order, wn, btype=btype, output="sos")

    # sosfiltfilt по оси сэмплов. На очень коротких сигналах дефолтный padlen
    # может превысить длину -> ValueError; тогда отключаем padding (padlen=0).
    filtered = sosfiltfilt(sos, y_f, axis=array_index, padlen=0)

    return convert_to_dtype(filtered, orig_dtype)

def _brickwall_mask(
    sr: int,
    hz: Union[float, int],
    kind: str  # "lowpass" | "highpass"
) -> np.ndarray:
    """
    Вещественная маска кирпичной стены с косинусным переходом в 1 бин.
    Возвращает массив формы (n_fft // 2 + 1,) в диапазоне [0, 1].
    """
    n_bins = n_fft // 2 + 1
    freqs = np.arange(n_bins, dtype=np.float64) * (float(sr) / n_fft)
    half_bin = (float(sr) / n_fft) * 0.5  # половина ширины бина -> переход = 1 бин
    f_low = float(hz) - half_bin
    f_high = float(hz) + half_bin

    if kind == "lowpass":
        mask = np.ones(n_bins, dtype=np.float32)
        mask[freqs >= f_high] = 0.0
        trans = (freqs >= f_low) & (freqs < f_high)
        if f_high > f_low:
            mask[trans] = 0.5 * (
                1.0 + np.cos(np.pi * (freqs[trans] - f_low) / (f_high - f_low))
            )
    else:  # highpass = инверсия lowpass (DC корректно вырезается)
        mask = np.zeros(n_bins, dtype=np.float32)
        mask[freqs >= f_high] = 1.0
        trans = (freqs >= f_low) & (freqs < f_high)
        if f_high > f_low:
            mask[trans] = 0.5 * (
                1.0 - np.cos(np.pi * (freqs[trans] - f_low) / (f_high - f_low))
            )
    return mask


def lowpass_fft(
    y: np.ndarray,
    sr: int,
    hz: Union[float, int]
) -> np.ndarray:
    """
    Фильтр низких частот (brickwall, линейная фаза) через STFT-маску.
    Оставляет всё ниже `hz`, плавно срезая полосу в 1 бин вокруг среза.

    Args:
        y: Аудио массив (любая ориентация: 1D / (ch, s) / (s, ch))
        sr: Частота дискретизации
        hz: Частота среза в Гц

    Returns:
        Массив той же формы и dtype, пропущенный через ФНЧ
    """
    orig_dtype = y.dtype
    channels, samples, array_index, flatten = get_info_array(y)
    y_f = convert_to_dtype(y, np.float32)

    sft = get_stft_obj(int(sr), n_fft=n_fft, hop=hop)
    mask = _brickwall_mask(sr, hz, "lowpass").reshape(-1, 1)  # (F, 1) для broadcast

    res_channels = []
    for ch in split_channels(y_f):
        spec = sft.stft(ch.astype(np.float32))
        spec_filtered = spec * mask
        res_wav = sft.istft(spec_filtered, k1=ch.shape[-1])
        res_channels.append(res_wav)
        del spec, spec_filtered

    if flatten:
        return convert_to_dtype(res_channels[0], orig_dtype)
    return multi_channel_array_from_arrays(
        *res_channels, index=array_index, dtype=orig_dtype
    )


def highpass_fft(
    y: np.ndarray,
    sr: int,
    hz: Union[float, int]
) -> np.ndarray:
    """
    Фильтр высоких частот (brickwall, линейная фаза) через STFT-маску.
    Оставляет всё выше `hz`, плавно срезая полосу в 1 бин вокруг среза.
    Постоянная составляющая (DC) при hz > 0 корректно удаляется.

    Args:
        y: Аудио массив (любая ориентация: 1D / (ch, s) / (s, ch))
        sr: Частота дискретизации
        hz: Частота среза в Гц

    Returns:
        Массив той же формы и dtype, пропущенный через ФВЧ
    """
    orig_dtype = y.dtype
    channels, samples, array_index, flatten = get_info_array(y)
    y_f = convert_to_dtype(y, np.float32)

    sft = get_stft_obj(int(sr), n_fft=n_fft, hop=hop)
    mask = _brickwall_mask(sr, hz, "highpass").reshape(-1, 1)  # (F, 1) для broadcast

    res_channels = []
    for ch in split_channels(y_f):
        spec = sft.stft(ch.astype(np.float32))
        spec_filtered = spec * mask
        res_wav = sft.istft(spec_filtered, k1=ch.shape[-1])
        res_channels.append(res_wav)
        del spec, spec_filtered

    if flatten:
        return convert_to_dtype(res_channels[0], orig_dtype)
    return multi_channel_array_from_arrays(
        *res_channels, index=array_index, dtype=orig_dtype
    )

def lowpass(
    y: np.ndarray,
    sr: int,
    hz: Union[float, int],
    order: int = 8
) -> np.ndarray:
    """
    Фильтр низких частот (IIR Баттерворта, нулевая фаза через sosfiltfilt).
    Оставляет всё ниже `hz`. Крутизна склона ~ 6 дБ/октаву на каждый порядок
    (order=8 -> ~48 дБ/окт). Фаза в полосе пропускания не искажается
    (двойной проход filtfilt компенсирует задержку).

    Args:
        y: Аудио массив (любая ориентация: 1D / (ch, s) / (s, ch))
        sr: Частота дискретизации
        hz: Частота среза в Гц (0 < hz < sr/2)
        order: Порядок фильтра (по умолчанию 8)

    Returns:
        Массив той же формы и dtype, пропущенный через ФНЧ
    """
    return _iir_filter(y, sr, hz, btype="low", order=order)


def highpass(
    y: np.ndarray,
    sr: int,
    hz: Union[float, int],
    order: int = 8
) -> np.ndarray:
    """
    Фильтр высоких частот (IIR Баттерворта, нулевая фаза через sosfiltfilt).
    Оставляет всё выше `hz`, постоянная составляющая (DC) удаляется.
    Крутизна склона ~ 6 дБ/октаву на каждый порядок.

    Args:
        y: Аудио массив (любая ориентация: 1D / (ch, s) / (s, ch))
        sr: Частота дискретизации
        hz: Частота среза в Гц (0 < hz < sr/2)
        order: Порядок фильтра (по умолчанию 8)

    Returns:
        Массив той же формы и dtype, пропущенный через ФВЧ
    """
    return _iir_filter(y, sr, hz, btype="high", order=order)

def write(
    path: str | Path, 
    y: np.ndarray, 
    sr: int, 
    bitrate: Union[int, str] = 320, 
    prefer_float: bool = False,
    metadata: dict = {}
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
        Путь к сохраненному файлу (Posix-вариант)
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not sr:
        raise Exception(_i18n("sr_required"))
    
    sr: int = int(sr)

    dtype = y.dtype
    channels, *_ = get_info_array(y)
    y = reshape(y, shape=("samples", "channels"))
    
    sample_format = SAMPLE_FORMATS_DICT.get(str(dtype), None)
    if not sample_format:
        sample_format = "f32le"
        y = convert_to_dtype(y, np.float32)
    
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)
    
    bitrate_val = bitrate_to_int(bitrate)
    if output_path.suffix == ".ogg":
        max_bitrate = get_ogg_bitrate(sr, channels)
        if bitrate_val > max_bitrate:
            bitrate_val = max_bitrate
    elif output_path.suffix == ".opus":
        max_bitrate = 256 * channels
        if bitrate_val > max_bitrate:
            bitrate_val = max_bitrate
    
    bitrate_fixed = 32 if bitrate_val < 32 else 320 if bitrate_val > 320 else bitrate_val

    output_path_str = output_path.as_posix()

    cmd = [ffmpeg_path, "-y", "-f", sample_format, "-ar", str(sr), "-ac", str(channels), 
           "-i", "-", *get_codec_args(output_path.suffix, prefer_float), "-ab", f"{bitrate_fixed}k", output_path_str]

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
        
        print_saved(output_path_str)

        if HAS_TAGLIB:
            if metadata:
                if isinstance(metadata, dict):
                    try:
                        with taglib.File(output_path_str, save_on_exit=True) as audio_metadata:
                            audio_metadata.tags = {k: [v] if isinstance(v, str) else v if isinstance(v, list) else [*v] if isinstance(v, tuple) else "" for k, v in metadata.items()}
                    except Exception as e1:
                        print(e1)

    except Exception as e:
        print(_i18n("write_critical_error", error=str(e)))
        process.kill()
        raise e

    return output_path_str


def multiwrite(
    arrays: Union[Tuple[np.ndarray, ...], List[np.ndarray]], 
    srs: Union[Tuple[int, ...], List[int]], 
    paths: Union[Tuple[str | Path, ...], List[str | Path]], 
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
        for array, sr, path in tqdm(save_arrays, desc=_i18n("multi_writing"), unit=_i18n("arrays")):
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

def get_audio_files_from_list(input_paths: Union[str | Path, List[str | Path]], only_files: bool = False) -> List[str]:
    """
    Получить список аудиофайлов из переданных путей
    
    Args:
        input_paths: Путь к файлу или директории или список путей
        only_files: Только файлы (не рекурсивно)
    
    Returns:
        Список путей к аудиофайлам
    """
    input_list: List[str] = []
    
    if isinstance(input_paths, (str, Path)):
        input_paths = [input_paths]
    
    for p_str in input_paths:
        p = Path(p_str)
        
        if p.is_dir():
            if not only_files:
                for file in p.rglob('*'):
                    if file.is_file() and check(file):
                        input_list.append(file.as_posix())
        elif p.is_file():
            if check(p):
                input_list.append(p.as_posix())

    return input_list
