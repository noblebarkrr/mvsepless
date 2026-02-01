import os
import subprocess
import numpy as np
import tempfile
from librosa import resample, stft as stft2, istft as istft2
from scipy.signal import stft, istft
from numpy.typing import DTypeLike

ffmpeg_path = "ffmpeg"
ffprobe_path = "ffprobe"

def average(*ints):
    numbers = len(ints)
    return sum(ints) / numbers

def check_installed():
    ffmpeg_version_output = subprocess.check_output(
        [ffmpeg_path, "-version"], text=True
    )

    ffprobe_version_output = subprocess.check_output(
        [ffprobe_path, "-version"], text=True
    )

SAMPLE_FORMATS_DICT = {
    "int16": "s16le",
    "int32": "s32le",
    "float32": "f32le",
    "float64": "f64le",
    np.int16: "s16le",
    np.int32: "s32le",
    np.float32: "f32le",
    np.float64: "f64le",
}

audio_formats = [
    'aac',       # AAC (обычно в ADTS контейнере)
    'ac3',       # Dolby Digital
    'ac4',       # Dolby AC-4
    'adts',      # ADTS AAC
    'aiff',      # Audio Interchange File Format
    'au',        # Sun AU
    'caf',       # Apple Core Audio Format
    'dts',       # DTS
    'eac3',      # Dolby Digital Plus
    'flac',      # Free Lossless Audio Codec
    'm4a',       # MPEG-4 Audio (обычно AAC)
    'mp3',       # MPEG Audio Layer 3
    'mp2',       # MPEG Audio Layer 2
    'ogg',       # Ogg Vorbis/Opus
    'oga',       # Ogg Audio
    'opus',      # Opus Audio
    'ra',        # RealAudio
    'raw',       # RAW PCM (различные типы)
    'snd',       # Sound
    'voc',       # Creative Voice
    'wav',       # Waveform Audio
    'wma',       # Windows Media Audio
    'wv',        # WavPack
]

video_formats_with_audio = [
    '3gp',       # 3GPP mobile video
    '3g2',       # 3GPP2 mobile video
    'asf',       # Advanced Systems Format
    'avi',       # Audio Video Interleaved
    'flv',       # Flash Video
    'f4v',       # Flash Video
    'm4v',       # MPEG-4 Video
    'mkv',       # Matroska Video
    'mov',       # QuickTime Movie
    'mp4',       # MPEG-4 Part 14
    'mpeg',      # MPEG-1/2
    'mpg',       # MPEG-1/2
    'mts',       # AVCHD
    'mxf',       # Material Exchange Format
    'ogv',       # Ogg Video
    'rm',        # RealMedia
    'rmvb',      # RealMedia Variable Bitrate
    'ts',        # MPEG Transport Stream
    'vob',       # DVD Video Object
    'webm',      # WebM (VP8/VP9 + Vorbis/Opus)
    'wmv',       # Windows Media Video
]

input_formats = video_formats_with_audio + audio_formats

output_formats = [
    "mp3",
    "wav",
    "flac",
    "ogg",
    "opus",
    "m4a",
    "aac",
    "ac3",
    "aiff",
    "wma"
]

input_extensions = [f".{of}" for of in input_formats]

output_extensions = [f".{of}" for of in output_formats]

codec_args = {
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

def get_codec_args(extension: str, prefer_float: bool) -> list | list[str]:
    if extension not in codec_args:
        return []
    return codec_args[extension][prefer_float]

allowed_chars = r"1234567890"

def sanitize_output(output: str) -> str:
    return "".join([char for char in output if char in allowed_chars])

def get_sr(path: str, stream: int = 0) -> int:
    cmd = [ffprobe_path, "-i", path, "-v", "quiet", "-hide_banner", "-show_entries", "stream=sample_rate", "-select_streams", f"a:{stream}", "-of", "compact=p=0:nk=1"]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    sample_rate = stdout.decode('utf-8').strip()
    sample_rate = sanitize_output(sample_rate)
    if sample_rate.isdigit():
        return int(sample_rate)
    else:
        return 0

def get_channels(path: str, stream: int = 0) -> int:
    cmd = [ffprobe_path, "-i", path, "-v", "quiet", "-hide_banner", "-show_entries", "stream=channels", "-select_streams", f"a:{stream}", "-of", "compact=p=0:nk=1"]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    channels = stdout.decode('utf-8').strip()
    channels = sanitize_output(channels)
    if channels.isdigit():
        return int(channels)
    else:
        return 0

def check(path: str) -> bool:
    channels = get_channels(path)
    sr = get_sr(path)
    return channels !=0 and sr != 0

def read(path: str, sr: int | None = None, mono: bool = False, dtype: DTypeLike = "float32", multi_channel: bool = False, num_channels: int = 2, stream: int = 0, flatten=False):
    output_format = SAMPLE_FORMATS_DICT.get(dtype, None)
    if not sr:
        sr = get_sr(path, stream)
    channels = 1 if mono else get_channels(path, stream) if multi_channel else num_channels
    if not output_format:
        output_format = "f32le"
        cmd = [ffmpeg_path, "-i", path, "-map", f"0:a:{stream}", "-vn", "-f", output_format, "-ac", str(channels), "-ar", str(sr), "-"]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
        )
        stdout, stderr = process.communicate()
        y = np.frombuffer(stdout, dtype=np.float32)
        y = convert_to_dtype(y, dtype)
    else:
        cmd = [ffmpeg_path, "-i", path, "-map", f"0:a:{stream}", "-vn", "-f", output_format, "-ac", str(channels), "-ar", str(sr), "-"]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
        )
        stdout, stderr = process.communicate()
        y = np.frombuffer(stdout, dtype=dtype)
    y = y.reshape((-1, channels)).T if not mono else y.flatten() if flatten else y.reshape((-1, 1)).T
    return y.copy(), sr

def multiread(paths: list | tuple, *args, **kwargs) -> tuple[list[np.ndarray], list[int]]:
    readed_files = []
    srs = []
    len_arrays = len(paths)
    for i, path in enumerate(paths, start=1):
        array, sr = read(path, *args, **kwargs)
        readed_files.append(array)
        srs.append(sr)
        print(f"{i}/{len_arrays} читаются...", end="\r")
    print("")
    return readed_files, srs

def bitrate_to_int(a: str | int | float) -> int:
    if isinstance(a, str):
        if a.endswith(("k", "K")):
            numeric_part = a[:-1]
            if numeric_part.isdigit():
                return int(numeric_part)
            else:
                return 320
        else:
            if a.isdigit():
                return int(a)
            else:
                return 320
    elif isinstance(a, (int, float)):
        return int(a)
    else:
        return 320

def get_info_array(y: np.ndarray) -> tuple[int, int, int | None, bool]:
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
    return channels, samples, array_index, flatten

def get_axis_from_array_index(index: int) -> int:
    if index == -1:
        return -1
    elif index == 1:
        return 0
    elif index == 0:
        return 1
    else:
        return -1

def get_duration_from_array(y: np.ndarray, sr: int | None = None) -> float | int:
    len_samples: int = get_info_array(y)[1]
    if sr is not None:
        return len_samples / sr
    else:
        return len_samples

def is_float(y: np.ndarray) -> bool:
    return np.issubdtype(y.dtype, np.floating)

def is_float_dtype(dtype: DTypeLike) -> bool:
    return np.issubdtype(dtype, np.floating)

def float_to_int(y: np.ndarray, dtype: DTypeLike) -> np.ndarray:
    info = np.iinfo(dtype)
    min_val = info.min
    max_val = info.max
    center_val = int(average(min_val, max_val))
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

def int_to_int(y: np.ndarray, dtype: DTypeLike) -> np.ndarray:
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
    info = np.iinfo(y.dtype)
    if info.min == 0:
        y_normalized = (y.astype(np.float64) + -int(average(info.min, info.max))) / info.max
    elif info.min < 0:
        abs_max = max(abs(info.min), abs(info.max))
        y_normalized = y.astype(np.float64) / abs_max
    return y_normalized.astype(dtype)

def float_to_float(y: np.ndarray, dtype: DTypeLike) -> np.ndarray:
    return y.astype(dtype)

def get_center_value_from_dtype(dtype: DTypeLike) -> int:
    if is_float_dtype(dtype):
        return 0
    else:
        info = np.iinfo(dtype)
        return int(average(info.min, info.max))

def convert_to_dtype(y: np.ndarray, dtype: DTypeLike) -> np.ndarray:
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

def dc_offset(y: np.ndarray, offset: float | int) -> np.ndarray:
    orig_dtype = y.dtype
    y = convert_to_dtype(y, np.float64)
    y = y + offset
    return convert_to_dtype(y, orig_dtype)

def gain(y: np.ndarray, gain: float | int) -> np.ndarray:
    orig_dtype = y.dtype
    y = convert_to_dtype(y, np.float64)
    y = y * gain
    return convert_to_dtype(y, orig_dtype)

def normalize(y: np.ndarray, target_peak: float | int = 1.0) -> np.ndarray:
    orig_dtype = y.dtype
    y = convert_to_dtype(y, np.float64)
    current_peak = np.max(np.abs(y))
    if current_peak > 0:
        scaling_factor = target_peak / current_peak
        y = y * scaling_factor
    return convert_to_dtype(y, orig_dtype)

def create_zero_array(samples: int, dtype: DTypeLike):
    return np.array([get_center_value_from_dtype(dtype) for ___ in range(samples)], dtype=dtype)

def split_channels(y: np.ndarray) -> tuple[np.ndarray]:
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
    
def split_mid_side(y: np.ndarray, var: int = 1, sr: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    channels, samples, array_index, flatten = get_info_array(y)
    axis = get_axis_from_array_index(array_index)
    if channels != 2:
        raise Exception("Аудио массив должен быть в стерео (2 канала)")
    orig_dtype = y.dtype
    y = convert_to_dtype(y, np.float64)
    channels_arrays = split_channels(y)
    left_channel = channels_arrays[0]
    right_channel = channels_arrays[1]
    mid_channel_one = (left_channel * 0.5) + (right_channel * 0.5)
    if var == 0:
        side_channel = np.stack([(left_channel + -mid_channel_one), (right_channel + -mid_channel_one)], axis=axis)
        mid_channel = y + -side_channel
    elif var == 1:
        mid_channel = np.stack([mid_channel_one, mid_channel_one], axis=axis)
        side_channel = y + -mid_channel
    elif var == 2:
        same_sign = (stereo_L * stereo_R) > 0
        center_mono = np.where(
            same_sign,
            np.minimum(np.abs(stereo_L), np.abs(stereo_R)) * np.sign(stereo_L),
            0.0
        )
        mid_channel = np.stack([center_mono, center_mono], axis=axis)
        stereo_L = left_channel - center_mono
        stereo_R = right_channel - center_mono
        side_channel = np.stack([stereo_L, stereo_R], axis=axis)
    elif var == 3:
        if sr:
            n_fft = 2048
            hop = 1024
            window = "lanczos"
            f, t, Lf = stft(left_channel, fs=sr, nperseg=n_fft, noverlap=n_fft-hop,
                        window=window, padded=False)
            _, _, Rf = stft(right_channel, fs=sr, nperseg=n_fft, noverlap=n_fft-hop,
                        window=window, padded=False)

            similarity_L = np.real(Lf * np.conj(Rf))
            similarity_R = np.real(Rf * np.conj(Lf))
            mask = (similarity_L > 0) & (similarity_R > 0)
            magL = np.abs(Lf)
            magR = np.abs(Rf)

            magC_L = np.minimum(magL, magR) * mask
            magC_R = np.minimum(magL, magR) * mask

            C_L = magC_L * np.exp(1j * np.angle(Rf))
            C_R = magC_R * np.exp(1j * np.angle(Lf))
            SL = Lf - C_L
            SR = Rf - C_R

            _, center_L = istft(C_L, fs=sr, nperseg=n_fft, noverlap=n_fft-hop, window=window)
            _, center_R = istft(C_R, fs=sr, nperseg=n_fft, noverlap=n_fft-hop, window=window)

            _, base_L = istft(SL, fs=sr, nperseg=n_fft, noverlap=n_fft-hop, window=window)
            _, base_R = istft(SR, fs=sr, nperseg=n_fft, noverlap=n_fft-hop, window=window)

            mid_channel = np.stack([center_L, center_R], axis=axis)
            side_channel = np.stack([base_L, base_R], axis=axis)
        else:
            raise Exception("Не указана частота дискретизации")
    elif var == 4:
        mid_channel = mid_channel_one
        side_channel = left_channel + -right_channel
    return convert_to_dtype(mid_channel, orig_dtype), convert_to_dtype(side_channel, orig_dtype)

def mid_side_to_stereo(y: np.ndarray, z: np.ndarray, index: int = -1, dtype: DTypeLike = np.float32):
    y, z = convert_to_dtype(y, np.float64), convert_to_dtype(z, np.float64)
    mid = multi_channel_array_from_arrays(y, y, index=index, dtype=np.float64)
    side = multi_channel_array_from_arrays(z, -z, index=index, dtype=np.float64)
    return convert_to_dtype(mid + side, dtype)

def mono_to_stereo(y: np.ndarray, index: int, num_channels: int = 2) -> np.ndarray:
    channels, samples, array_index, flatten = get_info_array(y)
    axis = get_axis_from_array_index(array_index)
    if index:
        new_axis = get_axis_from_array_index(index)
    else:
        new_axis = axis
    orig_dtype = y.dtype
    if channels == 1:
        if flatten:
            return np.stack([y for _ in range(num_channels)], axis=new_axis, dtype=orig_dtype)
        else:
            return np.stack([y.flatten() for _ in range(num_channels)], axis=new_axis, dtype=orig_dtype)
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
    axis = get_axis_from_array_index(array_index)
    y = convert_to_dtype(y, np.float64)
    if channels > 1:
        mono = create_zero_array(samples, np.float64)
        for ch in split_channels(y):
            mono = mono + gain(ch, (1 / channels))
        if not to_flatten:
            return y.reshape((-1, 1)).T if axis == 0 else y.reshape((-1, 1))
        else:
            return mono

def multi_channel_array_from_arrays(*arrays: tuple[np.ndarray], index: int = -1, dtype: DTypeLike) -> np.ndarray:
    return np.stack([convert_to_dtype(array, dtype) for array in arrays], axis=get_axis_from_array_index(index), dtype=dtype)

def reshape(y: np.ndarray, shape: tuple = ("channels", "samples")) -> np.ndarray:
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
        raise ValueError(f"Неизвестный формат shape: {shape}")

def easy_resampler(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    channels, samples, array_index, flatten = get_info_array(y)
    orig_dtype = y.dtype
    axis = get_axis_from_array_index(array_index)
    return convert_to_dtype(resample(y, orig_sr=orig_sr, target_sr=target_sr, axis=axis), orig_dtype)

def add_zero_to_end(y: np.ndarray, max_samples: int) -> np.ndarray:
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

def fit_arrays(arrays: tuple[np.ndarray] | list[np.ndarray], srs: tuple[int] | list[int], max_channels: int = 2, 
               min_sr: int = 44100, flatten: bool = False, max_samples: int = -1, extend: bool = True) -> tuple[np.ndarray]:
    if len(arrays) != len(srs):
        raise Exception("Количество массивов должно совпадать с количеством частот дискретизации")
    
    new_arrays = []
    
    arrays_with_srs = list(zip(arrays, srs))
    len_arrays = len(arrays_with_srs)
    
    if max_samples == -1:
        durations = [get_duration_from_array(array) for array, _ in arrays_with_srs]
        max_samples = max(durations) if durations else 0
    
    max_samples = 0

    for i, (array, sr) in enumerate(arrays_with_srs, start=1):
        channels1, samples1, array_index1, _ = get_info_array(array)
        
        a1 = easy_resampler(array, sr, min_sr)
        max_samples = max(max_samples, get_duration_from_array(array))
        
        if flatten:
            a1 = stereo_to_mono(a1, to_flatten=True)
        else:
            if max_channels >= 2:
                a1 = mono_to_stereo(a1, array_index1, max_channels)
            else:
                a1 = stereo_to_mono(a1)
        a1 = reshape(a1, shape=("channels", "samples"))
        new_arrays.append(a1)
        print(f"{i}/{len_arrays}", end="\r")
    print("")
    if extend:
        for i, array_ in enumerate(new_arrays):
            new_arrays[i] = add_zero_to_end(array_, max_samples)
            print(f"{i}/{len_arrays} удлиняется до максимума...", end="\r")
        print("")
    
    return tuple(new_arrays)

def substractor(y: np.ndarray, z: np.ndarray, sr1: int, sr2: int, spectrogram: bool = False) -> tuple[np.ndarray, int]:
    channels1, _, array_index1, flatten1 = get_info_array(y)
    channels2, _, array_index2, flatten2 = get_info_array(z)
    orig_dtype1 = y.dtype
    orig_dtype2 = z.dtype
    y = convert_to_dtype(y, np.float64)
    z = convert_to_dtype(z, np.float64)
    max_channels = max(channels1, channels2)
    min_sr = min(sr1, sr2)
    yz = fit_arrays([y, z], [sr1, sr2], max_channels=max_channels, min_sr=min_sr)
    y, z = yz[0], yz[1]
    if spectrogram:
        print("Используется вычитание из спектрограммы")
        n_fft = 2048
        hop = 1024
        window = "lanczos"
        substracted_specs = []
        substracted_wavs = []
        y_stfts = []
        for ch_1 in split_channels(y):
            f, t, spec = stft(ch_1, fs=min_sr, nperseg=n_fft, noverlap=n_fft-hop, window=window, padded=False)
            y_stfts.append(spec)
        z_stfts = []
        for ch_2 in split_channels(z):
            f, t, spec = stft(ch_2, fs=min_sr, nperseg=n_fft, noverlap=n_fft-hop, window=window, padded=False)
            z_stfts.append(spec)
        y_z_stfts = zip(y_stfts, z_stfts)
        for a1, a2 in y_z_stfts:
            mag1 = np.abs(a1)
            mag2 = np.abs(a2)
            mag_result = np.maximum(mag1 - mag2, 0)
            phase = np.angle(a1)
            substracted_specs.append(mag_result * np.exp(1j * phase))
        for sub_spec in substracted_specs:
            _, wav = istft(sub_spec, fs=min_sr, nperseg=n_fft, noverlap=n_fft-hop, window=window)
            substracted_wavs.append(wav)
        substracted = multi_channel_array_from_arrays(*substracted_wavs, index=1, dtype=orig_dtype1)
        return substracted, min_sr
    else:
        print("Используется вычитание противофазой")
        return convert_to_dtype(y + -z, orig_dtype1), min_sr

def stft_(wave, nfft, hl):
    wave_left = np.asfortranarray(wave[0])
    wave_right = np.asfortranarray(wave[1])
    spec_left = stft2(wave_left, n_fft=nfft, hop_length=hl)
    spec_right = stft2(wave_right, n_fft=nfft, hop_length=hl)
    spec = np.asfortranarray([spec_left, spec_right])
    return spec


def istft_(spec, hl, length):
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])
    wave_left = istft2(spec_left, hop_length=hl, length=length)
    wave_right = istft2(spec_right, hop_length=hl, length=length)
    wave = np.asfortranarray([wave_left, wave_right])
    return wave


def absmax(a, *, axis):
    dims = list(a.shape)
    dims.pop(axis)
    indices = np.ogrid[tuple(slice(0, d) for d in dims)]
    argmax = np.abs(a).argmax(axis=axis)
    indices = list(indices)
    indices.insert(axis % len(a.shape), argmax)
    return a[tuple(indices)]


def absmin(a, *, axis):
    dims = list(a.shape)
    dims.pop(axis)
    indices = np.ogrid[tuple(slice(0, d) for d in dims)]
    argmax = np.abs(a).argmin(axis=axis)
    indices.insert((len(a.shape) + axis) % len(a.shape), argmax)
    return a[tuple(indices)]


def lambda_max(arr, axis=None, key=None, keepdims=False):
    idxs = np.argmax(key(arr), axis)
    if axis is not None:
        idxs = np.expand_dims(idxs, axis)
        result = np.take_along_axis(arr, idxs, axis)
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result
    else:
        return arr.flatten()[idxs]


def lambda_min(arr, axis=None, key=None, keepdims=False):
    idxs = np.argmin(key(arr), axis)
    if axis is not None:
        idxs = np.expand_dims(idxs, axis)
        result = np.take_along_axis(arr, idxs, axis)
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result
    else:
        return arr.flatten()[idxs]

def ensemble(pred_track: list, srs: list, weights: list, algorithm: str, dtype: DTypeLike = np.float64):
    max_sr = int(max(*srs))
    pred_track = [convert_to_dtype(track, np.float64) for track in pred_track]
    pred_track = fit_arrays(pred_track, srs, max_channels=2, min_sr=max_sr)
    pred_track = np.array(pred_track)
    final_length = pred_track.shape[-1]
    mod_track = []
    for i in range(pred_track.shape[0]):
        if algorithm == "avg_wave":
            mod_track.append(pred_track[i] * weights[i])
        elif algorithm in ["median_wave", "min_wave", "max_wave"]:
            mod_track.append(pred_track[i])
        elif algorithm in ["avg_fft", "min_fft", "max_fft", "median_fft"]:
            spec = stft_(pred_track[i], nfft=2048, hl=1024)
            if algorithm in ["avg_fft"]:
                mod_track.append(spec * weights[i])
            else:
                mod_track.append(spec)
    pred_track: np.ndarray = np.array(mod_track)
    if algorithm in ["avg_wave"]:
        pred_track = pred_track.sum(axis=0)
        pred_track /= np.array(weights).sum().T
    elif algorithm in ["median_wave"]:
        pred_track = np.median(pred_track, axis=0)
    elif algorithm in ["min_wave"]:
        pred_track = np.array(pred_track)
        pred_track = lambda_min(pred_track, axis=0, key=np.abs)
    elif algorithm in ["max_wave"]:
        pred_track = np.array(pred_track)
        pred_track = lambda_max(pred_track, axis=0, key=np.abs)
    elif algorithm in ["avg_fft"]:
        pred_track = pred_track.sum(axis=0)
        pred_track /= np.array(weights).sum()
        pred_track = istft_(pred_track, 1024, final_length)
    elif algorithm in ["min_fft"]:
        pred_track = np.array(pred_track)
        pred_track = lambda_min(pred_track, axis=0, key=np.abs)
        pred_track = istft_(pred_track, 1024, final_length)
    elif algorithm in ["max_fft"]:
        pred_track = np.array(pred_track)
        pred_track = absmax(pred_track, axis=0)
        pred_track = istft_(pred_track, 1024, final_length)
    elif algorithm in ["median_fft"]:
        pred_track = np.array(pred_track)
        pred_track = np.median(pred_track, axis=0)
        pred_track = istft_(pred_track, 1024, final_length)
    print("Склеивание завершено")
    return convert_to_dtype(pred_track, dtype), max_sr

def concatenate(arrays: tuple[np.ndarray] | list[np.ndarray], srs: tuple[int] | list[int], dtype=np.float32):
    max_sr = int(max(*srs))
    arrayss = fit_arrays([convert_to_dtype(array, np.float64) for array in arrays], srs, max_channels=2, min_sr=max_sr, extend=False)
    result = np.concatenate(arrayss, axis=1, dtype=np.float64)
    print("Все массивы склеены")
    return convert_to_dtype(result, dtype), max_sr

def trim(y: np.ndarray, start: int = 0, end: int = -1) -> np.ndarray:
    channels, samples, array_index, flatten = get_info_array(y)
    end_index = samples
    _end = end if end > 0 and end <= end_index else end_index
    if flatten:
        return y[start:_end]
    elif array_index == 0:
        return y[start:_end, :]
    elif array_index == 1:
        return y[:, start:_end]

def reverse(y: np.ndarray) -> np.ndarray:
    channels, samples, array_index, flatten = get_info_array(y)
    if flatten:
        return np.flip(y)
    else:
        return np.flip(y, axis=array_index)

def write(path: str, y: np.ndarray, sr: int, bitrate: int | str = 320, prefer_float: bool = False) -> str:
    path = os.path.abspath(path)
    name, ext = os.path.splitext(path)
    dir = os.path.dirname(path)
    os.makedirs(dir, exist_ok=True)
    if not sr:
        raise Exception("Не указана частота дискретизации")
    dtype = y.dtype
    channels, *_ = get_info_array(y)
    y = reshape(y, shape=("samples", "channels"))
    sample_format = SAMPLE_FORMATS_DICT.get(str(dtype), None)
    if not sample_format:
        sample_format = "f32le"
        y = convert_to_dtype(y, np.float32)
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)
    audio_bytes = y.tobytes()
    bitrate = bitrate_to_int(bitrate)
    bitrate_fixed = 64 if bitrate < 64 else 320 if bitrate > 320 else bitrate
    cmd = [ffmpeg_path, "-y", "-f", sample_format, "-ar", str(sr), "-ac", str(channels), "-i", "-", *get_codec_args(ext, prefer_float), "-ab", f"{bitrate_fixed}k", path]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8
    )
    process.stdin.write(audio_bytes)
    process.stdin.close()
    process.wait()
    return path

def multiwrite(arrays: tuple[np.ndarray] | list[np.ndarray], srs: tuple[int] | list[int], paths: tuple[str] | list[int], bitrate: int | str = 320, prefer_float: bool = False, callable_func = None, strict: bool = False):
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
                    print(e)
                    exceptions.append(str(e))
    if not saved_paths:
        exceptions_str = '\n'.join(exceptions)
        raise Exception(f"Ни один из аудио-массивов не был записан без ошибок\nОшибки: {exceptions_str}")
    return tuple(saved_paths)