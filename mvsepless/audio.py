import os
import subprocess
import numpy as np
import tempfile
from numpy.typing import DTypeLike

ffmpeg_path = "ffmpeg"
ffprobe_path = "ffprobe"

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

allowed_chars = r"1234567890"

def sanitize_output(output):
    return "".join([char for char in output if char in allowed_chars])

def get_sr(path: str, stream: int = 0):
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

def get_channels(path: str, stream: int = 0):
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

def check(path):
    channels = get_channels(path)
    sr = get_sr(path)
    return channels !=0 and sr != 0

def read(path: str, sr: int | None = None, mono: bool = False, dtype: DTypeLike = "float32", multi_channel: bool = False, num_channels: int = 2, stream: int = 0, flatten=False):
    output_format = SAMPLE_FORMATS_DICT.get(dtype)
    if not sr:
        sr = get_sr(path, stream)
    channels = 1 if mono else get_channels(path, stream) if multi_channel else num_channels
    cmd = [ffmpeg_path, "-i", path, "-map", f"0:a:{stream}", "-vn", "-f", output_format, "-ac", str(channels), "-ar", str(sr), "-"]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
    )
    stdout, stderr = process.communicate()
    y = np.frombuffer(stdout, dtype=dtype)
    y = y.reshape((-1, channels)).T if not mono else y.flatten() if flatten else y.reshape((-1, 1)).T
    return y.copy(), sr

def bitrate_to_int(a):
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
        array_index = None
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

def get_duration_from_array(y: np.ndarray, sr: int | None = None):
    len_samples = get_info_array(y)[1]
    if sr is not None:
        return len_samples / sr
    else:
        return len_samples
    
def trim(y: np.ndarray, start: int = 0, end: int = -1):
    channels, samples, array_index, flatten = get_info_array(y)
    end_index = samples - 1
    _end = end if end > 0 and end <= end_index else end_index
    if flatten:
        return y[start:_end]
    elif array_index == 0:
        return y[start:_end, :]
    elif array_index == 1:
        return y[:, start:_end]

def reverse(y: np.ndarray):
    channels, samples, array_index, flatten = get_info_array(y)
    if flatten:
        return np.flip(y)
    else:
        return np.flip(y, axis=array_index)

def write(path, y: np.ndarray, sr: int, bitrate: int | str = 320):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dtype = y.dtype
    channels, samples, array_index, flatten = get_info_array(y)
    if flatten:
        y = y.reshape(-1, 1)
    else:
        if array_index == 1:
            y = y.T
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)
    audio_bytes = y.tobytes()
    sample_format = SAMPLE_FORMATS_DICT.get(str(dtype))
    bitrate = bitrate_to_int(bitrate)
    bitrate_fixed = 64 if bitrate < 64 else 320 if bitrate > 320 else bitrate
    cmd = [ffmpeg_path, "-y", "-f", sample_format, "-ar", str(sr), "-ac", str(channels), "-i", "-", "-ab", f"{bitrate_fixed}k", path]
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

