import os
import json
import subprocess
import numpy as np
from typing import Literal
from collections.abc import Callable
from pathlib import Path
from numpy.typing import DTypeLike

class Audio:
    def __init__(self):
        """
Чтение и запись аудио файла через ffmpeg

Поддерживаемые типы данных: - int16, int32, float32, float64
        """
        self.ffmpeg_path = os.environ.get("MVSEPLESS_FFMPEG", "ffmpeg")
        self.ffprobe_path = os.environ.get("MVSEPLESS_FFPROBE", "ffprobe")
        self.output_formats = ("mp3", "wav", "flac", "ogg", "opus", "m4a", "aac", "ac3", "aiff")
        self.input_formats = ("mp3", "wav", "flac", "ogg", "opus", "m4a", "aac", "ac3", "aiff", "mp4", "mkv", "webm", "avi", "mov", "ts")
        self.supported_dtypes = ("int16", "int32", "float32", "float64")
        self.dtypes_dict = {
            "int16": "s16le",
            "int32": "s32le",
            "float32": "f32le",
            "float64": "f64le",
            np.int16: "s16le",
            np.int32: "s32le",
            np.float32: "f32le",
            np.float64: "f64le",
        }
        self.bitrate_limit = {
            "mp3": {"min": 8, "max": 320},
            "aac": {"min": 8, "max": 512},
            "m4a": {"min": 8, "max": 512},
            "ac3": {"min": 32, "max": 640},
            "ogg": {"min": 64, "max": 500},
            "opus": {"min": 6, "max": 512},
        }
        self.sample_rates = {
            "mp3": {
                "supported": (44100, 48000, 32000, 22050, 24000, 16000, 11025, 12000, 8000)
            },
            "opus": {"supported": (48000, 24000, 16000, 12000, 8000)},
            "m4a": {
                "supported": (
                    96000,
                    88200,
                    64000,
                    48000,
                    44100,
                    32000,
                    24000,
                    22050,
                    16000,
                    12000,
                    11025,
                    8000,
                    7350,
                )
            },
            "aac": {
                "supported": (
                    96000,
                    88200,
                    64000,
                    48000,
                    44100,
                    32000,
                    24000,
                    22050,
                    16000,
                    12000,
                    11025,
                    8000,
                    7350,
                )
            },
            "ac3": {
                "supported": (
                    48000,
                    44100,
                    32000,
                )
            },
            "ogg": {"min": 6, "max": 192000},
            "wav": {"min": 0, "max": float("inf")},
            "aiff": {"min": 0, "max": float("inf")},
            "flac": {"min": 0, "max": 192000},
        }
        self.check_ffmpeg()
        self.check_ffprobe()

    def check_ffmpeg(self):
        """
        Проверяет, установлен ли ffmpeg?
        """
        try:
            ffmpeg_version_output = subprocess.check_output(
                [self.ffmpeg_path, "-version"], text=True
            )
        except FileNotFoundError:
            if "PYTEST_CURRENT_TEST" not in os.environ:
                raise FileNotFoundError("FFMPEG не установлен. Укажите путь к установленному FFMPEG через переменную окружения MVSEPLESS_FFMPEG")

    def check_ffprobe(self):
        """
        Проверяет, установлен ли ffprobe?
        """
        try:
            ffmpeg_version_output = subprocess.check_output(
                [self.ffprobe_path, "-version"], text=True
            )
        except FileNotFoundError:
            if "PYTEST_CURRENT_TEST" not in os.environ:
                raise FileNotFoundError("FFPROBE не установлен. Укажите путь к установленному FFPROBE через переменную окружения MVSEPLESS_FFPROBE")


    def fit_sr(
        self, 
        f: str | Literal["mp3", "wav", "flac", "ogg", "opus", "m4a", "aac", "ac3", "aiff"] = "mp3",
        sr: int = 44100
    ) -> int:
        """
        Исправляет значение частоты дисректизации выходного файла

        Параметры:
            f: Формат вывода
            sr: Частота дискретизации (целое число)
        Возвращает:
            sr: Исправленная частота дискретизации
        """
        format_info = self.sample_rates.get(f.lower())

        if not format_info:
            return None  # Формат не найден

        if "supported" in format_info:
            # Для форматов с конкретным списком
            supported_rates = format_info["supported"]
            if sr in supported_rates:
                return sr

            # Находим ближайшую поддерживаемую частоту
            return min(supported_rates, key=lambda x: abs(x - sr))

        elif "min" in format_info and "max" in format_info:
            # Для форматов с диапазоном - обрезаем до границ
            min_rate = format_info["min"]
            max_rate = format_info["max"]

            if sr < min_rate:
                return min_rate
            elif sr > max_rate:
                return max_rate
            else:
                return sr

        return None
    
    def fit_br(
        self, 
        f: str | Literal["mp3", "wav", "flac", "ogg", "opus", "m4a", "aac", "ac3", "aiff"] = "mp3", 
        br: int = 320
        ) -> int:
        """
        Исправляет значение битрейта выходного файла

        Параметры:
            f: Формат вывода
            br: Битрейт (целое число)
        Возвращает:
            br: Исправленный битрейт
        """
        if f not in self.bitrate_limit:
            raise ValueError(f"Формат {f} не поддерживается")

        limits = self.bitrate_limit[f]

        if br < limits["min"]:
            return limits["min"]
        elif br > limits["max"]:
            return limits["max"]
        else:
            return br
        
    def get_info(
        self,
        i: str | os.PathLike | Callable | None = None,
    ) -> dict[int, dict[int, float]]:
        """
        Получает информацию о аудио потоках из файла напрямую через FFMPEG

        Параметры:
            i: Путь к выходному файлу
        Возвращает:
            audio_info: Словарь с информацией о аудиопотоках вида:

                {Номер потока: 
                    {
                        "sample_rate": Частота дисректизации (является целым числом), 
                        "duration": Длительность аудиопотока (является числом с плавающей точкой)
                    }
                }
        """
        audio_info = {}
        if i:
            if os.path.exists(i):
                cmd = [self.ffprobe_path, "-i", i, "-v", "quiet", "-hide_banner", 
                    "-show_entries", "stream=index,sample_rate,duration", "-select_streams", "a", "-of", "json"]
                
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                stdout, stderr = process.communicate()

                if process.returncode != 0:
                    print(f"STDERR: {stderr.decode('utf-8')}")
                    print(f"STDOUT: {stdout.decode('utf-8')}")

                json_output = json.loads(stdout)
                streams = json_output["streams"]
                if not streams:
                    pass
                    
                else:
                    for a, stream in enumerate(streams):
                        audio_info[a] = {
                            "sample_rate": int(stream["sample_rate"]),
                            "duration": float(stream["duration"])
                        }

                return audio_info
                
            else:
                raise FileExistsError("Указанного файла не существует")
            
        else:
            raise TypeError("Не указан путь к файлу")

    def read(
        self,
        i: str | os.PathLike | Callable | None = None, 
        sr: int | None = None, 
        mono: bool = False,
        dtype: DTypeLike = np.float32,
        s: int = 0
    ) -> tuple[np.ndarray, int]:
        """
        Читает аудио-файл, преобразовывая его в массив с аудио данными напрямую через FFMPEG
        Является заменой soundfile.read() и librosa.load()

        Параметры:
            i: Путь к выходному файлу
            sr: Целевая частота дискретизации (Если не указана, то используется частота дискретизации входного файла)
            mono: Конвертация в моно (по умолчанию отключена)
            dtype: Тип данных (поддерживаются типы: int16, int32, float32, float64; по умолчанию - float32)
            s: Номер аудиопотока (по умолчанию 0)
        Возвращает:
            audio_array: Массив с аудио данными
            sr: Частота дискретизации массива
        """
        if i:
            if os.path.exists(i):
                if i.endswith(tuple([f".{of}" for of in self.input_formats])):
                    audio_info = self.get_info(i=i)
                    list_streams = list(audio_info.keys())
                    if audio_info.get(s, False):
                        print(f"Сейчас выбран поток 0:a:{s}")
                        stream = s
                    else:
                        if len(list_streams) != 0:
                            print(f"Количество потоков: {len(list_streams)}\nБудет выбран первый аудиопоток, как так выбранного потока 0:a:{s} не существует")
                            stream = 0
                        else:
                            raise Exception("В входном файле нет аудио потоков")
                    
                    sample_rate_input = audio_info[stream]["sample_rate"]

                    cmd = [
                        self.ffmpeg_path,
                        "-i", i,
                        "-map", f"0:a:{stream}", "-vn",
                        "-f", self.dtypes_dict[dtype],
                        "-ac", "1" if mono else "2",
                    ]
                    
                    if sr:
                        cmd.extend(["-ar", str(sr)])
                    else:
                        sr = sample_rate_input
                    
                    cmd.append("pipe:1")
                    
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=10**8
                    )
                    
                    try:

                        raw_audio, stderr = process.communicate(timeout=300)
                        
                        if process.returncode != 0:
                            raise RuntimeError(f"FFmpeg error: {stderr.decode()}")
                            
                    except subprocess.TimeoutExpired:
                        process.kill()
                        raise RuntimeError("FFmpeg timeout при чтении файла")
                    
                    audio_array = np.frombuffer(raw_audio, dtype=dtype)
                    
                    channels = 1 if mono else 2
                    audio_array = audio_array.reshape((-1, channels)).T

                    print(f"Частота дискретизации: {sr}")
                    
                    return audio_array, sr
                else:
                    raise TypeError("Указанный файл должен быть видео или аудио")
            else:
                raise FileExistsError("Указанного файла не существует")
                
        else:
            raise TypeError("Не указан путь к файлу")
    
    def write(
        self, 
        o: str | os.PathLike | Callable | None = None, 
        array: np.ndarray = np.array([], dtype=np.float32),
        sr: int = 44100, 
        of: str | Literal["mp3", "wav", "flac", "ogg", "opus", "m4a", "aac", "ac3", "aiff"] | None = None, 
        br: str | int | None = None
    ) -> str:
        """
        Записывает numpy-массив с аудио данными в файл напрямую через ffmpeg.
        Является заменой soundfile.write()

        Параметры:
            o: Путь к выходному файлу
            array: Массив с аудио данными (поддерживаются типы: int16, int32, float32, float64)
            sr: Частота дискретизации массива
            of: Формат вывода (по умолчанию mp3)
            br: Битрейт для кодеков, сжимающих аудио с потерями
        Возвращает:
            o: Путь к выходному файлу
        """
        if isinstance(array, np.ndarray):

            if len(array.shape) == 1:
                array = array.reshape(-1, 1)
            elif len(array.shape) == 2:
                if array.shape[0] == 2:
                    array = array.T
            else:
                raise ValueError("Input array must be 1D or 2D")

            if array.dtype == np.int16:
                input_format = "s16le"
            elif array.dtype == np.int32:
                input_format = "s32le"
            elif array.dtype == np.float32:
                input_format = "f32le"
            elif array.dtype == np.float64:
                input_format = "f64le"
            else:
                if np.issubdtype(array.dtype, np.floating):
                    array = np.clip(array, -1.0, 1.0)
                    array = (array * 32767).astype(np.int16)
                elif array.dtype != np.int16:
                    array = array.astype(np.int16)
                input_format = "s16le"

            if array.shape[1] == 1:
                audio_bytes = array.tobytes()

                channels = 1
                
            elif array.shape[1] == 2:
                audio_bytes = array.tobytes()

                channels = 2
            else:
                raise ValueError("numpy-массив должен содержать 1 или 2 канала")

        else:
            raise ValueError("Вход должен быть numpy-массивом")

        if o:
            output_dir = os.path.dirname(o)
            print(f"Выходная папка: {output_dir}")
            output_base = os.path.basename(o)
            output_name, output_ext = os.path.splitext(output_base)
            if output_dir != "":
                os.makedirs(output_dir, exist_ok=True)
            if output_ext == "":
                if of:
                    o += f".{of}"
                else:
                    o += f".mp3"
            elif output_ext == ".":
                if of:
                    o += f"{of}"
                else:
                    o += f"mp3"
        else:
            raise TypeError("Не указан путь к выходному файлу")
        
        if of:
            if of in self.output_formats:
                output_name, output_ext = os.path.splitext(o)
                if output_ext == f".{of}":
                    pass
                else:
                    o = f"{os.path.join(output_dir, output_name)}.{of}"
            else:
                raise ValueError(f"Неподдерживаемый формат: {of}")
        else:
            of = os.path.splitext(o)[1].strip(".")
            if of in self.output_formats:
                pass
            else:
                raise ValueError(f"Неподдерживаемый формат: {of}")

        if sr:
            if isinstance(sr, int):
                sample_rate_fixed = self.fit_sr(f=of, sr=sr)
            elif isinstance(sr, float):
                sr = int(sr)
                sample_rate_fixed = self.fit_sr(f=of, sr=sr)
            else:
                raise TypeError(f"Частота дискретизации должна быть числом\n\nЗначение: {sr}\nТип: {type(sr)}")
        else:
            raise TypeError("Не указана частота дискретизации")

        bitrate_fixed = "320k"

        if of not in ["wav", "flac", "aiff"]:
            if br:
                if isinstance(br, int):
                    bitrate_fixed = self.fit_br(f=of, br=br)
                elif isinstance(br, float):
                    bitrate_fixed = self.fit_br(f=of, br=int(br))
                elif isinstance(br, str):
                    bitrate_fixed = self.fit_br(f=of, br=int(br.strip("k").strip("K")))
                else:
                    bitrate_fixed = self.fit_br(output_format, 320)
            else:
                bitrate_fixed = self.fit_br(of, 320)

        format_settings = {
            "wav": [
                "-c:a",
                "pcm_f32le",
                "-sample_fmt",
                "flt",
            ],
            "aiff": [
                "-c:a",
                "pcm_f32le",
                "-sample_fmt",
                "flt",
            ],
            "flac": [
                "-c:a",
                "flac",
                "-compression_level",
                "12",
                "-sample_fmt",
                "s32",
            ],
            "mp3": [
                "-c:a",
                "libmp3lame",
                "-b:a",
                f"{bitrate_fixed}k",
            ],
            "ogg": [
                "-c:a",
                "libvorbis",
                "-b:a",
                f"{bitrate_fixed}k",
            ],
            "opus": [
                "-c:a",
                "libopus",
                "-b:a",
                f"{bitrate_fixed}k",
            ],
            "m4a": [
                "-c:a",
                "aac",
                "-b:a",
                f"{bitrate_fixed}k",
            ],
            "aac": [
                "-c:a",
                "aac",
                "-b:a",
                f"{bitrate_fixed}k",
            ],
            "ac3": [
                "-c:a",
                "ac3",
                "-b:a",
                f"{bitrate_fixed}k",
            ],
        }
        print(f"Формат вывода: {of}")

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-f",
            input_format,
            "-ar",
            str(sr),
            "-ac",
            str(channels),
            "-i",
            "pipe:0",
            "-ac",
            str(channels),
        ]

        cmd.extend(["-ar", str(sample_rate_fixed)])
        cmd.extend(format_settings[of])
        print(f"Полный путь к выходному файлу: {os.path.abspath(o)}")
        cmd.append(o)

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout, stderr = process.communicate(input=audio_bytes, timeout=300)
        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError("FFmpeg timeout: операция заняла слишком много времени")

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg завершился с ошибкой (код: {process.returncode})")

        return os.path.abspath(o)