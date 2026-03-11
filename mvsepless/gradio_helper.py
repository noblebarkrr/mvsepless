import os
import gradio as gr
import zipfile
from datetime import timezone, timedelta
import platform
import torch
from tqdm import tqdm
import urllib.request
import time
import yt_dlp
from typing import List, Optional, Tuple, Union, Any, Dict
from i18n import _i18n

tz = timezone(timedelta(hours=3))

cuda_available: bool = torch.cuda.is_available()
mps_available: bool = False  # torch.mps.is_available()
device_count: int = torch.cuda.device_count() if cuda_available else 0
all_ids: List[int] = list(range(device_count))

script_dir: str = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR: str = os.environ.get(
    "MVSEPLESS_DOWNLOAD_DIR", os.path.join(os.getcwd(), "downloaded")
)


def dw_file(url_model: str, local_path: str, retries: int = 180) -> None:
    """
    Скачать файл с поддержкой повторных попыток
    
    Args:
        url_model: URL файла
        local_path: Локальный путь для сохранения
        retries: Количество попыток
    """
    dir_name = os.path.dirname(local_path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)

    class TqdmUpTo(tqdm):
        def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None) -> None:
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    for attempt in range(retries):
        try:
            with TqdmUpTo(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc=os.path.basename(local_path),
            ) as t:
                urllib.request.urlretrieve(
                    url_model, local_path, reporthook=t.update_to
                )
            break
        except Exception as e:
            print(_i18n("download_attempt_failed", attempt=attempt + 1, retries=retries, error=str(e)))
            if attempt < retries - 1:
                print(_i18n("retrying"))
                time.sleep(2)
            else:
                print(_i18n("all_download_attempts_failed"))
                raise


def dw_yt_dlp(
    url: str,
    output_dir: Optional[str] = None,
    cookie: Optional[str] = None,
    output_format: str = "mp3",
    output_bitrate: str = "320",
    title: Optional[str] = None,
) -> Optional[str]:
    """
    Скачать аудио с YouTube с помощью yt-dlp
    
    Args:
        url: URL видео
        output_dir: Директория для сохранения
        cookie: Путь к файлу с cookies
        output_format: Формат выходного файла
        output_bitrate: Битрейт
        title: Название файла
    
    Returns:
        Путь к скачанному файлу или None
    """
    outtmpl = "%(title)s.%(ext)s" if title is None else f"{title}.%(ext)s"

    ydl_opts: Dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(
            DOWNLOAD_DIR if not output_dir else output_dir, outtmpl
        ),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": output_format,
                "preferredquality": output_bitrate,
            }
        ],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    if cookie and os.path.exists(cookie):
        ydl_opts["cookiefile"] = cookie

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            if "_type" in info and info["_type"] == "playlist":
                entry = info["entries"][0]
                filename = ydl.prepare_filename(entry)
            else:
                filename = ydl.prepare_filename(info)

            base, _c = os.path.splitext(filename)
            audio_file = base + f".{output_format}"

            return os.path.join(DOWNLOAD_DIR, audio_file)
        except Exception as e:
            print(_i18n("download_error", error=str(e)))
            return None


def str2bool(value: Union[str, bool, int, None]) -> bool:
    """
    Преобразовать строку в булево значение
    
    Args:
        value: Входное значение
    
    Returns:
        Булево значение
    """
    true_values = ['true', '1', 'yes', 'y', 't', 'on']
    false_values = ['false', '0', 'no', 'n', 'f', 'off']
    
    if isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in true_values:
            return True
        elif value_lower in false_values:
            return False
        else:
            raise ValueError(_i18n("str2bool_error", value=value))
    elif isinstance(value, bool):
        return value
    else:
        return bool(value)


def set_device(*args: Any, prefer_gpu: bool = True) -> str:
    """
    Установить устройство для вычислений
    
    Args:
        *args: Аргументы (могут содержать ID устройств)
        prefer_gpu: Предпочитать GPU
    
    Returns:
        Строка с указанием устройства
    """
    prefer_cuda_flag = prefer_gpu
    
    if args:
        if len(args) == 1 and isinstance(args[0], bool):
            prefer_cuda_flag = args[0]
            ids = None
        else:
            ids = []
            for arg in args:
                if isinstance(arg, list):
                    ids.extend(arg)
                elif isinstance(arg, int):
                    ids.append(arg)
                elif isinstance(arg, tuple):
                    ids.extend(list(arg))
            
            ids = sorted(set(ids))
            prefer_cuda_flag = prefer_gpu if ids else prefer_cuda_flag
    else:
        ids = None
    
    if ids is not None:
        if cuda_available and prefer_cuda_flag:
            valid_ids = [i for i in ids if i < device_count]
            if valid_ids:
                if len(valid_ids) == 1:
                    return f"cuda:{valid_ids[0]}"
                else:
                    return f"cuda:{','.join(map(str, valid_ids))}"
            else:
                return "cuda:0"
        elif mps_available and prefer_cuda_flag:
            return "mps"
        else:
            return "cpu"
    else:
        if cuda_available and prefer_cuda_flag:
            if device_count == 1:
                return "cuda:0"
            elif device_count > 1:
                return f"cuda:{','.join(map(str, all_ids))}"
            else:
                return "cpu"
        elif mps_available and prefer_cuda_flag:
            return "mps"
        else:
            return "cpu"


def easy_check_is_colab() -> bool:
    """
    Проверить, выполняется ли код в Google Colab
    
    Returns:
        True если в Colab
    """
    if platform.machine() == "x86_64" and "Linux" in platform.platform():
        try:
            import google.colab
            module_path: str = google.colab.__file__
            if module_path.startswith("/usr/local/lib/python") and module_path.endswith("/dist-packages/google/colab/__init__.py"):
                return True
            else:
                return False
        except ImportError:
            return False
    else:
        return False


class GradioHelper:
    """Вспомогательный класс для Gradio интерфейса"""
    
    def return_list(self, lst: List[Any], none: bool = False, **kwargs) -> gr.update:
        """
        Вернуть обновление для списка
        
        Args:
            lst: Список значений
            none: Добавить пустое значение
            **kwargs: Дополнительные аргументы
        
        Returns:
            Обновление Gradio
        """
        if lst:
            return gr.update(choices=lst, value=lst[0] if not none else None, **kwargs)
        else:
            return gr.update(choices=[], value=None, **kwargs)

    def return_audio(self, label: str, path: str) -> gr.update:
        """
        Вернуть обновление для аудио
        
        Args:
            label: Метка
            path: Путь к файлу
        
        Returns:
            Обновление Gradio
        """
        return gr.update(label=label, value=path)

    def get_file_size(self, path: Optional[str]) -> str:
        """
        Получить размер файла в человекочитаемом формате
        
        Args:
            path: Путь к файлу
        
        Returns:
            Строка с размером
        """
        if path:
            if os.path.exists(path):
                size_bytes = os.path.getsize(path)
            else:
                return _i18n("file_not_exists")
        else:
            return ""

        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        
        if size_bytes == 0:
            return "[0 B]"
        
        i = 0
        size_float = float(size_bytes)
        while size_float >= 1024 and i < len(units) - 1:
            size_float /= 1024
            i += 1
        
        return f"[{size_float:.1f} {units[i]}]" if i > 0 else f"[{int(size_float)} {units[i]}]"

    def return_audio_with_size(self, *args: Any, **kwargs) -> gr.update:
        """
        Вернуть аудио с размером в метке
        
        Args:
            *args: Позиционные аргументы
            **kwargs: Именованные аргументы
        
        Returns:
            Обновление Gradio
        """
        if "label" in kwargs and "value" in kwargs:
            kwargs["label"] = f"{self.get_file_size(kwargs['value'])} {kwargs['label']}"
        elif "label" not in kwargs and "value" in kwargs:
            kwargs["label"] = f"{self.get_file_size(kwargs['value'])}"
        return gr.update(**kwargs)

    def define_audio_with_size(self, *args: Any, **kwargs) -> gr.Audio:
        """
        Создать аудио компонент с размером в метке
        
        Args:
            *args: Позиционные аргументы
            **kwargs: Именованные аргументы
        
        Returns:
            Компонент Audio
        """
        if "label" in kwargs and "value" in kwargs:
            kwargs["label"] = f"{self.get_file_size(kwargs['value'])} {kwargs['label']}"
        elif "label" not in kwargs and "value" in kwargs:
            kwargs["label"] = f"{self.get_file_size(kwargs['value'])}"
        return gr.Audio(**kwargs)

    def create_archive_advanced(self, file_list: List[Tuple[str, List[Tuple[str, str]]]], archive_name: str = "archive.zip") -> str:
        """
        Создать ZIP архив из списка файлов
        
        Args:
            file_list: Список файлов для архивации
            archive_name: Имя архива
        
        Returns:
            Путь к созданному архиву
        """
        try:
            print(_i18n("creating_zip_archive"))
            with zipfile.ZipFile(
                archive_name, "w", zipfile.ZIP_DEFLATED
            ) as zipf:
                successful_files = 0

                for basename, stems in file_list:
                    for stem_name, stem_path in stems:
                        try:
                            if os.path.exists(stem_path) and os.path.isfile(
                                stem_path
                            ):
                                basename_ = os.path.basename(stem_path)
                                zipf.write(stem_path, basename_)
                                successful_files += 1
                                print(
                                    _i18n("file_added_to_zip", path=stem_path, name=basename)
                                )
                            else:
                                print(
                                    _i18n("file_not_found_for_zip", path=stem_path)
                                )

                        except Exception as e:
                            print(
                                _i18n("error_adding_to_zip", path=stem_path, error=str(e))
                            )

                print(_i18n("zip_created", path=archive_name))
                print(_i18n("files_added_count", count=successful_files))
                return os.path.abspath(archive_name)

        except Exception as e:
            print(_i18n("zip_creation_error", error=str(e)))
            return ""

    def extract_zip(self, zip_file_path: str, output_dir: Optional[str] = None) -> List[str]:
        """
        Распаковать ZIP архив
        
        Args:
            zip_file_path: Путь к ZIP архиву
            output_dir: Директория для распаковки
        
        Returns:
            Список распакованных файлов
        """
        if output_dir is None:
            output_dir = os.path.splitext(zip_file_path)[0]
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                if zip_ref.testzip() is not None:
                    print(_i18n("zip_corrupted_warning"))
                
                zip_ref.extractall(output_dir)
                
                file_count = len(zip_ref.namelist())
                print(_i18n("zip_extracted", count=file_count, dir=output_dir))
                
        except zipfile.BadZipFile:
            print(_i18n("zip_bad_file"))
            return []
        except PermissionError:
            print(_i18n("zip_permission_error"))
            return []
        except Exception as e:
            print(_i18n("zip_unknown_error", error=str(e)))
            return []
        finally:
            input_files: List[str] = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    input_files.append(os.path.join(root, file))
            return input_files


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=_i18n("download_audio_cli_description")
    )
    
    parser.add_argument("--url", type=str, required=True, help=_i18n("url_help"))
    parser.add_argument("--output_dir", type=str, default=None, help=_i18n("output_dir_help"))
    parser.add_argument("--cookie", type=str, default=None, help=_i18n("cookie_help"))
    parser.add_argument("--output_format", type=str, default="mp3", choices=["mp3", "wav", "flac", "ogg", "opus", "m4a", "aac"], help=_i18n("output_format_help"))
    parser.add_argument("--title", type=str, default=None, help=_i18n("title_help"))
    args = parser.parse_args()

    from audio import output_formats
    dw_yt_dlp(
        args.url,
        args.output_dir,
        args.cookie,
        args.output_format,
        "320",
        args.title,
    )