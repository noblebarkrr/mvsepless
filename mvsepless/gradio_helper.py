import os, gradio as gr, zipfile
from datetime import timezone, timedelta
import platform
import torch
import os
from tqdm import tqdm
import urllib.request
import time
import yt_dlp

tz = timezone(timedelta(hours=3))

cuda_available = torch.cuda.is_available()
mps_available = False #torch.mps.is_available()
device_count = torch.cuda.device_count() if cuda_available else 0
all_ids = list(range(device_count))

script_dir = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.environ.get(
    "MVSEPLESS_DOWNLOAD_DIR", os.path.join(os.getcwd(), "downloaded")
)

def dw_file(url_model: str, local_path: str, retries: int = 180):
    dir_name = os.path.dirname(local_path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)

    class TqdmUpTo(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
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
            print(f"Попытка {attempt + 1}/{retries} не удалась. Ошибка: {e}")
            if attempt < retries - 1:
                print("Повторная попытка...")
                time.sleep(2)
            else:
                print("Все попытки загрузки завершились неудачно")
                raise

def dw_yt_dlp(
    url,
    output_dir=None,
    cookie=None,
    output_format="mp3",
    output_bitrate="320",
    title=None,
):
    outtmpl = "%(title)s.%(ext)s" if title is None else f"{title}.%(ext)s"

    ydl_opts = {
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

            base, _ = os.path.splitext(filename)
            audio_file = base + f".{output_format}"

            return os.path.join(DOWNLOAD_DIR, audio_file)
        except Exception as e:
            return None

def str2bool(value):
    true_values = ['true', '1', 'yes', 'y', 't', 'on']
    false_values = ['false', '0', 'no', 'n', 'f', 'off']
    
    if isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in true_values:
            return True
        elif value_lower in false_values:
            return False
        else:
            raise ValueError(f"Не удалось преобразовать '{value}' в булево значение")
    elif isinstance(value, bool):
        return value
    else:
        return bool(value)

def set_device(*args, prefer_gpu=True):
 
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

    def return_list(self, list, none=False, **kwargs):
        if list:
            return gr.update(choices=list, value=list[0] if not none else None, **kwargs)
        else:
            return gr.update(choices=[], value=None, **kwargs)

    def return_audio(self, label, path):
        return gr.update(label=label, value=path)

    def get_file_size(self, path):
        if path:
            if os.path.exists(path):
                size_bytes = os.path.getsize(path)
            else:
                return "[Указанного файла не существует]"
        else:
            return ""

        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        
        if size_bytes == 0:
            return "[0 B]"
        
        i = 0
        while size_bytes >= 1024 and i < len(units) - 1:
            size_bytes /= 1024
            i += 1
        
        return f"[{size_bytes:.1f} {units[i]}]" if i > 0 else f"[{size_bytes} {units[i]}]"

    def return_audio_with_size(self, *args, **kwargs):
        if "label" in kwargs and "value" in kwargs:
            kwargs["label"] = f"{self.get_file_size(kwargs['value'])} {kwargs['label']}"
        elif not "label" in kwargs and "value" in kwargs:
            kwargs["label"] = f"{self.get_file_size(kwargs['value'])}"
        return gr.update(**kwargs)

    def define_audio_with_size(self, *args, **kwargs):
        if "label" in kwargs and "value" in kwargs:
            kwargs["label"] = f"{self.get_file_size(kwargs['value'])} {kwargs['label']}"
        elif not "label" in kwargs and "value" in kwargs:
            kwargs["label"] = f"{self.get_file_size(kwargs['value'])}"
        return gr.Audio(**kwargs)

    def create_archive_advanced(self, file_list, archive_name="archive.zip"):
        try:
            print("Генерация ZIP-архива с результатами разделения...")
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
                                    f"✓ Добавлен: {stem_path} -> {basename}"
                                )
                            else:
                                print(
                                    f"✗ Файл не найден или не является файлом: {stem_path}"
                                )

                        except Exception as e:
                            print(
                                f"✗ Ошибка при добавлении {stem_path}: {e}"
                            )

                print(f"\nАрхив создан: {archive_name}")
                print(f"Успешно добавлено файлов: {successful_files}")
                return os.path.abspath(archive_name)

        except Exception as e:
            print(f"Ошибка при создании архива: {e}")

    def extract_zip(self, zip_file_path, output_dir=None):
        """
        Распаковывает ZIP-архив с обработкой ошибок
        
        Args:
            zip_file_path: путь к ZIP-архиву
            output_dir: папка для распаковки (по умолчанию - текущая директория)
        """
        
        if output_dir is None:
            output_dir = os.path.splitext(zip_file_path)[0]
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                if zip_ref.testzip() is not None:
                    print("Предупреждение: архив может быть поврежден")
                
                zip_ref.extractall(output_dir)
                
                file_count = len(zip_ref.namelist())
                print(f"Успешно распаковано {file_count} файлов в {output_dir}")                
        except zipfile.BadZipFile:
            print("Ошибка: файл не является ZIP-архивом или поврежден")
            return []
        except PermissionError:
            print("Ошибка: нет прав на запись в указанную директорию")
            return []
        except Exception as e:
            print(f"Неизвестная ошибка: {e}")
            return []
        finally:
            input_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    input_files.append(os.path.join(root, file))
            return input_files
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Скачивание аудио-файлов с интернета"
    )
    
    parser.add_argument("--url", type=str, required=True, help="Ссылка на аудио файл")
    parser.add_argument("--output_dir", type=str, default=None, help="Папка для сохранения")
    parser.add_argument("--cookie", type=str, default=None, help="Путь к файлу куки (необязательно)")
    parser.add_argument("--output_format", type=str, default=output_formats[0], choices=output_formats, help="Формат файла (например, mp3, wav)")
    parser.add_argument("--title", type=str, default=None, help="Название файла (если не указано, возьмется из сети)")
    args = parser.parse_args()

    # Теперь все поля из args соответствуют вызову функции
    dw_yt_dlp(
        args.url,
        args.output_dir,
        args.cookie,
        args.output_format,
        "320",
        args.title,
    )