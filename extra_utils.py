import os
from typing import List, Optional, Tuple, Union, Any, Dict, Literal, Callable
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
from datetime import datetime
import requests
from pathlib import Path
import gradio as gr
import gc
from i18n import _i18n
from namer import Namer
import time
from datetime import timezone, timedelta
from audio import get_audio_files_from_list
import psutil
import os
import ctypes
import platform
import numpy as np
import yt_dlp
import hashlib

try:
    import spaces
except ImportError:
    spaces = None

zerogpu_available = False

def hf_spaces_gpu(*args, **kwargs):
    """Декоратор для GPU на HF Spaces с fallback для локального запуска"""
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return lambda f: f

if spaces is not None:
    try:
        if hasattr(spaces, 'GPU'):
            zerogpu_available = True
            print(_i18n("zerogpu=true"))
            hf_spaces_gpu = spaces.GPU
    except:
        pass  # Если что-то пошло не так, оставляем заглушку
        
import torch
tz = timezone(timedelta(hours=3))



class DownloadError(Exception): pass

base_c_params = {
    "input_file": {
        "interactive": True,
        "type": "filepath",
        "file_count": "single"
    },
    "input_files_multi": {
        "interactive": True,
        "type": "filepath",
        "file_count": "multiple"
    },
    "output_audio": {
        "interactive": False,
        "type": "filepath",
        "show_download_button": True
    },
    "base": {
        "interactive": True
    },
    "dropdown_multi": {
        "interactive": True,
        "multiselect": True
    },
    "dropdown": {
        "interactive": True,
        "multiselect": False
    }
}

def get_info():
    pass

def size_readable(size_bytes: int):
    if size_bytes == 0:
        return f"0 {_i18n('bytes')}"
    # Единицы измерения
    units = (_i18n('bytes'), _i18n('kbytes'), _i18n('mbytes'), _i18n('gbytes'), _i18n('tbytes'))
    i = 0
    # Делим на 1024, пока размер > 1024
    while size_bytes >= 1024 and i < len(units) - 1:
        size_bytes /= 1024
        i += 1
    return f"{size_bytes:.2f} {units[i]}"

def define_audio_with_size(basename: bool = False, **kwargs):
    path = kwargs.get("value", None)
    if not path:
        return gr.update(**kwargs)
    file_path = Path(path)
    if "label" in kwargs:
        temp_label = f"[{size_readable(file_path.stat().st_size)}] " + (file_path.stem if basename else kwargs["label"])
        kwargs["label"] = temp_label
    return gr.Audio(**kwargs)

def update_audio_with_size(basename: bool = False, **kwargs):
    path = kwargs.get("value", None)
    if not path:
        return gr.update(**kwargs)
    file_path = Path(path)
    if "label" in kwargs:
        temp_label = f"[{size_readable(file_path.stat().st_size)}] " + (file_path.stem if basename else kwargs["label"])
        kwargs["label"] = temp_label
    return gr.update(**kwargs)


class DownloadError(Exception):
    """Custom exception for download errors"""
    pass

def format_size(size_bytes: int) -> str:
    """
    Format file size with appropriate units using i18n
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string with units
    """
    if size_bytes < 1024:
        return f"{size_bytes} {_i18n('bytes')}"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} {_i18n('kbytes')}"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.2f} {_i18n('mbytes')}"
    elif size_bytes < 1024**4:
        return f"{size_bytes / (1024**3):.2f} {_i18n('gbytes')}"
    else:
        return f"{size_bytes / (1024**4):.2f} {_i18n('tbytes')}"


def dw_file(
    url_model: str,
    local_path: Union[str, Path],
    retries: int = 180,
    timeout: int = 300,
    chunk_size: int = 8192,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> None:
    """
    Download file with resume support and hash verification
    
    Args:
        url_model: File URL
        local_path: Local path for saving
        retries: Number of retry attempts
        timeout: Request timeout in seconds
        chunk_size: Download chunk size in bytes
        resume: Enable resume for partial downloads
        expected_hash: Expected hash value (if None and auto_detect_hash=True, try to get from server)
        hash_algorithm: Hash algorithm to use (md5, sha1, sha256, sha512)
        auto_detect_hash: Try to get hash from server automatically
        verify_after_download: Verify hash after download completion
        progress_callback: Optional callback for progress updates (current, total)
    
    Raises:
        DownloadError: If download fails or hash verification fails
    """
    local_path_ = Path(local_path)
    local_path_.parent.mkdir(parents=True, exist_ok=True)
    

    headers = {}
    
    for attempt in range(retries):
        try:
            with requests.Session() as session:
                session.headers.update({
                    "User-Agent": "Mozilla/5.0 (compatible; MVSepless/1.0)"
                })
                
                response = session.get(
                    url_model, 
                    stream=True, 
                    timeout=timeout, 
                    headers=headers
                )
                
                # Handle response status
                if response.status_code == 200:
                    # Full download (not resumed)
                    total_size = int(response.headers.get("content-length", 0))
                    mode = "wb"
                    initial_progress = 0
                    print(f"[{_i18n('status')}] {_i18n('download_start')} {format_size(total_size)}")
                    
                else:
                    raise DownloadError(f"HTTP {response.status_code}")
                
                # Download with progress bar
                with tqdm(
                    total=total_size,
                    desc=local_path_.name,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    initial=initial_progress,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                ) as progress_bar:
                    
                    with open(local_path_, mode) as f:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                progress_bar.update(len(chunk))
                                if progress_callback:
                                    progress_callback(progress_bar.n, total_size)
                
                print(f"[{_i18n('status')}] ✓ {_i18n('download_complete')}: {local_path_}")
                return
                
        except (requests.RequestException, DownloadError) as e:
            print(_i18n(
                "download_attempt_failed", 
                attempt=attempt + 1, 
                retries=retries, 
                error=str(e)
            ))
            
            if attempt < retries - 1:
                print(_i18n("retrying"))
            else:
                print(_i18n("all_download_attempts_failed"))
                raise DownloadError(f"{_i18n('download_error', error=str(e))}")

def dw_yt_dlp(
    url: str,
    output_dir: str | Path = None,
    cookie: str | Path = None,
    output_format: str = "mp3",
    output_bitrate: str = "320",
    title: str = None,
) -> str:
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
    if not output_dir:
        output_dir = Path(".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = "%(title)s.%(ext)s" if title is None else f"{title}.%(ext)s"
    output_path_template = output_dir / outtmpl

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_path_template),
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

    if cookie:
        cookie_path = Path(cookie)
        if cookie_path.exists():
            ydl_opts["cookiefile"] = cookie

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, process=True, download=True)
            if "_type" in info and info["_type"] == "playlist":
                entry = info["entries"][0]
                filename = ydl.prepare_filename(entry)
            else:
                filename = ydl.prepare_filename(info)

            output_file = Path(filename)
            audio_file = output_file.with_suffix(f".{output_format}")

            return audio_file.as_posix()
        except Exception as e:
            print(_i18n("download_error", error=str(e)))
            return None

def one_element_list_to_value(input_list: list | tuple):
    if input_list:
        return input_list[0]
    else:
        return None

def emergency_ram_clear():
    """
    Экстренная очистка — удаление ВСЕХ больших объектов в процессе.
    Использовать только если nuclear_clear_model недостаточно.
    """
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    
    # Удаляем все numpy массивы и тензоры из globals
    for name in list(globals().keys()):
        try:
            obj = globals()[name]
            if isinstance(obj, (np.ndarray, torch.Tensor)):
                if isinstance(obj, np.ndarray):
                    obj.fill(0)
                del globals()[name]
        except:
            pass
    
    for name in list(locals().keys()):
        try:
            obj = locals()[name]
            if isinstance(obj, (np.ndarray, torch.Tensor)) and name != 'self':
                if isinstance(obj, np.ndarray):
                    obj.fill(0)
                del locals()[name]
        except:
            pass
    
    for _ in range(5):
        gc.collect()
    
    system = platform.system()
    if system == "Linux":
        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except:
            pass
    elif system == "Windows":
        try:
            ctypes.windll.psapi.EmptyWorkingSet(ctypes.c_void_p(-1))
            ctypes.windll.kernel32.SetProcessWorkingSetSize(
                ctypes.c_void_p(-1), ctypes.c_size_t(-1), ctypes.c_size_t(-1)
            )
        except:
            pass
    
    mem_after = process.memory_info().rss
    freed = (mem_before - mem_after) / (1024 * 1024)
    print(f"[EMERGENCY] {_i18n('emeergency_ram')}: {freed:.1f} MB")
    
    return max(0, freed)

def linux_nuclear_ram_clear():
    """Linux-специфичная очистка: malloc_trim + madvise"""
    try:
        libc = ctypes.CDLL('libc.so.6')
        
        # malloc_trim(0) — возврат памяти из heap в ОС
        libc.malloc_trim(0)
        
        # Попытка очистить page cache (требует root, игнорируем ошибки)
        try:
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('1')
        except (PermissionError, IOError):
            pass
        
        # MADV_DONTNEED для больших numpy массивов в процессе
        MADV_DONTNEED = 4
        
        for obj in gc.get_objects():
            try:
                if isinstance(obj, np.ndarray) and obj.size > 100000:
                    # Помечаем страницы как "не нужны"
                    addr = obj.ctypes.data
                    size = obj.size * obj.itemsize
                    libc.madvise(
                        ctypes.c_void_p(addr),
                        ctypes.c_size_t(size),
                        MADV_DONTNEED
                    )
            except:
                pass
                
    except Exception as e:
        pass  # Не критично если не сработало

def windows_nuclear_ram_clear():
    """Windows-специфичная очистка: EmptyWorkingSet + SetProcessWorkingSetSize"""
    try:
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        
        current_process = ctypes.c_void_p(-1)
        
        try:
            psapi.EmptyWorkingSet(current_process)
        except:
            pass
        
        try:
            kernel32.SetProcessWorkingSetSize(
                current_process,
                ctypes.c_size_t(-1),
                ctypes.c_size_t(-1)
            )
        except:
            pass

        try:
            msvcrt = ctypes.cdll.msvcrt
            msvcrt._heapmin()
        except:
            pass

        MEM_RESET = 0x00080000
        PAGE_READWRITE = 0x04
        
        for obj in gc.get_objects():
            try:
                if isinstance(obj, np.ndarray) and obj.size > 100000:
                    addr = obj.ctypes.data
                    size = obj.size * obj.itemsize
                    kernel32.VirtualAlloc(
                        ctypes.c_void_p(addr),
                        ctypes.c_size_t(size),
                        MEM_RESET,
                        PAGE_READWRITE
                    )
            except:
                pass
                
    except Exception as e:
        pass

def nuclear_clear_model():
    """
    Ядерная очистка RAM. Возвращает освобожденные МБ.
    
    Args:
        aggressive: Если True, использует платформенно-специфичные вызовы
    """
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    
    gc.set_threshold(1, 1, 1)
    for generation in [0, 1, 2]:
        gc.collect(generation)
    gc.set_threshold(700, 10, 10)
    
    system = platform.system()
    if system == "Linux":
        linux_nuclear_ram_clear()
    elif system == "Windows":
        windows_nuclear_ram_clear()
    
    gc.collect()
    gc.collect()
    
    mem_after = process.memory_info().rss
    freed_mb = (mem_before - mem_after) / (1024 * 1024)
    
    if freed_mb > 0:
        print(f"[NUCLEAR] {_i18n('freed_ram')}: {freed_mb:.1f} MB")
    
    return max(0, freed_mb)

def extra_clear_torch_cache():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
            try:
                torch._C._cuda_clearCublasWorkspaces()
            except Exception: pass

    if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
        try:
            torch.compiler.reset()
        except Exception: pass
    
    if hasattr(torch._C, "_clear_dynamo_cache"):
        try:
            torch._C._clear_dynamo_cache()
        except Exception: pass

    if hasattr(torch._C, "_clear_cpu_caches"):
        torch._C._clear_cpu_caches()
    
    if hasattr(torch._C, "clear_autocast_cache"):
        torch._C.clear_autocast_cache()

    if hasattr(torch._C, "_jit_clear_class_registry"):
        torch._C._jit_clear_class_registry()
    
    if hasattr(torch._C, "_jit_pass_onnx_clear_scope_records"):
        try:
            torch._C._jit_pass_onnx_clear_scope_records()
        except Exception: pass

class UserDirectory:
    def __init__(self):
        self.user_directory = Path('.')

    def change_dir(self, dir: str):
        self.user_directory = Path(dir)

    def generate(self, name: str):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        generated_directory = self.user_directory / name / timestamp
        generated_directory.mkdir(parents=True, exist_ok=True)
        return generated_directory
    
    def generate_from_dir(self, dir: str):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        generated_directory = Path(dir) / timestamp
        generated_directory.mkdir(parents=True, exist_ok=True)
        return generated_directory

class InputFilesDatabase(UserDirectory):
    def __init__(self):
        super().__init__()
        self.input_dir_base = self.user_directory / "input"
        self.input_dir_base.mkdir(parents=True, exist_ok=True)
        self.input_base_json = self.input_dir_base / "inputs.json"
        self.input_base = []
        self.load()

    def _write_decorator(func):
        def wrapper(self, *args, **kwargs):
            results_ = func(self, *args, **kwargs)
            self.write()
            return results_
        return wrapper

    def _load_decorator(func):
        def wrapper(self, *args, **kwargs):
            self.load()
            results_ = func(self, *args, **kwargs)
            return results_
        return wrapper

    def write(self):
        self.input_base_json.write_text(json.dumps(self.input_base, ensure_ascii=False, indent=4), encoding="utf-8")

    def load(self):
        if self.input_base_json.exists():
            self.input_base = json.loads(self.input_base_json.read_text("utf-8"))
            print(_i18n("input_base_loaded"))

    @_write_decorator
    def upload(self, files, copy=False):
        input_dir = self.generate_from_dir(self.input_dir_base)
        uploaded_input_files = []
        valid_files = get_audio_files_from_list(files, only_files=True)
        for file in valid_files:
            new_file = Namer.iter(input_dir / Path(file).name)
            if copy:
                shutil.copy2(file, new_file)
            else:
                shutil.move(file, new_file)
            uploaded_input_files.append(new_file)
        self.input_base.extend(uploaded_input_files)
        return uploaded_input_files

    @_write_decorator
    def clear(self):
        for path in self.input_base:
            Path(path).unlink(missing_ok=True)
        self.input_base.clear()
        print(_i18n("input_base_cleared"))

    def get_input_list(self):
        return list(reversed(self.input_base))
    
class OutputDir(UserDirectory):
    def __init__(self, dir: str = "output_mvsepless"):
        super().__init__()
        self.output_dir_name = dir

    def gen_output_dir(self):
        return self.generate(self.output_dir_name)