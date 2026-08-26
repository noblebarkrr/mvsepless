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
import secrets
import platform
import numpy as np
import yt_dlp
import subprocess
import tempfile
from urllib.parse import urlparse, urlunparse
from queue import Queue, Empty
import threading

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

def print_current_device(device: str | torch.device):
    if isinstance(device, str):
        device_str = device
    else:
        device_str = str(device)
    print(_i18n("current_device", device=device_str))

def get_gdrive_dir():
    try:
        result = subprocess.run(['/bin/mount'], capture_output=True, text=True)
        for line in result.stdout.strip().split('\n'):
            if 'type fuse.drive' in line:
                parts = line.split(' type ')
                if len(parts) >= 2:
                    source_mount = parts[0]
                    source, mount_point = source_mount.split(' on ')
                    return mount_point
    except:
        pass
    return None

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

class DownloadError(Exception): pass
class FileExistsError(Exception): pass

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

def get_size_folder(folder: str | Path):
    folder_path = Path(folder)
    return sum([file.stat().st_size for file in folder_path.rglob('*') if file.is_file()])

def get_disk_usage(path="/content/drive/MyDrive", user_dir="", user_gdrive_dir="", list_subdirs=[]):
    try:
        usage = shutil.disk_usage(path)
        
        total_gb = size_readable(usage.total)
        used_gb = size_readable(usage.used)
        free_gb = size_readable(usage.free)
        return f"""{_i18n("all_space")}: {total_gb}
{_i18n("used_space")}: {used_gb}
{_i18n("free_space")}: {free_gb}"""
    except Exception as e:
        return ""

def get_size_from_path(path: Path):
    path = Path(path)
    return path.stat().st_size

def define_audio_with_size(basename: bool = False, **kwargs):
    path = kwargs.get("value", None)
    if not path:
        return gr.update(**kwargs)
    file_path = Path(path)
    if "label" in kwargs:
        size_str = _i18n("path_not_exist")
        if file_path.exists():
            size_str = size_readable(get_size_from_path(file_path))
        temp_label = f"[{size_str}] " + (file_path.stem if basename else kwargs["label"])
        kwargs["label"] = temp_label
    return gr.Audio(**kwargs)

def update_audio_with_size(basename: bool = False, **kwargs):
    path = kwargs.get("value", None)
    if not path:
        return gr.update(**kwargs)
    file_path = Path(path)
    if "label" in kwargs:
        size_str = _i18n("path_not_exist")
        if file_path.exists():
            size_str = size_readable(get_size_from_path(file_path))
        temp_label = f"[{size_str}] " + (file_path.stem if basename else kwargs["label"])
        kwargs["label"] = temp_label
    return gr.update(**kwargs)

def define_download_button_with_size(basename: bool = False, **kwargs):
    path = kwargs.get("value", None)
    if not path:
        return gr.update(**kwargs)
    file_path = Path(path)
    if "label" in kwargs:
        size_str = _i18n("path_not_exist")
        if file_path.exists():
            size_str = size_readable(get_size_from_path(file_path))
        temp_label = f"[{size_str}] " + (file_path.stem if basename else kwargs["label"])
        kwargs["label"] = temp_label
    return gr.DownloadButton(**kwargs)

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
    chunk_size: int = 8 * 1024 * 1024,  # Увеличен до 8MB для снижения накладных расходов
    progress_callback: Optional[Callable[[int, int], None]] = None,
    expected_hash: Optional[str] = None,
    hash_algorithm: str = "sha256",
    desc: Optional[str] = None
) -> None:
    """
    Высокопроизводительная загрузка с неблокирующим I/O и стабильным прогрессом в Gradio.
    """
    local_path_ = Path(local_path)
    local_path_.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MVSepless/1.0)",
        "Accept-Encoding": "identity" 
    }

    for attempt in range(retries):
        try:
            initial_pos = 0
            mode = "wb"
            
            if local_path_.exists():
                initial_pos = local_path_.stat().st_size
                headers["Range"] = f"bytes={initial_pos}-"
                mode = "ab"

            with requests.Session() as session:
                response = session.get(url_model, stream=True, timeout=timeout, headers=headers)
                
                if response.status_code == 206:
                    total_size = int(response.headers.get("content-length", 0)) + initial_pos
                elif response.status_code == 200:
                    if initial_pos > 0:
                        mode = "wb"
                        initial_pos = 0
                        del headers["Range"]
                    total_size = int(response.headers.get("content-length", 0))
                else:
                    raise DownloadError(f"HTTP {response.status_code}")

                if initial_pos == total_size and total_size > 0:
                    return

                display_name = desc if desc else local_path_.name
                
                # Очередь для разделения сетевого чтения и записи на диск
                write_queue: Queue = Queue(maxsize=10)
                write_error = None

                def writer_thread():
                    nonlocal write_error
                    try:
                        with open(local_path_, mode, buffering=chunk_size * 2) as f:
                            while True:
                                chunk = write_queue.get()
                                if chunk is None:
                                    break
                                f.write(chunk)
                    except Exception as e:
                        write_error = e

                # Запускаем отдельный поток для записи
                writer = threading.Thread(target=writer_thread, daemon=True)
                writer.start()

                try:
                    with tqdm(
                        total=total_size,
                        initial=initial_pos,
                        desc=display_name,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        disable=False,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                # Сеть читает и сразу кладет в очередь, не блокируясь на диске
                                write_queue.put(chunk)
                                pbar.update(len(chunk))
                                
                                # Мгновенное обновление для Gradio
                                if hasattr(pbar, 'container') and pbar.container is not None:
                                    try:
                                        pbar.refresh()
                                    except:
                                        pass
                                        
                                if progress_callback:
                                    progress_callback(pbar.n, total_size)
                    
                    # Сигнал завершения записи
                    write_queue.put(None)
                    writer.join(timeout=30)
                    
                    if write_error:
                        raise write_error
                        
                except Exception:
                    write_queue.put(None)
                    writer.join(timeout=5)
                    raise
                    
                return
                
        except (requests.RequestException, DownloadError) as e:
            print(_i18n("download_attempt_failed", attempt=attempt+1, retries=retries, error=str(e)))
            if attempt < retries - 1:
                wait_time = min(2 ** attempt, 60)
                time.sleep(wait_time)
            else:
                raise DownloadError(f"{_i18n('download_error', error=str(e))}")

def _download_chunk_worker(url: str, start: int, end: int, temp_path: Path, timeout: int, chunk_size: int):
    """Воркер для загрузки одного куска файла в отдельном потоке."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MVSepless/1.0)",
        "Range": f"bytes={start}-{end}"
    }
    try:
        with requests.Session() as s:
            resp = s.get(url, stream=True, timeout=timeout, headers=headers)
            if resp.status_code != 206:
                raise DownloadError(f"Server does not support ranges or HTTP {resp.status_code}")
            
            with open(temp_path, 'wb', buffering=chunk_size) as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        raise e

def dw_file_parts(
    urls: list[str],
    local_path: Union[str, Path],
    retries: int = 3,
    timeout: int = 300,
    max_workers: int = 4,
    chunk_size: int = 4 * 1024 * 1024
) -> None:
    """
    Параллельная загрузка частей файла с корректным отображением прогресса в Gradio.
    """
    local_path_ = Path(local_path)
    local_path_.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix="mvsepless_dl_"))
    temp_files = [temp_dir / f"part_{i}" for i in range(len(urls))]

    print(f"[{_i18n('status')}] {_i18n('downloading_parts')} ({len(urls)} {_i18n('parts')})...")

    # Очередь для передачи прогресса из рабочих потоков в основной
    progress_queue: Queue = Queue()

    try:
        # Узнаем размеры всех файлов
        sizes = []
        for url in urls:
            try:
                r = requests.head(url, timeout=timeout, allow_redirects=True)
                size = int(r.headers.get('content-length', 0))
                sizes.append(size)
            except:
                sizes.append(0)
        total_size = sum(sizes)

        def worker_with_progress(url, t_path, idx):
            """Воркер скачивает часть и кладёт размер каждого чанка в очередь."""
            try:
                h = {"User-Agent": "Mozilla/5.0 (compatible; MVSepless/1.0)"}
                with requests.Session() as s:
                    r = s.get(url, stream=True, timeout=timeout, headers=h)
                    with open(t_path, 'wb', buffering=chunk_size) as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                progress_queue.put(len(chunk))  # ← в очередь, не в pbar
            except Exception as e:
                progress_queue.put(e)  # Пробрасываем ошибку в основной поток

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(worker_with_progress, u, t, i)
                for i, (u, t) in enumerate(zip(urls, temp_files))
            ]

            # tqdm обновляется ТОЛЬКО в основном потоке — Gradio это видит
            with tqdm(
                total=total_size,
                desc=_i18n("combining_parts"),
                unit="B",
                unit_scale=True
            ) as pbar:
                active_workers = len(futures)
                while active_workers > 0:
                    try:
                        item = progress_queue.get(timeout=0.1)
                        if isinstance(item, Exception):
                            raise item
                        pbar.update(item)
                    except Empty:
                        pass
                    # Считаем завершённые воркеры
                    active_workers = sum(1 for f in futures if not f.done())

                # Дочитываем остаток очереди (на случай race condition)
                while not progress_queue.empty():
                    try:
                        item = progress_queue.get_nowait()
                        if isinstance(item, Exception):
                            raise item
                        pbar.update(item)
                    except Empty:
                        break

        # Склеивание
        print(f"[{_i18n('status')}] {_i18n('combining_parts')}...")
        with open(local_path_, 'wb', buffering=chunk_size) as outfile:
            for t_path in temp_files:
                with open(t_path, 'rb') as infile:
                    while True:
                        data = infile.read(chunk_size)
                        if not data:
                            break
                        outfile.write(data)

        print(f"[{_i18n('status')}] ✓ {_i18n('download_complete')}: {local_path_}")

    finally:
        for t_path in temp_files:
            if t_path.exists():
                t_path.unlink()
        if temp_dir.exists():
            temp_dir.rmdir()

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

def share_gradio_tunnel(server_name: str = "0.0.0.0", server_port: str | int = 7860, share_server_protocol: Literal ["http", "https"] | None = None, share_server_address: str | None = None):
    share_token = secrets.token_urlsafe(32)
    share_server_protocol = share_server_protocol or (
        "http" if share_server_address is not None else "https"
    )
    share_url = gr.networking.setup_tunnel(
        local_host=server_name,
        local_port=server_port,
        share_token=share_token,
        share_server_address=None,
        share_server_tls_certificate=None,
    )
    parsed_url = urlparse(share_url)
    share_url_ = urlunparse(
        (share_server_protocol,) + parsed_url[1:]
    )
    return share_url_