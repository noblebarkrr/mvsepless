import os
import subprocess
import argparse
import re
import sys
from typing import List, Optional, Tuple, Union
from i18n import _i18n


def get_latest_version(package_name: str, index_url: Optional[str] = None) -> Optional[str]:
    """
    Получает последнюю версию пакета из вывода pip index versions
    
    Args:
        package_name: Имя пакета
        index_url: URL индекса пакетов
    
    Returns:
        Последняя версия пакета или None
    """
    cmd = [sys.executable, "-m", "pip", "index", "versions", package_name]
    if index_url:
        cmd.extend(["--index-url", index_url])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # Не вызываем исключение при ошибке
        )
        
        if result.returncode != 0:
            print(_i18n("pip_index_warning", code=result.returncode))
            print(f"stderr: {result.stderr}")
            return None
            
    except Exception as e:
        print(_i18n("pip_index_error", error=str(e)))
        return None
    
    def parse_version_from_output(pip_output: str) -> Optional[str]:
        """
        Парсит версию из вывода pip
        
        Args:
            pip_output: Вывод pip
        
        Returns:
            Версия пакета или None
        """
        if not pip_output:
            return None
            
        lines = pip_output.split('\n')
        
        # Способ 1: Парсим первую строку
        if lines and lines[0].strip():
            first_line = lines[0].strip()
            
            # Версия в скобках (приоритетный способ)
            match = re.search(r'\(([^)]+)\)', first_line)
            if match:
                version = match.group(1).strip()
                return version
            
            # Версия после пробела
            match = re.search(r'\S+\s+([^\s]+)', first_line)
            if match:
                version = match.group(1).strip()
                # Проверяем, что это похоже на версию (содержит цифры)
                if re.search(r'\d', version):
                    return version
        
        # Способ 2: Ищем "Available versions:" и берем первую версию
        for i, line in enumerate(lines):
            if 'Available versions:' in line:
                # Проверяем следующие несколько строк на наличие версий
                for j in range(1, 4):  # Проверяем следующие 3 строки
                    if i + j < len(lines):
                        versions_line = lines[i + j].strip()
                        if versions_line:
                            # Разделяем по запятой и берем первую версию
                            versions = [v.strip() for v in versions_line.split(',') if v.strip()]
                            if versions:
                                return versions[0]
                break
        
        return None
    
    latest_version = parse_version_from_output(result.stdout)
    
    print(_i18n("version_retrieved", package=package_name, version=latest_version or _i18n("unknown")))
    
    return latest_version


def fno_compitable(index_url: Optional[str] = None) -> bool:
    """
    Проверяет совместимость с FNO (Fourier Neural Operator)
    
    Args:
        index_url: URL индекса пакетов
    
    Returns:
        True если совместимо
    """
    is_torch_2 = False
    fno_c = False
    latest_version_torch = get_latest_version("torch", index_url)
    
    if not latest_version_torch:
        print(_i18n("torch_version_not_found"))
        return False
        
    lvt = latest_version_torch.split(".")
    lvt = [int(n_) for n_ in lvt if n_.isdigit()]
    
    for i, num in enumerate(lvt, start=1):
        if i == 1:
            if num == 2:
                is_torch_2 = True
        elif i == 2:
            if num >= 4 and is_torch_2:
                fno_c = True
                
    return fno_c


def is_nvidia_gpu_present() -> bool:
    """
    Проверяет наличие NVIDIA GPU в системе
    
    Returns:
        True если GPU обнаружен
    """
    try:
        # Пытаемся выполнить команду nvidia-smi
        result = subprocess.run(
            ['nvidia-smi'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False  # Не поднимаем исключение при ошибке
        )
        
        # Если код возврата 0 — команда выполнилась успешно
        if result.returncode == 0:
            print(_i18n("nvidia_gpu_detected"))
            return True
        else:
            print(_i18n("nvidia_smi_error"))
            return False

    except FileNotFoundError:
        # Команда nvidia-smi не найдена в системе
        print(_i18n("nvidia_smi_not_found"))
        return False
    except Exception as e:
        print(_i18n("nvidia_check_error", error=str(e)))
        return False


cuda_available: bool = is_nvidia_gpu_present()


def install_uv() -> None:
    """Устанавливает uv - быстрый установщик пакетов Python"""
    print(_i18n("installing_uv"))
    result = subprocess.run([sys.executable, "-m", "pip", "install", "uv"])
    if result.returncode == 0:
        print(_i18n("uv_installed"))
    else:
        print(_i18n("uv_install_error"))


def install_requirements(requirements: List[str], force: bool = False, index_url: Optional[str] = None) -> None:
    """
    Устанавливает зависимости
    
    Args:
        requirements: Список зависимостей
        force: Принудительная установка
        index_url: URL индекса пакетов
    """
    if not requirements:
        return
        
    cmd = [sys.executable, "-m", "uv", "pip", "install", "--no-cache-dir", "-qq"]
    
    if force:
        cmd.append("--upgrade")
        cmd.append("--force-reinstall")
        
    if index_url:
        cmd.extend(["--index-url", index_url])
        
    for pkg in requirements:
        cmd.append(pkg)
        
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(_i18n("requirements_install_error", count=len(requirements)))


# Списки зависимостей
torch_requirements: List[str] = [
    "torch",
    "torchvision",
    "torchaudio",
    "torchcrepe",
]

universal_requirements: List[str] = [
    "numpy==2.0.2",
    "pandas",
    "scipy",
    "librosa",
    "samplerate==0.1.0",
    "matplotlib",
    "tqdm",
    "einops",
    "soundfile",
    "pydub",
    "webrtcvad",
    "audiomentations",
    "pedalboard",
    "ml_collections",
    "wandb",
    "bitsandbytes",
    "tokenizers",
    "huggingface-hub",
    "transformers",
    "diffq>=0.2.1",
    "julius>=0.2.3",
    "openunmix",
    "asteroid>=0.6.0",
    "pyloudnorm",
    "rotary_embedding_torch",
    "gradio<6.0",
    "omegaconf",
    "beartype",
    "spafe",
    "torch_audiomentations",
    "onnx>=1.17",
    "onnx2torch>=0.3.0",
    "onnxruntime-gpu>=1.17" if cuda_available else "onnxruntime>=1.17",
    "ml_dtypes",
    "resampy",
    "yt_dlp",
    "pyngrok",
    "praat-parselmouth",
    "faiss-cpu==1.11",
    "local-attention",
    "tenacity",
    "pyworld",
    "gdown"
]

torch_old_requirements: List[str] = [
    "torch==1.13.1",
    "torchvision==0.14.1",
    "torchaudio==0.13.1",
    "torchcrepe==0.0.24",
]

old_requirements: List[str] = [
    "numpy==1.26.4",
    "pandas==2.3.3",
    "scipy==1.15.3",
    "librosa==0.11.0",
    "samplerate==0.1.0",
    "matplotlib==3.10.8",
    "tqdm==4.67.1",
    "einops==0.8.1",
    "soundfile==0.13.1",
    "pydub==0.25.1",
    "webrtcvad==2.0.10",
    "audiomentations==0.43.1",
    "pedalboard==0.8.2",
    "ml_collections==1.1.0",
    "wandb==0.24.0",
    "bitsandbytes==0.45.0",
    "tokenizers==0.15.2",
    "huggingface-hub==0.34.2",
    "transformers==4.39.3",
    "diffq>=0.2.1",
    "julius>=0.2.3",
    "openunmix",
    "asteroid==0.6.0",
    "pyloudnorm",
    "rotary_embedding_torch==0.3.6",
    "gradio<6.0.0",
    "omegaconf==2.3.0",
    "beartype==0.22.9",
    "spafe==0.3.3",
    "torch_audiomentations==0.12.0",
    "onnx>=1.17",
    "onnx2torch>=0.3.0",
    "onnxruntime-gpu>=1.17" if cuda_available else "onnxruntime>=1.17",
    "ml_dtypes==0.5.4",
    "resampy==0.4.3",
    "yt_dlp",
    "pyngrok",
    "praat-parselmouth==0.4.7",
    "faiss-cpu==1.7.2",
    "local-attention==1.10.0",
    "tenacity==9.1.2",
    "pyworld==0.3.5",
    "gdown"
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=_i18n("installer_description"))
    parser.add_argument("--old", action="store_true", help=_i18n("old_deps_help"))
    parser.add_argument("--force", action="store_true", help=_i18n("force_install_help"))
    parser.add_argument("--index_url", type=str, default=None, help=_i18n("index_url_help"))
    args = parser.parse_args()
    
    if args.old:
        torch_reqs = torch_old_requirements
        reqs = old_requirements
        print(_i18n("installing_old_deps"))
    else:
        torch_reqs = torch_requirements
        reqs = universal_requirements
        if fno_compitable(args.index_url):
            reqs.append("neuraloperator==1.0.2")
            print(_i18n("fno_compatible_detected"))
            
    if args.force:
        print(_i18n("force_install_warning"))
        
    install_uv()
    
    print(_i18n("installing_torch"))
    install_requirements(torch_reqs, force=args.force, index_url=args.index_url)
    
    print(_i18n("installing_other_deps"))
    install_requirements(reqs, force=args.force)
    
    print(_i18n("installing_setuptools"))
    install_requirements(["setuptools<76.0"], force=True)
    
    print(_i18n("installation_complete"))
