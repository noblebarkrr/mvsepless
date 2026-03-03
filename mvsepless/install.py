import os, sys, subprocess, argparse, time
from check_colab import easy_check_is_colab
import json
import re

def get_latest_version(package_name):
    """Получает последнюю версию пакета из поля LATEST в выводе pip index"""
    result = subprocess.run(
        [os.sys.executable, "-m", "pip", "index", "versions", package_name],
        capture_output=True,
        text=True
    )
    
    def parse_latest_from_output(pip_output):
        """Парсит значение LATEST из вывода pip"""
        for line in pip_output.split('\n'):
            # Ищем строку с LATEST:
            if 'LATEST:' in line:
                # Извлекаем значение после LATEST:
                match = re.search(r'LATEST:\s+(\S+)', line)
                if match:
                    return match.group(1)
        return None
    
    latest_version = parse_latest_from_output(result.stdout)
    return latest_version


def fno_compitable():
    is_torch_2 = False
    fno_c = False
    latest_version_torch = get_latest_version("torch")
    lvt = latest_version_torch.split(".")
    lvt = [int(n_) for n_ in lvt if n_.isdigit()]
    for i, num in enumerate(lvt, start=1):
        if i == 1:
            if num == 2:
                is_torch_2 = True
        elif i == 2:
            if num >= 4:
                fno_c = True
    return fno_c

def is_nvidia_gpu_present():
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
            print("Видеокарта NVIDIA обнаружена!")
            return True
        else:
            print("Команда nvidia-smi выполнена с ошибкой. Возможно, видеокарта NVIDIA отсутствует или драйверы не установлены.")
            return False

    except FileNotFoundError:
        # Команда nvidia-smi не найдена в системе
        print("Утилита nvidia-smi не найдена. Вероятно, драйверы NVIDIA не установлены.")
        return False
    except Exception as e:
        print(f"Произошла неожиданная ошибка: {e}")
        return False

cuda_available = is_nvidia_gpu_present()
def install_uv():
    print("Установка uv...")
    result = subprocess.run([os.sys.executable, "-m", "pip", "install", "uv"])
    print("uv установлен")
def install_requirements(requirements: list, force=False):
    if requirements:
        cmd = [os.sys.executable, "-m", "uv", "pip", "install", "--no-cache-dir", "-qq"]
        if force:
            cmd.append("--upgrade")
            cmd.append("--force-reinstall")
        for pkg in requirements:
            cmd.append(pkg)
        print("Установка зависимостей через uv...")
        result = subprocess.run(cmd)
        print("Установка зависимостей завершена")

universal_requirements = [
    "torch",
    "torchvision",
    "torchaudio",
    "numpy==2.0.2",
    "pandas",
    "scipy",
    "librosa",
    "samplerate==0.1.0",
    "matplotlib",
    "tqdm",
    "einops",
    "protobuf",
    "soundfile",
    "pydub",
    "webrtcvad",
    "audiomentations",
    "pedalboard",
    "ml_collections",
    "timm",
    "wandb",
    "accelerate",
    "bitsandbytes",
    "tokenizers",
    "huggingface-hub",
    "transformers",
    "torchseg",
    "demucs==4.0.0",
    "asteroid>=0.6.0",
    "pyloudnorm",
    "prodigyopt",
    "torch_log_wmse",
    "rotary_embedding_torch",
    "gradio<6.0",
    "omegaconf",
    "beartype",
    "spafe",
    "torch_audiomentations",
    "auraloss",
    "onnx>=1.17",
    "onnx2torch>=0.3.0",
    "onnxruntime-gpu>=1.17" if cuda_available else "onnxruntime>=1.17",
    "ml_dtypes",
    "resampy",
    "yt_dlp",
    "pyngrok",
    "tabulate",
    "torchcrepe",
    "praat-parselmouth",
    "faiss-cpu==1.11",
    "local-attention",
    "tenacity",
    "pyworld",
    "gdown"
]
if fno_compitable():
    universal_requirements.append("neuraloperator==1.0.2")

old_requirements = [
    "torch==1.13.1",
    "torchvision==0.14.1",
    "torchaudio==0.13.1",
    "numpy==1.26.4",
    "pandas==2.3.3",
    "scipy==1.15.3",
    "librosa==0.11.0",
    "samplerate==0.1.0",
    "matplotlib==3.10.8",
    "tqdm==4.67.1",
    "einops==0.8.1",
    "protobuf==6.33.4",
    "soundfile==0.13.1",
    "pydub==0.25.1",
    "webrtcvad==2.0.10",
    "audiomentations==0.43.1",
    "pedalboard==0.8.2",
    "ml_collections==1.1.0",
    "timm==1.0.24",
    "wandb==0.24.0",
    "accelerate==1.2.1",
    "bitsandbytes==0.45.0",
    "tokenizers==0.15.2",
    "huggingface-hub==0.34.2",
    "transformers==4.39.3",
    "torchseg==0.0.1a4",
    "demucs==4.0.0",
    "asteroid==0.6.0",
    "pyloudnorm",
    "prodigyopt==1.1.2",
    "rotary_embedding_torch==0.3.6",
    "gradio<6.0.0",
    "omegaconf==2.3.0",
    "beartype==0.22.9",
    "spafe==0.3.3",
    "torch_audiomentations==0.12.0",
    "auraloss==0.4.0",
    "onnx>=1.17",
    "onnx2torch>=0.3.0",
    "onnxruntime-gpu>=1.17" if cuda_available else "onnxruntime>=1.17",
    "ml_dtypes==0.5.4",
    "resampy==0.4.3",
    "yt_dlp",
    "pyngrok",
    "tabulate",
    "torchcrepe==0.0.24",
    "praat-parselmouth==0.4.7",
    "faiss-cpu==1.7.2",
    "local-attention==1.10.0",
    "tenacity==9.1.2",
    "pyworld==0.3.5",
    "gdown"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Установщик зависимостей")
    parser.add_argument("--old", action="store_true", help="Старые зависимости (только python 3.10)")
    parser.add_argument("--force", action="store_true", help="Принудительная установка")
    args = parser.parse_args()
    if args.old:
        reqs = old_requirements
    else:
        reqs = universal_requirements
    if args.force:
        print("Предупреждение! Зависимости устанавливаются принудительно")
    install_uv()
    install_requirements(reqs, force=args.force)

    install_requirements(["setuptools<76.0"], force=True)
