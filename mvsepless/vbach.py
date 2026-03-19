import os
import gc
from gradio_helper import GradioHelper, tz, dw_file, easy_check_is_colab, str2bool, all_ids, set_device, zerogpu_available, hf_spaces_gpu
import torch
import ast
from torch import nn
import torch.nn.functional as F
import torchcrepe
import faiss
import librosa
import math
import numpy as np
from scipy import signal
import argparse
from functools import lru_cache
import pyworld
import parselmouth
import string
from transformers import HubertModel
from typing import Tuple, Any, Dict, List, Optional, Union, Callable
import sys
import json
import yaml
import shutil
from tqdm import tqdm
import urllib.request
import gdown
import requests
import zipfile
import tempfile
import secrets
import gradio as gr
import subprocess
from datetime import datetime, timezone, timedelta
from functools import wraps
from pathlib import Path

from separator import get_files_from_list
from audio import check, read, write, output_formats, split_mid_side, split_channels, easy_resampler, stereo_to_mono, mono_to_stereo, convert_to_dtype, gain, add_zero_to_end, multi_channel_array_from_arrays, trim, fit_arrays
from namer import Namer
from i18n import _i18n, CURRENT_LANGUAGE, set_language

script_dir: str = os.path.dirname(os.path.abspath(__file__))

FILTER_ORDER: int = 5
CUTOFF_FREQUENCY: int = 48
SAMPLE_RATE: int = 16000
bh, ah = signal.butter(
    N=FILTER_ORDER, Wn=CUTOFF_FREQUENCY, btype="high", fs=SAMPLE_RATE
)

from multiprocessing import cpu_count
from vbach_lib.fairseq import load_model_ensemble_and_task, load_checkpoint_to_cpu
from vbach_lib.algorithm.synthesizers import Synthesizer
from vbach_lib.predictors.FCPE import FCPEF0Predictor
from vbach_lib.predictors.RMVPE import RMVPE0Predictor
from vbach_lib.predictors.HPA_RMVPE import HPA_RMVPE

VBACH_ALT_PIPELINE_TIME_CHUNK: int = int(os.environ.get("VBACH_ALTPL_BASE_SEG", "10"))


def format_end_count_models(count: int) -> str:
    """
    Форматирование окончания для слова "модель" в зависимости от числа
    
    Args:
        count: Количество моделей
    
    Returns:
        Окончание слова
    """
    if CURRENT_LANGUAGE == "ru":
        if count % 10 == 1 and count % 100 != 11:
            return "ь"
        elif 2 <= count % 10 <= 4 and (count % 100 < 10 or count % 100 >= 20):
            return "и"
        else:
            return "ей"
    else:
        return "s" if count != 1 else ""


class UserDirectory:
    """Класс для управления пользовательской директорией"""
    
    def __init__(self) -> None:
        self.path: str = ""
    
    def change_dir(self, directory: str) -> None:
        """
        Изменить пользовательскую директорию
        
        Args:
            directory: Путь к директории
        """
        self.path = directory
        os.makedirs(directory, exist_ok=True)
    

user_directory: UserDirectory = UserDirectory()
IS_COLAB: bool = easy_check_is_colab()

if IS_COLAB:
    print(_i18n("msg_colab_detected"))
    result = subprocess.run(['/bin/mount'], capture_output=True, text=True)

    for line in result.stdout.strip().split('\n'):
        if 'type fuse.drive' in line:
            parts = line.split(' type ')
            if len(parts) >= 2:
                source_mount = parts[0]
                source, mount_point = source_mount.split(' on ')
                user_directory.change_dir(os.path.join(mount_point, "MyDrive", "mvsepless-data-gdrive"))
                os.makedirs(user_directory.path, exist_ok=True)
                print(_i18n("msg_gdrive_mounted", path=mount_point))
                break


def generate_secure_random(length: int = 10) -> str:
    """
    Генерация безопасной случайной строки
    
    Args:
        length: Длина строки
    
    Returns:
        Случайная строка
    """
    characters: str = string.ascii_letters + string.digits
    return "".join(secrets.choice(characters) for _c in range(length))


class VbachModelManager:
    """Менеджер моделей Vbach"""
    
    def __init__(self, user_directory: UserDirectory) -> None:
        """
        Инициализация менеджера моделей
        
        Args:
            user_directory: Пользовательская директория
        """
        self.user_directory: UserDirectory = user_directory
        self.rmvpe_path: str = os.path.join(script_dir, "vbach_lib", "predictors", "rmvpe.pt")
        self.hpa_rmvpe_path: str = os.path.join(script_dir, "vbach_lib", "predictors", "hpa_rmvpe.pt")
        self.fcpe_path: str = os.path.join(script_dir, "vbach_lib", "predictors", "fcpe.pt")
        
        self.custom_fairseq_huberts_dir: str = os.path.join(
            script_dir, "vbach_lib", "huberts", "fairseq"
        )
        self.custom_transformers_huberts_dir: str = os.path.join(
            script_dir, "vbach_lib", "huberts", "transformers"
        )
        
        self.huberts_fairseq_dict: Dict[str, Dict[str, str]] = {
            "hubert_base": {
                "url": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/fairseq/hubert_base.pt?download=true",
                "local_path": os.path.join(
                    self.custom_fairseq_huberts_dir, "hubert_base.pt"
                ),
            },
            "contentvec_base": {
                "url": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/fairseq/contentvec_base.pt?download=true",
                "local_path": os.path.join(
                    self.custom_fairseq_huberts_dir, "contentvec_base.pt"
                ),
            },
            "korean_hubert_base": {
                "url": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/fairseq/korean_hubert_base.pt?download=true",
                "local_path": os.path.join(
                    self.custom_fairseq_huberts_dir, "korean_hubert_base.pt"
                ),
            },
            "chinese_hubert_base": {
                "url": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/fairseq/chinese_hubert_base.pt?download=true",
                "local_path": os.path.join(
                    self.custom_fairseq_huberts_dir, "chinese_hubert_base.pt"
                ),
            },
            "portuguese_hubert_base": {
                "url": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/fairseq/portuguese_hubert_base.pt?download=true",
                "local_path": os.path.join(
                    self.custom_fairseq_huberts_dir, "portuguese_hubert_base.pt"
                ),
            },
            "japanese_hubert_base": {
                "url": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/fairseq/japanese_hubert_base.pt?download=true",
                "local_path": os.path.join(
                    self.custom_fairseq_huberts_dir, "japanese_hubert_base.pt"
                ),
            },
        }
        
        self.huberts_transformers_dict: Dict[str, Dict[str, str]] = {
            "contentvec": {
                "base_dir": os.path.join(
                    self.custom_transformers_huberts_dir, "contentvec"
                ),
                "url_bin": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/contentvec/pytorch_model.bin?download=true",
                "url_json": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/contentvec/config.json?download=true",
                "local_bin": os.path.join(
                    self.custom_transformers_huberts_dir,
                    "contentvec",
                    "pytorch_model.bin",
                ),
                "local_json": os.path.join(
                    self.custom_transformers_huberts_dir, "contentvec", "config.json"
                ),
            },
            "spin": {
                "base_dir": os.path.join(self.custom_transformers_huberts_dir, "spin"),
                "url_bin": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/spin/pytorch_model.bin?download=true",
                "url_json": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/spin/config.json?download=true",
                "local_bin": os.path.join(
                    self.custom_transformers_huberts_dir, "spin", "pytorch_model.bin"
                ),
                "local_json": os.path.join(
                    self.custom_transformers_huberts_dir, "spin", "config.json"
                ),
            },
            "spin-v2": {
                "base_dir": os.path.join(
                    self.custom_transformers_huberts_dir, "spinv2"
                ),
                "url_bin": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/spinv2/pytorch_model.bin?download=true",
                "url_json": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/spinv2/config.json?download=true",
                "local_bin": os.path.join(
                    self.custom_transformers_huberts_dir, "spinv2", "pytorch_model.bin"
                ),
                "local_json": os.path.join(
                    self.custom_transformers_huberts_dir, "spinv2", "config.json"
                ),
            },
            "chinese-hubert-base": {
                "base_dir": os.path.join(
                    self.custom_transformers_huberts_dir, "chinese_hubert_base"
                ),
                "url_bin": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/chinese_hubert_base/pytorch_model.bin?download=true",
                "url_json": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/chinese_hubert_base/config.json?download=true",
                "local_bin": os.path.join(
                    self.custom_transformers_huberts_dir,
                    "chinese_hubert_base",
                    "pytorch_model.bin",
                ),
                "local_json": os.path.join(
                    self.custom_transformers_huberts_dir,
                    "chinese_hubert_base",
                    "config.json",
                ),
            },
            "japanese-hubert-base": {
                "base_dir": os.path.join(
                    self.custom_transformers_huberts_dir, "japanese_hubert_base"
                ),
                "url_bin": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/japanese_hubert_base/pytorch_model.bin?download=true",
                "url_json": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/japanese_hubert_base/config.json?download=true",
                "local_bin": os.path.join(
                    self.custom_transformers_huberts_dir,
                    "japanese_hubert_base",
                    "pytorch_model.bin",
                ),
                "local_json": os.path.join(
                    self.custom_transformers_huberts_dir,
                    "japanese_hubert_base",
                    "config.json",
                ),
            },
            "korean-hubert-base": {
                "base_dir": os.path.join(
                    self.custom_transformers_huberts_dir, "korean_hubert_base"
                ),
                "url_bin": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/korean_hubert_base/pytorch_model.bin?download=true",
                "url_json": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/korean_hubert_base/config.json?download=true",
                "local_bin": os.path.join(
                    self.custom_transformers_huberts_dir,
                    "korean_hubert_base",
                    "pytorch_model.bin",
                ),
                "local_json": os.path.join(
                    self.custom_transformers_huberts_dir,
                    "korean_hubert_base",
                    "config.json",
                ),
            },
        }
        
        self.requirements: List[List[str]] = [
            [
                "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/predictors/rmvpe.pt?download=true",
                self.rmvpe_path,
            ],
            [
                "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/predictors/hpa_rmvpe.pt?download=true",
                self.hpa_rmvpe_path,
            ],
            [
                "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/predictors/fcpe.pt?download=true",
                self.fcpe_path,
            ],
        ]
        
        self.voicemodels_dir: str = os.path.join(user_directory.path, "vbach_models_cache")
        os.makedirs(self.voicemodels_dir, exist_ok=True)
        
        self.voicemodels_info: str = os.path.join(self.voicemodels_dir, "vbach_models.json")
        self.voicemodels: Dict[str, Dict[str, Optional[str]]] = {}
        
        self.download_requirements()
        self.check_hubert("hubert_base")
        self.check_and_load()

    def check_hubert(self, embedder_name: str) -> Optional[str]:
        """
        Проверить наличие Hubert модели и скачать при необходимости
        
        Args:
            embedder_name: Имя эмбеддера
        
        Returns:
            Путь к модели или None
        """
        if embedder_name in self.huberts_fairseq_dict:
            if not os.path.exists(
                self.huberts_fairseq_dict[embedder_name]["local_path"]
            ):
                dw_file(
                    self.huberts_fairseq_dict[embedder_name]["url"],
                    self.huberts_fairseq_dict[embedder_name]["local_path"],
                )
            return self.huberts_fairseq_dict[embedder_name]["local_path"]
        else:
            return None

    def check_hubert_transformers(self, embedder_name: str) -> Optional[str]:
        """
        Проверить наличие Hubert модели transformers и скачать при необходимости
        
        Args:
            embedder_name: Имя эмбеддера
        
        Returns:
            Путь к директории модели или None
        """
        if embedder_name in self.huberts_transformers_dict:
            os.makedirs(
                self.huberts_transformers_dict[embedder_name]["base_dir"], exist_ok=True
            )
            if not os.path.exists(
                self.huberts_transformers_dict[embedder_name]["local_bin"]
            ) and not os.path.exists(
                self.huberts_transformers_dict[embedder_name]["local_json"]
            ):
                dw_file(
                    self.huberts_transformers_dict[embedder_name]["url_bin"],
                    self.huberts_transformers_dict[embedder_name]["local_bin"],
                )
                dw_file(
                    self.huberts_transformers_dict[embedder_name]["url_json"],
                    self.huberts_transformers_dict[embedder_name]["local_json"],
                )
            return self.huberts_transformers_dict[embedder_name]["base_dir"]
        else:
            return None

    def write_voicemodels_info(self) -> None:
        """Записать информацию о голосовых моделях в файл"""
        with open(self.voicemodels_info, "w", encoding='utf-8') as f:
            json.dump(self.voicemodels, f, indent=4, ensure_ascii=False)

    def load_voicemodels_info(self) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Загрузить информацию о голосовых моделях из файла
        
        Returns:
            Словарь с информацией о моделях
        """
        with open(self.voicemodels_info, "r", encoding='utf-8') as f:
            return json.load(f)

    def add_voice_model(
        self,
        name: str,
        pth_path: Optional[str],
        index_path: Optional[str],
    ) -> None:
        """
        Добавить голосовую модель
        
        Args:
            name: Имя модели
            pth_path: Путь к PTH файлу
            index_path: Путь к индексному файлу
        """
        self.voicemodels[name] = {"pth": pth_path, "index": index_path}
        self.write_voicemodels_info()

    def del_voice_model(self, name: str) -> str:
        """
        Удалить голосовую модель
        
        Args:
            name: Имя модели
        
        Returns:
            Сообщение о результате
        """
        if name in self.parse_voice_models():
            pth: Optional[str] = self.voicemodels[name].get("pth", None)
            index: Optional[str] = self.voicemodels[name].get("index", None)
            
            if index and os.path.exists(index):
                os.remove(index)
            if pth and os.path.exists(pth):
                os.remove(pth)
                
            del self.voicemodels[name]
            self.write_voicemodels_info()
            return _i18n("model_deleted", model=name)
        else:
            return _i18n("model_not_found", model=name)

    def parse_voice_models(self) -> List[str]:
        """
        Получить список голосовых моделей
        
        Returns:
            Список имен моделей
        """
        return list(self.voicemodels.keys())

    def parse_pth_and_index(self, name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Получить пути к PTH и индексному файлу модели
        
        Args:
            name: Имя модели
        
        Returns:
            Кортеж (путь к PTH, путь к индексу)
        """
        pth: Optional[str] = self.voicemodels[name].get("pth", None)
        index: Optional[str] = self.voicemodels[name].get("index", None)
        return pth, index

    def check_and_load(self) -> None:
        """Проверить и загрузить информацию о моделях"""
        if os.path.exists(self.voicemodels_info):
            self.voicemodels = self.load_voicemodels_info()
        else:
            self.write_voicemodels_info()

    def clear_voicemodels_info(self) -> None:
        """Очистить информацию о голосовых моделях"""
        self.voicemodels = {}
        self.write_voicemodels_info()

    def download_requirements(self) -> None:
        """Скачать необходимые компоненты"""
        for url, file in self.requirements:
            if not os.path.exists(file):
                dw_file(url, file)

    def download_voice_model_file(self, url: str, zip_name: str) -> None:
        """
        Скачать файл голосовой модели
        
        Args:
            url: URL для скачивания
            zip_name: Имя ZIP файла
        """
        try:
            if "drive.google.com" in url:
                self.download_from_google_drive(url, zip_name)
            elif "pixeldrain.com" in url:
                self.download_from_pixeldrain(url, zip_name)
            elif "disk.yandex.ru" in url or "yadi.sk" in url:
                self.download_from_yandex(url, zip_name)
            else:
                dw_file(url, zip_name)
        except Exception as e:
            print(f"{_i18n('download_error')}: {e}")

    def download_from_google_drive(self, url: str, zip_name: str) -> None:
        """
        Скачать с Google Drive
        
        Args:
            url: URL файла
            zip_name: Имя для сохранения
        """
        file_id: str = (
            url.split("file/d/")[1].split("/")[0]
            if "file/d/" in url
            else url.split("id=")[1].split("&")[0]
        )
        gdown.download(id=file_id, output=str(zip_name), quiet=False)

    def download_from_pixeldrain(self, url: str, zip_name: str) -> None:
        """
        Скачать с Pixeldrain
        
        Args:
            url: URL файла
            zip_name: Имя для сохранения
        """
        file_id: str = url.split("pixeldrain.com/u/")[1]
        response = requests.get(f"https://pixeldrain.com/api/file/{file_id}")
        with open(zip_name, "wb") as f:
            f.write(response.content)

    def download_from_yandex(self, url: str, zip_name: str) -> None:
        """
        Скачать с Yandex Disk
        
        Args:
            url: URL файла
            zip_name: Имя для сохранения
        """
        yandex_public_key: str = f"download?public_key={url}"
        yandex_api_url: str = (
            f"https://cloud-api.yandex.net/v1/disk/public/resources/{yandex_public_key}"
        )
        response = requests.get(yandex_api_url)
        if response.status_code == 200:
            download_link: str = response.json().get("href", "")
            urllib.request.urlretrieve(download_link, zip_name)
        else:
            print(f"{_i18n('yandex_error')}: {response.status_code}")

    def extract_zip(self, zip_name: str, model_name: str) -> str:
        """
        Распаковать ZIP архив с моделью
        
        Args:
            zip_name: Путь к ZIP файлу
            model_name: Имя модели
        
        Returns:
            Сообщение о результате
        """
        model_dir: str = os.path.join(
            self.voicemodels_dir, f"{model_name}_{generate_secure_random(17)}"
        )
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_name, "r") as zip_ref:
                zip_ref.extractall(model_dir)
            os.remove(zip_name)

            added_voice_models: List[str] = []

            index_filepath: Optional[str] = None
            model_filepaths: List[str] = []
            
            for root, _c, files in os.walk(model_dir):
                for name in files:
                    file_path: str = os.path.join(root, name)
                    if (
                        name.endswith(".index")
                        and os.stat(file_path).st_size > 1024 * 100
                    ):
                        index_filepath = file_path
                    if (
                        name.endswith(".pth")
                        and os.stat(file_path).st_size > 1024 * 1024 * 20
                    ):
                        model_filepaths.append(file_path)

            if len(model_filepaths) == 1:
                self.add_voice_model(model_name, model_filepaths[0], index_filepath)
                added_voice_models.append(model_name)
            else:
                for i, pth in enumerate(model_filepaths):
                    self.add_voice_model(f"{model_name}_{i + 1}", pth, index_filepath)
                    added_voice_models.append(f"{model_name}_{i + 1}")
                    
            list_models_str: str = "\n".join(added_voice_models)
            return f"{_i18n('models_added')}:\n{list_models_str}"
            
        except Exception as e:
            return f"{_i18n('model_load_error')}: {e}"

    def install_model_zip(self, zip_source: str, model_name: str, mode: str = "url") -> str:
        """
        Установить модель из ZIP архива
        
        Args:
            zip_source: Путь к ZIP или URL
            model_name: Имя модели
            mode: Режим ("url" или "local")
        
        Returns:
            Сообщение о результате
        """
        if model_name in self.parse_voice_models():
            print(_i18n("model_overwrite_warning"))
            
        if mode == "url":
            with tempfile.TemporaryDirectory(
                prefix="vbach_temp_model", ignore_cleanup_errors=True
            ) as tmp:
                zip_path: str = os.path.join(tmp, "model.zip")
                self.download_voice_model_file(zip_source, zip_path)
                status: str = self.extract_zip(zip_path, model_name)
        elif mode == "local":
            status = self.extract_zip(zip_source, model_name)
        else:
            status = _i18n("invalid_mode")
            
        return status

    def install_model_files(
        self, 
        index: Optional[str], 
        pth: Optional[str], 
        model_name: str, 
        mode: str = "url"
    ) -> str:
        """
        Установить модель из отдельных файлов
        
        Args:
            index: Путь к индексному файлу или URL
            pth: Путь к PTH файлу или URL
            model_name: Имя модели
            mode: Режим ("url" или "local")
        
        Returns:
            Сообщение о результате
        """
        if model_name in self.parse_voice_models():
            print(_i18n("model_overwrite_warning"))
            
        model_dir: str = os.path.join(
            self.voicemodels_dir, f"{model_name}_{generate_secure_random(17)}"
        )
        os.makedirs(model_dir, exist_ok=True)
        
        local_index_path: Optional[str] = None
        local_pth_path: Optional[str] = None
        
        try:
            if mode == "url":
                if index:
                    local_index_path = os.path.join(model_dir, "model.index")
                    self.download_voice_model_file(index, local_index_path)
                if pth:
                    local_pth_path = os.path.join(model_dir, "model.pth")
                    self.download_voice_model_file(pth, local_pth_path)

            elif mode == "local":
                if index and os.path.exists(index):
                    local_index_path = os.path.join(
                        model_dir, os.path.basename(index)
                    )
                    shutil.copy(index, local_index_path)
                if pth and os.path.exists(pth):
                    local_pth_path = os.path.join(model_dir, os.path.basename(pth))
                    shutil.copy(pth, local_pth_path)
            else:
                return _i18n("invalid_mode")

            self.add_voice_model(model_name, local_pth_path, local_index_path)
            return _i18n("model_added", model=model_name)
            
        except Exception as e:
            return f"{_i18n('model_load_error')}: {e}"

    def get_list_installed_models(self) -> None:
        """
        Вывести список установленных моделей в стиле separator.py
        """
        models: List[str] = self.parse_voice_models()
        
        if not models:
            print(_i18n("no_models_installed"))
            return
        
        f_key: str = _i18n("model_name")
        s_key: str = _i18n("model_files")
        
        # Определяем максимальную ширину для форматирования
        name_width = max(len(f_key), max(len(model) for model in models)) + 2
        files_width = 60  # Фиксированная ширина для колонки файлов
        
        print("|-", "-" * name_width, "-+-", "-" * files_width, "-|", sep="")
        print(f"| {f_key:<{name_width}} | {s_key:<{files_width}} |")
        print("|-", "-" * name_width, "-+-", "-" * files_width, "-|", sep="")
        
        for model in models:
            pth, index = self.parse_pth_and_index(model)
            
            files_info = []
            if pth:
                pth_size = os.path.getsize(pth) if os.path.exists(pth) else 0
                pth_size_mb = pth_size / (1024 * 1024)
                files_info.append(f"PTH: {pth_size_mb:.1f} MB")
            else:
                files_info.append("PTH: None")
                
            if index and os.path.exists(index):
                idx_size = os.path.getsize(index)
                idx_size_mb = idx_size / (1024 * 1024)
                files_info.append(f"INDEX: {idx_size_mb:.1f} MB")
            else:
                files_info.append("INDEX: None")
            
            files_str = " | ".join(files_info)
            
            # Обрезаем если слишком длинно
            if len(files_str) > files_width:
                files_str = files_str[:files_width-3] + "..."
            
            print(f"| {model:<{name_width}} | {files_str:<{files_width}} |")
            print("|-", "-" * name_width, "-+-", "-" * files_width, "-|", sep="")
        
        print(_i18n("installed_models_count", count=len(models), end=format_end_count_models(len(models))))


model_manager: VbachModelManager = VbachModelManager(user_directory)
namer: Namer = Namer()

f0_methods: Tuple[str, ...] = (
    "rmvpe+",
    "hpa-rmvpe",
    "fcpe",
    "mangio-crepe",
    "mangio-crepe-tiny",
    "harvest",
    "pm",
    "pyin",
)

HPA_RMVPE_DIR: str = model_manager.hpa_rmvpe_path
RMVPE_DIR: str = model_manager.rmvpe_path
FCPE_DIR: str = model_manager.fcpe_path

input_audio_path2wav: Dict[str, np.ndarray] = {}


class HubertModelWithFinalProj(HubertModel):
    """Hubert модель с финальной проекцией"""
    
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


@lru_cache(maxsize=128)
def get_harvest_f0(
    input_audio_path: str, 
    fs: int, 
    f0max: float, 
    f0min: float, 
    frame_period: float
) -> np.ndarray:
    """
    Получить F0 с помощью Harvest
    
    Args:
        input_audio_path: Путь к аудиофайлу
        fs: Частота дискретизации
        f0max: Максимальная частота F0
        f0min: Минимальная частота F0
        frame_period: Период кадра
    
    Returns:
        Массив F0
    """
    audio: np.ndarray = input_audio_path2wav[input_audio_path]
    f0, t = pyworld.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0


class AudioProcessor:
    """Класс для обработки аудио"""
    
    @staticmethod
    def change_rms(
        sourceaudio: np.ndarray, 
        source_rate: int, 
        targetaudio: np.ndarray, 
        target_rate: int, 
        rate: float
    ) -> np.ndarray:
        """
        Изменить RMS (громкость) аудио
        
        Args:
            sourceaudio: Исходное аудио
            source_rate: Частота исходного аудио
            targetaudio: Целевое аудио
            target_rate: Частота целевого аудио
            rate: Коэффициент изменения
        
        Returns:
            Измененное аудио
        """
        rms1 = librosa.feature.rms(
            y=sourceaudio,
            frame_length=source_rate // 2 * 2,
            hop_length=source_rate // 2,
        )
        rms2 = librosa.feature.rms(
            y=targetaudio,
            frame_length=target_rate // 2 * 2,
            hop_length=target_rate // 2,
        )

        rms1 = F.interpolate(
            torch.from_numpy(rms1).float().unsqueeze(0),
            size=targetaudio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = F.interpolate(
            torch.from_numpy(rms2).float().unsqueeze(0),
            size=targetaudio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6)

        adjustedaudio: np.ndarray = (
            targetaudio
            * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1)).numpy()
        )
        return adjustedaudio


class VC:
    """Класс для голосового преобразования"""
    
    def __init__(self, tgt_sr: int, config: Any, stack: str = "fairseq") -> None:
        """
        Инициализация VC
        
        Args:
            tgt_sr: Целевая частота дискретизации
            config: Конфигурация
            stack: Стек ("fairseq" или "transformers")
        """
        self.x_pad: int = config.x_pad
        self.x_query: int = config.x_query
        self.x_center: int = config.x_center
        self.x_max: int = config.x_max
        self.is_half: bool = config.is_half
        self.sample_rate: int = 16000
        self.window: int = 160
        self.t_pad: int = self.sample_rate * self.x_pad
        self.t_pad_tgt: int = tgt_sr * self.x_pad
        self.t_pad2: int = self.t_pad * 2
        self.t_query: int = self.sample_rate * self.x_query
        self.t_center: int = self.sample_rate * self.x_center
        self.t_max: int = self.sample_rate * self.x_max
        self.time_step: float = self.window / self.sample_rate * 1000
        self.device: torch.device = config.device
        self.vc: Callable = self._vc_transformers if stack == "transformers" else self._vc

    def get_f0_mangio_crepe(
        self, 
        x: np.ndarray, 
        f0_min: int, 
        f0_max: int, 
        p_len: int, 
        hop_length: int, 
        model: str = "full"
    ) -> np.ndarray:
        """
        Получить F0 с помощью Mangio-Crepe
        
        Args:
            x: Аудиоданные
            f0_min: Минимальная частота F0
            f0_max: Максимальная частота F0
            p_len: Длина
            hop_length: Длина шага
            model: Модель ("full" или "tiny")
        
        Returns:
            Массив F0
        """
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        audio = torch.from_numpy(x).to(self.device, copy=True).unsqueeze(0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        pitch = torchcrepe.predict(
            audio,
            self.sample_rate,
            hop_length,
            f0_min,
            f0_max,
            model,
            batch_size=hop_length * 2,
            device=self.device,
            pad=True,
        )

        p_len = p_len or x.shape[0] // hop_length
        source = np.array(pitch.squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source,
        )
        f0 = np.nan_to_num(target)
        return f0

    def get_f0_rmvpe(
        self, 
        x: np.ndarray, 
        f0_min: int = 1, 
        f0_max: int = 40000, 
        *args, 
        **kwargs
    ) -> np.ndarray:
        """
        Получить F0 с помощью RMVPE
        
        Args:
            x: Аудиоданные
            f0_min: Минимальная частота F0
            f0_max: Максимальная частота F0
        
        Returns:
            Массив F0
        """
        if not hasattr(self, "model_rmvpe"):
            self.model_rmvpe = RMVPE0Predictor(
                RMVPE_DIR, is_half=self.is_half, device=self.device
            )
        f0 = self.model_rmvpe.infer_from_audio_with_pitch(
            x, thred=0.03, f0_min=f0_min, f0_max=f0_max
        )
        return f0
    
    def get_f0_hpa_rmvpe(
        self, 
        x: np.ndarray, 
        f0_min: int = 1, 
        f0_max: int = 40000, 
        *args, 
        **kwargs
    ) -> np.ndarray:
        """
        Получить F0 с помощью HPA-RMVPE
        
        Args:
            x: Аудиоданные
            f0_min: Минимальная частота F0
            f0_max: Максимальная частота F0
        
        Returns:
            Массив F0
        """
        if not hasattr(self, "model_hpa_rmvpe"):
            self.model_hpa_rmvpe = HPA_RMVPE(
                HPA_RMVPE_DIR, device=self.device, hpa=True
            )
        f0 = self.model_hpa_rmvpe.infer_from_audio_with_pitch(
            x, thred=0.03, f0_min=f0_min, f0_max=f0_max
        )
        return f0

    def get_f0_fcpe(
        self, 
        x: np.ndarray, 
        f0_min: int = 50, 
        f0_max: int = 1100, 
        p_len: Optional[int] = None
    ) -> np.ndarray:
        """
        Получить F0 с помощью FCPE
        
        Args:
            x: Аудиоданные
            f0_min: Минимальная частота F0
            f0_max: Максимальная частота F0
            p_len: Длина
        
        Returns:
            Массив F0
        """
        self.model_fcpe = FCPEF0Predictor(
            FCPE_DIR,
            f0_min=int(f0_min),
            f0_max=int(f0_max),
            dtype=torch.float32,
            device=self.device,
            sample_rate=self.sample_rate,
            threshold=0.03,
        )
        f0 = self.model_fcpe.compute_f0(x, p_len=p_len or len(x) // self.window)
        del self.model_fcpe
        gc.collect()
        return f0

    def get_f0_librosa(
        self, 
        x: np.ndarray, 
        p_len: int, 
        f0_min: int = 50, 
        f0_max: int = 1100, 
        hop_length: int = 160
    ) -> np.ndarray:
        """
        Получить F0 с помощью Librosa
        
        Args:
            x: Аудиоданные
            p_len: Длина
            f0_min: Минимальная частота F0
            f0_max: Максимальная частота F0
            hop_length: Длина шага
        
        Returns:
            Массив F0
        """
        f0, *_ = librosa.pyin(
            x.astype(np.float32),
            sr=self.sample_rate,
            fmin=f0_min,
            fmax=f0_max,
            hop_length=hop_length,
        )
        return self._resize_f0(f0, p_len)

    def _resize_f0(self, x: np.ndarray, target_len: int) -> np.ndarray:
        """
        Изменить размер массива F0
        
        Args:
            x: Исходный массив F0
            target_len: Целевая длина
        
        Returns:
            Измененный массив F0
        """
        source = np.array(x)
        source[source < 0.001] = np.nan

        output_f0 = np.nan_to_num(
            np.interp(
                np.arange(0, len(source) * target_len, len(source)) / target_len,
                np.arange(0, len(source)),
                source,
            )
        )
        return output_f0.astype(np.float32)

    def get_f0(
        self,
        inputaudio_path: str,
        x: np.ndarray,
        p_len: int,
        pitch: float,
        f0_method: str,
        filter_radius: int,
        hop_length: int,
        inp_f0: Optional[np.ndarray] = None,
        f0_min: int = 50,
        f0_max: int = 1100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Получить F0 выбранным методом
        
        Args:
            inputaudio_path: Путь к аудиофайлу
            x: Аудиоданные
            p_len: Длина
            pitch: Высота тона
            f0_method: Метод извлечения F0
            filter_radius: Радиус фильтра
            hop_length: Длина шага
            inp_f0: Входной F0
            f0_min: Минимальная частота F0
            f0_max: Максимальная частота F0
        
        Returns:
            Кортеж (f0_coarse, f0bak)
        """
        global input_audio_path2wav
        time_step: float = self.window / self.sample_rate * 1000
        f0_mel_min: float = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max: float = 1127 * np.log(1 + f0_max / 700)

        if f0_method in ["mangio-crepe", "mangio-crepe-tiny"]:
            f0 = self.get_f0_mangio_crepe(
                x,
                f0_min,
                f0_max,
                p_len,
                int(hop_length),
                "tiny" if f0_method == "mangio-crepe-tiny" else "full",
            )

        elif f0_method == "pyin":
            f0 = self.get_f0_librosa(x, p_len, f0_min, f0_max, hop_length)

        elif f0_method == "fcpe":
            f0 = self.get_f0_fcpe(x, f0_min, f0_max, p_len)

        elif f0_method == "harvest":
            input_audio_path2wav = {}
            input_audio_path2wav[inputaudio_path] = x.astype(np.double)
            f0 = get_harvest_f0(inputaudio_path, self.sample_rate, f0_max, f0_min, 10)
            if filter_radius > 2:
                f0 = signal.medfilt(f0, 3)
                
        elif f0_method == "pm":
            f0 = (
                parselmouth.Sound(x, self.sample_rate)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size: int = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )

        elif f0_method == "rmvpe+":
            f0 = self.get_f0_rmvpe(x=x, f0_min=f0_min, f0_max=f0_max)

        elif f0_method == "hpa-rmvpe":
            f0 = self.get_f0_hpa_rmvpe(x=x, f0_min=f0_min, f0_max=f0_max)

        else:
            raise ValueError(_i18n("unknown_f0_method", method=f0_method))

        f0 *= pow(2, pitch / 12)
        tf0: int = self.sample_rate // self.window
        
        if inp_f0 is not None:
            delta_t: int = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            shape: int = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[
                :shape
            ]

        f0bak: np.ndarray = f0.copy()
        f0_mel: np.ndarray = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse: np.ndarray = np.rint(f0_mel).astype(int)
        
        return f0_coarse, f0bak

    def _vc(
        self,
        model: nn.Module,
        net_g: nn.Module,
        sid: torch.Tensor,
        audio0: np.ndarray,
        pitch: Optional[torch.Tensor],
        pitchf: Optional[torch.Tensor],
        index: Optional[faiss.Index],
        big_npy: Optional[np.ndarray],
        index_rate: float,
        version: str,
        protect: float,
    ) -> np.ndarray:
        """
        Внутренний метод голосового преобразования (fairseq)
        
        Args:
            model: Модель Hubert
            net_g: Генератор
            sid: ID спикера
            audio0: Аудиоданные
            pitch: Высота тона
            pitchf: F0
            index: Индекс FAISS
            big_npy: Массив эмбеддингов
            index_rate: Коэффициент влияния индекса
            version: Версия модели
            protect: Защита согласных
        
        Returns:
            Преобразованные аудиоданные
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        feats = torch.from_numpy(audio0)
        feats = feats.half() if self.is_half else feats.float()

        if feats.dim() == 2:
            feats = feats.mean(-1)

        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs: Dict[str, Any] = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.is_half):
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

            if protect < 0.5 and pitch is not None and pitchf is not None:
                feats0 = feats.clone()

            if index is not None and big_npy is not None and index_rate != 0:
                npy = feats[0].cpu().numpy()
                npy = npy.astype("float32") if self.is_half else npy
                score, ix = index.search(npy, k=8)
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                npy = npy.astype("float16") if self.is_half else npy
                feats = (
                    torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                    + (1 - index_rate) * feats
                )

            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
            if protect < 0.5 and pitch is not None and pitchf is not None:
                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                    0, 2, 1
                )

            p_len: int = audio0.shape[0] // self.window
            if feats.shape[1] < p_len:
                p_len = feats.shape[1]
                if pitch is not None and pitchf is not None:
                    pitch = pitch[:, :p_len]
                    pitchf = pitchf[:, :p_len]

            if protect < 0.5 and pitch is not None and pitchf is not None:
                pitchff = pitchf.clone()
                pitchff[pitchf > 0] = 1
                pitchff[pitchf < 1] = protect
                pitchff = pitchff.unsqueeze(-1)
                feats = feats * pitchff + feats0 * (1 - pitchff)
                feats = feats.to(feats0.dtype)

            p_len_tensor = torch.tensor([p_len], device=self.device).long()

            if pitch is not None and pitchf is not None:
                audio1 = (
                    (net_g.infer(feats, p_len_tensor, pitch, pitchf, sid)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )
            else:
                audio1 = (
                    (net_g.infer(feats, p_len_tensor, sid)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )

        del feats, p_len_tensor, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio1

    def _vc_transformers(
        self,
        model: nn.Module,
        net_g: nn.Module,
        sid: torch.Tensor,
        audio0: np.ndarray,
        pitch: Optional[torch.Tensor],
        pitchf: Optional[torch.Tensor],
        index: Optional[faiss.Index],
        big_npy: Optional[np.ndarray],
        index_rate: float,
        version: str,
        protect: float,
    ) -> np.ndarray:
        """
        Внутренний метод голосового преобразования (transformers)
        
        Args:
            model: Модель Hubert
            net_g: Генератор
            sid: ID спикера
            audio0: Аудиоданные
            pitch: Высота тона
            pitchf: F0
            index: Индекс FAISS
            big_npy: Массив эмбеддингов
            index_rate: Коэффициент влияния индекса
            version: Версия модели
            protect: Защита согласных
        
        Returns:
            Преобразованные аудиоданные
        """
        with torch.no_grad():
            pitch_guidance: bool = pitch is not None and pitchf is not None
            feats = torch.from_numpy(audio0).float()
            feats = feats.mean(-1) if feats.dim() == 2 else feats
            assert feats.dim() == 1, feats.dim()
            feats = feats.view(1, -1).to(self.device)
            feats = model(feats)["last_hidden_state"]
            feats = (
                model.final_proj(feats[0]).unsqueeze(0) if version == "v1" else feats
            )
            feats0 = feats.clone() if pitch_guidance else None
            
            if index is not None and big_npy is not None and index_rate != 0:
                feats = self._retrieve_speaker_embeddings(feats, index, big_npy, index_rate)
                
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
            p_len: int = min(audio0.shape[0] // self.window, feats.shape[1])
            
            if pitch_guidance:
                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                    0, 2, 1
                )
                if pitch is not None and pitchf is not None:
                    pitch = pitch[:, :p_len]
                    pitchf = pitchf[:, :p_len]
                    
                if protect < 0.5:
                    pitchff = pitchf.clone()
                    pitchff[pitchf > 0] = 1
                    pitchff[pitchf < 1] = protect
                    feats = feats * pitchff.unsqueeze(-1) + feats0 * (
                        1 - pitchff.unsqueeze(-1)
                    )
                    feats = feats.to(feats0.dtype)
            else:
                pitch, pitchf = None, None
                
            p_len_tensor = torch.tensor([p_len], device=self.device).long()
            audio1 = (
                (net_g.infer(feats.float(), p_len_tensor, pitch, pitchf.float() if pitchf is not None else None, sid)[0][0, 0])
                .data.cpu()
                .float()
                .numpy()
            )
            
            del feats, feats0, p_len_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return audio1

    def pipeline(
        self,
        model: nn.Module,
        net_g: nn.Module,
        sid: int,
        audio: np.ndarray,
        inputaudio_path: str,
        pitch: float,
        f0_method: str,
        file_index: Optional[str],
        index_rate: float,
        pitch_guidance: bool,
        filter_radius: int,
        tgt_sr: int,
        resample_sr: int,
        volume_envelope: float,
        version: str,
        protect: float,
        hop_length: int,
        f0_file: Optional[Any],
        f0_min: int = 50,
        f0_max: int = 1100,
        add_text: str = ""
    ) -> np.ndarray:
        """
        Основной пайплайн обработки (оригинальный)
        
        Args:
            model: Модель Hubert
            net_g: Генератор
            sid: ID спикера
            audio: Аудиоданные
            inputaudio_path: Путь к аудиофайлу
            pitch: Высота тона
            f0_method: Метод извлечения F0
            file_index: Путь к индексному файлу
            index_rate: Коэффициент влияния индекса
            pitch_guidance: Использовать направление по высоте тона
            filter_radius: Радиус фильтра
            tgt_sr: Целевая частота дискретизации
            resample_sr: Частота ресемплинга
            volume_envelope: Огибающая громкости
            version: Версия модели
            protect: Защита согласных
            hop_length: Длина шага
            f0_file: Файл с F0
            f0_min: Минимальная частота F0
            f0_max: Максимальная частота F0
            add_text: Дополнительный текст для прогресса
        
        Returns:
            Преобразованные аудиоданные
        """
        if (
            file_index is not None
            and file_index != ""
            and os.path.exists(file_index)
            and index_rate != 0
        ):
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as e:
                print(f"{_i18n('faiss_error')}: {e}")
                index = big_npy = None
        else:
            index = big_npy = None
            
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts: List[int] = []
        
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )
                
        s: int = 0
        audio_opt: List[np.ndarray] = []
        t: Optional[int] = None
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len: int = audio_pad.shape[0] // self.window
        inp_f0: Optional[np.ndarray] = None
        
        if f0_file and hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = np.array(
                    [[float(i) for i in line.split(",")] for line in lines],
                    dtype="float32",
                )
            except Exception as e:
                print(f"{_i18n('f0_file_error')}: {e}")
                
        sid_tensor = torch.tensor(sid, device=self.device).unsqueeze(0).long()

        progress = gr.Progress()
        progress((2, 4), desc=f"{_i18n('calculating_f0')} {add_text}")

        if pitch_guidance:
            pitch_coarse, pitchf = self.get_f0(
                inputaudio_path,
                audio_pad,
                p_len,
                pitch,
                f0_method,
                filter_radius,
                hop_length,
                inp_f0,
                f0_min,
                f0_max,
            )
            pitch_coarse = pitch_coarse[:p_len]
            pitchf = pitchf[:p_len]
            if self.device.type == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch_tensor = torch.tensor(pitch_coarse, device=self.device).unsqueeze(0).long()
            pitchf_tensor = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        else:
            pitch_tensor = pitchf_tensor = None

        total_ts: int = len(opt_ts)

        for i, t in enumerate(opt_ts, start=1):
            progress((i, total_ts), desc=f"{_i18n('voice_synthesis')} {add_text}", unit=_i18n("chunks"))
            print(f"\r{_i18n('voice_synthesis')} {int((i / total_ts) * 100)}% {add_text}", end="")
            t = t // self.window * self.window
            
            if pitch_guidance:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid_tensor,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch_tensor[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_tensor is not None else None,
                        pitchf_tensor[:, s // self.window : (t + self.t_pad2) // self.window] if pitchf_tensor is not None else None,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid_tensor,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t
            
        if pitch_guidance:
            progress(1, desc=f"{_i18n('voice_synthesis_final')} {add_text}")
            print(f"\r{_i18n('voice_synthesis')} 100% {add_text}", end="")
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid_tensor,
                    audio_pad[t:] if t is not None else audio_pad,
                    pitch_tensor[:, t // self.window :] if (pitch_tensor is not None and t is not None) else pitch_tensor,
                    pitchf_tensor[:, t // self.window :] if (pitchf_tensor is not None and t is not None) else pitchf_tensor,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            progress(1, desc=f"{_i18n('voice_synthesis_final')} {add_text}")
            print(f"\r{_i18n('voice_synthesis')} 100% {add_text}", end="")
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid_tensor,
                    audio_pad[t:] if t is not None else audio_pad,
                    None,
                    None,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
            
        print("")
        audio_opt_array = np.concatenate(audio_opt)
        
        if volume_envelope != 1:
            audio_opt_array = AudioProcessor.change_rms(
                audio, self.sample_rate, audio_opt_array, tgt_sr, volume_envelope
            )
            
        if resample_sr >= self.sample_rate and tgt_sr != resample_sr:
            audio_opt_array = librosa.resample(
                audio_opt_array, orig_sr=tgt_sr, target_sr=resample_sr
            )

        audio_max = np.abs(audio_opt_array).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt_array = (audio_opt_array * max_int16).astype(np.int16)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio_opt_array

    def pipeline2(
        self,
        model: nn.Module,
        net_g: nn.Module,
        sid: int,
        audio: np.ndarray,
        inputaudio_path: str,
        pitch: float,
        f0_method: str,
        file_index: Optional[str],
        index_rate: float,
        pitch_guidance: bool,
        filter_radius: int,
        tgt_sr: int,
        resample_sr: int,
        volume_envelope: float,
        version: str,
        protect: float,
        hop_length: int,
        f0_file: Optional[Any],
        f0_min: int = 50,
        f0_max: int = 1100,
        add_text: str = ""
    ) -> np.ndarray:
        """
        Альтернативный пайплайн обработки (с разбиением на чанки)
        
        Args:
            model: Модель Hubert
            net_g: Генератор
            sid: ID спикера
            audio: Аудиоданные
            inputaudio_path: Путь к аудиофайлу
            pitch: Высота тона
            f0_method: Метод извлечения F0
            file_index: Путь к индексному файлу
            index_rate: Коэффициент влияния индекса
            pitch_guidance: Использовать направление по высоте тона
            filter_radius: Радиус фильтра
            tgt_sr: Целевая частота дискретизации
            resample_sr: Частота ресемплинга
            volume_envelope: Огибающая громкости
            version: Версия модели
            protect: Защита согласных
            hop_length: Длина шага
            f0_file: Файл с F0
            f0_min: Минимальная частота F0
            f0_max: Максимальная частота F0
            add_text: Дополнительный текст для прогресса
        
        Returns:
            Преобразованные аудиоданные
        """
        device = self.device
        audio = signal.filtfilt(bh, ah, audio)
        audio_len = len(audio)

        if (
            file_index
            and file_index != ""
            and os.path.exists(file_index)
            and index_rate != 0
        ):
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as e:
                print(f"{_i18n('faiss_error')}: {e}")
                index = big_npy = None
        else:
            index = big_npy = None

        inp_f0 = None
        if f0_file and hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = np.array(
                    [[float(i) for i in line.split(",")] for line in lines],
                    dtype="float32",
                )
            except Exception as e:
                print(f"{_i18n('f0_file_error')}: {e}")

        sid_tensor = torch.tensor(sid, device=device).unsqueeze(0).long()

        raw_chunk_size = self.get_max_memory_chunk(audio_len, model, net_g, version)
        offset = int(tgt_sr // 12.5)
        real_chunk_size = raw_chunk_size
        if real_chunk_size <= 0:
            raise ValueError(_i18n("chunk_size_error"))
        
        print(f"{_i18n('chunk_size')}: {real_chunk_size} | {int(real_chunk_size / self.sample_rate)} {_i18n('seconds')}")

        audio_pad = np.pad(audio, (offset, offset), mode="reflect")

        progress = gr.Progress()
        progress((2, 4), desc=f"{_i18n('calculating_f0')} {add_text}")

        pitch_tensor: Optional[torch.Tensor] = None
        pitchf_tensor: Optional[torch.Tensor] = None
        
        if pitch_guidance:
            p_len = len(audio_pad) // self.window
            pitch_coarse, pitchf = self.get_f0(
                inputaudio_path,
                audio_pad,
                p_len,
                pitch,
                f0_method,
                filter_radius,
                hop_length,
                inp_f0,
                f0_min,
                f0_max,
            )
            pitch_coarse = pitch_coarse[:p_len]
            pitchf = pitchf[:p_len]
            if device.type == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch_tensor = torch.tensor(pitch_coarse, device=device).unsqueeze(0).long()
            pitchf_tensor = torch.tensor(pitchf, device=device).unsqueeze(0).float()

        processed_chunks: List[Tuple[int, int, np.ndarray, int, int]] = []
        start = 0

        chunk_count: int = 0
        temp_start = 0
        while temp_start < audio_len:
            temp_end = min(temp_start + real_chunk_size, audio_len)
            chunk_count += 1
            temp_start = temp_end
        
        current_chunk = 0

        while start < audio_len:
            current_chunk += 1
            progress(
                (current_chunk, chunk_count), 
                desc=f"{_i18n('voice_synthesis_alt')} {add_text}", unit=_i18n("chunks")
            )
            print(f"\r{_i18n('voice_synthesis_alt')} {int((current_chunk / chunk_count) * 100)}% {add_text}", end="")
            
            end = min(start + real_chunk_size, audio_len)

            need_left = start > 0
            need_right = end < audio_len
            pad_left = offset if need_left else 0
            pad_right = offset if need_right else 0

            chunk_start_in_pad = start - pad_left
            chunk_end_in_pad = end + pad_right

            chunk_audio = audio_pad[
                chunk_start_in_pad + offset : chunk_end_in_pad + offset
            ]

            f0_start = (chunk_start_in_pad + offset) // self.window
            f0_end = (chunk_end_in_pad + offset) // self.window

            if pitch_guidance and pitch_tensor is not None and pitchf_tensor is not None:
                out = self.vc(
                    model,
                    net_g,
                    sid_tensor,
                    chunk_audio,
                    pitch_tensor[:, f0_start:f0_end],
                    pitchf_tensor[:, f0_start:f0_end],
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )
            else:
                out = self.vc(
                    model,
                    net_g,
                    sid_tensor,
                    chunk_audio,
                    None,
                    None,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )

            output_start = int(round((chunk_start_in_pad) / self.sample_rate * tgt_sr))
            output_end = output_start + len(out)

            processed_chunks.append(
                (output_start, output_end, out, pad_left, pad_right)
            )

            start = end

        if not processed_chunks:
            raise RuntimeError(_i18n("no_chunks_error"))

        max_output_end = max(end for _c, end, _c, _c, _c in processed_chunks)
        output = np.zeros(max_output_end, dtype=np.float32)
        weight = np.zeros(max_output_end, dtype=np.float32)

        for start_idx, end_idx, chunk, pad_left, pad_right in processed_chunks:
            chunk_len = len(chunk)
            if chunk_len != (end_idx - start_idx):
                end_idx = start_idx + chunk_len

            w = np.ones(chunk_len, dtype=np.float32)
            fade_len = int(round(offset / self.sample_rate * tgt_sr))

            if pad_left > 0 and fade_len > 0:
                actual_fade = min(fade_len, chunk_len)
                w[:actual_fade] = np.linspace(0, 1, actual_fade)
            if pad_right > 0 and fade_len > 0:
                actual_fade = min(fade_len, chunk_len)
                w[-actual_fade:] = np.linspace(1, 0, actual_fade)

            output_end = min(end_idx, len(output))
            chunk = chunk[: output_end - start_idx]
            w = w[: output_end - start_idx]

            output[start_idx:output_end] += chunk * w
            weight[start_idx:output_end] += w

        mask = weight > 1e-8
        output[mask] /= weight[mask]

        expected_final_len = int(round(audio_len / self.sample_rate * tgt_sr))
        print("")
        audio_opt = output[:expected_final_len]

        if volume_envelope != 1:
            audio_opt = AudioProcessor.change_rms(
                audio, self.sample_rate, audio_opt, tgt_sr, volume_envelope
            )
        if resample_sr >= self.sample_rate and tgt_sr != resample_sr:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )

        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return audio_opt

    def get_max_memory_chunk(
        self, audio_length: int, model: nn.Module, net_g: nn.Module, version: str
    ) -> int:
        """
        Рассчитывает оптимальный размер чанка на основе доступной памяти
        
        Args:
            audio_length: Длина аудио
            model: Модель Hubert
            net_g: Генератор
            version: Версия модели
        
        Returns:
            Оптимальный размер чанка
        """
        base_chunk_size = min(
            self.sample_rate * VBACH_ALT_PIPELINE_TIME_CHUNK, 
            audio_length
        )

        if self.device.type == "cuda" and torch.cuda.is_available() and not str2bool(os.environ.get("VBACH_ALTPL_PREF_BASE_SEG", "False")):
            try:
                torch.cuda.synchronize()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated(0)
                free_memory = total_memory - allocated

                usable_memory = free_memory * 0.2

                print(
                    f"{_i18n('vram_available')}: {free_memory/1024**3:.2f} GB, "
                    f"{_i18n('using')}: {usable_memory/1024**3:.2f} GB"
                )

                memory_per_second = 100 * 1024 * 1024

                max_seconds = usable_memory / memory_per_second
                max_seconds = int(max_seconds)
                chunk_seconds = max(10.0, max_seconds)
                chunk_size = int(chunk_seconds * self.sample_rate)

                chunk_size = max(self.window, (chunk_size // self.window) * self.window)

                min_chunk_size = self.sample_rate * 2
                chunk_size = max(chunk_size, min_chunk_size)

                chunk_size = min(chunk_size, audio_length)

                return chunk_size

            except Exception as e:
                print(f"{_i18n('chunk_calc_error')}: {e}")

        return min(base_chunk_size, audio_length)

    def _retrieve_speaker_embeddings(
        self, 
        feats: torch.Tensor, 
        index: faiss.Index, 
        big_npy: np.ndarray, 
        index_rate: float
    ) -> torch.Tensor:
        """
        Получить эмбеддинги спикера из индекса
        
        Args:
            feats: Эмбеддинги
            index: Индекс FAISS
            big_npy: Массив эмбеддингов
            index_rate: Коэффициент влияния индекса
        
        Returns:
            Обновленные эмбеддинги
        """
        npy = feats[0].cpu().numpy()
        score, ix = index.search(npy, k=8)
        weight = np.square(1 / score)
        weight /= weight.sum(axis=1, keepdims=True)
        npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
        feats = (
            torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
            + (1 - index_rate) * feats
        )
        return feats


def loadaudio(
    file_path: str, 
    target_sr: int, 
    stereo_mode: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Загрузить аудиофайл
    
    Args:
        file_path: Путь к файлу
        target_sr: Целевая частота дискретизации
        stereo_mode: Режим стерео
    
    Returns:
        Кортеж (mid, left, right)
    """
    try:
        mid: Optional[np.ndarray] = None
        left: Optional[np.ndarray] = None
        right: Optional[np.ndarray] = None
        
        if stereo_mode == "mono":
            mid, sr = read(path=file_path, sr=target_sr, mono=True, flatten=True)
        else:
            stereoaudio, sr = read(path=file_path, sr=target_sr, mono=False)
            if stereo_mode == "left/right":
                left, right = split_channels(stereoaudio)
            elif stereo_mode == "sim/dif":
                center, stereo_base = split_mid_side(stereoaudio, var=3, sr=target_sr)
                mid = stereo_to_mono(center, to_flatten=True)
                left, right = split_channels(stereo_base)
        return mid, left, right
    except Exception as e:
        raise RuntimeError(f"{_i18n('audio_load_error', file=file_path)}: {str(e)}")


class Config:
    """Конфигурация для VC"""
    
    def __init__(self, device_str: str) -> None:
        """
        Инициализация конфигурации
        
        Args:
            device_str: Строка устройства
        """
        self.device_str: str = device_str
        self.device_ids: Optional[List[int]] = None
        self.set_device(self.device_str)
        self.is_half: bool = False
        self.n_cpu: int = cpu_count()
        self.gpu_name: Optional[str] = None
        self.gpu_mem: Optional[int] = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def set_device(self, device_str: str) -> None:
        """
        Установить устройство
        
        Args:
            device_str: Строка устройства
        """
        if "cuda" in device_str.lower():
            if ":" in device_str:
                device_spec = device_str.split(":")[1]
                self.device_ids = [int(id) for id in device_spec.split(",") if id.isdigit()]
            else:
                self.device_ids = list(range(torch.cuda.device_count()))
            self.device = torch.device("cuda" if not self.device_ids else f"cuda:{self.device_ids[0]}")
        elif "mps" in device_str.lower():
            self.device_ids = None
            self.device = torch.device("mps")
        else:
            self.device_ids = None
            self.device = torch.device("cpu")

    def device_config(self) -> Tuple[int, int, int, int]:
        """
        Настройка параметров для устройства
        
        Returns:
            Кортеж (x_pad, x_query, x_center, x_max)
        """
        if self.device.type == "cuda":
            print(_i18n("using_cuda"))
            if self.device_ids:
                self.gpu_mem = self._configure_gpu(self.device_ids[0])
        elif self.device.type == "mps":
            print(_i18n("using_mps"))
        else:
            print(_i18n("using_cpu"))

        x_pad, x_query, x_center, x_max = (
            (3, 10, 60, 65) if self.is_half else (1, 6, 38, 41)
        )
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad, x_query, x_center, x_max = (1, 5, 30, 32)

        return x_pad, x_query, x_center, x_max

    def _configure_gpu(self, device_id: int) -> int:
        """
        Настройка GPU
        
        Args:
            device_id: ID устройства
        
        Returns:
            Объем памяти GPU в GB
        """
        self.gpu_name = torch.cuda.get_device_name(f"cuda:{device_id}")
        low_end_gpus = ["16", "P40", "P10", "1060", "1070", "1080"]
        if (
            any(gpu in self.gpu_name for gpu in low_end_gpus)
            and "V100" not in self.gpu_name.upper()
        ):
            self.is_half = False
        return int(
            torch.cuda.get_device_properties(self.device).total_memory
            / 1024
            / 1024
            / 1024
            + 0.4
        )


def load_hubert(
    device: torch.device, 
    is_half: bool, 
    model_path: str
) -> nn.Module:
    """
    Загрузить модель Hubert
    
    Args:
        device: Устройство
        is_half: Использовать половинную точность
        model_path: Путь к модели
    
    Returns:
        Модель Hubert
    """
    models, saved_cfg, task = load_model_ensemble_and_task([model_path], suffix="")
    hubert = models[0].to(device)
    hubert = hubert.half() if is_half else hubert.float()
    hubert.eval()
    return hubert


def get_vc(
    device: torch.device, 
    is_half: bool, 
    config: Any, 
    model_path: str, 
    stack: str
) -> Tuple[Dict[str, Any], str, nn.Module, int, VC, int]:
    """
    Загрузить модель VC
    
    Args:
        device: Устройство
        is_half: Использовать половинную точность
        config: Конфигурация
        model_path: Путь к модели
        stack: Стек
    
    Returns:
        Кортеж (cpt, version, net_g, tgt_sr, vc, use_f0)
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"{_i18n('model_not_found')}: {model_path}")

    try:
        cpt = torch.load(model_path, map_location="cpu", weights_only=True)

        required_keys = ["config", "weight"]
        missing_keys = [key for key in required_keys if key not in cpt]

        if missing_keys:
            raise ValueError(
                f"{_i18n('invalid_model_format', model=model_path)}. "
                f"{_i18n('missing_keys')}: {missing_keys}. "
                f"{_i18n('use_rvc_format')}"
            )

        tgt_sr = cpt["config"][-1]

        emb_weight_shape = cpt["weight"]["emb_g.weight"].shape
        cpt["config"][-3] = emb_weight_shape[0]

        use_f0 = cpt.get("f0", 1)
        version = cpt.get("version", "v1")
        vocoder = cpt.get("vocoder", "HiFi-GAN")

        text_enc_hidden_dim = 768 if version == "v2" else 256

        print(f"{_i18n('loading_model')}: {os.path.basename(model_path)}")
        print(f"{_i18n('version')}: {version}, F0: {use_f0}, {_i18n('sample_rate')}: {tgt_sr}Hz")
        print(f"{_i18n('speaker_count')}: {emb_weight_shape[0]}")

        net_g = Synthesizer(
            *cpt["config"],
            use_f0=use_f0,
            text_enc_hidden_dim=text_enc_hidden_dim,
            vocoder=vocoder,
        )

        if hasattr(net_g, "enc_q"):
            del net_g.enc_q
        else:
            print(f"{_i18n('enc_q_warning')}")

        missing_keys, unexpected_keys = net_g.load_state_dict(
            cpt["weight"], strict=False
        )

        if missing_keys:
            print(f"{_i18n('missing_keys_warning')}: {missing_keys}")

        if unexpected_keys:
            print(f"{_i18n('unexpected_keys_warning')}: {unexpected_keys}")

        net_g.eval()

        net_g = net_g.to(device)
        if is_half:
            net_g = net_g.half()
            print(f"{_i18n('half_precision')}")
        else:
            net_g = net_g.float()
            print(f"{_i18n('full_precision')}")

        vc = VC(tgt_sr, config, stack)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"{_i18n('model_loaded', device=str(device))}")

        return cpt, version, net_g, tgt_sr, vc, use_f0

    except torch.serialization.pickle.UnpicklingError as e:
        raise ValueError(
            f"{_i18n('corrupted_model')}: {model_path}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"{_i18n('model_load_error')}: {str(e)}") from e


def rvc_infer(
    index_path: Optional[str],
    index_rate: float,
    input_path: str,
    output_path: str,
    pitch: float,
    f0_method: str,
    cpt: Dict[str, Any],
    version: str,
    net_g: nn.Module,
    filter_radius: int,
    tgt_sr: int,
    volume_envelope: float,
    protect: float,
    hop_length: int,
    vc: VC,
    hubert_model: nn.Module,
    pitch_guidance: bool,
    f0_min: int = 50,
    f0_max: int = 1100,
    format_output: str = "wav",
    output_bitrate: str = "320k",
    stereo_mode: str = "mono",
    pipeline_mode: str = "orig",
    add_text: str = ""
) -> str:
    """
    Инференс RVC
    
    Args:
        index_path: Путь к индексному файлу
        index_rate: Коэффициент влияния индекса
        input_path: Путь к входному файлу
        output_path: Путь к выходному файлу
        pitch: Высота тона
        f0_method: Метод извлечения F0
        cpt: Чекпоинт модели
        version: Версия модели
        net_g: Генератор
        filter_radius: Радиус фильтра
        tgt_sr: Целевая частота дискретизации
        volume_envelope: Огибающая громкости
        protect: Защита согласных
        hop_length: Длина шага
        vc: Объект VC
        hubert_model: Модель Hubert
        pitch_guidance: Использовать направление по высоте тона
        f0_min: Минимальная частота F0
        f0_max: Максимальная частота F0
        format_output: Формат вывода
        output_bitrate: Битрейт
        stereo_mode: Режим стерео
        pipeline_mode: Режим пайплайна
        add_text: Дополнительный текст
    
    Returns:
        Путь к выходному файлу
    """
    if pipeline_mode == "alt":
        pipeline = vc.pipeline2
    else:
        pipeline = vc.pipeline

    mid, left, right = loadaudio(input_path, 16000, stereo_mode)

    if stereo_mode == "mono":
        if mid is None:
            raise ValueError(_i18n("mono_audio_none"))
            
        audio_opt = pipeline(
            hubert_model,
            net_g,
            0,
            mid,
            input_path,
            pitch,
            f0_method,
            index_path,
            index_rate,
            pitch_guidance,
            filter_radius,
            tgt_sr,
            0,
            volume_envelope,
            version,
            protect,
            hop_length,
            f0_file=None,
            f0_min=f0_min,
            f0_max=f0_max,
            add_text=add_text
        )

    elif stereo_mode == "left/right":
        if left is None or right is None:
            raise ValueError(_i18n("stereo_channels_none"))

        leftaudio_opt = pipeline(
            hubert_model,
            net_g,
            0,
            left,
            input_path,
            pitch,
            f0_method,
            index_path,
            index_rate,
            pitch_guidance,
            filter_radius,
            tgt_sr,
            0,
            volume_envelope,
            version,
            protect,
            hop_length,
            f0_file=None,
            f0_min=f0_min,
            f0_max=f0_max,
            add_text=f"{add_text} (L)"
        )
        rightaudio_opt = pipeline(
            hubert_model,
            net_g,
            0,
            right,
            input_path,
            pitch,
            f0_method,
            index_path,
            index_rate,
            pitch_guidance,
            filter_radius,
            tgt_sr,
            0,
            volume_envelope,
            version,
            protect,
            hop_length,
            f0_file=None,
            f0_min=f0_min,
            f0_max=f0_max,
            add_text=f"{add_text} (R)"
        )

        min_len = min(len(leftaudio_opt), len(rightaudio_opt))
        if min_len == 0:
            raise ValueError(_i18n("processed_audio_empty"))
        
        output_dtype = leftaudio_opt.dtype

        leftaudio_opt = trim(leftaudio_opt, 0, min_len)
        rightaudio_opt = trim(rightaudio_opt, 0, min_len)

        audio_opt = multi_channel_array_from_arrays(
            leftaudio_opt, 
            rightaudio_opt, 
            index=1, 
            dtype=output_dtype
        )

    elif stereo_mode == "sim/dif":
        if mid is None or left is None or right is None:
            raise ValueError(_i18n("mid_side_channels_none"))

        midaudio_opt = pipeline(
            hubert_model,
            net_g,
            0,
            mid,
            input_path,
            pitch,
            f0_method,
            index_path,
            index_rate,
            pitch_guidance,
            filter_radius,
            tgt_sr,
            0,
            volume_envelope,
            version,
            protect,
            hop_length,
            f0_file=None,
            f0_min=f0_min,
            f0_max=f0_max,
            add_text=f"{add_text} {_i18n('center')}"
        )
        leftaudio_opt = pipeline(
            hubert_model,
            net_g,
            0,
            left,
            input_path,
            pitch,
            f0_method,
            index_path,
            index_rate,
            pitch_guidance,
            filter_radius,
            tgt_sr,
            0,
            volume_envelope,
            version,
            protect,
            hop_length,
            f0_file=None,
            f0_min=f0_min,
            f0_max=f0_max,
            add_text=f"{add_text} {_i18n('stereo_base')} L"
        )
        rightaudio_opt = pipeline(
            hubert_model,
            net_g,
            0,
            right,
            input_path,
            pitch,
            f0_method,
            index_path,
            index_rate,
            pitch_guidance,
            filter_radius,
            tgt_sr,
            0,
            volume_envelope,
            version,
            protect,
            hop_length,
            f0_file=None,
            f0_min=f0_min,
            f0_max=f0_max,
            add_text=f"{add_text} {_i18n('stereo_base')} R"
        )

        min_len = min(len(midaudio_opt), len(leftaudio_opt), len(rightaudio_opt))
        if min_len == 0:
            raise ValueError(_i18n("processed_audio_empty"))
            
        output_dtype = leftaudio_opt.dtype
        midaudio_opt = trim(midaudio_opt, 0, min_len)
        leftaudio_opt = trim(leftaudio_opt, 0, min_len)
        rightaudio_opt = trim(rightaudio_opt, 0, min_len)
        difaudio_opt = multi_channel_array_from_arrays(
            leftaudio_opt, 
            rightaudio_opt, 
            index=1, 
            dtype=output_dtype
        )
        audio_opt = convert_to_dtype(
            (mono_to_stereo(midaudio_opt, index=1) + difaudio_opt), 
            output_dtype
        )
    else:
        raise ValueError(_i18n("unknown_stereo_mode"))

    output_path = write(
        namer.iter(output_path), audio_opt, tgt_sr, output_bitrate
    )
    return output_path


def load_rvc_model(voice_model: str) -> Tuple[str, Optional[str]]:
    """
    Загрузить RVC модель
    
    Args:
        voice_model: Имя голосовой модели
    
    Returns:
        Кортеж (путь к PTH, путь к индексу)
    """
    if voice_model in model_manager.parse_voice_models():
        rvc_model_path, rvc_index_path = model_manager.parse_pth_and_index(voice_model)

        if not rvc_model_path:
            raise ValueError(
                _i18n("model_file_missing", model=voice_model)
            )
        return rvc_model_path, rvc_index_path
    else:
        raise ValueError(
            _i18n("model_not_found", model=voice_model)
        )


def voice_conversion(
    voice_model: str,
    vocals_path: str,
    output_path: str,
    pitch: float,
    f0_method: str,
    index_rate: float,
    filter_radius: int,
    volume_envelope: float,
    protect: float,
    hop_length: int,
    f0_min: int,
    f0_max: int,
    format_output: str,
    output_bitrate: str,
    stereo_mode: str,
    embedder_name: str = "hubert_base",
    pipeline_mode: str = "orig",
    device: str = "cpu",
    add_text_progress: str = ""
) -> str:
    """
    Голосовое преобразование (fairseq)
    
    Args:
        voice_model: Имя голосовой модели
        vocals_path: Путь к вокалу
        output_path: Путь к выходному файлу
        pitch: Высота тона
        f0_method: Метод извлечения F0
        index_rate: Коэффициент влияния индекса
        filter_radius: Радиус фильтра
        volume_envelope: Огибающая громкости
        protect: Защита согласных
        hop_length: Длина шага
        f0_min: Минимальная частота F0
        f0_max: Максимальная частота F0
        format_output: Формат вывода
        output_bitrate: Битрейт
        stereo_mode: Режим стерео
        embedder_name: Имя эмбеддера
        pipeline_mode: Режим пайплайна
        device: Устройство
        add_text_progress: Дополнительный текст для прогресса
    
    Returns:
        Путь к выходному файлу
    """
    add_text: str = f"| {add_text_progress}" if add_text_progress else ""
    
    rvc_model_path, rvc_index_path = load_rvc_model(voice_model)
    
    progress = gr.Progress()
    progress((0, 4), desc=f"{_i18n('loading_rvc_model')} {add_text}")
    
    config = Config(device)
    progress((1, 4), desc=f"{_i18n('loading_hubert_model')} {add_text}")
    
    hubert_path = model_manager.check_hubert(embedder_name)
    if not hubert_path:
        raise ValueError(
            _i18n("embedder_not_found", embedder=embedder_name)
        )
        
    hubert_model = load_hubert(config.device, config.is_half, hubert_path)
    cpt, version, net_g, tgt_sr, vc, use_f0 = get_vc(
        config.device, config.is_half, config, rvc_model_path, "fairseq"
    )

    outputaudio = rvc_infer(
        rvc_index_path,
        index_rate,
        vocals_path,
        output_path,
        pitch,
        f0_method,
        cpt,
        version,
        net_g,
        filter_radius,
        tgt_sr,
        volume_envelope,
        protect,
        hop_length,
        vc,
        hubert_model,
        use_f0,
        f0_min,
        f0_max,
        format_output,
        output_bitrate,
        stereo_mode,
        pipeline_mode,
        add_text
    )

    del hubert_model, cpt, net_g, vc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return outputaudio


def voice_conversion_transformers(
    voice_model: str,
    vocals_path: str,
    output_path: str,
    pitch: float,
    f0_method: str,
    index_rate: float,
    filter_radius: int,
    volume_envelope: float,
    protect: float,
    hop_length: int,
    f0_min: int,
    f0_max: int,
    format_output: str,
    output_bitrate: str,
    stereo_mode: str,
    embedder_name: str = "contentvec",
    pipeline_mode: str = "orig",
    device: str = "cpu",
    add_text_progress: str = ""
) -> str:
    """
    Голосовое преобразование (transformers)
    
    Args:
        voice_model: Имя голосовой модели
        vocals_path: Путь к вокалу
        output_path: Путь к выходному файлу
        pitch: Высота тона
        f0_method: Метод извлечения F0
        index_rate: Коэффициент влияния индекса
        filter_radius: Радиус фильтра
        volume_envelope: Огибающая громкости
        protect: Защита согласных
        hop_length: Длина шага
        f0_min: Минимальная частота F0
        f0_max: Максимальная частота F0
        format_output: Формат вывода
        output_bitrate: Битрейт
        stereo_mode: Режим стерео
        embedder_name: Имя эмбеддера
        pipeline_mode: Режим пайплайна
        device: Устройство
        add_text_progress: Дополнительный текст для прогресса
    
    Returns:
        Путь к выходному файлу
    """
    add_text: str = f"| {add_text_progress}" if add_text_progress else ""
    
    progress = gr.Progress()
    progress((0, 4), desc=f"{_i18n('loading_rvc_model')} {add_text}")
    
    rvc_model_path, rvc_index_path = load_rvc_model(voice_model)

    config = Config(device)
    progress((1, 4), desc=f"{_i18n('loading_hubert_model')} {add_text}")
    
    hubert_path = model_manager.check_hubert_transformers(embedder_name)
    if not hubert_path:
        raise ValueError(
            _i18n("embedder_not_found", embedder=embedder_name)
        )
        
    hubert_model = HubertModelWithFinalProj.from_pretrained(hubert_path)
    hubert_model = hubert_model.to(config.device)
    cpt, version, net_g, tgt_sr, vc, use_f0 = get_vc(
        config.device, config.is_half, config, rvc_model_path, "transformers"
    )

    outputaudio = rvc_infer(
        rvc_index_path,
        index_rate,
        vocals_path,
        output_path,
        pitch,
        f0_method,
        cpt,
        version,
        net_g,
        filter_radius,
        tgt_sr,
        volume_envelope,
        protect,
        hop_length,
        vc,
        hubert_model,
        use_f0,
        f0_min,
        f0_max,
        format_output,
        output_bitrate,
        stereo_mode,
        pipeline_mode,
        add_text
    )

    del hubert_model, cpt, net_g, vc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return outputaudio


def vbach_inference(
    input_file: str,
    model_name: str,
    output_dir: str,
    output_name: str,
    output_format: str,
    output_bitrate: Union[str, int],
    pitch: int,
    method_pitch: str,
    format_name: bool = False,
    pipeline_mode: str = "orig",
    embedder_name: Optional[str] = "hubert_base",
    stack: str = "fairseq",
    add_params: Dict[str, Any] = {
        "index_rate": 0,
        "filter_radius": 3,
        "protect": 0.33,
        "rms": 0.25,
        "mangio_crepe_hop_length": 128,
        "f0_min": 50,
        "f0_max": 1100,
        "stereo_mode": "mono",
    },
    add_text_progress: str = "",
    device: str = "cpu"
) -> str:
    """
    Основная функция инференса Vbach
    
    Args:
        input_file: Путь к входному файлу
        model_name: Имя модели
        output_dir: Выходная директория
        output_name: Имя выходного файла
        output_format: Формат вывода
        output_bitrate: Битрейт
        pitch: Высота тона
        method_pitch: Метод извлечения F0
        format_name: Форматировать имя
        pipeline_mode: Режим пайплайна
        embedder_name: Имя эмбеддера
        stack: Стек
        add_params: Дополнительные параметры
        add_text_progress: Дополнительный текст для прогресса
        device: Устройство
    
    Returns:
        Путь к выходному файлу
    """
    if stack == "fairseq":
        vbach_convert = voice_conversion
    elif stack == "transformers":
        vbach_convert = voice_conversion_transformers
    else:
        raise ValueError(_i18n("unknown_stack", stack=stack))

    stereo_mode = add_params.get("stereo_mode", "mono")
    index_rate = add_params.get("index_rate", 0)
    filter_radius = add_params.get("filter_radius", 3)
    protect = add_params.get("protect", 0.33)
    rms = add_params.get("rms", 0.25)
    mangio_crepe_hop_length = add_params.get("mangio_crepe_hop_length", 128)
    f0_min = add_params.get("f0_min", 50)
    f0_max = add_params.get("f0_max", 1100)
    
    if not input_file:
        raise ValueError(_i18n("no_input_error"))
    if not os.path.exists(input_file):
        raise ValueError(_i18n("file_not_exists"))
    if not check(input_file):
        raise ValueError(_i18n("file_no_audio"))
        
    basename = os.path.splitext(os.path.basename(input_file))[0]

    final_output_name: Optional[str] = None

    print(_i18n("inference_started"))

    if format_name:
        cleaned_output_name_template = namer.sanitize(
            namer.dedup_template(
                output_name, keys=["NAME", "MODEL", "F0METHOD", "PITCH"]
            )
        )
        short_basename = namer.short_input_name_template(
            cleaned_output_name_template,
            MODEL=model_name,
            F0METHOD=method_pitch,
            PITCH=pitch,
            NAME=basename,
        )
        final_output_name = namer.template(
            cleaned_output_name_template,
            MODEL=model_name,
            F0METHOD=method_pitch,
            PITCH=pitch,
            NAME=short_basename,
        )
    else:
        final_output_name = output_name

    print(f"{_i18n('embedder')}: {embedder_name}")
    print(f"{_i18n('stack')}: {stack}")

    final_output_path = os.path.join(output_dir, f"{final_output_name}.{output_format}")
    
    output_converted_voice = vbach_convert(
        voice_model=model_name,
        vocals_path=input_file,
        output_path=final_output_path,
        pitch=pitch,
        f0_method=method_pitch,
        index_rate=index_rate,
        filter_radius=filter_radius,
        volume_envelope=rms,
        protect=protect,
        hop_length=mangio_crepe_hop_length,
        f0_min=f0_min,
        f0_max=f0_max,
        format_output=output_format,
        output_bitrate=str(output_bitrate),
        stereo_mode=stereo_mode,
        pipeline_mode=pipeline_mode,
        embedder_name=embedder_name,
        device=device,
        add_text_progress=add_text_progress
    )
    
    print(f"{_i18n('inference_complete')}\n{_i18n('output_path')}: \"{output_converted_voice}\"")
    return output_converted_voice


class History:
    """Класс для управления историей преобразований"""
    
    def __init__(self, user_directory: UserDirectory) -> None:
        """
        Инициализация истории
        
        Args:
            user_directory: Пользовательская директория
        """
        self.info: Dict[str, List] = {}
        self.user_directory: UserDirectory = user_directory
        self.path: str = os.path.join(self.user_directory.path, "history", "vbach.json")
        os.makedirs(os.path.join(self.user_directory.path, "history"), exist_ok=True)
        self.load_from_file()
    
    def _save_to_file(func):
        """Декоратор для автоматического сохранения после вызова метода"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self._write_file()
            return result
        return wrapper
    
    def _write_file(self) -> None:
        """Записывает текущее состояние в файл"""
        try:
            dir_path = os.path.dirname(self.path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.info, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"{_i18n('error_writing_file')}: {e}")
    
    @_save_to_file
    def add(
        self, 
        state: List, 
        model_name: str, 
        timestamp: str, 
        f0_method: str, 
        pitch: int
    ) -> None:
        """
        Добавить запись в историю
        
        Args:
            state: Состояние
            model_name: Имя модели
            timestamp: Временная метка
            f0_method: Метод извлечения F0
            pitch: Высота тона
        """
        self.info[f"{timestamp} / {model_name} / {f0_method} / {pitch}"] = state
    
    @_save_to_file
    def clear(self) -> None:
        """Очистить историю"""
        self.info = {}
    
    def get_list(self) -> List[str]:
        """
        Получить список записей истории
        
        Returns:
            Список ключей истории
        """
        return sorted([key for key in self.info], reverse=True)
    
    def get(self, key: str) -> List:
        """
        Получить запись истории по ключу
        
        Args:
            key: Ключ записи
        
        Returns:
            Запись истории
        """
        return self.info.get(key, [])
    
    def load_from_file(self) -> None:
        """Загрузить историю из файла"""
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                self.info = json.load(f)


class Vbach(GradioHelper):
    """Класс для Gradio интерфейса Vbach"""
    
    def __init__(self, user_directory: UserDirectory, device: str) -> None:
        """
        Инициализация Vbach интерфейса
        
        Args:
            user_directory: Пользовательская директория
            device: Устройство
        """
        super().__init__()
        self.device: str = device
        self.pitch_methods: Tuple[str, ...] = f0_methods
        self.hop_length_values: Tuple[int, int] = (8, 512)
        self.index_rates_values: Tuple[int, int] = (0, 1)
        self.filter_radius_values: Tuple[int, int] = (0, 7)
        self.protect_values: Tuple[float, float] = (0, 0.5)
        self.rms_values: Tuple[int, int] = (0, 1)
        self.f0_min_values: Tuple[int, int] = (50, 3000)
        self.f0_max_values: Tuple[int, int] = (300, 6000)
        self.fairseq_embedders: List[str] = list(
            model_manager.huberts_fairseq_dict.keys()
        )
        self.transformers_embedders: List[str] = list(
            model_manager.huberts_transformers_dict.keys()
        )
        self.last_converted_state: List = []
        self.input_files: List[str] = []
        self.user_directory: UserDirectory = user_directory

        model_manager.__init__(self.user_directory)
        self.input_base_dir: str = os.path.join(user_directory.path, "input")
        self.inputs_json_path: str = os.path.join(self.input_base_dir, "inputs.json")
        self.output_base_dir: str = os.path.join(user_directory.path, "output", "vbach")
        self.history: History = History(self.user_directory)
        self.load_from_file()

    def _write_file(self) -> None:
        """Записывает текущее состояние в файл"""
        try:
            with open(self.inputs_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.input_files, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"{_i18n('error_writing_file')}: {e}")

    def _save_to_file(func):
        """Декоратор для автоматического сохранения после вызова метода"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self._write_file()
            return result
        return wrapper

    def load_from_file(self) -> None:
        """Загрузить историю из файла"""
        if os.path.exists(self.inputs_json_path):
            with open(self.inputs_json_path, 'r', encoding='utf-8') as f:
                self.input_files = json.load(f)

    @_save_to_file
    def clean(self) -> None:
        """Очистить список входных файлов"""
        self.input_files = []

    @_save_to_file
    def upload_files(self, input_files: List[str], copy: bool = False) -> List[str]:
        """
        Загрузить файлы в пользовательскую директорию
        
        Args:
            input_files: Список путей к файлам
            copy: Копировать вместо перемещения
        
        Returns:
            Список путей к загруженным файлам
        """
        if input_files:
            input_dir: str = os.path.join(
                self.input_base_dir, 
                datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
            )
            os.makedirs(input_dir, exist_ok=True)
            
            valid_files: List[str] = [file for file in input_files if check(file)]
            valid_files_moved: List[str] = []
            
            if valid_files:
                for file in valid_files:
                    basename: str = os.path.basename(file)
                    output_path: str = os.path.join(input_dir, basename)
                    if copy:
                        shutil.copy(file, output_path)
                    else:
                        shutil.move(file, output_path)
                    valid_files_moved.append(output_path)
                    self.input_files.append(output_path)
            return valid_files_moved
        else:
            return []

    def vbach_convert_batch(
        self,
        input_files: List[str],
        model_name: str,
        pitch_method: str,
        pitch: float,
        hop_length: int,
        index_rate: float,
        filter_radius: int,
        rms: float,
        protect: float,
        f0_min: int,
        f0_max: int,
        output_name: str,
        format_name: bool,
        output_format: str,
        stereo_mode: str,
        alt_pipeline: bool,
        embedder_name: str,
        transformers_mode: bool,
    ) -> Tuple[gr.update, gr.update]:
        output_converted_files: List[str] = []
        progress = gr.Progress(track_tqdm=True)
        progress(progress=0, desc=_i18n("starting_conversion"))
        
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        
        if input_files:
            total_files = len(input_files)
            for i, file in enumerate(input_files, start=1):
                try:
                    print(f"{_i18n('processing_file', current=i, total=total_files, file=file)}")
                    progress(
                        progress=(i / total_files), 
                        desc=_i18n("processing_file_title", current=i, total=total_files)
                    )
                    gr.Warning(
                        title=_i18n("processing_file_title", current=i, total=total_files), 
                        message=file
                    )
                    
                    out_conv = vbach_inference(
                        input_file=file,
                        model_name=model_name,
                        output_dir=os.path.join(self.output_base_dir, timestamp),
                        output_name=output_name,
                        format_name=format_name if total_files == 1 else True,
                        output_format=output_format,
                        pitch=pitch,
                        method_pitch=pitch_method,
                        output_bitrate=320,
                        add_params={
                            "index_rate": index_rate,
                            "filter_radius": filter_radius,
                            "protect": protect,
                            "rms": rms,
                            "mangio_crepe_hop_length": hop_length,
                            "f0_min": f0_min,
                            "f0_max": f0_max,
                            "stereo_mode": stereo_mode,
                        },
                        pipeline_mode="alt" if alt_pipeline else "orig",
                        embedder_name=embedder_name,
                        stack="transformers" if transformers_mode else "fairseq",
                        add_text_progress=f"{i}/{total_files}",
                        device=self.device
                    )
                    output_converted_files.append(out_conv)
                except Exception as e:
                    print(f"{_i18n('error')}: {e}")
                    
        if output_converted_files:
            self.history.add(output_converted_files, model_name, timestamp, pitch_method, pitch)
            
        return gr.update(value=str(output_converted_files)), gr.update(visible=False)

    @hf_spaces_gpu
    def vbach_convert_batch_zero_gpu(
        self,
        input_files: List[str],
        model_name: str,
        pitch_method: str,
        pitch: float,
        hop_length: int,
        index_rate: float,
        filter_radius: int,
        rms: float,
        protect: float,
        f0_min: int,
        f0_max: int,
        output_name: str,
        format_name: bool,
        output_format: str,
        stereo_mode: str,
        alt_pipeline: bool,
        embedder_name: str,
        transformers_mode: bool,
    ) -> Tuple[gr.update, gr.update]:
        output_converted_files: List[str] = []
        progress = gr.Progress(track_tqdm=True)
        progress(progress=0, desc=_i18n("starting_conversion"))
        
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        
        if input_files:
            total_files = len(input_files)
            for i, file in enumerate(input_files, start=1):
                try:
                    print(f"{_i18n('processing_file', current=i, total=total_files, file=file)}")
                    progress(
                        progress=(i / total_files), 
                        desc=_i18n("processing_file_title", current=i, total=total_files)
                    )
                    gr.Warning(
                        title=_i18n("processing_file_title", current=i, total=total_files), 
                        message=file
                    )
                    
                    out_conv = vbach_inference(
                        input_file=file,
                        model_name=model_name,
                        output_dir=os.path.join(self.output_base_dir, timestamp),
                        output_name=output_name,
                        format_name=format_name if total_files == 1 else True,
                        output_format=output_format,
                        pitch=pitch,
                        method_pitch=pitch_method,
                        output_bitrate=320,
                        add_params={
                            "index_rate": index_rate,
                            "filter_radius": filter_radius,
                            "protect": protect,
                            "rms": rms,
                            "mangio_crepe_hop_length": hop_length,
                            "f0_min": f0_min,
                            "f0_max": f0_max,
                            "stereo_mode": stereo_mode,
                        },
                        pipeline_mode="alt" if alt_pipeline else "orig",
                        embedder_name=embedder_name,
                        stack="transformers" if transformers_mode else "fairseq",
                        add_text_progress=f"{i}/{total_files}",
                        device="cuda:0"
                    )
                    output_converted_files.append(out_conv)
                except Exception as e:
                    print(f"{_i18n('error')}: {e}")
                    
        if output_converted_files:
            self.history.add(output_converted_files, model_name, timestamp, pitch_method, pitch)
            
        return gr.update(value=str(output_converted_files)), gr.update(visible=False)

    def vbach_convert_duet(
        self,
        input_file: Optional[str],
        model_name1: str,
        model_name2: str,
        pitch_method1: str,
        pitch_method2: str,
        pitch1: float,
        pitch2: float,
        hop_length1: int,
        hop_length2: int,
        index_rate1: float,
        index_rate2: float,
        filter_radius1: int,
        filter_radius2: int,
        rms1: float,
        rms2: float,
        protect1: float,
        protect2: float,
        f0_min1: int,
        f0_min2: int,
        f0_max1: int,
        f0_max2: int,
        output_format: str,
        stereo_mode: str,
        alt_pipeline: bool,
        embedder_name1: str,
        embedder_name2: str,
        transformers_mode1: bool,
        transformers_mode2: bool,
        mix_duet: bool,
        mix_duet_ratio: float
    ) -> Tuple[Optional[Dict], Optional[Dict], gr.update]:
        output_1: Optional[str] = None
        output_2: Optional[str] = None
        
        progress = gr.Progress(track_tqdm=True)
        progress(progress=0, desc=_i18n("starting_conversion"))
        
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(self.output_base_dir, timestamp)
        
        if input_file:
            try:
                gr.Warning(title=_i18n("model_1"), message="")
                output_1 = vbach_inference(
                    input_file=input_file,
                    model_name=model_name1,
                    output_dir=output_dir,
                    output_name="NAME - MODEL 1 - F0METHOD - PITCH",
                    format_name=True,
                    output_format=output_format,
                    pitch=pitch1,
                    method_pitch=pitch_method1,
                    output_bitrate=320,
                    add_params={
                        "index_rate": index_rate1,
                        "filter_radius": filter_radius1,
                        "protect": protect1,
                        "rms": rms1,
                        "mangio_crepe_hop_length": hop_length1,
                        "f0_min": f0_min1,
                        "f0_max": f0_max1,
                        "stereo_mode": stereo_mode,
                    },
                    pipeline_mode="alt" if alt_pipeline else "orig",
                    embedder_name=embedder_name1,
                    stack="transformers" if transformers_mode1 else "fairseq",
                    add_text_progress=_i18n("model_1"),
                    device=self.device
                )
                
                gr.Warning(title=_i18n("model_2"), message="")
                output_2 = vbach_inference(
                    input_file=input_file,
                    model_name=model_name2,
                    output_dir=output_dir,
                    output_name="NAME - MODEL 2 - F0METHOD - PITCH",
                    format_name=True,
                    output_format=output_format,
                    pitch=pitch2,
                    method_pitch=pitch_method2,
                    output_bitrate=320,
                    add_params={
                        "index_rate": index_rate2,
                        "filter_radius": filter_radius2,
                        "protect": protect2,
                        "rms": rms2,
                        "mangio_crepe_hop_length": hop_length2,
                        "f0_min": f0_min2,
                        "f0_max": f0_max2,
                        "stereo_mode": stereo_mode,
                    },
                    pipeline_mode="alt" if alt_pipeline else "orig",
                    embedder_name=embedder_name2,
                    stack="transformers" if transformers_mode2 else "fairseq",
                    add_text_progress=_i18n("model_2"),
                    device=self.device
                )

            except Exception as e:
                print(f"{_i18n('error')}: {e}")
                return (
                    gr.update(value=None), 
                    gr.update(value=None)
                )

        if mix_duet and output_1 and output_2:
            input_file_basename = os.path.splitext(os.path.basename(input_file))[0] if input_file else "duet"
            mix1, sr1 = read(output_1)
            mix2, sr2 = read(output_2)
            max_sr = max(sr1, sr2)
            fitted_arrays = fit_arrays([mix1, mix2], [sr1, sr2], min_sr=max_sr)
            g1 = (1 - mix_duet_ratio) / 2
            g2 = (1 + mix_duet_ratio) / 2
            mixed_duet = gain(fitted_arrays[0], g1) + gain(fitted_arrays[1], g2)
            shorted_name = namer.short(input_file_basename, length=50)
            sanitized_name = namer.sanitize(f"{model_name1}, {model_name2} - {shorted_name}")
            output_mixed = write(
                os.path.join(output_dir, f"{sanitized_name}.{output_format}"), 
                mixed_duet, 
                max_sr
            )
            self.history.add(
                [output_mixed], 
                f"{model_name1}|{model_name2}", 
                timestamp, 
                f"{pitch_method1}|{pitch_method2}", 
                f"{pitch1}|{pitch2}"
            )
            return (
                self.return_audio_with_size(label=_i18n("mixed_result"), value=output_mixed),
                gr.update(label=_i18n("model_2_result"), value=None),
            )
        elif output_1 and output_2:
            self.history.add(
                [output_1, output_2], 
                f"{model_name1}|{model_name2}", 
                timestamp, 
                f"{pitch_method1}|{pitch_method2}", 
                f"{pitch1}|{pitch2}"
            )
            return (
                self.return_audio_with_size(label=_i18n("model_1_result"), value=output_1),
                self.return_audio_with_size(label=_i18n("model_2_result"), value=output_2),
            )
        else:
            return (
                gr.update(value=None), 
                gr.update(value=None)
            )

    def vbach_convert_duet_zero_gpu(
        self,
        input_file: Optional[str],
        model_name1: str,
        model_name2: str,
        pitch_method1: str,
        pitch_method2: str,
        pitch1: float,
        pitch2: float,
        hop_length1: int,
        hop_length2: int,
        index_rate1: float,
        index_rate2: float,
        filter_radius1: int,
        filter_radius2: int,
        rms1: float,
        rms2: float,
        protect1: float,
        protect2: float,
        f0_min1: int,
        f0_min2: int,
        f0_max1: int,
        f0_max2: int,
        output_format: str,
        stereo_mode: str,
        alt_pipeline: bool,
        embedder_name1: str,
        embedder_name2: str,
        transformers_mode1: bool,
        transformers_mode2: bool,
        mix_duet: bool,
        mix_duet_ratio: float
    ) -> Tuple[Optional[Dict], Optional[Dict], gr.update]:
        output_1: Optional[str] = None
        output_2: Optional[str] = None
        
        progress = gr.Progress(track_tqdm=True)
        progress(progress=0, desc=_i18n("starting_conversion"))
        
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(self.output_base_dir, timestamp)
        
        if input_file:
            try:
                gr.Warning(title=_i18n("model_1"), message="")
                output_1 = vbach_inference(
                    input_file=input_file,
                    model_name=model_name1,
                    output_dir=output_dir,
                    output_name="NAME - MODEL 1 - F0METHOD - PITCH",
                    format_name=True,
                    output_format=output_format,
                    pitch=pitch1,
                    method_pitch=pitch_method1,
                    output_bitrate=320,
                    add_params={
                        "index_rate": index_rate1,
                        "filter_radius": filter_radius1,
                        "protect": protect1,
                        "rms": rms1,
                        "mangio_crepe_hop_length": hop_length1,
                        "f0_min": f0_min1,
                        "f0_max": f0_max1,
                        "stereo_mode": stereo_mode,
                    },
                    pipeline_mode="alt" if alt_pipeline else "orig",
                    embedder_name=embedder_name1,
                    stack="transformers" if transformers_mode1 else "fairseq",
                    add_text_progress=_i18n("model_1"),
                    device="cuda:0"
                )
                
                gr.Warning(title=_i18n("model_2"), message="")
                output_2 = vbach_inference(
                    input_file=input_file,
                    model_name=model_name2,
                    output_dir=output_dir,
                    output_name="NAME - MODEL 2 - F0METHOD - PITCH",
                    format_name=True,
                    output_format=output_format,
                    pitch=pitch2,
                    method_pitch=pitch_method2,
                    output_bitrate=320,
                    add_params={
                        "index_rate": index_rate2,
                        "filter_radius": filter_radius2,
                        "protect": protect2,
                        "rms": rms2,
                        "mangio_crepe_hop_length": hop_length2,
                        "f0_min": f0_min2,
                        "f0_max": f0_max2,
                        "stereo_mode": stereo_mode,
                    },
                    pipeline_mode="alt" if alt_pipeline else "orig",
                    embedder_name=embedder_name2,
                    stack="transformers" if transformers_mode2 else "fairseq",
                    add_text_progress=_i18n("model_2"),
                    device="cuda:0"
                )

            except Exception as e:
                print(f"{_i18n('error')}: {e}")
                return (
                    gr.update(value=None), 
                    gr.update(value=None)
                )

        if mix_duet and output_1 and output_2:
            input_file_basename = os.path.splitext(os.path.basename(input_file))[0] if input_file else "duet"
            mix1, sr1 = read(output_1)
            mix2, sr2 = read(output_2)
            max_sr = max(sr1, sr2)
            fitted_arrays = fit_arrays([mix1, mix2], [sr1, sr2], min_sr=max_sr)
            g1 = (1 - mix_duet_ratio) / 2
            g2 = (1 + mix_duet_ratio) / 2
            mixed_duet = gain(fitted_arrays[0], g1) + gain(fitted_arrays[1], g2)
            shorted_name = namer.short(input_file_basename, length=50)
            sanitized_name = namer.sanitize(f"{model_name1}, {model_name2} - {shorted_name}")
            output_mixed = write(
                os.path.join(output_dir, f"{sanitized_name}.{output_format}"), 
                mixed_duet, 
                max_sr
            )
            self.history.add(
                [output_mixed], 
                f"{model_name1}|{model_name2}", 
                timestamp, 
                f"{pitch_method1}|{pitch_method2}", 
                f"{pitch1}|{pitch2}"
            )
            return (
                self.return_audio_with_size(label=_i18n("mixed_result"), value=output_mixed),
                gr.update(label=_i18n("model_2_result"), value=None),
            )
        elif output_1 and output_2:
            self.history.add(
                [output_1, output_2], 
                f"{model_name1}|{model_name2}", 
                timestamp, 
                f"{pitch_method1}|{pitch_method2}", 
                f"{pitch1}|{pitch2}"
            )
            return (
                self.return_audio_with_size(label=_i18n("model_1_result"), value=output_1),
                self.return_audio_with_size(label=_i18n("model_2_result"), value=output_2),
            )
        else:
            return (
                gr.update(value=None), 
                gr.update(value=None)
            )

    def UI(self) -> gr.Blocks:
        """
        Создать пользовательский интерфейс
        
        Returns:
            Блоки интерфейса Gradio
        """
        with gr.Blocks() as vbach_app:
            with gr.Tab(_i18n("tab_inference")):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            upload = gr.Files(
                                show_label=False, 
                                type="filepath", 
                                interactive=True
                            )
                            refresh_input_btn = gr.Button(
                                _i18n("refresh"), 
                                variant="primary", 
                                interactive=True
                            )
                            list_input_files = gr.Dropdown(
                                label=_i18n("select_input_files"),
                                choices=reversed(self.input_files) if self.input_files else [],
                                value=[],
                                multiselect=True,
                                interactive=True,
                                filterable=False, 
                                scale=15
                            )
                            
                            gr.on(
                                fn=lambda: gr.update(choices=reversed(self.input_files) if self.input_files else [], value=[]), 
                                outputs=list_input_files, 
                                trigger_mode="once"
                            )
                            
                            refresh_input_btn.click(
                                lambda: gr.update(choices=reversed(self.input_files) if self.input_files else [], value=[]), 
                                outputs=list_input_files
                            )
                                
                            @upload.upload(inputs=[upload], outputs=[list_input_files, upload])
                            def upload_files(input_files: List[str]) -> Tuple[gr.update, gr.update]:
                                files = self.upload_files(input_files)
                                return (
                                    gr.update(choices=reversed(self.input_files) if self.input_files else [], value=files),
                                    gr.update(value=[])
                                )
                                
                            converted_state = gr.Textbox(
                                label=_i18n("conversion_status"),
                                interactive=False,
                                value="",
                                visible=False,
                            )

                    with gr.Column():
                        with gr.Group():
                            with gr.Group():
                                model_name = gr.Dropdown(
                                    label=_i18n("model_name"), 
                                    interactive=True
                                )
                                model_list_refresh_btn = gr.Button(
                                    _i18n("refresh"), 
                                    variant="secondary", 
                                    interactive=True
                                )

                                @model_list_refresh_btn.click(outputs=[model_name])
                                def refresh_list_voice_models() -> gr.update:
                                    models = model_manager.parse_voice_models()
                                    first_model = models[0] if models else None
                                    return gr.update(choices=models, value=first_model)

                            with gr.Group():
                                pitch_method = gr.Dropdown(
                                    label=_i18n("f0_method"),
                                    choices=self.pitch_methods,
                                    value=self.pitch_methods[0] if self.pitch_methods else "rmvpe+",
                                    interactive=True,
                                    filterable=False
                                )
                                pitch = gr.Slider(
                                    label=_i18n("pitch"),
                                    minimum=-48,
                                    maximum=48,
                                    step=0.5,
                                    value=0,
                                    interactive=True,
                                )
                                hop_length = gr.Slider(
                                    label=_i18n("hop_length"),
                                    info=_i18n("hop_length_info"),
                                    minimum=self.hop_length_values[0],
                                    maximum=self.hop_length_values[1],
                                    step=8,
                                    value=128,
                                    interactive=True,
                                    visible=False,
                                )

                                @pitch_method.change(
                                    inputs=[pitch_method], outputs=[hop_length]
                                )
                                def show_mangio_crepe_hop_length(pitch_method: str) -> gr.update:
                                    return gr.update(
                                        visible=(
                                            pitch_method
                                            in ["mangio-crepe", "mangio-crepe-tiny", "pyin"]
                                        )
                                    )

                            with gr.Accordion(label=_i18n("additional_settings"), open=False):
                                with gr.Group():
                                    with gr.Accordion(label=_i18n("audio_processing"), open=False):
                                        with gr.Group():
                                            stereo_mode = gr.Radio(
                                                choices=["mono", "left/right", "sim/dif"],
                                                label=_i18n("stereo_mode"),
                                                info=_i18n("stereo_mode_info"),
                                                value="mono",
                                                interactive=True,
                                            )
                                            alt_pl = gr.Checkbox(
                                                label=_i18n("alt_pipeline"),
                                                info=_i18n("alt_pipeline_info"),
                                                value=False,
                                                interactive=True,
                                            )
                                    with gr.Accordion(label=_i18n("inference"), open=False):
                                        with gr.Group():
                                            with gr.Row():
                                                index_rate = gr.Slider(
                                                    label=_i18n("index_rate"),
                                                    info=_i18n("index_rate_info"),
                                                    minimum=self.index_rates_values[0],
                                                    maximum=self.index_rates_values[1],
                                                    step=0.05,
                                                    value=0,
                                                    interactive=True,
                                                )
                                                filter_radius = gr.Slider(
                                                    label=_i18n("filter_radius"),
                                                    info=_i18n("filter_radius_info"),
                                                    minimum=self.filter_radius_values[0],
                                                    maximum=self.filter_radius_values[1],
                                                    step=1,
                                                    value=3,
                                                    interactive=True,
                                                )
                                            with gr.Row():
                                                rms = gr.Slider(
                                                    label=_i18n("rms_envelope"),
                                                    info=_i18n("rms_info"),
                                                    minimum=self.rms_values[0],
                                                    maximum=self.rms_values[1],
                                                    step=0.05,
                                                    value=0.25,
                                                    interactive=True,
                                                )
                                                protect = gr.Slider(
                                                    label=_i18n("protect"),
                                                    info=_i18n("protect_info"),
                                                    minimum=self.protect_values[0],
                                                    maximum=self.protect_values[1],
                                                    step=0.05,
                                                    value=0.35,
                                                    interactive=True,
                                                )
                                    with gr.Accordion(label=_i18n("f0_range"), open=False):
                                        with gr.Group():
                                            with gr.Row():
                                                f0_min = gr.Slider(
                                                    label=_i18n("f0_min"),
                                                    minimum=self.f0_min_values[0],
                                                    maximum=self.f0_min_values[1],
                                                    step=10,
                                                    value=50,
                                                    interactive=True,
                                                )
                                                f0_max = gr.Slider(
                                                    label=_i18n("f0_max"),
                                                    minimum=self.f0_max_values[0],
                                                    maximum=self.f0_max_values[1],
                                                    step=10,
                                                    value=1100,
                                                    interactive=True,
                                                )
                                    with gr.Accordion(label=_i18n("embedder"), open=False):
                                        with gr.Group():
                                            embedder_name = gr.Radio(
                                                label=_i18n("hubert_model"),
                                                choices=self.fairseq_embedders,
                                                value=self.fairseq_embedders[0] if self.fairseq_embedders else "hubert_base",
                                            )
                                            transformers_mode = gr.Checkbox(
                                                label=_i18n("use_transformers"),
                                                value=False,
                                                interactive=True,
                                            )

                                        @transformers_mode.change(
                                            inputs=[transformers_mode], outputs=[embedder_name]
                                        )
                                        def change_embedders(tr_m: bool) -> gr.update:
                                            if tr_m:
                                                return gr.update(
                                                    value=self.transformers_embedders[0] if self.transformers_embedders else None,
                                                    choices=self.transformers_embedders,
                                                )
                                            else:
                                                return gr.update(
                                                    choices=self.fairseq_embedders,
                                                    value=self.fairseq_embedders[0] if self.fairseq_embedders else None,
                                                )

                                    with gr.Accordion(label=_i18n("output_filename"), open=False):
                                        with gr.Group():
                                            output_name = gr.Textbox(
                                                label=_i18n("output_filename"),
                                                interactive=True,
                                                value="NAME - MODEL - F0METHOD - PITCH",
                                            )
                                            format_output_name_check = gr.Checkbox(
                                                label=_i18n("format_name"),
                                                info=_i18n("format_name_info"),
                                                value=True,
                                                interactive=True,
                                            )

                            with gr.Group():
                                output_format = gr.Dropdown(
                                    label=_i18n("output_format"),
                                    interactive=True,
                                    choices=output_formats,
                                    value=output_formats[0] if output_formats else "wav",
                                    filterable=False,
                                )
                                status = gr.Textbox(
                                    container=False, 
                                    lines=4, 
                                    interactive=False, 
                                    max_lines=4, 
                                    visible=False
                                )
                                convert_btn = gr.Button(
                                    _i18n("convert_btn"), 
                                    variant="primary", 
                                    interactive=True
                                ).click(
                                    lambda: gr.update(visible=True), 
                                    outputs=[status]
                                )
                                
                @convert_btn.then(
                    inputs=[
                        list_input_files,
                        model_name,
                        pitch_method,
                        pitch,
                        hop_length,
                        index_rate,
                        filter_radius,
                        rms,
                        protect,
                        f0_min,
                        f0_max,
                        output_name,
                        format_output_name_check,
                        output_format,
                        stereo_mode,
                        alt_pl,
                        embedder_name,
                        transformers_mode,
                    ],
                    outputs=[converted_state, status],
                    queue=True
                )
                def vbach_convert_batch_fn(
                    input_files: List[str],
                    model_name: str,
                    pitch_method: str,
                    pitch: float,
                    hop_length: int,
                    index_rate: float,
                    filter_radius: int,
                    rms: float,
                    protect: float,
                    f0_min: int,
                    f0_max: int,
                    output_name: str,
                    format_name: bool,
                    output_format: str,
                    stereo_mode: str,
                    alt_pipeline: bool,
                    embedder_name: str,
                    transformers_mode: bool,
                ) -> Tuple[gr.update, gr.update]:

                    vbach_batch = self.vbach_convert_batch_zero_gpu if zerogpu_available else self.vbach_convert_batch
                    return vbach_batch(
                        input_files=input_files,
                        model_name=model_name,
                        pitch_method=pitch_method,
                        pitch=pitch,
                        hop_length=hop_length,
                        index_rate=index_rate,
                        filter_radius=filter_radius,
                        rms=rms,
                        protect=protect,
                        f0_min=f0_min,
                        f0_max=f0_max,
                        output_name=output_name,
                        format_name=format_name,
                        output_format=output_format,
                        stereo_mode=stereo_mode,
                        alt_pipeline=alt_pipeline,
                        embedder_name=embedder_name,
                        transformers_mode=transformers_mode
                    )



                with gr.Column(variant="panel"):
                    gr.Markdown(f"<center><h3>{_i18n('results')}</h3></center>")

                    with gr.Group():
                        with gr.Row(equal_height=True):
                            list_conversions = gr.Dropdown(
                                label=_i18n("select_conversion_results"),
                                choices=[],
                                value=None,
                                interactive=True, 
                                scale=14
                            )
                            
                            list_conversions.change(
                                lambda x: gr.update(value=str(self.history.get(x))), 
                                inputs=[list_conversions], 
                                outputs=[converted_state]
                            )
                            
                            refresh_conversions_btn = gr.Button(
                                _i18n("refresh"), 
                                scale=2, 
                                interactive=True
                            )
                            refresh_conversions_btn.click(
                                lambda: gr.update(choices=self.history.get_list(), value=None), 
                                outputs=[list_conversions]
                            )
                            
                            gr.on(
                                fn=lambda: gr.update(choices=self.history.get_list(), value=None), 
                                outputs=[list_conversions]
                            )

                @gr.render(inputs=[converted_state])
                def show_players_converted(state: str) -> None:
                    if state:
                        try:
                            output_converted_files = ast.literal_eval(state)
                            if output_converted_files:
                                with gr.Group():
                                    for conv_file in output_converted_files:
                                        basename = os.path.splitext(
                                            os.path.basename(conv_file)
                                        )[0]
                                        self.define_audio_with_size(
                                            label=basename,
                                            value=conv_file,
                                            type="filepath",
                                            interactive=False,
                                            show_download_button=True,
                                        )
                        except:
                            pass

            with gr.TabItem(_i18n("tab_duet")):
                with gr.Column():
                    with gr.Group():
                        upload_duet = gr.File(
                            show_label=False, 
                            type="filepath", 
                            interactive=True
                        )
                        refresh_input_btn_duet = gr.Button(
                            _i18n("refresh"), 
                            variant="primary", 
                            interactive=True
                        )
                        list_input_files_duet = gr.Dropdown(
                            label=_i18n("select_input_files"),
                            choices=self.input_files,
                            value=None,
                            multiselect=False,
                            interactive=True,
                            filterable=False, 
                            scale=15
                        )
                        
                        gr.on(
                            fn=lambda: gr.update(choices=reversed(self.input_files) if self.input_files else [], value=None), 
                            outputs=list_input_files_duet, 
                            trigger_mode="once"
                        )
                        
                        refresh_input_btn_duet.click(
                            lambda: gr.update(choices=reversed(self.input_files) if self.input_files else [], value=None), 
                            outputs=list_input_files_duet
                        )
                            
                        @upload_duet.upload(
                            inputs=[upload_duet], 
                            outputs=[list_input_files_duet, upload_duet]
                        )
                        def upload_files(input_file: str) -> Tuple[gr.update, gr.update]:
                            files = self.upload_files([input_file])
                            return (
                                gr.update(choices=reversed(self.input_files) if self.input_files else [], value=files[0] if files else None),
                                gr.update(value=None)
                            )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(f"<h3><center>{_i18n('model')} 1</center></h3>")
                            with gr.Group():
                                model_name1 = gr.Dropdown(
                                    label=_i18n("model_name"), 
                                    interactive=True
                                )

                                pitch_method1 = gr.Dropdown(
                                    label=_i18n("f0_method"),
                                    choices=self.pitch_methods,
                                    value=self.pitch_methods[0] if self.pitch_methods else "rmvpe+",
                                    interactive=True,
                                    filterable=False
                                )
                                pitch1 = gr.Slider(
                                    label=_i18n("pitch"),
                                    minimum=-48,
                                    maximum=48,
                                    step=0.5,
                                    value=0,
                                    interactive=True,
                                )
                                hop_length1 = gr.Slider(
                                    label=_i18n("hop_length"),
                                    info=_i18n("hop_length_info"),
                                    minimum=self.hop_length_values[0],
                                    maximum=self.hop_length_values[1],
                                    step=8,
                                    value=128,
                                    interactive=True,
                                    visible=False,
                                )

                                @pitch_method1.change(
                                    inputs=[pitch_method1], outputs=[hop_length1]
                                )
                                def show_mangio_crepe_hop_length(pitch_method: str) -> gr.update:
                                    return gr.update(
                                        visible=(
                                            pitch_method
                                            in ["mangio-crepe", "mangio-crepe-tiny", "pyin"]
                                        )
                                    )

                                with gr.Accordion(label=_i18n("additional_settings"), open=False):
                                    with gr.Group():
                                        with gr.Accordion(label=_i18n("inference"), open=False):
                                            with gr.Group():
                                                with gr.Row():
                                                    index_rate1 = gr.Slider(
                                                        label=_i18n("index_rate"),
                                                        info=_i18n("index_rate_info"),
                                                        minimum=self.index_rates_values[0],
                                                        maximum=self.index_rates_values[1],
                                                        step=0.05,
                                                        value=0,
                                                        interactive=True,
                                                    )
                                                    filter_radius1 = gr.Slider(
                                                        label=_i18n("filter_radius"),
                                                        info=_i18n("filter_radius_info"),
                                                        minimum=self.filter_radius_values[0],
                                                        maximum=self.filter_radius_values[1],
                                                        step=1,
                                                        value=3,
                                                        interactive=True,
                                                    )
                                                with gr.Row():
                                                    rms1 = gr.Slider(
                                                        label=_i18n("rms_envelope"),
                                                        info=_i18n("rms_info"),
                                                        minimum=self.rms_values[0],
                                                        maximum=self.rms_values[1],
                                                        step=0.05,
                                                        value=0.25,
                                                        interactive=True,
                                                    )
                                                    protect1 = gr.Slider(
                                                        label=_i18n("protect"),
                                                        info=_i18n("protect_info"),
                                                        minimum=self.protect_values[0],
                                                        maximum=self.protect_values[1],
                                                        step=0.05,
                                                        value=0.35,
                                                        interactive=True,
                                                    )
                                        with gr.Accordion(label=_i18n("f0_range"), open=False):
                                            with gr.Group():
                                                with gr.Row():
                                                    f0_min1 = gr.Slider(
                                                        label=_i18n("f0_min"),
                                                        minimum=self.f0_min_values[0],
                                                        maximum=self.f0_min_values[1],
                                                        step=10,
                                                        value=50,
                                                        interactive=True,
                                                    )
                                                    f0_max1 = gr.Slider(
                                                        label=_i18n("f0_max"),
                                                        minimum=self.f0_max_values[0],
                                                        maximum=self.f0_max_values[1],
                                                        step=10,
                                                        value=1100,
                                                        interactive=True,
                                                    )
                                        with gr.Accordion(label=_i18n("embedder"), open=False):
                                            with gr.Group():
                                                embedder_name1 = gr.Radio(
                                                    label=_i18n("hubert_model"),
                                                    choices=self.fairseq_embedders,
                                                    value=self.fairseq_embedders[0] if self.fairseq_embedders else "hubert_base",
                                                )
                                                transformers_mode1 = gr.Checkbox(
                                                    label=_i18n("use_transformers"),
                                                    value=False,
                                                    interactive=True,
                                                )

                                            @transformers_mode1.change(
                                                inputs=[transformers_mode1], 
                                                outputs=[embedder_name1]
                                            )
                                            def change_embedders(tr_m: bool) -> gr.update:
                                                if tr_m:
                                                    return gr.update(
                                                        value=self.transformers_embedders[0] if self.transformers_embedders else None,
                                                        choices=self.transformers_embedders,
                                                    )
                                                else:
                                                    return gr.update(
                                                        choices=self.fairseq_embedders,
                                                        value=self.fairseq_embedders[0] if self.fairseq_embedders else None,
                                                    )

                        with gr.Column():
                            gr.Markdown(f"<h3><center>{_i18n('model')} 2</center></h3>")
                            with gr.Group():
                                model_name2 = gr.Dropdown(
                                    label=_i18n("model_name"), 
                                    interactive=True
                                )

                                pitch_method2 = gr.Dropdown(
                                    label=_i18n("f0_method"),
                                    choices=self.pitch_methods,
                                    value=self.pitch_methods[0] if self.pitch_methods else "rmvpe+",
                                    interactive=True,
                                    filterable=False
                                )
                                pitch2 = gr.Slider(
                                    label=_i18n("pitch"),
                                    minimum=-48,
                                    maximum=48,
                                    step=0.5,
                                    value=0,
                                    interactive=True,
                                )
                                hop_length2 = gr.Slider(
                                    label=_i18n("hop_length"),
                                    info=_i18n("hop_length_info"),
                                    minimum=self.hop_length_values[0],
                                    maximum=self.hop_length_values[1],
                                    step=8,
                                    value=128,
                                    interactive=True,
                                    visible=False,
                                )

                                @pitch_method2.change(
                                    inputs=[pitch_method2], outputs=[hop_length2]
                                )
                                def show_mangio_crepe_hop_length(pitch_method: str) -> gr.update:
                                    return gr.update(
                                        visible=(
                                            pitch_method
                                            in ["mangio-crepe", "mangio-crepe-tiny", "pyin"]
                                        )
                                    )

                                with gr.Accordion(label=_i18n("additional_settings"), open=False):
                                    with gr.Group():
                                        with gr.Accordion(label=_i18n("inference"), open=False):
                                            with gr.Group():
                                                with gr.Row():
                                                    index_rate2 = gr.Slider(
                                                        label=_i18n("index_rate"),
                                                        info=_i18n("index_rate_info"),
                                                        minimum=self.index_rates_values[0],
                                                        maximum=self.index_rates_values[1],
                                                        step=0.05,
                                                        value=0,
                                                        interactive=True,
                                                    )
                                                    filter_radius2 = gr.Slider(
                                                        label=_i18n("filter_radius"),
                                                        info=_i18n("filter_radius_info"),
                                                        minimum=self.filter_radius_values[0],
                                                        maximum=self.filter_radius_values[1],
                                                        step=1,
                                                        value=3,
                                                        interactive=True,
                                                    )
                                                with gr.Row():
                                                    rms2 = gr.Slider(
                                                        label=_i18n("rms_envelope"),
                                                        info=_i18n("rms_info"),
                                                        minimum=self.rms_values[0],
                                                        maximum=self.rms_values[1],
                                                        step=0.05,
                                                        value=0.25,
                                                        interactive=True,
                                                    )
                                                    protect2 = gr.Slider(
                                                        label=_i18n("protect"),
                                                        info=_i18n("protect_info"),
                                                        minimum=self.protect_values[0],
                                                        maximum=self.protect_values[1],
                                                        step=0.05,
                                                        value=0.35,
                                                        interactive=True,
                                                    )
                                        with gr.Accordion(label=_i18n("f0_range"), open=False):
                                            with gr.Group():
                                                with gr.Row():
                                                    f0_min2 = gr.Slider(
                                                        label=_i18n("f0_min"),
                                                        minimum=self.f0_min_values[0],
                                                        maximum=self.f0_min_values[1],
                                                        step=10,
                                                        value=50,
                                                        interactive=True,
                                                    )
                                                    f0_max2 = gr.Slider(
                                                        label=_i18n("f0_max"),
                                                        minimum=self.f0_max_values[0],
                                                        maximum=self.f0_max_values[1],
                                                        step=10,
                                                        value=1100,
                                                        interactive=True,
                                                    )
                                        with gr.Accordion(label=_i18n("embedder"), open=False):
                                            with gr.Group():
                                                embedder_name2 = gr.Radio(
                                                    label=_i18n("hubert_model"),
                                                    choices=self.fairseq_embedders,
                                                    value=self.fairseq_embedders[0] if self.fairseq_embedders else "hubert_base",
                                                )
                                                transformers_mode2 = gr.Checkbox(
                                                    label=_i18n("use_transformers"),
                                                    value=False,
                                                    interactive=True,
                                                )

                                            @transformers_mode2.change(
                                                inputs=[transformers_mode2], 
                                                outputs=[embedder_name2]
                                            )
                                            def change_embedders(tr_m: bool) -> gr.update:
                                                if tr_m:
                                                    return gr.update(
                                                        value=self.transformers_embedders[0] if self.transformers_embedders else None,
                                                        choices=self.transformers_embedders,
                                                    )
                                                else:
                                                    return gr.update(
                                                        choices=self.fairseq_embedders,
                                                        value=self.fairseq_embedders[0] if self.fairseq_embedders else None,
                                                    )

                    with gr.Group():
                        model_list_refresh_btn = gr.Button(
                            _i18n("refresh_models"), 
                            variant="secondary", 
                            interactive=True
                        )
                        
                        @model_list_refresh_btn.click(outputs=[model_name1, model_name2])
                        def refresh_list_voice_models() -> Tuple[gr.update, gr.update]:
                            models = model_manager.parse_voice_models()
                            first_model = models[0] if models else None
                            return (
                                gr.update(choices=models, value=first_model), 
                                gr.update(choices=models, value=first_model)
                            )
                            
                        stereo_mode_duet = gr.Radio(
                            choices=["mono", "left/right", "sim/dif"],
                            label=_i18n("stereo_mode"),
                            info=_i18n("stereo_mode_info"),
                            value="mono",
                            interactive=True,
                        )
                        alt_pl_duet = gr.Checkbox(
                            label=_i18n("alt_pipeline"),
                            info=_i18n("alt_pipeline_info"),
                            value=False,
                            interactive=True,
                        )
                        mix_duet = gr.Checkbox(
                            label=_i18n("mix_voices"),
                            value=False,
                            interactive=True,
                        )
                        mix_duet_ratio = gr.Slider(
                            label=_i18n("voice_balance"),
                            info=_i18n("voice_balance_info"),
                            minimum=-1,
                            maximum=1,
                            step=0.05,
                            value=0,
                            interactive=True,
                            visible=False
                        )

                        output_format_duet = gr.Dropdown(
                            label=_i18n("output_format"),
                            interactive=True,
                            choices=output_formats,
                            value=output_formats[0] if output_formats else "wav",
                            filterable=False,
                        )
                        convert_btn_duet = gr.Button(
                            _i18n("convert_btn"), 
                            variant="primary", 
                            interactive=True
                        )
                        
                    with gr.Row(equal_height=True):
                        output_duet_audio_1 = gr.Audio(
                            label=_i18n("model_1_result"),
                            type="filepath",
                            interactive=False,
                            show_download_button=True,
                        )
                        output_duet_audio_2 = gr.Audio(
                            label=_i18n("model_2_result"),
                            type="filepath",
                            interactive=False,
                            show_download_button=True,
                        )
                        
                        @mix_duet.change(
                            inputs=mix_duet, 
                            outputs=[mix_duet_ratio, output_duet_audio_1, output_duet_audio_2]
                        )
                        def mix_duet_change_fn(x: bool) -> Tuple[gr.update, gr.update, gr.update]:
                            if x:
                                return (
                                    gr.update(visible=x), 
                                    gr.update(label=_i18n("mixed_result"), value=None), 
                                    gr.update(visible=False, value=None)
                                )
                            else:
                                return (
                                    gr.update(visible=x), 
                                    gr.update(label=_i18n("model_1_result"), value=None), 
                                    gr.update(visible=True, value=None)
                                )

                    @convert_btn_duet.click(
                        inputs=[
                            list_input_files_duet,
                            model_name1, model_name2,
                            pitch_method1, pitch_method2,
                            pitch1, pitch2,
                            hop_length1, hop_length2,
                            index_rate1, index_rate2,
                            filter_radius1, filter_radius2,
                            rms1, rms2,
                            protect1, protect2,
                            f0_min1, f0_min2,
                            f0_max1, f0_max2,
                            output_format_duet,
                            stereo_mode_duet,
                            alt_pl_duet,
                            embedder_name1, embedder_name2,
                            transformers_mode1, transformers_mode2,
                            mix_duet, mix_duet_ratio
                        ],
                        outputs=[output_duet_audio_1, output_duet_audio_2],
                        queue=True
                    )
                    def vbach_convert_duet_fn(
                        input_file: Optional[str],
                        model_name1: str,
                        model_name2: str,
                        pitch_method1: str,
                        pitch_method2: str,
                        pitch1: float,
                        pitch2: float,
                        hop_length1: int,
                        hop_length2: int,
                        index_rate1: float,
                        index_rate2: float,
                        filter_radius1: int,
                        filter_radius2: int,
                        rms1: float,
                        rms2: float,
                        protect1: float,
                        protect2: float,
                        f0_min1: int,
                        f0_min2: int,
                        f0_max1: int,
                        f0_max2: int,
                        output_format: str,
                        stereo_mode: str,
                        alt_pipeline: bool,
                        embedder_name1: str,
                        embedder_name2: str,
                        transformers_mode1: bool,
                        transformers_mode2: bool,
                        mix_duet: bool,
                        mix_duet_ratio: float
                    ) -> Tuple[Optional[Dict], Optional[Dict], gr.update]:
                        vbach_duet_ = self.vbach_convert_duet_zero_gpu if zerogpu_available else self.vbach_convert_duet_zero_gpu
                        return vbach_duet_(
                                input_file=input_file,
                                model_name1=model_name1,
                                model_name2=model_name2,
                                pitch_method1=pitch_method1,
                                pitch_method2=pitch_method2,
                                pitch1=pitch1,
                                pitch2=pitch2,
                                hop_length1=hop_length1,
                                hop_length2=hop_length2,
                                index_rate1=index_rate1,
                                index_rate2=index_rate2,
                                filter_radius1=filter_radius1,
                                filter_radius2=filter_radius2,
                                rms1=rms1,
                                rms2=rms2,
                                protect1=protect1,
                                protect2=protect2,
                                f0_min1=f0_min1,
                                f0_min2=f0_min2,
                                f0_max1=f0_max1,
                                f0_max2=f0_max2,
                                output_format=output_format,
                                stereo_mode=stereo_mode,
                                alt_pipeline=alt_pipeline,
                                embedder_name1=embedder_name1,
                                embedder_name2=embedder_name2,
                                transformers_mode1=transformers_mode1,
                                transformers_mode2=transformers_mode2,
                                mix_duet=mix_duet,
                                mix_duet_ratio=mix_duet_ratio
                            )

            with gr.TabItem(_i18n("tab_manager")):
                with gr.TabItem(_i18n("tab_download_url")):
                    with gr.TabItem(_i18n("tab_zip")):
                        with gr.Group():
                            url_zip = gr.Textbox(
                                label=_i18n("zip_url"), 
                                interactive=True
                            )
                            url_zip_model_name = gr.Textbox(
                                label=_i18n("model_name"), 
                                interactive=True
                            )
                            url_zip_download_btn = gr.Button(
                                _i18n("download_btn"), 
                                variant="primary", 
                                interactive=True
                            )
                            url_zip_output = gr.Textbox(
                                label=_i18n("status"), 
                                interactive=False, 
                                lines=5
                            )
                            
                            url_zip_download_btn.click(
                                lambda x, y: model_manager.install_model_zip(
                                    x,
                                    namer.short(
                                        namer.sanitize(y), length=40
                                    ),
                                    "url",
                                ),
                                inputs=[url_zip, url_zip_model_name],
                                outputs=url_zip_output,
                            )

                    with gr.TabItem(_i18n("tab_files")):
                        with gr.Group():
                            url_pth = gr.Textbox(
                                label=_i18n("pth_url"), 
                                interactive=True
                            )
                            url_index = gr.Textbox(
                                label=_i18n("index_url_optional"), 
                                interactive=True
                            )
                            url_file_model_name = gr.Textbox(
                                label=_i18n("model_name"), 
                                interactive=True
                            )
                            url_file_download_btn = gr.Button(
                                _i18n("download_btn"), 
                                variant="primary", 
                                interactive=True
                            )
                            url_file_output = gr.Textbox(
                                label=_i18n("status"), 
                                interactive=False, 
                                lines=5
                            )
                            
                            url_file_download_btn.click(
                                lambda x, y, z: model_manager.install_model_files(
                                    x,
                                    y,
                                    namer.short(
                                        namer.sanitize(z), length=40
                                    ),
                                    "url",
                                ),
                                inputs=[url_index, url_pth, url_file_model_name],
                                outputs=url_file_output,
                            )

                with gr.Tab(_i18n("tab_upload_local")):
                    with gr.TabItem(_i18n("tab_zip")):
                        with gr.Group():
                            local_zip = gr.File(
                                label=_i18n("zip_file"),
                                file_types=[".zip"],
                                file_count="single",
                                interactive=True
                            )
                            local_zip_model_name = gr.Textbox(
                                label=_i18n("model_name"), 
                                interactive=True
                            )
                            local_zip_upload_btn = gr.Button(
                                _i18n("upload_btn"), 
                                variant="primary", 
                                interactive=True
                            )
                            local_zip_output = gr.Textbox(
                                label=_i18n("status"), 
                                interactive=False, 
                                lines=5
                            )
                            
                            local_zip_upload_btn.click(
                                lambda x, y: model_manager.install_model_zip(
                                    x,
                                    namer.short(
                                        namer.sanitize(y), length=40
                                    ),
                                    "local",
                                ),
                                inputs=[local_zip, local_zip_model_name],
                                outputs=local_zip_output,
                            )

                    with gr.TabItem(_i18n("tab_files")):
                        with gr.Group():
                            with gr.Row():
                                local_pth = gr.File(
                                    label=_i18n("pth_file"),
                                    file_types=[".pth"],
                                    file_count="single",
                                    interactive=True
                                )
                                local_index = gr.File(
                                    label=_i18n("index_file_optional"),
                                    file_types=[".index"],
                                    file_count="single",
                                    interactive=True
                                )
                            local_file_model_name = gr.Textbox(
                                label=_i18n("model_name"), 
                                interactive=True
                            )
                            local_file_upload_btn = gr.Button(
                                _i18n("upload_btn"), 
                                variant="primary", 
                                interactive=True
                            )
                            local_file_output = gr.Textbox(
                                label=_i18n("status"), 
                                interactive=False, 
                                lines=5
                            )
                            
                            local_file_upload_btn.click(
                                lambda x, y, z: model_manager.install_model_files(
                                    x,
                                    y,
                                    namer.short(
                                        namer.sanitize(z), length=40
                                    ),
                                    "local",
                                ),
                                inputs=[local_index, local_pth, local_file_model_name],
                                outputs=local_file_output,
                            )

                with gr.TabItem(_i18n("tab_delete_model")):
                    with gr.Group():
                        delete_model_name = gr.Dropdown(
                            label=_i18n("model_name"),
                            choices=model_manager.parse_voice_models(),
                            interactive=True,
                            filterable=False,
                        )
                        delete_refresh_btn = gr.Button(
                            _i18n("refresh"), 
                            interactive=True
                        )
                        delete_btn = gr.Button(
                            _i18n("delete"), 
                            variant="stop", 
                            interactive=True
                        )
                        
                        @delete_refresh_btn.click(
                            inputs=None, outputs=delete_model_name
                        )
                        def refresh_list_voice_models() -> gr.update:
                            models = model_manager.parse_voice_models()
                            first_model = models[0] if models else None
                            return gr.update(choices=models, value=first_model)

                        delete_output = gr.Textbox(
                            label=_i18n("status"), 
                            interactive=False, 
                            lines=5
                        )
                        
                        delete_btn.click(
                            fn=model_manager.del_voice_model,
                            inputs=delete_model_name,
                            outputs=delete_output,
                        )

                @gr.on(
                    inputs=None, 
                    outputs=[delete_model_name, model_name, model_name1, model_name2]
                )
                def refresh_all_models() -> Tuple[gr.update, gr.update, gr.update, gr.update]:
                    models = model_manager.parse_voice_models()
                    first_model = models[0] if models else None
                    return (
                        gr.update(choices=models, value=first_model),
                        gr.update(choices=models, value=first_model),
                        gr.update(choices=models, value=first_model),
                        gr.update(choices=models, value=first_model)
                    )
                    
        return vbach_app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vbach - RVC форк")
    
    # Основные подкоманды
    subparsers = parser.add_subparsers(dest="mode", help=_i18n("mode"), required=True)
    
    # CLI режим
    cli_parser = subparsers.add_parser("cli", help=_i18n("cli_mode"))
    cli_parser.add_argument("--input", nargs="*", help=_i18n("input_path_help"))
    cli_parser.add_argument(
        "--output_dir", type=str, required=True, help=_i18n("output_dir_help")
    )
    cli_parser.add_argument(
        "--output_format",
        type=str,
        default="wav",
        choices=output_formats,
        help=_i18n("output_format_help"),
    )
    cli_parser.add_argument(
        "--output_bitrate", type=str, default="320k", help=_i18n("output_bitrate_help")
    )
    cli_parser.add_argument(
        "--format_name",
        action="store_true",
        help=_i18n("format_name_help"),
    )
    cli_parser.add_argument(
        "--output_name",
        type=str,
        default="NAME_STEM",
        help=_i18n("output_name_help"),
    )
    cli_parser.add_argument(
        "--model_name",
        type=str,
        default="model",
        help=_i18n("model_name_help"),
    )
    cli_parser.add_argument(
        "--index_rate",
        type=float,
        default=0,
        help=_i18n("index_rate_help"),
        metavar="[0.0-1.0]",
    )
    cli_parser.add_argument(
        "--stereo_mode",
        type=str,
        default="mono",
        choices=["mono", "left/right", "sim/dif"],
        help=_i18n("stereo_mode_help"),
    )
    cli_parser.add_argument(
        "--method_pitch",
        type=str,
        default="rmvpe+",
        help=_i18n("f0_method_help"),
    )
    cli_parser.add_argument(
        "--pitch", type=int, default=0, help=_i18n("pitch_help")
    )
    cli_parser.add_argument(
        "--hop_length",
        type=int,
        default=128,
        help=_i18n("hop_length_help"),
    )
    cli_parser.add_argument(
        "--filter_radius", type=int, default=3, help=_i18n("filter_radius_help")
    )
    cli_parser.add_argument(
        "--rms",
        type=float,
        default=0.25,
        help=_i18n("rms_help"),
    )
    cli_parser.add_argument(
        "--protect", type=float, default=0.33, help=_i18n("protect_help")
    )
    cli_parser.add_argument(
        "--f0_min", type=int, default=50, help=_i18n("f0_min_help")
    )
    cli_parser.add_argument(
        "--f0_max", type=int, default=1100, help=_i18n("f0_max_help")
    )
    cli_parser.add_argument(
        "--alt_pipeline",
        action="store_true",
        help=_i18n("alt_pipeline_help"),
    )
    cli_parser.add_argument(
        "--use_transformers",
        action="store_true",
        help=_i18n("use_transformers_help"),
    )
    cli_parser.add_argument(
        "--embedder_name",
        type=str,
        default="hubert_base",
        help=_i18n("embedder_name_help"),
    )
    
    # App режим
    app_parser = subparsers.add_parser("app", help=_i18n("app_mode"))
    app_parser.add_argument(
        "--port", 
        type=int, 
        default=7860, 
        help=_i18n("port_help")
    )
    app_parser.add_argument(
        "--share",
        action="store_true",
        help=_i18n("share_help"),
    )
    app_parser.add_argument(
        "--debug",
        action="store_true",
        help=_i18n("debug_help"),
    )

    model_manager_parser = subparsers.add_parser(
        "model_manager", help=_i18n("model_manager_help")
    )
    vbach_model_manager_parser = model_manager_parser.add_subparsers(
        title="vbach_commands", dest="vbach_command", required=True
    )

    install_local_parser = vbach_model_manager_parser.add_parser(
        "install_local", help=_i18n("install_local_help")
    )
    install_local_parser.add_argument(
        "--model_name", required=True, help=_i18n("model_name_help")
    )
    install_local_parser.add_argument("--pth", required=True, help=_i18n("pth_path_help"))
    install_local_parser.add_argument(
        "--index", required=False, help=_i18n("index_path_help")
    )

    install_url_zip_parser = vbach_model_manager_parser.add_parser(
        "install_url_zip", help=_i18n("install_url_zip_help")
    )
    install_url_zip_parser.add_argument(
        "--model_name", required=True, help=_i18n("model_name_help")
    )
    install_url_zip_parser.add_argument("--url", required=True, help=_i18n("zip_url_help"))

    install_url_files_parser = vbach_model_manager_parser.add_parser(
        "install_url_files", help=_i18n("install_url_files_help")
    )
    install_url_files_parser.add_argument(
        "--model_name", required=True, help=_i18n("model_name_help")
    )
    install_url_files_parser.add_argument(
        "--pth_url", required=True, help=_i18n("pth_url_help")
    )
    install_url_files_parser.add_argument(
        "--index_url", required=False, help=_i18n("index_url_help")
    )

    list_parser = vbach_model_manager_parser.add_parser(
        "list", help=_i18n("list_models_help")
    )

    remove_voice_model = vbach_model_manager_parser.add_parser(
        "remove", help=_i18n("remove_model_help")
    )
    remove_voice_model.add_argument(
        "--model_name", required=True, help=_i18n("model_name_help")
    )

    args = parser.parse_args()

    if args.mode == "cli":
        if not args.input:
            cli_parser.error(_i18n("input_required"))
            
        list_valid_files = get_files_from_list(args.input)
        if list_valid_files:
            for i, vocals_file in enumerate(list_valid_files, start=1):
                print(_i18n('processing_file', current=i, total=len(list_valid_files), file=vocals_file))
                vbach_inference(
                    input_file=vocals_file,
                    model_name=args.model_name,
                    output_dir=args.output_dir,
                    output_name=args.output_name,
                    output_bitrate=args.output_bitrate,
                    output_format=args.output_format,
                    pitch=args.pitch,
                    method_pitch=args.method_pitch,
                    format_name=(True if len(list_valid_files) > 1 else args.format_name),
                    add_params={
                        "index_rate": args.index_rate,
                        "filter_radius": args.filter_radius,
                        "protect": args.protect,
                        "rms": args.rms,
                        "mangio_crepe_hop_length": args.hop_length,
                        "f0_min": args.f0_min,
                        "f0_max": args.f0_max,
                        "stereo_mode": args.stereo_mode,
                    },
                    pipeline_mode="alt" if args.alt_pipeline else "orig",
                    embedder_name=args.embedder_name,
                    stack="transformers" if args.use_transformers else "fairseq",
                    device=set_device()
                )
        else:
            sys.exit(1)
    
    elif args.mode == "app":
        Vbach(user_directory, set_device(0)).UI().launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
            allowed_paths=["/"],
            debug=args.debug,
            inbrowser=True
        )

    elif args.mode == "model_manager":
        if args.vbach_command == "install_local":
            status = model_manager.install_model_files(
                args.index, args.pth, args.model_name, mode="local"
            )
            print(status)

        elif args.vbach_command == "install_url_zip":
            status = model_manager.install_model_zip(
                args.url, args.model_name, mode="url"
            )
            print(status)

        elif args.vbach_command == "install_url_files":
            status = model_manager.install_model_files(
                args.index_url, args.pth_url, args.model_name, mode="url"
            )
            print(status)

        elif args.vbach_command == "list":
            model_manager.get_list_installed_models()

        elif args.vbach_command == "remove":
            status = model_manager.del_voice_model(args.model_name)

            print(status)

