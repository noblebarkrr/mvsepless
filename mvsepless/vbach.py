import os
import gc
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
from typing import Tuple, Any, Dict
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
from separator import get_files_from_list
from datetime import datetime, timezone, timedelta
from functools import wraps
script_dir = os.path.dirname(os.path.abspath(__file__))

FILTER_ORDER = 5
CUTOFF_FREQUENCY = 48
SAMPLE_RATE = 16000
bh, ah = signal.butter(
    N=FILTER_ORDER, Wn=CUTOFF_FREQUENCY, btype="high", fs=SAMPLE_RATE
)
from multiprocessing import cpu_count
from audio import check, read, write, output_formats, split_mid_side, split_channels, easy_resampler, stereo_to_mono, mono_to_stereo, convert_to_dtype, gain, add_zero_to_end, multi_channel_array_from_arrays, trim, fit_arrays
from namer import Namer
from gradio_helper import GradioHelper, tz, dw_file, easy_check_is_colab, str2bool, all_ids, set_device
from vbach_lib.fairseq import load_model_ensemble_and_task, load_checkpoint_to_cpu
from vbach_lib.algorithm.synthesizers import Synthesizer
from vbach_lib.predictors.FCPE import FCPEF0Predictor
from vbach_lib.predictors.RMVPE import RMVPE0Predictor
from vbach_lib.predictors.HPA_RMVPE import HPA_RMVPE

VBACH_ALT_PIPELINE_TIME_CHUNK = int(os.environ.get("VBACH_ALTPL_BASE_SEG", "10"))

class UserDirectory:
    path = ""
    def change_dir(self, dir: str):
        self.path = dir
        os.makedirs(dir, exist_ok=True)
    
user_directory = UserDirectory()
IS_COLAB = easy_check_is_colab()

if IS_COLAB:

    print("[VBach] Обнаружена среда выполнения Colab")
    result = subprocess.run(['/bin/mount'], capture_output=True, text=True)

    for line in result.stdout.strip().split('\n'):
        if 'type fuse.drive' in line:
            parts = line.split(' type ')
            if len(parts) >= 2:
                source_mount = parts[0]
                source, mount_point = source_mount.split(' on ')
                user_directory.change_dir(os.path.join(mount_point, "MyDrive", "mvsepless-data-gdrive"))
                os.makedirs(user_directory.path, exist_ok=True)
                print(f"[VBach] Обнаружен привязанный Google Диск\nПуть к привязанному диску: {mount_point}")
                break

def generate_secure_random(length=10):
    characters = string.ascii_letters + string.digits
    return "".join(secrets.choice(characters) for _ in range(length))

class VbachModelManager:
    def __init__(self, user_directory):
        self.user_directory = user_directory
        self.rmvpe_path = os.path.join(script_dir, "vbach_lib", "predictors", "rmvpe.pt")
        self.hpa_rmvpe_path = os.path.join(script_dir, "vbach_lib", "predictors", "hpa_rmvpe.pt")
        self.fcpe_path = os.path.join(script_dir, "vbach_lib", "predictors", "fcpe.pt")
        self.custom_fairseq_huberts_dir = os.path.join(
            script_dir, "vbach_lib", "huberts", "fairseq"
        )
        self.custom_transformers_huberts_dir = os.path.join(
            script_dir, "vbach_lib", "huberts", "transformers"
        )
        self.huberts_fairseq_dict = {
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
        self.huberts_transformers_dict = {
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
        self.requirements = [
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
        self.voicemodels_dir = os.path.join(user_directory.path, "vbach_models_cache")
        os.makedirs(self.voicemodels_dir, exist_ok=True)
        self.voicemodels_info = os.path.join(self.voicemodels_dir, "vbach_models.json")
        self.voicemodels: Dict[str, Dict[str, str]] = {}
        self.download_requirements()
        self.check_hubert("hubert_base")
        self.check_and_load()

    def check_hubert(self, embedder_name):
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

    def check_hubert_transformers(self, embedder_name):
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

    def write_voicemodels_info(self):
        with open(self.voicemodels_info, "w") as f:
            json.dump(self.voicemodels, f, indent=4)

    def load_voicemodels_info(self):
        with open(self.voicemodels_info, "r") as f:
            return json.load(f)

    def add_voice_model(
        self,
        name,
        pth_path,
        index_path,
    ):
        self.voicemodels[name] = {"pth": pth_path, "index": index_path}
        self.write_voicemodels_info()

    def del_voice_model(self, name):
        if name in self.parse_voice_models():
            pth = self.voicemodels[name].get("pth", None)
            index = self.voicemodels[name].get("index", None)
            if index:
                os.remove(index)
            if pth:
                os.remove(pth)
            del self.voicemodels[name]
            self.write_voicemodels_info()
            return f"Модель {name} удалена"
        else:
            return f"Модель не была удалена, как так её не существует"

    def parse_voice_models(self):
        list_models = list(self.voicemodels.keys())
        return list_models

    def parse_pth_and_index(self, name):
        pth = self.voicemodels[name].get("pth", None)
        index = self.voicemodels[name].get("index", None)
        return pth, index

    def check_and_load(self):
        if os.path.exists(self.voicemodels_info):
            self.voicemodels = self.load_voicemodels_info()
        else:
            self.write_voicemodels_info()

    def clear_voicemodels_info(self):
        self.voicemodels: Dict[str, Dict[str, str]] = {}
        self.write_voicemodels_info()

    def download_requirements(self):
        for url, file in self.requirements:
            if not os.path.exists(file):
                dw_file(url, file)

    def download_voice_model_file(self, url, zip_name):
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
            print(e)

    def download_from_google_drive(self, url, zip_name):
        file_id = (
            url.split("file/d/")[1].split("/")[0]
            if "file/d/" in url
            else url.split("id=")[1].split("&")[0]
        )
        gdown.download(id=file_id, output=str(zip_name), quiet=False)

    def download_from_pixeldrain(self, url, zip_name):
        file_id = url.split("pixeldrain.com/u/")[1]
        response = requests.get(f"https://pixeldrain.com/api/file/{file_id}")
        with open(zip_name, "wb") as f:
            f.write(response.content)

    def download_from_yandex(self, url, zip_name):
        yandex_public_key = f"download?public_key={url}"
        yandex_api_url = (
            f"https://cloud-api.yandex.net/v1/disk/public/resources/{yandex_public_key}"
        )
        response = requests.get(yandex_api_url)
        if response.status_code == 200:
            download_link = response.json().get("href")
            urllib.request.urlretrieve(download_link, zip_name)
        else:
            print(response.status_code)

    def extract_zip(self, zip_name, model_name):
        model_dir = os.path.join(
            self.voicemodels_dir, f"{model_name}_{generate_secure_random(17)}"
        )
        os.makedirs(model_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_name, "r") as zip_ref:
                zip_ref.extractall(model_dir)
            os.remove(zip_name)

            added_voice_models = []

            index_filepath, model_filepaths = None, []
            for root, _, files in os.walk(model_dir):
                for name in files:
                    file_path = os.path.join(root, name)
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
            list_models_str = "\n".join(added_voice_models)
            return f"Добавленные модели:\n{list_models_str}"
        except Exception as e:
            return f"Произошла ошибка при загрузке модели: {e}"

    def install_model_zip(self, zip, model_name, mode="url"):
        if model_name in self.parse_voice_models():
            print(
                "Эта модель уже есть в списке установленных моделей. Она будут перезаписана"
            )
        if mode == "url":
            with tempfile.TemporaryDirectory(
                prefix="vbach_temp_model", ignore_cleanup_errors=True
            ) as tmp:
                zip_path = os.path.join(tmp, "model.zip")
                self.download_voice_model_file(zip, zip_path)
                status = self.extract_zip(zip_path, model_name)
        if mode == "local":
            status = self.extract_zip(zip, model_name)
        return status

    def install_model_files(self, index, pth, model_name, mode="url"):
        if model_name in self.parse_voice_models():
            print(
                "Эта модель уже есть в списке установленных моделей. Она будут перезаписана"
            )
        model_dir = os.path.join(
            self.voicemodels_dir, f"{model_name}_{generate_secure_random(17)}"
        )
        os.makedirs(model_dir, exist_ok=True)
        local_index_path = None
        local_pth_path = None
        try:
            if mode == "url":
                if index:
                    local_index_path = os.path.join(model_dir, "model.index")
                    self.download_voice_model_file(index, local_index_path)
                if pth:
                    local_pth_path = os.path.join(model_dir, "model.pth")
                    self.download_voice_model_file(pth, local_pth_path)

            if mode == "local":
                if index:
                    if os.path.exists(index):
                        local_index_path = os.path.join(
                            model_dir, os.path.basename(index)
                        )
                        shutil.copy(index, local_index_path)
                if pth:
                    if os.path.exists(pth):
                        local_pth_path = os.path.join(model_dir, os.path.basename(pth))
                        shutil.copy(pth, local_pth_path)

            self.add_voice_model(model_name, local_pth_path, local_index_path)
            return f"Модель {model_name} добавлена"
        except Exception as e:
            return f"Произошла ошибка при загрузке модели: {e}"

model_manager = VbachModelManager(user_directory)
namer = Namer()

f0_methods = (
    "rmvpe+",
    "hpa-rmvpe",
    "fcpe",
    "mangio-crepe",
    "mangio-crepe-tiny",
    "harvest",
    "pm",
    "pyin",
)
HPA_RMVPE_DIR = model_manager.hpa_rmvpe_path
RMVPE_DIR = model_manager.rmvpe_path
FCPE_DIR = model_manager.fcpe_path

input_audio_path2wav = {}


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


@lru_cache
def get_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    audio = input_audio_path2wav[input_audio_path]
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
    @staticmethod
    def change_rms(sourceaudio, source_rate, targetaudio, target_rate, rate):
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

        adjustedaudio = (
            targetaudio
            * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1)).numpy()
        )
        return adjustedaudio


class VC:
    def __init__(self, tgt_sr, config, stack="fairseq"):
        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.is_half = config.is_half
        self.sample_rate = 16000
        self.window = 160
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max
        self.time_step = self.window / self.sample_rate * 1000
        self.device = config.device
        self.vc = self._vc_transformers if stack == "transformers" else self._vc

    def get_f0_mangio_crepe(self, x, f0_min, f0_max, p_len, hop_length, model="full"):
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

    def get_f0_rmvpe(self, x, f0_min=1, f0_max=40000, *args, **kwargs):
        if not hasattr(self, "model_rmvpe"):
            self.model_rmvpe = RMVPE0Predictor(
                RMVPE_DIR, is_half=self.is_half, device=self.device
            )
        f0 = self.model_rmvpe.infer_from_audio_with_pitch(
            x, thred=0.03, f0_min=f0_min, f0_max=f0_max
        )
        return f0
    
    def get_f0_hpa_rmvpe(self, x, f0_min=1, f0_max=40000, *args, **kwargs):
        if not hasattr(self, "model_hpa_rmvpe"):
            self.model_hpa_rmvpe = HPA_RMVPE(
                HPA_RMVPE_DIR, device=self.device, hpa=True
            )
        f0 = self.model_hpa_rmvpe.infer_from_audio_with_pitch(
            x, thred=0.03, f0_min=f0_min, f0_max=f0_max
        )
        return f0

    def get_f0_librosa(self, x, p_len, f0_min=50, f0_max=1100, hop_length=160):

        f0, *_ = librosa.pyin(
            x.astype(np.float32),
            sr=self.sample_rate,
            fmin=f0_min,
            fmax=f0_max,
            hop_length=hop_length,
        )

        return self._resize_f0(f0, p_len)

    def _resize_f0(self, x, target_len):
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
        inputaudio_path,
        x,
        p_len,
        pitch,
        f0_method,
        filter_radius,
        hop_length,
        inp_f0=None,
        f0_min=50,
        f0_max=1100,
    ):
        global input_audio_path2wav
        time_step = self.window / self.sample_rate * 1000
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

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
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )

        elif f0_method == "rmvpe+":
            f0 = self.get_f0_rmvpe(x=x, f0_min=f0_min, f0_max=f0_max)

        elif f0_method == "hpa-rmvpe":
            f0 = self.get_f0_hpa_rmvpe(x=x, f0_min=f0_min, f0_max=f0_max)

        elif f0_method == "fcpe":
            self.model_fcpe = FCPEF0Predictor(
                FCPE_DIR,
                f0_min=int(f0_min),
                f0_max=int(f0_max),
                dtype=torch.float32,
                device=self.device,
                sample_rate=self.sample_rate,
                threshold=0.03,
            )
            f0 = self.model_fcpe.compute_f0(x, p_len=p_len)
            del self.model_fcpe
            gc.collect()

        f0 *= pow(2, pitch / 12)
        tf0 = self.sample_rate // self.window
        if inp_f0 is not None:
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[
                :shape
            ]

        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(int)
        return f0_coarse, f0bak

    def _vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        feats = torch.from_numpy(audio0)
        feats = feats.half() if self.is_half else feats.float()

        if feats.dim() == 2:
            feats = feats.mean(-1)

        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
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

            p_len = audio0.shape[0] // self.window
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

            p_len = torch.tensor([p_len], device=self.device).long()

            if pitch is not None and pitchf is not None:
                audio1 = (
                    (net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )
            else:
                audio1 = (
                    (net_g.infer(feats, p_len, sid)[0][0, 0]).data.cpu().float().numpy()
                )

        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio1

    def _vc_transformers(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):
        with torch.no_grad():
            pitch_guidance = pitch != None and pitchf != None
            feats = torch.from_numpy(audio0).float()
            feats = feats.mean(-1) if feats.dim() == 2 else feats
            assert feats.dim() == 1, feats.dim()
            feats = feats.view(1, -1).to(self.device)
            feats = model(feats)["last_hidden_state"]
            feats = (
                model.final_proj(feats[0]).unsqueeze(0) if version == "v1" else feats
            )
            feats0 = feats.clone() if pitch_guidance else None
            if index:
                feats = self._retrieve_speaker_embeddings(
                    feats, index, big_npy, index_rate
                )
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
            p_len = min(audio0.shape[0] // self.window, feats.shape[1])
            if pitch_guidance:
                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                    0, 2, 1
                )
                pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len]
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
            p_len = torch.tensor([p_len], device=self.device).long()
            audio1 = (
                (net_g.infer(feats.float(), p_len, pitch, pitchf.float(), sid)[0][0, 0])
                .data.cpu()
                .float()
                .numpy()
            )
            del feats, feats0, p_len
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        inputaudio_path,
        pitch,
        f0_method,
        file_index,
        index_rate,
        pitch_guidance,
        filter_radius,
        tgt_sr,
        resample_sr,
        volume_envelope,
        version,
        protect,
        hop_length,
        f0_file,
        f0_min=50,
        f0_max=1100,
        add_text=""
    ):
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
                print(f"Произошла ошибка при чтении индекса FAISS: {e}")
                index = big_npy = None
        else:
            index = big_npy = None
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
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
        s = 0
        audio_opt = []
        t = None
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
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
                print(f"Произошла ошибка при чтении файла F0: {e}")
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()

        progress = gr.Progress()
        progress((2, 4), desc=f"Вычисление кривой F0 {add_text}", unit="")

        if pitch_guidance:
            pitch, pitchf = self.get_f0(
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
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if self.device.type == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()

        progress = gr.Progress()
        total_ts = len(opt_ts)

        for i, t in enumerate(opt_ts, start=1):
            progress((i, total_ts), desc=f"Синтез голоса... {add_text}", unit="чанков")
            print(f"\rСинтез голоса... {int((i / total_ts) * 100)}% {add_text}", end="")
            t = t // self.window * self.window
            if pitch_guidance:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
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
                        sid,
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
            progress(1, desc=f"Синтез голоса... [Финал] {add_text}")
            print(f"\rСинтез голоса... 100% {add_text}", end="")
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            progress(1, desc=f"Синтез голоса... [Финал] {add_text}")
            print(f"\rСинтез голоса... 100% {add_text}", end="")
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
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
        audio_opt = np.concatenate(audio_opt)
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

        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio_opt

    def pipeline2(
        self,
        model,
        net_g,
        sid,
        audio,
        inputaudio_path,
        pitch,
        f0_method,
        file_index,
        index_rate,
        pitch_guidance,
        filter_radius,
        tgt_sr,
        resample_sr,
        volume_envelope,
        version,
        protect,
        hop_length,
        f0_file,
        f0_min=50,
        f0_max=1100,
        add_text=""
    ):

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
                print(f"Ошибка при чтении FAISS индекса: {e}")
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
                print(f"Ошибка при чтении F0 файла: {e}")

        sid_tensor = torch.tensor(sid, device=device).unsqueeze(0).long()

        raw_chunk_size = self.get_max_memory_chunk(audio_len, model, net_g, version)
        offset = int(tgt_sr // 12.5)
        real_chunk_size = raw_chunk_size
        if real_chunk_size <= 0:
            raise ValueError("Chunk size too small")
        
        print(f"Размер чанка: {real_chunk_size} | {int(real_chunk_size / self.sample_rate)} cекунд")

        audio_pad = np.pad(audio, (offset, offset), mode="reflect")
        padded_len = len(audio_pad)

        progress = gr.Progress()
        progress((2, 4), desc=f"Вычисление кривой F0 {add_text}", unit="")

        pitch_tensor = pitchf_tensor = None
        if pitch_guidance:
            p_len = len(audio_pad) // self.window
            pitch, pitchf = self.get_f0(
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
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if device.type == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch_tensor = torch.tensor(pitch, device=device).unsqueeze(0).long()
            pitchf_tensor = torch.tensor(pitchf, device=device).unsqueeze(0).float()

        processed_chunks = []
        start = 0

        chunk_count = 0
        temp_start = 0
        while temp_start < audio_len:
            temp_end = min(temp_start + real_chunk_size, audio_len)
            chunk_count += 1
            temp_start = temp_end
        
        current_chunk = 0

        progress = gr.Progress()

        while start < audio_len:
            current_chunk += 1
            progress((current_chunk, chunk_count), desc=f"Синтез голоса... [ALT] {add_text}", unit="чанков")
            print(f"\rСинтез голоса... [ALT] {int((current_chunk / chunk_count) * 100)}% {add_text}", end="")
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

            if pitch_guidance:
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

            input_duration_sec = len(chunk_audio) / self.sample_rate

            output_start = int(round((chunk_start_in_pad) / self.sample_rate * tgt_sr))
            output_end = output_start + len(out)

            processed_chunks.append(
                (output_start, output_end, out, pad_left, pad_right)
            )

            start = end

        if not processed_chunks:
            raise RuntimeError("No chunks processed")

        max_output_end = max(end for _, end, _, _, _ in processed_chunks)
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

        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt

    def get_max_memory_chunk(
        self, audio_length: int, model, net_g, version: str
    ) -> int:
        """
        Рассчитывает оптимальный размер чанка на основе доступной памяти
        """
        base_chunk_size = min(self.sample_rate * VBACH_ALT_PIPELINE_TIME_CHUNK, audio_length)

        if self.device.type == "cuda" and torch.cuda.is_available() and not str2bool(os.environ.get("VBACH_ALTPL_PREF_BASE_SEG", False)):
            try:
                torch.cuda.synchronize()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated(0)
                free_memory = total_memory - allocated

                usable_memory = free_memory * 0.2

                print(
                    f"Доступно видеопамяти: {free_memory/1024**3:.2f} GB, "
                    f"используем: {usable_memory/1024**3:.2f} GB"
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
                print(f"Ошибка при расчете размера чанка: {e}")

        return min(base_chunk_size, audio_length)

    def _retrieve_speaker_embeddings(self, feats, index, big_npy, index_rate):
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

def loadaudio(file_path: str, target_sr: int, stereo_mode: str) -> np.ndarray:
    try:
        mid, left, right = None, None, None
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
        raise RuntimeError(f"Ошибка загрузки аудио '{file_path}': {str(e)}")


class Config:
    def __init__(self, device):
        self.device_str = device
        self.set_device(self.device_str)
        self.is_half = False
        self.n_cpu = cpu_count()
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def set_device(self, device_str):
        if "cuda" in device_str.lower():
            # Извлекаем ID устройств для CUDA
            if ":" in device_str:
                device_spec = device_str.split(":")[1]
                self.device_ids = [int(id) for id in device_spec.split(",") if id.isdigit()]
            else:
                # Если указано просто "cuda", используем все доступные GPU
                self.device_ids = list(range(torch.cuda.device_count()))
            self.device = torch.device("cuda" if not self.device_ids else f"cuda:{self.device_ids[0]}")
        elif "mps" in device_str.lower():
            self.device_ids = None
            self.device = torch.device("mps")
        else:
            self.device_ids = None
            self.device = torch.device("cpu")

    def device_config(self):
        if self.device.type == "cuda":
            print("Используется устройство CUDA")
            self.gpu_mem = self._configure_gpu(self.device_ids[0])
        elif self.device.type == "mps":
            print("Используется устройство MPS")
        else:
            print("Используется CPU")

        x_pad, x_query, x_center, x_max = (
            (3, 10, 60, 65) if self.is_half else (1, 6, 38, 41)
        )
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad, x_query, x_center, x_max = (1, 5, 30, 32)

        return x_pad, x_query, x_center, x_max

    def _configure_gpu(self, device_id):
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


def load_hubert(device, is_half, model_path):
    models, saved_cfg, task = load_model_ensemble_and_task([model_path], suffix="")
    hubert = models[0].to(device)
    hubert = hubert.half() if is_half else hubert.float()
    hubert.eval()
    return hubert


def get_vc(
    device: torch.device, is_half: bool, config: Any, model_path: str, stack: Any
) -> Tuple[Dict[str, Any], str, torch.nn.Module, int, VC, int]:

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")

    try:
        cpt = torch.load(model_path, map_location="cpu", weights_only=True)

        required_keys = ["config", "weight"]
        missing_keys = [key for key in required_keys if key not in cpt]

        if missing_keys:
            raise ValueError(
                f"Некорректный формат модели {model_path}. "
                f"Отсутствующие ключи: {missing_keys}. "
                "Используйте модель RVC формата."
            )

        tgt_sr = cpt["config"][-1]

        emb_weight_shape = cpt["weight"]["emb_g.weight"].shape
        cpt["config"][-3] = emb_weight_shape[0]

        use_f0 = cpt.get("f0", 1)
        version = cpt.get("version", "v1")
        vocoder = cpt.get("vocoder", "HiFi-GAN")

        text_enc_hidden_dim = 768 if version == "v2" else 256

        print(f"Загружаем модель: {os.path.basename(model_path)}")
        print(f"Версия: {version}, F0: {use_f0}, Частота: {tgt_sr}Hz")
        print(f"Количество спикеров: {emb_weight_shape[0]}")

        net_g = Synthesizer(
            *cpt["config"],
            use_f0=use_f0,
            text_enc_hidden_dim=text_enc_hidden_dim,
            vocoder=vocoder,
        )

        if hasattr(net_g, "enc_q"):
            del net_g.enc_q
        else:
            print("Предупреждение: слой enc_q не найден в модели")

        missing_keys, unexpected_keys = net_g.load_state_dict(
            cpt["weight"], strict=False
        )

        if missing_keys:
            print(
                f"Предупреждение: отсутствующие ключи при загрузке модели: {missing_keys}"
            )

        if unexpected_keys:
            print(
                f"Предупреждение: неожиданные ключи при загрузке модели: {unexpected_keys}"
            )

        net_g.eval()

        net_g = net_g.to(device)
        if is_half:
            net_g = net_g.half()
            print("Модель переведена в половинную точность (float16)")
        else:
            net_g = net_g.float()
            print("Модель использует полную точность (float32)")

        vc = VC(tgt_sr, config, stack)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Модель успешно загружена на устройство: {device}")

        return cpt, version, net_g, tgt_sr, vc, use_f0

    except torch.serialization.pickle.UnpicklingError as e:
        raise ValueError(
            f"Файл {model_path} поврежден или имеет неверный формат"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке модели: {str(e)}") from e


def rvc_infer(
    index_path,
    index_rate,
    input_path,
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
    pitch_guidance,
    f0_min=50,
    f0_max=1100,
    format_output="wav",
    output_bitrate="320k",
    stereo_mode="mono",
    pipeline_mode="orig",
    add_text=""
) -> str:

    if pipeline_mode == "alt":
        pipeline = vc.pipeline2
    else:
        pipeline = vc.pipeline

    mid, left, right = loadaudio(input_path, 16000, stereo_mode)

    if stereo_mode == "mono":
        if mid is None:
            raise ValueError("Mono audio data is None")
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
            raise ValueError("Left or right audio channel is None")

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
            raise ValueError("Processed audio is empty")
        
        output_dtype = leftaudio_opt.dtype

        leftaudio_opt = trim(leftaudio_opt, 0, min_len)
        rightaudio_opt = trim(rightaudio_opt, 0, min_len)

        audio_opt = multi_channel_array_from_arrays(leftaudio_opt, rightaudio_opt, index=1, dtype=output_dtype)

    elif stereo_mode == "sim/dif":
        if mid is None or left is None or right is None:
            raise ValueError("Mid, left or right audio channel is None")

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
            add_text=f"{add_text} (Центр)"
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
            add_text=f"{add_text} (Стерео-база L)"
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
            add_text=f"{add_text} (Стерео-база R)"
        )

        min_len = min(len(midaudio_opt), len(leftaudio_opt), len(rightaudio_opt))
        if min_len == 0:
            raise ValueError("Processed audio is empty")
        output_dtype = leftaudio_opt.dtype
        midaudio_opt = trim(midaudio_opt, 0, min_len)
        leftaudio_opt = trim(leftaudio_opt, 0, min_len)
        rightaudio_opt = trim(rightaudio_opt, 0, min_len)
        difaudio_opt = multi_channel_array_from_arrays(leftaudio_opt, rightaudio_opt, index=1, dtype=output_dtype)
        audio_opt = convert_to_dtype((mono_to_stereo(midaudio_opt, index=1) + difaudio_opt), output_dtype)

    output_path = write(
        namer.iter(output_path), audio_opt, tgt_sr, output_bitrate
    )
    return output_path


def load_rvc_model(voice_model):

    if voice_model in model_manager.parse_voice_models():
        rvc_model_path, rvc_index_path = model_manager.parse_pth_and_index(voice_model)

        if not rvc_model_path:
            raise ValueError(
                f"[91mФайла для модели {voice_model} не существует. "
                "Возможно, вы неправильно её установили.[0m"
            )

    else:
        raise ValueError(
            f"[91mМодели {voice_model} не существует. "
            "Возможно, вы неправильно ввели имя.[0m"
        )

    return rvc_model_path, rvc_index_path


def voice_conversion(
    voice_model,
    vocals_path,
    output_path,
    pitch,
    f0_method,
    index_rate,
    filter_radius,
    volume_envelope,
    protect,
    hop_length,
    f0_min,
    f0_max,
    format_output,
    output_bitrate,
    stereo_mode,
    embedder_name="hubert_base",
    pipeline_mode="orig",
    device="cpu",
    add_text_progress=""
):
    _add_text = ""
    if add_text_progress != "" or add_text_progress is not None:
        _add_text = f"| {add_text_progress}"
    rvc_model_path, rvc_index_path = load_rvc_model(voice_model)
    progress = gr.Progress()
    progress((0, 4), desc=f"Загрузка RVC модели {_add_text}", unit="")
    config = Config(device)
    progress((1, 4), desc=f"Загрузка Hubert модели {_add_text}", unit="")
    hubert_path = model_manager.check_hubert(embedder_name)
    if not hubert_path:
        raise ValueError(
            f"[91mЭмбеддера {embedder_name} не существует. "
            "Возможно, вы неправильно ввели имя.[0m"
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
        _add_text
    )

    del hubert_model, cpt, net_g, vc
    gc.collect()
    torch.cuda.empty_cache()
    return outputaudio


def voice_conversion_transformers(
    voice_model,
    vocals_path,
    output_path,
    pitch,
    f0_method,
    index_rate,
    filter_radius,
    volume_envelope,
    protect,
    hop_length,
    f0_min,
    f0_max,
    format_output,
    output_bitrate,
    stereo_mode,
    embedder_name="contentvec",
    pipeline_mode="orig",
    device="cpu",
    add_text_progress=""
):
    _add_text = ""
    if add_text_progress != "" or add_text_progress is not None:
        _add_text = f"| {add_text_progress}"
    progress = gr.Progress()
    progress((0, 4), desc=f"Загрузка RVC модели {_add_text}", unit="")
    rvc_model_path, rvc_index_path = load_rvc_model(voice_model)

    config = Config(device)
    progress((1, 4), desc=f"Загрузка Hubert модели {_add_text}", unit="")
    hubert_path = model_manager.check_hubert_transformers(embedder_name)
    if not hubert_path:
        raise ValueError(
            f"[91mЭмбеддера {embedder_name} не существует. "
            "Возможно, вы неправильно ввели имя.[0m"
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
        _add_text
    )

    del hubert_model, cpt, net_g, vc
    gc.collect()
    torch.cuda.empty_cache()
    return outputaudio


def vbach_inference(
    input_file: str,
    model_name: str,
    output_dir: str,
    output_name: str,
    output_format: str,
    output_bitrate: str | int,
    pitch: int,
    method_pitch: str,
    format_name: bool = False,
    pipeline_mode: str = "orig",
    embedder_name: str | None = "hubert_base",
    stack: str = "fairseq",
    add_params: dict = {
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
):

    if stack == "fairseq":
        vbach_convert = voice_conversion
    elif stack == "transformers":
        vbach_convert = voice_conversion_transformers

    stereo_mode = add_params.get("stereo_mode", "mono")
    index_rate = add_params.get("index_rate", 0)
    filter_radius = add_params.get("filter_radius", 3)
    protect = add_params.get("protect", 0.33)
    rms = add_params.get("rms", 0.25)
    mangio_crepe_hop_length = add_params.get("mangio_crepe_hop_length", 0)
    f0_min = add_params.get("f0_min", 50)
    f0_max = add_params.get("f0_max", 1100)
    if not input_file:
        raise ValueError("Входной файл не указан")
    if not os.path.exists(input_file):
        raise ValueError("Входного файла не существует")
    if not check(input_file):
        raise ValueError("Входной файл не содержит аудио")
    basename = os.path.splitext(os.path.basename(input_file))[0]

    final_output_name = None

    print("Инференс запущен")

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

    print(f"Эмбеддер: {embedder_name}", f"Стэк: {stack}", sep="\n")

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
        output_bitrate=output_bitrate,
        stereo_mode=stereo_mode,
        pipeline_mode=pipeline_mode,
        embedder_name=embedder_name,
        device=device,
        add_text_progress=add_text_progress
    )
    print(f'Инференс завершен\nПуть к выходному файлу: "{output_converted_voice}"')
    return output_converted_voice

class History:
    def __init__(self, user_directory):
        self.info = {}
        self.user_directory = user_directory
        self.path = os.path.join(self.user_directory.path, "history", "vbach.json")
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
    
    def _write_file(self):
        """Записывает текущее состояние в файл"""
        try:
            dir = os.path.dirname(self.path)
            if dir != "":
                os.makedirs(dir, exist_ok=True)
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.info, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Ошибка при записи в файл: {e}")
    
    @_save_to_file
    def add(self, state, model_name, timestamp, f0_method, pitch):
        self.info[f"{timestamp} / {model_name} / {f0_method} / {pitch}"] = state
    
    @_save_to_file
    def clear(self):
        self.info = {}
    
    def get_list(self):
        return sorted([key for key in self.info], reverse=True)
    
    def get(self, key):
        return self.info.get(key, [])
    
    def load_from_file(self):
        """Загрузить историю из файла"""
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                self.info = json.load(f)

class Vbach(GradioHelper):
    def __init__(self, user_directory, device):
        super().__init__()
        self.device = device
        self.pitch_methods = f0_methods
        self.hop_length_values = (8, 512)
        self.index_rates_values = (0, 1)
        self.filter_radius_values = (0, 7)
        self.protect_values = (0, 0.5)
        self.rms_values = (0, 1)
        self.f0_min_values = (50, 3000)
        self.f0_max_values = (300, 6000)
        self.fairseq_embedders = list(
            model_manager.huberts_fairseq_dict.keys()
        )
        self.transformers_embedders = list(
            model_manager.huberts_transformers_dict.keys()
        )
        self.last_converted_state = []
        self.input_files = []
        self.user_directory = user_directory

        model_manager.__init__(self.user_directory)
        self.input_base_dir = os.path.join(user_directory.path, "input")
        self.inputs_json_path = os.path.join(self.input_base_dir, "inputs.json")
        self.output_base_dir = os.path.join(user_directory.path, "output", "vbach")
        self.history = History(self.user_directory)
        self.load_from_file()

    def _write_file(self):
        """Записывает текущее состояние в файл"""
        try:
            with open(self.inputs_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.input_files, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Ошибка при записи в файл: {e}")

    def _save_to_file(func):
        """Декоратор для автоматического сохранения после вызова метода"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self._write_file()
            return result
        return wrapper

    def load_from_file(self):
        """Загрузить историю из файла"""
        if os.path.exists(self.inputs_json_path):
            with open(self.inputs_json_path, 'r', encoding='utf-8') as f:
                self.input_files = json.load(f)

    @_save_to_file
    def clean(self):
        self.input_files = []

    @_save_to_file
    def upload_files(self, input_files: list, copy: bool = False):
        if input_files: 
            input_dir = os.path.join(self.input_base_dir, datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S"))
            os.makedirs(input_dir, exist_ok=True)
            valid_files = [file for file in input_files if check(file)]
            valid_files_moved = []
            if valid_files:
                for file in valid_files:
                    basename = os.path.basename(file)
                    output_path = os.path.join(input_dir, basename)
                    if copy:
                        shutil.copy(file, output_path)
                    else:
                       shutil.move(file, output_path)
                    valid_files_moved.append(output_path)
                    self.input_files.append(output_path)
            return valid_files_moved
        else:
            return []

    def UI(self):
        with gr.Blocks() as vbach_app:
            with gr.Tab("Инференс"):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            upload = gr.Files(show_label=False, type="filepath", interactive=True)
                            refresh_input_btn = gr.Button("Обновить", variant="primary", interactive=True)
                            list_input_files = gr.Dropdown(
                                label="Загрузить файлы",
                                choices=reversed(self.input_files),
                                value=[],
                                multiselect=True,
                                interactive=True,
                                filterable=False, scale=15
                            )
                            gr.on(fn=lambda: gr.update(choices=reversed(self.input_files), value=[]), outputs=list_input_files, trigger_mode="once")
                            refresh_input_btn.click(lambda: gr.update(choices=reversed(self.input_files), value=[]), outputs=list_input_files)
                                
                            @upload.upload(inputs=[upload], outputs=[list_input_files, upload])
                            def upload_files(input_files):
                                files = self.upload_files(input_files)
                                return gr.update(
                                    choices=reversed(self.input_files), value=files
                                ), gr.update(value=[])
                            converted_state = gr.Textbox(
                                label="Состояние разделения",
                                interactive=False,
                                value="",
                                visible=False,
                            )

                    with gr.Column():
                        with gr.Group():
                            with gr.Group():
                                model_name = gr.Dropdown(label="Имя модели", interactive=True)
                                model_list_refresh_btn = gr.Button(
                                    "Обновить", variant="secondary", interactive=True
                                )

                                @model_list_refresh_btn.click(outputs=[model_name])
                                def refresh_list_voice_models():
                                    models = []
                                    models = model_manager.parse_voice_models()
                                    first_model = None
                                    if len(models) > 0:
                                        first_model = models[0]
                                    return gr.update(choices=models, value=first_model)

                            with gr.Group():
                                pitch_method = gr.Dropdown(
                                    label="Метод извлечения высоты тона",
                                    choices=self.pitch_methods,
                                    value=self.pitch_methods[0],
                                    interactive=True,
                                    filterable=False
                                )
                                pitch = gr.Slider(
                                    label="Высота тона",
                                    minimum=-48,
                                    maximum=48,
                                    step=0.5,
                                    value=0,
                                    interactive=True,
                                )
                                hop_length = gr.Slider(
                                    label="Длина шага",
                                    info="Длина шага влияет на точность передачи высоты тона\nЧем меньше длина шага - тем точнее будет передана высота тона",
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
                                def show_mangio_crepe_hop_length(pitch_method):
                                    return gr.update(
                                        visible=(
                                            True
                                            if pitch_method
                                            in ["mangio-crepe", "mangio-crepe-tiny", "pyin"]
                                            else False
                                        )
                                    )

                            with gr.Accordion(label="Дополнительные настройки", open=False):
                                with gr.Group():
                                    with gr.Accordion(label="Обработка аудио", open=False):
                                        with gr.Group():
                                            stereo_mode = gr.Radio(
                                                choices=["mono", "left/right", "sim/dif"],
                                                label="Стерео режим",
                                                info="mono - монофоническая обработка аудио, \nleft/right - обработка левого и правого каналов отдельно, \nsim/dif - обработка фантомного центра и стерео-базы, разделенную на левый и правый каналы",
                                                value="mono",
                                                interactive=True,
                                            )
                                            alt_pl = gr.Checkbox(
                                                label="Альтернативный пайплайн",
                                                info="Аудио нарезается на фиксированные чанки с перекрытием, что исключает любые щелчки на выходе (исключение - если есть щелчки в самой модели из-за грязного датасета)\nРазмер чанка вычисляется на основе 20% свободной видеопамяти",
                                                value=False,
                                                interactive=True,
                                            )
                                    with gr.Accordion(label="Инференс", open=False):
                                        with gr.Group():
                                            with gr.Row():
                                                index_rate = gr.Slider(
                                                    label="Влияние индекса",
                                                    info="Чем ниже значение, тем больше голос похож на исходный; чем выше, тем ближе к модели",
                                                    minimum=self.index_rates_values[0],
                                                    maximum=self.index_rates_values[1],
                                                    step=0.05,
                                                    value=0,
                                                    interactive=True,
                                                )
                                                filter_radius = gr.Slider(
                                                    label="Радиус фильтра",
                                                    info="Сглаживает результаты извлечения тона\nМожет снизить дыхание и шумы на выходе",
                                                    minimum=self.filter_radius_values[0],
                                                    maximum=self.filter_radius_values[1],
                                                    step=1,
                                                    value=3,
                                                    interactive=True,
                                                )
                                            with gr.Row():
                                                rms = gr.Slider(
                                                    label="Соотношение огибающих громкости",
                                                    info="Значение 0 - огибающая громкости как у входного аудио, 1 - как у выходного сигнала",
                                                    minimum=self.rms_values[0],
                                                    maximum=self.rms_values[1],
                                                    step=0.05,
                                                    value=0.25,
                                                    interactive=True,
                                                )
                                                protect = gr.Slider(
                                                    label="Защита согласных",
                                                    info="Предовращает роботизацию дыхания и согласных (Может влиять на четкость речи)\nЗначение 0.5 - выключает защиту, 0 - максимальная защита",
                                                    minimum=self.protect_values[0],
                                                    maximum=self.protect_values[1],
                                                    step=0.05,
                                                    value=0.35,
                                                    interactive=True,
                                                )
                                    with gr.Accordion(label="Диапазон определения высоты тона", open=False):
                                        with gr.Group():
                                            with gr.Row():
                                                f0_min = gr.Slider(
                                                    label="Нижний предел диапазона определения высоты тона",
                                                    minimum=self.f0_min_values[0],
                                                    maximum=self.f0_min_values[1],
                                                    step=10,
                                                    value=50,
                                                    interactive=True,
                                                )
                                                f0_max = gr.Slider(
                                                    label="Верхний предел диапазона определения высоты тона",
                                                    minimum=self.f0_max_values[0],
                                                    maximum=self.f0_max_values[1],
                                                    step=10,
                                                    value=1100,
                                                    interactive=True,
                                                )
                                    with gr.Accordion(label="Эмбеддер", open=False):
                                        with gr.Group():
                                            embedder_name = gr.Radio(
                                                label="Модель Hubert",
                                                choices=self.fairseq_embedders,
                                                value=self.fairseq_embedders[0],
                                            )
                                            transformers_mode = gr.Checkbox(
                                                label="Использовать стек Transformers",
                                                value=False,
                                                interactive=True,
                                            )

                                        @transformers_mode.change(
                                            inputs=[transformers_mode], outputs=[embedder_name]
                                        )
                                        def change_embedders(tr_m):
                                            if tr_m:
                                                return gr.update(
                                                    value=self.transformers_embedders[0],
                                                    choices=self.transformers_embedders,
                                                )
                                            else:
                                                return gr.update(
                                                    choices=self.fairseq_embedders,
                                                    value=self.fairseq_embedders[0],
                                                )

                                    with gr.Accordion(label="Имя выходного файла", open=False):
                                        with gr.Group():
                                            output_name = gr.Textbox(
                                                label="Имя выходного файла",
                                                interactive=True,
                                                value="NAME - MODEL - F0METHOD - PITCH",
                                            )
                                            format_output_name_check = gr.Checkbox(
                                                label="Форматировать имя",
                                                info="Используйте ключи: \nNAME - имя входного файла без расширения, \nPITCH - высота тона, \nF0METHOD - метод извлечения высота тона, \nMODEL - имя голосовой модели",
                                                value=True,
                                                interactive=True,
                                            )

                            with gr.Group():
                                output_format = gr.Dropdown(
                                    label="Формат выходного файла",
                                    interactive=True,
                                    choices=output_formats,
                                    value=output_formats[0],
                                    filterable=False,
                                )
                                status = gr.Textbox(
                                    container=False, lines=4, interactive=False, max_lines=4, visible=False
                                )
                                convert_btn = gr.Button(
                                    "Преобразовать", variant="primary", interactive=True
                                ).click(lambda: gr.update(visible=True), outputs=[status])
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
                )
                def vbach_convert_batch(
                    ifl,
                    mn,
                    pm,
                    p,
                    hl,
                    ir,
                    fr,
                    rms,
                    pr,
                    f0min,
                    f0max,
                    on,
                    fn,
                    of,
                    sm,
                    alt_pipeline,
                    em_n,
                    tr_m,
                ):
                    output_converted_files = []
                    progress = gr.Progress(track_tqdm=True)
                    progress(
                        progress=0, desc=f"Начало преобразования"
                    )
                    timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
                    if ifl:
                        for i, file in enumerate(ifl, start=1):
                            try:
                                print(f"Файл {i} из {len(ifl)}: {file}")
                                progress(
                                    progress=(i / len(ifl)), desc=f"Файл {i} из {len(ifl)}"
                                )
                                gr.Warning(title=f"Файл {i} из {len(ifl)}: {file}", message="")
                                out_conv = vbach_inference(
                                    input_file=file,
                                    model_name=mn,
                                    output_dir=os.path.join(self.output_base_dir, timestamp),
                                    output_name=on,
                                    format_name=True if len(ifl) > 1 else fn,
                                    output_format=of,
                                    pitch=p,
                                    method_pitch=pm,
                                    output_bitrate=320,
                                    add_params={
                                        "index_rate": ir,
                                        "filter_radius": fr,
                                        "protect": pr,
                                        "rms": rms,
                                        "mangio_crepe_hop_length": hl,
                                        "f0_min": f0min,
                                        "f0_max": f0max,
                                        "stereo_mode": sm,
                                    },
                                    pipeline_mode="alt" if alt_pipeline == True else "orig",
                                    embedder_name=em_n,
                                    stack="transformers" if tr_m == True else "fairseq",
                                    add_text_progress=f"{i} из {len(ifl)}",
                                    device=self.device
                                )
                                output_converted_files.append(out_conv)
                            except Exception as e:
                                print(e)
                    if output_converted_files:
                        self.history.add(output_converted_files, mn, timestamp, pm, p)
                    return gr.update(value=str(output_converted_files)), gr.update(visible=False)

                with gr.Column(variant="panel"):
                    gr.Markdown("<center><h3>Результаты</h3></center>")

                    with gr.Group():
                        with gr.Row(equal_height=True):
                            list_conversions = gr.Dropdown(
                                label="Выберите результаты преобразования",
                                choices=[],
                                value=None,
                                interactive=True, scale=14
                            )
                            list_conversions.change(lambda x: gr.update(value=str(self.history.get(x))), inputs=[list_conversions], outputs=[converted_state])
                            refresh_conversions_btn = gr.Button("Обновить", scale=2, interactive=True)
                            refresh_conversions_btn.click(lambda: gr.update(choices=self.history.get_list(), value=None), outputs=[list_conversions])
                            gr.on(fn=lambda: gr.update(choices=self.history.get_list(), value=None), outputs=[list_conversions])

                @gr.render(inputs=[converted_state])
                def show_players_converted(state):
                    if state != "":
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
            with gr.TabItem("Дуэт"):
                with gr.Column():
                    with gr.Group():
                        upload_duet = gr.File(show_label=False, type="filepath", interactive=True)
                        refresh_input_btn_duet = gr.Button("Обновить", variant="primary", interactive=True)
                        list_input_files_duet = gr.Dropdown(
                            label="Загрузить файл",
                            choices=self.input_files,
                            value=None,
                            multiselect=False,
                            interactive=True,
                            filterable=False, scale=15
                        )
                        gr.on(fn=lambda: gr.update(choices=reversed(self.input_files), value=None), outputs=list_input_files_duet, trigger_mode="once")
                        refresh_input_btn_duet.click(lambda: gr.update(choices=reversed(self.input_files), value=None), outputs=list_input_files_duet)
                            
                        @upload_duet.upload(inputs=[upload_duet], outputs=[list_input_files_duet, upload_duet])
                        def upload_files(input_file):
                            files = self.upload_files([input_file])
                            return gr.update(
                                choices=reversed(self.input_files), value=files[0]
                            ), gr.update(value=None)
                        

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("<h3><center>Модель 1</center></h3>")
                            with gr.Group():
                                model_name1 = gr.Dropdown(label="Имя модели", interactive=True)

                                pitch_method1 = gr.Dropdown(
                                    label="Метод извлечения высоты тона",
                                    choices=self.pitch_methods,
                                    value=self.pitch_methods[0],
                                    interactive=True,
                                    filterable=False
                                )
                                pitch1 = gr.Slider(
                                    label="Высота тона",
                                    minimum=-48,
                                    maximum=48,
                                    step=0.5,
                                    value=0,
                                    interactive=True,
                                )
                                hop_length1 = gr.Slider(
                                    label="Длина шага",
                                    info="Длина шага влияет на точность передачи высоты тона\nЧем меньше длина шага - тем точнее будет передана высота тона",
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
                                def show_mangio_crepe_hop_length(pitch_method):
                                    return gr.update(
                                        visible=(
                                            True
                                            if pitch_method
                                            in ["mangio-crepe", "mangio-crepe-tiny", "pyin"]
                                            else False
                                        )
                                    )

                                with gr.Accordion(label="Дополнительные настройки", open=False):
                                    with gr.Group():
                                        with gr.Accordion(label="Инференс", open=False):
                                            with gr.Group():
                                                with gr.Row():
                                                    index_rate1 = gr.Slider(
                                                        label="Влияние индекса",
                                                        info="Чем ниже значение, тем больше голос похож на исходный; чем выше, тем ближе к модели",
                                                        minimum=self.index_rates_values[0],
                                                        maximum=self.index_rates_values[1],
                                                        step=0.05,
                                                        value=0,
                                                        interactive=True,
                                                    )
                                                    filter_radius1 = gr.Slider(
                                                        label="Радиус фильтра",
                                                        info="Сглаживает результаты извлечения тона\nМожет снизить дыхание и шумы на выходе",
                                                        minimum=self.filter_radius_values[0],
                                                        maximum=self.filter_radius_values[1],
                                                        step=1,
                                                        value=3,
                                                        interactive=True,
                                                    )
                                                with gr.Row():
                                                    rms1 = gr.Slider(
                                                        label="Соотношение огибающих громкости",
                                                        info="Значение 0 - огибающая громкости как у входного аудио, 1 - как у выходного сигнала",
                                                        minimum=self.rms_values[0],
                                                        maximum=self.rms_values[1],
                                                        step=0.05,
                                                        value=0.25,
                                                        interactive=True,
                                                    )
                                                    protect1 = gr.Slider(
                                                        label="Защита согласных",
                                                        info="Предовращает роботизацию дыхания и согласных (Может влиять на четкость речи)\nЗначение 0.5 - выключает защиту, 0 - максимальная защита",
                                                        minimum=self.protect_values[0],
                                                        maximum=self.protect_values[1],
                                                        step=0.05,
                                                        value=0.35,
                                                        interactive=True,
                                                    )
                                        with gr.Accordion(label="Диапазон определения высоты тона", open=False):
                                            with gr.Group():
                                                with gr.Row():
                                                    f0_min1 = gr.Slider(
                                                        label="Нижний предел диапазона определения высоты тона",
                                                        minimum=self.f0_min_values[0],
                                                        maximum=self.f0_min_values[1],
                                                        step=10,
                                                        value=50,
                                                        interactive=True,
                                                    )
                                                    f0_max1 = gr.Slider(
                                                        label="Верхний предел диапазона определения высоты тона",
                                                        minimum=self.f0_max_values[0],
                                                        maximum=self.f0_max_values[1],
                                                        step=10,
                                                        value=1100,
                                                        interactive=True,
                                                    )
                                        with gr.Accordion(label="Эмбеддер", open=False):
                                            with gr.Group():
                                                embedder_name1 = gr.Radio(
                                                    label="Модель Hubert",
                                                    choices=self.fairseq_embedders,
                                                    value=self.fairseq_embedders[0],
                                                )
                                                transformers_mode1 = gr.Checkbox(
                                                    label="Использовать стек Transformers",
                                                    value=False,
                                                    interactive=True,
                                                )

                                            @transformers_mode1.change(
                                                inputs=[transformers_mode1], outputs=[embedder_name1]
                                            )
                                            def change_embedders(tr_m):
                                                if tr_m:
                                                    return gr.update(
                                                        value=self.transformers_embedders[0],
                                                        choices=self.transformers_embedders,
                                                    )
                                                else:
                                                    return gr.update(
                                                        choices=self.fairseq_embedders,
                                                        value=self.fairseq_embedders[0],
                                                    )

                        with gr.Column():
                            gr.Markdown("<h3><center>Модель 2</center></h3>")
                            with gr.Group():
                                model_name2 = gr.Dropdown(label="Имя модели", interactive=True)

                                pitch_method2 = gr.Dropdown(
                                    label="Метод извлечения высоты тона",
                                    choices=self.pitch_methods,
                                    value=self.pitch_methods[0],
                                    interactive=True,
                                    filterable=False
                                )
                                pitch2 = gr.Slider(
                                    label="Высота тона",
                                    minimum=-48,
                                    maximum=48,
                                    step=0.5,
                                    value=0,
                                    interactive=True,
                                )
                                hop_length2 = gr.Slider(
                                    label="Длина шага",
                                    info="Длина шага влияет на точность передачи высоты тона\nЧем меньше длина шага - тем точнее будет передана высота тона",
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
                                def show_mangio_crepe_hop_length(pitch_method):
                                    return gr.update(
                                        visible=(
                                            True
                                            if pitch_method
                                            in ["mangio-crepe", "mangio-crepe-tiny", "pyin"]
                                            else False
                                        )
                                    )

                                with gr.Accordion(label="Дополнительные настройки", open=False):
                                    with gr.Group():
                                        with gr.Accordion(label="Инференс", open=False):
                                            with gr.Group():
                                                with gr.Row():
                                                    index_rate2 = gr.Slider(
                                                        label="Влияние индекса",
                                                        info="Чем ниже значение, тем больше голос похож на исходный; чем выше, тем ближе к модели",
                                                        minimum=self.index_rates_values[0],
                                                        maximum=self.index_rates_values[1],
                                                        step=0.05,
                                                        value=0,
                                                        interactive=True,
                                                    )
                                                    filter_radius2 = gr.Slider(
                                                        label="Радиус фильтра",
                                                        info="Сглаживает результаты извлечения тона\nМожет снизить дыхание и шумы на выходе",
                                                        minimum=self.filter_radius_values[0],
                                                        maximum=self.filter_radius_values[1],
                                                        step=1,
                                                        value=3,
                                                        interactive=True,
                                                    )
                                                with gr.Row():
                                                    rms2 = gr.Slider(
                                                        label="Соотношение огибающих громкости",
                                                        info="Значение 0 - огибающая громкости как у входного аудио, 1 - как у выходного сигнала",
                                                        minimum=self.rms_values[0],
                                                        maximum=self.rms_values[1],
                                                        step=0.05,
                                                        value=0.25,
                                                        interactive=True,
                                                    )
                                                    protect2 = gr.Slider(
                                                        label="Защита согласных",
                                                        info="Предовращает роботизацию дыхания и согласных (Может влиять на четкость речи)\nЗначение 0.5 - выключает защиту, 0 - максимальная защита",
                                                        minimum=self.protect_values[0],
                                                        maximum=self.protect_values[1],
                                                        step=0.05,
                                                        value=0.35,
                                                        interactive=True,
                                                    )
                                        with gr.Accordion(label="Диапазон определения высоты тона", open=False):
                                            with gr.Group():
                                                with gr.Row():
                                                    f0_min2 = gr.Slider(
                                                        label="Нижний предел диапазона определения высоты тона",
                                                        minimum=self.f0_min_values[0],
                                                        maximum=self.f0_min_values[1],
                                                        step=10,
                                                        value=50,
                                                        interactive=True,
                                                    )
                                                    f0_max2 = gr.Slider(
                                                        label="Верхний предел диапазона определения высоты тона",
                                                        minimum=self.f0_max_values[0],
                                                        maximum=self.f0_max_values[1],
                                                        step=10,
                                                        value=1100,
                                                        interactive=True,
                                                    )
                                        with gr.Accordion(label="Эмбеддер", open=False):
                                            with gr.Group():
                                                embedder_name2 = gr.Radio(
                                                    label="Модель Hubert",
                                                    choices=self.fairseq_embedders,
                                                    value=self.fairseq_embedders[0],
                                                )
                                                transformers_mode2 = gr.Checkbox(
                                                    label="Использовать стек Transformers",
                                                    value=False,
                                                    interactive=True,
                                                )

                                            @transformers_mode2.change(
                                                inputs=[transformers_mode2], outputs=[embedder_name2]
                                            )
                                            def change_embedders(tr_m):
                                                if tr_m:
                                                    return gr.update(
                                                        value=self.transformers_embedders[0],
                                                        choices=self.transformers_embedders,
                                                    )
                                                else:
                                                    return gr.update(
                                                        choices=self.fairseq_embedders,
                                                        value=self.fairseq_embedders[0],
                                                    )

                    with gr.Group():
                        model_list_refresh_btn = gr.Button(
                            "Обновить список моделей", variant="secondary", interactive=True
                        )
                        @model_list_refresh_btn.click(outputs=[model_name1, model_name2])
                        def refresh_list_voice_models():
                            models = []
                            models = model_manager.parse_voice_models()
                            first_model = None
                            if len(models) > 0:
                                first_model = models[0]
                            return gr.update(choices=models, value=first_model), gr.update(choices=models, value=first_model)
                        stereo_mode_duet = gr.Radio(
                            choices=["mono", "left/right", "sim/dif"],
                            label="Стерео режим",
                            info="mono - монофоническая обработка аудио, \nleft/right - обработка левого и правого каналов отдельно, \nsim/dif - обработка фантомного центра и стерео-базы, разделенную на левый и правый каналы",
                            value="mono",
                            interactive=True,
                        )
                        alt_pl_duet = gr.Checkbox(
                            label="Альтернативный пайплайн",
                            info="Аудио нарезается на фиксированные чанки с перекрытием, что исключает любые щелчки на выходе (исключение - если есть щелчки в самой модели из-за грязного датасета)\nРазмер чанка вычисляется на основе 20% свободной видеопамяти",
                            value=False,
                            interactive=True,
                        )
                        mix_duet = gr.Checkbox(
                            label="Смешать два голоса в один выходной файл",
                            value=False,
                            interactive=True,
                        )
                        mix_duet_ratio = gr.Slider(
                            label="Баланс между двумя голосами",
                            info="Регулирует громкость между первым и вторым голосом: значение -1 = только первый голос, значение 1 = только второй голос",
                            minimum=-1,
                            maximum=1,
                            step=0.05,
                            value=0,
                            interactive=True,
                            visible=False
                        )

                        output_format_duet = gr.Dropdown(
                            label="Формат выходного файла",
                            interactive=True,
                            choices=output_formats,
                            value=output_formats[0],
                            filterable=False,
                        )
                        status_duet = gr.Textbox(
                            container=False, lines=3, interactive=False, max_lines=3, visible=False
                        )
                        convert_btn_duet = gr.Button(
                            "Преобразовать", variant="primary", interactive=True
                        ).click(lambda: gr.update(visible=True), outputs=[status_duet])
                    with gr.Row(equal_height=True):
                        output_duet_audio_1 = gr.Audio(
                            label="Результат модели 1",
                            type="filepath",
                            interactive=False,
                            show_download_button=True,
                        )
                        output_duet_audio_2 = gr.Audio(
                            label="Результат модели 2",
                            type="filepath",
                            interactive=False,
                            show_download_button=True,
                        )
                        @mix_duet.change(inputs=mix_duet, outputs=[mix_duet_ratio, output_duet_audio_1, output_duet_audio_2])
                        def mix_duet_change_fn(x):
                            match x:
                                case True:
                                    return gr.update(visible=x), gr.update(label="Общий результат", value=None), gr.update(visible=False, value=None)
                                case False:
                                    return gr.update(visible=x), gr.update(label="Результат модели 1", value=None), gr.update(visible=True, value=None)

                    @convert_btn_duet.then(
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
                        outputs=[output_duet_audio_1, output_duet_audio_2, status_duet],
                    )
                    def vbach_convert_duet(
                        ifile_,
                        mn1,
                        mn2,
                        pm1,
                        pm2,
                        p1,
                        p2,
                        hl1,
                        hl2,
                        ir1,
                        ir2,
                        fr1,
                        fr2,
                        rms1,
                        rms2,
                        pr1,
                        pr2,
                        f0min1,
                        f0min2,
                        f0max1,
                        f0max2,
                        of,
                        sm,
                        alt_pipeline,
                        em_n1,
                        em_n2,
                        tr_m1,
                        tr_m2,
                        mix_d,
                        mix_d_ratio
                    ):
                        output_1 = None
                        output_2 = None
                        output_mixed = None
                        progress = gr.Progress(track_tqdm=True)
                        progress(
                            progress=0, desc=f"Начало преобразования"
                        )
                        
                        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
                        output_dir = os.path.join(self.output_base_dir, timestamp)
                        if ifile_:
                            try:
                                gr.Warning(title=f"Модель 1", message="")
                                output_1 = vbach_inference(
                                    input_file=ifile_,
                                    model_name=mn1,
                                    output_dir=output_dir,
                                    output_name="NAME - MODEL 1 - F0METHOD - PITCH",
                                    format_name=True,
                                    output_format=of,
                                    pitch=p1,
                                    method_pitch=pm1,
                                    output_bitrate=320,
                                    add_params={
                                        "index_rate": ir1,
                                        "filter_radius": fr1,
                                        "protect": pr1,
                                        "rms": rms1,
                                        "mangio_crepe_hop_length": hl1,
                                        "f0_min": f0min1,
                                        "f0_max": f0max1,
                                        "stereo_mode": sm,
                                    },
                                    pipeline_mode="alt" if alt_pipeline == True else "orig",
                                    embedder_name=em_n1,
                                    stack="transformers" if tr_m1 == True else "fairseq",
                                    add_text_progress=f"Модель 1",
                                    device=self.device
                                )
                                gr.Warning(title=f"Модель 2", message="")
                                output_2 = vbach_inference(
                                    input_file=ifile_,
                                    model_name=mn2,
                                    output_dir=output_dir,
                                    output_name="NAME - MODEL 2 - F0METHOD - PITCH",
                                    format_name=True,
                                    output_format=of,
                                    pitch=p2,
                                    method_pitch=pm2,
                                    output_bitrate=320,
                                    add_params={
                                        "index_rate": ir2,
                                        "filter_radius": fr2,
                                        "protect": pr2,
                                        "rms": rms2,
                                        "mangio_crepe_hop_length": hl2,
                                        "f0_min": f0min2,
                                        "f0_max": f0max2,
                                        "stereo_mode": sm,
                                    },
                                    pipeline_mode="alt" if alt_pipeline == True else "orig",
                                    embedder_name=em_n2,
                                    stack="transformers" if tr_m2 == True else "fairseq",
                                    add_text_progress=f"Модель 2",
                                    device=self.device
                                )

                            except Exception as e:
                                print(e)
                                return gr.update(value=None), gr.update(value=None), gr.update(visible=False)

                        match mix_d:
                            case True:
                                input_file_basename = os.path.splitext(os.path.basename(ifile_))[0]
                                mix1, sr1 = read(output_1)
                                mix2, sr2 = read(output_2)
                                max_sr = max(sr1, sr2)
                                fited_arrays = fit_arrays([mix1, mix2], [sr1, sr2], min_sr=max_sr)
                                g1 = (1 - mix_d_ratio) / 2
                                g2 = (1 + mix_d_ratio) / 2
                                mixed_duet = gain(fited_arrays[0], g1) + gain(fited_arrays[1], g2)
                                shorted_name = namer.short(input_file_basename, length=50)
                                sanitized_name = namer.sanitize(f"{mn1}, {mn2} - {shorted_name}")
                                output_mixed = write(os.path.join(output_dir, f"{sanitized_name}.{of}"), mixed_duet, max_sr)
                                self.history.add([output_mixed], f"{mn1}|{mn2}", timestamp, f"{pm1}|{pm2}", f"{p1}|{p2}")
                                return self.return_audio_with_size(label="Общий результат", value=output_mixed), gr.update(label="Результат модели 2", value=None), gr.update(visible=False)
                            case False:
                                self.history.add([output_1, output_2], f"{mn1}|{mn2}", timestamp, f"{pm1}|{pm2}", f"{p1}|{p2}")
                                return self.return_audio_with_size(label="Результат модели 1", value=output_1), self.return_audio_with_size(label="Результат модели 2", value=output_2), gr.update(visible=False)

            with gr.TabItem("Менеджер"):
                with gr.TabItem("Загрузить по ссылке"):
                    with gr.TabItem("Через zip файл"):
                        with gr.Group():
                            url_zip = gr.Text(label="Ссылка на zip файл", interactive=True)
                            url_zip_model_name = gr.Text(
                                label="Имя модели", interactive=True
                            )
                            url_zip_download_btn = gr.Button(
                                "Загрузить", variant="primary", interactive=True
                            )
                            url_zip_output = gr.Text(
                                label="Статус", interactive=False, lines=5
                            )
                            url_zip_download_btn.click(
                                (
                                    lambda x, y: model_manager.install_model_zip(
                                        x,
                                        namer.short(
                                            namer.sanitize(y), length=40
                                        ),
                                        "url",
                                    )
                                ),
                                inputs=[url_zip, url_zip_model_name],
                                outputs=url_zip_output,
                            )

                    with gr.TabItem("Через отдельные файлы"):
                        with gr.Group():
                            url_pth = gr.Text(label="Ссылка на *.pth файл", interactive=True)
                            url_index = gr.Text(
                                label="Ссылка на *.index файл (необязательно)", interactive=True
                            )
                            url_file_model_name = gr.Text(
                                label="Имя модели", interactive=True
                            )
                            url_file_download_btn = gr.Button(
                                "Загрузить", variant="primary", interactive=True
                            )
                            url_file_output = gr.Text(
                                label="Статус", interactive=False, lines=5
                            )
                            url_file_download_btn.click(
                                (
                                    lambda x, y, z: model_manager.install_model_files(
                                        x,
                                        y,
                                        namer.short(
                                            namer.sanitize(z), length=40
                                        ),
                                        "url",
                                    )
                                ),
                                inputs=[url_index, url_pth, url_file_model_name],
                                outputs=url_file_output,
                            )

                with gr.Tab("Загрузить с устройства"):
                    with gr.Tab("Через zip файл"):
                        with gr.Group():
                            local_zip = gr.File(
                                label="zip файл",
                                file_types=[".zip"],
                                file_count="single",
                                interactive=True
                            )
                            local_zip_model_name = gr.Text(
                                label="Имя модели", interactive=True
                            )
                            local_zip_upload_btn = gr.Button(
                                "Загрузить", variant="primary", interactive=True
                            )
                            local_zip_output = gr.Text(
                                label="Статус", interactive=False, lines=5
                            )
                            local_zip_upload_btn.click(
                                (
                                    lambda x, y: model_manager.install_model_zip(
                                        x,
                                        namer.short(
                                            namer.sanitize(y), length=40
                                        ),
                                        "local",
                                    )
                                ),
                                inputs=[local_zip, local_zip_model_name],
                                outputs=local_zip_output,
                            )

                    with gr.TabItem("Через отдельные файлы"):
                        with gr.Group():
                            with gr.Row():
                                local_pth = gr.File(
                                    label="*.pth файл",
                                    file_types=[".pth"],
                                    file_count="single",
                                    interactive=True
                                )
                                local_index = gr.File(
                                    label="*.index файл (необязательно)",
                                    file_types=[".index"],
                                    file_count="single",
                                    interactive=True
                                )
                            local_file_model_name = gr.Text(
                                label="Имя модели", interactive=True
                            )
                            local_file_upload_btn = gr.Button(
                                "Загрузить", variant="primary", interactive=True
                            )
                            local_file_output = gr.Text(
                                label="Статус", interactive=False, lines=5
                            )
                            local_file_upload_btn.click(
                                (
                                    lambda x, y, z: model_manager.install_model_files(
                                        x,
                                        y,
                                        namer.short(
                                            namer.sanitize(z), length=40
                                        ),
                                        "local",
                                    )
                                ),
                                inputs=[local_index, local_pth, local_file_model_name],
                                outputs=local_file_output,
                            )

                with gr.TabItem("Удалить модель"):
                    with gr.Group():
                        delete_model_name = gr.Dropdown(
                            label="Имя модели",
                            choices=model_manager.parse_voice_models(),
                            interactive=True,
                            filterable=False,
                        )
                        delete_refresh_btn = gr.Button("Обновить", interactive=True)
                        delete_btn = gr.Button("Удалить", variant="stop", interactive=True)
                        @delete_refresh_btn.click(
                            inputs=None, outputs=delete_model_name
                        )
                        def refresh_list_voice_models():
                            models = []
                            models = model_manager.parse_voice_models()
                            first_model = None
                            if len(models) > 0:
                                first_model = models[0]
                            return gr.update(choices=models, value=first_model)

                        delete_output = gr.Text(
                            label="Статус", interactive=False, lines=5
                        )
                        delete_btn.click(
                            fn=model_manager.del_voice_model,
                            inputs=delete_model_name,
                            outputs=delete_output,
                        )

                @gr.on(fn="decorator", inputs=None, outputs=[delete_model_name, model_name, model_name1, model_name2])
                def refresh_list_voice_models():
                    models = []
                    models = model_manager.parse_voice_models()
                    first_model = None
                    if len(models) > 0:
                        first_model = models[0]
                    return gr.update(choices=models, value=first_model), gr.update(choices=models, value=first_model), gr.update(choices=models, value=first_model), gr.update(choices=models, value=first_model)
        return vbach_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vbach - форк Polgen-RVC 1.2.0")
    
    # Основные подкоманды
    subparsers = parser.add_subparsers(dest="mode", help="Режим работы", required=True)
    
    # CLI режим
    cli_parser = subparsers.add_parser("cli", help="Консольный режим")
    cli_parser.add_argument("--input", nargs="*", help="Путь к входному файлу или папке")
    cli_parser.add_argument(
        "--output_dir", type=str, required=True, help="Путь для сохранения результатов"
    )
    cli_parser.add_argument(
        "--output_format",
        type=str,
        default="wav",
        choices=output_formats,
        help="Формат выходных файлов",
    )
    cli_parser.add_argument(
        "--output_bitrate", type=str, default="320k", help="Битрейт выходного файла"
    )
    cli_parser.add_argument(
        "--format_name",
        action="store_true",
        help="Форматировать имя выходного файла",
    )
    cli_parser.add_argument(
        "--output_name",
        type=str,
        default="NAME_STEM",
        help="Имя выходного файла",
    )
    cli_parser.add_argument(
        "--model_name",
        type=str,
        default="model",
        help="Имя голосовой модели",
    )

    cli_parser.add_argument(
        "--index_rate",
        type=float,
        default=0,
        help="Интенсивность использования индексного файла (от 0.0 до 1.0)",
        metavar="[0.0-1.0]",
    )
    cli_parser.add_argument(
        "--stereo_mode",
        type=str,
        default="mono",
        choices=["mono", "left/right", "sim/dif"],
        help="Режим каналов: моно или стерео",
    )
    cli_parser.add_argument(
        "--method_pitch",
        type=str,
        default="rmvpe+",
        help="Метод извлечения pitch (тона)",
    )
    cli_parser.add_argument(
        "--pitch", type=int, default=0, help="Корректировка тона в полутонах"
    )
    cli_parser.add_argument(
        "--hop_length",
        type=int,
        default=128,
        help="Длина hop (в семплах) для обработки",
    )
    cli_parser.add_argument(
        "--filter_radius", type=int, default=3, help="Радиус фильтра для сглаживания"
    )
    cli_parser.add_argument(
        "--rms",
        type=float,
        default=0.25,
        help="Масштабирование огибающей громкости (RMS)",
    )
    cli_parser.add_argument(
        "--protect", type=float, default=0.33, help="Защита для глухих согласных звуков"
    )
    cli_parser.add_argument(
        "--f0_min", type=int, default=50, help="Минимальная частота pitch (F0) в Hz"
    )
    cli_parser.add_argument(
        "--f0_max", type=int, default=1100, help="Максимальная частота pitch (F0) в Hz"
    )
    cli_parser.add_argument(
        "--alt_pipeline",
        action="store_true",
        help="Альтернативный пайплайн",
    )
    cli_parser.add_argument(
        "--use_transformers",
        action="store_true",
        help="Использовать transformers",
    )
    cli_parser.add_argument(
        "--embedder_name",
        type=str,
        default="hubert_base",
        help="Имя Hubert модели",
    )
    
    # App режим
    app_parser = subparsers.add_parser("app", help="Веб-интерфейс (Gradio)")
    app_parser.add_argument(
        "--port", 
        type=int, 
        default=7860, 
        help="Порт для запуска сервера Gradio (по умолчанию: 7860)"
    )
    app_parser.add_argument(
        "--share",
        action="store_true",
        help="Создать публичную ссылку для приложения Gradio",
    )
    app_parser.add_argument(
        "--debug",
        action="store_true",
        help="Включить режим отладки",
    )

    model_manager_parser = subparsers.add_parser(
        "model_manager", help="Установка голосовых моделей в Vbach"
    )
    vbach_model_manager_parser = model_manager_parser.add_subparsers(
        title="vbach_commands", dest="vbach_command", required=True
    )

    install_local_parser = vbach_model_manager_parser.add_parser(
        "install_local", help="Установка голосовой модели по локальным файлам"
    )
    install_local_parser.add_argument(
        "--model_name", required=True, help="Имя голосовой модели"
    )
    install_local_parser.add_argument("--pth", required=True, help="Путь к *.pth файлу")
    install_local_parser.add_argument(
        "--index", required=False, help="Путь к *.index файлу"
    )

    install_url_zip_parser = vbach_model_manager_parser.add_parser(
        "install_url_zip", help="Установка голосовой модели по URL (архив с файлами)"
    )
    install_url_zip_parser.add_argument(
        "--model_name", required=True, help="Имя голосовой модели"
    )
    install_url_zip_parser.add_argument("--url", required=True, help="URL *.zip файла")

    install_url_files_parser = vbach_model_manager_parser.add_parser(
        "install_url_files", help="Установка голосовой модели по URL (отдельные файлы)"
    )
    install_url_files_parser.add_argument(
        "--model_name", required=True, help="Имя голосовой модели"
    )
    install_url_files_parser.add_argument(
        "--pth_url", required=True, help="URL *.pth файла"
    )
    install_url_files_parser.add_argument(
        "--index_url", required=False, help="URL *.index файла"
    )

    list_parser = vbach_model_manager_parser.add_parser(
        "list", help="Список установленных моделей"
    )

    remove_voice_model = vbach_model_manager_parser.add_parser("remove", help="Удаление модели")
    remove_voice_model.add_argument(
        "--model_name", required=True, help="Имя голосовой модели"
    )

    args = parser.parse_args()

    if args.mode == "cli":
        if not args.input:
            cli_parser.error("Для CLI режима требуется указать --input")
        list_valid_files = get_files_from_list(args.input)
        if list_valid_files:
            for i, vocals_file in enumerate(list_valid_files, start=1):
                print(f"Файл {i} из {len(list_valid_files)}: {vocals_file}")
                vbach_inference(
                    input_file=vocals_file,
                    model_name=args.model_name,
                    output_dir=args.output_dir,
                    output_name=args.output_name,
                    output_bitrate=args.output_bitrate,
                    output_format=args.output_format,
                    pitch=args.pitch,
                    method_pitch=args.method_pitch,
                    format_name=(
                        True if len(list_valid_files) > 1 else args.format_name
                    ),
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
                    pipeline_mode="alt" if args.alt_pipeline == True else "orig",
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
            models = model_manager.parse_voice_models()
            if models:
                print("Установленные модели:")
                for model in models:
                    print(f"  - {model}")
            else:
                print("Нет установленных моделей")

        elif args.vbach_command == "remove":
            status = model_manager.del_voice_model(args.model_name)
            print(status)
