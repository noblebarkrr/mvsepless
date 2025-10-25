import os
import shutil
from tqdm import tqdm
import urllib.request
import gdown
import json
import requests
import zipfile
import tempfile
import secrets
import string
import argparse
from typing import Dict, Any

current_dir = os.getcwd()

def generate_secure_random(length=10):
    """Генерирует криптографически безопасную случайную строку"""
    characters = string.ascii_letters + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))

class ModelManager:
    def __init__(self):
        self.rmvpe_path = os.path.join(current_dir, "vbach", "models", "predictors", "rmvpe.pt")
        self.fcpe_path = os.path.join(current_dir, "vbach", "models", "predictors", "fcpe.pt")
        self.hubert_path = os.path.join(current_dir, "vbach", "models", "embedders", "hubert_base.pt")
        self.requirements = [["https://huggingface.co/Politrees/RVC_resources/resolve/main/predictors/rmvpe.pt", self.rmvpe_path], ["https://huggingface.co/Politrees/RVC_resources/resolve/main/predictors/fcpe.pt", self.fcpe_path], ["https://huggingface.co/Politrees/RVC_resources/resolve/main/embedders/hubert_base.pt", self.hubert_path]]
        self.voicemodels_dir = os.environ.get("VBACH_MODELS_DIR", os.path.join(current_dir, "voice_models"))
        os.makedirs(self.voicemodels_dir, exist_ok=True)
        self.voicemodels_info = os.path.join(self.voicemodels_dir, "models.json")
        self.voicemodels: Dict[str, Dict[str, str]] = {}
        self.download_requirements()
        self.check_and_load()
        pass
    
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

    def del_voice_model(
        self, name
    ):
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
    
    def download_file(self, url_model, local_path):
        dir_name = os.path.dirname(local_path)
        if dir_name != "":
            os.makedirs(dir_name, exist_ok=True)
        class TqdmUpTo(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

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

    def download_requirements(self):
        for url, file in self.requirements:
            if not os.path.exists(file):
                self.download_file(url_model=url, local_path=file)

    def download_voice_model_file(self, url, zip_name):
        try:
            if "drive.google.com" in url:
                self.download_from_google_drive(url, zip_name)
            elif "pixeldrain.com" in url:
                self.download_from_pixeldrain(url, zip_name)
            elif "disk.yandex.ru" in url or "yadi.sk" in url:
                self.download_from_yandex(url, zip_name)
            else:
                self.download_file(url, zip_name)
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
        yandex_api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources/{yandex_public_key}"
        response = requests.get(yandex_api_url)
        if response.status_code == 200:
            download_link = response.json().get("href")
            urllib.request.urlretrieve(download_link, zip_name)
        else:
            print(response.status_code)

    def extract_zip(self, zip_name, model_name):
        model_dir = os.path.join(self.voicemodels_dir, f"{model_name}_{generate_secure_random(17)}")
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
                    if name.endswith(".index") and os.stat(file_path).st_size > 1024 * 100:
                        index_filepath = file_path
                    if name.endswith(".pth") and os.stat(file_path).st_size > 1024 * 1024 * 20:
                        model_filepaths.append(file_path)

            if len(model_filepaths) == 1:
                self.add_voice_model(model_name, model_filepaths[0], index_filepath)
                added_voice_models.append(model_name)
            else:
                for i, pth in enumerate(model_filepaths):
                    self.add_voice_model(f"{model_name}_{i + 1}", pth, index_filepath)
                    added_voice_models.append(f"{model_name}_{i + 1}")
            list_models_str = '\n'.join(added_voice_models)
            return f"Добавленные модели:\n{list_models_str}"
        except Exception as e:
            return f"Произошла ошибка при загрузке модели: {e}"

    def install_model_zip(self, zip, model_name, mode="url"):
        if model_name in self.parse_voice_models():
            print("Эта модель уже есть в списке установленных моделей. Она будут перезаписана")
        if mode == "url":
            with tempfile.TemporaryDirectory(prefix="vbach_temp_model", ignore_cleanup_errors=True) as tmp:
                zip_path = os.path.join(tmp, "model.zip")
                self.download_voice_model_file(zip, zip_path)
                status = self.extract_zip(zip_path, model_name)
        if mode == "local":
            status = self.extract_zip(zip, model_name)
        return status

    def install_model_files(self, index, pth, model_name, mode="url"):
        if model_name in self.parse_voice_models():
            print("Эта модель уже есть в списке установленных моделей. Она будут перезаписана")
        model_dir = os.path.join(self.voicemodels_dir, f"{model_name}_{generate_secure_random(17)}")
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
                        local_index_path = os.path.join(model_dir, os.path.basename(index))
                        shutil.copy(index, local_index_path)
                if pth:
                    if os.path.exists(pth):
                        local_pth_path = os.path.join(model_dir, os.path.basename(pth))
                        shutil.copy(pth, local_pth_path)

            self.add_voice_model(model_name, local_pth_path, local_index_path)
            return f"Модель {model_name} добавлена"
        except Exception as e:
            return f"Произошла ошибка при загрузке модели: {e}"

model_manager = ModelManager()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Model Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Local install
    local_parser = subparsers.add_parser("install_local", help="Install model from local files")
    local_parser.add_argument("--name", required=True, help="Model name")
    local_parser.add_argument("--pth", required=True, help="Path to .pth file")
    local_parser.add_argument("--index", required=False, help="Path to .index file")

    # Install from URL (zip)
    url_parser = subparsers.add_parser("install_url_zip", help="Install model from zip URL")
    url_parser.add_argument("--name", required=True, help="Model name")
    url_parser.add_argument("--url", required=True, help="URL to zip file")

    # Install from URL (files)
    url_files_parser = subparsers.add_parser("install_url_files", help="Install model from .pth and .index URLs")
    url_files_parser.add_argument("--name", required=True, help="Model name")
    url_files_parser.add_argument("--pth_url", required=True, help="URL to .pth file")
    url_files_parser.add_argument("--index_url", required=False, help="URL to .index file")

    list_parser = subparsers.add_parser("list", help="Install model from local files")

    args = parser.parse_args()

    if args.command == "install_local":
      status = model_manager.install_model_files(args.index, args.pth, args.name, mode="local")
      print(status)
    elif args.command == "install_url_zip":
      status = model_manager.install_model_zip(args.url, args.name, mode="url")
      print(status)
    elif args.command == "install_url_files":
      status = model_manager.install_model_files(args.index_url, args.pth_url, args.name, mode="url")
      print(status)
    elif args.command == "list":
      status = model_manager.parse_voice_models()
      status_str = '\n'.join(status)

      print(f"Установленные модели:\n\n{status_str}")

