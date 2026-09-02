import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*show_api.*") # Предупреждения скрыты
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*theme.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*css.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*head.*")
import gradio as gr
import sys
import json
import zipfile
from urllib.parse import urlparse
from pathlib import Path, PurePosixPath
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
from extra_utils import tz, define_audio_with_size, define_download_button_with_size, update_audio_with_size, base_c_params, easy_check_is_colab, get_gdrive_dir, one_element_list_to_value, dw_file, dw_file_legacy, dw_yt_dlp, get_disk_usage, share_gradio_tunnel
from inference import Separator, PresetExecutor, add_params, add_params_list, ensemble_types, BASE_DIR, get_stems_from_config_simple, custom_model_types, default_add_params
from vbach_lib.infer import VbachConverter, stereo_modes
from vbach_lib.f0_extractor import f0_methods, crepe_like_f0_methods, f0_extract_and_write
from vbach_lib.hubert_manager import download_hubert, huberts_fairseq, huberts_transformers
from audio import output_formats, get_audio_files_from_list, check, check_taglib_not_installed, read
from datetime import datetime
from typing import Any
from namer import Namer
from i18n import _i18n
from args_parser import parse_app_args
import tempfile
import librosa
import shutil
from tqdm import tqdm
from copy import deepcopy
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import scipy.interpolate
from scipy.ndimage import zoom
from PIL import Image
import io
import base64
import numpy as np
import asyncio
import uvicorn

def generate_add_params_component():
    add_params_components = []
    for tab, components in add_params.items():
          with gr.Tab(_i18n(tab)):
              for component_name, params in components.items():
                  component_type = params["component"]
                  if component_type == "slider":
                      add_params_components.append(gr.Slider(label=_i18n(component_name), minimum=params["minimum"], maximum=params["maximum"], step=params["step"], value=params["default"], info=_i18n(params.get("info", "")), **base_c_params["base"]))
                  elif component_type == "number":
                      add_params_components.append(gr.Number(label=_i18n(component_name), minimum=params["minimum"], maximum=params["maximum"], value=params["default"], info=_i18n(params.get("info", "")), **base_c_params["base"]))
                  elif component_type == "checkbox":
                      add_params_components.append(gr.Checkbox(label=_i18n(component_name), value=params["default"], info=_i18n(params.get("info", "")), **base_c_params["base"]))
    return add_params_components

def melspectrogram_full_reassigned(
    *,
    y=None,
    sr: float = 22050,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 256,
    fmin: float = 40.0,
    **kwargs,
) -> np.ndarray:
    fmax = sr / 2.0
    freqs, times, S_reassigned = librosa.reassigned_spectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, fill_nan=True
    )
    mels = librosa.hz_to_mel(freqs)
    min_mel = librosa.hz_to_mel(fmin)
    max_mel = librosa.hz_to_mel(fmax)
    mel_bins = np.round((mels - min_mel) / (max_mel - min_mel) * (n_mels - 1))
    time_bins = np.round(times * sr / hop_length)
    n_frames = S_reassigned.shape[1]
    S_sharp = np.zeros((n_mels, n_frames))
    valid = (
        (mel_bins >= 0) & (mel_bins < n_mels) &
        (time_bins >= 0) & (time_bins < n_frames) &
        ~np.isnan(mel_bins) & ~np.isnan(time_bins)
    )
    flat_mel_indices = mel_bins[valid].astype(int)
    flat_time_indices = time_bins[valid].astype(int)
    flat_magnitudes = np.abs(S_reassigned[valid])
    target_indices = np.ravel_multi_index((flat_mel_indices, flat_time_indices), S_sharp.shape)
    np.add.at(S_sharp.ravel(), target_indices, flat_magnitudes)
    return librosa.amplitude_to_db(S_sharp, ref=np.max)


def apply_colormap_audacity(gray):
    """Чёрный/синий → красный → оранжевый → жёлтый → белый (палитра Audacity)"""
    audacity_colors = np.array([
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.8, 0.0, 0.0],
        [1.0, 0.6, 0.0], [1.0, 1.0, 0.6], [1.0, 1.0, 1.0]
    ])
    x = np.array([0, 0.15, 0.4, 0.75, 0.9, 1.0])
    r_interp = scipy.interpolate.interp1d(x, audacity_colors[:, 0])
    g_interp = scipy.interpolate.interp1d(x, audacity_colors[:, 1])
    b_interp = scipy.interpolate.interp1d(x, audacity_colors[:, 2])
    lut_x = np.linspace(0, 1, 256)
    lut = np.stack([r_interp(lut_x), g_interp(lut_x), b_interp(lut_x)], axis=1)
    return (lut[gray] * 255).astype(np.uint8)

def f0_corrector_analyze_worker(audio_path: str, f0_path: str) -> dict:
    """Тяжёлая часть анализа — выполняется в рабочем потоке, не блокируя event loop."""
    with open(f0_path, "r", encoding="utf-8") as f:
        f0_data = json.load(f)
    freqs = np.array(f0_data.get("freqs", []))
    sample_rate = f0_data.get("sample_rate", 16000)
    window = f0_data.get("window", 160)
    method = f0_data.get("method", "rmvpe+")
    n_mels = 128
    n_fft = 512
    hop_length = window // 4
    internal_sample_rate = sample_rate // 4
    zoom_factor_height = 2
    y, sr = read(audio_path, sr=internal_sample_rate, mono=True, flatten=True)
    S = melspectrogram_full_reassigned(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=0
    )
    spec_height = n_mels * zoom_factor_height
    src_rows = (1.0 - np.arange(spec_height) / (spec_height - 1)) * (n_mels - 1)
    low = np.clip(np.floor(src_rows).astype(int), 0, n_mels - 1)
    high = np.clip(low + 1, 0, n_mels - 1)
    frac = (src_rows - np.floor(src_rows))[:, None]
    S_interp = S[low] * (1.0 - frac) + S[high] * frac
    db_min, db_max = -80.0, 0.0
    S_norm = np.clip((S_interp - db_min) / (db_max - db_min), 0, 1) * 255
    colored = apply_colormap_audacity(S_norm.astype(np.uint8))
    img = Image.fromarray(colored, mode='RGB')
    n_frames = S.shape[1]
    if len(freqs) < n_frames:
        freqs = np.pad(freqs, (0, n_frames - freqs.size), mode="constant")
    elif len(freqs) > n_frames:
        freqs = freqs[:n_frames]
    duration = len(y) / sr
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return {
        "success": True,
        "spectrogram": base64.b64encode(buffer.getvalue()).decode(),
        "spec_width": n_frames,
        "spec_height": n_mels * zoom_factor_height,
        "times": np.linspace(0, duration, n_frames).tolist(),
        "freqs": freqs.tolist(),
        "sample_rate": internal_sample_rate,
        "original_sample_rate": sample_rate,
        "window": window,
        "method": method,
        "duration": float(duration),
        "n_mels": n_mels
    }

USER_DIR = ""
GDRIVE_DIR = get_gdrive_dir()
def generate_user_dir_from_gdrive():
    global GDRIVE_DIR
    if GDRIVE_DIR:
        user_dir = Path(GDRIVE_DIR, "MyDrive", "mvsepless-data")
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir.as_posix()
    else:
        return None
GDRIVE_USER_DIR = generate_user_dir_from_gdrive()

def get_default_user_dir():
    if easy_check_is_colab():
        if GDRIVE_DIR:
            print(_i18n("gdrive_mount_found"))
            return GDRIVE_USER_DIR
        else:
            return USER_DIR
    else:
        return USER_DIR

DEFAULT_USER_DIR = get_default_user_dir()

def rename_user_dir_path(path: str, mode=0):
    global GDRIVE_USER_DIR, USER_DIR
    if path:
        if mode == 0:
            return (PurePosixPath(GDRIVE_USER_DIR) / PurePosixPath(path).relative_to(USER_DIR)).as_posix()
        elif mode == 1:
            return (PurePosixPath(USER_DIR) / PurePosixPath(path).relative_to(GDRIVE_USER_DIR)).as_posix()
    else:
        return None

base_names_app_dirs = (
    "input",
    "output_mvsepless",
    "history",
    "ensemble_flows",
    "vbach_models",
    "f0_curves",
    "custom_separation_models",
    "vbach_output",
    "iterative_ensemble_flows",
    "presets"
)

def copy_to_gdrive():
    global GDRIVE_DIR, GDRIVE_USER_DIR, USER_DIR
    if GDRIVE_DIR:
        copied_dirs = []
        dirs = [[dir, Path(USER_DIR, dir)] for dir in base_names_app_dirs]
        for (dir_name, dir_path) in tqdm(dirs, desc=_i18n("copy_to_gdrive"), unit=_i18n("dirs")):
            if dir_path.exists():
                shutil.copytree(dir_path, Path(GDRIVE_USER_DIR, dir_name), dirs_exist_ok=True)
                copied_dirs.append("")
        print(_i18n("copied_dirs")+": "+str(len(copied_dirs)))
        print(_i18n("copy_to_gdrive_done"))
        gr.Info(title=_i18n("copy_to_gdrive_done"), message="")

def copy_to_runtime():
    global GDRIVE_DIR, GDRIVE_USER_DIR, USER_DIR
    if GDRIVE_DIR:
        copied_dirs = []
        dirs = [[dir, Path(GDRIVE_USER_DIR, dir)] for dir in base_names_app_dirs]
        for (dir_name, dir_path) in tqdm(dirs, desc=_i18n("copy_to_current_user_dir"), unit=_i18n("dirs")):
            if dir_path.exists():
                shutil.copytree(dir_path, Path(USER_DIR, dir_name), dirs_exist_ok=True)
                copied_dirs.append("")
        print(_i18n("copied_dirs")+": "+str(len(copied_dirs)))
        print(_i18n("copy_to_gdrive_done"))
        gr.Info(title=_i18n("copy_to_gdrive_done"), message="")

def generate_zip_archive(files: str | Path | list[str | Path] | tuple[str | Path, ...], output_path: str | Path):

    if isinstance(files, (str, Path)):
        input_files = [files]
    else:
        input_files = files

    added_files = []

    output_path_ = Path(output_path)
    output_path_.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path_, mode="w") as zip_file:
        for path in input_files:
            p = Path(path)
            if p.exists():
                name = p.name
                name_list = zip_file.namelist()
                if name not in name_list:
                    zip_file.write(p, name)
                else:
                    zip_file.write(p, Namer.iter_in_list(Namer.short(name), name_list))
                added_files.append(name)

    print(_i18n("added_files")+": "+str(len(added_files)))
    return output_path_.as_posix()

def get_zip_output_path(name):
    temp_dir = Path(tempfile.gettempdir())
    timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
    return Namer.iter(temp_dir / f"{name}_{timestamp}.zip")


class UserDirectory:
    def __init__(self, custom_dir=USER_DIR):
        self.user_directory = Path(custom_dir if custom_dir else DEFAULT_USER_DIR)

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



class PresetLessApp(UserDirectory):
    def __init__(self):
        super().__init__()
        self.base_dir = self.user_directory / base_names_app_dirs[9]
        self.base_dir.mkdir(exist_ok=True)

    def get_list(self):
        return [p.stem for p in self.base_dir.glob("*.json")]
    
    def load_preset(self, name: str):
        path = self.base_dir / (name + ".json")
        return json.loads(path.read_text(encoding="utf-8"))

    def save_preset(self, name: str, data: dict):
        path = self.base_dir / (Namer.sanitize(name) + ".json")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")

    def delete_preset(self, name: str):
        path = self.base_dir / (name + ".json")
        path.unlink(missing_ok=True)

class AutoEnsembleApp(UserDirectory):
    def __init__(self, separator: Separator = "Separator"):
        UserDirectory.__init__(self)
        self.separator = separator
        self.base_dir = self.user_directory / base_names_app_dirs[3]
        self.base_dir.mkdir(exist_ok=True)

    def get_list(self):
        return [p.stem for p in self.base_dir.glob("*.json")]
    
    def load_preset(self, name: str):
        path = self.base_dir / (name + ".json")
        state = json.loads(path.read_text(encoding="utf-8"))
        state, warns_str = self.separator.validate_flow(state, non_exists_warn=True)
        return state, warns_str

    def save_preset(self, name: str, data: dict):
        path = self.base_dir / (Namer.sanitize(name) + ".json")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")

    def delete_preset(self, name: str):
        path = self.base_dir / (name + ".json")
        path.unlink(missing_ok=True)

class InputFilesDatabase(UserDirectory):
    def __init__(self):
        super().__init__()
        self.input_dir_base = self.user_directory / base_names_app_dirs[0]
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

    @_write_decorator
    def update_data(self, mode: int):
        current_data = deepcopy(self.input_base)
        new_data = []
        if self.input_base_json.exists():
            new_data: list = json.loads(self.input_base_json.read_text("utf-8"))

        new_data2 = []
        new_data_to_merge = []

        for file_path in new_data:
            new_data2.append(rename_user_dir_path(file_path, mode=mode))

        for path2 in new_data2:
            if path2 not in current_data:
                new_data_to_merge.append(path2)

        self.input_base = list(dict.fromkeys([*current_data, *new_data_to_merge]))

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
    def __init__(self, dir: str = base_names_app_dirs[1]):
        super().__init__()
        self.output_dir_name = dir

    def gen_output_dir(self):
        return self.generate(self.output_dir_name)

class History(UserDirectory):
    def __init__(self, name: str = "mvsepless"):
        super().__init__()
        self.history_dir_base = self.user_directory / base_names_app_dirs[2]
        self.history_dir_base.mkdir(parents=True, exist_ok=True)
        self.history_dict_json = self.history_dir_base / f"{name}.json"
        self.history_dict = {}
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
        self.history_dict_json.write_text(json.dumps(self.history_dict, ensure_ascii=False, indent=4), encoding="utf-8")

    def load(self):
        if self.history_dict_json.exists():
            self.history_dict = json.loads(self.history_dict_json.read_text("utf-8"))
            print(_i18n("history_loaded"))

    @_write_decorator
    def update_data(self, mode: int):
        current_data = deepcopy(self.history_dict)
        new_data = {}
        if self.history_dict_json.exists():
            new_data: dict = json.loads(self.history_dict_json.read_text("utf-8"))
            
        new_data_to_merge = {}

        for key, state in new_data.items():
            new_state = []
            for basename, stems_list in state:
                new_stems_list = [basename]
                new_stems_list.append([[stem_name, rename_user_dir_path(stem_path, mode=mode)] for stem_name, stem_path in stems_list])
                new_state.append(deepcopy(new_stems_list))
            new_data[key] = deepcopy(new_state)

        for key2, state2 in new_data.items():
            if key2 not in list(current_data.keys()) and state2 != current_data.get(key2):
                new_data_to_merge[key2] = state2

        self.history_dict: dict = {
            **current_data,
            **new_data_to_merge
        }

    def get_list(self, update_from_file=False):
        if update_from_file:
            self.load()
        return deepcopy(list(reversed([key_h for key_h in self.history_dict])))

    @_write_decorator
    def add_to_history(self, model_name: str, state: list):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        self.history_dict.update([(f"{timestamp} | {model_name}", deepcopy(state))])

    def get_from_history(self, key: str):
        return deepcopy(self.history_dict.get(key, None))

class HistoryPresetless(History):
    def __init__(self):
        super().__init__("presetless")

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

    @_write_decorator
    def update_data(self, mode: int):
        current_data = deepcopy(self.history_dict)
        new_data = {}
        if self.history_dict_json.exists():
            new_data: dict = json.loads(self.history_dict_json.read_text("utf-8"))
            
        new_data_to_merge = {}

        for key, state in new_data.items():
            new_state = []
            if state:
                new_state = [[stem_name, rename_user_dir_path(stem_path, mode=mode)] for stem_name, stem_path in state]
            new_data[key] = deepcopy(new_state)

        for key2, state2 in new_data.items():
            if key2 not in list(current_data.keys()) and state2 != current_data.get(key2):
                new_data_to_merge[key2] = state2

        self.history_dict: dict = {
            **current_data,
            **new_data_to_merge
        }

    @_write_decorator
    def add_to_history(self, name_preset: str, state: list):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        self.history_dict.update([(f"{timestamp} | {name_preset}", deepcopy(state))])
    
class HistoryAutoEnsemble(History):
    def __init__(self):
        super().__init__("ensembless")

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
    
    @_write_decorator
    def update_data(self, mode: int):
        current_data = deepcopy(self.history_dict)
        new_data = {}
        if self.history_dict_json.exists():
            new_data: dict = json.loads(self.history_dict_json.read_text("utf-8"))
        new_data_to_merge = {}

        for key, state in new_data.items():
            new_state = [
                rename_user_dir_path(state[0], mode=mode),  # result
                rename_user_dir_path(state[1], mode=mode),  # invert
                [rename_user_dir_path(stem_path, mode=mode) for stem_path in state[2]]  # primary_stems_list
            ]
            new_data[key] = deepcopy(new_state)
        for key2, state2 in new_data.items():
            if key2 not in list(current_data.keys()) and state2 != current_data.get(key2):
                new_data_to_merge[key2] = state2

        self.history_dict: dict = {
            **current_data,
            **new_data_to_merge
        }

    @_write_decorator
    def add_to_history(self, etype: str, output: str, inverted_output: str, primary_stems_list: list = []):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        self.history_dict.update([(f"{timestamp} | {etype}", (output, inverted_output, primary_stems_list))])

    def get_from_history(self, key: str):
        return deepcopy(self.history_dict.get(key, (None, None, [])))

class HistoryManualEnsemble(History):
    def __init__(self):
        super().__init__("manual_ensembless")

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

    @_write_decorator
    def update_data(self, mode: int):
        current_data = deepcopy(self.history_dict)
        new_data = {}
        if self.history_dict_json.exists():
            new_data: dict = json.loads(self.history_dict_json.read_text("utf-8"))
        new_data_to_merge = {}

        for key, state in new_data.items():
            new_state = None
            if state:
                new_state = rename_user_dir_path(state, mode=mode)
            new_data[key] = deepcopy(new_state)

        for key2, state2 in new_data.items():
            if key2 not in list(current_data.keys()) and state2 != current_data.get(key2):
                new_data_to_merge[key2] = state2

        self.history_dict: dict = {
            **current_data,
            **new_data_to_merge
        }

    @_write_decorator
    def add_to_history(self, etype: str, state: str):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        self.history_dict.update([(f"{timestamp} | {etype}", deepcopy(state))])

    def get_from_history(self, key: str):
        return deepcopy(self.history_dict.get(key, None))

class HistorySubtractor(History):
    def __init__(self):
        super().__init__("subtract")

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

    @_write_decorator
    def update_data(self, mode: int):
        current_data = deepcopy(self.history_dict)
        new_data = {}
        if self.history_dict_json.exists():
            new_data: dict = json.loads(self.history_dict_json.read_text("utf-8"))
        new_data_to_merge = {}

        for key, state in new_data.items():
            new_state = None
            if state:
                new_state = rename_user_dir_path(state, mode=mode)
            new_data[key] = deepcopy(new_state)

        for key2, state2 in new_data.items():
            if key2 not in list(current_data.keys()) and state2 != current_data.get(key2):
                new_data_to_merge[key2] = state2

        self.history_dict: dict = {
            **current_data,
            **new_data_to_merge
        }

    @_write_decorator
    def add_to_history(self, itype: str, state: str):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        self.history_dict.update([(f"{timestamp} | {itype}", deepcopy(state))])

class HistoryVbach(History):
    def __init__(self):
        super().__init__("vbach")

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

    @_write_decorator
    def update_data(self, mode: int):
        current_data = deepcopy(self.history_dict)
        new_data = {}
        if self.history_dict_json.exists():
            new_data: dict = json.loads(self.history_dict_json.read_text("utf-8"))
        new_data_to_merge = {}

        for key, state in new_data.items():
            new_state = []
            if state:
                new_state = [rename_user_dir_path(file_path, mode=mode) for file_path in state]
            new_data[key] = deepcopy(new_state)

        for key2, state2 in new_data.items():
            if key2 not in list(current_data.keys()) and state2 != current_data.get(key2):
                new_data_to_merge[key2] = state2

        self.history_dict: dict = {
            **current_data,
            **new_data_to_merge
        }

    @_write_decorator
    def add_to_history(self, model_name: str, f0_method: str, pitch: int, output_files: list):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        self.history_dict.update([(f"{timestamp} | {model_name} | {f0_method} | {pitch}", deepcopy(output_files))])

    def get_from_history(self, key: str):
        return deepcopy(self.history_dict.get(key, []))

class HistoryIterativeEnsemble(History):
    def __init__(self):
        super().__init__("iterative_ensembless")
    
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
    
    @_write_decorator
    def update_data(self, mode: int):
        current_data = deepcopy(self.history_dict)
        new_data = {}
        if self.history_dict_json.exists():
            new_data: dict = json.loads(self.history_dict_json.read_text("utf-8"))
        new_data_to_merge = {}

        for key, state in new_data.items():
            new_state = None
            if state:
                new_state = [
                    rename_user_dir_path(state[0], mode=mode),  # result
                    [rename_user_dir_path(path, mode=mode) for path in state[1]]  # intermediate files
                ]
            new_data[key] = deepcopy(new_state)

        for key2, state2 in new_data.items():
            if key2 not in list(current_data.keys()) and state2 != current_data.get(key2):
                new_data_to_merge[key2] = state2

        self.history_dict: dict = {
            **current_data,
            **new_data_to_merge
        }

    @_write_decorator
    def add_to_history(self, result_path: str, intermediate_files: list, count_iters: int):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        self.history_dict.update([(f"{timestamp} | {count_iters} iterations", (result_path, intermediate_files))])

    def get_from_history(self, key: str):
        state = self.history_dict.get(key, (None, []))
        if state:
            return state[0], state[1]
        return None, []

class HistoryPhaseFixer(History):
    def __init__(self):
        super().__init__("phase_fixer")

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

    @_write_decorator
    def update_data(self, mode: int):
        current_data = deepcopy(self.history_dict)
        new_data = {}
        if self.history_dict_json.exists():
            new_data: dict = json.loads(self.history_dict_json.read_text("utf-8"))
        new_data_to_merge = {}
        for key, state in new_data.items():
            new_state = None
            if state:
                new_state = rename_user_dir_path(state, mode=mode)
            new_data[key] = deepcopy(new_state)
        for key2, state2 in new_data.items():
            if key2 not in list(current_data.keys()) and state2 != current_data.get(key2):
                new_data_to_merge[key2] = state2
        self.history_dict: dict = {
            **current_data,
            **new_data_to_merge
        }

    @_write_decorator
    def add_to_history(self, settings_str: str, state: str):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        self.history_dict.update([(f"{timestamp} | {settings_str}", deepcopy(state))])

class IterativeEnsembleApp(UserDirectory, Separator):
    def __init__(self, separator: Separator = "Separator"):
        UserDirectory.__init__(self)
        self.separator = separator
        self.base_dir = self.user_directory / base_names_app_dirs[8]
        self.base_dir.mkdir(exist_ok=True)
        
    def get_list(self):
        return [p.stem for p in self.base_dir.glob("*.json")]
    
    def load_preset(self, name: str):
        path = self.base_dir / (name + ".json")
        state = json.loads(path.read_text(encoding="utf-8"))
        state, warns_str = self.separator.validate_flow(state, non_exists_warn=True, iterative=True)
        return state, warns_str

    def save_preset(self, name: str, data: dict):
        path = self.base_dir / (name + ".json")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")

    def delete_preset(self, name: str):
        path = self.base_dir / (name + ".json")
        path.unlink(missing_ok=True)

class VbachModelsDir(UserDirectory):
    """Manage Vbach models directory and model discovery"""
    
    def __init__(self):
        super().__init__()
        self.vbach_models_base = self.user_directory / base_names_app_dirs[4]
        self.pth_models_dir = self.vbach_models_base / "pth"
        self.index_models_dir = self.vbach_models_base / "index"
        self.pth_models_dir.mkdir(parents=True, exist_ok=True)
        self.index_models_dir.mkdir(parents=True, exist_ok=True)
        
        self.supported_extensions = (".pth", ".ckpt", ".pt", ".th", ".chpt")
        
    def get_pth_models(self):
        """Get list of available checkpoint models"""
        models = []
        for ext in self.supported_extensions:
            models.extend([f.as_posix() for f in self.pth_models_dir.glob(f"*{ext}")])
        return models
    
    def get_index_files(self):
        """Get list of available index files"""
        return [f.as_posix() for f in self.index_models_dir.glob("*.index")]

    def extract_zip(self, zip_path: str | Path):
        status = ""
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_extracted_path = Path(tmpdirname)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_extracted_path)
                print(_i18n("vbach_model_zip_unpacked"))
                gr.Info(title=_i18n("vbach_model_zip_unpacked"), message="")
            indexes = []
            pths = []
            for ext in self.supported_extensions:
                pths.extend([f.as_posix() for f in tmp_extracted_path.rglob(f"*{ext}")])
            indexes.extend([f.as_posix() for f in tmp_extracted_path.rglob("*.index")])
            if not pths and not indexes:
                print(_i18n("vbach_model_zip_not_model_files"))
                gr.Info(title=_i18n("vbach_model_zip_not_model_files"), message="")
            status += self.upload_index_model(indexes)+"\n"
            status += self.upload_pth_model(pths)+"\n"
        return status

    def get_pth_name_from_link(self, url):
        clean_url = urlparse(url)._replace(query="", fragment="").geturl()
        file_name = PurePosixPath(PurePosixPath(clean_url).name)
        if file_name.suffix not in self.supported_extensions:
            file_name = file_name.with_suffix(".pth")
        return str(file_name)
    
    def get_index_name_from_link(self, url):
        clean_url = urlparse(url)._replace(query="", fragment="").geturl()
        file_name = PurePosixPath(PurePosixPath(clean_url).name)
        if file_name.suffix != ".index":
            file_name = file_name.with_suffix(".index")
        return str(file_name)

    def download_model(self, zip_url=None, pth_url=None, index_url=None):
        status = ""
        if zip_url:
            temp_zip = Path(tempfile.mkstemp(suffix=".zip")[1])
            dw_file(zip_url, temp_zip)
            status = self.extract_zip(temp_zip)
        else:
            index_status = None
            pth_status = None
            if index_url:
                dw_file(index_url, Namer.iter(self.index_models_dir / self.get_index_name_from_link(index_url)))
                index_status = _i18n("vbach_model_index_downloaded")+"\n"
            if pth_url:
                dw_file(pth_url, Namer.iter(self.pth_models_dir / self.get_pth_name_from_link(pth_url)))
                pth_status = _i18n("vbach_model_pth_downloaded")+"\n"
            status_list = [status_unit for status_unit in [index_status, pth_status] if status_unit]
            status = "".join(status_list).rsplit("\n", maxsplit=1)[0]
            print(status)
            gr.Info(message="<b>"+status.replace("\n", "<br>")+"</b>", title="")
        return status

    def upload_pth_model(self, pth_paths: str | Path | list[str | Path]):
        added_files = []
        if isinstance(pth_paths, (str, Path)):
            pth_paths = [pth_paths]

        for p_str in pth_paths:
            p = Path(p_str)
            dst = Namer.iter(self.pth_models_dir / p.name)
            if p.suffix in self.supported_extensions:
                shutil.move(p, dst)
                added_files.append(dst)
        status = _i18n("vbach_added_pths")+": "+str(len(added_files))
        print(status)
        gr.Info(message="<b>"+status.replace("\n", "<br>")+"</b>", title="")
        return status

    def upload_index_model(self, index_paths: str | Path):
        added_files = []
        if isinstance(index_paths, (str, Path)):
            index_paths = [index_paths]

        for p_str in index_paths:
            p = Path(p_str)
            if p.exists():
                dst = Namer.iter(self.index_models_dir / p.name)
                if p.suffix in [".index"]:
                    shutil.move(p, dst)
                    added_files.append(dst)
        status = _i18n("vbach_added_indexes")+": "+str(len(added_files))
        print(status)
        gr.Info(message="<b>"+status.replace("\n", "<br>")+"</b>", title="")
        return status

    def get_model_name(self, model_path: str | Path) -> str:
        """Extract model name from path"""
        return Path(model_path).stem

class F0GenerateOutPath(UserDirectory):
    def __init__(self):
        super().__init__()
        self.f0_curves_dir = self.user_directory / base_names_app_dirs[5]
        self.f0_curves_dir.mkdir(parents=True, exist_ok=True)

    def generate_output_path(self, name: str, f0_method: str):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        generated_path = self.f0_curves_dir / f"{timestamp}_{f0_method}_{Namer.short(name, length=90)}.json"
        return generated_path.as_posix()

class CustomSeparationModelsDir(UserDirectory):
    """Manage custom separation models directory and model discovery"""
    
    def __init__(self):
        super().__init__()
        self.custom_models_base = self.user_directory / base_names_app_dirs[6]
        self.checkpoints_dir = self.custom_models_base / "checkpoints"
        self.configs_dir = self.custom_models_base / "configs"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        
        self.supported_extensions = (".pth", ".ckpt", ".pt", ".th", ".chpt")
        self.config_extensions = (".yaml", ".yml")
        
    def get_checkpoints(self):
        """Get list of available checkpoint files"""
        checkpoints = []
        for ext in self.supported_extensions:
            checkpoints.extend([f.as_posix() for f in self.checkpoints_dir.glob(f"*{ext}")])
        return checkpoints
    
    def get_configs(self):
        """Get list of available config files"""
        configs = []
        for ext in self.config_extensions:
            configs.extend([f.as_posix() for f in self.configs_dir.glob(f"*{ext}")])
        return configs

    def get_checkpoint_name_from_link(self, url):
        clean_url = urlparse(url)._replace(query="", fragment="").geturl()
        file_name = PurePosixPath(PurePosixPath(clean_url).name)
        if file_name.suffix not in self.supported_extensions:
            file_name = file_name.with_suffix(".pth")
        return str(file_name)
    
    def get_config_name_from_link(self, url):
        clean_url = urlparse(url)._replace(query="", fragment="").geturl()
        file_name = PurePosixPath(PurePosixPath(clean_url).name)
        if file_name.suffix not in self.config_extensions:
            file_name = file_name.with_suffix(".yaml")
        return str(file_name)

    def download_model(self, zip_url=None, checkpoint_url=None, config_url=None):
        """Download model from URL"""
        status = ""
        config_status = None
        checkpoint_status = None
        if config_url:
            dw_file_legacy(config_url, Namer.iter(self.configs_dir / self.get_config_name_from_link(config_url)))
            config_status = _i18n("custom_model_config_downloaded")+"\n"
        if checkpoint_url:
            dw_file(checkpoint_url, Namer.iter(self.checkpoints_dir / self.get_checkpoint_name_from_link(checkpoint_url)))
            checkpoint_status = _i18n("custom_model_checkpoint_downloaded")+"\n"
        status_list = [status_unit for status_unit in [config_status, checkpoint_status] if status_unit]
        status = "".join(status_list).rsplit("\n", maxsplit=1)[0]
        print(status)
        gr.Info(message="<b>"+status.replace("\n", "<br>")+"</b>", title="")
        return status

    def upload_checkpoint(self, checkpoint_paths: str | Path | list[str | Path]):
        """Upload checkpoint files"""
        added_files = []
        if isinstance(checkpoint_paths, (str, Path)):
            checkpoint_paths = [checkpoint_paths]

        for p_str in checkpoint_paths:
            p = Path(p_str)
            dst = Namer.iter(self.checkpoints_dir / p.name)
            if p.suffix in self.supported_extensions:
                shutil.move(p, dst)
                added_files.append(dst)
        status = _i18n("custom_added_checkpoints")+": "+str(len(added_files))
        print(status)
        gr.Info(message="<b>"+status.replace("\n", "<br>")+"</b>", title="")
        return status

    def upload_config(self, config_paths: str | Path | list[str | Path]):
        """Upload config files"""
        added_files = []
        if isinstance(config_paths, (str, Path)):
            config_paths = [config_paths]

        for p_str in config_paths:
            p = Path(p_str)
            if p.exists():
                dst = Namer.iter(self.configs_dir / p.name)
                if p.suffix in self.config_extensions:
                    shutil.move(p, dst)
                    added_files.append(dst)
        status = _i18n("custom_added_configs")+": "+str(len(added_files))
        print(status)
        gr.Info(message="<b>"+status.replace("\n", "<br>")+"</b>", title="")
        return status

    def get_model_pair(self, checkpoint_path: str, config_path: str) -> tuple[str, str]:
        """Get checkpoint and config paths"""
        return checkpoint_path, config_path

class App(Separator):
    def __init__(self, source: str = "hface",
                 custom_model_info_path: str | Path | None = None,
                 custom_models_dir: str | Path | None = None):
        self.separator = Separator(
            source=source,
            custom_model_info_path=custom_model_info_path,
            custom_models_dir=custom_models_dir
        )
        self.input_files = InputFilesDatabase()
        self.output_dir = OutputDir()
        self.history = History()
        self.vbach_converter = VbachConverter()
        self.vbach_model_manager = VbachModelsDir()
        self.auto_ensemble_app = AutoEnsembleApp(
            self.separator
        )
        self.auto_ensemble_history_app = HistoryAutoEnsemble()
        self.manual_ensemble_history_app = HistoryManualEnsemble()
        self.subtract_history_app = HistorySubtractor()
        self.vbach_history_app = HistoryVbach()
        self.f0_gen_output_path = F0GenerateOutPath()
        self.custom_sep_model_manager = CustomSeparationModelsDir()
        self.iterative_ensemble_app = IterativeEnsembleApp(
            self.separator
        )
        self.iterative_ensemble_history_app = HistoryIterativeEnsemble()
        self.preset_history = HistoryPresetless()
        self.preset_manager = PresetLessApp()
        self.phase_fixer_history_app = HistoryPhaseFixer()
        self.add_params_dict = {}
        # Мультипользовательский inbox: ключ — session_hash Gradio,
        # чтобы пользователи не получали чужие загруженные файлы
        self.f0_corrector_inbox = {}
        # Ограничение параллельных тяжёлых анализов (иначе один пользователь
        # кладёт event loop всем остальным)
        self._f0_analyze_semaphore = asyncio.Semaphore(2)

    def update_model_name(self, model_name):
        stems = self.separator.get_stems(model_name)
        return gr.update(value=False, visible=len(stems) > 2), gr.update(value=[], choices=stems)

    def update_model_name_ensemble(self, model_name):
        stems = self.separator.get_stems(model_name)
        if stems:
            first_value = stems[0]
        else:
            first_value = None
        return gr.update(value=first_value, choices=stems)

    def update_add_params(self, *add_params_values):
        return dict(zip(add_params_list, add_params_values))

    def get_actual_history_list(self, value, state):
        current_history = self.history.get_list()
        if current_history == state:
            return gr.skip()
        return gr.update(choices=current_history, value=value), current_history

    def get_actual_preset_history_list(self, value, state):
        current_history = self.preset_history.get_list()
        if current_history == state:
            return gr.skip()
        return gr.update(choices=current_history, value=value), current_history

    def get_actual_auto_ensemble_history_list(self, value, state):
        current_history = self.auto_ensemble_history_app.get_list()
        if current_history == state:
            return gr.skip()
        return gr.update(choices=current_history, value=value), current_history
    
    def get_actual_manual_ensemble_history_list(self, value, state):
        current_history = self.manual_ensemble_history_app.get_list()
        if current_history == state:
            return gr.skip()
        return gr.update(choices=current_history, value=value), current_history

    def get_actual_auto_ensemble_flows_list(self, value, state):
        current_flows = self.auto_ensemble_app.get_flows()
        if current_flows == state:
            return gr.skip()
        return gr.update(choices=current_flows, value=value), current_flows

    def get_actual_subtract_history_list(self, value, state):
        current_history = self.subtract_history_app.get_list()
        if current_history == state:
            return gr.skip()
        return gr.update(choices=current_history, value=value), current_history

    def get_actual_input_list(self, value, state):
        current_files = self.input_files.get_input_list()
        if current_files == state:
            return gr.skip()
        return gr.update(choices=current_files, value=value), current_files

    def get_actual_vbach_history_list(self, value, state):
        """Get updated history list"""
        current_history = self.vbach_history_app.get_list()
        if current_history == state:
            return gr.skip()
        return gr.update(choices=current_history, value=value), current_history
    
    def get_actual_vbach_models_list(self, value, state):
        """Get updated models list"""
        current_models = self.vbach_model_manager.get_pth_models()
        if current_models == state:
            return gr.skip()
        return gr.update(choices=current_models, value=value), current_models
    
    def get_actual_vbach_index_list(self, value, state):
        """Get updated index files list"""
        current_indexes = self.vbach_model_manager.get_index_files()
        if current_indexes == state:
            return gr.skip()
        return gr.update(choices=current_indexes, value=value), current_indexes

    def get_actual_custom_sep_checkpoints_list(self, value, state):
        """Get updated checkpoints list"""
        current_checkpoints = self.custom_sep_model_manager.get_checkpoints()
        if current_checkpoints == state:
            return gr.skip()
        return gr.update(choices=current_checkpoints, value=value), current_checkpoints

    def get_actual_custom_sep_configs_list(self, value, state):
        """Get updated configs list"""
        current_configs = self.custom_sep_model_manager.get_configs()
        if current_configs == state:
            return gr.skip()
        return gr.update(choices=current_configs, value=value), current_configs

    def get_actual_iterative_ensemble_history_list(self, value, state):
        current_history = self.iterative_ensemble_history_app.get_list()
        if current_history == state:
            return gr.skip()
        return gr.update(choices=current_history, value=value), current_history

    def get_actual_iterative_ensemble_flows_list(self, value, state):
        current_flows = self.iterative_ensemble_app.get_flows()
        if current_flows == state:
            return gr.skip()
        return gr.update(choices=current_flows, value=value), current_flows

    def get_actual_phase_fixer_history_list(self, value, state):
        current_history = self.phase_fixer_history_app.get_list()
        if current_history == state:
            return gr.skip()
        return gr.update(choices=current_history, value=value), current_history

    def f0_path_allowed(self, path: str) -> bool:
        """Мультипользовательская защита: разрешаем только файлы из директорий приложения и tempdir."""
        try:
            p = Path(path).resolve()
        except Exception:
            return False
        if not p.is_file():
            return False
        roots = [Path(self.input_files.user_directory).resolve(), Path(tempfile.gettempdir()).resolve()]
        if GDRIVE_USER_DIR:
            roots.append(Path(GDRIVE_USER_DIR).resolve())
        return any(p.is_relative_to(r) for r in roots)

    @staticmethod
    def _purge_old_f0_tempfiles(session_dir: Path, max_age_hours: int = 24):
        cutoff = datetime.now(tz).timestamp() - max_age_hours * 3600
        for f in session_dir.glob("*.json"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass

    def UI(self, theme=None, hf_space_mode=False) -> gr.Blocks:
        global GDRIVE_DIR, IS_CUSTOM_DIR
        all_models = self.separator.get_all_models()
        default_model = all_models[0]
        stems_default = self.separator.get_stems(default_model)
        ext_inst_visible_default = len(stems_default) > 2

        app = FastAPI()

        F0C_TRIGGER_JS = (
            "() => { try {"
            "  const send = () => { document.querySelectorAll('iframe').forEach(function(f){"
            "    try { f.contentWindow.postMessage({ type: 'f0_corrector_inbox_check' }, '*'); } catch(e){}"
            "  }); };"
            "  send(); setTimeout(send, 400); setTimeout(send, 900);"   # burst: ловим любой тайминг загрузки/занятости
            "} catch(e){} }"
        )

        # --- API Эндпоинты для пресетов ---
        @app.get("/presetless")
        def get_presets():
            return {"presets": self.preset_manager.get_list()}

        @app.get("/presetless/{name}")
        def get_preset(name: str):
            try:
                return self.preset_manager.load_preset(name)
            except Exception as e:
                return {"error": str(e)}

        @app.post("/presetless/{name}")
        async def save_preset(name: str, request: Request):
            try:
                data = await request.json()
                self.preset_manager.save_preset(name, data)
                return {"status": "ok"}
            except Exception as e:
                return {"error": str(e)}
            
        @app.delete("/presetless/{name}")
        def delete_preset(name: str):
            try:
                self.preset_manager.delete_preset(name)
                return {"status": "ok"}
            except Exception as e:
                return {"error": str(e)}

        @app.get("/auto_ensemble_preset")
        def get_auto_ensemble_presets():
            return {"presets": self.auto_ensemble_app.get_list()}

        @app.get("/auto_ensemble_preset/{name}")
        def get_auto_ensemble_preset(name: str):
            try:
                state, warns_str = self.auto_ensemble_app.load_preset(name)
                if warns_str:
                    return {"state": state, "warning": warns_str + "\n" + _i18n("ensemble_run_error_with_incorrect_flow")}
                return {"state": state}
            except Exception as e:
                return {"error": str(e)}

        @app.post("/auto_ensemble_preset/{name}")
        async def save_auto_ensemble_preset(name: str, request: Request):
            try:
                data = await request.json()
                self.auto_ensemble_app.save_preset(name, data)
                return {"status": "ok"}
            except Exception as e:
                return {"error": str(e)}
            
        @app.delete("/auto_ensemble_preset/{name}")
        def delete_auto_ensemble_preset(name: str):
            try:
                self.auto_ensemble_app.delete_preset(name)
                return {"status": "ok"}
            except Exception as e:
                return {"error": str(e)}

        @app.get("/iter_ensemble_preset")
        def get_iter_ensemble_presets():
            return {"presets": self.iterative_ensemble_app.get_list()}

        @app.get("/iter_ensemble_preset/{name}")
        def get_iter_ensemble_preset(name: str):
            try:
                state, warns_str = self.iterative_ensemble_app.load_preset(name)
                if warns_str:
                    return {"state": state, "warning": warns_str + "\n" + _i18n("ensemble_run_error_with_incorrect_flow")}
                return {"state": state}
            except Exception as e:
                return {"error": str(e)}

        @app.post("/iter_ensemble_preset/{name}")
        async def save_iter_ensemble_preset(name: str, request: Request):
            try:
                data = await request.json()
                self.iterative_ensemble_app.save_preset(name, data)
                return {"status": "ok"}
            except Exception as e:
                return {"error": str(e)}
            
        @app.delete("/iter_ensemble_preset/{name}")
        def delete_iter_ensemble_preset(name: str):
            try:
                self.iterative_ensemble_app.delete_preset(name)
                return {"status": "ok"}
            except Exception as e:
                return {"error": str(e)}
        # -----------------------------------

        PRESETLESS_HTML_CONTENT = """<!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
            <title>""" + f"{_i18n('preset_node_title')}" + """</title>
            <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap" rel="stylesheet">
            <style>

            /* -- PRESETLESS --- */

            :root {
                /* Основной фон (рабочее пространство) — делаем чуть более серым */
                --bg-color: var(--background-fill-primary, #eef1f5);
                
                --grid-color: var(--border-color-secondary, #d4d8dd); /* Слегка затемнили сетку для контраста */
                --sidebar-w: 250px; 
                --topbar-h: 60px;
                
                /* Фон панелей и нод — убрали #ffffff, заменили на мягкий светло-серый */
                --node-bg: var(--background-fill-secondary, #f6f8fa); 
                
                --node-header: var(--block-background-fill, #e6ebf1);
                --border-color: var(--border-color-primary, #d0d7de);
                --text-main: var(--body-text-color, #333333);
                --port-color: var(--color-accent-soft, #facc15);
                --port-border: var(--border-color-primary, #000000);
                --btn-blue: var(--color-accent, #007bff);
                --label-bg: var(--background-fill-primary, #e2ebf3);
                --label-border: var(--border-color-primary, #cdddeb);
                --primary: var(--color-accent, #007bff);
                --primary-hover: var(--color-accent-soft, #0056b3);
                --node-padding: 12px;
                --node-gap: 8px;
            }

                /* Fallback для темной темы, если CSS-переменные Gradio не пробросились во фрейм */
                @media (prefers-color-scheme: dark) {
                    :root {
                        --bg-color: var(--background-fill-primary, #0b0f19);
                        --grid-color: var(--border-color-secondary, #1f2937);
                        --node-bg: var(--background-fill-secondary, #1f2937);
                        --node-header: var(--block-background-fill, #374151);
                        --border-color: var(--border-color-primary, #374151);
                        --text-main: var(--body-text-color, #f3f4f6);
                        --port-border: var(--border-color-primary, #4b5563);
                        --btn-blue: var(--color-accent, #3b82f6);
                        --label-bg: var(--background-fill-primary, #111827);
                        --label-border: var(--border-color-primary, #374151);
                        --primary: var(--color-accent, #3b82f6);
                        --primary-hover: var(--color-accent-soft, #60a5fa);
                    }
                }

                .dark {
                    --bg-color: var(--background-fill-primary, #0b0f19);
                    --grid-color: var(--border-color-secondary, #1f2937);
                    --node-bg: var(--background-fill-secondary, #1f2937);
                    --node-header: var(--block-background-fill, #374151);
                    --border-color: var(--border-color-primary, #374151);
                    --text-main: var(--body-text-color, #f3f4f6);
                    --port-border: var(--border-color-primary, #4b5563);
                    --btn-blue: var(--color-accent, #3b82f6);
                    --label-bg: var(--background-fill-primary, #111827);
                    --label-border: var(--border-color-primary, #374151);
                    --primary: var(--color-accent, #3b82f6);
                    --primary-hover: var(--color-accent-soft, #60a5fa);
                }

                * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Montserrat', sans-serif; user-select: none; }
                body { width: 100vw; height: 100vh; overflow: hidden; display: flex; flex-direction: column; background: var(--bg-color); color: var(--text-main);}
                .topbar { height: var(--topbar-h); background: var(--node-bg); border-bottom: 1px solid var(--border-color); display: flex; align-items: center; padding: 0 20px; gap: 10px; z-index: 9999999; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }
                .topbar h1 { font-size: 20px; font-weight: 600; color: var(--text-main); margin-right: 15px; }
                input[type="text"] { padding: 7px 12px; border: 1px solid var(--border-color); border-radius: 4px; font-size: 14px; outline: none; background: var(--bg-color); color: var(--text-main); }
                .btn { padding: 8px 16px; border-radius: 4px; border: 1px solid var(--border-color); background: var(--node-bg); color: var(--text-main); cursor: pointer; font-weight: 500; font-size: 14px; transition: 0.2s; white-space: nowrap; }
                .btn:hover { background: var(--grid-color); } 
                .btn-blue { color: var(--btn-blue); border-color: var(--btn-blue); } 
                .btn-blue:hover { background: rgba(0, 123, 255, 0.1); }
                .btn-green { color: #28a745; border-color: #28a745; } 
                .btn-green:hover { background: rgba(40, 167, 69, 0.1); }
                .btn-red { color: #dc3545; border-color: #dc3545; } 
                .btn-red:hover { background: rgba(220, 53, 69, 0.1); }
                #sidebar-toggle { display: none; }

                /* === МОДАЛЬНЫЕ ОКНА === */
                .modal-overlay {
                    position: fixed;
                    top: 0; left: 0;
                    width: 100vw; height: 100vh;
                    background: rgba(0,0,0,0.4);
                    display: none;
                    align-items: center;
                    justify-content: center;
                    z-index: 99999999;
                    opacity: 0;
                    transition: opacity 0.2s;
                    padding: 16px;          /* ← отступ от краёв экрана */
                    box-sizing: border-box;
                }
                .modal-overlay.show { display: flex; opacity: 1; }
                .modal-box {
                    background: var(--node-bg);
                    padding: 25px;
                    border-radius: 8px;
                    width: 100%;            /* ← было min-width:320px */
                    max-width: 420px;       /* ← было max-width:90vw */
                    max-height: 90vh;       /* ← новое: ограничение высоты */
                    overflow-y: auto;       /* ← новое: скролл если контент не влезает */
                    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                    text-align: center;
                    box-sizing: border-box;
                }
                .modal-text { margin-bottom: 25px; font-size: 15px; font-weight: 500; color: var(--text-main); word-wrap: break-word; }
                .modal-buttons { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
                .modal-buttons .btn { min-width: 100px; }

                /* === ИНФО ПАНЕЛЬ === */
                .info-panel {
                    position: absolute;
                    top: 20px;
                    right: 20px;
                    background: var(--node-header);
                    color: var(--text-main);
                    border: 1px solid var(--border-color);
                    padding: 12px 16px;
                    border-radius: 6px;
                    font-size: 12px;
                    pointer-events: none;
                    z-index: 50000000;
                    min-width: 150px;
                    width: max-content;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
                }

                /* === КАСКАДНЫЕ ФИЛЬТРУЕМЫЕ СЕЛЕКТОРЫ (Интеграция) === */
                .custom-select-container { position: relative; box-sizing: border-box; min-width: 0; width: 100%; }
                .custom-select-input-wrapper { position: relative; }
                .custom-select-input { width: 100%; box-sizing: border-box; padding: 7px 30px 7px 10px; border: 1px solid var(--border-color); border-radius: 4px; font-size: 10px; background: var(--node-bg); color: var(--text-main); min-height: 35px; cursor: pointer; white-space: nowrap; pointer-events: auto; }
                .custom-select-input:focus { outline: none; border-color: var(--primary); }
                .custom-select-input-wrapper::after { content: '▼'; font-size: 10px; color: var(--text-main); position: absolute; right: 10px; top: 50%; transform: translateY(-50%); pointer-events: none; transition: transform 0.2s; opacity: 0.5; }
                .custom-select-container.open .custom-select-input-wrapper::after { transform: translateY(-50%) rotate(180deg); }
                .custom-select-options { position: absolute; top: 100%; left: 0; width: 100%; background: var(--node-bg); border: 1px solid var(--border-color); border-radius: 4px; margin-top: 4px; max-height: 200px; overflow-y: auto; overflow-x: auto; z-index: 999; display: none; box-shadow: 0 5px 15px rgba(0,0,0,0.15); pointer-events: auto; }
                .custom-select-container.open .custom-select-options { display: block; }
                .custom-select-container.open { z-index: 9999; }
                .custom-option { padding: 9px 12px; cursor: pointer; font-size: 13px; color: var(--text-main); transition: background 0.2s; white-space: pre-wrap; overflow: hidden; overflow-wrap: anywhere;  }
                .custom-option:hover { background: var(--grid-color); color: var(--primary-hover); }
                .custom-option.selected { background: var(--primary); color: #fff; font-weight: 600; }
                .custom-option.disabled { color: #aaa; cursor: not-allowed; background: var(--bg-color); }

                /* === ОСНОВНОЙ РАБОЧИЙ ПРОСТРАНСТВО === */
                .main-container { display: flex; flex: 1; height: calc(100vh - var(--topbar-h)); overflow: hidden;}
                .sidebar { width: var(--sidebar-w); background: var(--node-bg); border-right: 1px solid var(--border-color); overflow-y: auto; padding: 15px; display: flex; flex-direction: column; gap: 10px; z-index: 10; }
                .node-item { padding: 10px; border: 1px solid var(--border-color); border-radius: 6px; text-align: center; cursor: grab; font-weight: 500; font-size: 13px; color: var(--text-main); transition: 0.2s; background: var(--node-bg);}
                .node-item:hover { box-shadow: 0 2px 5px rgba(0,0,0,0.1); border-color: var(--primary); }
                .workspace { flex: 1; position: relative; overflow: hidden; background-color: var(--bg-color); background-image: radial-gradient(var(--grid-color) 1px, transparent 1px); background-size: 20px 20px; cursor: grab; touch-action: none; }
                .workspace:active { cursor: grabbing; }
                .custom-select-options { touch-action: pan-y; }
                /* Индикатор масштаба — всплывает при pinch и зуме колесом */
                #pinch-zoom-badge { position: absolute; left: 50%; bottom: 26px; transform: translateX(-50%) translateY(10px); background: var(--node-header); color: var(--text-main); border: 1px solid var(--border-color); border-left: 3px solid var(--primary); padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: 600; font-variant-numeric: tabular-nums; box-shadow: 0 4px 14px rgba(0,0,0,0.2); opacity: 0; pointer-events: none; transition: opacity 0.25s ease, transform 0.25s ease; z-index: 60000000; }
                #pinch-zoom-badge.show { opacity: 1; transform: translateX(-50%) translateY(0); }
                .transform-layer { position: absolute; top: 0; left: 0; transform-origin: 0 0; }
                svg { position: absolute; top: 0; left: 0; pointer-events: none; overflow: visible}
                .link-path { fill: none; stroke: var(--primary); stroke-width: 8; pointer-events: stroke; cursor: pointer; transition: stroke-width 0.2s; }
                .link-path:hover { stroke: var(--primary-hover); stroke-width: 12; }

                /* ===== РАСШИРЕНИЕ НОДЫ ===== */
                .node { position: absolute; background: var(--node-bg); border: 1px solid var(--border-color); border-radius: 8px; width: max-content !important; min-width: 180px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); cursor: default; display: flex; flex-direction: column; padding-bottom: 10px; will-change: left, top, transform; backface-visibility: hidden; transform: translateZ(0); -webkit-user-drag: none; }
                .node-header { position: relative; background: var(--node-header); color: var(--text-main); padding: 10px 70px 10px 50px; border-bottom: 1px solid var(--border-color); border-top-left-radius: 7px; border-top-right-radius: 7px; font-weight: 600; font-size: 14px; text-align: center; cursor: grab; white-space: nowrap; flex-shrink: 0; -webkit-user-drag: none; touch-action: none; }
                .node-header:active, .node:active .node-header { cursor: grabbing !important; }
                .node-close { position: absolute; right: 7px; top: 50%; transform: translateY(-50%); cursor: pointer; color: #dc3545; font-size: 30px; line-height: 1; transition: color 0.2s; padding: 0px 4px; }
                .node-close:hover { color: #ff6666; transform: scale(1.6) translateY(-30%); }
                .node-content { padding: var(--node-padding); display: flex; flex-direction: column; gap: var(--node-gap); font-size: 12px; width: 100%; min-width: 0; flex: 1; }
                .node-content select, .node-content input[type="text"] { padding: 6px 32px 6px 10px; border: 1px solid var(--border-color); border-radius: 4px; font-size: 12px; background: var(--bg-color); color: var(--text-main); min-height: 32px; box-sizing: border-box; transition: border-color 0.2s; width: max-content !important; min-width: 100%; field-sizing: content; }
                .node-content input[type="number"] { padding: 6px 10px; border: 1px solid var(--border-color); border-radius: 4px; font-size: 12px; background: var(--bg-color); color: var(--text-main); min-height: 32px; box-sizing: border-box; transition: border-color 0.2s; width: max-content !important; min-width: 100%; field-sizing: content; }
                .node-content select:focus, .node-content input[type="text"]:focus, .node-content input[type="number"]:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 2px rgba(0,123,255,0.1); }

                .toggle-container { display: flex; justify-content: space-between; align-items: center; width: 100%; padding: 2px 0; gap: 10px; }
                .toggle-label { flex: 1; font-size: 12px; color: var(--text-main); min-width: 0; word-wrap: break-word; overflow-wrap: break-word; }
                .toggle-switch { position: relative; display: inline-block; width: 34px !important; min-width: 34px !important; max-width: 34px !important; height: 20px !important; flex-shrink: 0; }
                .toggle-switch input { opacity: 0; width: 0; height: 0; margin: 0; }
                .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: var(--border-color); transition: .3s; border-radius: 20px; }
                .slider:before { position: absolute; content: ""; height: 14px; width: 14px; left: 3px; bottom: 3px; background-color: var(--node-bg); transition: .3s; border-radius: 50%; box-shadow: 0 1px 3px rgba(0,0,0,0.3); }
                input:checked + .slider { background-color: var(--btn-blue); } 
                input:checked + .slider:before { transform: translateX(14px); }

                .ports { display: flex; justify-content: space-between; position: absolute; width: 100%; top: 50%; pointer-events: none; height: 0; }
                .port { width: 64px; height: 28px; position: absolute; pointer-events: auto; cursor: crosshair; display: flex; align-items: center; justify-content: center; transition: transform 0.1s ease; }
                .port:hover { transform: scale(1.4); } 
                .port-in { left: -32px; } 
                .port-out { right: -32px; }
                .port-in::before { content: ""; width: 18px; height: 18px; background: var(--port-color); border: 2px solid var(--port-border); border-radius: 50%; box-sizing: border-box; }
                .port-out::before { content: ""; width: 18px; height: 18px; background: var(--node-bg); border: 2px solid var(--port-border); border-radius: 50%; box-sizing: border-box; }
                .port-label-container { position: relative; width: 100%; min-height: 28px; display: flex; align-items: center; padding: 0px 0; }
                .port-label-in { padding-left: 18px; flex: 1; display: flex; align-items: center; }
                .port-label-out { padding-right: 18px; flex: 1; display: flex; align-items: center; justify-content: flex-end; }
                .port-label-text { background-color: var(--label-bg); border: 1px solid var(--label-border); padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 500; color: var(--text-main); white-space: nowrap; max-width: 100%; overflow: hidden; text-overflow: ellipsis; }
                .port-group { display: flex; flex-direction: column; gap: 0px; width: 100%; }

                #import-zone.dragover { background-color: rgba(0, 123, 255, 0.1); border: 2px dashed var(--btn-blue); }

                .node-content .custom-select-container { width: max-content !important; min-width: 100%; }
                .node-content .custom-select-input-wrapper { width: max-content !important; min-width: 100%; }
                .node-content .custom-select-input { width: max-content !important; min-width: 100% !important; font-size: 12px; min-height: 32px; padding: 6px 30px 6px 10px; box-sizing: border-box; }
                .node[style*="width"] { width: max-content !important; }

                /* ===== СТАТУСЫ НОД (Прогресс из бэкенда) ===== */
                @keyframes blink-blue {
                    0% { box-shadow: 0 0 5px var(--primary); border-color: var(--primary); }
                    50% { box-shadow: 0 0 20px var(--primary); border-color: var(--primary); }
                    100% { box-shadow: 0 0 5px var(--primary); border-color: var(--primary); }
                }

                .node.status-active {
                    animation: blink-blue 2s infinite;
                    border-color: var(--primary);
                    z-index: 50;
                }

                .node.status-success {
                    box-shadow: 0 0 15px rgba(40, 167, 69, 0.6);
                    border-color: #28a745;
                }

                .node.status-error {
                    box-shadow: 0 0 15px rgba(220, 53, 69, 0.6);
                    border-color: #dc3545;
                }

                /* ===== ЦЕЛЕВОЙ ИНСТРУМЕНТ В ЗАГОЛОВКЕ ===== */
                .node-subtitle {
                    background: var(--primary); /* Используем цвет соединения (акцентный синий) */
                    color: #ffffff; /* Белый текст */
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-size: 10px;
                    margin-left: 8px;
                    vertical-align: middle;
                    font-weight: 500;
                    pointer-events: none;
                    display: none; /* Скрыто по умолчанию */
                }

                /* ===== ВЫДЕЛЕНИЕ ЦЕЛЕВОГО ИНСТРУМЕНТА В ПОРТАХ ===== */
                .port-label-text.target-stem {
                    background-color: var(--primary); /* Используем акцентный синий цвет */
                    color: #ffffff; /* Белый текст для контраста */
                    border-color: var(--primary-hover);
                    font-weight: 600;
                    box-shadow: 0 0 8px rgba(0, 123, 255, 0.3);
                }


                .editor-locked .node-header,
                .editor-locked .node-close,
                .editor-locked .port,
                .editor-locked .node-content input,
                .editor-locked .node-content select,
                .editor-locked .node-content .toggle-switch,
                .editor-locked .sidebar .node-item,
                .editor-locked .topbar .btn-green,
                .editor-locked .topbar .btn-red,
                .editor-locked .topbar #import-btn,
                .editor-locked .topbar #preset-name,
                .editor-locked .topbar #preset-select-container,
                .editor-locked .link-path {
                    pointer-events: none !important;
                    cursor: not-allowed !important;
                    opacity: 0.6 !important;
                }
                /* Разрешаем панорамирование и зум */
                .editor-locked #workspace {
                    cursor: grab !important;
                }
                .editor-locked #workspace:active {
                    cursor: grabbing !important;
                }





                @media (max-width: 1024px) {
                    .node-item { font-size: 10px; flex-shrink: 0; }
                    .topbar { flex-wrap: wrap; height: auto; padding: 10px; justify-content: space-between; }
                    .topbar h1 { display: none; }
                    .custom-select-container { width: 100% !important; margin-bottom: 5px; }
                    .topbar input[type="text"] { flex: 1 1 auto; width: 100%; min-width: 150px; margin-bottom: 5px; }
                    .topbar .btn { flex: 1 1 auto; margin: 2px; font-size: 12px; padding: 6px 10px; }
                    #sidebar-toggle { display: none !important; }
                    .main-container { flex-direction: column; }
                    .sidebar { 
                        position: relative; 
                        left: 0; 
                        top: 0; 
                        bottom: auto;
                        width: 100%; 
                        height: auto; 
                        flex-direction: row;
                        overflow-x: auto;
                        overflow-y: hidden;
                        padding: 10px;
                        padding-bottom: 35px;
                        z-index: 10; 
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                        border-right: none;
                        border-bottom: 1px solid var(--border-color);
                    }
                }

            </style>
        </head>
        <body>

            <!-- МОДАЛЬНОЕ ОКНО -->
            <div id="custom-modal-overlay" class="modal-overlay">
                <div class="modal-box">
                    <div class="modal-text" id="modal-text"></div>
                    <div class="modal-buttons">
                        <button class="btn btn-blue" id="modal-btn-ok">""" + f"{_i18n('preset_node_ok')}" + """</button>
                        <button class="btn btn-red" id="modal-btn-cancel" style="display:none;">""" + f"{_i18n('preset_node_cancel')}" + """</button>
                    </div>
                </div>
            </div>

            <div class="topbar">
                <button id="sidebar-toggle" class="btn btn-blue" onclick="toggleSidebar()">""" + f"{_i18n('preset_node_toggle_sidebar')}" + """</button>
                <div class="custom-select-container" id="preset-select-container" data-value="" style="width: 220px;" onmouseenter="fetchServerPresets()">
                    <div class="custom-select-input-wrapper">
                        <input type="text" class="custom-select-input" placeholder=""" + f"\"{_i18n('preset_node_saved')}\"" + """ onfocus="openDropdown(this)" onclick="openDropdown(this)" oninput="filterOptions(this)" autocomplete="off">
                    </div>
                    <div class="custom-select-options" id="server-preset-list">
                        <div class="custom-option disabled">""" + f"{_i18n('preset_node_loading')}" + """</div>
                    </div>
                </div>

                <input type="text" id="preset-name" placeholder=""" + f"\"{_i18n('preset_node_preset_name')}\"" + """ value="">
                
                <button class="btn btn-green" onclick="saveServerPreset()">""" + f"{_i18n('preset_node_save')}" + """</button>
                <button class="btn btn-red" onclick="deleteServerPreset()" style="margin-right: 5px;">""" + f"{_i18n('preset_node_delete')}" + """</button>

                <span style="border-left: 1px solid var(--border-color); height: 30px; margin: 0 5px;" class="desktop-only"></span>

                <button class="btn btn-blue" onclick="exportJSON()">""" + f"{_i18n('preset_node_preset_download')}" + """</button>
                <button class="btn" id="import-btn">""" + f"{_i18n('preset_node_preset_upload')}" + """</button>
                <input type="file" id="import-file" style="display:none" accept=".json">
                
                <button class="btn btn-red" onclick="clearWorkspace()" style="margin-left: auto;">""" + f"{_i18n('preset_node_clear')}" + """</button>
            </div>

            <div class="main-container">
                <div class="sidebar" id="sidebar"></div>
                <div class="workspace" id="workspace">
                    <div id="pinch-zoom-badge">100%</div>
                    
                    <!-- ИНФОРМАЦИОННАЯ ПАНЕЛЬ -->
                    <div id="info-panel" class="info-panel">
                        <div id="info-links">Связей: 0</div>
                        <div id="info-nodes">Всего нод: 0</div>
                        <div style="margin-top: 5px;">Выходные ноды:</div>
                        <div id="info-outputs"></div>
                    </div>

                    <div class="transform-layer" id="transform-layer">
                        <svg id="svg-canvas"></svg>
                        <div id="nodes-container"></div>
                    </div>
                </div>
            </div>

            <script>

                const urlParams = new URLSearchParams(window.location.search);
                if (urlParams.get('__theme') === 'dark') {
                    document.documentElement.classList.add('dark');
                    document.body.classList.add('dark');
                }

                window.addEventListener('message', function(event) {
                    // Проверяем как прямую строку, так и возможное поле в объекте
                    let themeData = event.data;
                    if (typeof event.data === 'object' && event.data !== null) {
                        // Зависит от того, как именно Gradio сейчас присылает тему
                        themeData = event.data.theme || event.data.type; 
                    }
                    
                    if (themeData === 'theme_dark' || themeData === 'dark') {
                        document.documentElement.classList.add('dark');
                        document.body.classList.add('dark');
                    } else if (themeData === 'theme_light' || themeData === 'light') {
                        document.documentElement.classList.remove('dark');
                        document.body.classList.remove('dark');
                    }
                });
                // === Custom Modal Logic ===
                function customAlert(msg) {
                    return new Promise(resolve => {
                        const overlay = document.getElementById('custom-modal-overlay');
                        document.getElementById('modal-text').innerText = msg;
                        const btnOk = document.getElementById('modal-btn-ok');
                        const btnCancel = document.getElementById('modal-btn-cancel');
                        
                        // Скрываем кнопку отмены, так как это просто Alert
                        if (btnCancel) btnCancel.style.display = 'none';
                        
                        btnOk.onclick = () => { 
                            overlay.classList.remove('show'); 
                            resolve(); 
                        };
                        
                        // Убираем style.display = 'flex' и полагаемся на CSS класс .show
                        overlay.classList.add('show');
                    });
                }

                function customConfirm(msg) {
                    return new Promise(resolve => {
                        const overlay = document.getElementById('custom-modal-overlay');
                        document.getElementById('modal-text').innerText = msg;
                        const btnOk = document.getElementById('modal-btn-ok');
                        const btnCancel = document.getElementById('modal-btn-cancel');
                        
                        btnCancel.style.display = 'inline-block';
                        
                        btnOk.onclick = () => { 
                            overlay.classList.remove('show'); 
                            resolve(true); 
                        };
                        btnCancel.onclick = () => { 
                            overlay.classList.remove('show'); 
                            resolve(false); 
                        };
                        
                        overlay.classList.add('show');
                    });
                }

                // === Sidebar Toggle ===
                function toggleSidebar() {
                    document.getElementById('sidebar').classList.toggle('open');
                }

                // === Dropdowns ===
                function openDropdown(input) {
                    const container = input.closest('.custom-select-container');
                    if (container.classList.contains('open')) return;

                    document.querySelectorAll('.custom-select-container').forEach(c => {
                        if (c !== container) closeDropdown(c);
                    });

                    const node = container.closest('.node');
                    if (node) {
                        document.querySelectorAll('.node').forEach(n => n.style.zIndex = 1);
                        node.style.zIndex = 100;
                    }

                    container.classList.add('open');
                    if (container.classList.contains('filterable')) {
                        setTimeout(() => { input.setSelectionRange(0, input.value.length); }, 0);
                    }
                    container.querySelectorAll('.custom-option').forEach(opt => opt.style.display = '');
                }

                function closeDropdown(container) {
                    if (!container.classList.contains('open')) return;
                    container.classList.remove('open');
                    
                    const input = container.querySelector('.custom-select-input');
                    const val = container.getAttribute('data-value');
                    
                    if (container.classList.contains('filterable')) {
                        if (val) {
                            const selectedOpt = container.querySelector(`.custom-option[data-value="${val}"]`);
                            if (selectedOpt) {
                                const currentText = input.value.trim();
                                if (currentText && currentText !== selectedOpt.innerText) {} else { input.value = selectedOpt.innerText; }
                            } else {
                                if (!input.value.trim()) { input.value = ''; }
                            }
                        } else {
                            if (!input.value.trim()) { input.value = ''; }
                        }
                    } else {
                        if (val && container.id !== 'preset-select-container') {
                            const selectedOpt = container.querySelector(`.custom-option[data-value="${val}"]`);
                            if (selectedOpt) input.value = selectedOpt.innerText;
                        } else if (container.id !== 'preset-select-container') {
                            if (!input.hasAttribute('readonly')) input.value = '';
                        } else if (container.id === 'preset-select-container') {
                            input.value = ''; 
                        }
                    }
                }

                function filterOptions(input) {
                    const container = input.closest('.custom-select-container');
                    const query = input.value.toLowerCase().trim();
                    const options = container.querySelectorAll('.custom-option');

                    container.classList.add('open');
                    options.forEach(opt => {
                        if (opt.classList.contains('disabled')) return;
                        const text = opt.innerText.toLowerCase();
                        opt.style.display = text.includes(query) ? '' : 'none';
                    });
                }

                function selectOption(optionElement) {
                    if (optionElement.classList.contains('disabled')) return;

                    const container = optionElement.closest('.custom-select-container');
                    const input = container.querySelector('.custom-select-input');
                    const val = optionElement.getAttribute('data-value');
                    
                    container.setAttribute('data-value', val);
                    input.value = optionElement.innerText;
                    
                    container.querySelectorAll('.custom-option').forEach(opt => opt.classList.remove('selected'));
                    optionElement.classList.add('selected');
                    container.classList.remove('open');

                    if (container.id === 'preset-select-container') {
                        loadServerPreset(val);
                        input.value = '';
                    } else if (input.dataset.param) {
                        if (typeof updateDynamicPorts === 'function') {
                            updateDynamicPorts(input);
                        }
                    }
                }

                document.addEventListener('click', function(e) {
                    if (!e.target.closest('.custom-select-container')) {
                        document.querySelectorAll('.custom-select-container').forEach(c => closeDropdown(c));
                    }
                    if (window.innerWidth <= 768 && !e.target.closest('.sidebar') && !e.target.closest('#sidebar-toggle')) {
                        document.getElementById('sidebar').classList.remove('open');
                    }
                });

                // ОБНОВЛЕНИЕ ИНФО-ПАНЕЛИ
                function updateInfoPanel() {
                    const panelLinks = document.getElementById('info-links');
                    const panelNodes = document.getElementById('info-nodes');
                    const panelOutputs = document.getElementById('info-outputs');
                    
                    panelLinks.innerText = `""" + f"{_i18n('preset_node_info_links')}" + """${links.length}`;
                    panelNodes.innerText = `""" + f"{_i18n('preset_node_info_nodes')}" + """${Object.keys(nodes).length}`;
                    
                    let outputsHTML = '';
                    Object.values(nodes).forEach(n => {
                        if (n.type === 'output_file') {
                            let stem = n.params.name_stem || "output";
                            outputsHTML += `<div style="margin-left:8px">- ${stem}</div>`;
                        }
                    });
                    if (outputsHTML === '') outputsHTML = `<div style="margin-left:8px">""" + f"{_i18n('preset_node_info_none')}" + """</div>`;
                    panelOutputs.innerHTML = outputsHTML;
                }

                let stateTimeout;
                window.sendStateToParent = function() {
                    updateInfoPanel();
                    clearTimeout(stateTimeout);
                    stateTimeout = setTimeout(() => {
                        const name = document.getElementById('preset-name').value || "preset";
                        const data = { nodes, links, name };
                        window.parent.postMessage({ type: 'update_preset', payload: data }, '*');
                    }, 100); 
                };

                // ДОБАВЛЕННАЯ ФУНКЦИЯ ДЛЯ ОБНОВЛЕНИЯ СТАТУСА
                function updateNodeStatus(nodeId, status) {
                    const el = document.getElementById(nodeId);
                    if (!el) return;
                    
                    el.classList.remove('status-active', 'status-success', 'status-error');
                    
                    if (status) {
                        el.classList.add('status-' + status);
                    }
                }

                window.addEventListener('message', (e) => {
                    if (e.data) {
                        if (e.data.type === 'set_preset') {
                            loadJSON(e.data.payload);
                        }
                        
                        // Отслеживание прогресса конкретной ноды
                        if (e.data.type === 'node_status') {
                            const { nodeId, status } = e.data.payload;
                            updateNodeStatus(nodeId, status); // <--- Теперь функция будет корректно вызвана
                        }
                        
                        // Сброс статуса всех нод
                        if (e.data.type === 'reset_all_statuses') {
                            document.querySelectorAll('.node').forEach(el => {
                                el.classList.remove('status-active', 'status-success', 'status-error');
                            });
                        }
                    }
                });

                async function fetchServerPresets() {
                    try {
                        const res = await fetch('/presetless');
                        const data = await res.json();
                        const list = document.getElementById('server-preset-list');
                        list.innerHTML = '';
                        
                        if (!data.presets || data.presets.length === 0) {
                            list.innerHTML = '<div class="custom-option disabled">""" + f"{_i18n('preset_node_no_presets')}" + """</div>';
                            return;
                        }
                        
                        data.presets.forEach(p => {
                            let div = document.createElement('div');
                            div.className = 'custom-option';
                            div.setAttribute('data-value', p);
                            div.innerText = p;
                            div.onclick = function() { selectOption(this); };
                            list.appendChild(div);
                        });
                    } catch(e) {
                        console.error(""" + f"\"{_i18n('preset_node_err_fetch')}\"" + """, e);
                    }
                }

                async function loadServerPreset(name) {
                    try {
                        const res = await fetch(`/presetless/${name}`);
                        const data = await res.json();
                        if (data.error) throw new Error(data.error);
                        
                        loadJSON(data);
                        document.getElementById('preset-name').value = name;
                        sendStateToParent();
                    } catch(e) {
                        await customAlert(""" + f"\"{_i18n('preset_node_err_load')}\"" + """ + e.message);
                    }
                }

                async function saveServerPreset() {
                    const name = document.getElementById('preset-name').value.trim();
                    if (!name) {
                        await customAlert(""" + f"\"{_i18n('preset_node_err_name_empty')}\"" + """);
                        return;
                    }
                    const data = { nodes, links, name };
                    try {
                        const res = await fetch(`/presetless/${name}`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(data)
                        });
                        const result = await res.json();
                        if (result.error) throw new Error(result.error);
                        await customAlert(`""" + f"{_i18n('preset_node_save_success')}" + """`);
                    } catch(e) {
                        await customAlert(""" + f"\"{_i18n('preset_node_error_save')}\"" + """ + e.message);
                    }
                }

                async function deleteServerPreset() {
                    const name = document.getElementById('preset-name').value.trim();
                    if (!name) return customAlert('""" + f"{_i18n('preset_node_err_name_delete')}" + """');

                    if (!(await customConfirm('""" + f"{_i18n('preset_node_confirm_delete')}" + """ "' + name + '"?'))) return;

                    try {
                        const res = await fetch(`/presetless/${name}`, { method: 'DELETE' });
                        const result = await res.json();

                        if (result.error) throw new Error(result.error);

                        clearWorkspace();
                        await customAlert('""" + f"{_i18n('preset_node_delete_success')}" + """');
                        fetchServerPresets();
                    } catch (e) {
                        await customAlert('""" + f"{_i18n('preset_node_error_delete')}" + """' + e.message);
                    }
                }

                const NODE_DEFINITIONS = {
                    "input_file": { title: """ + f"\"{_i18n('preset_node_input_file')}\"" + """, params: {} },
                    "output_file": { title: """ + f"\"{_i18n('preset_node_output_file')}\"" + """, params: { name_stem: "output", output_format: "mp3", prefer_float: false }, html: `
                        <label>""" + f"{_i18n('preset_node_name_stem')}" + """</label>
                        <input type="text" data-param="name_stem" placeholder=""" + f"\"{_i18n('preset_node_stem_name')}\"" + """>
                        <label>""" + f"{_i18n('preset_node_output_format')}" + """</label>
                        <div class="custom-select-container" data-param="output_format">
                            <div class="custom-select-input-wrapper">
                                <input type="text" class="custom-select-input" data-param="output_format" placeholder=""" + f"\"{_i18n('preset_node_choose_format')}\"" + """ onclick="openDropdown(this)" readonly autocomplete="off">
                            </div>
                            <div class="custom-select-options">
                                """ + "".join([f"<div class='custom-option' data-value='{of}'>{of}</div>" for of in output_formats]) + """
                            </div>
                        </div>
                        <div class="toggle-container">
                            <label>""" + f"{_i18n('preset_node_prefer_float')}" + """</label>
                            <label class="toggle-switch"><input type="checkbox" data-param="prefer_float"><span class="slider"></span></label>
                        </div>
                    ` },
                    "gain": { title: """ + f"\"{_i18n('preset_node_gain')}\"" + """, params: { gain: 1.0 }, html: `<label>""" + f"{_i18n('preset_node_gain_factor')}" + """</label><input type="number" step="0.01" min="0" max="20" data-param="gain">` },
                    "normalize": { title: """ + f"\"{_i18n('preset_node_normalize')}\"" + """, params: { peak: 1.0 }, html: `<label>""" + f"{_i18n('preset_node_peak')}" + """</label><input type="number" step="0.01" min="0" max="20" data-param="peak">` },
                    "trim": { title: """ + f"\"{_i18n('preset_node_trim')}\"" + """, params: { start: 0, end: 30 }, html: `
                        <label>""" + f"{_i18n('start_sec')}" + """</label>
                        <input type="number" step="0.1" min="0" data-param="start">
                        <label>""" + f"{_i18n('end_sec')}" + """</label>
                        <input type="number" step="0.1" min="0.1" data-param="end">
                    ` },
                    "filter": { title: """ + f"\"{_i18n('preset_node_filter')}\"" + """, params: { kind: "hp", fft_mode: true, cutoff: 100 }, html: `
                        <label>""" + f"{_i18n('preset_node_kind')}" + """</label>
                        <div class="custom-select-container" data-param="kind">
                            <div class="custom-select-input-wrapper">
                                <input type="text" class="custom-select-input" data-param="kind" placeholder=""" + f"\"{_i18n('preset_node_choose_type')}\"" + """ onclick="openDropdown(this)" readonly autocomplete="off">
                            </div>
                            <div class="custom-select-options">
                                <div class="custom-option" data-value="hp">""" + f"{_i18n('highpass')}" + """</div>
                                <div class="custom-option" data-value="lp">""" + f"{_i18n('lowpass')}" + """</div>
                            </div>
                        </div>
                        <label>""" + f"{_i18n('preset_node_cutoff')}" + """</label>
                        <input type="number" step="1" min="10" max="22050" data-param="cutoff">
                        <div class="toggle-container">
                            <label>""" + f"{_i18n('preset_node_use_spectrogram')}" + """</label>
                            <label class="toggle-switch"><input type="checkbox" data-param="fft_mode" checked><span class="slider"></span></label>
                        </div>
                    ` },
                    "phase_shift": { title: """ + f"\"{_i18n('preset_node_phase_shift')}\"" + """, params: { degrees: 90 }, html: `
                        <label>""" + f"{_i18n('phase_angle')}" + """</label>
                        <input type="number" step="1" min="-360" max="360" data-param="degrees">
                    ` },
                    "phase_correct": { title: """ + f"\"{_i18n('preset_node_phase_correct')}\"" + """, params: { transfer_magnitude: false, transfer_phase: true, freq_blend_phases: true, low_cutoff: 500, high_cutoff: 5000 }, html: `
                        <div class="toggle-container">
                            <label>""" + f"{_i18n('preset_node_transfer_magnitude')}" + """</label>
                            <label class="toggle-switch"><input type="checkbox" data-param="transfer_magnitude" checked><span class="slider"></span></label>
                        </div>
                        <div class="toggle-container">
                            <label>""" + f"{_i18n('preset_node_transfer_phase')}" + """</label>
                            <label class="toggle-switch"><input type="checkbox" data-param="transfer_phase" checked><span class="slider"></span></label>
                        </div>
                        <div class="toggle-container">
                            <label>""" + f"{_i18n('preset_node_freq_blend_phases')}" + """</label>
                            <label class="toggle-switch"><input type="checkbox" data-param="freq_blend_phases" checked><span class="slider"></span></label>
                        </div>
                        <label>""" + f"{_i18n('preset_node_low_cutoff')}" + """</label>
                        <input type="number" step="10" min="20" max="20000" data-param="low_cutoff">
                        <label>""" + f"{_i18n('preset_node_high_cutoff')}" + """</label>
                        <input type="number" step="10" min="20" max="20000" data-param="high_cutoff">
                    ` },
                    "mix": { title: """ + f"\"{_i18n('preset_node_mix')}\"" + """, params: { num_inputs: 2 }, html: `<label>""" + f"{_i18n('preset_node_inputs')}" + """</label><input type="number" min="1" max="10" data-param="num_inputs" onchange="updateDynamicPorts(this)">` },
                    "ensemble": { title: """ + f"\"{_i18n('preset_node_ensemble')}\"" + """, params: { num_inputs: 2, type: "avg_fft" }, html: `
                        <label>""" + f"{_i18n('preset_node_inputs')}" + """</label>
                        <input type="number" min="1" max="10" data-param="num_inputs" onchange="updateDynamicPorts(this)">
                        <label>""" + f"{_i18n('preset_node_ensemble_type')}" + """</label>
                        <div class="custom-select-container" data-param="type">
                            <div class="custom-select-input-wrapper">
                                <input type="text" class="custom-select-input" data-param="type" placeholder=""" + f"\"{_i18n('preset_node_choose_type')}\"" + """ onclick="openDropdown(this)" readonly autocomplete="off">
                            </div>
                            <div class="custom-select-options">
                                <div class="custom-option" data-value="avg_fft">avg_fft</div>
                                <div class="custom-option" data-value="min_fft">min_fft</div>
                                <div class="custom-option" data-value="max_fft">max_fft</div>
                                <div class="custom-option" data-value="median_fft">median_fft</div>
                            </div>
                        </div>
                    ` },
                    "split_stereo": { title: """ + f"\"{_i18n('preset_node_split_stereo')}\"" + """, params: { var: "left/right" }, html: `
                        <label>""" + f"{_i18n('preset_node_stereo_mode')}" + """</label>
                        <div class="custom-select-container" data-param="var">
                            <div class="custom-select-input-wrapper">
                                <input type="text" class="custom-select-input" data-param="var" placeholder=""" + f"\"{_i18n('preset_node_choose_mode')}\"" + """ onclick="openDropdown(this)" readonly autocomplete="off">
                            </div>
                            <div class="custom-select-options">
                                <div class="custom-option" data-value="left/right">""" + f"{_i18n('preset_node_leftright')}" + """</div>
                                <div class="custom-option" data-value="mid/side">""" + f"{_i18n('preset_node_midside')}" + """</div>
                                <div class="custom-option" data-value="sim/dif">""" + f"{_i18n('preset_node_simdif')}" + """</div>
                            </div>
                        </div>
                    ` },
                    "join_stereo": { title: """ + f"\"{_i18n('preset_node_join_stereo')}\"" + """, params: { var: "left/right" }, html: `
                        <label>""" + f"{_i18n('preset_node_stereo_mode')}" + """</label>
                        <div class="custom-select-container" data-param="var">
                            <div class="custom-select-input-wrapper">
                                <input type="text" class="custom-select-input" data-param="var" placeholder=""" + f"\"{_i18n('preset_node_choose_mode')}\"" + """ onclick="openDropdown(this)" readonly autocomplete="off">
                            </div>
                            <div class="custom-select-options">
                                <div class="custom-option" data-value="left/right">""" + f"{_i18n('preset_node_leftright')}" + """</div>
                                <div class="custom-option" data-value="mid/side">""" + f"{_i18n('preset_node_midside')}" + """</div>
                                <div class="custom-option" data-value="sim/dif">""" + f"{_i18n('preset_node_simdif')}" + """</div>
                            </div>
                        </div>
                    ` },
                    "subtract": { title: """ + f"\"{_i18n('preset_node_subtract')}\"" + """, params: { use_spectrogram: false }, html: `<div class="toggle-container"><label>""" + f"{_i18n('preset_node_use_spectrogram')}" + """</label><label class="toggle-switch"><input type="checkbox" data-param="use_spectrogram"><span class="slider"></span></label></div>` },
                    "invert": { title: """ + f"\"{_i18n('preset_node_invert')}\"" + """, params: {} },
                    "separate": { title: """ + f"\"{_i18n('preset_node_separate')}\"" + """, params: { model_name: "" }, html: `
                        <label>""" + f"{_i18n('preset_node_model_name')}" + """</label>
                        <div class="custom-select-container filterable">
                            <div class="custom-select-input-wrapper">
                                <input type="text" class="custom-select-input" data-param="model_name" placeholder=""" + f"\"{_i18n('preset_node_choose_model')}\"" + """ onfocus="openDropdown(this)" onclick="openDropdown(this)" oninput="filterOptions(this)" autocomplete="off">
                            </div>
                            <div class="custom-select-options custom-model-options"></div>
                        </div>
                    ` }
                };

                let modelsData = {}; let nodes = {}; let links = [];
                let transform = { x: 0, y: 0, scale: 1 };
                let draggedNode = null; let dragOffset = { x: 0, y: 0 };
                let isPanning = false; let panStart = { x: 0, y: 0 };
                let connectingPort = null; let tempLink = null; let idCounter = 1;
                let touchDragElement = null; let touchDragType = null;

                const workspace = document.getElementById('workspace');
                const transformLayer = document.getElementById('transform-layer');
                const nodesContainer = document.getElementById('nodes-container');
                const svgCanvas = document.getElementById('svg-canvas');

                async function init() {
                    try {""" + """                      modelsData = """ + json.dumps(self.separator.info, ensure_ascii=False, indent=0) + ";" + """
                    } catch (e) {
                        console.error("Failed to load models.json", e);
                        modelsData = { "NO_MODELS": { "stems": ["Vocals", "Instrumental"] } }; 
                    }
                    buildSidebar();
                    setupEvents();
                    setupDragDrop();
                    setupImport();
                    
                    document.getElementById('preset-name').addEventListener('input', sendStateToParent);
                    sendStateToParent();
                }


                function buildSidebar() {
                    const sb = document.getElementById('sidebar');
                    for (let type in NODE_DEFINITIONS) {
                        let div = document.createElement('div');
                        div.className = 'node-item';
                        div.innerText = NODE_DEFINITIONS[type].title;
                        div.draggable = true;
                        
                        div.addEventListener('dragstart', e => { e.dataTransfer.setData('type', type); });
                        
                        div.addEventListener('touchstart', e => {
                            e.preventDefault(); 
                            touchDragType = type;
                            const touch = e.touches[0];
                            touchDragElement = div.cloneNode(true);
                            touchDragElement.style.position = 'fixed'; touchDragElement.style.margin = '0';
                            touchDragElement.style.width = div.offsetWidth + 'px';
                            touchDragElement.style.left = (touch.clientX - div.offsetWidth / 2) + 'px';
                            touchDragElement.style.top = (touch.clientY - div.offsetHeight / 2) + 'px';
                            touchDragElement.style.opacity = '0.9'; touchDragElement.style.zIndex = '9999';
                            touchDragElement.style.pointerEvents = 'none'; touchDragElement.style.boxShadow = '0 5px 15px rgba(0,0,0,0.2)';
                            document.body.appendChild(touchDragElement);
                        }, {passive: false});
                        
                        sb.appendChild(div);
                    }
                }

                window.deleteNode = function(nodeId) {
                    delete nodes[nodeId];
                    const el = document.getElementById(nodeId);
                    if (el) el.remove();
                    links = links.filter(l => l.fromNode !== nodeId && l.toNode !== nodeId);
                    drawLinks();
                    sendStateToParent(); 
                }

                function addNode(type, x, y, id = null, loadedParams = null, isImport = false) {
                    // Ограничение: не больше одной ноды "input_file"
                    if (type === 'input_file') {
                        const existingInputs = Object.values(nodes).filter(n => n.type === 'input_file');
                        if (existingInputs.length >= 1) {
                            if (!isImport) {
                                customAlert(""" + f"\"{_i18n('preset_node_err_max_input')}\"" + """);
                            }
                            return null;
                        }
                    }

                    const def = NODE_DEFINITIONS[type];
                    const nodeId = id || `node_${idCounter++}`;
                    
                    let params = JSON.parse(JSON.stringify(def.params || {}));
                    if (loadedParams) params = { ...params, ...loadedParams };

                    const node = { id: nodeId, type, x, y, params, ins: [], outs: [] };
                    nodes[nodeId] = node;

                    if (type === 'separate') {
                        const modelKeys = Object.keys(modelsData);
                        if (modelKeys.length > 0) {
                            // Если модель не задана или её нет в списке доступных, берем первую по умолчанию
                            if (!node.params.model_name || !modelsData[node.params.model_name]) {
                                // Берем ключ (имя) первой модели в объекте
                                const firstModelName = modelKeys[0];
                                // Устанавливаем её по умолчанию
                                node.params.model_name = firstModelName;
                            }
                            // Сразу пересчитываем выходные порты под эту модель
                            calculatePorts(node.id); 
                        }
                    }

                    renderNode(nodeId);
                    sendStateToParent(); 
                    return nodeId;
                }

                function renderNode(nodeId) {
                    const node = nodes[nodeId];
                    const def = NODE_DEFINITIONS[node.type];
                    let el = document.getElementById(nodeId);
                    
                    if (!el) {
                        el = document.createElement('div'); el.className = 'node'; el.id = nodeId;
                        el.style.left = node.x + 'px'; el.style.top = node.y + 'px';
                        nodesContainer.appendChild(el);

                        let header = document.createElement('div'); 
                        header.className = 'node-header'; 

                        let titleText = document.createElement('span');
                        titleText.className = 'node-title-text';
                        titleText.innerText = def.title;
                        header.appendChild(titleText);

                        // Контейнер для целевого инструмента
                        let subtitle = document.createElement('span');
                        subtitle.className = 'node-subtitle';
                        header.appendChild(subtitle);

                        header.addEventListener('mousedown', e => startNodeDrag(e, nodeId));
                        header.addEventListener('touchstart', e => startNodeDrag(e, nodeId, true), {passive: false});

                        let closeBtn = document.createElement('span'); closeBtn.className = 'node-close'; closeBtn.innerHTML = '&times;';
                        closeBtn.addEventListener('mousedown', e => { e.stopPropagation(); deleteNode(nodeId); });
                        closeBtn.addEventListener('touchstart', e => { e.stopPropagation(); e.preventDefault(); deleteNode(nodeId); }, {passive: false});

                        header.appendChild(closeBtn);

                        let content = document.createElement('div'); content.className = 'node-content';
                        if (def.html) content.innerHTML = def.html;

                        el.appendChild(header); el.appendChild(content);

                        if (node.type === 'separate') {
                            const optionsContainer = content.querySelector('.custom-model-options');
                            const modelKeys = Object.keys(modelsData);

                            // Если модель почему-то не выбрана в параметрах — берем первую по умолчанию
                            if (!node.params.model_name && modelKeys.length > 0) {
                                node.params.model_name = modelKeys[0];
                                calculatePorts(nodeId);
                            }

                            // Заполняем кастомный выпадающий список доступными моделями
                            if (optionsContainer) {
                                optionsContainer.innerHTML = '';
                                modelKeys.forEach(key => {
                                    let opt = document.createElement('div');
                                    opt.className = 'custom-option';
                                    opt.setAttribute('data-value', key);
                                    // Используем читаемое имя модели, если оно есть, иначе ключ
                                    opt.innerText = modelsData[key].name || key;
                                    optionsContainer.appendChild(opt);
                                });
                            }
                        }

                        content.querySelectorAll('.custom-select-container').forEach(container => {
                            const input = container.querySelector('.custom-select-input');
                            const param = input.dataset.param;
                            
                            const isFilterable = container.classList.contains('filterable');
                            
                            if (!isFilterable) {
                                input.setAttribute('readonly', 'readonly');
                                input.removeAttribute('oninput');
                            }
                            
                            container.querySelectorAll('.custom-option').forEach(opt => {
                                opt.onclick = function(e) {
                                    e.stopPropagation();
                                    selectOption(this);
                                    const val = this.getAttribute('data-value');
                                    node.params[param] = val;
                                    if (param === 'var' || param === 'type' || param === 'model_name') {
                                        calculatePorts(nodeId);
                                        renderNode(nodeId);
                                    }
                                    sendStateToParent();
                                };
                            });
                        });

                        content.querySelectorAll('[data-param]').forEach(inp => {
                            if (inp.closest('.custom-select-container')) return;
                            
                            if (node.params[inp.dataset.param] !== undefined) {
                                if (inp.type === 'checkbox') inp.checked = node.params[inp.dataset.param];
                                else inp.value = node.params[inp.dataset.param];
                            }
                            inp.addEventListener('input', e => {
                                node.params[inp.dataset.param] = inp.type === 'checkbox' ? inp.checked : (inp.type === 'number' ? parseFloat(inp.value) : inp.value);
                                sendStateToParent();
                            });
                        });
                    }

                    // === ОБНОВЛЕНИЕ ЗАГОЛОВКА ДЛЯ SEPARATE ===
                    let currentHeader = el.querySelector('.node-header');
                    if (currentHeader) {
                        let currentSubtitle = currentHeader.querySelector('.node-subtitle');
                        if (currentSubtitle) {
                            if (node.type === 'separate' && node.params.model_name) {
                                currentSubtitle.style.display = 'inline-block';
                                currentSubtitle.innerText = node.params.model_name;
                            } else {
                                currentSubtitle.style.display = 'none';
                            }
                        }
                    }

                    el.querySelectorAll('.custom-select-container').forEach(container => {
                        const input = container.querySelector('.custom-select-input');
                        const param = input.dataset.param;
                        const value = node.params[param];
                        
                        if (value !== undefined) {
                            container.setAttribute('data-value', value);
                            const selectedOpt = container.querySelector(`.custom-option[data-value="${value}"]`);
                            if (selectedOpt) {
                                input.value = selectedOpt.innerText;
                                container.querySelectorAll('.custom-option').forEach(opt => opt.classList.remove('selected'));
                                selectedOpt.classList.add('selected');
                            }
                        }
                    });

                    calculatePorts(nodeId);
                    el.querySelectorAll('.port-label-container').forEach(e => e.remove());
                    
                    const maxPorts = Math.max(node.ins.length, node.outs.length);
                    for(let i=0; i<maxPorts; i++) {
                        let pCont = document.createElement('div'); pCont.className = 'port-label-container';
                        
                        if (i < node.ins.length) {
                            let pIn = document.createElement('div'); pIn.className = 'port port-in';
                            pIn.dataset.node = nodeId; pIn.dataset.port = i; pIn.dataset.type = 'in';
                            setupPortEvents(pIn);
                            
                            let lIn = document.createElement('div'); lIn.className = 'port-label-in';
                            let portName = node.ins[i];
                            if (portName !== 'audio' && portName !== '0') {
                                let span = document.createElement('span'); span.className = 'port-label-text'; span.innerText = portName;
                                lIn.appendChild(span);
                            }
                            pCont.appendChild(pIn); pCont.appendChild(lIn);
                        } else { pCont.appendChild(document.createElement('div')); }

                        if (i < node.outs.length) {
                            let pOut = document.createElement('div'); pOut.className = 'port port-out';
                            pOut.dataset.node = nodeId; pOut.dataset.port = i; pOut.dataset.type = 'out';
                            setupPortEvents(pOut);
                            
                            let lOut = document.createElement('div'); lOut.className = 'port-label-out';
                            let portName = node.outs[i];
                            if (portName !== 'audio' && portName !== '0') {
                                let span = document.createElement('span'); span.className = 'port-label-text'; span.innerText = portName;
                                
                                // === ВЫДЕЛЕНИЕ ЦЕЛЕВОГО ИНСТРУМЕНТА ===
                                if (node.type === 'separate') {
                                    const modelName = node.params.model_name;
                                    // Получаем данные модели или берем первую по умолчанию
                                    const model = modelsData[modelName] || Object.values(modelsData)[0];
                                    
                                    // Проверяем наличие target_instrument и сравниваем без учета регистра
                                    if (model && model.target_instrument && portName.toLowerCase() === model.target_instrument.toLowerCase()) {
                                        span.classList.add('target-stem');
                                        span.title = 'Целевой инструмент модели (' + model.target_instrument + ')';
                                    }
                                }
                                
                                lOut.appendChild(span);
                            }
                            pCont.appendChild(lOut); pCont.appendChild(pOut);
                        }
                        el.appendChild(pCont);
                    }
                    drawLinks();
                }

                function calculatePorts(nodeId) {
                    const node = nodes[nodeId]; const p = node.params;
                    node.ins = []; node.outs = [];
                    
                    if (node.type === 'input_file') { node.outs = ["audio"]; } 
                    else if (node.type === 'gain') { node.ins = ["audio"]; node.outs = ["audio"]; }
                    else if (node.type === 'normalize') { node.ins = ["audio"]; node.outs = ["audio"]; } 
                    else if (node.type === 'trim' || node.type === 'phase_shift' || node.type === 'filter' || node.type === 'stereo_to_mono') { node.ins = ["audio"]; node.outs = ["audio"]; }
                    else if (node.type === 'phase_correct') { node.ins = ["target", "source"]; node.outs = ["audio"]; }
                    else if (node.type === 'mix' || node.type === 'ensemble') {
                        let num = Math.max(1, p.num_inputs || 1);
                        for(let i=0; i<num; i++) node.ins.push("audio"); node.outs = ["audio"];
                    } 
                    else if (node.type === 'split_stereo') {
                        node.ins = ["audio"];
                        if (p.var === 'sim/dif') node.outs = ["sim", "dif"];
                        else if (p.var === 'mid/side') node.outs = ["mid", "side"];
                        else node.outs = ["left", "right"];
                    } 
                    else if (node.type === 'join_stereo') {
                        if (p.var === 'sim/dif') node.ins = ["sim", "dif"];
                        else if (p.var === 'mid/side') node.ins = ["mid", "side"];
                        else node.ins = ["left", "right"];
                        node.outs = ["0"]; 
                    } 
                    else if (node.type === 'subtract') { node.ins = ["orig", "stem"]; node.outs = ["subtracted"]; } 
                    else if (node.type === 'invert') { node.ins = ["audio"]; node.outs = ["inverted"]; } 
                    else if (node.type === 'separate') {
                        node.ins = ["audio"];
                        const model = modelsData[p.model_name] || Object.values(modelsData)[0];
                        if(model && model.stems) node.outs = model.stems;
                    } 
                    else if (node.type === 'output_file') { node.ins = ["audio"]; node.outs = []; }
                    
                    links = links.filter(l => {
                        let fromNode = nodes[l.fromNode]; let toNode = nodes[l.toNode];
                        if(!fromNode || !toNode) return false;
                        return l.fromPort < fromNode.outs.length && l.toPort < toNode.ins.length;
                    });
                }

                window.updateDynamicPorts = function(el) {
                    const nodeId = el.closest('.node').id;
                    const node = nodes[nodeId];
                    
                    if (el.classList.contains('custom-select-input')) {
                        const container = el.closest('.custom-select-container');
                        const selectedOpt = container.querySelector('.custom-option.selected');
                        if (selectedOpt) {
                            node.params[el.dataset.param] = selectedOpt.getAttribute('data-value');
                        }
                    } else {
                        node.params[el.dataset.param] = el.type === 'number' ? parseFloat(el.value) : el.value;
                    }
                    
                    calculatePorts(nodeId);
                    renderNode(nodeId);
                    sendStateToParent();
                }

                function setupPortEvents(portEl) {
                    portEl.ondragstart = () => false;
                    portEl.addEventListener('mousedown', e => { 
                        if (e.button !== 0) return;
                        e.stopPropagation(); startLink(portEl); 
                    });
                    portEl.addEventListener('touchstart', e => { 
                        e.stopPropagation(); startLink(portEl); 
                    }, {passive: false});
                    portEl.addEventListener('contextmenu', e => {
                        e.preventDefault(); e.stopPropagation();
                        const n = portEl.dataset.node, p = parseInt(portEl.dataset.port), t = portEl.dataset.type;
                        links = links.filter(l => !( (t==='in' && l.toNode===n && l.toPort===p) || (t==='out' && l.fromNode===n && l.fromPort===p) ));
                        drawLinks();
                        sendStateToParent(); 
                    });
                }

                function startLink(portEl) {
                    if (connectingPort) return;
                    connectingPort = { node: portEl.dataset.node, port: parseInt(portEl.dataset.port), type: portEl.dataset.type };
                    tempLink = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                    tempLink.setAttribute('class', 'link-path'); tempLink.style.stroke = '#888'; tempLink.style.strokeDasharray = "5,5";
                    svgCanvas.appendChild(tempLink);
                }

                // Проверка на зацикливание графа (DFS)
                function createsCycle(fromId, toId) {
                    if (fromId === toId) return true;
                    let visited = new Set();
                    let stack = [toId];
                    while(stack.length > 0) {
                        let curr = stack.pop();
                        if (curr === fromId) return true;
                        if (!visited.has(curr)) {
                            visited.add(curr);
                            links.forEach(l => {
                                if (l.fromNode === curr && !visited.has(l.toNode)) {
                                    stack.push(l.toNode);
                                }
                            });
                        }
                    }
                    return false;
                }

                function finishLink(portEl) {
                    if (!connectingPort) return;
                    const t1 = connectingPort.type, t2 = portEl.dataset.type;
                    if (t1 !== t2 && connectingPort.node !== portEl.dataset.node) {
                        const from = t1 === 'out' ? connectingPort : { node: portEl.dataset.node, port: parseInt(portEl.dataset.port) };
                        const to = t1 === 'in' ? connectingPort : { node: portEl.dataset.node, port: parseInt(portEl.dataset.port) };
                        
                        // Проверка 1: Одно соединение в один входной порт
                        const existingLink = links.find(l => l.toNode === to.node && l.toPort === to.port);
                        if (existingLink) {
                            if (existingLink.fromNode !== from.node || existingLink.fromPort !== from.port) {
                                customAlert(""" + f"\"{_i18n('preset_node_err_multi_input')}\"" + """);
                            }
                            cleanupTempLink();
                            return;
                        }

                        // Проверка 2: Рекурсивные соединения запрещены
                        if (createsCycle(from.node, to.node)) {
                            customAlert(""" + f"\"{_i18n('preset_node_err_recursive')}\"" + """);
                            cleanupTempLink();
                            return;
                        }

                        links.push({ fromNode: from.node, fromPort: from.port, toNode: to.node, toPort: to.port });
                    }
                    cleanupTempLink();
                    drawLinks();
                    sendStateToParent(); 
                }

                function cleanupTempLink() { if (tempLink) tempLink.remove(); tempLink = null; connectingPort = null; }
                
                function getLocalMousePos(clientX, clientY) {
                    const rect = workspace.getBoundingClientRect();
                    return { x: (clientX - rect.left - transform.x) / transform.scale, y: (clientY - rect.top - transform.y) / transform.scale };
                }

                function getPortPos(nodeId, portIdx, type) {
                    const nodeEl = document.getElementById(nodeId); if (!nodeEl) return {x:0, y:0};
                    const ports = nodeEl.querySelectorAll(`.port-${type}`); if(!ports[portIdx]) return {x:0, y:0};
                    const portEl = ports[portIdx]; const rect = portEl.getBoundingClientRect();
                    const workspaceRect = workspace.getBoundingClientRect();
                    return {
                        x: (rect.left + rect.width / 2 - workspaceRect.left - transform.x) / transform.scale,
                        y: (rect.top + rect.height / 2 - workspaceRect.top - transform.y) / transform.scale
                    };
                }

                let lastStatuses = {};
                let sseStarted = false;

                function lockEditor() {
                    document.body.classList.add('editor-locked');
                    
                    // Очищаем ранее полученный прогресс (статусы нод) при старте выполнения
                    document.querySelectorAll('.node').forEach(el => {
                        el.classList.remove('status-active', 'status-success', 'status-error');
                    });
                    
                    // Сбрасываем кэш статусов, чтобы новые статусы корректно применились
                    lastStatuses = {};
                }

                function unlockEditor() {
                    document.body.classList.remove('editor-locked');
                }
                // --- Безопасное чтение session_hash (никаких внешних функций) ---
                function readSessionHashFromUrl() {
                    try {
                        const p = new URLSearchParams(window.location.search);
                        const h = p.get('session_hash') || p.get('session_id');
                        if (h && h !== 'undefined' && h !== 'null' && h !== '') return h;
                    } catch (e) {}
                    return '';
                }
                function readSessionHashFromParent() {
                    try { if (window.parent && window.parent.__gradio_session_hash) return window.parent.__gradio_session_hash; } catch (e) {}
                    try { if (window.parent && window.parent.gradio_config && window.parent.gradio_config.session_hash) return window.parent.gradio_config.session_hash; } catch (e) {}
                    return '';
                }

                function startSSE(hash) {
                    if (sseStarted) return;
                    if (!hash || hash === 'undefined' || hash === 'null') return;
                    sseStarted = true;
                    console.log('[preset-editor] SSE start, session_hash =', hash);
                    const eventSource = new EventSource('/preset_status?session_hash=' + encodeURIComponent(hash));
                    eventSource.onmessage = function (event) {
                        let statuses = {};
                        try { statuses = JSON.parse(event.data); } catch (e) { return; }
                        
                        // Обработка глобальной блокировки
                        if (statuses.hasOwnProperty('_locked')) {
                            if (statuses._locked) {
                                lockEditor();
                            } else {
                                unlockEditor();
                            }
                        }
                        
                        if (Object.keys(statuses).length === 0 && Object.keys(lastStatuses).length > 0) {
                            document.querySelectorAll('.node').forEach(el => {
                                el.classList.remove('status-active', 'status-success', 'status-error');
                            });
                        }
                        for (const [nodeId, status] of Object.entries(statuses)) {
                            if (nodeId === '_locked') continue; // Пропускаем служебный ключ
                            if (lastStatuses[nodeId] !== status) updateNodeStatus(nodeId, status);
                        }
                        lastStatuses = statuses;
                    };
                    eventSource.onerror = function (err) { console.error("Ошибка SSE соединения:", err); };
                }

                // (1) Приём hash от родителя — регистрируем ПЕРВЫМ делом, до любых setTimeout
                window.addEventListener('message', function (e) {
                    if (e.data && e.data.type === 'gradio_session_hash' && e.data.hash) {
                        startSSE(e.data.hash);
                    }
                });

                // (2) Bootstrap: URL -> родитель(синхронно) -> активный запрос у родителя
                (function bootstrapSSE() {
                    const h = readSessionHashFromUrl() || readSessionHashFromParent();
                    if (h) startSSE(h);
                    function askParent() {
                        if (sseStarted) return;
                        // на всякий случай ещё раз пробуем прочитать синхронно
                        const hh = readSessionHashFromParent();
                        if (hh) { startSSE(hh); return; }
                        try { window.parent.postMessage({ type: 'request_session_hash' }, '*'); } catch (e) {}
                    }
                    askParent();
                    setTimeout(askParent, 1500);
                    setTimeout(askParent, 4000);
                })();

                function drawLinks() {
                    svgCanvas.innerHTML = '';
                    links.forEach(l => {
                        const p1 = getPortPos(l.fromNode, l.fromPort, 'out'); const p2 = getPortPos(l.toNode, l.toPort, 'in');
                        if ((p1.x === 0 && p1.y === 0) || (p2.x === 0 && p2.y === 0)) return;
                        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                        path.setAttribute('class', 'link-path'); path.setAttribute('d', `M ${p1.x} ${p1.y} C ${p1.x + 80} ${p1.y}, ${p2.x - 80} ${p2.y}, ${p2.x} ${p2.y}`);
                        path.addEventListener('mousedown', e => {
                            if (e.button === 0 || e.button === 2) {
                                e.preventDefault(); e.stopPropagation();
                                links = links.filter(link => link !== l);
                                drawLinks(); sendStateToParent(); 
                            }
                        });
                        path.addEventListener('contextmenu', e => { e.preventDefault(); e.stopPropagation(); });
                        svgCanvas.appendChild(path);
                    });
                }

                function startNodeDrag(e, nodeId, isTouch = false) {
                    e.stopPropagation(); draggedNode = nodeId;
                    
                    document.querySelectorAll('.node').forEach(n => n.style.zIndex = 1);
                    document.getElementById(nodeId).style.zIndex = 100;

                    const clientX = isTouch ? e.touches[0].clientX : e.clientX;
                    const clientY = isTouch ? e.touches[0].clientY : e.clientY;
                    const localPos = getLocalMousePos(clientX, clientY); const n = nodes[nodeId];
                    dragOffset = { x: localPos.x - n.x, y: localPos.y - n.y };
                }

                function applyTransform() {
                    transformLayer.style.transform = `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`;
                    drawLinks();
                }

                function handleDragAndPan(clientX, clientY, movementX, movementY, isTouch = false) {
                    if (isPanning) {
                        const mx = movementX !== undefined ? movementX : (clientX - panStart.x);
                        const my = movementY !== undefined ? movementY : (clientY - panStart.y);
                        transform.x += mx; transform.y += my;
                        if (isTouch) { panStart = { x: clientX, y: clientY }; }
                        applyTransform();
                    } else if (draggedNode) {
                        const localPos = getLocalMousePos(clientX, clientY);
                        const n = nodes[draggedNode];
                        n.x = localPos.x - dragOffset.x; n.y = localPos.y - dragOffset.y;
                        const el = document.getElementById(draggedNode);
                        el.style.left = n.x + 'px'; el.style.top = n.y + 'px';
                        drawLinks();
                    } else if (tempLink && connectingPort) {
                        const localPos = getLocalMousePos(clientX, clientY);
                        const mx = localPos.x; const my = localPos.y;
                        const p = getPortPos(connectingPort.node, connectingPort.port, connectingPort.type);
                        if (connectingPort.type === 'out') {
                            tempLink.setAttribute('d', `M ${p.x} ${p.y} C ${p.x + 80} ${p.y}, ${mx - 80} ${my}, ${mx} ${my}`);
                        } else { tempLink.setAttribute('d', `M ${mx} ${my} C ${mx + 80} ${my}, ${p.x - 80} ${p.y}, ${p.x} ${p.y}`); }
                    }
                }

                function setupEvents() {
                    workspace.addEventListener('contextmenu', e => e.preventDefault());
                    window.addEventListener('mousemove', e => { handleDragAndPan(e.clientX, e.clientY, e.movementX, e.movementY); });
                    window.addEventListener('touchmove', e => {
                        if (isPinching) { if (e.cancelable) e.preventDefault(); return; }
                        if (isPanning || draggedNode || tempLink || touchDragElement) { e.preventDefault(); } 
                        const touch = e.touches[0];
                        if (touchDragElement) {
                            touchDragElement.style.left = (touch.clientX - touchDragElement.offsetWidth / 2) + 'px';
                            touchDragElement.style.top = (touch.clientY - touchDragElement.offsetHeight / 2) + 'px';
                        } else { handleDragAndPan(touch.clientX, touch.clientY, undefined, undefined, true); }
                    }, {passive: false});

                    const stopDrag = (e) => {
                        if (isPinching) return;
                        if (touchDragElement) {
                            if (e && e.changedTouches) {
                                const touch = e.changedTouches[0];
                                const rect = workspace.getBoundingClientRect();
                                if (touch.clientX >= rect.left && touch.clientX <= rect.right && touch.clientY >= rect.top && touch.clientY <= rect.bottom) {
                                    const localPos = getLocalMousePos(touch.clientX, touch.clientY);
                                    addNode(touchDragType, localPos.x, localPos.y);
                                }
                            }
                            touchDragElement.remove(); touchDragElement = null; touchDragType = null;
                        }
                        if (connectingPort) {
                            let targetPort = null;
                            if (e && e.changedTouches) {
                                const touch = e.changedTouches[0];
                                const targetEl = document.elementFromPoint(touch.clientX, touch.clientY);
                                if (targetEl) targetPort = targetEl.closest('.port');
                            } else if (e && e.target) { targetPort = e.target.closest('.port'); }
                            if (targetPort) { finishLink(targetPort); } 
                            else { cleanupTempLink(); drawLinks(); }
                        }
                        
                        if(draggedNode) {
                            sendStateToParent();
                        }

                        isPanning = false; draggedNode = null;
                    };

                    window.addEventListener('mouseup', stopDrag);
                    window.addEventListener('touchend', stopDrag);

                    workspace.addEventListener('mousedown', e => {
                        if (e.target === workspace || e.target === transformLayer || e.target.tagName === 'svg') {
                            isPanning = true; panStart = { x: e.clientX, y: e.clientY };
                        }
                    });
                    workspace.addEventListener('touchstart', e => {
                        if (e.touches.length !== 1 || isPinching) return;
                        if (e.target === workspace || e.target === transformLayer || e.target.tagName === 'svg') {
                            isPanning = true; panStart = { x: e.touches[0].clientX, y: e.touches[0].clientY };
                        }
                    }, {passive: true});
                    workspace.addEventListener('wheel', e => {
                        if (e.target.closest('.custom-select-options')) {
                            return;
                        }
                        e.preventDefault(); const zoomIntensity = 0.1;
                        const wheel = e.deltaY < 0 ? 1 : -1; const zoom = Math.exp(wheel * zoomIntensity);
                        const rect = workspace.getBoundingClientRect(); const mouseX = e.clientX - rect.left; const mouseY = e.clientY - rect.top;
                        transform.x = mouseX - (mouseX - transform.x) * zoom; transform.y = mouseY - (mouseY - transform.y) * zoom;
                        transform.scale *= zoom; applyTransform();
                        showZoomBadge();
                    });
                }

                function setupDragDrop() {
                    workspace.addEventListener('dragover', e => e.preventDefault());
                    workspace.addEventListener('drop', e => {
                        e.preventDefault();
                        const type = e.dataTransfer.getData('type');
                        if (type && NODE_DEFINITIONS[type]) {
                            const localPos = getLocalMousePos(e.clientX, e.clientY);
                            addNode(type, localPos.x, localPos.y);
                        }
                    });
                }

                window.exportJSON = function() {
                    const name = document.getElementById('preset-name').value || "preset";
                    const data = { nodes, links, name };
                    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url; a.download = `${name}.json`; a.click();
                    URL.revokeObjectURL(url);
                }

                window.clearWorkspace = function() {
                    nodes = {}; links = [];
                    document.getElementById('preset-name').value = ''
                    nodesContainer.innerHTML = ''; svgCanvas.innerHTML = '';
                    idCounter = 1;
                    sendStateToParent(); 
                }


                function loadJSON(data) {
                    clearWorkspace();
                    if (data.name) document.getElementById('preset-name').value = data.name;
                    let maxId = 0;
                    let hasInputFile = false; // Трекинг наличия input_file
                    
                    for (let id in data.nodes) {
                        const n = data.nodes[id];
                        if (n.type === 'input_file') {
                            if (hasInputFile) continue; // Игнорируем дублирующие входные ноды
                            hasInputFile = true;
                        }
                        addNode(n.type, n.x, n.y, id, n.params, true); // true = загрузка импорта
                        const idNum = parseInt(id.replace('node_', '')); 
                        if (idNum > maxId) maxId = idNum;
                    }
                    idCounter = maxId + 1;
                    
                    // Загружаем только связи, ведущие на существующие ноды (чтобы удалить висячие хвосты от удаленных input_file)
                    links = (data.links || []).filter(l => nodes[l.fromNode] && nodes[l.toNode]); 
                    drawLinks();
                    sendStateToParent(); 
                }

                function setupImport() {
                    const impBtn = document.getElementById('import-btn'); const impFile = document.getElementById('import-file');
                    impBtn.addEventListener('click', () => impFile.click());
                    impFile.addEventListener('change', e => handleFile(e.target.files[0]));
                    impBtn.addEventListener('dragover', e => { e.preventDefault(); impBtn.classList.add('dragover'); });
                    impBtn.addEventListener('dragleave', e => impBtn.classList.remove('dragover'));
                    impBtn.addEventListener('drop', e => {
                        e.preventDefault(); impBtn.classList.remove('dragover');
                        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
                    });
                }

                function handleFile(file) {
                    if (!file) return;
                    const reader = new FileReader();
                    reader.onload = async (e) => {
                        try {
                            const data = JSON.parse(e.target.result); loadJSON(data);
                        } catch(err) { await customAlert(""" + f"\"{_i18n('preset_node_invalid_json')}\"" + """); }
                    };
                    reader.readAsText(file);
                }

                // Добавляем обработчик для мультитач зума
                // ==================== МУЛЬТИТАЧ ЗУМ (pinch = зум + пан одновременно) ====================
                // Предсказуемость:
                //  1. Точка мира между пальцами "приклеена" к пальцам: разведение — зум вокруг неё,
                //     движение обоих пальцев — пан, работает одновременно (как в картах/Figma).
                //  2. Формула абсолютная — от базовой линии жеста, без накопления ошибки и дрейфа.
                //  3. Отпустил один палец — жест бесшовно продолжается как пан оставшимся.
                //  4. Второй палец во время переноса ноды/создания связи — жесты безопасно отменяются.
                const PINCH_MIN_SCALE = 0.1;
                const PINCH_MAX_SCALE = 8;
                let isPinching = false;
                let pinchBaseline = null;   // { dist, scale, anchorX, anchorY }
                let pinchLastTouch = null;  // опорная точка для fallback-пана одним пальцем
                let zoomBadgeTimer = null;

                function showZoomBadge() {
                    const badge = document.getElementById('pinch-zoom-badge');
                    if (!badge) return;
                    badge.textContent = Math.round(transform.scale * 100) + '%';
                    badge.classList.add('show');
                    clearTimeout(zoomBadgeTimer);
                    zoomBadgeTimer = setTimeout(() => badge.classList.remove('show'), 700);
                }
                function pinchDist(e) {
                    const dx = e.touches[0].clientX - e.touches[1].clientX;
                    const dy = e.touches[0].clientY - e.touches[1].clientY;
                    return Math.hypot(dx, dy);
                }
                function pinchMid(e) {
                    const rect = workspace.getBoundingClientRect();
                    return {
                        x: (e.touches[0].clientX + e.touches[1].clientX) / 2 - rect.left,
                        y: (e.touches[0].clientY + e.touches[1].clientY) / 2 - rect.top
                    };
                }
                function startPinch(e) {
                    isPinching = true;
                    isPanning = false;
                    draggedNode = null;
                    if (connectingPort) cleanupTempLink();
                    if (touchDragElement) { touchDragElement.remove(); touchDragElement = null; touchDragType = null; }
                    const mid = pinchMid(e);
                    pinchBaseline = {
                        dist: pinchDist(e),
                        scale: transform.scale,
                        // Точка мира под центром жеста — она всё время будет оставаться под пальцами
                        anchorX: (mid.x - transform.x) / transform.scale,
                        anchorY: (mid.y - transform.y) / transform.scale
                    };
                    pinchLastTouch = null;
                }
                function applyPinch(e) {
                    const base = pinchBaseline;
                    if (!base || base.dist <= 0) return;
                    const mid = pinchMid(e);
                    const dist = pinchDist(e);
                    const newScale = Math.max(PINCH_MIN_SCALE, Math.min(PINCH_MAX_SCALE, base.scale * (dist / base.dist)));
                    transform.scale = newScale;
                    transform.x = mid.x - base.anchorX * newScale;
                    transform.y = mid.y - base.anchorY * newScale;
                    applyTransform();
                    showZoomBadge();
                }
                function endPinch(e) {
                    if (!isPinching) return;
                    if (e.touches.length >= 1) {
                        // Жест продолжается оставшимся пальцем — переключаемся на пан
                        const t = e.touches[0];
                        pinchLastTouch = { x: t.clientX, y: t.clientY };
                        pinchBaseline = null;
                    } else {
                        isPinching = false;
                        pinchBaseline = null;
                        pinchLastTouch = null;
                    }
                }
                workspace.addEventListener('touchstart', function(e) {
                    if (e.touches.length === 2 && !e.target.closest('.port') && !e.target.closest('.node-header')) {
                        if (e.cancelable) e.preventDefault();
                        startPinch(e);
                    }
                }, {passive: false});
                workspace.addEventListener('touchmove', function(e) {
                    if (!isPinching) return;
                    if (e.touches.length >= 2) {
                        if (e.cancelable) e.preventDefault();
                        pinchLastTouch = null;
                        applyPinch(e);
                    } else if (e.touches.length === 1) {
                        // Один палец остался после щипка — пан без рывка
                        if (e.cancelable) e.preventDefault();
                        const t = e.touches[0];
                        if (pinchLastTouch) {
                            transform.x += t.clientX - pinchLastTouch.x;
                            transform.y += t.clientY - pinchLastTouch.y;
                            applyTransform();
                        }
                        pinchLastTouch = { x: t.clientX, y: t.clientY };
                    }
                }, {passive: false});
                workspace.addEventListener('touchend', endPinch);
                workspace.addEventListener('touchcancel', endPinch);

                init();
            </script>
        </body>
        </html>
        """





















#########################################












        AUTO_ENSEMBLE_PRESET_HTML_CONTENT = """<!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
            <title>""" + f"{_i18n('auto_ensemble_preset_editor_title')}" + """</title>
            <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap" rel="stylesheet">
            <style>



                /* -- AUTO ENSMEBLE --- */





                :root {
                    --bg-color: var(--background-fill-primary, #eef1f5);
                    --grid-color: var(--border-color-secondary, #d4d8dd);
                    --node-bg: var(--background-fill-secondary, #f6f8fa);
                    --node-header: var(--block-background-fill, #e6ebf1);
                    --border-color: var(--border-color-primary, #d0d7de);
                    --text-main: var(--body-text-color, #333333);
                    --btn-blue: var(--color-accent, #007bff);
                    --primary: var(--color-accent, #007bff);
                    --primary-hover: var(--color-accent-soft, #0056b3);
                }

                @media (prefers-color-scheme: dark) {
                    :root {
                        --bg-color: var(--background-fill-primary, #0b0f19);
                        --grid-color: var(--border-color-secondary, #1f2937);
                        --node-bg: var(--background-fill-secondary, #1f2937);
                        --node-header: var(--block-background-fill, #374151);
                        --border-color: var(--border-color-primary, #374151);
                        --text-main: var(--body-text-color, #f3f4f6);
                    }
                }

                .dark {
                    --bg-color: var(--background-fill-primary, #0b0f19);
                    --grid-color: var(--border-color-secondary, #1f2937);
                    --node-bg: var(--background-fill-secondary, #1f2937);
                    --node-header: var(--block-background-fill, #374151);
                    --border-color: var(--border-color-primary, #374151);
                    --text-main: var(--body-text-color, #f3f4f6);
                }

                * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Montserrat', sans-serif; }
                body { width: 100vw; height: 100vh; overflow: hidden; display: flex; flex-direction: column; background: var(--bg-color); color: var(--text-main); }
                
                .topbar { height: 60px; background: var(--node-bg); border-bottom: 1px solid var(--border-color); display: flex; align-items: center; padding: 0 20px; gap: 10px; z-index: 100; box-shadow: 0 2px 5px rgba(0,0,0,0.02); flex-shrink: 0; }
                input[type="text"], input[type="number"] { padding: 7px 12px; border: 1px solid var(--border-color); border-radius: 4px; font-size: 14px; outline: none; background: var(--bg-color); color: var(--text-main); }
                .btn { padding: 8px 16px; border-radius: 4px; border: 1px solid var(--border-color); background: var(--node-bg); color: var(--text-main); cursor: pointer; font-weight: 500; font-size: 14px; transition: 0.2s; white-space: nowrap; }
                .btn:hover { background: var(--grid-color); }
                .btn-blue { color: var(--btn-blue); border-color: var(--btn-blue); }
                .btn-green { color: #28a745; border-color: #28a745; }
                .btn-red { color: #dc3545; border-color: #dc3545; }

                /* Модальные окна */
                .modal-overlay {
                    position: fixed;
                    top: 0; left: 0;
                    width: 100vw; height: 100vh;
                    background: rgba(0,0,0,0.4);
                    display: none;
                    align-items: center;
                    justify-content: center;
                    z-index: 99999999;
                    opacity: 0;
                    transition: opacity 0.2s;
                    padding: 16px;          /* ← отступ от краёв экрана */
                    box-sizing: border-box;
                }
                .modal-overlay.show { display: flex; opacity: 1; }
                .modal-box {
                    background: var(--node-bg);
                    padding: 25px;
                    border-radius: 8px;
                    width: 100%;            /* ← было min-width:320px */
                    max-width: 420px;       /* ← было max-width:90vw */
                    max-height: 90vh;       /* ← новое: ограничение высоты */
                    overflow-y: auto;       /* ← новое: скролл если контент не влезает */
                    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                    text-align: center;
                    box-sizing: border-box;
                }
                .modal-text { margin-bottom: 25px; font-size: 15px; font-weight: 500; color: var(--text-main); word-wrap: break-word; }
                .modal-buttons { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
                .modal-buttons .btn { min-width: 100px; }

                /* Каскадные селекторы (Gradio-like) */
                .custom-select-container { position: relative; width: 100%; box-sizing: border-box; }
                .custom-select-input-wrapper { position: relative; width: 100%; }
                .custom-select-input { width: 100%; padding-right: 30px; border: 1px solid var(--border-color); border-radius: 4px; font-size: 14px; background: var(--bg-color); color: var(--text-main); cursor: pointer; text-overflow: ellipsis; white-space: nowrap; overflow: hidden; }
                .custom-select-input-wrapper::after { content: '▼'; font-size: 10px; color: var(--text-main); position: absolute; right: 10px; top: 50%; transform: translateY(-50%); pointer-events: none; opacity: 0.7; }
                .custom-select-options { position: absolute; top: 100%; left: 0; width: 100%; background: var(--node-bg); border: 1px solid var(--border-color); border-radius: 4px; margin-top: 4px; max-height: 200px; overflow-y: auto; z-index: 999; display: none; box-shadow: 0 5px 15px rgba(0,0,0,0.15); }
                .custom-select-container.open .custom-select-options { display: block; }
                .custom-option { padding: 9px 12px; cursor: pointer; font-size: 13px; color: var(--text-main); transition: background 0.2s; white-space: pre-wrap; overflow: hidden; overflow-wrap: anywhere;  }
                .custom-option:hover { background: var(--grid-color); color: var(--primary-hover); }
                .custom-option.disabled { color: #aaa; cursor: not-allowed; }
                .custom-option.disabled:hover { background: transparent; color: #aaa; }

                /* Рабочая область (Списочный редактор) */
                .main-container { flex: 1; padding: 20px; overflow-y: auto; background-color: var(--bg-color); }
                .editor-layout { max-width: 1000px; margin: 0 auto; display: flex; flex-direction: column; gap: 15px; }
                
                .ensemble-item { background: var(--node-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 15px; display: flex; align-items: center; justify-content: space-between; gap: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.02); transition: 0.2s; }
                .ensemble-item:hover { border-color: var(--primary); }
                .ensemble-item-number { font-weight: 600; color: var(--text-main); opacity: 0.5; width: 25px; text-align: center; }
                
                .checkbox-container { display: flex; align-items: center; gap: 8px; font-size: 13px; cursor: pointer; user-select: none; }
                .checkbox-container input[type="checkbox"] { cursor: pointer; width: 16px; height: 16px; accent-color: var(--primary); }
                
                .item-controls { display: flex; gap: 10px; }
                .add-btn-wrapper { text-align: center; margin-top: 25px; }
                .empty-state { text-align: center; color: var(--text-main); opacity: 0.6; font-size: 14px; margin-top: 40px; }

                .toggle-container {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                .toggle-switch {
                    position: relative;
                    display: inline-block;
                    width: 34px;
                    height: 20px;
                    flex-shrink: 0;
                }
                .toggle-switch input {
                    opacity: 0;
                    width: 0;
                    height: 0;
                    margin: 0;
                }
                .slider {
                    position: absolute;
                    cursor: pointer;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background-color: var(--border-color);
                    transition: .3s;
                    border-radius: 20px;
                }
                .slider:before {
                    position: absolute;
                    content: "";
                    height: 14px;
                    width: 14px;
                    left: 3px;
                    bottom: 3px;
                    background-color: var(--node-bg);
                    transition: .3s;
                    border-radius: 50%;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                }
                input:checked + .slider {
                    background-color: var(--btn-blue);
                }
                input:checked + .slider:before {
                    transform: translateX(14px);
                }
                .weights-number {
                    max-width: 10ch
                }

                @media (max-width: 1024px) {
                    /* Topbar responsive adjustments */
                    .topbar { flex-wrap: wrap; height: auto; padding: 10px; justify-content: space-between; }
                    .topbar h1 { display: none; }
                    .custom-select-container { width: 100% !important; margin-bottom: 5px; }
                    .topbar input[type="text"] { flex: 1 1 auto; width: 100%; min-width: 150px; margin-bottom: 5px; }
                    .topbar .btn { flex: 1 1 auto; margin: 2px; font-size: 12px; padding: 6px 10px; }
                    #sidebar-toggle { display: none !important; }
                    .main-container { flex-direction: column; }

                    /* Custom select full width on mobile */
                    .custom-select-container { width: 100% !important; margin-bottom: 5px; }

                    .custom-select-input {
                        font-size: 13px;
                        padding: 8px 30px 8px 12px;
                    }

                    .custom-select-options {
                        max-height: 150px;
                    }

                    /* Hide sidebar toggle if exists */
                    #sidebar-toggle {
                        display: none !important;
                    }

                    /* Main container */
                    .main-container {
                        flex-direction: column;
                        padding: 12px;
                    }



                    /* Ensemble items - mobile friendly */
                    .ensemble-item {
                        display: flex;
                        flex-direction: column;
                        gap: 12px;
                        padding: 15px;
                        align-items: stretch;
                        border-radius: 6px;
                        transition: border-color 0.2s;
                    }

                    .ensemble-item-number {
                        display: none; /* Hide numbers on mobile for cleaner look */
                        /* Or keep minimal: width: 20px; font-size: 12px; */
                    }

                    /* Checkbox container full width */
                    .toggle-container {
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }

                    .toggle-switch {
                        position: relative;
                        display: inline-block;
                        width: 34px;
                        height: 20px;
                        flex-shrink: 0;
                        margin-left: auto; /* ← Перемещает только переключатель вправо */
                    }

                    .weights-number {
                        margin-left: auto;
                    }

                    /* Item controls - button group */
                    .item-controls {
                        display: flex;
                        gap: 8px;
                        width: 100%;
                        justify-content: flex-end;
                        flex-wrap: wrap;
                    }

                    .item-controls .btn {
                        flex: 1 1 auto;
                        min-width: 60px;
                        padding: 8px 12px;
                        font-size: 12px;
                        text-align: center;
                    }

                    /* Add button wrapper */
                    .add-btn-wrapper {
                        margin-top: 20px;
                        text-align: center;
                    }

                    .add-btn-wrapper .btn {
                        width: 100%;
                        max-width: 300px;
                        padding: 12px 20px;
                        font-size: 14px;
                    }

                    /* Empty state */
                    .empty-state {
                        font-size: 13px;
                        margin-top: 30px;
                        padding: 20px;
                    }
                }
            



            
            



            </style>
        </head>
        <body>

            <!-- Модальное окно подтверждений/ошибок -->
            <div id="custom-modal-overlay" class="modal-overlay">
                <div class="modal-box">
                    <div class="modal-text" id="modal-text"></div>
                    <div class="modal-buttons">
                        <button class="btn btn-blue" id="modal-btn-ok">""" + f"{_i18n('preset_node_ok')}" + """</button>
                        <button class="btn btn-red" id="modal-btn-cancel" style="display:none;">""" + f"{_i18n('preset_node_cancel')}" + """</button>
                    </div>
                </div>
            </div>

            <!-- Верхняя панель (Topbar) -->
            <div class="topbar">
                <div class="custom-select-container" id="preset-select-container" data-value="" style="width: 220px;" onmouseenter="fetchServerPresets()">
                    <div class="custom-select-input-wrapper">
                        <input type="text" class="custom-select-input" placeholder=""" + f"\"{_i18n('preset_node_saved')}\"" + """ onfocus="openDropdown(this)" onclick="openDropdown(this)" oninput="filterOptions(this)" autocomplete="off">
                    </div>
                    <div class="custom-select-options" id="server-preset-list">
                        <div class="custom-option disabled">""" + f"{_i18n('preset_node_loading')}" + """</div>
                    </div>
                </div>

                <input type="text" id="preset-name" placeholder=""" + f"\"{_i18n('preset_node_preset_name')}\"" + """ value="">
                <button class="btn btn-green" onclick="saveServerPreset()">""" + f"{_i18n('preset_node_save')}" + """</button>
                <button class="btn btn-red" onclick="deleteServerPreset()">""" + f"{_i18n('preset_node_delete')}" + """</button>

                <span style="border-left: 1px solid var(--border-color); height: 30px; margin: 0 5px;" class="desktop-only"></span>

                <button class="btn btn-blue" onclick="exportJSON()">""" + f"{_i18n('preset_node_preset_download')}" + """</button>
                <button class="btn" id="import-btn">""" + f"{_i18n('preset_node_preset_upload')}" + """</button>
                <input type="file" id="import-file" style="display:none" accept=".json">

                <button class="btn btn-red" style="margin-left: auto;" onclick="clearEditor()">""" + f"{_i18n('clear')}" + """</button>
            </div>

            <!-- Рабочая область -->
            <div class="main-container">
                <div class="editor-layout" id="editor-layout">
                    <!-- Элементы ансамбля рендерятся здесь -->
                </div>
                <div class="add-btn-wrapper">
                    <button class="btn btn-blue" onclick="addEnsembleItem()">""" + f"{_i18n('add_model')}" + """</button>
                </div>
            </div>

            <script>
                let modelsData = {};
                let ensembleState = [];
                let idCounter = 1;

                // Инициализация
                const urlParams = new URLSearchParams(window.location.search);
                if (urlParams.get('__theme') === 'dark') document.documentElement.classList.add('dark');

                window.addEventListener('message', function (event) {
                    let themeData = typeof event.data === 'object' ? (event.data.theme || event.data.type) : event.data;
                    if (themeData === 'theme_dark' || themeData === 'dark') document.documentElement.classList.add('dark');
                    else if (themeData === 'theme_light' || themeData === 'light') document.documentElement.classList.remove('dark');
                });

                async function init() {
                    try {""" + """                      modelsData = """ + json.dumps(self.separator.info, ensure_ascii=False, indent=0) + ";" + """
                    } catch (e) {
                        console.error("Failed to load models.json", e);
                        modelsData = { "NO_MODELS": { "stems": ["Vocals", "Instrumental"] } }; 
                    }
                }
                window.onload = init;

                // Прием данных
                window.addEventListener('message', (e) => {
                    if (e.data && e.data.type === 'update_auto_ensemble_preset') {
                        loadJSON(e.data.payload);
                    }
                });

                function exportJSON() {
                    const name = document.getElementById('preset-name').value || "auto_ensemble_preset";
                    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(ensembleState, null, 4));
                    const downloadAnchorNode = document.createElement('a');
                    downloadAnchorNode.setAttribute("href", dataStr);
                    downloadAnchorNode.setAttribute("download", name + ".json");
                    document.body.appendChild(downloadAnchorNode);
                    downloadAnchorNode.click();
                    downloadAnchorNode.remove();
                }

                function setupImport() {
                    const importBtn = document.getElementById('import-btn');
                    const importFile = document.getElementById('import-file');
                    
                    if (importBtn && importFile) {
                        // 1. Стандартный клик по кнопке
                        importBtn.onclick = () => importFile.click();
                        
                        // 2. События Drag-and-Drop для кнопки импорта
                        importBtn.addEventListener('dragover', (e) => {
                            e.preventDefault();
                            importBtn.style.opacity = '0.7';
                            importBtn.style.border = '2px dashed var(--btn-blue)';
                        });
                        
                        importBtn.addEventListener('dragleave', (e) => {
                            e.preventDefault();
                            importBtn.style.opacity = '1';
                            importBtn.style.border = '1px solid var(--border-color)';
                        });
                        
                        importBtn.addEventListener('drop', (e) => {
                            e.preventDefault();
                            importBtn.style.opacity = '1';
                            importBtn.style.border = '1px solid var(--border-color)';
                            
                            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                                processFile(e.dataTransfer.files[0]);
                            }
                        });

                        // 3. Выбор файла через диалоговое окно
                        importFile.onchange = e => {
                            const file = e.target.files[0];
                            if (!file) return;
                            processFile(file);
                            importFile.value = ""; // Сбрасываем value, чтобы можно было загрузить тот же файл снова
                        };
                    }
                }

                // Обработчик чтения файла
                function processFile(file) {
                    const reader = new FileReader();
                    reader.onload = async event => {
                        try {
                            const json = JSON.parse(event.target.result);
                            
                            // Запуск JS-аналога функции validate_flow
                            const validatedFlow = await validateFlowJS(json);
                            
                            loadJSON(validatedFlow);
                            document.getElementById('preset-name').value = json.name || "";
                            sendStateToParent();
                            
                        } catch (err) {
                            await customAlert(""" + f"\"{_i18n('preset_node_invalid_json')}\"" + """ + "\\n" + err.message);
                        }
                    };
                    reader.readAsText(file);
                }

                // Реализация validate_flow на JS с учетом локализации _i18n
                async function validateFlowJS(flow) {
                    let errors = [];
                    let warns = [];
                    let validated_flow = [];
                    
                    if (!flow) {
                        throw new Error(""" + f"\"{_i18n('flow_empty')}\"" + """);
                    }
                    
                    if (!Array.isArray(flow)) {
                        throw new Error(""" + f"\"{_i18n('flow_validation_error', error=_i18n('flow_not_list'))}\"" + """);
                    }

                    for (let i = 0; i < flow.length; i++) {
                        let model_flow = flow[i];
                        let valid = true;
                        let error_parts = [];
                        
                        if (!Array.isArray(model_flow)) {
                            let errMsg = """ + f"\"{_i18n('flow_item_not_list', type='__TYPE__')}\"" + """.replace('__TYPE__', typeof model_flow);
                            errors.push({ index: i, error: errMsg });
                            continue;
                        }
                        
                        if (model_flow.length !== 3 && model_flow.length !== 4) {
                            let errMsg = """ + f"\"{_i18n('flow_invalid_length', length='__LEN__', expected='3 or 4')}\"" + """.replace('__LEN__', model_flow.length);
                            errors.push({ index: i, error: errMsg });
                            continue;
                        }
                        
                        let model_name = model_flow[0];
                        let primary_stem = model_flow[1];
                        let invert = model_flow[2];
                        let weight = model_flow.length === 4 ? model_flow[3] : 1.0;
                        let has_weight = model_flow.length === 4;
                        
                        let model_exists = false;
                        
                        // Валидация model_name
                        if (typeof model_name !== 'string') {
                            let errMsg = """ + f"\"{_i18n('flow_invalid_type', field='model_name', expected='str', got='__GOT__')}\"" + """.replace('__GOT__', typeof model_name);
                            error_parts.push(errMsg);
                            valid = false;
                        } else {
                            model_exists = modelsData.hasOwnProperty(model_name);
                            if (!model_exists && !modelsData["NO_MODELS"]) {
                                let errMsg = """ + f"\"{_i18n('flow_model_not_found', model='__MOD__', available='self.get_all_models()')}\"" + """.replace('__MOD__', model_name);
                                warns.push(errMsg);
                            }
                        }
                        
                        // Валидация primary_stem
                        if (typeof primary_stem !== 'string') {
                            let errMsg = """ + f"\"{_i18n('flow_invalid_type', field='primary_stem', expected='str', got='__GOT__')}\"" + """.replace('__GOT__', typeof primary_stem);
                            error_parts.push(errMsg);
                            valid = false;
                        } else if (model_exists && !modelsData["NO_MODELS"]) {
                            let available_stems = modelsData[model_name].stems || [];
                            if (available_stems.length > 0 && !available_stems.includes(primary_stem)) {
                                let errMsg = """ + f"\"{_i18n('flow_stem_not_found', stem='__STEM__', model='__MOD__', available='__AVAIL__')}\"" + """
                                    .replace('__STEM__', primary_stem)
                                    .replace('__MOD__', model_name)
                                    .replace('__AVAIL__', available_stems.join(', '));
                                warns.push(errMsg);
                            }
                        }
                        
                        // Валидация invert
                        if (typeof invert !== 'boolean') {
                            let errMsg = """ + f"\"{_i18n('flow_invalid_type', field='invert', expected='bool', got='__GOT__')}\"" + """.replace('__GOT__', typeof invert);
                            error_parts.push(errMsg);
                            valid = false;
                        }
                        
                        // Валидация weight
                        if (has_weight) {
                            if (typeof weight === 'string') {
                                let parsed = parseFloat(weight);
                                if (isNaN(parsed)) {
                                    error_parts.push(""" + f"\"{_i18n('flow_weight_empty')}\"" + """);
                                    valid = false;
                                    weight = 1.0;
                                } else {
                                    weight = parsed;
                                }
                            } else if (typeof weight !== 'number') {
                                let errMsg = """ + f"\"{_i18n('flow_invalid_type', field='weight', expected='number', got='__GOT__')}\"" + """.replace('__GOT__', typeof weight);
                                error_parts.push(errMsg);
                                valid = false;
                                weight = 1.0;
                            }
                        }
                        
                        if (valid) {
                            validated_flow.push([model_name, primary_stem, invert, weight]);
                        } else {
                            errors.push({ index: i, error: error_parts.join('; ') });
                        }
                    }
                    
                    if (errors.length > 0) {
                        let error_messages = errors.map(e => `#${e.index}: ${e.error}`).join("\\n");
                        let finalErr = """ + f"\"{_i18n('flow_validation_errors', count='__CNT__')}\"" + """.replace('__CNT__', errors.length) + "\\n" + error_messages;
                        throw new Error(finalErr);
                    }
                    
                    if (warns.length > 0) {
                        await customAlert(warns.join("\\n\\n"));
                    }
                    
                    return validated_flow;
                }

                function loadJSON(state) {
                    const data = typeof state === 'string' ? JSON.parse(state) : state;
                    if (Array.isArray(data)) {
                        ensembleState = data.map(item => {
                            if (Array.isArray(item) && item.length >= 4) {
                                return [item[0] || '', item[1] || '', item[2] || false, item[3] !== undefined ? item[3] : 1.0];
                            }
                            return ['', '', false, 1.0];
                        });
                    } else {
                        ensembleState = [];
                    }
                    renderEditor();
                }

                function customAlert(msg) {
                    return new Promise(resolve => {
                        const overlay = document.getElementById('custom-modal-overlay');
                        document.getElementById('modal-text').innerText = msg;
                        const btnOk = document.getElementById('modal-btn-ok');
                        document.getElementById('modal-btn-cancel').style.display = 'none';
                        btnOk.onclick = () => { overlay.classList.remove('show'); resolve(); };
                        overlay.classList.add('show');
                    });
                }

                function customConfirm(msg) {
                    return new Promise(resolve => {
                        const overlay = document.getElementById('custom-modal-overlay');
                        document.getElementById('modal-text').innerText = msg;
                        const btnOk = document.getElementById('modal-btn-ok');
                        const btnCancel = document.getElementById('modal-btn-cancel');
                        btnCancel.style.display = 'inline-block';
                        btnOk.onclick = () => { overlay.classList.remove('show'); resolve(true); };
                        btnCancel.onclick = () => { overlay.classList.remove('show'); resolve(false); };
                        overlay.classList.add('show');
                    });
                }

                function toggleDropdown(el) {
                    const container = el.closest('.custom-select-container');
                    container.classList.toggle('open');
                    if (container.classList.contains('open')) fetchServerPresets();
                }

                function selectModel(option, index) {
                    const val = option.getAttribute('data-value');
                    ensembleState[index][0] = val;

                    let targetStem = '';
                    if (modelsData[val] && modelsData[val].stems) {
                        targetStem = modelsData[val].target_instrument;
                        if (!targetStem || !modelsData[val].stems.includes(targetStem)) {
                            targetStem = modelsData[val].stems[0];
                        }
                    }
                    ensembleState[index][1] = targetStem || '';
                    sendStateToParent();
                    renderEditor();
                }

                function selectStem(option, index) {
                    if (option.classList.contains('disabled')) return;
                    ensembleState[index][1] = option.getAttribute('data-value');
                    sendStateToParent();
                    renderEditor();
                }

                function openDropdown(input) {
                    const container = input.closest('.custom-select-container');
                    if (container.classList.contains('open')) return;
                    document.querySelectorAll('.custom-select-container').forEach(c => c.classList.remove('open'));
                    container.classList.add('open');
                    setTimeout(() => { input.setSelectionRange(input.value.length, input.value.length); }, 0);
                    container.querySelectorAll('.custom-option').forEach(opt => opt.style.display = '');
                }

                function closeDropdown(container) {
                    if (!container.classList.contains('open')) return;
                    container.classList.remove('open');
                    const input = container.querySelector('.custom-select-input');
                    const val = container.getAttribute('data-value');
                    if (val) {
                        input.value = val;
                    } else if (!input.hasAttribute('readonly')) {
                        input.value = '';
                    }
                }

                function filterOptions(input) {
                    const container = input.closest('.custom-select-container');
                    const query = input.value.toLowerCase().trim();
                    const options = container.querySelectorAll('.custom-option');
                    container.classList.add('open');

                    options.forEach(opt => {
                        if (opt.classList.contains('disabled')) return;
                        opt.style.display = opt.innerText.toLowerCase().includes(query) ? '' : 'none';
                    });
                }

                document.addEventListener('click', e => {
                    if (!e.target.closest('.custom-select-container')) {
                        document.querySelectorAll('.custom-select-container').forEach(c => closeDropdown(c));
                    }
                });

                function addEnsembleItem(model = '', stem = '', invert = false, weight = 1.0) {
                    ensembleState.push([model, stem, invert, weight]);
                    sendStateToParent();
                    renderEditor();
                }

                function removeEnsembleItem(index) {
                    ensembleState.splice(index, 1);
                    sendStateToParent();
                    renderEditor();
                }

                function renderEditor() {
                    const container = document.getElementById('editor-layout');
                    container.innerHTML = '';

                    if (ensembleState.length === 0) {
                        return;
                    }

                    ensembleState.forEach((item, index) => {
                        const model = item[0] || '';
                        const stem = item[1] || '';
                        const invert = item[2] || false;
                        const weight = item[3] !== undefined ? item[3] : 1.0;

                        let modelOptionsHtml = '';
                        for (const shortName of Object.keys(modelsData)) {
                            modelOptionsHtml += `<div class="custom-option" data-value="${shortName}" onclick="selectModel(this, ${index})">${shortName}</div>`;
                        }

                        let stemOptionsHtml = '<div class="custom-option disabled">""" + f"{_i18n('separate_no_model')}" + """</div>';
                        if (model && modelsData[model] && modelsData[model].stems) {
                            stemOptionsHtml = '';
                            modelsData[model].stems.forEach(s => {
                                stemOptionsHtml += `<div class="custom-option" data-value="${s}" onclick="selectStem(this, ${index})">${s}</div>`;
                            });
                        }

                        const div = document.createElement('div');
                        div.className = 'ensemble-item';
                        div.innerHTML = `
                                            <div class="ensemble-item-number">${index + 1}</div>
                                            
                                            <div class="custom-select-container" data-value="${model}" style="flex:2">
                                                <div class="custom-select-input-wrapper">
                                                    <input type="text" class="custom-select-input" placeholder=""" + f"\"{_i18n('model_name')}\"" + """ value="${model}" onfocus="openDropdown(this)" onclick="openDropdown(this)" oninput="filterOptions(this)" autocomplete="off">
                                                </div>
                                                <div class="custom-select-options">${modelOptionsHtml}</div>
                                            </div>

                                            <div class="custom-select-container" data-value="${stem}" style="flex:1.5">
                                                <div class="custom-select-input-wrapper">
                                                    <input type="text" class="custom-select-input" placeholder="${model ? '""" + _i18n("primary_stem") + """' : '""" + _i18n("separate_no_model") + """'}" value="${stem}" onfocus="openDropdown(this)" onclick="openDropdown(this)" oninput="filterOptions(this)" autocomplete="off" ${!model ? 'readonly' : ''}>
                                                </div>
                                                <div class="custom-select-options">${stemOptionsHtml}</div>
                                            </div>

                                            <!-- Переключатель (switch) вместо чекбокса -->
                                            <div class="toggle-container" style="flex:0.8; display: flex; align-items: center; gap: 8px;">
                                                <span style="font-size: 13px; color: var(--text-main);">""" + _i18n("invert") + """</span>
                                                <label class="toggle-switch">
                                                    <input type="checkbox" ${invert ? 'checked' : ''} onchange="ensembleState[${index}][2] = this.checked; sendStateToParent();">
                                                    <span class="slider"></span>
                                                </label>
                                            </div>

                                            <div style="flex:0.8; display: flex; align-items: center; gap: 5px;">
                                                <span style="font-size: 12px; color: var(--text-main); " class="weights-text">""" + _i18n("weights") + """:</span>
                                                <input type="number" class="weights-number" step="0.1" value="${weight}" onchange="ensembleState[${index}][3] = parseFloat(this.value) || 1.0; sendStateToParent();">
                                            </div>

                                            <div class="item-controls">
                                                <button class="btn btn-red" onclick="removeEnsembleItem(${index})" title=""" + f"\"{_i18n('delete')}\"" + """>✕</button>
                                            </div>
                                        `;
                        container.appendChild(div);
                    });
                }

                function clearEditor() {
                    ensembleState = [];
                    document.getElementById('preset-name').value = '';
                    sendStateToParent();
                    renderEditor();
                }

                // API Функции
                async function fetchServerPresets() {
                    try {
                        const res = await fetch('/auto_ensemble_preset');
                        const data = await res.json();
                        const list = document.getElementById('server-preset-list');
                        list.innerHTML = '';

                        if (data.presets && data.presets.length > 0) {
                            data.presets.forEach(p => {
                                const opt = document.createElement('div');
                                opt.className = 'custom-option';
                                opt.innerText = p;
                                opt.onclick = () => {
                                    loadAutoEnsemblePreset(p);
                                    // Закрываем текущий выпадающий список после выбора
                                    closeDropdown(opt.closest('.custom-select-container')); 
                                };
                                list.appendChild(opt);
                            });
                        }
                    } catch (e) {
                        console.error('""" + f"{_i18n('preset_node_err_fetch')}" + """', e);
                    }
                }

                async function loadAutoEnsemblePreset(name) {
                    try {
                        const res = await fetch(`/auto_ensemble_preset/${name}`);
                        const data = await res.json();

                        if (data.error) throw new Error(data.error);

                        if (data.warning) {
                            await customAlert(data.warning);
                        }

                        if (data.state) {
                            loadJSON(data.state);
                            document.getElementById('preset-name').value = name;
                            sendStateToParent();
                            renderEditor();
                        }

                    } catch (e) {
                        await customAlert(""" + f"\"{_i18n('preset_node_err_load')}\"" + """ + e.message);
                    }
                }

                async function saveServerPreset() {
                    const name = document.getElementById('preset-name').value.trim();
                    if (!name) return customAlert('""" + f"{_i18n('preset_node_err_name_empty')}" + """');

                    try {
                        const res = await fetch(`/auto_ensemble_preset/${name}`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(ensembleState)
                        });

                        const result = await res.json();
                        if (result.error) throw new Error(result.error);

                        await customAlert('""" + f"{_i18n('preset_node_save_success')}" + """');
                        fetchServerPresets();
                    } catch (e) {
                        await customAlert('""" + f"{_i18n('preset_node_error_save')}" + """' + e.message);
                    }
                }

                async function deleteServerPreset() {
                    const name = document.getElementById('preset-name').value.trim();
                    if (!name) return customAlert('""" + f"{_i18n('preset_node_err_name_delete')}" + """');

                    if (!(await customConfirm('""" + f"{_i18n('preset_node_confirm_delete')}" + """ "' + name + '"?'))) return;

                    try {
                        const res = await fetch(`/auto_ensemble_preset/${name}`, { method: 'DELETE' });
                        const result = await res.json();

                        if (result.error) throw new Error(result.error);

                        clearEditor();
                        await customAlert('""" + f"{_i18n('preset_node_delete_success')}" + """');
                        fetchServerPresets();
                    } catch (e) {
                        await customAlert('""" + f"{_i18n('preset_node_error_delete')}" + """' + e.message);
                    }
                }

                let stateTimeout;
                window.sendStateToParent = function () {
                    clearTimeout(stateTimeout);
                    stateTimeout = setTimeout(() => {
                        const data = ensembleState;
                        window.parent.postMessage({ type: 'update_auto_ensemble_preset', payload: data }, '*');
                    }, 100);
                };

                // Подключаем импорт
                setupImport();
            </script>
        </body>
        </html>
        """

        ITERATIVE_ENSEMBLE_PRESET_HTML_CONTENT = """<!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
            <title>""" + f"{_i18n('auto_ensemble_preset_editor_title')}" + """</title>
            <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap" rel="stylesheet">
            <style>



                /* -- ITERATIVE ENSEMBLE --- */





                :root {
                    --bg-color: var(--background-fill-primary, #eef1f5);
                    --grid-color: var(--border-color-secondary, #d4d8dd);
                    --node-bg: var(--background-fill-secondary, #f6f8fa);
                    --node-header: var(--block-background-fill, #e6ebf1);
                    --border-color: var(--border-color-primary, #d0d7de);
                    --text-main: var(--body-text-color, #333333);
                    --btn-blue: var(--color-accent, #007bff);
                    --primary: var(--color-accent, #007bff);
                    --primary-hover: var(--color-accent-soft, #0056b3);
                }

                @media (prefers-color-scheme: dark) {
                    :root {
                        --bg-color: var(--background-fill-primary, #0b0f19);
                        --grid-color: var(--border-color-secondary, #1f2937);
                        --node-bg: var(--background-fill-secondary, #1f2937);
                        --node-header: var(--block-background-fill, #374151);
                        --border-color: var(--border-color-primary, #374151);
                        --text-main: var(--body-text-color, #f3f4f6);
                    }
                }

                .dark {
                    --bg-color: var(--background-fill-primary, #0b0f19);
                    --grid-color: var(--border-color-secondary, #1f2937);
                    --node-bg: var(--background-fill-secondary, #1f2937);
                    --node-header: var(--block-background-fill, #374151);
                    --border-color: var(--border-color-primary, #374151);
                    --text-main: var(--body-text-color, #f3f4f6);
                }

                * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Montserrat', sans-serif; }
                body { width: 100vw; height: 100vh; overflow: hidden; display: flex; flex-direction: column; background: var(--bg-color); color: var(--text-main); }
                
                .topbar { height: 60px; background: var(--node-bg); border-bottom: 1px solid var(--border-color); display: flex; align-items: center; padding: 0 20px; gap: 10px; z-index: 100; box-shadow: 0 2px 5px rgba(0,0,0,0.02); flex-shrink: 0; }
                input[type="text"], input[type="number"] { padding: 7px 12px; border: 1px solid var(--border-color); border-radius: 4px; font-size: 14px; outline: none; background: var(--bg-color); color: var(--text-main); }
                .btn { padding: 8px 16px; border-radius: 4px; border: 1px solid var(--border-color); background: var(--node-bg); color: var(--text-main); cursor: pointer; font-weight: 500; font-size: 14px; transition: 0.2s; white-space: nowrap; }
                .btn:hover { background: var(--grid-color); }
                .btn-blue { color: var(--btn-blue); border-color: var(--btn-blue); }
                .btn-green { color: #28a745; border-color: #28a745; }
                .btn-red { color: #dc3545; border-color: #dc3545; }

                /* Модальные окна */
                .modal-overlay {
                    position: fixed;
                    top: 0; left: 0;
                    width: 100vw; height: 100vh;
                    background: rgba(0,0,0,0.4);
                    display: none;
                    align-items: center;
                    justify-content: center;
                    z-index: 99999999;
                    opacity: 0;
                    transition: opacity 0.2s;
                    padding: 16px;          /* ← отступ от краёв экрана */
                    box-sizing: border-box;
                }
                .modal-overlay.show { display: flex; opacity: 1; }
                .modal-box {
                    background: var(--node-bg);
                    padding: 25px;
                    border-radius: 8px;
                    width: 100%;            /* ← было min-width:320px */
                    max-width: 420px;       /* ← было max-width:90vw */
                    max-height: 90vh;       /* ← новое: ограничение высоты */
                    overflow-y: auto;       /* ← новое: скролл если контент не влезает */
                    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                    text-align: center;
                    box-sizing: border-box;
                }
                .modal-text { margin-bottom: 25px; font-size: 15px; font-weight: 500; color: var(--text-main); word-wrap: break-word; }
                .modal-buttons { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
                .modal-buttons .btn { min-width: 100px; }

                /* Каскадные селекторы (Gradio-like) */
                .custom-select-container { position: relative; width: 100%; box-sizing: border-box; }
                .custom-select-input-wrapper { position: relative; width: 100%; }
                .custom-select-input { width: 100%; padding-right: 30px; border: 1px solid var(--border-color); border-radius: 4px; font-size: 14px; background: var(--bg-color); color: var(--text-main); cursor: pointer; text-overflow: ellipsis; white-space: nowrap; overflow: hidden; }
                .custom-select-input-wrapper::after { content: '▼'; font-size: 10px; color: var(--text-main); position: absolute; right: 10px; top: 50%; transform: translateY(-50%); pointer-events: none; opacity: 0.7; }
                .custom-select-options { position: absolute; top: 100%; left: 0; width: 100%; background: var(--node-bg); border: 1px solid var(--border-color); border-radius: 4px; margin-top: 4px; max-height: 200px; overflow-y: auto; z-index: 999; display: none; box-shadow: 0 5px 15px rgba(0,0,0,0.15); }
                .custom-select-container.open .custom-select-options { display: block; }
                .custom-option { padding: 9px 12px; cursor: pointer; font-size: 13px; color: var(--text-main); transition: background 0.2s; white-space: pre-wrap; overflow: hidden; overflow-wrap: anywhere;  }
                .custom-option:hover { background: var(--grid-color); color: var(--primary-hover); }
                .custom-option.disabled { color: #aaa; cursor: not-allowed; }
                .custom-option.disabled:hover { background: transparent; color: #aaa; }

                /* Рабочая область (Списочный редактор) */
                .main-container { flex: 1; padding: 20px; overflow-y: auto; background-color: var(--bg-color); }
                .editor-layout { max-width: 1000px; margin: 0 auto; display: flex; flex-direction: column; gap: 15px; }
                
                .ensemble-item { background: var(--node-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 15px; display: flex; align-items: center; justify-content: space-between; gap: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.02); transition: 0.2s; }
                .ensemble-item:hover { border-color: var(--primary); }
                .ensemble-item-number { font-weight: 600; color: var(--text-main); opacity: 0.5; width: 25px; text-align: center; }
                
                .checkbox-container { display: flex; align-items: center; gap: 8px; font-size: 13px; cursor: pointer; user-select: none; }
                .checkbox-container input[type="checkbox"] { cursor: pointer; width: 16px; height: 16px; accent-color: var(--primary); }
                
                .item-controls { display: flex; gap: 10px; }
                .add-btn-wrapper { text-align: center; margin-top: 25px; }
                .empty-state { text-align: center; color: var(--text-main); opacity: 0.6; font-size: 14px; margin-top: 40px; }

                .toggle-container {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                .toggle-switch {
                    position: relative;
                    display: inline-block;
                    width: 34px;
                    height: 20px;
                    flex-shrink: 0;
                }
                .toggle-switch input {
                    opacity: 0;
                    width: 0;
                    height: 0;
                    margin: 0;
                }
                .slider {
                    position: absolute;
                    cursor: pointer;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background-color: var(--border-color);
                    transition: .3s;
                    border-radius: 20px;
                }
                .slider:before {
                    position: absolute;
                    content: "";
                    height: 14px;
                    width: 14px;
                    left: 3px;
                    bottom: 3px;
                    background-color: var(--node-bg);
                    transition: .3s;
                    border-radius: 50%;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                }
                input:checked + .slider {
                    background-color: var(--btn-blue);
                }
                input:checked + .slider:before {
                    transform: translateX(14px);
                }
                .weights-number {
                    max-width: 10ch
                }

                @media (max-width: 1024px) {
                    /* Topbar responsive adjustments */
                    .topbar { flex-wrap: wrap; height: auto; padding: 10px; justify-content: space-between; }
                    .topbar h1 { display: none; }
                    .custom-select-container { width: 100% !important; margin-bottom: 5px; }
                    .topbar input[type="text"] { flex: 1 1 auto; width: 100%; min-width: 150px; margin-bottom: 5px; }
                    .topbar .btn { flex: 1 1 auto; margin: 2px; font-size: 12px; padding: 6px 10px; }
                    #sidebar-toggle { display: none !important; }
                    .main-container { flex-direction: column; }

                    /* Custom select full width on mobile */
                    .custom-select-container { width: 100% !important; margin-bottom: 5px; }

                    .custom-select-input {
                        font-size: 13px;
                        padding: 8px 30px 8px 12px;
                    }

                    .custom-select-options {
                        max-height: 150px;
                    }

                    /* Hide sidebar toggle if exists */
                    #sidebar-toggle {
                        display: none !important;
                    }

                    /* Main container */
                    .main-container {
                        flex-direction: column;
                        padding: 12px;
                    }



                    /* Ensemble items - mobile friendly */
                    .ensemble-item {
                        display: flex;
                        flex-direction: column;
                        gap: 12px;
                        padding: 15px;
                        align-items: stretch;
                        border-radius: 6px;
                        transition: border-color 0.2s;
                    }

                    .ensemble-item-number {
                        display: none; /* Hide numbers on mobile for cleaner look */
                        /* Or keep minimal: width: 20px; font-size: 12px; */
                    }

                    /* Checkbox container full width */
                    .toggle-container {
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }

                    .toggle-switch {
                        position: relative;
                        display: inline-block;
                        width: 34px;
                        height: 20px;
                        flex-shrink: 0;
                        margin-left: auto; /* ← Перемещает только переключатель вправо */
                    }

                    .weights-number {
                        margin-left: auto;
                    }

                    /* Item controls - button group */
                    .item-controls {
                        display: flex;
                        gap: 8px;
                        width: 100%;
                        justify-content: flex-end;
                        flex-wrap: wrap;
                    }

                    .item-controls .btn {
                        flex: 1 1 auto;
                        min-width: 60px;
                        padding: 8px 12px;
                        font-size: 12px;
                        text-align: center;
                    }

                    /* Add button wrapper */
                    .add-btn-wrapper {
                        margin-top: 20px;
                        text-align: center;
                    }

                    .add-btn-wrapper .btn {
                        width: 100%;
                        max-width: 300px;
                        padding: 12px 20px;
                        font-size: 14px;
                    }

                    /* Empty state */
                    .empty-state {
                        font-size: 13px;
                        margin-top: 30px;
                        padding: 20px;
                    }
                }
            



            
            



            </style>
        </head>
        <body>

            <!-- Модальное окно подтверждений/ошибок -->
            <div id="custom-modal-overlay" class="modal-overlay">
                <div class="modal-box">
                    <div class="modal-text" id="modal-text"></div>
                    <div class="modal-buttons">
                        <button class="btn btn-blue" id="modal-btn-ok">""" + f"{_i18n('preset_node_ok')}" + """</button>
                        <button class="btn btn-red" id="modal-btn-cancel" style="display:none;">""" + f"{_i18n('preset_node_cancel')}" + """</button>
                    </div>
                </div>
            </div>

            <!-- Верхняя панель (Topbar) -->
            <div class="topbar">
                <div class="custom-select-container" id="preset-select-container" data-value="" style="width: 220px;" onmouseenter="fetchServerPresets()">
                    <div class="custom-select-input-wrapper">
                        <input type="text" class="custom-select-input" placeholder=""" + f"\"{_i18n('preset_node_saved')}\"" + """ onfocus="openDropdown(this)" onclick="openDropdown(this)" oninput="filterOptions(this)" autocomplete="off">
                    </div>
                    <div class="custom-select-options" id="server-preset-list">
                        <div class="custom-option disabled">""" + f"{_i18n('preset_node_loading')}" + """</div>
                    </div>
                </div>

                <input type="text" id="preset-name" placeholder=""" + f"\"{_i18n('preset_node_preset_name')}\"" + """ value="">
                <button class="btn btn-green" onclick="saveServerPreset()">""" + f"{_i18n('preset_node_save')}" + """</button>
                <button class="btn btn-red" onclick="deleteServerPreset()">""" + f"{_i18n('preset_node_delete')}" + """</button>

                <span style="border-left: 1px solid var(--border-color); height: 30px; margin: 0 5px;" class="desktop-only"></span>

                <button class="btn btn-blue" onclick="exportJSON()">""" + f"{_i18n('preset_node_preset_download')}" + """</button>
                <button class="btn" id="import-btn">""" + f"{_i18n('preset_node_preset_upload')}" + """</button>
                <input type="file" id="import-file" style="display:none" accept=".json">

                <button class="btn btn-red" style="margin-left: auto;" onclick="clearEditor()">""" + f"{_i18n('clear')}" + """</button>
            </div>

            <!-- Рабочая область -->
            <div class="main-container">
                <div class="editor-layout" id="editor-layout">
                    <!-- Элементы ансамбля рендерятся здесь -->
                </div>
                <div class="add-btn-wrapper">
                    <button class="btn btn-blue" onclick="addEnsembleItem()">""" + f"{_i18n('add_model')}" + """</button>
                </div>
            </div>

            <script>
                let modelsData = {};
                let ensembleState = [];
                let idCounter = 1;

                // Инициализация
                const urlParams = new URLSearchParams(window.location.search);
                if (urlParams.get('__theme') === 'dark') document.documentElement.classList.add('dark');

                window.addEventListener('message', function (event) {
                    let themeData = typeof event.data === 'object' ? (event.data.theme || event.data.type) : event.data;
                    if (themeData === 'theme_dark' || themeData === 'dark') document.documentElement.classList.add('dark');
                    else if (themeData === 'theme_light' || themeData === 'light') document.documentElement.classList.remove('dark');
                });

                async function init() {
                    try {""" + """                      modelsData = """ + json.dumps(self.separator.info, ensure_ascii=False, indent=0) + ";" + """
                    } catch (e) {
                        console.error("Failed to load models.json", e);
                        modelsData = { "NO_MODELS": { "stems": ["Vocals", "Instrumental"] } }; 
                    }
                }
                window.onload = init;

                // Прием данных
                window.addEventListener('message', (e) => {
                    if (e.data && e.data.type === 'update_auto_ensemble_preset') {
                        loadJSON(e.data.payload);
                    }
                });

                function exportJSON() {
                    const name = document.getElementById('preset-name').value || "auto_ensemble_preset";
                    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(ensembleState, null, 4));
                    const downloadAnchorNode = document.createElement('a');
                    downloadAnchorNode.setAttribute("href", dataStr);
                    downloadAnchorNode.setAttribute("download", name + ".json");
                    document.body.appendChild(downloadAnchorNode);
                    downloadAnchorNode.click();
                    downloadAnchorNode.remove();
                }

                function setupImport() {
                    const importBtn = document.getElementById('import-btn');
                    const importFile = document.getElementById('import-file');
                    
                    if (importBtn && importFile) {
                        importBtn.onclick = () => importFile.click();
                        
                        importBtn.addEventListener('dragover', (e) => {
                            e.preventDefault();
                            importBtn.style.opacity = '0.7';
                            importBtn.style.border = '2px dashed var(--btn-blue)';
                        });
                        
                        importBtn.addEventListener('dragleave', (e) => {
                            e.preventDefault();
                            importBtn.style.opacity = '1';
                            importBtn.style.border = '1px solid var(--border-color)';
                        });
                        
                        importBtn.addEventListener('drop', (e) => {
                            e.preventDefault();
                            importBtn.style.opacity = '1';
                            importBtn.style.border = '1px solid var(--border-color)';
                            
                            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                                processFile(e.dataTransfer.files[0]);
                            }
                        });

                        importFile.onchange = e => {
                            const file = e.target.files[0];
                            if (!file) return;
                            processFile(file);
                            importFile.value = ""; 
                        };
                    }
                }

                function processFile(file) {
                    const reader = new FileReader();
                    reader.onload = async event => {
                        try {
                            const json = JSON.parse(event.target.result);
                            const validatedFlow = await validateFlowJS(json);
                            
                            loadJSON(validatedFlow);
                            document.getElementById('preset-name').value = json.name || "";
                            sendStateToParent();
                            
                        } catch (err) {
                            await customAlert(""" + f"\"{_i18n('preset_node_invalid_json')}\"" + """ + "\\n" + err.message);
                        }
                    };
                    reader.readAsText(file);
                }

                // Реализация validate_flow на JS (с учетом iterative=True)
                async function validateFlowJS(flow) {
                    let errors = [];
                    let warns = [];
                    let validated_flow = [];
                    
                    if (!flow) {
                        throw new Error(""" + f"\"{_i18n('flow_empty')}\"" + """);
                    }
                    
                    if (!Array.isArray(flow)) {
                        throw new Error(""" + f"\"{_i18n('flow_validation_error', error=_i18n('flow_not_list'))}\"" + """);
                    }

                    for (let i = 0; i < flow.length; i++) {
                        let model_flow = flow[i];
                        let valid = true;
                        let error_parts = [];
                        
                        if (!Array.isArray(model_flow)) {
                            let errMsg = """ + f"\"{_i18n('flow_item_not_list', type='__TYPE__')}\"" + """.replace('__TYPE__', typeof model_flow);
                            errors.push({ index: i, error: errMsg });
                            continue;
                        }
                        
                        if (model_flow.length !== 3 && model_flow.length !== 4) {
                            let errMsg = """ + f"\"{_i18n('flow_invalid_length', length='__LEN__', expected='3 or 4')}\"" + """.replace('__LEN__', model_flow.length);
                            errors.push({ index: i, error: errMsg });
                            continue;
                        }
                        
                        let model_name = model_flow[0];
                        let primary_stem = model_flow[1];
                        let invert = model_flow[2];
                        let weight = model_flow.length === 4 ? model_flow[3] : 1.0;
                        let has_weight = model_flow.length === 4;
                        
                        let model_exists = false;
                        
                        // Валидация model_name
                        if (typeof model_name !== 'string') {
                            let errMsg = """ + f"\"{_i18n('flow_invalid_type', field='model_name', expected='str', got='__GOT__')}\"" + """.replace('__GOT__', typeof model_name);
                            error_parts.push(errMsg);
                            valid = false;
                        } else {
                            model_exists = modelsData.hasOwnProperty(model_name);
                            if (!model_exists && !modelsData["NO_MODELS"]) {
                                let errMsg = """ + f"\"{_i18n('flow_model_not_found', model='__MOD__', available='self.get_all_models()')}\"" + """.replace('__MOD__', model_name);
                                warns.push(errMsg);
                            }
                        }
                        
                        // Валидация primary_stem
                        if (typeof primary_stem !== 'string') {
                            let errMsg = """ + f"\"{_i18n('flow_invalid_type', field='primary_stem', expected='str', got='__GOT__')}\"" + """.replace('__GOT__', typeof primary_stem);
                            error_parts.push(errMsg);
                            valid = false;
                        } else if (model_exists && !modelsData["NO_MODELS"]) {
                            let available_stems = modelsData[model_name].stems || [];
                            if (available_stems.length > 0 && !available_stems.includes(primary_stem)) {
                                let errMsg = """ + f"\"{_i18n('flow_stem_not_found', stem='__STEM__', model='__MOD__', available='__AVAIL__')}\"" + """
                                    .replace('__STEM__', primary_stem)
                                    .replace('__MOD__', model_name)
                                    .replace('__AVAIL__', available_stems.join(', '));
                                warns.push(errMsg);
                            }
                        }
                        
                        // Валидация invert
                        if (typeof invert !== 'boolean') {
                            let errMsg = """ + f"\"{_i18n('flow_invalid_type', field='invert', expected='bool', got='__GOT__')}\"" + """.replace('__GOT__', typeof invert);
                            error_parts.push(errMsg);
                            valid = false;
                        }
                        
                        // Валидация weight (проверяем на ошибки, но позже отбрасываем)
                        if (has_weight) {
                            if (typeof weight === 'string') {
                                let parsed = parseFloat(weight);
                                if (isNaN(parsed)) {
                                    error_parts.push(""" + f"\"{_i18n('flow_weight_empty')}\"" + """);
                                    valid = false;
                                }
                            } else if (typeof weight !== 'number') {
                                let errMsg = """ + f"\"{_i18n('flow_invalid_type', field='weight', expected='number', got='__GOT__')}\"" + """.replace('__GOT__', typeof weight);
                                error_parts.push(errMsg);
                                valid = false;
                            }
                        }
                        
                        if (valid) {
                            // iterative=True отбрасывает weight и оставляет только [str, str, bool]
                            validated_flow.push([model_name, primary_stem, invert]);
                        } else {
                            errors.push({ index: i, error: error_parts.join('; ') });
                        }
                    }
                    
                    if (errors.length > 0) {
                        let error_messages = errors.map(e => `#${e.index}: ${e.error}`).join("\\n");
                        let finalErr = """ + f"\"{_i18n('flow_validation_errors', count='__CNT__')}\"" + """.replace('__CNT__', errors.length) + "\\n" + error_messages;
                        throw new Error(finalErr);
                    }
                    
                    if (warns.length > 0) {
                        await customAlert(warns.join("\\n\\n"));
                    }
                    
                    return validated_flow;
                }

                function loadJSON(state) {
                    const data = typeof state === 'string' ? JSON.parse(state) : state;
                    if (Array.isArray(data)) {
                        ensembleState = data.map(item => {
                            if (Array.isArray(item) && item.length >= 3) {
                                return [item[0] || '', item[1] || '', item[2] || false];
                            }
                            return ['', '', false];
                        });
                    } else {
                        ensembleState = [];
                    }
                    renderEditor();
                }

                function customAlert(msg) {
                    return new Promise(resolve => {
                        const overlay = document.getElementById('custom-modal-overlay');
                        document.getElementById('modal-text').innerText = msg;
                        const btnOk = document.getElementById('modal-btn-ok');
                        document.getElementById('modal-btn-cancel').style.display = 'none';
                        btnOk.onclick = () => { overlay.classList.remove('show'); resolve(); };
                        overlay.classList.add('show');
                    });
                }

                function customConfirm(msg) {
                    return new Promise(resolve => {
                        const overlay = document.getElementById('custom-modal-overlay');
                        document.getElementById('modal-text').innerText = msg;
                        const btnOk = document.getElementById('modal-btn-ok');
                        const btnCancel = document.getElementById('modal-btn-cancel');
                        btnCancel.style.display = 'inline-block';
                        btnOk.onclick = () => { overlay.classList.remove('show'); resolve(true); };
                        btnCancel.onclick = () => { overlay.classList.remove('show'); resolve(false); };
                        overlay.classList.add('show');
                    });
                }

                function toggleDropdown(el) {
                    const container = el.closest('.custom-select-container');
                    container.classList.toggle('open');
                    if (container.classList.contains('open')) fetchServerPresets();
                }

                function selectModel(option, index) {
                    const val = option.getAttribute('data-value');
                    ensembleState[index][0] = val;

                    let targetStem = '';
                    if (modelsData[val] && modelsData[val].stems) {
                        targetStem = modelsData[val].target_instrument;
                        if (!targetStem || !modelsData[val].stems.includes(targetStem)) {
                            targetStem = modelsData[val].stems[0];
                        }
                    }
                    ensembleState[index][1] = targetStem || '';
                    sendStateToParent();
                    renderEditor();
                }

                function selectStem(option, index) {
                    if (option.classList.contains('disabled')) return;
                    ensembleState[index][1] = option.getAttribute('data-value');
                    sendStateToParent();
                    renderEditor();
                }

                function openDropdown(input) {
                    const container = input.closest('.custom-select-container');
                    if (container.classList.contains('open')) return;
                    document.querySelectorAll('.custom-select-container').forEach(c => c.classList.remove('open'));
                    container.classList.add('open');
                    setTimeout(() => { input.setSelectionRange(input.value.length, input.value.length); }, 0);
                    container.querySelectorAll('.custom-option').forEach(opt => opt.style.display = '');
                }

                function closeDropdown(container) {
                    if (!container.classList.contains('open')) return;
                    container.classList.remove('open');
                    const input = container.querySelector('.custom-select-input');
                    const val = container.getAttribute('data-value');
                    if (val) {
                        input.value = val;
                    } else if (!input.hasAttribute('readonly')) {
                        input.value = '';
                    }
                }

                function filterOptions(input) {
                    const container = input.closest('.custom-select-container');
                    const query = input.value.toLowerCase().trim();
                    const options = container.querySelectorAll('.custom-option');
                    container.classList.add('open');

                    options.forEach(opt => {
                        if (opt.classList.contains('disabled')) return;
                        opt.style.display = opt.innerText.toLowerCase().includes(query) ? '' : 'none';
                    });
                }

                document.addEventListener('click', e => {
                    if (!e.target.closest('.custom-select-container')) {
                        document.querySelectorAll('.custom-select-container').forEach(c => closeDropdown(c));
                    }
                });

                function addEnsembleItem(model = '', stem = '', invert = false) {
                    ensembleState.push([model, stem, invert]);
                    sendStateToParent();
                    renderEditor();
                }

                function removeEnsembleItem(index) {
                    ensembleState.splice(index, 1);
                    sendStateToParent();
                    renderEditor();
                }

                function renderEditor() {
                    const container = document.getElementById('editor-layout');
                    container.innerHTML = '';

                    if (ensembleState.length === 0) {
                        return;
                    }

                    ensembleState.forEach((item, index) => {
                        const model = item[0] || '';
                        const stem = item[1] || '';
                        const invert = item[2] || false;

                        let modelOptionsHtml = '';
                        for (const shortName of Object.keys(modelsData)) {
                            modelOptionsHtml += `<div class="custom-option" data-value="${shortName}" onclick="selectModel(this, ${index})">${shortName}</div>`;
                        }

                        let stemOptionsHtml = '<div class="custom-option disabled">""" + f"{_i18n('separate_no_model')}" + """</div>';
                        if (model && modelsData[model] && modelsData[model].stems) {
                            stemOptionsHtml = '';
                            modelsData[model].stems.forEach(s => {
                                stemOptionsHtml += `<div class="custom-option" data-value="${s}" onclick="selectStem(this, ${index})">${s}</div>`;
                            });
                        }

                        const div = document.createElement('div');
                        div.className = 'ensemble-item';
                        div.innerHTML = `
                                            <div class="ensemble-item-number">${index + 1}</div>
                                            
                                            <div class="custom-select-container" data-value="${model}" style="flex:2">
                                                <div class="custom-select-input-wrapper">
                                                    <input type="text" class="custom-select-input" placeholder=""" + f"\"{_i18n('model_name')}\"" + """ value="${model}" onfocus="openDropdown(this)" onclick="openDropdown(this)" oninput="filterOptions(this)" autocomplete="off">
                                                </div>
                                                <div class="custom-select-options">${modelOptionsHtml}</div>
                                            </div>

                                            <div class="custom-select-container" data-value="${stem}" style="flex:1.5">
                                                <div class="custom-select-input-wrapper">
                                                    <input type="text" class="custom-select-input" placeholder="${model ? '""" + _i18n("primary_stem") + """' : '""" + _i18n("separate_no_model") + """'}" value="${stem}" onfocus="openDropdown(this)" onclick="openDropdown(this)" oninput="filterOptions(this)" autocomplete="off" ${!model ? 'readonly' : ''}>
                                                </div>
                                                <div class="custom-select-options">${stemOptionsHtml}</div>
                                            </div>

                                            <div class="toggle-container" style="flex:0.8; display: flex; align-items: center; gap: 8px;">
                                                <span style="font-size: 13px; color: var(--text-main);">""" + _i18n("invert") + """</span>
                                                <label class="toggle-switch">
                                                    <input type="checkbox" ${invert ? 'checked' : ''} onchange="ensembleState[${index}][2] = this.checked; sendStateToParent();">
                                                    <span class="slider"></span>
                                                </label>
                                            </div>

                                            <div class="item-controls">
                                                <button class="btn btn-red" onclick="removeEnsembleItem(${index})" title=""" + f"\"{_i18n('delete')}\"" + """>✕</button>
                                            </div>
                                        `;
                        container.appendChild(div);
                    });
                }

                function clearEditor() {
                    ensembleState = [];
                    document.getElementById('preset-name').value = '';
                    sendStateToParent();
                    renderEditor();
                }

                async function fetchServerPresets() {
                    try {
                        const res = await fetch('/iter_ensemble_preset');
                        const data = await res.json();
                        const list = document.getElementById('server-preset-list');
                        list.innerHTML = '';

                        if (data.presets && data.presets.length > 0) {
                            data.presets.forEach(p => {
                                const opt = document.createElement('div');
                                opt.className = 'custom-option';
                                opt.innerText = p;
                                opt.onclick = () => {
                                    loadAutoEnsemblePreset(p);
                                    // Закрываем текущий выпадающий список после выбора
                                    closeDropdown(opt.closest('.custom-select-container')); 
                                };
                                list.appendChild(opt);
                            });
                        }
                    } catch (e) {
                        console.error('""" + f"{_i18n('preset_node_err_fetch')}" + """', e);
                    }
                }

                async function loadAutoEnsemblePreset(name) {
                    try {
                        const res = await fetch(`/iter_ensemble_preset/${name}`);
                        const data = await res.json();

                        if (data.error) throw new Error(data.error);

                        if (data.warning) {
                            await customAlert(data.warning);
                        }

                        if (data.state) {
                            loadJSON(data.state);
                            document.getElementById('preset-name').value = name;
                            sendStateToParent();
                            renderEditor();
                        }

                    } catch (e) {
                        await customAlert(""" + f"\"{_i18n('preset_node_err_load')}\"" + """ + e.message);
                    }
                }

                async function saveServerPreset() {
                    const name = document.getElementById('preset-name').value.trim();
                    if (!name) return customAlert('""" + f"{_i18n('preset_node_err_name_empty')}" + """');

                    try {
                        const res = await fetch(`/iter_ensemble_preset/${name}`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(ensembleState)
                        });

                        const result = await res.json();
                        if (result.error) throw new Error(result.error);

                        await customAlert('""" + f"{_i18n('preset_node_save_success')}" + """');
                        fetchServerPresets();
                    } catch (e) {
                        await customAlert('""" + f"{_i18n('preset_node_error_save')}" + """' + e.message);
                    }
                }

                async function deleteServerPreset() {
                    const name = document.getElementById('preset-name').value.trim();
                    if (!name) return customAlert('""" + f"{_i18n('preset_node_err_name_delete')}" + """');

                    if (!(await customConfirm('""" + f"{_i18n('preset_node_confirm_delete')}" + """ "' + name + '"?'))) return;

                    try {
                        const res = await fetch(`/iter_ensemble_preset/${name}`, { method: 'DELETE' });
                        const result = await res.json();

                        if (result.error) throw new Error(result.error);

                        clearEditor();
                        await customAlert('""" + f"{_i18n('preset_node_delete_success')}" + """');
                        fetchServerPresets();
                    } catch (e) {
                        await customAlert('""" + f"{_i18n('preset_node_error_delete')}" + """' + e.message);
                    }
                }

                let stateTimeout;
                window.sendStateToParent = function () {
                    clearTimeout(stateTimeout);
                    stateTimeout = setTimeout(() => {
                        const data = ensembleState;
                        window.parent.postMessage({ type: 'update_iter_ensemble_preset', payload: data }, '*');
                    }, 100);
                };

                setupImport();
            </script>
        </body>
        </html>
        """
































        F0_CORRECTOR_HTML_CONTENT = """<!DOCTYPE html>
        <html lang="ru">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>""" + f"{_i18n('f0_corrector_editor_title')}" + """</title>
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>

        /* -- F0 CORRECTOR --- */
        :root {
            --bg-color: var(--background-fill-primary, #eef1f5);
            --grid-color: var(--border-color-secondary, #d4d8dd);
            --node-bg: var(--background-fill-secondary, #f6f8fa);
            --node-header: var(--block-background-fill, #e6ebf1);
            --border-color: var(--border-color-primary, #d0d7de);
            --text-main: var(--body-text-color, #333333);
            --text-white: #f3f4f6;
            --btn-blue: var(--color-accent, #007bff);
            --btn-red: var(--button-cancel-background-fill, #dc3545);
            --primary: var(--color-accent, #007bff);
            --primary-hover: var(--color-accent-soft, #0056b3);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --bg-color: var(--background-fill-primary, #0b0f19);
                --grid-color: var(--border-color-secondary, #1f2937);
                --node-bg: var(--background-fill-secondary, #1f2937);
                --node-header: var(--block-background-fill, #374151);
                --border-color: var(--border-color-primary, #374151);
                --text-main: var(--body-text-color, #f3f4f6);
            }
        }

        .dark {
            --bg-color: var(--background-fill-primary, #0b0f19);
            --grid-color: var(--border-color-secondary, #1f2937);
            --node-bg: var(--background-fill-secondary, #1f2937);
            --node-header: var(--block-background-fill, #374151);
            --border-color: var(--border-color-primary, #374151);
            --text-main: var(--body-text-color, #f3f4f6);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Montserrat', sans-serif; }
        html, body { width: 100%; height: 100%; }
        body { overflow: hidden; display: flex; flex-direction: column; background: var(--bg-color); color: var(--text-main); height: 100vh; height: 100dvh; }

        /* --- Topbar --- */
        .topbar { min-height: 56px; background: var(--node-bg); border-bottom: 1px solid var(--border-color); display: flex; align-items: center; padding: 8px 16px; gap: 10px; z-index: 100; box-shadow: 0 2px 5px rgba(0,0,0,0.02); flex-shrink: 0; flex-wrap: wrap; }
        input[type="text"] { padding: 7px 12px; border: 1px solid var(--border-color); border-radius: 4px; font-size: 14px; outline: none; background: var(--bg-color); color: var(--text-main); }
        .topbar input[type="text"] { padding: 7px 25px 7px 12px; border: 1px solid var(--border-color); border-radius: 4px; font-size: 14px; outline: none; background: var(--bg-color); color: var(--text-main); }

        /* --- Buttons --- */
        .btn { padding: 8px 16px; border-radius: 4px; border: 1px solid var(--border-color); background: var(--node-bg); color: var(--text-main); cursor: pointer; font-weight: 500; font-size: 14px; transition: background 0.2s, color 0.2s, transform 0.1s, box-shadow 0.2s; white-space: nowrap; }
        .btn:hover { background: var(--grid-color); }
        .btn:active { transform: scale(0.96); }
        .btn-blue { color: var(--btn-blue); border-color: var(--btn-blue); }
        .btn-blue:hover { background: rgba(0, 123, 255, 0.1); }
        .btn-green { color: #28a745; border-color: #28a745; }
        .btn-green:hover { background: rgba(40, 167, 69, 0.1); }
        .btn-undo { background: var(--btn-red); color: var(--text-white); border-color: var(--btn-red); }
        .btn-redo { background: var(--primary); color: var(--text-white); border-color: var(--primary); }

        /* --- Modals --- */
        .modal-overlay {
            position: fixed;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            background: rgba(0,0,0,0.4);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 99999999;
            opacity: 0;
            transition: opacity 0.2s;
            padding: 16px;          /* ← отступ от краёв экрана */
            box-sizing: border-box;
        }
        .modal-overlay.show { display: flex; opacity: 1; }
        .modal-box {
            background: var(--node-bg);
            padding: 25px;
            border-radius: 8px;
            width: 100%;            /* ← было min-width:320px */
            max-width: 420px;       /* ← было max-width:90vw */
            max-height: 90vh;       /* ← новое: ограничение высоты */
            overflow-y: auto;       /* ← новое: скролл если контент не влезает */
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            text-align: center;
            box-sizing: border-box;
        }
        .modal-text { margin-bottom: 25px; font-size: 15px; font-weight: 500; color: var(--text-main); word-wrap: break-word; }
        .modal-buttons { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
        .modal-buttons .btn { min-width: 100px; }

        /* --- Custom Selects --- */
        .custom-select-container { position: relative; box-sizing: border-box; min-width: 0; width: 100%; }
        .custom-select-input-wrapper { position: relative; }
        .custom-select-input { width: 100%; box-sizing: border-box; padding: 7px 30px 7px 10px; border: 1px solid var(--border-color); border-radius: 4px; font-size: 12px; background: var(--node-bg); color: var(--text-main); min-height: 35px; cursor: pointer; white-space: nowrap; transition: border-color 0.25s, box-shadow 0.25s; }
        .custom-select-input:focus { outline: none; border-color: var(--primary); }
        .custom-select-input-wrapper::after { content: '▼'; font-size: 10px; color: var(--text-main); position: absolute; right: 10px; top: 50%; transform: translateY(-50%); pointer-events: none; opacity: 0.5; }
        .custom-select-options { position: absolute; top: 100%; left: 0; width: 100%; background: var(--node-bg); border: 1px solid var(--border-color); border-radius: 4px; margin-top: 4px; max-height: 220px; overflow-y: auto; z-index: 999; display: none; box-shadow: 0 5px 15px rgba(0,0,0,0.15); }
        .custom-select-container.open .custom-select-options { display: block; }
        .custom-select-container.open { z-index: 9999; }
        .custom-option { padding: 7px 36px 7px 12px; cursor: pointer; font-size: 13px; transition: background 0.2s, color 0.2s; white-space: pre-wrap; overflow: hidden; overflow-wrap: anywhere; user-select: none; color: var(--text-main); }
        .custom-option:hover { background: var(--grid-color); color: var(--primary-hover); }
        .custom-option.selected { background: var(--primary); color: #fff; font-weight: 600; }
        .custom-option.disabled { color: #aaa; cursor: not-allowed; background: var(--bg-color); }
        .custom-select-container.flash .custom-select-input {
            border-color: #28a745;
            background: rgba(40,167,69,0.06);
            animation: flash-pulse 1.5s ease-out;
        }
        @keyframes flash-pulse {
            0%   { box-shadow: 0 0 0 0   rgba(40,167,69,0.55); }
            70%  { box-shadow: 0 0 0 6px rgba(40,167,69,0.0);  }
            100% { box-shadow: 0 0 0 3px rgba(40,167,69,0.18); }
        }

        /* --- Toolbar Layout (Обновлено) --- */
        .toolbar { display: flex; flex-direction: column; gap: 8px; padding: 8px 12px; background: var(--node-bg); border-bottom: 1px solid var(--border-color); flex-shrink: 0; }
        .toolbar-row { display: flex; width: 100%; gap: 8px; flex-wrap: wrap; align-items: center; }
        .btn-group { display: flex; gap: 6px; flex-wrap: wrap; }
        .toolbar-export { display: flex; gap: 6px; margin-left: auto; }
        .toolbar .btn { padding: 5px 10px; } 
        .tb-sep { width: 1px; height: 24px; background: var(--border-color); margin: 0 4px; flex-shrink: 0; }
        .spacer { flex: 1; }

        /* --- Tool Buttons --- */
        .tool-btn { padding: 7px 12px; border-radius: 4px; border: 1px solid var(--border-color); background: var(--node-bg); color: var(--text-main); cursor: pointer; font-size: 13px; font-weight: 500; transition: background 0.15s, color 0.15s, box-shadow 0.15s, transform 0.1s; white-space: nowrap; }
        .tool-btn:hover { background: var(--grid-color); }
        .tool-btn:active { transform: scale(0.95); }
        .tool-btn.active-tool { border-color: var(--primary); color: var(--primary); background: rgba(0, 123, 255, 0.08); box-shadow: inset 0 0 0 1px var(--primary), 0 0 10px rgba(0,123,255,0.15); }

        /* --- Main Area & Viewport --- */
        .main-area { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-height: 0; }
        .viewport { flex: 1; position: relative; overflow: hidden; min-height: 0; cursor: grab; touch-action: none; background-color: var(--bg-color); background-image: radial-gradient(var(--grid-color) 1px, transparent 1px); background-size: 20px 20px; }
        .canvas-container { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
        .canvas-container canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; touch-action: none; -webkit-touch-callout: none; -webkit-user-select: none; user-select: none; }

        #placeholder { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; text-align: center; color: var(--text-main); opacity: 0.55; font-size: 15px; pointer-events: none; line-height: 1.9; padding: 20px; z-index: 10; }
        #placeholder small { display: block; font-size: 12px; opacity: 0.75; margin-top: 6px; }
        #loading-overlay { position: absolute; inset: 0; background: rgba(0,0,0,0.5); display: none; align-items: center; justify-content: center; flex-direction: column; gap: 14px; color: #fff; font-size: 14px; z-index: 100; }
        #loading-overlay.show { display: flex; }
        .spinner { width: 38px; height: 38px; border: 4px solid rgba(255,255,255,0.25); border-top-color: #fff; border-radius: 50%; animation: spin 0.8s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* --- Toast / Snackbar: fixed вне overflow-hidden, поэтому не обрезается --- */
        #toast {
            position: fixed; left: 50%; bottom: 22px;
            transform: translateX(-50%) translateY(calc(100% + 32px));
            display: flex; align-items: center; gap: 11px;
            max-width: min(92vw, 540px);
            padding: 11px 16px 14px;
            background: var(--node-header); color: var(--text-main);
            border: 1px solid var(--border-color); border-left: 4px solid #28a745;
            border-radius: 11px;
            box-shadow: 0 12px 34px rgba(0,0,0,0.30), 0 2px 8px rgba(0,0,0,0.18);
            font-size: 13px; font-weight: 600; letter-spacing: .2px; line-height: 1.35;
            pointer-events: none; opacity: 0; overflow: hidden;
            z-index: 100000; white-space: nowrap;
            transition: transform .42s cubic-bezier(.18,.89,.32,1.28), opacity .3s ease;
        }
        #toast .toast-ico {
            flex: 0 0 auto; width: 21px; height: 21px;
            display: grid; place-items: center; border-radius: 50%;
            background: rgba(40,167,69,0.16); color: #28a745; font-size: 12px;
            transition: background .3s ease, color .3s ease;
        }
        #toast .toast-msg { overflow: hidden; text-overflow: ellipsis; }
        #toast::after {
            content: ""; position: absolute; left: 0; bottom: 0;
            height: 3px; width: 100%;
            background: linear-gradient(90deg, #28a745, #6ee09a);
            transform-origin: left center; transform: scaleX(0);
        }
        #toast.show { transform: translateX(-50%) translateY(0); opacity: 1; }
        #toast.show::after { animation: toast-progress 2.6s linear forwards; }
        @keyframes toast-progress { from { transform: scaleX(1); } to { transform: scaleX(0); } }
        .selection-rect { position: absolute; border: 2px dashed var(--primary); background: rgba(0, 123, 255, 0.1); pointer-events: none; display: none; z-index: 50; }
        .tooltip { position: fixed; background: rgba(0,0,0,0.85); color: #fff; padding: 6px 10px; border-radius: 4px; font-size: 12px; pointer-events: none; display: none; z-index: 10000; white-space: nowrap; line-height: 1.5; }

        /* --- Statusbar --- */
        .statusbar {
            display: flex; align-items: center;
            gap: 4px 14px;
            padding: 6px 14px;
            background: var(--node-bg);
            border-top: 1px solid var(--border-color);
            font-size: 12px; color: var(--text-main);
            flex-wrap: wrap;
            flex-shrink: 0;
            overflow-x: auto; overflow-y: hidden;
            scrollbar-width: thin;
            -webkit-overflow-scrolling: touch;
        }
        /* Логические группы — атомарные: не рвутся и не переносятся по частям */
        .st-group { display: flex; align-items: center; gap: 14px; flex-shrink: 0; }
        .st-group > span { white-space: nowrap; }
        .st-zoom { gap: 6px; }
        .statusbar .spacer { flex: 1 1 16px; min-width: 16px; }
        .zoom-ctl {
            cursor: pointer; padding: 1px 9px; font-weight: 700;
            border: 1px solid var(--border-color); border-radius: 4px;
            user-select: none; transition: background 0.15s;
            display: inline-flex; align-items: center; line-height: 1;
        }
        .zoom-ctl:hover { background: var(--grid-color); }
        #st-zoom { cursor: pointer; min-width: 46px; text-align: center; display: inline-block; }
        /* Табличные цифры + резерв ширины: значения обновляются на лету,
        не меняя ширину — статусбар не дёргается и не переносится ложно */
        .statusbar b { font-variant-numeric: tabular-nums; }
        #st-time { display: inline-block; min-width: 8ch; }
        #st-freq { display: inline-block; min-width: 10ch; }
        #st-f0   { display: inline-block; min-width: 10ch; }
        .sel-hidden { display: none; }
        .zoom-ctl:hover { background: var(--grid-color); }
        #st-zoom { cursor: pointer; min-width: 46px; text-align: center; display: inline-block; }
        .sel-hidden { display: none; }

        /* --- Модалка имени кривой --- */
        .dl-option { display: flex; align-items: center; gap: 10px; padding: 9px 12px; border: 1px solid var(--border-color); border-radius: 6px; margin-bottom: 8px; cursor: pointer; transition: border-color .2s, background .2s, transform .15s; text-align: left; font-size: 13px; color: var(--text-main); background: var(--bg-color); user-select: none; }
        .dl-option:hover { border-color: var(--primary); background: rgba(0, 123, 255, 0.07); transform: translateX(3px); }
        .dl-option input[type="radio"] { accent-color: var(--primary); cursor: pointer; width: 15px; height: 15px; flex-shrink: 0; }
        .dl-option.checked { border-color: var(--primary); box-shadow: inset 0 0 0 1px var(--primary); background: rgba(0, 123, 255, 0.08); }
        #download-name-input[readonly] { opacity: 0.75; cursor: default; }

        /* --- Mobile Responsiveness (Обновлено) --- */
        @media (max-width: 820px) {
            /* ===== MODALS ===== */
            .modal-overlay { padding: 10px; }
            .modal-box {
                min-width: 0 !important;   /* ← переопределяет inline min-width:380px */
                max-width: 100% !important;
                width: 100%;
                padding: 16px;
                max-height: 85vh;
            }
            .modal-text { font-size: 14px; margin-bottom: 16px; }
            .modal-buttons { flex-direction: column; }
            .modal-buttons .btn { width: 100%; min-width: 0; }
            /* Радио-опции на всю ширину */
            .dl-option { font-size: 12px; padding: 8px 10px; }
            #download-name-input { font-size: 13px; }
            /* ===== TOPBAR ===== */
            .topbar { padding: 8px 10px; gap: 6px; }
            .topbar .custom-select-container { width: 100% !important; }
            .topbar .btn { width: 100%; padding: 3px 12px }
            /* ===== TOOLBAR ===== */
            .toolbar {
                overflow: visible;
                max-height: none;
                padding: 6px 8px;
                gap: 8px;
            }
            .toolbar-row { gap: 3px; }
            .tb-sep,
            .toolbar-row .spacer { display: none; }
            .btn-group {
                flex: 1 1 100%;
                flex-wrap: wrap;
                justify-content: center;
                gap: 3px;
            }
            .btn-group .btn {
                flex: 1 1 84px;
                min-width: 0;
                max-width: 100%;
                height: auto;
                padding: 4px 10px;
                font-size: 10px;
                line-height: 1.15;
                white-space: normal;
                overflow-wrap: anywhere;
                text-align: center;
                justify-content: center;
                display: inline-flex;
                align-items: center;
                gap: 4px;
            }
            .btn-group .tool-btn {
                flex: 1 1 84px;
                min-width: 0;
                max-width: 100%;
                height: auto;
                padding: 4px 6px;
                font-size: 11.5px;
                line-height: 1.15;
                white-space: normal;
                overflow-wrap: anywhere;
                text-align: center;
                justify-content: center;
                display: inline-flex;
                align-items: center;
                gap: 4px;
            }
            .toolbar-export {
                flex: 1 1 100%;
                flex-wrap: wrap;
                margin-left: 0;
                justify-content: center;
                gap: 6px;
            }
            .toolbar-export .btn {
                flex: 1 1 45%;
                min-width: 0;
                max-width: 100%;
                height: auto;
                padding: 4px 10px;
                font-size: 11.5px;
                line-height: 1.15;
                white-space: normal;
                overflow-wrap: anywhere;
                justify-content: center;
                display: flex;
                align-items: center;
                gap: 6px;
            }
            /* ===== STATUSBAR ===== */
            .statusbar {
                overflow: hidden;
                flex-wrap: wrap;
                gap: 4px 12px;
                padding: 2px 10px;
                font-size: 10px;
            }
            .statusbar .spacer { display: none; }
            .st-group {
                flex: 0 1 auto;
                flex-wrap: wrap;
                min-width: 0;
                gap: 5px 12px;
            }
            .st-group > span {
                white-space: nowrap;
                max-width: 100%;
            }
            .st-zoom { flex: 0 0 auto; }
        }
        
        </style>
        </head>
        <body>
        <div id="custom-modal-overlay" class="modal-overlay">
        <div class="modal-box">
        <div class="modal-text" id="modal-text"></div>
        <div class="modal-buttons">
        <button class="btn btn-blue" id="modal-btn-ok">""" + f"{_i18n('preset_node_ok')}" + """</button>
        </div>
        </div>
        </div>
        <div id="pitch-shift-modal" class="modal-overlay">
            <div class="modal-box">
                <div class="modal-text" style="font-size: 16px; margin-bottom: 15px;">""" + f"{_i18n('f0_corrector_pitch_shift_title')}" + """</div>
                <div style="margin-bottom: 20px;">
                    <label style="display:block; margin-bottom:8px; font-size:14px; color:var(--text-main); text-align:left;">""" + f"{_i18n('f0_corrector_pitch_shift_label')}" + """</label>
                    <input type="text" id="pitch-shift-input" step="0.001" style="width: 100%; padding: 8px 12px; border: 1px solid var(--border-color); border-radius: 4px; font-size: 14px; background: var(--bg-color); color: var(--text-main); box-sizing: border-box;" placeholder=""" + f"\"{_i18n('f0_corrector_pitch_shift_placeholder')}\"" + """>
                </div>
                <div class="modal-buttons">
                    <button class="btn btn-blue" id="pitch-shift-apply">""" + f"{_i18n('f0_corrector_pitch_shift_apply')}" + """</button>
                    <button class="btn btn-red" id="pitch-shift-cancel">""" + f"{_i18n('preset_node_cancel')}" + """</button>
                </div>
            </div>
        </div>
        <div id="download-modal" class="modal-overlay">
            <div class="modal-box" style="min-width: 380px;">
                <div class="modal-text" style="font-size: 16px; margin-bottom: 15px;">""" + f"{_i18n('f0_corrector_download_title')}" + """</div>
                <div style="margin-bottom: 14px;">
                    <label class="dl-option checked" id="dl-opt-default">
                        <input type="radio" name="dl-name-variant" value="default" checked>
                        <span>""" + f"{_i18n('f0_corrector_name_default')}" + """</span>
                    </label>
                    <label class="dl-option" id="dl-opt-source">
                        <input type="radio" name="dl-name-variant" value="source">
                        <span>""" + f"{_i18n('f0_corrector_name_source')}" + """</span>
                    </label>
                    <label class="dl-option" id="dl-opt-custom">
                        <input type="radio" name="dl-name-variant" value="custom">
                        <span>""" + f"{_i18n('f0_corrector_name_custom')}" + """</span>
                    </label>
                </div>
                <div style="margin-bottom: 20px;">
                    <label style="display:block; margin-bottom:8px; font-size:13px; color:var(--text-main); text-align:left;">""" + f"{_i18n('f0_corrector_curve_name')}" + """</label>
                    <input type="text" id="download-name-input" style="width: 100%; padding: 8px 12px; border: 1px solid var(--border-color); border-radius: 4px; font-size: 14px; background: var(--bg-color); color: var(--text-main); box-sizing: border-box;" readonly>
                </div>
                <div class="modal-buttons">
                    <button class="btn btn-blue" id="download-apply">⬇ """ + f"{_i18n('f0_corrector_download_json')}" + """</button>
                    <button class="btn btn-red" id="download-cancel">""" + f"{_i18n('preset_node_cancel')}" + """</button>
                </div>
            </div>
        </div>
        <div class="topbar">
        <div class="custom-select-container filterable" id="audio-select-container" data-value="" style="width: 280px;" onmouseenter="fetchCorrectorFiles()">
        <div class="custom-select-input-wrapper">
        <input type="text" class="custom-select-input" placeholder=""" + f"\"{_i18n('f0_corrector_audio_file')}\"" + """ onfocus="openCorrectorDropdown(this)" onclick="openCorrectorDropdown(this)" oninput="filterCorrectorOptions(this)" autocomplete="off">
        </div>
        <div class="custom-select-options"><div class="custom-option disabled">""" + f"{_i18n('preset_node_loading')}" + """</div></div>
        </div>
        <div class="custom-select-container filterable" id="f0-select-container" data-value="" style="width: 280px;" onmouseenter="fetchCorrectorFiles()">
        <div class="custom-select-input-wrapper">
        <input type="text" class="custom-select-input" placeholder=""" + f"\"{_i18n('f0_corrector_f0_curve')}\"" + """ onfocus="openCorrectorDropdown(this)" onclick="openCorrectorDropdown(this)" oninput="filterCorrectorOptions(this)" autocomplete="off">
        </div>
        <div class="custom-select-options"><div class="custom-option disabled">""" + f"{_i18n('preset_node_loading')}" + """</div></div>
        </div>
        <button class="btn btn-blue" onclick="analyze()">▶ """ + f"{_i18n('f0_corrector_analyze')}" + """</button>
        </div>
        <div class="main-area">
        <div class="toolbar">
            <!-- ВЕРХНИЙ РЯД: Навигация, выделение и экспорт -->
            <div class="toolbar-row">
                <!-- Группа 1: История и навигация -->
                <div class="btn-group">
                    <button class="btn btn-undo" onclick="undo()" title="Ctrl+Z">""" + f"{_i18n('f0_corrector_undo')}" + """</button>
                    <button class="btn btn-redo" onclick="redo()" title="Ctrl+Y">""" + f"{_i18n('f0_corrector_redo')}" + """</button>
                    <button class="btn" onclick="resetF0()">""" + f"{_i18n('f0_corrector_reset_f0')}" + """</button>
                </div>
                
                <span class="tb-sep"></span>

                <!-- Группа 2: Работа с кривой -->
                <div class="btn-group">
                    <button class="btn" onclick="deselectAll()">""" + f"{_i18n('f0_corrector_deselect')}" + """</button>
                    <button class="btn" onclick="resetView()">⛶ """ + f"{_i18n('f0_corrector_fit_view')}" + """</button>
                </div>
                
                <span class="spacer"></span>
                
                <!-- Группа 3: Экспорт -->
                <div class="toolbar-export">
                    <button class="btn btn-blue" onclick="downloadJSON()">⬇ """ + f"{_i18n('f0_corrector_download_json')}" + """</button>
                    <button class="btn btn-green" onclick="sendToInference()">➜ """ + f"{_i18n('f0_corrector_send_to_inference')}" + """</button>
                </div>
            </div>

            <!-- НИЖНИЙ РЯД: Инструменты редактирования -->
            <div class="toolbar-row">
                <!-- Группа 4: Кисти и курсоры -->
                <div class="btn-group">
                    <button class="tool-btn active-tool" id="btn-pan" onclick="setTool('pan')">""" + f"{_i18n('f0_corrector_tool_pan')}" + """</button>
                    <button class="tool-btn" id="btn-cursor" onclick="setTool('cursor')">""" + f"{_i18n('f0_corrector_tool_cursor')}" + """</button>
                    <button class="tool-btn" id="btn-select" onclick="setTool('select')">""" + f"{_i18n('f0_corrector_tool_select')}" + """</button>
                    <button class="tool-btn" id="btn-pencil" onclick="setTool('pencil')">""" + f"{_i18n('f0_corrector_tool_pencil')}" + """</button>
                    <button class="tool-btn" id="btn-eraser" onclick="setTool('eraser')">""" + f"{_i18n('f0_corrector_tool_eraser')}" + """</button>
                </div>

                <span class="tb-sep"></span>

                <div class="btn-group">
                    <button class="btn" onclick="shiftPitch()">""" + f"{_i18n('f0_corrector_pitch_shift_title')}" + """</button>
                    <button class="btn" onclick="smoothF0()">""" + f"{_i18n('f0_corrector_smooth')}" + """</button>
                    <button class="btn" onclick="autoFixOctaveJumps()">""" + f"{_i18n('f0_corrector_auto_fix_octave')}" + """</button>
                </div>
            </div>
        </div>
        <div class="viewport" id="viewport">
        <div class="canvas-container" id="canvasContainer">
        <canvas id="specCanvas"></canvas>
        <canvas id="gridCanvas"></canvas>
        <canvas id="f0Canvas"></canvas>
        </div>
        <div id="placeholder">🎼<br>""" + f"{_i18n('f0_corrector_placeholder')}" + """<small style="white-space: pre-wrap;">          1–5 — """ + f"{_i18n('f0_corrector_tools_hotkeys')}" + """</small></div>
        <div id="loading-overlay"><div class="spinner"></div><div>""" + f"{_i18n('f0_corrector_analyzing')}" + """</div></div>
        </div>
        <div class="statusbar">
            <div class="st-group">
                <span>""" + f"{_i18n('f0_corrector_points')}" + """: <b id="st-points">0</b></span>
                <span>""" + f"{_i18n('f0_corrector_sample_rate')}" + """: <b id="st-sr">0</b></span>
                <span>""" + f"{_i18n('f0_corrector_size')}" + """: <b id="st-size">0×0</b></span>
                <span>""" + f"{_i18n('f0_corrector_method')}" + """: <b id="st-method">—</b></span>
            </div>
            <div class="st-group st-zoom">
                <span class="zoom-ctl" onclick="changeZoom(-0.5)">−</span>
                <b id="st-zoom" onclick="resetView()">231%</b>
                <span class="zoom-ctl" onclick="changeZoom(0.5)">+</span>
            </div>
            <span class="spacer"></span>
            <div class="st-group">
                <span>""" + f"{_i18n('f0_corrector_time')}" + """: <b id="st-time">—</b></span>
                <span>""" + f"{_i18n('f0_corrector_freq')}" + """: <b id="st-freq">—</b></span>
                <span>""" + f"{_i18n('f0_corrector_curve_f0')}" + """: <b id="st-f0">—</b></span>
                <span id="st-sel-wrap" class="sel-hidden">""" + f"{_i18n('f0_corrector_selected_frames')}" + """: <b id="st-sel">0</b></span>
            </div>
        </div>
        <div id="toast"><span class="toast-ico">✓</span><span class="toast-msg"></span></div>
        <div class="tooltip" id="tooltip"></div>
        <script>
        // ==================== THEME ====================
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.get('__theme') === 'dark') { document.documentElement.classList.add('dark'); document.body.classList.add('dark'); }
        window.addEventListener('message', function(event) {
        let themeData = event.data;
        if (typeof event.data === 'object' && event.data !== null) themeData = event.data.theme || event.data.type;
        if (themeData === 'theme_dark' || themeData === 'dark') { document.documentElement.classList.add('dark'); document.body.classList.add('dark'); }
        else if (themeData === 'theme_light' || themeData === 'light') { document.documentElement.classList.remove('dark'); document.body.classList.remove('dark'); }
        });
        // ==================== SESSION (multi-user) ====================
        const _sessionParams = new URLSearchParams(window.location.search);
        let SESSION_HASH = _sessionParams.get('session_hash') || '';
        function apiGet(url) {
        const sep = url.includes('?') ? '&' : '?';
        return fetch(url + sep + 'session_hash=' + encodeURIComponent(SESSION_HASH || ''));
        }
        // Если hash ещё не известен — запрашиваем у родителя (head-скрипт ответит)
        if (!SESSION_HASH) { try { window.parent.postMessage({ type: 'request_session_hash' }, '*'); } catch(e) {} }
        window.addEventListener('message', (ev) => {
            if (ev.data && ev.data.type === 'gradio_session_hash' && ev.data.hash && !SESSION_HASH) {
                SESSION_HASH = ev.data.hash;
                fetchCorrectorFiles(); pollInbox(3);
            }
            // Сигнал от родителя: «таб показали / кнопка нажата» — читаем inbox сами
            if (ev.data && ev.data.type === 'f0_corrector_inbox_check') {
                fetchCorrectorFiles(); pollInbox(3);
                if (editorInitialized && !_renderedOk) updateTransform();   // мягкая починка рендера
            }
        });
        function onCorrectorBecameVisible() {
            fetchCorrectorFiles();
            pollInbox(3);
        }
        // ==================== UI HELPERS ====================
        function customAlert(msg) {
        return new Promise(resolve => {
        const overlay = document.getElementById('custom-modal-overlay');
        document.getElementById('modal-text').innerText = msg;
        document.getElementById('modal-btn-ok').onclick = () => { overlay.classList.remove('show'); resolve(); };
        overlay.classList.add('show');
        });
        }
        let toastTimer;
        function showToast(msg) {
            const t = document.getElementById('toast');
            const msgEl = t.querySelector('.toast-msg');
            const icoEl = t.querySelector('.toast-ico');
            if (msgEl) msgEl.textContent = msg;
            // ⬇ в начале строки = скачивание файла, иначе = успех/инфо
            if (icoEl) {
                const isDownload = typeof msg === 'string' && msg.indexOf('⬇') === 0;
                icoEl.textContent = isDownload ? '⬇' : '✓';
                icoEl.style.background = isDownload ? 'rgba(0,123,255,0.16)' : 'rgba(40,167,69,0.16)';
                icoEl.style.color = isDownload ? 'var(--primary)' : '#28a745';
            }
            // reflow-трюк: гарантированно перезапускает анимацию прогресс-полосы
            t.classList.remove('show');
            void t.offsetWidth;
            t.classList.add('show');
            clearTimeout(toastTimer);
            toastTimer = setTimeout(() => t.classList.remove('show'), 2600);
        }
        function showTooltipMessage(msg) {
        const tooltip = document.getElementById('tooltip');
        tooltip.innerHTML = msg;
        tooltip.style.display = 'block';
        tooltip.style.left = '50%';
        tooltip.style.top = '20px';
        tooltip.style.transform = 'translateX(-50%)';
        setTimeout(() => { tooltip.style.display = 'none'; tooltip.style.transform = 'none'; }, 2000);
        }
        // ==================== MEL / COORDS ====================
        const f_sp = 200 / 3;
        const min_log_hz = 1000.0;
        const min_log_mel = min_log_hz / f_sp;
        const logstep = Math.log(6.4) / 27.0;
        function hzToMel(hz) {
        if (hz >= min_log_hz) return min_log_mel + Math.log(hz / min_log_hz) / logstep;
        return hz / f_sp;
        }
        function melToHz(mel) {
        if (mel >= min_log_mel) return min_log_hz * Math.exp(logstep * (mel - min_log_mel));
        return mel * f_sp;
        }
        function getMaxMel() { return hzToMel(f0Data.sample_rate / 2); }
        function freqToY(freq, height) {
        if (freq <= 0) return height;
        return height - (hzToMel(freq) / getMaxMel()) * height;
        }
        function yToFreq(y, height) {
        return melToHz(((height - y) / height) * getMaxMel());
        }
        // ==================== STATE ====================
        const isTouchDevice = ('ontouchstart' in window) || navigator.maxTouchPoints > 0;
        let currentTool = 'pan';
        let historyStack = [];
        let historyIndex = -1;
        const MAX_HISTORY = 50;
        let justFinishedSelecting = false;
        let selectedIndices = new Set();
        let isSelecting = false;
        let selectionStart = { x: 0, y: 0 };
        let selectionRectElem;
        let isPinching = false;
        let pinchBaseline = null;   // { dist, scale, anchorX, anchorY } — базовая линия жеста
        let pinchLastTouch = null;  // опорная точка fallback-пана одним пальцем
        let isDraggingGroup = false;
        let initialFrequencies = new Map();
        let dragStartY = 0;
        let f0Data = {
        times: [], freqs: [], originalFreqs: [],
        sample_rate: 16000, original_sample_rate: 16000,
        window: 160, method: 'rmvpe+',
        n_mels: 256, duration: 0, spec_width: 0, spec_height: 0
        };
        let specCanvas, gridCanvas, f0Canvas;
        let specCtx, gridCtx, f0Ctx;
        let spectrogramImage = new Image();
        let points = [];
        let selectedPoint = null;
        let isDragging = false;
        let isPanning = false;
        let lastX, lastY;
        let lastProcessedIndex = null;
        let animationFrameId = null;
        let scale = 1, panX = 0, panY = 0;
        let editorInitialized = false;
        let _renderedOk = false;     // спектрограмма уже успешно отрисована в canvas ненулевого размера
        let _needsRelayout = false;  // onload прошёл «в фоне» — при появлении вкладки нужен resetView
        const viewport = document.getElementById('viewport');
        const canvasContainer = document.getElementById('canvasContainer');
        // ==================== RENDERING ====================
        function updateTransform() {
            const rect = viewport.getBoundingClientRect();
            const w = Math.round(rect.width);
            const h = Math.round(rect.height);

            // Вкладка скрыта (неактивный таб Gradio → display:none → viewport 0×0).
            // НЕ создаём canvas 0×0 (это ломает контекст) и НЕ рисуем в пустоту.
            // Помечаем relayout только если картинка готова, но нормально ещё не рисовали.
            if (w <= 0 || h <= 0) {
                if (editorInitialized && !_renderedOk) _needsRelayout = true;
                return;
            }

            // Вернулись на вкладку после фонового анализа — пересчитать scale/pan под реальный размер.
            if (_needsRelayout) {
                _needsRelayout = false;
                if (spectrogramImage && spectrogramImage.complete && spectrogramImage.naturalWidth > 0) {
                    resetView();   // resetView сам вызовет updateTransform уже без флага
                    return;
                }
            }

            if (specCanvas.width !== w || specCanvas.height !== h) {
                specCanvas.width = w; specCanvas.height = h;
                gridCanvas.width = w; gridCanvas.height = h;
                f0Canvas.width = w; f0Canvas.height = h;
                renderGrid();
            }
            if (specCtx && spectrogramImage.complete && spectrogramImage.naturalWidth > 0) {
                specCtx.clearRect(0, 0, w, h);
                specCtx.save();
                specCtx.translate(panX, panY);
                specCtx.scale(scale, scale);
                specCtx.imageSmoothingEnabled = scale >= 4;
                specCtx.drawImage(spectrogramImage, 0, 0);
                specCtx.restore();
            }
            renderGrid();
            renderF0();
            _renderedOk = true;   // зафиксировали, что рендер прошёл при валидном размере
        }
        function updatePoints() {
        const baseWidth = spectrogramImage.naturalWidth || f0Data.spec_width;
        const baseHeight = spectrogramImage.naturalHeight || f0Data.spec_height;
        if (!baseWidth || !baseHeight || !f0Data.freqs.length) return;
        points = f0Data.freqs.map((freq, i) => {
        const x = f0Data.freqs.length > 1 ? (i / (f0Data.freqs.length - 1)) * baseWidth : 0;
        const y = freqToY(freq, baseHeight);
        return { x, y, freq, index: i };
        });
        }
        function renderGrid() {
        if (!gridCtx) return;
        const w = gridCanvas.width, h = gridCanvas.height;
        gridCtx.clearRect(0, 0, w, h);
        if (!spectrogramImage.complete || !spectrogramImage.naturalWidth) return;
        gridCtx.save();
        gridCtx.translate(panX, panY);
        gridCtx.scale(scale, scale);
        const viewLeft = -panX / scale;
        const viewRight = viewLeft + (w / scale);
        const viewHeight = spectrogramImage.naturalHeight;
        const viewWidth = spectrogramImage.naturalWidth;
        gridCtx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        gridCtx.lineWidth = 1 / scale;
        gridCtx.setLineDash([5 / scale, 5 / scale]);
        gridCtx.beginPath();
        let timeStep = scale > 4 ? 0.1 : (scale > 2 ? 0.5 : (scale < 0.5 ? 5 : 1));
        if (f0Data.duration) {
        for (let sec = 0; sec <= f0Data.duration; sec += timeStep) {
        const x = (sec / f0Data.duration) * viewWidth;
        if (x >= viewLeft - 10 && x <= viewRight + 10) {
        gridCtx.moveTo(x, 0); gridCtx.lineTo(x, viewHeight);
        }
        }
        }
        const freqStep = scale > 2 ? 250 : 500;
        for (let freq = freqStep; freq <= 8000; freq += freqStep) {
        const y = freqToY(freq, viewHeight);
        if (y >= 0 && y <= viewHeight) {
        gridCtx.moveTo(0, y); gridCtx.lineTo(viewWidth, y);
        }
        }
        gridCtx.stroke();
        gridCtx.setLineDash([]);
        gridCtx.font = (12 / scale) + 'px monospace';
        gridCtx.lineJoin = 'round';
        const strokeWidth = 3 / scale;
        if (f0Data.duration) {
        for (let sec = 0; sec <= f0Data.duration; sec += timeStep) {
        const x = (sec / f0Data.duration) * viewWidth;
        if (x >= viewLeft - 10 && x <= viewRight + 10) {
        const label = sec.toFixed(timeStep < 1 ? 1 : 0) + 's';
        const tx = x + (4 / scale);
        const ty = Math.max(15 / scale, ((-panY + 15) / scale));
        gridCtx.strokeStyle = 'black'; gridCtx.lineWidth = strokeWidth;
        gridCtx.strokeText(label, tx, ty);
        gridCtx.fillStyle = 'white'; gridCtx.fillText(label, tx, ty);
        }
        }
        }
        for (let freq = freqStep; freq <= 8000; freq += freqStep) {
        const y = freqToY(freq, viewHeight);
        if (y >= 0 && y <= viewHeight) {
        const freqLabel = freq + ' Hz';
        const fx = Math.max(5 / scale, (-panX + 5) / scale);
        const fy = y - 2 / scale;
        gridCtx.strokeStyle = 'black'; gridCtx.lineWidth = strokeWidth;
        gridCtx.strokeText(freqLabel, fx, fy);
        gridCtx.fillStyle = 'white'; gridCtx.fillText(freqLabel, fx, fy);
        }
        }
        gridCtx.restore();
        }
        function renderF0() {
        if (!f0Ctx) return;
        const w = f0Canvas.width, h = f0Canvas.height;
        f0Ctx.clearRect(0, 0, w, h);
        if (!points.length) return;
        f0Ctx.save();
        f0Ctx.translate(panX, panY);
        f0Ctx.scale(scale, scale);
        const viewLeft = -panX / scale;
        const viewRight = viewLeft + (w / scale);
        // Curve shadow
        f0Ctx.lineJoin = 'round'; f0Ctx.lineCap = 'round';
        f0Ctx.strokeStyle = 'rgba(0,0,0,0.55)';
        f0Ctx.lineWidth = 3.5 / scale;
        strokeF0Path(viewLeft, viewRight);
        // Curve main
        f0Ctx.strokeStyle = '#00e5ff';
        f0Ctx.lineWidth = 1.8 / scale;
        strokeF0Path(viewLeft, viewRight);
        // Points
        if (scale > 0.5) {
        for (let i = 0; i < points.length; i++) {
        const p = points[i];
        if (p.x < viewLeft - 10) continue;
        if (p.x > viewRight + 10) break;
        if (p.freq <= 0) continue;
        const isSelected = (selectedPoint && p.index === selectedPoint.index) || selectedIndices.has(p.index);
        if (isSelected || scale > 1.5) {
        f0Ctx.beginPath();
        f0Ctx.arc(p.x, p.y, (isSelected ? 5 : 2.5) / scale, 0, Math.PI * 2);
        f0Ctx.fillStyle = isSelected ? '#ff0055' : '#00d4ff';
        f0Ctx.fill();
        f0Ctx.strokeStyle = '#fff';
        f0Ctx.lineWidth = 1 / scale;
        f0Ctx.stroke();
        }
        }
        }
        f0Ctx.restore();
        updateSelStatus();
        }
        function strokeF0Path(viewLeft, viewRight) {
        f0Ctx.beginPath();
        let prev = false;
        for (let i = 0; i < points.length; i++) {
        const p = points[i];
        if (p.x < viewLeft - 50) continue;
        if (p.x > viewRight + 50) break;
        if (p.freq > 0) {
        if (prev) f0Ctx.lineTo(p.x, p.y); else f0Ctx.moveTo(p.x, p.y);
        prev = true;
        } else prev = false;
        }
        f0Ctx.stroke();
        }
        // ==================== ZOOM / VIEW ====================
        function changeZoom(delta) {
        const oldScale = scale;
        scale = Math.max(0.5, Math.min(15, scale + delta));
        if (scale !== oldScale) {
        const rect = viewport.getBoundingClientRect();
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        panX = centerX - (centerX - panX) * (scale / oldScale);
        panY = centerY - (centerY - panY) * (scale / oldScale);
        updateTransform();
        document.getElementById('st-zoom').textContent = Math.round(scale * 100) + '%';
        }
        }
        function resetView() {
            if (!spectrogramImage.complete || !spectrogramImage.naturalWidth) return;
            const rect = viewport.getBoundingClientRect();
            const w = rect.width, h = rect.height;
            if (w <= 0 || h <= 0) { _needsRelayout = true; return; }   // <-- новая защита
            const imgW = spectrogramImage.naturalWidth, imgH = spectrogramImage.naturalHeight;
            scale = h / imgH;
            const scaledW = imgW * scale;
            panX = scaledW <= w ? (w - scaledW) / 2 : 0;
            panY = 0;
            updateTransform();
            document.getElementById('st-zoom').textContent = Math.round(scale * 100) + '%';
        }
        // ==================== HISTORY ====================
        let historyDirty = false;

        function markDirty() {
            historyDirty = true;
        }

        function saveState() {
            if (!historyDirty && historyIndex >= 0 && historyStack.length > 0) {
                return;
            }

            if (historyIndex < historyStack.length - 1) {
                historyStack = historyStack.slice(0, historyIndex + 1);
            }

            historyStack.push([...f0Data.freqs]);

            if (historyStack.length > MAX_HISTORY) {
                historyStack.shift();
            }

            historyIndex = historyStack.length - 1;
            historyDirty = false;
        }

        function undo() {
            if (historyIndex > 0) {
                historyIndex--;
                f0Data.freqs = [...historyStack[historyIndex]];
                historyDirty = false;
                syncAndRender();
            }
        }

        function redo() {
            if (historyIndex < historyStack.length - 1) {
                historyIndex++;
                f0Data.freqs = [...historyStack[historyIndex]];
                historyDirty = false;
                syncAndRender();
            }
        }

        function syncAndRender() {
            updatePoints();
            renderF0();
            updateStatusCounts();
        }
        // ==================== TOOLS ====================
        function setTool(t) {
        currentTool = t;
        document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active-tool'));
        const el = document.getElementById('btn-' + t);
        if (el) el.classList.add('active-tool');
        if (t === 'pencil') viewport.style.cursor = 'crosshair';
        else if (t === 'eraser') viewport.style.cursor = 'not-allowed';
        else if (t === 'pan') viewport.style.cursor = 'grab';
        else if (t === 'select') viewport.style.cursor = 'cell';
        else viewport.style.cursor = 'default';
        }
        function findNearestPoint(x, y, threshold) {
        if (threshold === undefined) threshold = 10;
        let nearest = null; let minDist = threshold / scale;
        for (let i = 0; i < points.length; i++) {
        const dist = Math.sqrt((points[i].x - x) * (points[i].x - x) + (points[i].y - y) * (points[i].y - y));
        if (dist < minDist) { minDist = dist; nearest = points[i]; }
        }
        return nearest;
        }
        function applyTool(canvasX, canvasY) {
        const totalPoints = f0Data.freqs.length;
        const baseWidth = spectrogramImage.naturalWidth || f0Data.spec_width;
        const baseHeight = spectrogramImage.naturalHeight || f0Data.spec_height;
        if (!baseWidth || !baseHeight || !totalPoints) return;
        const currentIndex = Math.round((canvasX / baseWidth) * (totalPoints - 1));
        if (currentIndex >= 0 && currentIndex < totalPoints) {
        const newFreq = currentTool === 'pencil'
        ? Math.max(0, Math.min(yToFreq(canvasY, baseHeight), f0Data.original_sample_rate / 2))
        : 0;
        if (lastProcessedIndex !== null && Math.abs(lastProcessedIndex - currentIndex) > 1) {
        const start = Math.min(lastProcessedIndex, currentIndex);
        const end = Math.max(lastProcessedIndex, currentIndex);
        const startFreq = f0Data.freqs[lastProcessedIndex];
        for (let i = start; i <= end; i++) {
        const t = (currentIndex === lastProcessedIndex) ? 0 : (i - lastProcessedIndex) / (currentIndex - lastProcessedIndex);
        f0Data.freqs[i] = currentTool === 'pencil'
        ? Math.max(0, Math.min(startFreq + t * (newFreq - startFreq), f0Data.original_sample_rate / 2))
        : 0;
        }
        } else {
        f0Data.freqs[currentIndex] = newFreq;
        }
        lastProcessedIndex = currentIndex;
        markDirty();
        updatePoints();
        renderF0();
        }
        }
        // ==================== INTERACTIONS ====================
        function setupInteractions() {
        const tooltip = document.getElementById('tooltip');
        if (!selectionRectElem) {
        selectionRectElem = document.createElement('div');
        selectionRectElem.className = 'selection-rect';
        viewport.appendChild(selectionRectElem);
        }
        function getCoords(e) {
        let clientX, clientY;
        if (e.touches && e.touches.length > 0) {
        clientX = e.touches[0].clientX; clientY = e.touches[0].clientY;
        } else {
        clientX = e.clientX; clientY = e.clientY;
        }
        const rect = f0Canvas.getBoundingClientRect();
        const screenX = clientX - rect.left;
        const screenY = clientY - rect.top;
        return {
        x: (screenX - panX) / scale,
        y: (screenY - panY) / scale,
        screenX: clientX,
        screenY: clientY
        };
        }
        function getPinchDist(e) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        return Math.sqrt(dx * dx + dy * dy);
        }
        const onDown = (e) => {
        if (e.type === 'mousedown' && e.button !== 0) return;
        if (e.touches && e.touches.length === 2) {
        if (e.cancelable) e.preventDefault();
        isPinching = true; isPanning = false; isDragging = false; isSelecting = false;
        if (selectionRectElem) selectionRectElem.style.display = 'none';
        const rect = viewport.getBoundingClientRect();
        const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2 - rect.left;
        const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2 - rect.top;
        pinchBaseline = {
        dist: getPinchDist(e),
        scale: scale,
        // Точка мира под центром жеста — всё время будет оставаться под пальцами
        anchorX: (midX - panX) / scale,
        anchorY: (midY - panY) / scale
        };
        pinchLastTouch = null;
        return;
        }
        isPinching = false;
        isPinching = false;
        const coords = getCoords(e);
        justFinishedSelecting = false;
        if (e.cancelable) e.preventDefault();
        if (currentTool === 'select') {
        isSelecting = true;
        selectionStart = { x: coords.screenX, y: coords.screenY };
        selectionRectElem.style.display = 'block';
        const vr = viewport.getBoundingClientRect();
        selectionRectElem.style.left = (coords.screenX - vr.left) + 'px';
        selectionRectElem.style.top = (coords.screenY - vr.top) + 'px';
        selectionRectElem.style.width = '0px'; selectionRectElem.style.height = '0px';
        return;
        }
        if (currentTool === 'pan') {
        isPanning = true; lastX = coords.screenX; lastY = coords.screenY;
        viewport.style.cursor = 'grabbing'; return;
        }
        if (currentTool === 'pencil' || currentTool === 'eraser') {
        isDragging = true; lastProcessedIndex = null;
        saveState();
        applyTool(coords.x, coords.y); return;
        }
        if (currentTool === 'cursor') {
        const point = findNearestPoint(coords.x, coords.y);
        if (point) {
        isDragging = true; dragStartY = coords.y;
        if (selectedIndices.has(point.index)) {
        isDraggingGroup = true; initialFrequencies.clear();
        selectedIndices.forEach(idx => initialFrequencies.set(idx, f0Data.freqs[idx]));
        } else {
        isDraggingGroup = false; selectedIndices.clear(); selectedPoint = point;
        initialFrequencies.clear(); initialFrequencies.set(point.index, f0Data.freqs[point.index]);
        }
        renderF0();
        } else {
        selectedIndices.clear(); renderF0();
        }
        }
        };
        const onMove = (e) => {
        if (e.touches && e.touches.length === 2) {
        isPinching = true;
        if (e.cancelable) e.preventDefault();
        pinchLastTouch = null;
        const base = pinchBaseline;
        if (base && base.dist > 0) {
        const newScale = Math.max(0.5, Math.min(15, base.scale * (getPinchDist(e) / base.dist)));
        const rect = viewport.getBoundingClientRect();
        const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2 - rect.left;
        const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2 - rect.top;
        scale = newScale;
        panX = midX - base.anchorX * newScale;
        panY = midY - base.anchorY * newScale;
        updateTransform();
        document.getElementById('st-zoom').textContent = Math.round(scale * 100) + '%';
        }
        return;
        }
        if (isPinching && e.touches && e.touches.length === 1) {
        // Один палец остался после щипка — продолжаем как пан без рывка
        if (e.cancelable) e.preventDefault();
        const t = e.touches[0];
        if (pinchLastTouch) {
        panX += t.clientX - pinchLastTouch.x;
        panY += t.clientY - pinchLastTouch.y;
        updateTransform();
        }
        pinchLastTouch = { x: t.clientX, y: t.clientY };
        return;
        }
        if (isPinching) return;
        const coords = getCoords(e);
        if ((isDragging || isSelecting) && currentTool !== 'pan') {
        if (e.cancelable) e.preventDefault();
        }
        if (animationFrameId) cancelAnimationFrame(animationFrameId);
        animationFrameId = requestAnimationFrame(() => {
        if (isDragging && (currentTool === 'pencil' || currentTool === 'eraser')) {
        applyTool(coords.x, coords.y); return;
        }
        if (isSelecting) {
        const left = Math.min(selectionStart.x, coords.screenX);
        const top = Math.min(selectionStart.y, coords.screenY);
        const width = Math.abs(selectionStart.x - coords.screenX);
        const height = Math.abs(selectionStart.y - coords.screenY);
        const viewRect = viewport.getBoundingClientRect();
        selectionRectElem.style.left = (left - viewRect.left) + 'px';
        selectionRectElem.style.top = (top - viewRect.top) + 'px';
        selectionRectElem.style.width = width + 'px';
        selectionRectElem.style.height = height + 'px';
        const rect = f0Canvas.getBoundingClientRect();
        selectedIndices.clear();
        points.forEach((p, i) => {
        const sx = p.x * scale + panX + rect.left;
        const sy = p.y * scale + panY + rect.top;
        if (sx >= left && sx <= left + width && sy >= top && sy <= top + height) {
        selectedIndices.add(i);
        }
        });
        renderF0();
        }
        if (isDragging && (selectedPoint || isDraggingGroup)) {
            const baseHeight = spectrogramImage.naturalHeight || f0Data.spec_height;
            const fStart = yToFreq(dragStartY, baseHeight);
            const fCurrent = yToFreq(coords.y, baseHeight);

            if (fStart > 0 && fCurrent > 0) {
                const ratio = fCurrent / fStart;

                initialFrequencies.forEach((startFreq, idx) => {
                    if (startFreq > 0) {
                        f0Data.freqs[idx] = Math.min(
                            f0Data.original_sample_rate / 2,
                            Math.max(0, parseFloat((startFreq * ratio).toFixed(2)))
                        );
                    }
                });

                markDirty();
            }

            updatePoints();
            renderF0();
            return;
        } else if (isPanning) {
        panX += coords.screenX - lastX; panY += coords.screenY - lastY;
        lastX = coords.screenX; lastY = coords.screenY;
        updateTransform();
        }
        });
        // Hover status
        updateHoverStatus(coords);
        // Tooltip
        const point = findNearestPoint(coords.x, coords.y, 15);
        if (point && point.freq > 0) {
        tooltip.style.display = 'block';
        tooltip.style.left = (coords.screenX + 10) + 'px';
        tooltip.style.top = (coords.screenY - 30) + 'px';
        tooltip.innerHTML = point.freq.toFixed(1) + ' Hz<br>' + (f0Data.times[point.index] !== undefined ? f0Data.times[point.index].toFixed(3) : '—') + 's';
        } else {
        tooltip.style.display = 'none';
        }
        };
        const onUp = (e) => {
        if (isPinching) {
        if (e.touches && e.touches.length > 0) {
        // Жест продолжается оставшимся пальцем — переключаемся на пан
        const t = e.touches[0];
        pinchLastTouch = { x: t.clientX, y: t.clientY };
        pinchBaseline = null;
        return;
        }
        isPinching = false; pinchBaseline = null; pinchLastTouch = null;
        isDragging = false; isSelecting = false; isPanning = false;
        viewport.style.cursor = currentTool === 'pan' ? 'grab' : (currentTool === 'select' ? 'cell' : (currentTool === 'pencil' ? 'crosshair' : (currentTool === 'eraser' ? 'not-allowed' : 'default')));
        return;
        }
        if (e.touches && e.touches.length > 0) return;
        if (isSelecting) {
        isSelecting = false; selectionRectElem.style.display = 'none';
        justFinishedSelecting = true; setTimeout(() => { justFinishedSelecting = false; }, 50);
        }
        if (isDragging && (currentTool === 'pencil' || currentTool === 'eraser' || selectedPoint || isDraggingGroup)) {
        saveState(); updateStatusCounts();
        }
        isDragging = false; isDraggingGroup = false; isPanning = false;
        selectedPoint = null; initialFrequencies.clear();
        viewport.style.cursor = currentTool === 'pan' ? 'grab' : (currentTool === 'select' ? 'cell' : (currentTool === 'pencil' ? 'crosshair' : (currentTool === 'eraser' ? 'not-allowed' : 'default')));
        };
        viewport.addEventListener('mousedown', onDown);
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onUp);
        viewport.addEventListener('touchstart', onDown, { passive: false });
        window.addEventListener('touchmove', onMove, { passive: false });
        window.addEventListener('touchend', onUp);
        window.addEventListener('touchcancel', onUp);
        viewport.addEventListener('wheel', (e) => {
            e.preventDefault();
            if (e.shiftKey) {
                // Горизонтальный скролл (Shift + Wheel)
                const dx = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
                panX -= dx;
                updateTransform();
            } else if (e.ctrlKey) {
                // Вертикальный скролл (Ctrl + Wheel)
                const dy = Math.abs(e.deltaY) > Math.abs(e.deltaX) ? e.deltaY : e.deltaX;
                panY -= dy;
                updateTransform();
            } else {
                changeZoom(e.deltaY > 0 ? -0.1 : 0.1);
            }
        }, { passive: false });
        viewport.addEventListener('contextmenu', (e) => e.preventDefault());
        }
        // ==================== ACTIONS ====================
        function deselectAll() { selectedIndices.clear(); selectedPoint = null; renderF0(); }
        function resetF0() {
            if (!editorInitialized) return;

            saveState();

            const targetIndices = selectedIndices.size > 0
                ? Array.from(selectedIndices)
                : f0Data.freqs.map((_, i) => i);

            targetIndices.forEach(i => {
                f0Data.freqs[i] = f0Data.originalFreqs[i] || 0;
            });

            markDirty();
            saveState();

            updatePoints();
            renderF0();
            updateStatusCounts();
        }
        function transpose(semitones) {
            if (!editorInitialized) return;

            saveState();

            const factor = Math.pow(2, semitones / 12);
            const targetIndices = selectedIndices.size > 0
                ? Array.from(selectedIndices)
                : f0Data.freqs.map((_, i) => i);

            targetIndices.forEach(i => {
                if (f0Data.freqs[i] > 0) {
                    f0Data.freqs[i] = Math.min(
                        f0Data.freqs[i] * factor,
                        f0Data.original_sample_rate / 2
                    );
                }
            });

            markDirty();
            saveState();

            updatePoints();
            renderF0();
            updateStatusCounts();
        }
        // === Логика модального окна для Pitch Shift ===
        let pitchShiftResolve = null;

        function openPitchShiftModal() {
            return new Promise(resolve => {
                pitchShiftResolve = resolve;
                const modal = document.getElementById('pitch-shift-modal');
                const input = document.getElementById('pitch-shift-input');
                input.value = '0';
                modal.classList.add('show');
                setTimeout(() => { input.focus(); input.select(); }, 100);
            });
        }

        function closePitchShiftModal() {
            const modal = document.getElementById('pitch-shift-modal');
            modal.classList.remove('show');
        }

        document.getElementById('pitch-shift-apply').onclick = () => {
            const input = document.getElementById('pitch-shift-input');
            const val = parseFloat(input.value);
            closePitchShiftModal();
            if (pitchShiftResolve) {
                pitchShiftResolve(isNaN(val) ? 0 : val);
                pitchShiftResolve = null;
            }
        };

        document.getElementById('pitch-shift-cancel').onclick = () => {
            closePitchShiftModal();
            if (pitchShiftResolve) {
                pitchShiftResolve(null);
                pitchShiftResolve = null;
            }
        };

        // Обработка Enter и Escape в поле ввода
        document.getElementById('pitch-shift-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                document.getElementById('pitch-shift-apply').click();
            } else if (e.key === 'Escape') {
                document.getElementById('pitch-shift-cancel').click();
            }
        });

        // === Обновленная функция сдвига питча ===
        async function shiftPitch() {
            if (!editorInitialized) return;
            const semitones = await openPitchShiftModal();
            if (semitones === null || semitones === 0) return;
            saveState();
            const factor = Math.pow(2, semitones / 12);
            const targetIndices = selectedIndices.size > 0 ? Array.from(selectedIndices) : f0Data.freqs.map((_, i) => i);
            targetIndices.forEach(i => {
                if (f0Data.freqs[i] > 0) f0Data.freqs[i] = parseFloat((f0Data.freqs[i] * factor).toFixed(2));
            });
            updatePoints(); 
            renderF0(); 
            updateStatusCounts();
        }
        function smoothF0() {
            if (!editorInitialized) return;

            saveState();

            const windowSize = 2;
            const newFreqs = [...f0Data.freqs];
            const targetIndices = selectedIndices.size > 0
                ? Array.from(selectedIndices)
                : f0Data.freqs.map((_, i) => i);

            for (let i of targetIndices) {
                if (f0Data.freqs[i] <= 0) continue;

                let sum = 0;
                let count = 0;

                for (let j = -windowSize; j <= windowSize; j++) {
                    const idx = i + j;
                    if (idx >= 0 && idx < f0Data.freqs.length && f0Data.freqs[idx] > 0) {
                        sum += f0Data.freqs[idx];
                        count++;
                    }
                }

                if (count > 0) {
                    newFreqs[i] = sum / count;
                }
            }

            f0Data.freqs = newFreqs;

            markDirty();
            saveState();

            updatePoints();
            renderF0();
            updateStatusCounts();
        }
        function autoFixOctaveJumps() {
        if (!editorInitialized || !f0Data.freqs) return;
        saveState();
        const newFreqs = [...f0Data.freqs];
        const threshold = 1.5;
        let lastValidFreq = 0;
        let lastValidIndex = -1;
        let correctionsCount = 0;
        for (let i = 0; i < newFreqs.length; i++) {
        let freq = newFreqs[i];
        if (freq > 0) {
        const isTarget = selectedIndices.size === 0 || selectedIndices.has(i);
        const isCloseEnough = lastValidIndex !== -1 && (i - lastValidIndex) < 50;
        if (isTarget && lastValidFreq > 0 && isCloseEnough) {
        let ratio = freq / lastValidFreq;
        if (ratio >= threshold || ratio <= (1 / threshold)) {
        let octaves = Math.round(Math.log2(ratio));
        if (octaves !== 0) {
        freq = freq / Math.pow(2, octaves);
        newFreqs[i] = parseFloat(freq.toFixed(2));
        correctionsCount++;
        }
        }
        }
        lastValidFreq = newFreqs[i];
        lastValidIndex = i;
        }
        }
        if (correctionsCount > 0) {
            f0Data.freqs = newFreqs;
            markDirty();
            saveState();
            updatePoints();
            renderF0();
            updateStatusCounts();
            showTooltipMessage('✨ """ + f"{_i18n('f0_corrector_fixed_points')}" + """' + correctionsCount);
        } else {
            showTooltipMessage('✨ 0');
        }
        }
        // ==================== EXPORT / SEND ====================
        function buildOutput() {
        return {
        method: f0Data.method,
        sample_rate: f0Data.original_sample_rate,
        window: f0Data.window,
        p_len: f0Data.freqs.length,
        freqs: f0Data.freqs.map(f => f > 0 ? parseFloat(f.toFixed(2)) : 0)
        };
        }
        // ==================== DOWNLOAD MODAL ====================
        function getDatetimeStamp() {
            const d = new Date();
            const pad = n => String(n).padStart(2, '0');
            return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) + '_' + pad(d.getHours()) + '-' + pad(d.getMinutes()) + '-' + pad(d.getSeconds());
        }
        function defaultCurveName() { return 'f0_corrected_' + getDatetimeStamp(); }
        function sourceCurveName() {
            const src = (f0Data.source_name && f0Data.source_name.length) ? f0Data.source_name : 'audio';
            return 'f0_corrected_' + src + '_' + getDatetimeStamp();
        }
        let downloadNameResolve = null;
        let currentDownloadVariant = 'default';
        function updateDownloadNamePreview() {
            const input = document.getElementById('download-name-input');
            document.querySelectorAll('.dl-option').forEach(o => o.classList.remove('checked'));
            if (currentDownloadVariant === 'default') {
                input.value = defaultCurveName(); input.readOnly = true;
                document.getElementById('dl-opt-default').classList.add('checked');
            } else if (currentDownloadVariant === 'source') {
                input.value = sourceCurveName(); input.readOnly = true;
                document.getElementById('dl-opt-source').classList.add('checked');
            } else {
                input.readOnly = false;
                document.getElementById('dl-opt-custom').classList.add('checked');
                setTimeout(() => { input.focus(); input.select(); }, 50);
            }
        }
        function openDownloadModal() {
            return new Promise(resolve => {
                downloadNameResolve = resolve;
                currentDownloadVariant = 'default';
                const defRadio = document.querySelector('input[name="dl-name-variant"][value="default"]');
                if (defRadio) defRadio.checked = true;
                updateDownloadNamePreview();
                document.getElementById('download-modal').classList.add('show');
            });
        }
        function closeDownloadModal() { document.getElementById('download-modal').classList.remove('show'); }
        document.querySelectorAll('input[name="dl-name-variant"]').forEach(r => {
            r.addEventListener('change', () => { currentDownloadVariant = r.value; updateDownloadNamePreview(); });
        });
        document.getElementById('download-apply').onclick = () => {
            let name = document.getElementById('download-name-input').value.trim();
            if (!name) name = defaultCurveName();
            closeDownloadModal();
            if (downloadNameResolve) { downloadNameResolve(name); downloadNameResolve = null; }
        };
        document.getElementById('download-cancel').onclick = () => {
            closeDownloadModal();
            if (downloadNameResolve) { downloadNameResolve(null); downloadNameResolve = null; }
        };
        document.getElementById('download-name-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') document.getElementById('download-apply').click();
            else if (e.key === 'Escape') document.getElementById('download-cancel').click();
        });
        async function downloadJSON() {
            if (!editorInitialized) { await customAlert('""" + f"{_i18n('f0_corrector_analyze_first')}" + """'); return; }
            const name = await openDownloadModal();
            if (!name) return;
            const jsonStr = JSON.stringify(buildOutput(), null, 2);
            const blob = new Blob([jsonStr], { type: "application/json" });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = name.toLowerCase().endsWith('.json') ? name : name + '.json'; a.click();
            window.URL.revokeObjectURL(url);
            showToast('⬇ ' + a.download);
        }
        async function sendToInference() {
        if (!editorInitialized) { await customAlert('""" + f"{_i18n('f0_corrector_analyze_first')}" + """'); return; }
        try {
        const res = await fetch('/f0_corrector/send_to_inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(Object.assign(buildOutput(), { session_hash: SESSION_HASH }))
        });
        const result = await res.json();
        if (result.error) throw new Error(result.error);
        window.parent.postMessage({ type: 'f0_corrector_send_to_inference', payload: { path: result.path } }, '*');
        await customAlert('""" + f"{_i18n('f0_corrector_sent_success')}" + """');
        } catch (e) {
        await customAlert('""" + f"{_i18n('f0_corrector_error_send')}" + """' + e.message);
        }
        }
        // ==================== ANALYZE / INIT ====================
        function showLoading(v) {
        document.getElementById('loading-overlay').classList.toggle('show', v);
        const ph = document.getElementById('placeholder');
        if (ph && v) ph.style.display = 'none';
        }
        function initEditor(data) {
        f0Data = {
        times: data.times || [],
        freqs: data.freqs || [],
        originalFreqs: [...(data.freqs || [])],
        sample_rate: data.sample_rate || 16000,
        original_sample_rate: data.original_sample_rate || data.sample_rate || 16000,
        window: data.window || 160,
        method: data.method || 'rmvpe+',
        n_mels: data.spec_height || 256,
        duration: data.duration || 0,
        spec_width: data.spec_width || 0,
        spec_height: data.spec_height || 0,
        source_name: ''
        };
        specCanvas = document.getElementById('specCanvas');
        gridCanvas = document.getElementById('gridCanvas');
        f0Canvas = document.getElementById('f0Canvas');
        specCtx = specCanvas.getContext('2d');
        gridCtx = gridCanvas.getContext('2d');
        f0Ctx = f0Canvas.getContext('2d');
        spectrogramImage = new Image();
        spectrogramImage.onload = () => {
        showLoading(false);
        const ph = document.getElementById('placeholder');
        if (ph) ph.style.display = 'none';
        editorInitialized = true;
        updatePoints();
        resetView();
        historyStack = [];
        historyIndex = -1;
        saveState();
        updateStatusCounts();
        };
        spectrogramImage.onerror = () => {
        showLoading(false);
        customAlert('""" + f"{_i18n('f0_corrector_error_analyze')}" + """Image load failed');
        };
        const src = data.spectrogram || '';
        spectrogramImage.src = src.startsWith('data:') ? src : ('data:image/png;base64,' + src);
        setupInteractions();
        }
        async function analyze() {
            if (analyzeInFlight) return;
            const audioPath = document.getElementById('audio-select-container').getAttribute('data-value');
            const f0Path = document.getElementById('f0-select-container').getAttribute('data-value');
            if (!audioPath) { await customAlert('""" + f"{_i18n('f0_corrector_select_audio_first')}" + """'); return; }
            if (!f0Path) { await customAlert('""" + f"{_i18n('f0_corrector_select_f0_first')}" + """'); return; }
            analyzeInFlight = true;
            showLoading(true);
            try {
                const res = await fetch('/f0_corrector/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ audio_path: audioPath, f0_path: f0Path })
                });
                const data = await res.json();
                if (!data.success) throw new Error(data.error || 'Unknown');
                initEditor(data);
                f0Data.source_name = audioPath.split(/[\\/]/).pop().replace(/\\.[^.]+$/, '');
            } catch (e) {
                showLoading(false);
                await customAlert('""" + f"{_i18n('f0_corrector_error_analyze')}" + """' + e.message);
            } finally {
                analyzeInFlight = false;
            }
        }
        // ==================== DROPDOWNS ====================
        function openCorrectorDropdown(input) {
        const container = input.closest('.custom-select-container');
        if (container.classList.contains('open')) return;
        document.querySelectorAll('.custom-select-container').forEach(c => { if (c !== container) c.classList.remove('open'); });
        container.classList.add('open');
        fetchCorrectorFiles();
        setTimeout(() => input.setSelectionRange(0, input.value.length), 0);
        }
        function filterCorrectorOptions(input) {
        const container = input.closest('.custom-select-container');
        const query = input.value.toLowerCase().trim();
        container.classList.add('open');
        container.querySelectorAll('.custom-option').forEach(opt => {
        if (opt.classList.contains('disabled')) return;
        opt.style.display = opt.innerText.toLowerCase().includes(query) ? '' : 'none';
        });
        }
        function selectCorrectorOption(opt) {
        if (opt.classList.contains('disabled')) return;
        const container = opt.closest('.custom-select-container');
        container.setAttribute('data-value', opt.getAttribute('data-value'));
        container.querySelector('.custom-select-input').value = opt.innerText;
        container.querySelectorAll('.custom-option').forEach(o => o.classList.remove('selected'));
        opt.classList.add('selected');
        container.classList.remove('open');
        }
        function populateOptions(containerId, paths) {
        const container = document.getElementById(containerId);
        const list = container.querySelector('.custom-select-options');
        const current = container.getAttribute('data-value');
        list.innerHTML = '';
        if (!paths || !paths.length) {
        list.innerHTML = '<div class="custom-option disabled">—</div>';
        return;
        }
        paths.forEach(p => {
        const div = document.createElement('div');
        div.className = 'custom-option' + (p === current ? ' selected' : '');
        div.setAttribute('data-value', p);
        div.innerText = p.split(/[\\/]/).pop();
        div.title = p;
        div.onclick = function(e) { e.stopPropagation(); selectCorrectorOption(this); };
        list.appendChild(div);
        });
        }
        async function fetchCorrectorFiles() {
        try {
        const res = await apiGet('/f0_corrector/files');
        const data = await res.json();
        [['audio-select-container', data.input_files], ['f0-select-container', data.f0_curves]].forEach(([id, paths]) => {
        const c = document.getElementById(id);
        const list = c.querySelector('.custom-select-options');
        const needsFill = c.classList.contains('open') || !list.children.length || list.querySelector('.custom-option.disabled');
        if (needsFill) populateOptions(id, paths);
        });
        } catch (e) { console.error(e); }
        }
        function setSelectValue(containerId, path) {
        if (!path) return;
        const container = document.getElementById(containerId);
        container.setAttribute('data-value', path);
        container.querySelector('.custom-select-input').value = path.split(/[\\/]/).pop();
        container.querySelectorAll('.custom-option').forEach(o => o.classList.remove('selected'));
        container.classList.add('flash');
        setTimeout(() => container.classList.remove('flash'), 1600);
        }
        document.addEventListener('click', (e) => {
        if (!e.target.closest('.custom-select-container'))
        document.querySelectorAll('.custom-select-container').forEach(c => c.classList.remove('open'));
        });
        // ==================== INBOX ====================
        let inboxRetryToken = 0;
        let analyzeInFlight = false;
        function apiDelete(url) {
            const sep = url.includes('?') ? '&' : '?';
            return fetch(url + sep + 'session_hash=' + encodeURIComponent(SESSION_HASH || ''), { method: 'DELETE' });
        }
        // Пара берётся не из памяти, а из того, что сейчас реально стоит в селекторах.
        // Покрывает все сценарии без накопления «половинок» и без таймеров:
        //   кнопка прислала оба разом; загрузка по очереди; догрузка одного к готовой паре (кейс с видео).
        function currentPair() {
            return {
                a: document.getElementById('audio-select-container').getAttribute('data-value') || '',
                f: document.getElementById('f0-select-container').getAttribute('data-value') || ''
            };
        }
        function applyInboxData(data) {
            if (!data || (!data.audio_path && !data.f0_path)) return false;
            if (data.audio_path) setSelectValue('audio-select-container', data.audio_path);
            if (data.f0_path)    setSelectValue('f0-select-container', data.f0_path);
            const { a, f } = currentPair();          // setSelectValue синхронна — пара уже свежая
            if (a && f && !analyzeInFlight) {
                showToast('""" + f"{_i18n('f0_corrector_auto_analyze')}" + """');
                analyze();
            } else if (!analyzeInFlight) {
                showToast('""" + f"{_i18n('f0_corrector_files_received')}" + """');
            }
            return true;
        }
        async function pollInboxOnce() {
            try {
                if (!SESSION_HASH) {
                    console.warn('[f0c] poll skipped: no SESSION_HASH yet');
                    try { window.parent.postMessage({ type: 'request_session_hash' }, '*'); } catch(e) {}
                    return false;
                }
                const res = await apiGet('/f0_corrector/inbox');
                const data = await res.json();
                if (!data || (!data.audio_path && !data.f0_path)) return false;
                applyInboxData(data);
                try { await apiDelete('/f0_corrector/inbox'); } catch(e) {}   // ack
                return true;
            } catch (e) { console.error('[f0c] poll error', e); return false; }
        }
        async function pollInbox(retries) {
            retries = retries || 0;
            if (retries <= 0) { pollInboxOnce(); return; }
            const token = ++inboxRetryToken;
            for (let i = 0; i <= retries; i++) {
                if (token !== inboxRetryToken) return;
                if (await pollInboxOnce()) return;
                await new Promise(r => setTimeout(r, Math.min(400 + i * 350, 1600)));
            }
        }
        // ==================== STATUS ====================
        function updateStatusCounts() {
        document.getElementById('st-points').textContent = f0Data.freqs.filter(f => f > 0).length + ' / ' + f0Data.freqs.length;
        document.getElementById('st-sr').textContent = f0Data.original_sample_rate + ' Hz';
        document.getElementById('st-size').textContent = f0Data.spec_width + '×' + f0Data.spec_height;
        document.getElementById('st-method').textContent = f0Data.method || '—';
        updateSelStatus();
        }
        function updateSelStatus() {
        const wrap = document.getElementById('st-sel-wrap');
        if (selectedIndices.size > 0) {
        wrap.classList.remove('sel-hidden');
        document.getElementById('st-sel').textContent = selectedIndices.size;
        } else wrap.classList.add('sel-hidden');
        }
        function updateHoverStatus(coords) {
        if (!editorInitialized || !f0Data.freqs.length) return;
        const baseWidth = spectrogramImage.naturalWidth || f0Data.spec_width;
        const baseHeight = spectrogramImage.naturalHeight || f0Data.spec_height;
        if (!baseWidth || !baseHeight) return;
        const frame = Math.round((coords.x / baseWidth) * (f0Data.freqs.length - 1));
        if (frame >= 0 && frame < f0Data.freqs.length) {
        const t = f0Data.times[frame] !== undefined ? f0Data.times[frame] : (frame / f0Data.freqs.length * f0Data.duration);
        document.getElementById('st-time').textContent = t.toFixed(3) + ' s';
        document.getElementById('st-f0').textContent = f0Data.freqs[frame] > 0 ? f0Data.freqs[frame].toFixed(1) + ' Hz' : '—';
        }
        document.getElementById('st-freq').textContent =
        (coords.y >= 0 && coords.y <= baseHeight) ? yToFreq(coords.y, baseHeight).toFixed(1) + ' Hz' : '—';
        }
        // ==================== HOTKEYS / INIT ====================
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;

            if ((e.ctrlKey || e.metaKey) && e.code === 'KeyZ') {
                e.preventDefault();
                if (e.shiftKey) {
                    redo();
                } else {
                    undo();
                }
            } else if ((e.ctrlKey || e.metaKey) && e.code === 'KeyY') {
                e.preventDefault();
                redo();
            } else if (e.key === '1') {
                setTool('pan');
            } else if (e.key === '2') {
                setTool('cursor');
            } else if (e.key === '3') {
                setTool('select');
            } else if (e.key === '4') {
                setTool('pencil');
            } else if (e.key === '5') {
                setTool('eraser');
            } else if (e.key === 'Escape') {
                deselectAll();
            }
        });
        window.addEventListener('resize', () => { if (editorInitialized) updateTransform(); });
        if (document.fonts && document.fonts.ready) {
        document.fonts.ready.then(() => { if (editorInitialized) updateTransform(); });
        }
        const _f0cResizeObserver = new ResizeObserver(() => { if (editorInitialized) updateTransform(); });
        _f0cResizeObserver.observe(document.getElementById('viewport'));
        setTool('pan');
        </script>
        </body>
        </html>
        """






















































































        self.sessions_statuses = {}

        @app.get("/preset_status")
        async def get_preset_status_stream(request: Request, session_id: str = None, session_hash: str = None):
            # Используем session_hash если передан, иначе session_id
            session_key = session_hash or session_id
            
            # Если ключ сессии не передан или равен 'undefined'/'null'/'', возвращаем пустой ответ
            if not session_key or session_key in ('undefined', 'null', ''):
                async def empty_generator():
                    yield "data: {}\n\n"
                return StreamingResponse(empty_generator(), media_type="text/event-stream")
            
            # Если сессии еще нет в словаре, создаем для нее пустой статус
            if session_key not in self.sessions_statuses:
                self.sessions_statuses[session_key] = {}

            async def event_generator():
                last_statuses = None
                try:
                    while True:
                        if await request.is_disconnected():
                            break
                        current_statuses = self.sessions_statuses[session_key].copy()
                        if current_statuses != last_statuses:
                            yield f"data: {json.dumps(current_statuses)}\n\n"
                            last_statuses = current_statuses.copy()
                        await asyncio.sleep(0.3)
                finally:
                    pass

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        @app.get("/preset_node_editor")
        def get_editor():
            return HTMLResponse(content=PRESETLESS_HTML_CONTENT)

        @app.get("/auto_ensemble_preset_editor")
        def auto_ensemble_preset_editor():
            return HTMLResponse(content=AUTO_ENSEMBLE_PRESET_HTML_CONTENT)

        @app.get("/iter_ensemble_preset_editor")
        def iter_ensemble_editor():
            return HTMLResponse(content=ITERATIVE_ENSEMBLE_PRESET_HTML_CONTENT)


        # --- API Эндпоинты для корректора F0 ---
        @app.get("/f0_corrector/files")
        def get_f0_corrector_files():
            return {
                "input_files": self.input_files.get_input_list(),
                "f0_curves": [
                    p.as_posix() for p in sorted(
                        self.f0_gen_output_path.f0_curves_dir.glob("*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True
                    )
                ]
            }

        @app.post("/f0_corrector/analyze")
        async def f0_corrector_analyze(request: Request):
            try:
                data = await request.json()
                audio_path = data.get("audio_path")
                f0_path = data.get("f0_path")
                if not audio_path or not self.f0_path_allowed(audio_path):
                    return JSONResponse({"success": False, "error": _i18n("path_not_exist")}, status_code=403)
                if not f0_path or not self.f0_path_allowed(f0_path):
                    return JSONResponse({"success": False, "error": _i18n("path_not_exist")}, status_code=403)
                # Тяжёлая работа — в потоке и не более N параллельно на весь сервер
                async with self._f0_analyze_semaphore:
                    payload = await asyncio.to_thread(f0_corrector_analyze_worker, audio_path, f0_path)
                return JSONResponse(payload)
            except Exception as e:
                return JSONResponse({"success": False, "error": str(e)}, status_code=500)

        @app.post("/f0_corrector/send_to_inference")
        async def f0_corrector_send_to_inference(request: Request):
            try:
                data = await request.json()
                freqs = [float(f) for f in data.get("freqs", [])]
                output = {
                    "method": data.get("method", "rmvpe+"),
                    "sample_rate": data.get("sample_rate", 16000),
                    "window": data.get("window", 160),
                    "p_len": len(freqs),
                    "freqs": freqs
                }
                # Per-session директория: нет коллизий и чужих файлов
                sid = "".join(c for c in str(data.get("session_hash", "") or "shared") if c.isalnum() or c in "-_")[:64] or "shared"
                temp_dir = Path(tempfile.gettempdir()) / "f0_corrector" / sid
                temp_dir.mkdir(parents=True, exist_ok=True)
                # self._purge_old_f0_tempfiles(temp_dir)
                timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
                out_path = Namer.iter(temp_dir / f"f0_corrected_{timestamp}.json")
                Path(out_path).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
                return {"success": True, "path": out_path}
            except Exception as e:
                return JSONResponse({"success": False, "error": str(e)}, status_code=500)

        @app.get("/f0_corrector/inbox")
        def get_f0_corrector_inbox(request: Request):
            sid = request.query_params.get("session_hash", "")
            if not sid:
                return {}
            # peek, не pop: удаляет клиент после обработки (идемпотентно при ретраях)
            return dict(self.f0_corrector_inbox.get(sid, {}))

        @app.delete("/f0_corrector/inbox")
        def ack_f0_corrector_inbox(request: Request):
            sid = request.query_params.get("session_hash", "")
            if sid:
                self.f0_corrector_inbox.pop(sid, None)
            return {"ok": True}

        @app.get("/f0_corrector_editor")
        def f0_corrector_editor():
            return HTMLResponse(content=F0_CORRECTOR_HTML_CONTENT)
        # ---------------------------------------

















        gradio_head_script = """
        <script>
        (function () {
            function arm() {
                var ids = ['f0c_upload_audio', 'f0c_upload_f0'];
                var timer = null, armed = 0;
                function ping() {
                    clearTimeout(timer);
                    timer = setTimeout(function () {
                        var f = document.getElementById('f0c-editor-iframe');
                        if (f && f.contentWindow) {
                            try { f.contentWindow.postMessage({ type: 'f0_corrector_inbox_check' }, '*'); } catch (e) {}
                        }
                    }, 300);
                }
                ids.forEach(function (id) {
                    var el = document.getElementById(id);
                    if (el) {
                        new MutationObserver(ping).observe(el, { childList: true, subtree: true, attributes: true, attributeFilter: ['class'] });
                        armed++;
                    }
                });
                return armed === ids.length;
            }
            if (!arm()) {
                var mo = new MutationObserver(function () { if (arm()) mo.disconnect(); });
                var start = function () { mo.observe(document.body, { childList: true, subtree: true }); };
                if (document.body) start(); else document.addEventListener('DOMContentLoaded', start);
            }
        })();



        


        (function () {
            function arm(el) {
                if (!el || el.__f0cVisObs) return;
                el.__f0cVisObs = true;
                var io = new IntersectionObserver(function (entries) {
                    var e = entries[0]; if (!e) return;
                    var r = e.boundingClientRect;
                    if (e.isIntersecting && r.width > 0 && r.height > 0) {
                        try { el.contentWindow.postMessage({ type: 'f0_corrector_inbox_check' }, '*'); } catch (_) {}
                    }
                }, { threshold: 0.01 });
                io.observe(el);
            }
            function tryArm() {
                var el = document.getElementById('f0c-editor-iframe');
                if (el) { arm(el); return true; }
                return false;
            }
            if (!tryArm()) {
                var mo = new MutationObserver(function () { if (tryArm()) mo.disconnect(); });
                var start = function () { mo.observe(document.body, { childList: true, subtree: true }); };
                if (document.body) start(); else document.addEventListener('DOMContentLoaded', start);
            }
        })();






        (function () {
            window.__gradio_session_hash = window.__gradio_session_hash || '';

            function broadcastHash(hash) {
                try {
                    var iframes = document.querySelectorAll('iframe');
                    for (var i = 0; i < iframes.length; i++) {
                        try {
                            iframes[i].contentWindow.postMessage(
                                { type: 'gradio_session_hash', hash: hash }, '*'
                            );
                        } catch (e) {}
                    }
                } catch (e) {}
            }

            function setSessionHash(hash) {
                if (!hash || hash === 'undefined' || hash === 'null') return;
                if (hash === window.__gradio_session_hash) return;
                window.__gradio_session_hash = hash;
                console.log('[mvsepless] Gradio session hash captured:', hash);
                broadcastHash(hash);
            }

            function extractFromUrl(url) {
                if (!url) return;
                var m = String(url).match(/[?&]session_hash=([^&]+)/);
                if (m) setSessionHash(decodeURIComponent(m[1]));
            }

            function extractFromBody(body) {
                if (typeof body !== 'string' || !body) return;
                // быстрый чек, чтобы не парсить всё подряд
                if (body.indexOf('session_hash') === -1) return;
                try {
                    var obj = JSON.parse(body);
                    if (obj && obj.session_hash) setSessionHash(obj.session_hash);
                } catch (e) {}
            }

            /* --- fetch --- */
            if (window.fetch) {
                var origFetch = window.fetch;
                window.fetch = function (input, init) {
                    try {
                        var url = (typeof input === 'string') ? input
                                : (input && input.url ? input.url : '');
                        extractFromUrl(url);
                        if (init && init.body) extractFromBody(init.body);
                    } catch (e) {}
                    return origFetch.apply(this, arguments);
                };
            }

            /* --- XMLHttpRequest --- */
            if (window.XMLHttpRequest) {
                var origOpen = XMLHttpRequest.prototype.open;
                XMLHttpRequest.prototype.open = function (method, url) {
                    try { extractFromUrl(url); } catch (e) {}
                    return origOpen.apply(this, arguments);
                };
                var origSend = XMLHttpRequest.prototype.send;
                XMLHttpRequest.prototype.send = function (body) {
                    try { extractFromBody(body); } catch (e) {}
                    return origSend.apply(this, arguments);
                };
            }

            /* --- EventSource (Gradio держит /queue/data?session_hash=...) --- */
            if (window.EventSource) {
                var OrigES = window.EventSource;
                function PatchedES(url, opts) {
                    try {
                        extractFromUrl(typeof url === 'string' ? url : (url && url.url ? url.url : ''));
                    } catch (e) {}
                    return new OrigES(url, opts);
                }
                PatchedES.prototype = OrigES.prototype;
                PatchedES.CONNECTING = OrigES.CONNECTING;
                PatchedES.OPEN = OrigES.OPEN;
                PatchedES.CLOSED = OrigES.CLOSED;
                window.EventSource = PatchedES;
            }

            /* Ответ iframe, который запросил hash после своей загрузки */
            window.addEventListener('message', function (event) {
                if (event.data && event.data.type === 'request_session_hash') {
                    if (window.__gradio_session_hash && event.source) {
                        try {
                            event.source.postMessage(
                                { type: 'gradio_session_hash', hash: window.__gradio_session_hash }, '*'
                            );
                        } catch (e) {}
                    }
                }
            });
        })();




        /* =========================================================
        Старая логика приёма состояний пресетов из iframe
        ========================================================= */
        window.getGradioSessionHash = function () {
            return window.__gradio_session_hash || '';
        };




        window.addEventListener('message', (event) => {
            if (event.data && event.data.type === 'update_preset') {
                const stateBox = document.querySelector('#hidden_preset_state textarea');
                if (stateBox) {
                    stateBox.value = JSON.stringify(event.data.payload, null, 2);
                    stateBox.dispatchEvent(new Event('input', { bubbles: true }));
                    stateBox.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
            else if (event.data && event.data.type === 'update_auto_ensemble_preset') {
                const stateBox = document.querySelector('#hidden_auto_ensemble_preset_state textarea');
                if (stateBox) {
                    stateBox.value = JSON.stringify(event.data.payload, null, 2);
                    stateBox.dispatchEvent(new Event('input', { bubbles: true }));
                    stateBox.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
            else if (event.data && event.data.type === 'update_iter_ensemble_preset') {
                const stateBox = document.querySelector('#hidden_iter_ensemble_preset_state textarea');
                if (stateBox) {
                    stateBox.value = JSON.stringify(event.data.payload, null, 2);
                    stateBox.dispatchEvent(new Event('input', { bubbles: true }));
                    stateBox.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
            else if (event.data && event.data.type === 'f0_corrector_send_to_inference') {
                const stateBox = document.querySelector('#hidden_f0_corrector_state textarea');
                if (stateBox) {
                    stateBox.value = event.data.payload.path;
                    stateBox.dispatchEvent(new Event('input', { bubbles: true }));
                    stateBox.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
        });
        </script>
        """

        custom_css = """
        #hidden_preset_state { 
            display: none !important; 
        }
        #hidden_auto_ensemble_preset_state { 
            display: none !important; 
        }
        #hidden_iter_ensemble_preset_state { 
            display: none !important; 
        }
        #hidden_f0_corrector_state { 
            display: none !important; 
        }
        """


        with gr.Blocks(theme=theme, head=gradio_head_script, css=custom_css) as mvsepless_app:
            sep_input_state = gr.State([])
            auto_ensemble_input_state = gr.State([])
            manual_ensemble_input_state = gr.State([])
            sep_history_list_state = gr.State([])
            auto_ensemble_history_state = gr.State([])
            manual_ensemble_history_state = gr.State([])
            subtract_1_input_state = gr.State([])
            subtract_2_input_state = gr.State([])
            subtract_history_state = gr.State([])
            vbach_input_state = gr.State([])
            vbach_history_state = gr.State([])
            vbach_models_state = gr.State([])
            vbach_index_state = gr.State([])
            vbach_custom_models_state = gr.State([])
            vbach_custom_index_state = gr.State([])
            vbach_custom_input_state = gr.State([])
            f0_input_state = gr.State([])
            custom_sep_checkpoints_state = gr.State([])
            custom_sep_configs_state = gr.State([])
            iterative_ensemble_input_state = gr.State([])
            iterative_ensemble_history_state = gr.State([])
            presetless_input_state = gr.State([])
            presetless_history_state = gr.State([])
            phase_fixer_target_state = gr.State([])
            phase_fixer_source_state = gr.State([])
            phase_fixer_history_state = gr.State([])
            with gr.Tab(_i18n("separation_tab")):
                with gr.Tab(_i18n("inference")):
                    sep_state = gr.State()
                    with gr.Row():
                        with gr.Column():
                            sep_upload_files = gr.File(show_label=False, **base_c_params["input_files_multi"])
                            with gr.Group():
                                sep_input_files = gr.Dropdown(container=False, allow_custom_value=True, **base_c_params["dropdown_multi"])
                                sep_input_files.focus(self.get_actual_input_list, inputs=[sep_input_files, sep_input_state], outputs=[sep_input_files, sep_input_state], show_progress="hidden")
                                sep_add_uploaded_files = gr.Checkbox(label=_i18n("add_uploaded_files_to_current_list"), value=False, **base_c_params["base"])
                                sep_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                                @sep_upload_files.upload(inputs=[sep_upload_files, sep_input_files, sep_add_uploaded_files], outputs=[sep_upload_files, sep_input_files])
                                def upload_files_fn(files: list, input_files: list, sep_add_uploaded_files: bool):
                                    uploaded_files = self.input_files.upload(files)
                                    if sep_add_uploaded_files:
                                        return gr.update(value=None), gr.update(choices=self.input_files.get_input_list(), value=[*uploaded_files, *input_files])
                                    return gr.update(value=None), gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)
                                @gr.render(inputs=[sep_input_files, sep_input_preview_check])
                                def preview_inputs(input: list, preview: bool):
                                    if preview:
                                        if input:
                                            for f_ in input:
                                                define_audio_with_size(basename=True, label="", value=f_, **base_c_params["output_audio"])
                        with gr.Column():
                            with gr.Group():
                                sep_model_name = gr.Dropdown(label=_i18n("model_name"), choices=all_models, value=default_model, **base_c_params["base"])
                                custom_sep_model_type = gr.Dropdown(
                                    label=_i18n("model_type"),
                                    choices=custom_model_types,
                                    value=custom_model_types[0], visible=False,
                                    **base_c_params["base"]
                                )
                                
                                custom_sep_checkpoint = gr.Dropdown(
                                    label=_i18n("checkpoint_path"), 
                                    multiselect=True, allow_custom_value=True, visible=False,
                                    max_choices=1,
                                    **base_c_params["base"]
                                )
                                custom_sep_checkpoint.focus(
                                    self.get_actual_custom_sep_checkpoints_list,
                                    inputs=[custom_sep_checkpoint, custom_sep_checkpoints_state],
                                    outputs=[custom_sep_checkpoint, custom_sep_checkpoints_state],
                                    show_progress="hidden"
                                )
                                
                                custom_sep_config = gr.Dropdown(
                                    label=_i18n("config_path"), 
                                    multiselect=True,  allow_custom_value=True, visible=False,
                                    max_choices=1,
                                    **base_c_params["base"]
                                )
                                custom_sep_config.focus(
                                    self.get_actual_custom_sep_configs_list,
                                    inputs=[custom_sep_config, custom_sep_configs_state],
                                    outputs=[custom_sep_config, custom_sep_configs_state],
                                    show_progress="hidden"
                                )
                                sep_use_custom_model = gr.Checkbox(label=_i18n("use_custom_model"), value=False, **base_c_params["base"])
                                sep_selected_stems = gr.CheckboxGroup(label=_i18n("select_stems"), info=_i18n("select_stems_info"), choices=stems_default, value=[], **base_c_params["base"])
                                sep_extract_instrumental = gr.Checkbox(label=_i18n("extract_instrumental"), visible=ext_inst_visible_default, value=False, **base_c_params["base"])
                                sep_model_name.change(self.update_model_name, inputs=sep_model_name, outputs=[sep_extract_instrumental, sep_selected_stems])
                                @sep_use_custom_model.change(inputs=[sep_use_custom_model], outputs=[sep_model_name, custom_sep_model_type, custom_sep_checkpoint, custom_sep_config, sep_selected_stems, sep_extract_instrumental])
                                def use_custom_fn(is_custom: bool):
                                    if is_custom:
                                        return gr.update(visible=False), gr.update(value=custom_model_types[0], choices=custom_model_types, visible=True), gr.update(value=[], visible=True), gr.update(value=[], visible=True), gr.update(choices=[], value=[], visible=True), gr.update(value=False, visible=False)
                                    else:
                                        return gr.update(visible=True, choices=all_models, value=default_model), gr.update(value=custom_model_types[0], choices=custom_model_types, visible=False), gr.update(value=[], visible=False), gr.update(value=[], visible=False), gr.update(choices=stems_default, value=[], visible=True), gr.update(value=False, visible=False)

                                @custom_sep_config.input(inputs=[custom_sep_config, custom_sep_model_type], outputs=[sep_extract_instrumental, sep_selected_stems])
                                def get_stems_from_config_fn(path: str, model_type: str):
                                    stems = get_stems_from_config_simple(one_element_list_to_value(path), model_type)
                                    return gr.update(value=False, visible=len(stems) > 2), gr.update(value=[], choices=stems)

                                sep_use_spec_invert = gr.Checkbox(label=_i18n("use_spec_invert"), value=False, **base_c_params["base"])
                                sep_sum_stems = gr.Checkbox(label=_i18n("invert_plus"), info=_i18n("invert_plus_info"), value=False, **base_c_params["base"])
                                with gr.Accordion(label=_i18n("separation_params"), open=False):
                                    add_params_comp_seq = generate_add_params_component()
                                    add_params_user_state = gr.State(default_add_params)

                                    for comp in add_params_comp_seq:
                                        comp.change(
                                            fn=self.update_add_params,
                                            inputs=[*add_params_comp_seq], outputs=add_params_user_state,
                                            show_progress="hidden"
                                        )
                                sep_template = gr.Textbox(label=_i18n("output_template"), info=_i18n("output_template_info"), value="NAME_(STEM)_MODEL", **base_c_params["base"])
                                sep_output_format = gr.Dropdown(label=_i18n("output_format"), choices=output_formats, value=output_formats[0], filterable=False, **base_c_params["base"])
                                sep_prefer_float = gr.Checkbox(label=_i18n("prefer_float"), value=False, **base_c_params["base"])
                                separate_btn = gr.Button(_i18n("separate"), variant="primary", **base_c_params["base"])
                    with gr.Group():
                        with gr.Row(equal_height=True):
                            with gr.Column(min_width=110):
                                gr.Markdown("<h4><center>"+_i18n("history")+"</center></h4>")
                            sep_history = gr.Dropdown(container=False, scale=13, multiselect=True, max_choices=1, **base_c_params["base"])
                            sep_history.focus(self.get_actual_history_list, inputs=[sep_history, sep_history_list_state], outputs=[sep_history, sep_history_list_state], show_progress="hidden")
                            @sep_history.input(inputs=sep_history, outputs=sep_state)
                            def separation_show_history_fn(key: list):
                                state = self.history.get_from_history(one_element_list_to_value(key))
                                return state
                        sep_off_players_output = gr.Checkbox(label=_i18n("off_audio_players_output"), info=_i18n("off_audio_players_output_info"), value=False, **base_c_params["base"])
                        @separate_btn.click(inputs=[sep_input_files, custom_sep_model_type, custom_sep_checkpoint, custom_sep_config, sep_model_name, sep_selected_stems, sep_extract_instrumental, sep_use_spec_invert, sep_template, sep_output_format, sep_sum_stems, add_params_user_state, sep_prefer_float, sep_use_custom_model], outputs=[sep_state, sep_upload_files], trigger_mode="once", concurrency_id="mvsepless_app_inference")
                        def separator_wrap(input_files: list, model_type: str, checkpoint: list, config: list, model_name: str, sel_stems: list, ext_inst: bool, spec_invert: bool, tmpl: str, output_format: str, sum_stems: bool, add_params: dict, pref_f: bool, is_custom: bool, progress=gr.Progress(track_tqdm=True)):
                            results = []
                            if is_custom:
                                checkpoint_path = one_element_list_to_value(checkpoint)
                                config_path = one_element_list_to_value(config)
                                
                                if not checkpoint_path or not config_path:
                                    gr.Warning(_i18n("paths_not_specified"))
                                    return [], gr.skip()
                                
                                results = self.separator.custom_separate(
                                    input_files=input_files, 
                                    output_dir=self.output_dir.gen_output_dir(), 
                                    output_format=output_format, 
                                    template=tmpl,
                                    model_type=model_type,
                                    ckpt=checkpoint_path,
                                    conf=config_path,
                                    extract_instrumental=ext_inst,
                                    use_spec_invert=spec_invert,
                                    invert_plus=sum_stems,
                                    prefer_float=pref_f,
                                    selected_stems=sel_stems,
                                    add_params=add_params
                                )
                                
                                model_name = Path(checkpoint_path).stem
                            else:
                                results = self.separator.separate(
                                    input_files=input_files, output_dir=self.output_dir.gen_output_dir(), 
                                    output_format=output_format, template=tmpl, model_name=model_name, extract_instrumental=ext_inst, 
                                    use_spec_invert=spec_invert, invert_plus=sum_stems, prefer_float=pref_f, selected_stems=sel_stems, add_params=add_params
                                )
                            self.history.add_to_history(model_name, results)
                            return results, gr.skip()
                        @gr.render(inputs=[sep_state, sep_off_players_output])
                        def show_players(state, off_players_output: bool):
                            if state:
                                zip_is_generated = False
                                all_stems_dict = {}
                                all_stems = set(stem_name_ for stem_list in (stems_list_ for basename_, stems_list_ in state) for stem_name_, stem_path_ in stem_list)
                                all_files = []
                                for stem in all_stems:
                                    all_stems_dict[stem] = []
                                for basename, stems_list in state:
                                    with gr.Group():
                                        gr.Markdown(f"<h4><center>{basename}</center></h4>")
                                        for stem_name, stem_path in stems_list:
                                            all_files.append(stem_path)
                                            all_stems_dict[stem_name].append(stem_path) 
                                            with gr.Row(equal_height=True):
                                                if off_players_output:
                                                    output_audio = define_download_button_with_size(
                                                        value=stem_path,
                                                        label=stem_name,
                                                        **base_c_params["base"], variant="huggingface",
                                                        scale=15,
                                                    )
                                                else:
                                                    output_audio = define_audio_with_size(
                                                        value=stem_path,
                                                        label=stem_name,
                                                        **base_c_params["output_audio"],
                                                        scale=15,
                                                    )
                                                reuse_btn = gr.Button(
                                                    _i18n("reuse_btn"), 
                                                    variant="secondary", **base_c_params["base"]
                                                )
                                                @reuse_btn.click(
                                                    inputs=[sep_input_files, sep_add_uploaded_files],
                                                    outputs=sep_input_files,
                                                )
                                                def reuse_fn(input_files: list, sep_add_uploaded_files: bool, stem=deepcopy(stem_path)) -> gr.update:
                                                    uploaded_files = self.input_files.upload([stem], copy=True)
                                                    if sep_add_uploaded_files:
                                                        return gr.update(choices=self.input_files.get_input_list(), value=[*uploaded_files, *input_files])
                                                    return gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)
                                                
                                with gr.Column(variant="panel"):
                                    with gr.Column(variant="panel"):
                                        for stem_a in all_stems:
                                            if len(all_stems_dict[stem_a]) > 1:
                                                reuse_all_stem_btn = gr.Button(_i18n("reuse_all_stem", stem=stem_a), variant="huggingface", **base_c_params["base"])
                                                @reuse_all_stem_btn.click(outputs=sep_input_files)
                                                def reuse_all_stem_fn(stem=deepcopy(stem_a)):
                                                    uploaded_files = self.input_files.upload(all_stems_dict[stem], copy=True)
                                                    return gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)
                                        
                                    reuse_all_stems_btn = gr.Button(_i18n("reuse_all_stems"), variant="primary", **base_c_params["base"])
                                    @reuse_all_stems_btn.click(outputs=sep_input_files)
                                    def reuse_all_stems_fn():
                                        uploaded_files = self.input_files.upload(all_files, copy=True)
                                        return gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)
                                    
                                    generate_zip_btn = gr.DownloadButton(label=_i18n("generate_zip_archive"), variant="huggingface", **base_c_params["base"])
                                    @generate_zip_btn.click(outputs=generate_zip_btn, trigger_mode="once")
                                    def generate_zip_fn():
                                        nonlocal zip_is_generated
                                        if zip_is_generated:
                                            return gr.skip()
                                        else:
                                            zip_file = generate_zip_archive(all_files, get_zip_output_path("mvsepless"))
                                            zip_is_generated = True
                                            return gr.DownloadButton(label=_i18n("download_zip_archive"), variant="huggingface", value=zip_file, **base_c_params["base"])

                            else:
                                gr.Markdown("<h3><center>"+_i18n("not_separated")+"</center></h3>", container=True)

                if not hf_space_mode:
                    with gr.Tab(_i18n("presets_tab")):
                        presetless_state = gr.State()
                        
                        with gr.Row():
                            with gr.Column():
                                presetless_upload_file = gr.File(show_label=False, **base_c_params["input_file"])
                                with gr.Group():
                                    presetless_input_file = gr.Dropdown(container=False, allow_custom_value=True, multiselect=True, max_choices=1, **base_c_params["base"])
                                    presetless_input_file.focus(self.get_actual_input_list, inputs=[presetless_input_file, presetless_input_state], outputs=[presetless_input_file, presetless_input_state], show_progress="hidden")
                                    presetless_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                                    @presetless_upload_file.upload(inputs=presetless_upload_file, outputs=[presetless_upload_file, presetless_input_file])
                                    def upload_file_fn(file: str):
                                        uploaded_files = self.input_files.upload([file])
                                        all_uploaded_files = self.input_files.get_input_list()
                                        if uploaded_files:
                                            first_value = [uploaded_files[0]]
                                        else:
                                            first_value = []
                                        return gr.update(value=None), gr.update(choices=all_uploaded_files, value=first_value)
                                    @gr.render(inputs=[presetless_input_file, presetless_input_preview_check])
                                    def preview_input(input: list, preview: bool):
                                        if preview:
                                            if input:
                                                define_audio_with_size(basename=True, label="", value=one_element_list_to_value(input), **base_c_params["output_audio"])

                            with gr.Column():
                                with gr.Group():
                                    with gr.Accordion(label=_i18n("separation_params"), open=False):
                                        presetless_add_params_comp_seq = generate_add_params_component()
                                        presetless_add_params_user_state = gr.State(default_add_params)
                                        
                                        for comp in presetless_add_params_comp_seq:
                                            comp.input(
                                                fn=self.update_add_params, 
                                                inputs=[*presetless_add_params_comp_seq], outputs=presetless_add_params_user_state,
                                                show_progress="hidden"
                                            )
                                    
                                    presetless_template = gr.Textbox(
                                        label=_i18n("output_template"), 
                                        info=_i18n("output_template2_info"), 
                                        value="NAME_(STEM)", 
                                        **base_c_params["base"]
                                    )
                                    presetless_run_button = gr.Button(_i18n("separate"), variant="primary", **base_c_params["base"])

                        with gr.Group():
                            presetless_preset_state = gr.Textbox(elem_id="hidden_preset_state")

                            node_editor = gr.HTML("""
                                <div id="preset-editor-container">
                                    <iframe 
                                        id="preset-editor-iframe"
                                        src="about:blank" 
                                        width="100%" 
                                        height="900px" 
                                        style="border:none;"
                                        onload="
                                            const urlParams = new URLSearchParams(window.location.search);
                                            const themeValue = urlParams.get('__theme') || 'light';
                                            const targetTheme = (themeValue === 'dark') ? 'dark' : 'light';
                                            
                                            // Функция для получения session_hash
                                            function getSessionHash() {
                                                try {
                                                    if (window.parent && window.parent.gradio_config) {
                                                        return window.parent.gradio_config.session_hash || '';
                                                    }
                                                } catch(e) {}
                                                try {
                                                    if (window.gradio_config) {
                                                        return window.gradio_config.session_hash || '';
                                                    }
                                                } catch(e) {}
                                                return '';
                                            }
                                            
                                            const sessionHash = getSessionHash();
                                            if(this.src.includes('about:blank')) {
                                                this.src = '/preset_node_editor?__theme=' + targetTheme + '&session_hash=' + sessionHash;
                                            }
                                        ">
                                    </iframe>
                                </div>
                                """, padding=False
                            )

                        with gr.Group():
                            with gr.Row(equal_height=True):
                                with gr.Column(min_width=110):
                                    gr.Markdown("<h4><center>"+_i18n("history")+"</center></h4>")
                                presetless_history = gr.Dropdown(container=False, scale=13, multiselect=True, max_choices=1, **base_c_params["base"])
                                presetless_history.focus(
                                    self.get_actual_preset_history_list,
                                    inputs=[presetless_history, presetless_history_state], 
                                    outputs=[presetless_history, presetless_history_state], 
                                    show_progress="hidden"
                                )
                                
                                @presetless_history.input(inputs=presetless_history, outputs=presetless_state)
                                def custom_separation_show_history_fn(key: list):
                                    state = self.preset_history.get_from_history(one_element_list_to_value(key))
                                    return state
                            presetless_off_players_output = gr.Checkbox(label=_i18n("off_audio_players_output"), info=_i18n("off_audio_players_output_info"), value=False, **base_c_params["base"])
                            @gr.render(inputs=[presetless_state, presetless_off_players_output])
                            def show_players(state, off_players_output: bool):
                                if state:
                                    zip_is_generated = False
                                    all_files = []

                                    for stem_name, stem_path in state:
                                        all_files.append(stem_path)
                                        with gr.Row(equal_height=True):
                                            if off_players_output:
                                                output_audio = define_download_button_with_size(
                                                    value=stem_path,
                                                    label=stem_name,
                                                    **base_c_params["base"], variant="huggingface",
                                                    scale=15,
                                                )
                                            else:
                                                output_audio = define_audio_with_size(
                                                    value=stem_path,
                                                    label=stem_name,
                                                    **base_c_params["output_audio"],
                                                    scale=15,
                                                )
                                            reuse_btn = gr.Button(
                                                _i18n("reuse_btn"), 
                                                variant="secondary", **base_c_params["base"]
                                            )
                                            @reuse_btn.click(
                                                inputs=[presetless_input_file],
                                                outputs=presetless_input_file,
                                            )
                                            def reuse_fn(input_file: str, stem=deepcopy(stem_path)) -> gr.update:
                                                uploaded_files = self.input_files.upload([stem], copy=True)
                                                all_uploaded_files = self.input_files.get_input_list()
                                                if all_uploaded_files:
                                                    first_value = [all_uploaded_files[0]]
                                                else:
                                                    first_value = []
                                                return gr.update(choices=all_uploaded_files, value=first_value)
                                            
                                    generate_zip_btn = gr.DownloadButton(label=_i18n("generate_zip_archive"), variant="huggingface", **base_c_params["base"])
                                    @generate_zip_btn.click(outputs=generate_zip_btn, trigger_mode="once")
                                    def generate_zip_fn():
                                        nonlocal zip_is_generated
                                        if zip_is_generated:
                                            return gr.skip()
                                        else:
                                            zip_file = generate_zip_archive(all_files, get_zip_output_path("mvsepless"))
                                            zip_is_generated = True
                                            return gr.DownloadButton(label=_i18n("download_zip_archive"), variant="huggingface", value=zip_file, **base_c_params["base"])
                                else:
                                    gr.Markdown("<h3><center>"+_i18n("not_separated")+"</center></h3>", container=True)


                        @presetless_run_button.click(
                            inputs=[presetless_preset_state, presetless_input_file, presetless_template, presetless_add_params_user_state],
                            outputs=[presetless_upload_file, presetless_state]
                        )
                        def execute_preset(preset_json: dict | str,
                                        input_file: list,
                                        template: str, 
                                        add_params: dict,
                                        request: gr.Request, # <-- ДОБАВЛЕНО: нативный объект запроса Gradio
                                        progress=gr.Progress(track_tqdm=True)):
                            # Получаем уникальный ID сессии (конкретной вкладки браузера)
                            session_hash = request.session_hash  # Получаем корректный хеш
                            
                            # Блокируем редактор перед началом выполнения
                            self.sessions_statuses[session_hash] = {"_locked": True}
                            
                            if isinstance(preset_json, str):
                                preset = json.loads(preset_json)
                            else:
                                preset = preset_json
                            if not preset or "nodes" not in preset:
                                gr.Warning(_i18n("preset_flow_invalid"))
                                self.sessions_statuses[session_hash]["_locked"] = False
                                return []
                            # Callback для обновления словаря
                            def progress_callback(data):
                                # data = {"nodeId": "node_1", "status": "active" | "success" | "error"}
                                if "nodeId" in data and "status" in data:
                                    # <-- ИСПРАВЛЕНО: обновляем статус внутри словаря конкретной сессии
                                    self.sessions_statuses[session_hash][data["nodeId"]] = data["status"]
                            preset_name = preset.get("name", "no_named_preset")
                            preset_executor = PresetExecutor(
                                input_file=one_element_list_to_value(input_file),
                                output_dir=self.output_dir.gen_output_dir(),
                                template=template,
                                add_params=add_params,
                                model_manager=self.separator
                            )
                            try:
                                result = preset_executor.execute_preset(
                                    preset=preset, progress_callback=progress_callback
                                )
                            finally:
                                # Разблокируем редактор после завершения (успешно или с ошибкой)
                                self.sessions_statuses[session_hash]["_locked"] = False
                                
                            self.preset_history.add_to_history(preset_name, result)
                            return gr.skip(), result

                with gr.Tab(_i18n("ensemble_tab")):
                    with gr.Tab(_i18n("auto_ensemble_tab")):
                        auto_ensemble_user_flow_state = gr.BrowserState([])
                        with gr.Row():
                            with gr.Column():
                                auto_ensemble_upload_file = gr.File(show_label=False, **base_c_params["input_file"])
                                with gr.Group():
                                    auto_ensemble_input_file = gr.Dropdown(container=False, allow_custom_value=True, multiselect=True, max_choices=1, **base_c_params["base"])
                                    auto_ensemble_input_file.focus(self.get_actual_input_list, inputs=[auto_ensemble_input_file, auto_ensemble_input_state], outputs=[auto_ensemble_input_file, auto_ensemble_input_state], show_progress="hidden")
                                    auto_ensemble_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                                    @auto_ensemble_upload_file.upload(inputs=auto_ensemble_upload_file, outputs=[auto_ensemble_upload_file, auto_ensemble_input_file])
                                    def upload_file_fn(file: str):
                                        uploaded_files = self.input_files.upload([file])
                                        all_uploaded_files = self.input_files.get_input_list()
                                        if uploaded_files:
                                            first_value = [uploaded_files[0]]
                                        else:
                                            first_value = []
                                        return gr.update(value=None), gr.update(choices=all_uploaded_files, value=first_value)
                                    @gr.render(inputs=[auto_ensemble_input_file, auto_ensemble_input_preview_check])
                                    def preview_input(input: list, preview: bool):
                                        if preview:
                                            if input:
                                                define_audio_with_size(basename=True, label="", value=one_element_list_to_value(input), **base_c_params["output_audio"])
                            with gr.Column():
                                with gr.Group():
                                    auto_ensemble_save_primary_stems = gr.Checkbox(label=_i18n("enable_save_primary_stems"), value=False, **base_c_params["base"])
                                    auto_ensemble_use_spec_invert = gr.Checkbox(label=_i18n("use_spec_invert"), value=False, **base_c_params["base"])
                                    auto_ensemble_type = gr.Dropdown(label=_i18n("ensemble_type"), info=_i18n("ensemble_type_info"), choices=ensemble_types, value=ensemble_types[0], filterable=False, **base_c_params["base"])
                                    auto_ensemble_template = gr.Textbox(label=_i18n("output_template"), info=_i18n("output_etemplate_info"), value="NAME_(COUNT)_TYPE", **base_c_params["base"])
                                    auto_ensemble_format = gr.Dropdown(label=_i18n("output_format"), choices=output_formats, value=output_formats[0], filterable=False, **base_c_params["base"])
                                    auto_ensemble_prefer_float = gr.Checkbox(label=_i18n("prefer_float"), value=False, **base_c_params["base"])
                                    auto_ensemble_run_btn = gr.Button(_i18n("run_ensemble"), variant="primary", **base_c_params["base"])
                        with gr.Column():

                            with gr.Group():
                                auto_ensemble_preset_state = gr.Textbox(elem_id="hidden_auto_ensemble_preset_state")

                                auto_ensembless_preset_editor = gr.HTML(
                                    """
                                    <iframe 
                                        src="about:blank" 
                                        width="100%" 
                                        height="600px" 
                                        style="border:none;"
                                        onload="
                                            const urlParams = new URLSearchParams(window.location.search);
                                            const themeValue = urlParams.get('__theme');
                                            const targetTheme = (themeValue === 'dark') ? 'dark' : 'light';
                                            
                                            if(this.src.includes('about:blank')) {
                                                this.src = '/auto_ensemble_preset_editor?__theme=' + targetTheme;
                                            }
                                        ">
                                    </iframe>
                                    """, max_height="800px", padding=False
                                )


                            with gr.Group():
                                with gr.Row(equal_height=True):
                                    with gr.Column(min_width=110):
                                        gr.Markdown("<h4><center>"+_i18n("history")+"</center></h4>")
                                    auto_ensemble_history = gr.Dropdown(container=False, scale=13, multiselect=True, max_choices=1, **base_c_params["base"])
                            with gr.Row():
                                with gr.Column():
                                    auto_ensemble_output_audio = gr.Audio(label=_i18n("ensemble_result"), value=None, **base_c_params["output_audio"])
                                    auto_ensemble_ioutput_audio = gr.Audio(label=_i18n("inverted_result"), value=None, **base_c_params["output_audio"])
                                    with gr.Row(equal_height=True):
                                        auto_ensemble_output_audio_reuse_btn = gr.Button(
                                            _i18n("reuse_output_btn"), 
                                            variant="secondary", visible=False, **base_c_params["base"]
                                        )
                                        auto_ensemble_ioutput_reuse_btn = gr.Button(
                                            _i18n("reuse_invert_btn"), 
                                            variant="huggingface", visible=False, **base_c_params["base"]
                                        )
                                    @auto_ensemble_output_audio_reuse_btn.click(
                                        inputs=[auto_ensemble_output_audio],
                                        outputs=auto_ensemble_input_file,
                                    )
                                    def reuse_fn(stem_audio: str) -> gr.update:
                                        uploaded_files = self.input_files.upload([stem_audio], copy=True)
                                        all_uploaded_files = self.input_files.get_input_list()
                                        if all_uploaded_files:
                                            first_value = [all_uploaded_files[0]]
                                        else:
                                            first_value = []
                                        return gr.update(choices=all_uploaded_files, value=first_value)
                                    @auto_ensemble_ioutput_reuse_btn.click(
                                        inputs=[auto_ensemble_ioutput_audio],
                                        outputs=auto_ensemble_input_file,
                                    )
                                    def reuse_fn(stem_audio: str) -> gr.update:
                                        uploaded_files = self.input_files.upload([stem_audio], copy=True)
                                        all_uploaded_files = self.input_files.get_input_list()
                                        if uploaded_files:
                                            first_value = [uploaded_files[0]]
                                        else:
                                            first_value = []
                                        return gr.update(choices=all_uploaded_files, value=first_value)
                                    
                                    auto_ensemble_zip_is_generated = gr.State(False)
                                    auto_ensemble_generate_zip_btn = gr.DownloadButton(label=_i18n("generate_zip_archive"), variant="huggingface", visible=False, **base_c_params["base"])
        
                                with gr.Column():
                                    with gr.Group():
                                        gr.Markdown("<h3><center>"+_i18n("saved_primary_stems")+"</center></h3>", container=True)
                                        auto_ensebmle_primary_stems_off_players_output = gr.Checkbox(label=_i18n("off_audio_players_output"), info=_i18n("off_audio_players_output_info"), value=False, **base_c_params["base"])
                                        auto_ensemble_primary_stems_state = gr.State([])
                                        @gr.render(inputs=[auto_ensemble_primary_stems_state, auto_ensebmle_primary_stems_off_players_output])
                                        def preview_pr_stems(input: list, off_players_output: bool):
                                            if input:
                                                for f_ in input:
                                                    if off_players_output:
                                                        eoutput_audio = define_download_button_with_size(basename=True, label="", value=f_, scale=15, **base_c_params["base"], variant="huggingface")
                                                    else:
                                                        eoutput_audio = define_audio_with_size(basename=True, label="", value=f_, scale=15, **base_c_params["output_audio"])
                                                    ereuse_btn = gr.Button(
                                                        _i18n("reuse_btn"), 
                                                        variant="secondary", **base_c_params["base"]
                                                    )
                                                    @ereuse_btn.click(
                                                        inputs=[eoutput_audio],
                                                        outputs=auto_ensemble_input_file,
                                                    )
                                                    def reuse_fn(stem_audio: str) -> gr.update:
                                                        uploaded_files = self.input_files.upload([stem_audio], copy=True)
                                                        all_uploaded_files = self.input_files.get_input_list()
                                                        if all_uploaded_files:
                                                            first_value = [all_uploaded_files[0]]
                                                        else:
                                                            first_value = []
                                                        return gr.update(choices=all_uploaded_files, value=first_value)
                                            else:
                                                gr.Markdown("<h3><center>"+_i18n("not_ensembled_with_primary_stems")+"</center></h3>", container=True)

                        @auto_ensemble_run_btn.click(inputs=[auto_ensemble_input_file, auto_ensemble_template, auto_ensemble_type, auto_ensemble_use_spec_invert, auto_ensemble_format, auto_ensemble_save_primary_stems, auto_ensemble_preset_state, auto_ensemble_prefer_float], outputs=[auto_ensemble_output_audio, auto_ensemble_ioutput_audio, auto_ensemble_primary_stems_state, auto_ensemble_upload_file, auto_ensemble_output_audio_reuse_btn, auto_ensemble_ioutput_reuse_btn, auto_ensemble_generate_zip_btn, auto_ensemble_zip_is_generated], concurrency_id="mvsepless_app_inference_ensemble")
                        def auto_ensemble_wrapper_fn(input_file: list, template: str, etype: str, spec_invert: bool, out_format: str, save_pr_stems: bool, flow: list[list], pref_f: bool, progress=gr.Progress(track_tqdm=True)):
                            out, iout, pr_stems = self.separator.auto_ensemble(input_file=one_element_list_to_value(input_file), output_dir=self.output_dir.gen_output_dir(), flow=json.loads(flow), template=template, etype=etype, output_format=out_format, use_spec_invert=spec_invert, save_primary_stems=save_pr_stems, prefer_float=pref_f)
                            self.auto_ensemble_history_app.add_to_history(etype, out, iout, pr_stems)
                            return update_audio_with_size(label=_i18n("ensemble_result"), value=out), update_audio_with_size(label=_i18n("inverted_result"), value=iout), pr_stems, gr.skip(), gr.update(visible=True), gr.update(visible=True), gr.DownloadButton(label=_i18n("generate_zip_archive"), variant="huggingface", visible=True, **base_c_params["base"]), gr.update(value=False)

                        auto_ensemble_history.focus(self.get_actual_auto_ensemble_history_list, inputs=[auto_ensemble_history, auto_ensemble_history_state], outputs=[auto_ensemble_history, auto_ensemble_history_state], show_progress="hidden")
                        @auto_ensemble_history.input(inputs=auto_ensemble_history, outputs=[auto_ensemble_output_audio, auto_ensemble_ioutput_audio, auto_ensemble_primary_stems_state, auto_ensemble_output_audio_reuse_btn, auto_ensemble_ioutput_reuse_btn, auto_ensemble_generate_zip_btn, auto_ensemble_zip_is_generated])
                        def auto_ensemble_show_history_fn(key: list):
                            out, iout, pr_stems = self.auto_ensemble_history_app.get_from_history(one_element_list_to_value(key))
                            visible = all([out, iout])
                            return update_audio_with_size(label=_i18n("ensemble_result"), value=out), update_audio_with_size(label=_i18n("inverted_result"), value=iout), pr_stems, gr.update(visible=visible), gr.update(visible=visible), gr.DownloadButton(label=_i18n("generate_zip_archive"), variant="huggingface", visible=visible, **base_c_params["base"]), gr.update(value=False)

                        @auto_ensemble_generate_zip_btn.click(inputs=[auto_ensemble_output_audio, auto_ensemble_ioutput_audio, auto_ensemble_primary_stems_state, auto_ensemble_zip_is_generated], outputs=[auto_ensemble_generate_zip_btn, auto_ensemble_zip_is_generated], trigger_mode="once")
                        def generate_zip_fn(out, iout, e_state, zip_is_generated):
                            all_files = []

                            if out:
                                all_files.append(out)

                            if iout:
                                all_files.append(iout)

                            if e_state:
                                all_files.extend(e_state)

                            if zip_is_generated:
                                return gr.skip(), gr.skip()
                            else:
                                zip_file = generate_zip_archive(all_files, get_zip_output_path("ensembless"))
                                zip_is_generated = True
                                return gr.DownloadButton(label=_i18n("download_zip_archive"), variant="huggingface", value=zip_file, **base_c_params["base"]), zip_is_generated

                    with gr.Tab(_i18n("iterative_ensemble_tab")):
                        iterative_ensemble_user_flow_state = gr.BrowserState([])
                        with gr.Row():
                            with gr.Column():
                                iterative_ensemble_upload_file = gr.File(show_label=False, **base_c_params["input_file"])
                                with gr.Group():
                                    iterative_ensemble_input_file = gr.Dropdown(container=False, allow_custom_value=True, multiselect=True, max_choices=1, **base_c_params["base"])
                                    iterative_ensemble_input_file.focus(
                                        self.get_actual_input_list, 
                                        inputs=[iterative_ensemble_input_file, iterative_ensemble_input_state], 
                                        outputs=[iterative_ensemble_input_file, iterative_ensemble_input_state], 
                                        show_progress="hidden"
                                    )
                                    iterative_ensemble_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                                    
                                    @iterative_ensemble_upload_file.upload(inputs=iterative_ensemble_upload_file, outputs=[iterative_ensemble_upload_file, iterative_ensemble_input_file])
                                    def upload_file_fn(file: str):
                                        uploaded_files = self.input_files.upload([file])
                                        all_uploaded_files = self.input_files.get_input_list()
                                        if uploaded_files:
                                            first_value = [uploaded_files[0]]
                                        else:
                                            first_value = []
                                        return gr.update(value=None), gr.update(choices=all_uploaded_files, value=first_value)
                                    
                                    @gr.render(inputs=[iterative_ensemble_input_file, iterative_ensemble_input_preview_check])
                                    def preview_input(input: list, preview: bool):
                                        if preview:
                                            if input:
                                                define_audio_with_size(basename=True, label="", value=one_element_list_to_value(input), **base_c_params["output_audio"])
                            
                            with gr.Column():
                                with gr.Group():
                                    iterative_ensemble_num_iters = gr.Number(
                                        label=_i18n("num_iters"), 
                                        minimum=1, 
                                        maximum=20, 
                                        value=4, 
                                        step=1, 
                                        **base_c_params["base"]
                                    )
                                    iterative_ensemble_save_intermediate = gr.Checkbox(
                                        label=_i18n("save_intermediate"), 
                                        value=False, 
                                        **base_c_params["base"]
                                    )
                                    iterative_ensemble_template = gr.Textbox(
                                        label=_i18n("output_template"), 
                                        info=_i18n("output_iterative_template_info"),
                                        value="NAME_ITER", 
                                        **base_c_params["base"]
                                    )
                                    iterative_ensemble_format = gr.Dropdown(
                                        label=_i18n("output_format"), 
                                        choices=output_formats, 
                                        value=output_formats[0], 
                                        filterable=False, 
                                        **base_c_params["base"]
                                    )
                                    iterative_ensemble_prefer_float = gr.Checkbox(label=_i18n("prefer_float"), value=False, **base_c_params["base"])
                                    iterative_ensemble_run_btn = gr.Button(
                                        _i18n("run_iterative_ensemble"), 
                                        variant="primary", 
                                        **base_c_params["base"]
                                    )
                        
                        with gr.Column():
                            with gr.Group():
                                iterative_ensembless_state = gr.Textbox(elem_id="hidden_iter_ensemble_preset_state")

                                iterative_ensembless_preset_editor = gr.HTML(
                                    """
                                    <iframe 
                                        src="about:blank" 
                                        width="100%" 
                                        height="600px" 
                                        style="border:none;"
                                        onload="
                                            const urlParams = new URLSearchParams(window.location.search);
                                            const themeValue = urlParams.get('__theme');
                                            const targetTheme = (themeValue === 'dark') ? 'dark' : 'light';
                                            
                                            if(this.src.includes('about:blank')) {
                                                this.src = '/iter_ensemble_preset_editor?__theme=' + targetTheme;
                                            }
                                        ">
                                    </iframe>
                                    """, max_height="800px", padding=False
                                )
                        
                        with gr.Group():
                            with gr.Row(equal_height=True):
                                with gr.Column(min_width=110):
                                    gr.Markdown("<h4><center>"+_i18n("history")+"</center></h4>")
                                iterative_ensemble_history = gr.Dropdown(
                                    container=False, 
                                    scale=13, 
                                    multiselect=True, 
                                    max_choices=1, 
                                    **base_c_params["base"]
                                )
                                iterative_ensemble_history.focus(
                                    self.get_actual_iterative_ensemble_history_list,
                                    inputs=[iterative_ensemble_history, iterative_ensemble_history_state],
                                    outputs=[iterative_ensemble_history, iterative_ensemble_history_state],
                                    show_progress="hidden"
                                )
                        
                        with gr.Row():
                            with gr.Column():
                                iterative_ensemble_output_audio = gr.Audio(
                                    label=_i18n("ensemble_result"), 
                                    value=None, 
                                    **base_c_params["output_audio"]
                                )
                                iterative_ensemble_intermediate_state = gr.State([])
                                with gr.Row(equal_height=True):
                                    iterative_ensemble_output_reuse_btn = gr.Button(
                                        _i18n("reuse_output_btn"), 
                                        variant="secondary", 
                                        visible=False, 
                                        **base_c_params["base"]
                                    )
                                @iterative_ensemble_output_reuse_btn.click(
                                    inputs=[iterative_ensemble_output_audio],
                                    outputs=iterative_ensemble_input_file,
                                )
                                def reuse_fn(stem_audio: str) -> gr.update:
                                    uploaded_files = self.input_files.upload([stem_audio], copy=True)
                                    all_uploaded_files = self.input_files.get_input_list()
                                    if all_uploaded_files:
                                        first_value = [all_uploaded_files[0]]
                                    else:
                                        first_value = []
                                    return gr.update(choices=all_uploaded_files, value=first_value)
                                
                                iterative_ensemble_zip_is_generated = gr.State(False)
                                iterative_ensemble_generate_zip_btn = gr.DownloadButton(
                                    label=_i18n("generate_zip_archive"), 
                                    variant="huggingface", 
                                    visible=False, 
                                    **base_c_params["base"]
                                )

                            with gr.Column():
                                with gr.Group():
                                    gr.Markdown("<h3><center>"+_i18n("intermediate_results")+"</center></h3>", container=True)
                                    iterative_ensemble_intermediate_off_players_output = gr.Checkbox(label=_i18n("off_audio_players_output"), info=_i18n("off_audio_players_output_info"), value=False, **base_c_params["base"])
                                    @gr.render(inputs=[iterative_ensemble_intermediate_state, iterative_ensemble_intermediate_off_players_output])
                                    def preview_intermediate_results(input: list, off_players_output: bool):
                                        if input:
                                            for f_ in input:
                                                if off_players_output:
                                                    define_download_button_with_size(basename=True, label="", value=f_, scale=15, **base_c_params["base"], variant="huggingface")
                                                else:
                                                    define_audio_with_size(basename=True, label="", value=f_, scale=15, **base_c_params["output_audio"])
                                        else:
                                            gr.Markdown("<h3><center>"+_i18n("no_intermediate_results")+"</center></h3>", container=True)
                        
                        @iterative_ensemble_run_btn.click(
                            inputs=[iterative_ensemble_input_file, iterative_ensemble_num_iters, iterative_ensemble_save_intermediate, 
                                    iterative_ensemble_template, iterative_ensemble_format, iterative_ensembless_state, iterative_ensemble_prefer_float], 
                            outputs=[iterative_ensemble_output_audio, iterative_ensemble_intermediate_state, iterative_ensemble_upload_file, 
                                    iterative_ensemble_output_reuse_btn, iterative_ensemble_generate_zip_btn, iterative_ensemble_zip_is_generated], 
                            trigger_mode="once", 
                            concurrency_id="mvsepless_app_inference_ensemble"
                        )
                        def iterative_ensemble_wrapper_fn(input_file: list, num_iters: int, save_intermediate: bool, template: str, out_format: str, flow: list[list], pref_f: bool, progress=gr.Progress(track_tqdm=True)):
                            if not flow:
                                gr.Warning(_i18n("flow_empty"))
                                return update_audio_with_size(label=_i18n("ensemble_result"), value=None), [], gr.skip(), gr.update(visible=False), gr.DownloadButton(label=_i18n("generate_zip_archive"), variant="huggingface", visible=False, **base_c_params["base"]), gr.update(value=False)
                            
                            result_path, intermediate_files = self.separator.iterative_ensemble(
                                input_file=one_element_list_to_value(input_file),
                                output_dir=self.output_dir.gen_output_dir(),
                                flow=json.loads(flow),
                                num_iters=num_iters,
                                output_format=out_format,
                                template=template,
                                save_intermediate=save_intermediate,
                                prefer_float=pref_f
                            )
                            
                            self.iterative_ensemble_history_app.add_to_history(result_path, intermediate_files, num_iters)
                            
                            return update_audio_with_size(label=_i18n("ensemble_result"), value=result_path), intermediate_files, gr.skip(), gr.update(visible=True), gr.DownloadButton(label=_i18n("generate_zip_archive"), variant="huggingface", visible=True, **base_c_params["base"]), gr.update(value=False)
                        
                        @iterative_ensemble_history.input(
                            inputs=iterative_ensemble_history, 
                            outputs=[iterative_ensemble_output_audio, iterative_ensemble_intermediate_state, iterative_ensemble_output_reuse_btn, 
                                    iterative_ensemble_generate_zip_btn, iterative_ensemble_zip_is_generated]
                        )
                        def iterative_ensemble_show_history_fn(key: list):
                            result, intermediate = self.iterative_ensemble_history_app.get_from_history(one_element_list_to_value(key))
                            return update_audio_with_size(label=_i18n("ensemble_result"), value=result), intermediate, gr.update(visible=result is not None), gr.DownloadButton(label=_i18n("generate_zip_archive"), variant="huggingface", visible=result is not None, **base_c_params["base"]), gr.update(value=False)
                        
                        @iterative_ensemble_generate_zip_btn.click(
                            inputs=[iterative_ensemble_output_audio, iterative_ensemble_intermediate_state, iterative_ensemble_zip_is_generated], 
                            outputs=[iterative_ensemble_generate_zip_btn, iterative_ensemble_zip_is_generated], 
                            trigger_mode="once"
                        )
                        def generate_zip_fn(out, intermediate_state, zip_is_generated):
                            all_files = []
                            
                            if out:
                                all_files.append(out)
                            
                            if intermediate_state:
                                all_files.extend(intermediate_state)
                            
                            if zip_is_generated:
                                return gr.skip(), gr.skip()
                            else:
                                zip_file = generate_zip_archive(all_files, get_zip_output_path("iterative_ensembless"))
                                zip_is_generated = True
                                return gr.DownloadButton(label=_i18n("download_zip_archive"), variant="huggingface", value=zip_file, **base_c_params["base"]), zip_is_generated
                
                    with gr.Tab(_i18n("man_ensemble_tab")):
                        with gr.Row():
                            with gr.Column():
                                manual_ensemble_upload_files = gr.File(show_label=False, **base_c_params["input_files_multi"])
                                with gr.Group():
                                    manual_ensemble_input_files = gr.Dropdown(container=False, allow_custom_value=True, **base_c_params["dropdown_multi"])
                                    manual_ensemble_input_files.focus(self.get_actual_input_list, inputs=[manual_ensemble_input_files, manual_ensemble_input_state], outputs=[manual_ensemble_input_files, manual_ensemble_input_state], show_progress="hidden")
                                    manual_ensemble_add_uploaded_files = gr.Checkbox(label=_i18n("add_uploaded_files_to_current_list"), value=False, **base_c_params["base"])
                                    manual_ensemble_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                                    @manual_ensemble_upload_files.upload(inputs=[manual_ensemble_upload_files, manual_ensemble_input_files, manual_ensemble_add_uploaded_files], outputs=[manual_ensemble_upload_files, manual_ensemble_input_files])
                                    def upload_files_fn(files: list, input_files: list, add_uploaded_files: bool):
                                        uploaded_files = self.input_files.upload(files)
                                        if add_uploaded_files:
                                            return gr.update(value=None), gr.update(choices=self.input_files.get_input_list(), value=[*uploaded_files, *input_files])
                                        return gr.update(value=None), gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)
                                    @gr.render(inputs=[manual_ensemble_input_files, manual_ensemble_input_preview_check])
                                    def preview_inputs(input: list, preview: bool):
                                        if preview:
                                            if input:
                                                for f_ in input:
                                                    define_audio_with_size(basename=True, label="", value=f_, **base_c_params["output_audio"])

                            with gr.Column():
                                with gr.Group():
                                    manual_ensemble_weights = gr.Textbox(label=_i18n("weights_only_for_avg_fft"), info=_i18n("weights_split"), value="", **base_c_params["base"])
                                    manual_ensemble_type = gr.Dropdown(label=_i18n("ensemble_type"), info=_i18n("ensemble_type_info"), choices=ensemble_types, value=ensemble_types[0], filterable=False, **base_c_params["base"])
                                    manual_ensemble_template = gr.Textbox(label=_i18n("output_template"), info=_i18n("output_metemplate_info"), value="ensemble_(COUNT)_TYPE", **base_c_params["base"])
                                    manual_ensemble_format = gr.Dropdown(label=_i18n("output_format"), choices=output_formats, value=output_formats[0], filterable=False, **base_c_params["base"])
                                    manual_ensemble_prefer_float = gr.Checkbox(label=_i18n("prefer_float"), value=False, **base_c_params["base"])
                                    manual_ensemble_run_btn = gr.Button(_i18n("run_ensemble"), variant="primary", **base_c_params["base"])

                        with gr.Group():
                            with gr.Row(equal_height=True):
                                with gr.Column(min_width=110):
                                    gr.Markdown("<h4><center>"+_i18n("history")+"</center></h4>")
                                manual_ensemble_history = gr.Dropdown(container=False, scale=13, multiselect=True, max_choices=1, **base_c_params["base"])

                        with gr.Row(equal_height=True):
                            manual_ensemble_output_audio = gr.Audio(label=_i18n("ensemble_result"), value=None, scale=15, **base_c_params["output_audio"])
                            manual_ensemble_reuse_btn = gr.Button(
                                _i18n("reuse_btn"), 
                                variant="secondary", visible=False, **base_c_params["base"]
                            )
                            @manual_ensemble_reuse_btn.click(
                                inputs=[manual_ensemble_output_audio, manual_ensemble_input_files, manual_ensemble_add_uploaded_files],
                                outputs=manual_ensemble_input_files,
                            )
                            def reuse_fn(stem_audio: str, input_files: list, add_uploaded_files: bool) -> gr.update:
                                uploaded_files = self.input_files.upload([stem_audio], copy=True)
                                if add_uploaded_files:
                                    return gr.update(choices=self.input_files.get_input_list(), value=[*uploaded_files, *input_files])
                                return gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)


                        @manual_ensemble_run_btn.click(inputs=[manual_ensemble_input_files, manual_ensemble_type, manual_ensemble_template, manual_ensemble_format, manual_ensemble_weights, manual_ensemble_prefer_float], outputs=[manual_ensemble_output_audio, manual_ensemble_reuse_btn])
                        def manual_ensemble_wrapper_fn(input_files: list, etype: str, template: str, out_format: str, weights_str: str, pref_f: bool, progress=gr.Progress(track_tqdm=True)):
                            weights_str = weights_str.strip(" ")
                            if weights_str != "":
                                weights = [float(weight) for weight in weights_str.split(",")]
                            else:
                                weights = []
                            result = self.separator.manual_ensemble(input_files, self.output_dir.gen_output_dir(), weights, template, etype, out_format, pref_f)
                            self.manual_ensemble_history_app.add_to_history(etype, result)
                            return update_audio_with_size(label=_i18n("ensemble_result"), value=result), gr.update(visible=True)

                        manual_ensemble_history.focus(self.get_actual_manual_ensemble_history_list, inputs=[manual_ensemble_history, manual_ensemble_history_state], outputs=[manual_ensemble_history, manual_ensemble_history_state], show_progress="hidden")
                        @manual_ensemble_history.input(inputs=manual_ensemble_history, outputs=[manual_ensemble_output_audio, manual_ensemble_reuse_btn])
                        def manual_ensemble_show_fn(key: list):
                            output = self.manual_ensemble_history_app.get_from_history(one_element_list_to_value(key))
                            return update_audio_with_size(label=_i18n("ensemble_result"), value=output), gr.update(visible=output is not None)

            with gr.Tab(_i18n("extras_tab")):
                with gr.Tab(_i18n("subtract_tab")):
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                gr.Markdown("<h3><center>"+_i18n("original")+"</center></h3>", container=True)
                                subtract_1_upload_file = gr.File(show_label=False, **base_c_params["input_file"])
                            with gr.Group():
                                subtract_1_input_file = gr.Dropdown(container=False, allow_custom_value=True, multiselect=True, max_choices=1, **base_c_params["base"])
                                subtract_1_input_file.focus(self.get_actual_input_list, inputs=[subtract_1_input_file, subtract_1_input_state], outputs=[subtract_1_input_file, subtract_1_input_state], show_progress="hidden")
                                subtract_1_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                                @subtract_1_upload_file.upload(inputs=subtract_1_upload_file, outputs=[subtract_1_upload_file, subtract_1_input_file])
                                def upload_file_fn(file: str):
                                    uploaded_files = self.input_files.upload([file])
                                    all_uploaded_files = self.input_files.get_input_list()
                                    if uploaded_files:
                                        first_value = [uploaded_files[0]]
                                    else:
                                        first_value = []
                                    return gr.update(value=None), gr.update(choices=all_uploaded_files, value=first_value)
                                @gr.render(inputs=[subtract_1_input_file, subtract_1_input_preview_check])
                                def preview_input(input: list, preview: bool):
                                    if preview:
                                        if input:
                                            define_audio_with_size(basename=True, label="", value=one_element_list_to_value(input), **base_c_params["output_audio"])
                        with gr.Column():
                            with gr.Group():
                                gr.Markdown("<h3><center>"+_i18n("stem")+"</center></h3>", container=True)
                                subtract_2_upload_file = gr.File(show_label=False, **base_c_params["input_file"])
                            with gr.Group():
                                subtract_2_input_file = gr.Dropdown(container=False, allow_custom_value=True, multiselect=True, max_choices=1, **base_c_params["base"])
                                subtract_2_input_file.focus(self.get_actual_input_list, inputs=[subtract_2_input_file, subtract_2_input_state], outputs=[subtract_2_input_file, subtract_2_input_state], show_progress="hidden")
                                subtract_2_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                                @subtract_2_upload_file.upload(inputs=subtract_2_upload_file, outputs=[subtract_2_upload_file, subtract_2_input_file])
                                def upload_file_fn(file: str):
                                    uploaded_files = self.input_files.upload([file])
                                    all_uploaded_files = self.input_files.get_input_list()
                                    if uploaded_files:
                                        first_value = [uploaded_files[0]]
                                    else:
                                        first_value = []
                                    return gr.update(value=None), gr.update(choices=all_uploaded_files, value=first_value)
                                @gr.render(inputs=[subtract_2_input_file, subtract_2_input_preview_check])
                                def preview_input(input: list, preview: bool):
                                    if preview:
                                        if input:
                                            define_audio_with_size(basename=True, label="", value=one_element_list_to_value(input), **base_c_params["output_audio"])

                    with gr.Group():
                        subtract_use_spec_invert = gr.Checkbox(label=_i18n("use_spec_invert"), value=False, **base_c_params["base"])
                        subtract_template = gr.Textbox(label=_i18n("output_template"), info=_i18n("output_itemplate_info"), value="NAME_(TYPE)_inverted", **base_c_params["base"])
                        subtract_output_format = gr.Dropdown(label=_i18n("output_format"), choices=output_formats, value=output_formats[0], filterable=False, **base_c_params["base"])
                        subtract_prefer_float = gr.Checkbox(label=_i18n("prefer_float"), value=False, **base_c_params["base"])
                        subtract_run_btn = gr.Button(_i18n("subtract"), variant="primary", **base_c_params["base"])

                    with gr.Group():
                        with gr.Row(equal_height=True):
                            with gr.Column(min_width=110):
                                gr.Markdown("<h4><center>"+_i18n("history")+"</center></h4>")
                            subtract_history = gr.Dropdown(container=False, scale=13, multiselect=True, max_choices=1, **base_c_params["base"])

                    with gr.Row(equal_height=True):
                        subtract_output_audio = gr.Audio(label=_i18n("inverted_result"), value=None, scale=15, **base_c_params["output_audio"])
                        subtract_output_audio_reuse_btn = gr.Button(
                            _i18n("reuse_btn"), 
                            variant="secondary", visible=False, **base_c_params["base"]
                        )
                        @subtract_output_audio_reuse_btn.click(
                            inputs=[subtract_output_audio],
                            outputs=subtract_1_input_file,
                        )
                        def reuse_fn(stem_audio: str) -> gr.update:
                            uploaded_files = self.input_files.upload([stem_audio], copy=True)
                            return gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)

                        @subtract_run_btn.click(inputs=[subtract_1_input_file, subtract_2_input_file, subtract_output_format, subtract_use_spec_invert, subtract_template, subtract_prefer_float], outputs=[subtract_output_audio, subtract_output_audio_reuse_btn])
                        def subtract_run_fn(input_1: list, input_2: list, out_format: str, spec_invert: bool, template: str, pref_f: bool, progress=gr.Progress(track_tqdm=True)):
                            result = self.separator.subtract(one_element_list_to_value(input_1), one_element_list_to_value(input_2), self.output_dir.gen_output_dir(), out_format, spec_invert, template, pref_f)
                            self.subtract_history_app.add_to_history(("spectrogram" if spec_invert else "waveform"), result)
                            return update_audio_with_size(label=_i18n("inverted_result"), value=result), gr.update(visible=True)

                        subtract_history.focus(self.get_actual_subtract_history_list, inputs=[subtract_history, subtract_history_state], outputs=[subtract_history, subtract_history_state], show_progress="hidden")
                        @subtract_history.input(inputs=subtract_history, outputs=[subtract_output_audio, subtract_output_audio_reuse_btn])
                        def subtract_show_fn(key: list):
                            output = self.subtract_history_app.get_from_history(one_element_list_to_value(key))
                            return update_audio_with_size(label=_i18n("inverted_result"), value=output), gr.update(visible=output is not None)


                with gr.Tab(_i18n("phase_fixer_tab")):
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                gr.Markdown("<h3><center>"+_i18n("phase_fixer_target")+"</center></h3>", container=True)

                                phase_fixer_target_upload_file = gr.File(show_label=False, **base_c_params["input_file"])
                            with gr.Group():
                                phase_fixer_target_input_file = gr.Dropdown(container=False, allow_custom_value=True, multiselect=True, max_choices=1, **base_c_params["base"])
                                phase_fixer_target_input_file.focus(self.get_actual_input_list, inputs=[phase_fixer_target_input_file, phase_fixer_target_state], outputs=[phase_fixer_target_input_file, phase_fixer_target_state], show_progress="hidden")
                                phase_fixer_target_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                                @phase_fixer_target_upload_file.upload(inputs=phase_fixer_target_upload_file, outputs=[phase_fixer_target_upload_file, phase_fixer_target_input_file])
                                def upload_phase_fixer_target_fn(file: str):
                                    uploaded_files = self.input_files.upload([file])
                                    all_uploaded_files = self.input_files.get_input_list()
                                    if uploaded_files:
                                        first_value = [uploaded_files[0]]
                                    else:
                                        first_value = []
                                    return gr.update(value=None), gr.update(choices=all_uploaded_files, value=first_value)
                                @gr.render(inputs=[phase_fixer_target_input_file, phase_fixer_target_preview_check])
                                def preview_phase_fixer_target(input: list, preview: bool):
                                    if preview:
                                        if input:
                                            define_audio_with_size(basename=True, label="", value=one_element_list_to_value(input), **base_c_params["output_audio"])
                        with gr.Column():
                            with gr.Group():
                                gr.Markdown("<h3><center>"+_i18n("phase_fixer_source")+"</center></h3>", container=True)

                                phase_fixer_source_upload_file = gr.File(show_label=False, **base_c_params["input_file"])
                            with gr.Group():
                                phase_fixer_source_input_file = gr.Dropdown(container=False, allow_custom_value=True, multiselect=True, max_choices=1, **base_c_params["base"])
                                phase_fixer_source_input_file.focus(self.get_actual_input_list, inputs=[phase_fixer_source_input_file, phase_fixer_source_state], outputs=[phase_fixer_source_input_file, phase_fixer_source_state], show_progress="hidden")
                                phase_fixer_source_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                                @phase_fixer_source_upload_file.upload(inputs=phase_fixer_source_upload_file, outputs=[phase_fixer_source_upload_file, phase_fixer_source_input_file])
                                def upload_phase_fixer_source_fn(file: str):
                                    uploaded_files = self.input_files.upload([file])
                                    all_uploaded_files = self.input_files.get_input_list()
                                    if uploaded_files:
                                        first_value = [uploaded_files[0]]
                                    else:
                                        first_value = []
                                    return gr.update(value=None), gr.update(choices=all_uploaded_files, value=first_value)
                                @gr.render(inputs=[phase_fixer_source_input_file, phase_fixer_source_preview_check])
                                def preview_phase_fixer_source(input: list, preview: bool):
                                    if preview:
                                        if input:
                                            define_audio_with_size(basename=True, label="", value=one_element_list_to_value(input), **base_c_params["output_audio"])
                    with gr.Row():
                        phase_fixer_swap_btn = gr.Button(_i18n("phase_fixer_swap"), variant="secondary", **base_c_params["base"])
                    with gr.Group():
                        with gr.Accordion(label=_i18n("phase_fixer_settings"), open=True):
                            with gr.Row():
                                with gr.Column():
                                    phase_fixer_transfer_phase = gr.Checkbox(label=_i18n("preset_node_transfer_phase"), info=_i18n("phase_fixer_transfer_phase_info"), value=True, **base_c_params["base"])
                                    phase_fixer_transfer_magnitude = gr.Checkbox(label=_i18n("preset_node_transfer_magnitude"), info=_i18n("phase_fixer_transfer_magnitude_info"), value=False, **base_c_params["base"])
                                with gr.Column():
                                    phase_fixer_freq_blend = gr.Checkbox(label=_i18n("preset_node_freq_blend_phases"), info=_i18n("phase_fixer_freq_blend_info"), value=True, **base_c_params["base"])
                                    with gr.Row():
                                        phase_fixer_low_cutoff = gr.Slider(label=_i18n("preset_node_low_cutoff"), minimum=20, maximum=20000, value=500, step=10, **base_c_params["base"])
                                        phase_fixer_high_cutoff = gr.Slider(label=_i18n("preset_node_high_cutoff"), minimum=20, maximum=20000, value=5000, step=10, **base_c_params["base"])
                        phase_fixer_template = gr.Textbox(label=_i18n("output_template"), info=_i18n("output_pftemplate_info"), value="NAME_TYPE", **base_c_params["base"])
                        phase_fixer_output_format = gr.Dropdown(label=_i18n("output_format"), choices=output_formats, value=output_formats[0], filterable=False, **base_c_params["base"])
                        phase_fixer_prefer_float = gr.Checkbox(label=_i18n("prefer_float"), value=False, **base_c_params["base"])
                        phase_fixer_run_btn = gr.Button(_i18n("phase_fixer_run"), variant="primary", **base_c_params["base"])
                    with gr.Group():
                        with gr.Row(equal_height=True):
                            with gr.Column(min_width=110):
                                gr.Markdown("<h4><center>"+_i18n("history")+"</center></h4>")
                            phase_fixer_history = gr.Dropdown(container=False, scale=13, multiselect=True, max_choices=1, **base_c_params["base"])
                    with gr.Row(equal_height=True):
                        phase_fixer_output_audio = gr.Audio(label=_i18n("phase_fixer_result"), value=None, scale=15, **base_c_params["output_audio"])
                        phase_fixer_reuse_btn = gr.Button(_i18n("reuse_btn"), variant="secondary", visible=False, **base_c_params["base"])

                    @phase_fixer_swap_btn.click(inputs=[phase_fixer_target_input_file, phase_fixer_source_input_file], outputs=[phase_fixer_target_input_file, phase_fixer_source_input_file], show_progress="hidden")
                    def phase_fixer_swap_fn(target: list, source: list):
                        return gr.update(value=source), gr.update(value=target)

                    @phase_fixer_reuse_btn.click(inputs=[phase_fixer_output_audio], outputs=phase_fixer_target_input_file)
                    def phase_fixer_reuse_fn(stem_audio: str) -> gr.update:
                        uploaded_files = self.input_files.upload([stem_audio], copy=True)
                        return gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)

                    @phase_fixer_run_btn.click(
                        inputs=[phase_fixer_target_input_file, phase_fixer_source_input_file, phase_fixer_output_format,
                                phase_fixer_transfer_magnitude, phase_fixer_transfer_phase, phase_fixer_freq_blend,
                                phase_fixer_low_cutoff, phase_fixer_high_cutoff, phase_fixer_template, phase_fixer_prefer_float],
                        outputs=[phase_fixer_output_audio, phase_fixer_reuse_btn],
                        trigger_mode="once", concurrency_id="mvsepless_app_inference"
                    )
                    def phase_fixer_run_fn(target: list, source: list, out_format: str, transfer_mag: bool, transfer_phase: bool,
                                        freq_blend: bool, low_cutoff: int, high_cutoff: int, template: str, pref_f: bool,
                                        progress=gr.Progress(track_tqdm=True)):
                        target_val = one_element_list_to_value(target)
                        source_val = one_element_list_to_value(source)
                        if not target_val or not source_val:
                            gr.Warning(_i18n("paths_not_specified"))
                            return gr.skip(), gr.skip()
                        result = self.separator.phase_fixer(
                            target=target_val,
                            source=source_val,
                            output_dir=self.output_dir.gen_output_dir(),
                            output_format=out_format,
                            template=template,
                            transfer_magnitude=transfer_mag,
                            transfer_phase=transfer_phase,
                            freq_blend_phases=freq_blend,
                            low_cutoff=int(low_cutoff),
                            high_cutoff=int(high_cutoff),
                            prefer_float=pref_f
                        )
                        settings_str = f"mag:{_i18n('yes') if transfer_mag else _i18n('no')} | phase:{_i18n('yes') if transfer_phase else _i18n('no')} | blend:{_i18n('yes') if freq_blend else _i18n('no')}"
                        self.phase_fixer_history_app.add_to_history(settings_str, result)
                        return update_audio_with_size(label=_i18n("phase_fixer_result"), value=result), gr.update(visible=True)

                    phase_fixer_history.focus(self.get_actual_phase_fixer_history_list, inputs=[phase_fixer_history, phase_fixer_history_state], outputs=[phase_fixer_history, phase_fixer_history_state], show_progress="hidden")

                    @phase_fixer_history.input(inputs=phase_fixer_history, outputs=[phase_fixer_output_audio, phase_fixer_reuse_btn])
                    def phase_fixer_show_fn(key: list):
                        output = self.phase_fixer_history_app.get_from_history(one_element_list_to_value(key))
                        return update_audio_with_size(label=_i18n("phase_fixer_result"), value=output), gr.update(visible=output is not None)












            with gr.Tab(_i18n("vbach_tab")):
                vbach_inner_tabs = gr.Tabs()
                with vbach_inner_tabs:
                    with gr.Tab(_i18n("inference"), id="vbach_infer"):
                        with gr.Row():
                            with gr.Column():
                                vbach_upload_files = gr.File(show_label=False, **base_c_params["input_files_multi"])
                                with gr.Group():
                                    vbach_input_files = gr.Dropdown(container=False, allow_custom_value=True, **base_c_params["dropdown_multi"])
                                    vbach_input_files.focus(
                                        self.get_actual_input_list, 
                                        inputs=[vbach_input_files, vbach_input_state], 
                                        outputs=[vbach_input_files, vbach_input_state], 
                                        show_progress="hidden"
                                    )
                                    vbach_input_add_uploaded_files = gr.Checkbox(label=_i18n("add_uploaded_files_to_current_list"), value=False, **base_c_params["base"])
                                    vbach_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                                    
                                    @vbach_upload_files.upload(inputs=[vbach_upload_files, vbach_input_files, vbach_input_add_uploaded_files], outputs=[vbach_upload_files, vbach_input_files])
                                    def upload_files_fn(files: list, input_files: list, add_uploaded_files: bool):
                                        uploaded_files = self.input_files.upload(files)
                                        if add_uploaded_files:
                                            return gr.update(value=None), gr.update(choices=self.input_files.get_input_list(), value=[*uploaded_files, *input_files])
                                        return gr.update(value=None), gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)
                                    
                                    @gr.render(inputs=[vbach_input_files, vbach_input_preview_check])
                                    def preview_inputs(input: list, preview: bool):
                                        if preview and input:
                                            for f_ in input:
                                                define_audio_with_size(basename=True, label="", value=f_, **base_c_params["output_audio"])
                            
                            with gr.Column():
                                with gr.Group():
                                    vbach_model_path = gr.Dropdown(
                                        label=_i18n("model_path"), multiselect=True, allow_custom_value=True, max_choices=1, **base_c_params["base"])
                                    vbach_model_path.focus(
                                        self.get_actual_vbach_models_list,
                                        inputs=[vbach_model_path, vbach_models_state],
                                        outputs=[vbach_model_path, vbach_models_state],
                                        show_progress="hidden"
                                    )
                                    
                                    vbach_index_path = gr.Dropdown(label=_i18n("index_path"), allow_custom_value=True, multiselect=True, max_choices=1, **base_c_params["base"])
                                    vbach_index_path.focus(
                                        self.get_actual_vbach_index_list,
                                        inputs=[vbach_index_path, vbach_index_state],
                                        outputs=[vbach_index_path, vbach_index_state],
                                        show_progress="hidden"
                                    )
                                    
                                    vbach_pitch = gr.Slider(
                                        label=_i18n("pitch"),
                                        minimum=-36,
                                        maximum=36,
                                        step=0.1,
                                        value=0,
                                        **base_c_params["base"]
                                    )
                                    
                                    vbach_f0method = gr.Dropdown(
                                        label=_i18n("f0_method"),
                                        choices=f0_methods,
                                        value=f0_methods[0],
                                        **base_c_params["base"]
                                    )

                                    vbach_crepe_hop_length = gr.Slider(
                                        label=_i18n("crepe_hop_length"),
                                        info=_i18n("crepe_hop_length_info"),
                                        minimum=24,
                                        maximum=512,
                                        step=8,
                                        value=128,
                                        visible=False,
                                        **base_c_params["base"]
                                    )
                                    vbach_f0method.change(lambda x: gr.update(visible=x in crepe_like_f0_methods), inputs=vbach_f0method, outputs=vbach_crepe_hop_length, show_progress="hidden")

                                    vbach_index_rate = gr.Slider(
                                        label=_i18n("index_rate"),
                                        info=_i18n("index_rate_info"),
                                        minimum=0,
                                        maximum=1,
                                        step=0.05,
                                        value=0,
                                        **base_c_params["base"]
                                    )
                                    
                                    vbach_volume_envelope = gr.Slider(
                                        label=_i18n("volume_envelope"),
                                        info=_i18n("volume_envelope_info"),
                                        minimum=0,
                                        maximum=1,
                                        step=0.05,
                                        value=0,
                                        **base_c_params["base"]
                                    )
                                    
                                    vbach_protect = gr.Slider(
                                        label=_i18n("protect"),
                                        info=_i18n("protect_info"),
                                        minimum=0,
                                        maximum=0.5,
                                        step=0.05,
                                        value=0.35,
                                        **base_c_params["base"]
                                    )
                                    
                                    with gr.Accordion(label=_i18n("advanced_params"), open=False):
                                        vbach_use_transformers = gr.Checkbox(
                                            label=_i18n("vbach_use_transformers"),
                                            value=False,
                                            **base_c_params["base"]
                                        )
                                        
                                        vbach_embedder = gr.Dropdown(
                                            label=_i18n("vbach_embedder"),
                                            info=_i18n("vbach_embedder_info"),
                                            choices=huberts_fairseq,
                                            value=huberts_fairseq[0],
                                            **base_c_params["dropdown"]
                                        )
                                        
                                        @vbach_use_transformers.change(inputs=vbach_use_transformers, outputs=vbach_embedder, show_progress="hidden")
                                        def show_embedders_vbach_fn(transformers: bool):
                                            if transformers:
                                                return gr.update(choices=huberts_transformers, value=huberts_transformers[0])
                                            else:
                                                return gr.update(choices=huberts_fairseq, value=huberts_fairseq[0])
                                        
                                        vbach_f0_min = gr.Slider(
                                            label=_i18n("f0_min"),
                                            minimum=50,
                                            maximum=1100,
                                            value=50,
                                            step=1,
                                            **base_c_params["base"]
                                        )
                                        vbach_f0_max = gr.Slider(
                                            label=_i18n("f0_max"),
                                            minimum=350,
                                            maximum=3500,
                                            value=1100,
                                            step=1,
                                            **base_c_params["base"]
                                        )

                                        vbach_chunk_duration = gr.Number(
                                            label=_i18n("chunk_duration"),
                                            minimum=1,
                                            maximum=30,
                                            value=7,
                                            step=1,
                                            **base_c_params["base"]
                                        )
                                        
                                        vbach_stereo_mode = gr.Dropdown(
                                            label=_i18n("stereo_mode"),
                                            info=_i18n("stereo_mode_info"),
                                            choices=stereo_modes,
                                            value=stereo_modes[0],
                                            **base_c_params["base"]
                                        )
                                    
                                    vbach_template = gr.Textbox(
                                        label=_i18n("output_template"), 
                                        info=_i18n("output_vbach_template_info"),
                                        value="MODEL_NAME_F0METHOD_PITCH", 
                                        **base_c_params["base"]
                                    )
                                    vbach_output_format = gr.Dropdown(
                                        label=_i18n("output_format"), 
                                        choices=output_formats, 
                                        value=output_formats[0], 
                                        filterable=False, 
                                        **base_c_params["base"]
                                    )
                                    vbach_convert_btn = gr.Button(_i18n("convert"), variant="primary", **base_c_params["base"])
                        
                        with gr.Group():
                            with gr.Row(equal_height=True):
                                with gr.Column(min_width=110):
                                    gr.Markdown(f"<h4><center>{_i18n('history')}</center></h4>")
                                vbach_history = gr.Dropdown(container=False, scale=13, multiselect=True, max_choices=1, **base_c_params["base"])
                                vbach_history.focus(
                                    self.get_actual_vbach_history_list,
                                    inputs=[vbach_history, vbach_history_state],
                                    outputs=[vbach_history, vbach_history_state],
                                    show_progress="hidden"
                                )
                            vbach_off_players_output = gr.Checkbox(label=_i18n("off_audio_players_output"), info=_i18n("off_audio_players_output_info"), value=False, **base_c_params["base"])
                            vbach_state = gr.State()
                            @gr.render(inputs=[vbach_state, vbach_off_players_output])
                            def show_results(state, off_players_output: bool):
                                if state:
                                    zip_is_generated = False
                                    for result_path in state:
                                        with gr.Group():
                                            if off_players_output:
                                                define_download_button_with_size(
                                                    value=result_path,
                                                    label=Path(result_path).stem,
                                                    **base_c_params["base"],
                                                    variant="huggingface",
                                                    scale=15,
                                                )
                                            else:
                                                define_audio_with_size(
                                                    value=result_path,
                                                    label=Path(result_path).stem,
                                                    **base_c_params["output_audio"]
                                                )

                                    generate_zip_btn = gr.DownloadButton(label=_i18n("generate_zip_archive"), variant="huggingface", **base_c_params["base"])
                                    @generate_zip_btn.click(outputs=generate_zip_btn, trigger_mode="once")
                                    def generate_zip_fn():
                                        nonlocal zip_is_generated
                                        if zip_is_generated:
                                            return gr.skip()
                                        else:
                                            zip_file = generate_zip_archive(state, get_zip_output_path("vbach"))
                                            zip_is_generated = True
                                            return gr.DownloadButton(label=_i18n("download_zip_archive"), variant="huggingface", value=zip_file, **base_c_params["base"])
                                else:
                                    gr.Markdown(f"<h3><center>{_i18n('no_conversion_results')}</center></h3>", container=True)

                        
                        @vbach_convert_btn.click(
                            inputs=[vbach_input_files, vbach_model_path, vbach_index_path, vbach_pitch, vbach_f0method, 
                                    vbach_index_rate, vbach_volume_envelope, vbach_protect, vbach_crepe_hop_length,
                                    vbach_chunk_duration, vbach_stereo_mode, vbach_embedder, vbach_use_transformers,
                                    vbach_template, vbach_output_format, vbach_f0_min, vbach_f0_max],
                            outputs=[vbach_state, vbach_upload_files],
                            trigger_mode="once", concurrency_id="mvsepless_app_inference_vbach"
                        )
                        def convert_wrapper(
                            input_files: list, model_path: str, index_path: str, pitch: float, f0_method: str,
                            index_rate: float, volume_envelope: float, protect: float, hop_length: int,
                            chunk_duration: int, stereo_mode: str, embedder_model: str, use_transformers: bool,
                            template: str, output_format: str, f0_min: int, f0_max: int, progress=gr.Progress(track_tqdm=True)
                        ):
                            
                            if not model_path:
                                gr.Warning(_i18n("model_not_selected"))
                                return [], gr.skip()
                            
                            output_dir = self.output_dir.generate(base_names_app_dirs[7])
                            results = self.vbach_converter.convert_audio(
                                audio_input=input_files,
                                output_dir=output_dir,
                                model_path=one_element_list_to_value(model_path),
                                index_path=one_element_list_to_value(index_path),
                                pitch=pitch,
                                f0_method=f0_method,
                                index_rate=index_rate,
                                volume_envelope=volume_envelope,
                                protect=protect,
                                hop_length=hop_length,
                                embedder_model=embedder_model,
                                use_transformers=use_transformers,
                                output_format=output_format,
                                stereo_mode=stereo_mode,
                                chunk_duration=chunk_duration,
                                f0_min=f0_min,
                                f0_max=f0_max,
                                template=template
                            )
                            
                            self.vbach_history_app.add_to_history(Path(one_element_list_to_value(model_path)).stem, f0_method, pitch, results)
                            return results, gr.skip()
                                    
                        @vbach_history.input(inputs=vbach_history, outputs=vbach_state)
                        def show_history_fn(key: list):
                            state = self.vbach_history_app.get_from_history(one_element_list_to_value(key))
                            return state

                    with gr.Tab(_i18n("f0_extraction_tab"), id="vbach_f0_extract"):
                        with gr.Row():
                            with gr.Column():
                                f0_upload_file = gr.File(show_label=False, **base_c_params["input_file"])
                                with gr.Group():
                                    f0_input_file = gr.Dropdown(container=False, allow_custom_value=True, multiselect=True, max_choices=1, **base_c_params["base"])
                                    f0_input_file.focus(self.get_actual_input_list, inputs=[f0_input_file, f0_input_state], outputs=[f0_input_file, f0_input_state], show_progress="hidden")
                                    f0_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                                    
                                    @f0_upload_file.upload(inputs=f0_upload_file, outputs=[f0_upload_file, f0_input_file])
                                    def upload_f0_file_fn(file: str):
                                        uploaded_files = self.input_files.upload([file])
                                        all_uploaded_files = self.input_files.get_input_list()
                                        if uploaded_files:
                                            first_value = [uploaded_files[0]]
                                        else:
                                            first_value = []
                                        return gr.update(value=None), gr.update(choices=all_uploaded_files, value=first_value)
                                    
                                    @gr.render(inputs=[f0_input_file, f0_input_preview_check])
                                    def preview_f0_input(input: list, preview: bool):
                                        if preview and input:
                                            define_audio_with_size(basename=True, label="", value=one_element_list_to_value(input), **base_c_params["output_audio"])
                            
                            with gr.Column():
                                with gr.Group():
                                    f0_method_dropdown = gr.Dropdown(
                                        label=_i18n("f0_method"),
                                        choices=f0_methods,
                                        value=f0_methods[0],
                                        **base_c_params["base"]
                                    )
                                    
                                    f0_min_slider = gr.Slider(
                                        label=_i18n("f0_min"),
                                        minimum=50,
                                        maximum=1100,
                                        value=50,
                                        step=1,
                                        **base_c_params["base"]
                                    )
                                    
                                    f0_max_slider = gr.Slider(
                                        label=_i18n("f0_max"),
                                        minimum=350,
                                        maximum=3500,
                                        value=1100,
                                        step=1,
                                        **base_c_params["base"]
                                    )
                                    
                                    f0_extract_btn = gr.Button(_i18n("extract_f0"), variant="primary", **base_c_params["base"])
                        
                        with gr.Group():
                            with gr.Row(equal_height=True):
                                with gr.Column(min_width=110):
                                    gr.Markdown(_i18n("f0_file_info"), container=True)
                                    gr.Markdown(f"<h4><center>{_i18n('f0_extraction_results')}</center></h4>")
                                    f0_result_file = gr.File(value=None, label=_i18n("download_f0_json"), type="filepath", interactive=False)
                                    with gr.Row(equal_height=True):
                                        f0_to_corrector_btn = gr.Button(_i18n("f0_send_to_corrector"), variant="secondary", **base_c_params["base"])
                                        f0_to_custom_inference_btn = gr.Button(_i18n("f0_send_to_custom_inference"), variant="primary", **base_c_params["base"])

                        @f0_extract_btn.click(
                            inputs=[f0_input_file, f0_method_dropdown, f0_min_slider, f0_max_slider],
                            outputs=[f0_result_file, f0_upload_file],
                            trigger_mode="once", concurrency_id="mvsepless_app_inference"
                        )
                        def extract_f0_wrapper(input_file: list, method: str, f0_min: int, f0_max: int, progress=gr.Progress(track_tqdm=True)):
                            if not input_file:
                                gr.Warning(_i18n("no_audio_selected"))
                                return gr.skip(), gr.skip()
                            
                            audio_path = one_element_list_to_value(input_file)
                            input_name = Path(audio_path).stem
                            result_path = f0_extract_and_write(
                                audio_path, 
                                f0_method=method, 
                                f0_min=f0_min, 
                                f0_max=f0_max, 
                                output_path=Namer.iter(self.f0_gen_output_path.generate_output_path(input_name, method))
                            )
                            gr.Info(title=_i18n("f0_extraction_complete"), message="")
                            return result_path, gr.skip()

                    with gr.Tab(_i18n("f0_corrector_tab"), id="vbach_f0_correct"):
                        with gr.Row():
                            with gr.Column():
                                f0c_upload_audio = gr.File(label=_i18n("f0_corrector_upload_audio"), elem_id="f0c_upload_audio", type="filepath", **base_c_params["base"])
                                def f0c_upload_audio_fn(file: str, request: gr.Request):
                                    if not file:
                                        return gr.skip()
                                    uploaded = self.input_files.upload([file])
                                    if uploaded:
                                        self.f0_corrector_inbox.setdefault(request.session_hash, {})["audio_path"] = Path(uploaded[0]).as_posix()
                                    return gr.update(value=None)
                                f0c_upload_audio.upload(
                                    f0c_upload_audio_fn, inputs=f0c_upload_audio, outputs=f0c_upload_audio
                                ).then(js=F0C_TRIGGER_JS)                       # без inputs — выполнится гарантированно
                            with gr.Column():
                                f0c_upload_f0 = gr.File(label=_i18n("f0_corrector_upload_f0"), elem_id="f0c_upload_f0", file_types=[".json"], type="filepath", **base_c_params["base"])
                                def f0c_upload_f0_fn(file: str, request: gr.Request):
                                    if not file:
                                        return gr.skip()
                                    dst = Namer.iter(self.f0_gen_output_path.f0_curves_dir / Path(file).name)
                                    shutil.copy2(file, dst)
                                    self.f0_corrector_inbox.setdefault(request.session_hash, {})["f0_path"] = str(dst)
                                    return gr.update(value=None)
                                f0c_upload_f0.upload(
                                    f0c_upload_f0_fn, inputs=f0c_upload_f0, outputs=f0c_upload_f0
                                ).then(js=F0C_TRIGGER_JS)

                        with gr.Group():
                            f0_corrector_state = gr.Textbox(elem_id="hidden_f0_corrector_state")
                            f0_corrector_editor = gr.HTML(
                                """
                                <iframe 
                                    id="f0c-editor-iframe"
                                    src="about:blank" 
                                    width="100%" 
                                    height="600px" 
                                    style="border:none;"
                                    onload="
                                        const urlParams = new URLSearchParams(window.location.search)
                                        const themeValue = urlParams.get('__theme');
                                        const targetTheme = (themeValue === 'dark') ? 'dark' : 'light';
                                        function getSessionHash() {
                                            try {
                                                if (window.parent && window.parent.gradio_config) {
                                                    return window.parent.gradio_config.session_hash || '';
                                                }
                                            } catch(e) {}
                                            try {
                                                if (window.gradio_config) {
                                                    return window.gradio_config.session_hash || '';
                                                }
                                            } catch(e) {}
                                            return '';
                                        }
                                        const sessionHash = getSessionHash();
                                        if(this.src.includes('about:blank')) {
                                            this.src = '/f0_corrector_editor?__theme=' + targetTheme + '&session_hash=' + sessionHash;
                                        }
                                    ">
                                </iframe>
                                """, max_height="900px", padding=False
                            )

                    with gr.Tab(_i18n("vbach_inference_custom_f0"), id="vbach_custom_f0"):
                        with gr.Row():
                            with gr.Column():
                                vbach_custom_upload_file = gr.File(show_label=False, **base_c_params["input_file"])
                                with gr.Group():
                                    vbach_custom_input_file = gr.Dropdown(container=False, allow_custom_value=True, multiselect=True, max_choices=1, **base_c_params["base"])
                                    vbach_custom_input_file.focus(self.get_actual_input_list, inputs=[vbach_custom_input_file, vbach_custom_input_state], outputs=[vbach_custom_input_file, vbach_custom_input_state], show_progress="hidden")
                                    vbach_custom_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                                    
                                    @vbach_custom_upload_file.upload(inputs=vbach_custom_upload_file, outputs=[vbach_custom_upload_file, vbach_custom_input_file])
                                    def upload_vbach_custom_file_fn(file: str):
                                        uploaded_files = self.input_files.upload([file])
                                        all_uploaded_files = self.input_files.get_input_list()
                                        if uploaded_files:
                                            first_value = [uploaded_files[0]]
                                        else:
                                            first_value = []
                                        return gr.update(value=None), gr.update(choices=all_uploaded_files, value=first_value)
                                    
                                    @gr.render(inputs=[vbach_custom_input_file, vbach_custom_input_preview_check])
                                    def preview_vbach_custom_input(input: list, preview: bool):
                                        if preview and input:
                                            define_audio_with_size(basename=True, label="", value=one_element_list_to_value(input), **base_c_params["output_audio"])
                            
                            with gr.Column():
                                with gr.Group():
                                    vbach_custom_model_path = gr.Dropdown(
                                        label=_i18n("model_path"), multiselect=True, allow_custom_value=True, max_choices=1, **base_c_params["base"])
                                    vbach_custom_model_path.focus(
                                        self.get_actual_vbach_models_list,
                                        inputs=[vbach_custom_model_path, vbach_custom_models_state],
                                        outputs=[vbach_custom_model_path, vbach_custom_models_state],
                                        show_progress="hidden"
                                    )
                                    
                                    vbach_custom_index_path = gr.Dropdown(
                                        label=_i18n("index_path"), multiselect=True, allow_custom_value=True, max_choices=1, **base_c_params["base"])
                                    vbach_custom_index_path.focus(
                                        self.get_actual_vbach_index_list,
                                        inputs=[vbach_custom_index_path, vbach_custom_index_state],
                                        outputs=[vbach_custom_index_path, vbach_custom_index_state],
                                        show_progress="hidden"
                                    )
                                    gr.Markdown(_i18n("f0_file_info"), container=True)
                                    vbach_custom_f0_file = gr.File(
                                        label=_i18n("f0_json_file"),
                                        file_types=[".json"],
                                        type="filepath",
                                        **base_c_params["base"]
                                    )
                                    
                                    vbach_custom_pitch = gr.Slider(
                                        label=_i18n("pitch"),
                                        minimum=-36,
                                        maximum=36,
                                        step=0.1,
                                        value=0,
                                        **base_c_params["base"]
                                    )
                                    
                                    vbach_custom_index_rate = gr.Slider(
                                        label=_i18n("index_rate"),
                                        info=_i18n("index_rate_info"),
                                        minimum=0,
                                        maximum=1,
                                        step=0.05,
                                        value=0,
                                        **base_c_params["base"]
                                    )
                                    
                                    vbach_custom_volume_envelope = gr.Slider(
                                        label=_i18n("volume_envelope"),
                                        info=_i18n("volume_envelope_info"),
                                        minimum=0,
                                        maximum=1,
                                        step=0.05,
                                        value=0,
                                        **base_c_params["base"]
                                    )
                                    
                                    vbach_custom_protect = gr.Slider(
                                        label=_i18n("protect"),
                                        info=_i18n("protect_info"),
                                        minimum=0,
                                        maximum=0.5,
                                        step=0.05,
                                        value=0.35,
                                        **base_c_params["base"]
                                    )
                                    
                                    with gr.Accordion(label=_i18n("advanced_params"), open=False):
                                        vbach_custom_use_transformers = gr.Checkbox(
                                            label=_i18n("vbach_use_transformers"),
                                            value=False,
                                            **base_c_params["base"]
                                        )
                                        
                                        vbach_custom_embedder = gr.Dropdown(
                                            label=_i18n("vbach_embedder"),
                                            info=_i18n("vbach_embedder_info"),
                                            choices=huberts_fairseq,
                                            value=huberts_fairseq[0],
                                            **base_c_params["dropdown"]
                                        )
                                        
                                        @vbach_custom_use_transformers.change(inputs=vbach_custom_use_transformers, outputs=vbach_custom_embedder, show_progress="hidden")
                                        def show_embedders_vbach_custom_fn(transformers: bool):
                                            if transformers:
                                                return gr.update(choices=huberts_transformers, value=huberts_transformers[0])
                                            else:
                                                return gr.update(choices=huberts_fairseq, value=huberts_fairseq[0])
                                        
                                        vbach_custom_f0_min = gr.Slider(
                                            label=_i18n("f0_min"),
                                            minimum=50,
                                            maximum=1100,
                                            value=50,
                                            step=1,
                                            **base_c_params["base"]
                                        )
                                        vbach_custom_f0_max = gr.Slider(
                                            label=_i18n("f0_max"),
                                            minimum=350,
                                            maximum=3500,
                                            value=1100,
                                            step=1,
                                            **base_c_params["base"]
                                        )

                                        vbach_custom_chunk_duration = gr.Number(
                                            label=_i18n("chunk_duration"),
                                            minimum=1,
                                            maximum=30,
                                            value=7,
                                            step=1,
                                            **base_c_params["base"]
                                        )
                                        
                                    
                                    vbach_custom_template = gr.Textbox(
                                        label=_i18n("output_template"), 
                                        info=_i18n("output_vbach_custom_template_info"),
                                        value="MODEL_NAME_F0METHOD_PITCH", 
                                        **base_c_params["base"]
                                    )
                                    vbach_custom_output_format = gr.Dropdown(
                                        label=_i18n("output_format"), 
                                        choices=output_formats, 
                                        value=output_formats[0], 
                                        filterable=False, 
                                        **base_c_params["base"]
                                    )
                                    vbach_custom_convert_btn = gr.Button(_i18n("convert_custom_f0"), variant="primary", **base_c_params["base"])
                        
                        with gr.Row(equal_height=True):
                            vbach_custom_output_audio = gr.Audio(
                                value=None,
                                label=_i18n("vbach_result"),
                                **base_c_params["output_audio"]
                            )
                        
                        @vbach_custom_convert_btn.click(
                            inputs=[vbach_custom_input_file, vbach_custom_model_path, vbach_custom_index_path, 
                                    vbach_custom_pitch, vbach_custom_f0_file, vbach_custom_index_rate, 
                                    vbach_custom_volume_envelope, vbach_custom_protect, vbach_custom_embedder, 
                                    vbach_custom_use_transformers, vbach_custom_template, vbach_custom_output_format, 
                                    vbach_custom_f0_min, vbach_custom_f0_max, vbach_custom_chunk_duration],
                            outputs=[vbach_custom_output_audio, vbach_custom_upload_file],
                            trigger_mode="once", concurrency_id="mvsepless_app_inference_vbach"
                        )
                        def convert_custom_f0_wrapper(
                            input_files: list, model_path: str, index_path: str, pitch: float, f0_file_path: str, 
                            index_rate: float, volume_envelope: float, protect: float, embedder_model: str, 
                            use_transformers: bool, template: str, output_format: str, f0_min: int, f0_max: int, 
                            chunk_duration: int, progress=gr.Progress(track_tqdm=True)
                        ):
                            if not model_path:
                                gr.Warning(_i18n("model_not_selected"))
                                return update_audio_with_size(label=_i18n("vbach_result"), value=None), gr.skip()
                            
                            if not f0_file_path:
                                gr.Warning(_i18n("no_f0_file_selected"))
                                return update_audio_with_size(label=_i18n("vbach_result"), value=None), gr.skip()
                            
                            output_dir = self.output_dir.generate(base_names_app_dirs[7])
                            
                            result = self.vbach_converter.convert_audio_custom_f0(
                                audio_input=one_element_list_to_value(input_files),
                                output_dir=output_dir,
                                model_path=one_element_list_to_value(model_path),
                                index_path=one_element_list_to_value(index_path),
                                pitch=pitch,
                                f0_file=f0_file_path,
                                index_rate=index_rate,
                                volume_envelope=volume_envelope,
                                protect=protect,
                                embedder_model=embedder_model,
                                use_transformers=use_transformers,
                                output_format=output_format,
                                f0_min=f0_min,
                                f0_max=f0_max,
                                chunk_duration=chunk_duration,
                                template=template
                            )
                            self.vbach_history_app.add_to_history(Path(one_element_list_to_value(model_path)).stem, "custom", pitch, [result])
                            return update_audio_with_size(label="", basename=True, value=result), gr.skip()
                        
                    @f0_corrector_state.input(
                        inputs=f0_corrector_state,
                        outputs=[vbach_custom_f0_file, vbach_inner_tabs]
                    )

                    def f0_corrector_to_inference_fn(path: str):
                        if path and Path(path).exists():
                            gr.Info(title=_i18n("f0_corrector_sent_success"), message="")
                            return gr.update(value=path), gr.Tabs(selected="vbach_custom_f0")
                        return gr.skip(), gr.skip()

                    def f0_send_to_corrector_fn(input_file: list, f0_path: str, request: gr.Request):
                        if not f0_path or not Path(f0_path).exists():
                            gr.Warning(_i18n("no_f0_extracted"))
                            return gr.skip()
                        payload_d = {"f0_path": f0_path}
                        audio_path = one_element_list_to_value(input_file)
                        if audio_path:
                            payload_d["audio_path"] = audio_path
                        self.f0_corrector_inbox.setdefault(request.session_hash, {}).update(payload_d)  # данные — в inbox ДО ответа
                        return gr.Tabs(selected="vbach_f0_correct")

                    f0_to_corrector_btn.click(
                        f0_send_to_corrector_fn,
                        inputs=[f0_input_file, f0_result_file],
                        outputs=vbach_inner_tabs
                    ).then(js=F0C_TRIGGER_JS)                                   # сигнал, не данные

                    @f0_to_custom_inference_btn.click(
                        inputs=[f0_input_file, f0_result_file],
                        outputs=[vbach_custom_input_file, vbach_custom_f0_file, vbach_inner_tabs]
                    )
                    def f0_send_to_custom_inference_fn(input_file: list, f0_path: str):
                        if not f0_path or not Path(f0_path).exists():
                            gr.Warning(_i18n("no_f0_extracted"))
                            return gr.skip(), gr.skip(), gr.skip()
                        audio = one_element_list_to_value(input_file)
                        return (
                            gr.update(value=[audio]) if audio else gr.skip(),
                            gr.update(value=f0_path),
                            gr.Tabs(selected="vbach_custom_f0")
                        )










            with gr.Tab(_i18n("upload_manager")):
                with gr.Tab(_i18n("upload_audio")):
                    with gr.Row():
                        with gr.Accordion(label=_i18n("upload_from_zip"), open=True):
                            with gr.Group():
                                gr.Markdown("<h3><center>"+_i18n("upload_zip_placeholder")+"</center></h3>")
                                upload_zip_file = gr.File(
                                    show_label=False, 
                                    type="filepath", 
                                    file_count="single", 
                                    file_types=[".zip"],
                                    **base_c_params["base"]
                                )
                                upload_zip_extract_btn = gr.Button(_i18n("extract_and_upload"), variant="primary", **base_c_params["base"])
                                upload_zip_status = gr.Textbox(label=_i18n("status"), value="", interactive=False, lines=3, max_lines=3)
                                
                                @upload_zip_extract_btn.click(inputs=upload_zip_file, outputs=[upload_zip_status, upload_zip_file])
                                def extract_and_upload_zip(zip_path: str, progress=gr.Progress(track_tqdm=True)):
                                    if not zip_path:
                                        return _i18n("path_not_specified"), gr.skip()
                                    
                                    extracted_files = []
                                    with tempfile.TemporaryDirectory() as tmpdirname:
                                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                            zip_ref.extractall(tmpdirname)
                                        
                                        tmp_path = Path(tmpdirname)
                                        audio_files = get_audio_files_from_list([f.as_posix() for f in tmp_path.rglob("*")], only_files=False)
                                        
                                        if audio_files:
                                            uploaded = self.input_files.upload(audio_files, copy=True)
                                            extracted_files.extend(uploaded)
                                            status = _i18n("uploaded_files_count", count=len(uploaded))
                                        else:
                                            status = _i18n("no_audio_files_in_zip")
                                    
                                    gr.Info(title=status, message="")
                                    return status, gr.update(value=None)
                        
                        with gr.Accordion(label=_i18n("upload_from_files"), open=True):
                            with gr.Group():
                                upload_files_component = gr.File(
                                    show_label=False,
                                    **base_c_params["input_files_multi"]
                                )
                                upload_files_btn = gr.Button(_i18n("upload"), variant="primary", **base_c_params["base"])
                                upload_files_status = gr.Textbox(label=_i18n("status"), value="", interactive=False, lines=3, max_lines=3)
                                
                                @upload_files_btn.click(inputs=upload_files_component, outputs=[upload_files_status, upload_files_component])
                                def upload_audio_files(files: list, progress=gr.Progress(track_tqdm=True)):
                                    if not files:
                                        return _i18n("paths_not_specified"), gr.skip()
                                    
                                    uploaded = self.input_files.upload(files, copy=True)
                                    status = _i18n("uploaded_files_count", count=len(uploaded))
                                    gr.Info(title=status, message="")
                                    return status, gr.update(value=[])

                    with gr.Accordion(label=_i18n("upload_from_url"), open=False):
                        with gr.Group():
                            gr.Markdown("<h3><center>"+_i18n("supported_yt_dlp_info")+"</center></h3>")
                            
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=3):
                                    upload_url_input = gr.Textbox(
                                        label=_i18n("audio_url"), 
                                        placeholder="https://example.com/audio.mp3", 
                                        **base_c_params["base"]
                                    )
                                with gr.Column(scale=1):
                                    upload_url_btn = gr.Button(_i18n("download_and_upload"), variant="primary", **base_c_params["base"])
                            
                            with gr.Row():
                                with gr.Column(scale=1):
                                    upload_url_format = gr.Dropdown(
                                        label=_i18n("output_format"),
                                        choices=output_formats,
                                        value=output_formats[0],
                                        **base_c_params["dropdown"]
                                    )
                                with gr.Column(scale=1):
                                    upload_url_bitrate = gr.Dropdown(
                                        label=_i18n("bitrate"),
                                        choices=["64", "128", "192", "256", "320"],
                                        value="320",
                                        **base_c_params["dropdown"]
                                    )
                            
                            with gr.Accordion(label=_i18n("cookie_settings"), open=False):
                                gr.Markdown(_i18n("cookie_explanation"))
                                
                                with gr.Row():
                                    with gr.Column(scale=2):
                                        upload_cookie_file = gr.File(
                                            label=_i18n("cookie_file"),
                                            file_types=[".txt", ".netscape"],
                                            **base_c_params["input_file"]
                                        )
                                    with gr.Column(scale=1):
                                        upload_cookie_status = gr.Textbox(
                                            label=_i18n("cookie_status"),
                                            value=_i18n("cookie_not_loaded"),
                                            interactive=False,
                                            lines=5
                                        )
                                
                                @upload_cookie_file.change(inputs=upload_cookie_file, outputs=upload_cookie_status)
                                def cookie_file_selected(file: str):
                                    if file and Path(file).exists():
                                        return _i18n("cookie_loaded", path=file)
                                    return _i18n("cookie_not_loaded")
                            
                            upload_url_status = gr.Textbox(
                                label=_i18n("status"), 
                                value="", 
                                interactive=False, 
                                lines=5, 
                                max_lines=5
                            )
                            
                            @upload_url_btn.click(
                                inputs=[upload_url_input, upload_url_format, upload_url_bitrate, upload_cookie_file], 
                                outputs=[upload_url_status, upload_url_input]
                            )
                            def download_from_url(url: str, audio_format: str, bitrate: str, cookie_path: str, progress=gr.Progress(track_tqdm=True)):
                                if not url:
                                    return _i18n("path_not_specified"), gr.skip()
                                
                                try:
                                    # Используем dw_yt_dlp для скачивания
                                    output_dir = self.input_files.input_dir_base / "yt_downloads"
                                    output_dir.mkdir(parents=True, exist_ok=True)
                                    
                                    result_path = dw_yt_dlp(
                                        url=url,
                                        output_dir=output_dir,
                                        output_format=audio_format,
                                        output_bitrate=bitrate,
                                        cookie=cookie_path if cookie_path else None
                                    )
                                    
                                    if result_path and Path(result_path).exists():
                                        # Загружаем скачанный файл в базу входных файлов
                                        uploaded = self.input_files.upload([result_path], copy=True)
                                        status = _i18n("downloaded_and_uploaded", count=len(uploaded))
                                        gr.Info(title=status, message="")
                                        return status, gr.update(value="")
                                    else:
                                        return _i18n("download_failed_no_file"), gr.skip()
                                        
                                except Exception as e:
                                    return _i18n("download_error", error=str(e)), gr.skip()
                    
                    with gr.Accordion(label=_i18n("upload_from_path"), open=False):
                        with gr.Group():
                            upload_path_input = gr.Textbox(label=_i18n("folder_path"), placeholder="/path/to/audio/folder", **base_c_params["base"])
                            upload_path_btn = gr.Button(_i18n("scan_and_upload"), variant="primary", **base_c_params["base"])
                            upload_path_status = gr.Textbox(label=_i18n("status"), value="", interactive=False, lines=5, max_lines=5)
                            
                            @upload_path_btn.click(inputs=upload_path_input, outputs=[upload_path_status, upload_path_input])
                            def upload_from_path(path: str, progress=gr.Progress(track_tqdm=True)):
                                if not path:
                                    return _i18n("path_not_specified"), gr.skip()
                                
                                path_obj = Path(path)
                                if not path_obj.exists():
                                    return _i18n("path_not_exist"), gr.skip()
                                
                                if path_obj.is_file():
                                    if check(path_obj):
                                        uploaded = self.input_files.upload([path_obj.as_posix()], copy=True)
                                        status = _i18n("uploaded_files_count", count=len(uploaded))
                                    else:
                                        status = _i18n("file_is_not_audio")
                                elif path_obj.is_dir():
                                    audio_files = get_audio_files_from_list([f.as_posix() for f in path_obj.rglob("*")], only_files=False)
                                    if audio_files:
                                        uploaded = self.input_files.upload(audio_files, copy=True)
                                        status = _i18n("uploaded_files_count", count=len(uploaded))
                                    else:
                                        status = _i18n("no_audio_files_in_directory")
                                else:
                                    status = _i18n("invalid_path")
                                
                                gr.Info(title=status, message="")
                                return status, gr.skip()
                    
                with gr.Tab(_i18n("download_model")):
                    with gr.Tab(_i18n("separation_tab")):
                        with gr.Row():
                            # ===== Колонка 1: скачивание из списка =====
                            with gr.Column():
                                with gr.Group():
                                    gr.Markdown("<h3><center>"+_i18n("from_catalog")+"</center></h3>", container=True)
                                    sep_download_model_name = gr.Dropdown(label=_i18n("model_name"), choices=all_models, value=default_model, **base_c_params["base"])
                                    sep_download_btn = gr.Button(_i18n("download"), variant="primary", **base_c_params["base"])
                                    sep_download_status = gr.Textbox(label=_i18n("status"), value="", interactive=False, lines=3, max_lines=3)
                                    @sep_download_btn.click(inputs=sep_download_model_name, outputs=sep_download_status)
                                    def download_sep_model_fn(name: str, progress=gr.Progress(track_tqdm=True)):
                                        status = self.separator.download(name, return_status=True)
                                        gr.Info(title=status, message="")
                                        return status
                            # ===== Колонка 2: кастомные модели =====
                            with gr.Column():
                                with gr.Group():
                                    gr.Markdown("<h3><center>"+_i18n("custom_separation_models_tab")+"</center></h3>", container=True)
                                    with gr.Accordion(label=_i18n("download_from_internet"), open=True):
                                        gr.Markdown("<h3><center>"+_i18n("supported_only_direct_links")+"</center></h3>")
                                        with gr.Group():
                                            custom_sep_url_checkpoint = gr.Textbox(label=_i18n("custom_checkpoint_link"), **base_c_params["base"])
                                            custom_sep_url_config = gr.Textbox(label=_i18n("custom_config_link"), **base_c_params["base"])
                                            custom_sep_url_model_btn = gr.Button(_i18n("download_and_move_to_models_dir"), variant="primary", **base_c_params["base"])
                                            custom_sep_url_model_status = gr.Textbox(label=_i18n("status"), value="", interactive=False, lines=3, max_lines=3)
                                            @custom_sep_url_model_btn.click(inputs=[custom_sep_url_checkpoint, custom_sep_url_config], outputs=custom_sep_url_model_status)
                                            def download_custom_sep_files_fn(url_checkpoint: str, url_config: str, progress=gr.Progress(track_tqdm=True)):
                                                status = self.custom_sep_model_manager.download_model(checkpoint_url=url_checkpoint, config_url=url_config)
                                                return status
                                    with gr.Accordion(label=_i18n("download_from_local_device"), open=False):
                                        with gr.Row():
                                            with gr.Column():
                                                with gr.Group():
                                                    gr.Markdown("<h3><center>"+_i18n("custom_checkpoint_placeholder")+"</center></h3>")
                                                    custom_sep_local_checkpoint = gr.File(show_label=False, type="filepath", file_count="multiple", file_types=[".pth", ".ckpt", ".pt", ".chpt"], **base_c_params["base"])
                                                    @custom_sep_local_checkpoint.upload(inputs=custom_sep_local_checkpoint, outputs=custom_sep_local_checkpoint)
                                                    def upload_custom_sep_checkpoint_fn(files: list, progress=gr.Progress(track_tqdm=True)):
                                                        self.custom_sep_model_manager.upload_checkpoint(files)
                                                        return gr.update(value=[])
                                            with gr.Column():
                                                with gr.Group():
                                                    gr.Markdown("<h3><center>"+_i18n("custom_config_placeholder")+"</center></h3>")
                                                    custom_sep_local_config = gr.File(show_label=False, type="filepath", file_count="multiple", file_types=[".yaml", ".yml"], **base_c_params["base"])
                                                    @custom_sep_local_config.upload(inputs=custom_sep_local_config, outputs=custom_sep_local_config)
                                                    def upload_custom_sep_config_fn(files: list, progress=gr.Progress(track_tqdm=True)):
                                                        self.custom_sep_model_manager.upload_config(files)
                                                        return gr.update(value=[])
                                                    
                    with gr.Tab(_i18n("vbach_models_tab")):
                        gr.Markdown("<h3><center>"+_i18n("supported_only_direct_links")+"</center></h3>")
                        with gr.Row():
                            with gr.Column():
                                with gr.Group():
                                    gr.Markdown("<h3><center>"+_i18n("download_model_files_from_zip")+"</center></h3>", container=True)
                                    with gr.Accordion(label=_i18n("download_from_internet"), open=True):
                                        with gr.Group():
                                            vbach_url_model_zip = gr.Textbox(label=_i18n("vbach_zip_link"), **base_c_params["base"])
                                            vbach_url_model_zip_btn = gr.Button(_i18n("download_and_unzip"), variant="primary", **base_c_params["base"])
                                            vbach_url_model_zip_status = gr.Textbox(label=_i18n("status"), value="", interactive=False, lines=3, max_lines=3)
                                            @vbach_url_model_zip_btn.click(inputs=vbach_url_model_zip, outputs=vbach_url_model_zip_status)
                                            def download_vbach_zip_fn(url: str, progress=gr.Progress(track_tqdm=True)):
                                                status = self.vbach_model_manager.download_model(zip_url=url)
                                                return status

                                    with gr.Accordion(label=_i18n("download_from_local_device"), open=False):
                                        with gr.Group():
                                            gr.Markdown("<h3><center>"+_i18n("vbach_zip_placeholder")+"</center></h3>")
                                            vbach_local_model_zip = gr.File(show_label=False, type="filepath", file_count="single", file_types=[".zip"], **base_c_params["base"])
                                            @vbach_local_model_zip.upload(inputs=vbach_local_model_zip, outputs=vbach_local_model_zip)
                                            def upload_vbach_zip_fn(file: str, progress=gr.Progress(track_tqdm=True)):
                                                self.vbach_model_manager.extract_zip(file)
                                                return gr.update(value=None)

                            with gr.Column():
                                with gr.Group():
                                    gr.Markdown("<h3><center>"+_i18n("download_model_files")+"</center></h3>", container=True)
                                    with gr.Accordion(label=_i18n("download_from_internet"), open=True):
                                        with gr.Group():
                                            vbach_url_model_pth = gr.Textbox(label=_i18n("vbach_pth_link"), **base_c_params["base"])
                                            vbach_url_model_index = gr.Textbox(label=_i18n("vbach_index_link"), **base_c_params["base"])
                                            vbach_url_model_pth_btn = gr.Button(_i18n("download_and_move_to_models_dir"), variant="primary", **base_c_params["base"])
                                            vbach_url_model_pth_status = gr.Textbox(label=_i18n("status"), value="", interactive=False, lines=3, max_lines=3)
                                            @vbach_url_model_pth_btn.click(inputs=[vbach_url_model_pth, vbach_url_model_index], outputs=vbach_url_model_pth_status)
                                            def download_vbach_pth_fn(url_pth: str, url_index: str, progress=gr.Progress(track_tqdm=True)):
                                                status = self.vbach_model_manager.download_model(pth_url=url_pth, index_url=url_index)
                                                return status
                                    with gr.Accordion(label=_i18n("download_from_local_device"), open=False):
                                        with gr.Row():
                                            with gr.Column():
                                                with gr.Group():
                                                    gr.Markdown("<h3><center>"+_i18n("vbach_checkpoint_pth_placeholder")+"</center></h3>")
                                                    vbach_local_model_pth = gr.File(show_label=False, type="filepath", file_count="multiple", file_types=[".pth"], **base_c_params["base"])
                                                    @vbach_local_model_pth.upload(inputs=vbach_local_model_pth, outputs=vbach_local_model_pth)
                                                    def upload_vbach_pth_fn(files: list, progress=gr.Progress(track_tqdm=True)):
                                                        self.vbach_model_manager.upload_pth_model(files)
                                                        return gr.update(value=[])
                                            with gr.Column():
                                                with gr.Group():
                                                    gr.Markdown("<h3><center>"+_i18n("vbach_index_file_placeholder")+"</center></h3>")
                                                    vbach_local_model_index = gr.File(show_label=False, type="filepath", file_count="multiple", file_types=[".index"], **base_c_params["base"])
                                                    @vbach_local_model_index.upload(inputs=vbach_local_model_index, outputs=vbach_local_model_index)
                                                    def upload_vbach_index_fn(files: list, progress=gr.Progress(track_tqdm=True)):
                                                        self.vbach_model_manager.upload_index_model(files)
                                                        return gr.update(value=[])
                                        
            if GDRIVE_USER_DIR:
                with gr.Tab(_i18n("google_drive")):
                    gdrive_info = gr.Textbox(lines=3, label=_i18n("status"), interactive=False)
                    gr.Timer().tick(lambda: gr.update(value=get_disk_usage(GDRIVE_DIR)), outputs=gdrive_info)
                    copy_to_gdrive_btn = gr.Button(_i18n("copy_from_current_user_dir_to_gdrive"), **base_c_params["base"])
                    @copy_to_gdrive_btn.click()
                    def copy_to_gdrive_fn():
                        copy_to_gdrive()
                        self.input_files.update_data(0)
                        self.history.update_data(0)
                        self.auto_ensemble_history_app.update_data(0)
                        self.manual_ensemble_history_app.update_data(0)
                        self.subtract_history_app.update_data(0)
                        self.vbach_history_app.update_data(0)
                        self.iterative_ensemble_history_app.update_data(0)
                        self.preset_history.update_data(0)
                        self.phase_fixer_history_app.update_data(0)

        return gr.mount_gradio_app(app, mvsepless_app, path="/", allowed_paths=["/"])

    def launch(self, theme: Any = None, hf_space_mode: bool = False, server_name: str = "0.0.0.0", server_port: int | str = 8000, share: bool = True):
        if not server_name:
            server_name = "0.0.0.0"
        if not server_port:
            server_port = 7860
        app = self.UI(theme, hf_space_mode)
        if share:
            share_url = share_gradio_tunnel(server_name, server_port)
            print(_i18n("public_url") +": " + share_url)
        uvicorn.run(app, host=server_name, port=server_port)

theme = gr.themes.Base(
    primary_hue=gr.themes.Color(c100="#D5E8F2", c200="#ABD1E6", c300="#80BBD9", c400="#56A4CC", c50="#EAF4F9", c500="#2C8DC0", c600="#0276B3", c700="#025886", c800="#013B5A", c900="#001E2D", c950="#000F16"),
    secondary_hue=gr.themes.Color(c100="#CDCDD6", c200="#9A9BAD", c300="#686985", c400="#35375C", c50="#E6E6EB", c500="#020533", c600="#000637", c700="#00052E", c800="#000424", c900="#00031A", c950="#000210"),
    neutral_hue=gr.themes.Color(c100="#DBEAFE", c200="#BFDBFE", c300="#93C5FD", c400="#60A5FA", c50="#EFF6FF", c500="#3B82F6", c600="#2563EB", c700="#1D4ED8", c800="#1E40AF", c900="#1E3A8A", c950="#172554"),
    spacing_size="sm",
    radius_size="none",
    font=[gr.themes.GoogleFont('Montserrat'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
).set(
    background_fill_primary='#F7F8FA',
    background_fill_primary_dark='*secondary_950',
    background_fill_secondary='*background_fill_primary',
    background_fill_secondary_dark='*background_fill_primary',
    border_color_accent_dark='*secondary_50',
    color_accent='*primary_600',
    color_accent_soft_dark='*neutral_800',
    link_text_color='*secondary_700',
    block_background_fill_dark='*secondary_800',
    block_border_color_dark='*neutral_800',
    input_background_fill='*body_background_fill',
    input_background_fill_dark='*secondary_900',
    input_border_width='1px',
    input_border_width_dark='1px',
    button_primary_background_fill_hover='*body_background_fill',
    button_primary_background_fill_hover_dark='*neutral_400',
    button_primary_text_color_hover='*body_text_color',
    button_secondary_background_fill='*body_background_fill',
    button_secondary_background_fill_dark='*body_background_fill',
    button_secondary_background_fill_hover='*secondary_600',
    button_secondary_background_fill_hover_dark='*neutral_500',
    button_secondary_border_color_dark='*primary_500',
    button_secondary_border_color_hover='*secondary_600',
    button_secondary_text_color_dark='*neutral_600',
    button_secondary_text_color_hover='*button_primary_text_color',
    button_secondary_text_color_hover_dark='*button_primary_text_color',
    button_cancel_background_fill='red',
    button_cancel_background_fill_dark='red',
    button_cancel_background_fill_hover='white',
    button_cancel_background_fill_hover_dark='white',
    button_cancel_text_color='*button_primary_text_color',
    button_cancel_text_color_dark='*button_primary_text_color',
    button_cancel_text_color_hover='red',
    button_cancel_text_color_hover_dark='red',
    checkbox_label_background_fill_hover='*neutral_300',
    table_even_background_fill='*body_background_fill',
    table_even_background_fill_dark='*body_background_fill',
    table_odd_background_fill='*body_background_fill',
    table_odd_background_fill_dark='*body_background_fill'
)

if __name__ == "__main__":
    check_taglib_not_installed()
    args = parse_app_args()

    app = App(
        source=args.model_source,
        custom_model_info_path=args.custom_model_info_path,
        custom_models_dir=args.custom_models_dir
    )

    if not args.full:
        app.update_info()

    app.launch(
        theme=theme,
        hf_space_mode=not args.full,
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )