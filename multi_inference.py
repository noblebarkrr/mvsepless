# multi_inference.py  (исправлен 24.07.2025)
import os
import time
import shutil
import sys
import gc
import argparse
import json
import tempfile
import base64
import subprocess   # <- безопасный CLI
import traceback
import warnings
import importlib.util
import torch
from datetime import datetime
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.utils import mediainfo
import gradio as gr
from pyngrok import ngrok

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from model_list import models_data
from utils.preedit_config import conf_editor
from utils.download_models import download_model
from assets.translations import MVSEPLESS_TRANSLATIONS as TRANSLATIONS
from separator.ensemble import ensemble_audio_files
from separator.audio_writer import write_audio_file

########### CONSTANTS
LANGS = ["ru", "en"]
FONTS_DIR = os.path.join(SCRIPT_DIR, "assets", "fonts")
CONFIG_UI_PATH = os.path.join(SCRIPT_DIR, "config.json")
FAVICON_PATH = os.path.join(SCRIPT_DIR, "assets", "mvsepless.png")
MODELS_CACHE_DIR = os.path.join(SCRIPT_DIR, "separator", "models_cache")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
ENSEMBLESS_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "ensembless_output")
GRADIO_HOST = "0.0.0.0"
GRADIO_PORT = 7860
GRADIO_SHARE = True
GRADIO_DEBUG = True
GRADIO_AUTH = None
CURRENT_LANG = "ru"
GRADIO_MAX_FILE_SIZE = "10000MB"
OUTPUT_FORMATS = ["mp3", "wav", "flac", "ogg", "opus", "m4a", "aac", "aiff"]
CALL_METHODS = ["cli", "direct"]
GOOGLE_FONT = "Tektur"
N_FFT = 2048
WIN_LENGTH = 2048
HOP_LENGTH = WIN_LENGTH // 4
plugins_dir = os.path.join(SCRIPT_DIR, "plugins")

DEFAULT_CONFIG = {
    "inference": {
        "output_dir": OUTPUT_DIR,
        "models_cache_dir": MODELS_CACHE_DIR,
        "output_bitrate": "320k",
        "call_method": "cli"
    },
    "settings": {
        "font": GOOGLE_FONT,
        "auth": GRADIO_AUTH,
        "language": CURRENT_LANG,
        "max_file_size": GRADIO_MAX_FILE_SIZE,
        "port": GRADIO_PORT,
        "debug": GRADIO_DEBUG,
        "share": GRADIO_SHARE
    }
}

def unload_model():
    # удалить из sys.modules все, что начинается с models.
    for k in list(sys.modules):
        if k.startswith('models'):
            del sys.modules[k]
    torch.cuda.empty_cache()
    gc.collect()

def restart_ui():
    python = sys.executable
    os.execl(python, python, *sys.argv)

def upload_plugin_list(files):
    if not files:
        return 
    for file in files:
        try:
            shutil.copy(file, os.path.join(plugins_dir, os.path.basename(file)))
        except Exception as e:
            print(f"Error copying plugin: {e}")
    gr.Warning(t("restart_warning"))
    time.sleep(2)
    restart_ui()

class ConfigManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = DEFAULT_CONFIG.copy()

    def load(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                for section in DEFAULT_CONFIG:
                    self.config.setdefault(section, {})
                    for key in DEFAULT_CONFIG[section]:
                        self.config[section].setdefault(key, DEFAULT_CONFIG[section][key])
            except Exception as e:
                print(f"Error loading config: {e}")
                self.config = DEFAULT_CONFIG.copy()
        else:
            self.save()
        return self.config

    def save(self):
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False

    def update(self, section, key, value):
        if section in self.config and key in self.config[section]:
            self.config[section][key] = value
            self.save()
            return True
        return False

    def get(self, section, key, default=None):
        return self.config.get(section, {}).get(key, default)

    def validate_paths(self):
        paths = [
            self.get("inference", "output_dir"),
            self.get("inference", "models_cache_dir"),
            plugins_dir,
            FONTS_DIR,
            ENSEMBLESS_OUTPUT_DIR
        ]
        for path in paths:
            if path and not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

config_manager = ConfigManager(CONFIG_UI_PATH)
CONFIG = config_manager.load()
config_manager.validate_paths()

########### HELPERS
def t(key, **kwargs):
    lang = CONFIG["settings"]["language"]
    translation = TRANSLATIONS.get(lang, {}).get(key, key)
    return translation.format(**kwargs) if kwargs else translation

def get_font_files():
    fonts = []
    if os.path.exists(FONTS_DIR):
        for filename in os.listdir(FONTS_DIR):
            path = os.path.join(FONTS_DIR, filename)
            if os.path.isfile(path) and path.lower().endswith((".ttf", ".otf", ".woff", ".eot")):
                fonts.append(os.path.abspath(path))
    return fonts

def write_fonts(files):
    for file in files:
        try:
            shutil.copy(file, os.path.join(FONTS_DIR, os.path.basename(file)))
        except Exception as e:
            print(f"Error copying font: {e}")
    return gr.Warning(t("fonts_uploaded"))



########### SINGLE INFERENCE
def single_multi_inference(
    input_file,
    output_dir,
    model_type,
    model_name,
    ext_inst,
    vr_aggr=5,
    output_format="wav",
    template="NAME_(STEM)_MODEL",
    selected_stems=None
):
    if selected_stems is None:
        selected_stems = []

    output_bitrate = CONFIG.get("inference", {}).get("output_bitrate", "320k")
    call_method = CONFIG.get("inference", {}).get("call_method", "cli")

    print(f"Starting inference: {model_type}/{model_name}, bitrate={output_bitrate}, method={call_method}, stems={selected_stems}")
    os.makedirs(output_dir, exist_ok=True)

    # MSST
    if model_type in ["mel_band_roformer", "bs_roformer", "mdx23c", "scnet", "htdemucs", "bandit", "bandit_v2"]:
        info = models_data[model_type][model_name]
        conf, ckpt = download_model(
            CONFIG["inference"]["models_cache_dir"],
            model_name,
            model_type,
            info["checkpoint_url"],
            info["config_url"]
        )
        if model_type != "htdemucs":
            conf_editor(conf)

        if call_method == "cli":
            cmd = [
                "python", "-m", "separator.msst_separator",
                f'--input "{input_file}"',
                f'--store_dir "{output_dir}"',
                f'--model_type "{model_type}"',
                f'--model_name "{model_name}"',
                f'--config_path "{conf}"',
                f'--start_check_point "{ckpt}"',
                f'--output_format "{output_format}"',
                f'--output_bitrate "{output_bitrate}"',
                f'--template "{template}"',
                "--save_results_info"
            ]
            if ext_inst:
                cmd.append("--extract_instrumental")
            if selected_stems:
                instruments = " ".join(f'"{s}"' for s in selected_stems)
                cmd.append(f'--selected_instruments {instruments}')
            subprocess.run(" ".join(cmd), shell=True, check=True)

            results_path = os.path.join(output_dir, "results.json")
            if os.path.exists(results_path):
                with open(results_path, encoding="utf-8") as f:
                    return json.load(f)
            else:
                return [("None", "/none/none.mp3")]

        elif call_method == "direct":
            from separator.msst_separator import mvsep_offline
            try:
                results = mvsep_offline(
                    input_path=input_file,
                    store_dir=output_dir,
                    model_type=model_type,
                    config_path=conf,
                    start_check_point=ckpt,
                    extract_instrumental=ext_inst,
                    output_format=output_format,
                    output_bitrate=output_bitrate,
                    model_name=model_name,
                    template=template,
                    selected_instruments=selected_stems
                )
            except Exception as e:
                print(e)
                return [("None", "/none/none.mp3")]
            finally:
                unload_model()
                return results

    # VR/MDX
    elif model_type in ["vr", "mdx"]:
        info = models_data[model_type][model_name]

        # custom VR
        if model_type == "vr" and info.get("custom_vr", False):
            conf, ckpt = download_model(
                CONFIG["inference"]["models_cache_dir"],
                model_name,
                model_type,
                info["checkpoint_url"],
                info["config_url"]
            )
            primary_stem = info["primary_stem"]

            if call_method == "cli":
                cmd = [
                    "python", "-m", "separator.uvr_sep", "custom_vr",
                    f'--input_file "{input_file}"',
                    f'--ckpt_path "{ckpt}"',
                    f'--config_path "{conf}"',
                    f'--bitrate "{output_bitrate}"',
                    f'--model_name "{model_name}"',
                    f'--template "{template}"',
                    f'--output_format "{output_format}"',
                    f'--primary_stem "{primary_stem}"',
                    f'--aggression {vr_aggr}',
                    f'--output_dir "{output_dir}"'
                ]
                if selected_stems:
                    instruments = " ".join(f'"{s}"' for s in selected_stems)
                    cmd.append(f'--selected_instruments {instruments}')
                subprocess.run(" ".join(cmd), shell=True, check=True)

                results_path = os.path.join(output_dir, "results.json")
                if os.path.exists(results_path):
                    with open(results_path, encoding="utf-8") as f:
                        return json.load(f)
                    
                else:
                    return [("None", "/none/none.mp3")]

            elif call_method == "direct":
                from separator.uvr_sep import custom_vr_separate
                try:
                    results = custom_vr_separate(
                        input_file=input_file,
                        ckpt_path=ckpt,
                        config_path=conf,
                        bitrate=output_bitrate,
                        model_name=model_name,
                        template=template,
                        output_format=output_format,
                        primary_stem=primary_stem,
                        aggression=vr_aggr,
                        output_dir=output_dir,
                        selected_instruments=selected_stems
                    )
                except Exception as e:
                    print(e)
                    return [("None", "/none/none.mp3")]
                finally:
                    unload_model()
                    return results
        # standard UVR
        else:
            if call_method == "cli":
                cmd = [
                    "python", "-m", "separator.uvr_sep", "uvr",
                    f'--input_file "{input_file}"',
                    f'--output_dir "{output_dir}"',
                    f'--template "{template}"',
                    f'--bitrate "{output_bitrate}"',
                    f'--model_dir "{CONFIG["inference"]["models_cache_dir"]}"',
                    f'--model_type "{model_type}"',
                    f'--model_name "{model_name}"',
                    f'--output_format "{output_format}"',
                    f'--aggression {vr_aggr}'
                ]
                if selected_stems:
                    instruments = " ".join(f'"{s}"' for s in selected_stems)
                    cmd.append(f'--selected_instruments {instruments}')
                subprocess.run(" ".join(cmd), shell=True, check=True)

                results_path = os.path.join(output_dir, "results.json")
                if os.path.exists(results_path):
                    with open(results_path, encoding="utf-8") as f:
                        return json.load(f)
                    
                else:
                    return [("None", "/none/none.mp3")]

            elif call_method == "direct":
                from separator.uvr_sep import non_custom_uvr_inference
                try:
                    results = non_custom_uvr_inference(
                        input_file=input_file,
                        output_dir=output_dir,
                        template=template,
                        bitrate=output_bitrate,
                        model_dir=CONFIG["inference"]["models_cache_dir"],
                        model_type=model_type,
                        model_name=model_name,
                        output_format=output_format,
                        aggression=vr_aggr,
                        selected_instruments=selected_stems
                    )
                except Exception as e:
                    print(e)
                    return [("None", "/none/none.mp3")]
                finally:
                    unload_model()
                    return results

    return [("None", "/none/none.mp3")]

########### GRADIO WRAPPERS
def mvsepless(
    input_audio,
    output_dir,
    model_type,
    model_name,
    ext_inst,
    vr_aggr=5,
    output_format="wav",
    template="NAME_(STEM)_MODEL",
    selected_stems=None,
    progress=gr.Progress(track_tqdm=True)
):
    if selected_stems is None:
        selected_stems = []

    if not input_audio:
        return t("no_audio"), None
    if not model_type or not model_name:
        return t("no_model"), None
    if not output_dir:
        return t("no_output_dir"), None

    model_fullname = models_data[model_type][model_name].get("full_name", model_name)
    output_text = f"{t('algorithm', model_fullname=model_fullname)}\n{t('output_format_info', output_format=output_format)}"

    if isinstance(input_audio, list):   # batch
        batch_results = []
        total = len(input_audio)
        for i, file in enumerate(input_audio):
            progress((i + 1) / total, desc=t("processing", base_name=os.path.basename(file)))
            res = single_multi_inference(file, output_dir, model_type, model_name,
                                         ext_inst, vr_aggr, output_format, template, selected_stems)
            if res:
                batch_results.append((os.path.splitext(os.path.basename(file))[0], res))
        return output_text, batch_results
    else:
        res = single_multi_inference(input_audio, output_dir, model_type, model_name,
                                     ext_inst, vr_aggr, output_format, template, selected_stems)
        return output_text, res

def mvsepless_sep_gradio(a1, a2, b, c, d, e, f, g, h, i_stem, batch, local_check, progress=gr.Progress(track_tqdm=True)):
    output_dir = b or CONFIG["inference"]["output_dir"]
    print(f"Inference settings: output_dir={output_dir}, bitrate={CONFIG['inference']['output_bitrate']}, method={CONFIG['inference']['call_method']}")

    if batch:
        if not a1:
            text, output = mvsepless(a1, output_dir, c, d, e, f, g, h, i_stem)
            return text, None
        elif isinstance(a1, list):
            text, batch_separated = mvsepless(a1, output_dir, c, d, e, f, g, h, i_stem)
            return text, batch_separated
    else:
        input_audio = a2 if local_check else a1
        if not input_audio:
            results = [gr.update(visible=False)] * 20
            return (gr.update(value=t("no_audio")),) + tuple(results)

        text, output_audio = mvsepless(input_audio, output_dir, c, d, e, f, g, h, i_stem)
        results = []
        if output_audio:
            for i, (stem, file) in enumerate(output_audio[:20]):
                results.append(gr.update(visible=True, label=stem, value=file))
        while len(results) < 20:
            results.append(gr.update(visible=False, label=None, value=None))
        return (gr.update(value=text),) + tuple(results)

########### ENSEMBLE HELPERS
def batch_show_names(out):
    names = [name for name, _ in out]
    return gr.update(choices=names, value=None, visible=True)

def batch_show_results(out, namefile):
    batch_names = []
    if namefile:
        for name, stems in out:
            if name == namefile:
                for i, (stem, output_file) in enumerate(stems[:20]):
                    batch_names.append(gr.update(visible=True, label=stem, value=output_file))
                break
    while len(batch_names) < 20:
        batch_names.append(gr.update(visible=False, label=None, value=None))
    return batch_names      # список, а не tuple

def resample_audio(audio_path):
    if not audio_path:
        gr.Warning(t("error_no_audio"))
        return None
    try:
        original_name = os.path.splitext(os.path.basename(audio_path))[0]
        folder_path = os.path.dirname(audio_path)
        audio = AudioSegment.from_file(audio_path)
        audio_resampled = audio.set_frame_rate(44100)
        resampled_path = os.path.join(folder_path, f"resampled_{original_name}.wav")
        audio_resampled.export(resampled_path, format="wav")
        return resampled_path
    except Exception as e:
        print(f"Resampling error: {e}")
        gr.Warning(t("resample_error"))
        return None

########### AUDIO PROCESSING
def load_audio(filepath):
    if not filepath:
        return None, None
    try:
        return librosa.load(filepath, sr=None, mono=False)
    except Exception as e:
        print(f"Audio loading error: {e}")
        return None, None

def process_channel(y1_ch, y2_ch, sr, method):
    if method == "waveform":
        return y1_ch - y2_ch
    elif method == "spectrogram":
        S1 = librosa.stft(y1_ch, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
        S2 = librosa.stft(y2_ch, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
        mag1, mag2 = np.abs(S1), np.abs(S2)
        mag_result = np.maximum(mag1 - mag2, 0)
        phase = np.angle(S1)
        S_result = mag_result * np.exp(1j * phase)
        return librosa.istft(S_result, n_fft=N_FFT, hop_length=HOP_LENGTH,
                             win_length=WIN_LENGTH, length=len(y1_ch))
    return y1_ch

def process_audio(audio1_path, audio2_path, out_format, method):
    y1, sr1 = load_audio(audio1_path)
    y2, sr2 = load_audio(audio2_path)
    if sr1 is None or sr2 is None:
        raise gr.Error(t("error_both_audio"))

    channels1 = 1 if y1.ndim == 1 else y1.shape[0]
    channels2 = 1 if y2.ndim == 1 else y2.shape[0]
    y1 = y1.T if channels1 > 1 else y1.reshape(-1, 1)
    y2 = y2.T if channels2 > 1 else y2.reshape(-1, 1)

    if sr1 != sr2:
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
        sr2 = sr1

    min_len = min(len(y1), len(y2))
    y1, y2 = y1[:min_len], y2[:min_len]

    result_channels = []
    for c in range(channels1):
        y1_ch = y1[:, c]
        y2_ch = y2[:, min(c, channels2 - 1)] if channels2 > 1 else y2[:, 0]
        result_channels.append(process_channel(y1_ch, y2_ch, sr1, method))

    result = np.column_stack(result_channels) if len(result_channels) > 1 else np.array(result_channels[0])
    max_val = np.max(np.abs(result))
    if max_val > 0:
        result = result * 0.9 / max_val

    output_dir_path = CONFIG["inference"]["output_dir"]
    inverted_wav = os.path.join(output_dir_path, "inverted.wav")
    sf.write(inverted_wav, result, sr1)
    inverted = os.path.join(output_dir_path, f"inverted_ensemble.{out_format}")
    write_audio_file(inverted, result.T, sr1, out_format, "320k")
    return inverted, inverted_wav

def manual_ensem(input_audios, method, weights, out_format):
    if not input_audios:
        raise gr.Error(t("error_no_files"))
    try:
        weights = [float(w.strip()) for w in weights.split(",")]
        if len(weights) != len(input_audios):
            weights = [1.0] * len(input_audios)
    except Exception:
        weights = [1.0] * len(input_audios)

    temp_dir = tempfile.mkdtemp(prefix="mvsep_", dir=CONFIG["inference"]["output_dir"])
    audio_data, max_length = [], 0
    for file in input_audios:
        data, sr = librosa.load(file, sr=None, mono=False)
        if data.ndim == 1:
            data = np.stack([data, data])
        elif data.shape[0] != 2:
            data = data.T
        audio_data.append((file, data, sr))
        max_length = max(max_length, data.shape[1])

    padded_files = []
    for file, data, sr in audio_data:
        if data.shape[1] < max_length:
            data = np.pad(data, ((0, 0), (0, max_length - data.shape[1])), mode='constant')
        padded_path = os.path.join(temp_dir, os.path.basename(file))
        sf.write(padded_path, data.T, sr)
        padded_files.append(padded_path)

    output_name = f"ensemble_{method}"
    output, output_wav = ensemble_audio_files(
        padded_files,
        output=os.path.join(temp_dir, output_name),
        ensemble_type=method,
        weights=weights,
        out_format=out_format
    )
    return output, output_wav

########### ENSEMBLESS
def ensembless(input_audio, input_settings, ensemble_type, out_format):
    if not input_audio or not input_settings:
        raise gr.Error(t("error_input_data"))

    progress = gr.Progress()
    progress(0, desc=f"{t('process1')}...")
    base_name = os.path.splitext(os.path.basename(input_audio))[0]
    temp_dir = tempfile.mkdtemp(prefix="mvsep_", dir=ENSEMBLESS_OUTPUT_DIR)
    source_files, output_s_files, output_s_weights, block_count = [], [], [], len(input_settings)

    for i, (input_model, weight, s_stem) in enumerate(input_settings):
        progress(i / block_count, desc=f"{t('process2')} {i+1}/{block_count}")
        model_type, model_name = input_model.split(" / ")
        output_s_dir = os.path.join(temp_dir, f"{model_type}_{model_name}_stems")
        os.makedirs(output_s_dir, exist_ok=True)
        output = single_multi_inference(
            input_audio, output_s_dir, model_type, model_name,
            True, 10, "wav", "STEM_MODEL", []
        )
        if output:
            for stem, file in output:
                source_files.append(file)
                if stem == s_stem:
                    output_s_files.append(file)
                    output_s_weights.append(float(weight))

    progress(0.9, desc=f"{t('process3')}...")
    if not output_s_files:
        raise gr.Error(t("error_no_stems"))

    # align lengths
    audio_data, max_length = [], 0
    for file in output_s_files:
        data, sr = sf.read(file)
        if data.ndim == 1:
            data = np.stack([data, data])
        elif data.shape[0] != 2:
            data = data.T
        audio_data.append((file, data, sr))
        max_length = max(max_length, data.shape[1])

    padded_files = []
    for file, data, sr in audio_data:
        if data.shape[1] < max_length:
            data = np.pad(data, ((0, 0), (0, max_length - data.shape[1])), mode='constant')
        padded_path = os.path.join(temp_dir, os.path.basename(file))
        sf.write(padded_path, data.T, sr)
        padded_files.append(padded_path)

    progress(0.95, desc=f"{t('process4')}...")
    output_name = f"ensemble_{base_name}_{ensemble_type}"
    output, output_wav = ensemble_audio_files(
        padded_files,
        output=os.path.join(temp_dir, output_name),
        ensemble_type=ensemble_type,
        weights=output_s_weights,
        out_format=out_format
    )
    return output, output_wav, source_files

########### MODEL DROPDOWN HELPERS
def get_model_types():
    return list(models_data.keys())

def get_models_by_type(model_type):
    return list(models_data[model_type].keys()) if model_type in models_data else []

def get_stems_by_model(model_type, model_name):
    if model_type in models_data and model_name in models_data[model_type]:
        return models_data[model_type][model_name].get('stems', [])
    return []

def update_model_dropdown(model_type):
    models = get_models_by_type(model_type)
    return gr.Dropdown(choices=models, value=models[0] if models else None)

def update_stem_dropdown(model_type, model_name):
    stems = get_stems_by_model(model_type, model_name)
    return gr.Dropdown(choices=stems, value=stems[0] if stems else None)

########### ENSEMBLE MANAGER
class EnsembleManager:
    def __init__(self):
        self.models = []

    def add_model(self, model_type, model_name, stem, weight):
        self.models.append({
            'type': model_type,
            'name': model_name,
            'stem': stem,
            'weight': float(weight)
        })
        return self.get_df()

    def remove_model(self, index):
        if 0 <= index < len(self.models):
            del self.models[index]
        return self.get_df()

    def clear_models(self):
        self.models = []
        return self.get_df()

    def get_df(self):
        if not self.models:
            return pd.DataFrame(columns=["#", t("model_type"), t("model_name"), t("stem"), t("weight")])
        data = [[i+1, m['type'], m['name'], m['stem'], m['weight']] for i, m in enumerate(self.models)]
        return pd.DataFrame(data, columns=["#", t("model_type"), t("model_name"), t("stem"), t("weight")])

    def get_settings(self):
        return [(f"{m['type']} / {m['name']}", m['weight'], m['stem']) for m in self.models]

manager = EnsembleManager()

def mvsepless_theme():
    font = CONFIG["settings"]["font"]
    return gr.themes.Default(
        primary_hue="violet",
        secondary_hue="cyan",
        neutral_hue="blue",
        spacing_size="sm",
        font=[gr.themes.GoogleFont(font), 'ui-sans-serif', 'system-ui', 'sans-serif'],
    ).set(
        body_text_color='*neutral_950',
        body_text_color_subdued='*neutral_500',
        background_fill_primary='*neutral_200',
        background_fill_primary_dark='*neutral_800',
        border_color_accent='*primary_950',
        border_color_accent_dark='*neutral_700',
        border_color_accent_subdued='*primary_500',
        border_color_primary='*primary_800',
        border_color_primary_dark='*neutral_400',
        color_accent_soft='*primary_100',
        color_accent_soft_dark='*neutral_800',
        link_text_color='*secondary_700',
        link_text_color_active='*secondary_700',
        link_text_color_hover='*secondary_800',
        link_text_color_visited='*secondary_600',
        link_text_color_visited_dark='*secondary_700',
        block_background_fill='*background_fill_secondary',
        block_background_fill_dark='*neutral_950',
        block_label_background_fill='*secondary_400',
        block_label_text_color='*neutral_800',
        panel_background_fill='*background_fill_primary',
        checkbox_background_color='*background_fill_secondary',
        checkbox_label_background_fill_dark='*neutral_900',
        input_background_fill_dark='*neutral_900',
        input_background_fill_focus='*neutral_100',
        input_background_fill_focus_dark='*neutral_950',
        button_small_radius='*radius_sm',
        button_secondary_background_fill='*neutral_400',
        button_secondary_background_fill_dark='*neutral_500',
        button_secondary_background_fill_hover_dark='*neutral_950'
    )

def create_mvsepless_app():
    gr.HTML(f"<h1><center>{t('app_title')}</center></h1>")

    with gr.Tabs():
        with gr.Tab(t("separation")):
            with gr.Column():
                with gr.Group(visible=True) as upload_group:
                    input_audio = gr.Audio(show_label=False, type="filepath", interactive=True, elem_classes="fixed-height", sources="upload")
                    input_audios = gr.Files(show_label=False, type="filepath", visible=False, interactive=True, file_types=[".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff"], elem_classes="fixed-height")
                input_file_explorer = gr.FileExplorer(show_label=False, root_dir="/content", file_count="single", visible=False, elem_classes="fixed-height")
                local_check = gr.Checkbox(label=t("local_path"), value=False, interactive=True)
                batch_separation = gr.Checkbox(label=t("batch_processing"), value=False, interactive=True, info=t("batch_info"))

            with gr.Tab(t("inference")):
                batch_results_state = gr.State()
                with gr.Row(equal_height=False):
                    with gr.Column(variant="panel"):
                        with gr.Group():
                            with gr.Row():
                                model_type = gr.Dropdown(label=t("model_type"), choices=get_model_types(), value=get_model_types()[0], interactive=True, filterable=False)
                                model_name = gr.Dropdown(label=t("model_name"), choices=get_models_by_type(get_model_types()[0]), value=get_models_by_type(get_model_types()[0])[0], interactive=True, filterable=False)
                            ext_inst = gr.Checkbox(label=t("extract_instrumental"), visible=True, value=True, interactive=True, info=t("extract_info"))
                            vr_aggr_slider = gr.Slider(label=t("vr_aggressiveness"), minimum=0, maximum=100, step=1, visible=False, interactive=True, value=5)
                            stems = gr.CheckboxGroup(label=t("stems_list"), choices=get_stems_by_model(get_model_types()[0], get_models_by_type(get_model_types()[0])[0]), value=[], interactive=False, info=t("stems_info", target_instrument="vocals"))
                            with gr.Column():
                                with gr.Accordion(t("template"), open=False):
                                    template_info = gr.Markdown(f"""{t("template_info")}""", line_breaks=True)
                                    template = gr.Text(label=t("template"), value="NAME_(STEM)_MODEL", interactive=True)
                                output_format = gr.Dropdown(label=t("output_format"), choices=OUTPUT_FORMATS, value="mp3", interactive=True, filterable=False)
                            single_separate_btn = gr.Button(t("separate_btn"), variant="primary", interactive=True, size="lg")
                            batch_separate_btn = gr.Button(t("separate_btn"), variant="primary", visible=False, interactive=True, size="lg")
                    with gr.Column(variant="panel"):
                        with gr.Group():
                            output_info = gr.Textbox(label=t("separation_info"), lines=3)
                            batch_select_dir = gr.Dropdown(label=t("select_file"), visible=False, interactive=True, filterable=False)
                            output_stems = []
                            for _ in range(10):
                                with gr.Row():
                                    audio1 = gr.Audio(visible=False, interactive=False, type="filepath", show_download_button=True)
                                    audio2 = gr.Audio(visible=False, interactive=False, type="filepath", show_download_button=True)
                                    output_stems.extend([audio1, audio2])

            with gr.Tab(t("auto_ensemble")):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(f"### {t('model_selection')}")
                        e_model_type = gr.Dropdown(choices=get_model_types(), label=t("model_type"), value=get_model_types()[0], filterable=False)
                        e_model_name = gr.Dropdown(choices=get_models_by_type(get_model_types()[0]), label=t("model_name"), interactive=True, filterable=False)
                        e_stem = gr.Dropdown(choices=get_stems_by_model(get_model_types()[0], get_models_by_type(get_model_types()[0])[0]), label=t("stem_selection"), interactive=True, filterable=False)
                        e_weight = gr.Slider(label=t("weight"), value=1.0, minimum=0.1, maximum=10.0, step=0.1)
                        e_add_btn = gr.Button(t("add_button"), variant="primary")
                    with gr.Column(scale=2):
                        gr.Markdown(f"### {t('current_ensemble')}")
                        ensemble_df = gr.Dataframe(value=manager.get_df(), headers=["#", t("model_type"), t("model_name"), t("stem"), t("weight")], datatype=["str", "str", "str", "str", "number"], interactive=False)
                        with gr.Row(equal_height=True):
                            remove_idx = gr.Number(label=t("remove_index"), precision=0, minimum=1, interactive=True)
                            with gr.Column():
                                remove_btn = gr.Button(t("remove_button"), variant="stop")
                                clear_btn = gr.Button(t("clear_button"), variant="stop")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(f"### {t('input_audio')}")
                        resample_local_btn = gr.Button(t("resample"))
                        e_input_audio_resampled = gr.Textbox(label=t("resampled_path"), interactive=False, lines=3, max_lines=3)
                        gr.Markdown(f"### {t('settings')}")
                        ensemble_type = gr.Dropdown(choices=['avg_wave', 'median_wave', 'min_wave', 'max_wave', 'avg_fft', 'median_fft', 'min_fft', 'max_fft'], value='avg_fft', label=t("method"), filterable=False)
                        e_output_format = gr.Dropdown(choices=OUTPUT_FORMATS, value="mp3", label=t("output_format"), filterable=False)
                        e_run_btn = gr.Button(t("run_button"), variant="primary")
                    with gr.Column(scale=2):
                        with gr.Tab(t('results')):
                            e_output_audio = gr.Audio(label=t("results"), type="filepath", interactive=False, show_download_button=True)
                            e_output_wav = gr.Text(label="Результат в WAV", interactive=False, visible=False)
                            gr.Markdown(f"###### {t('inverted_result')}")
                            invert_method = gr.Radio(choices=["waveform", "spectrogram"], label=t("invert_method"), value="waveform")
                            invert_btn = gr.Button(t("invert_button"))
                            inverted_output_audio = gr.Audio(label=t("inverted_result"), type="filepath", interactive=False, show_download_button=True)
                            inverted_wav = gr.Text(label="Инвертированный результат в WAV", interactive=False, visible=False)
                        with gr.Tab(t('result_source')):
                            result_source = gr.Files(interactive=False, label=t('result_source'))

        with gr.Tab(t("plugins")):
            plugins = []
            with gr.Tab(t('upload')):
                upload_plugin_files = gr.Files(label=t('upload'), file_types=[".py"])
                upload_btn = gr.Button(t('upload_btn'))
            if os.path.exists(plugins_dir) and os.path.isdir(plugins_dir):
                try:
                    for filename in os.listdir(plugins_dir):
                        if filename.endswith(".py") and filename != "__init__.py":
                            file_path = os.path.join(plugins_dir, filename)
                            module_name = filename[:-3]
                            spec = importlib.util.spec_from_file_location(module_name, file_path)
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            plugin_func, name_func = None, None
                            for attr_name in dir(module):
                                attr = getattr(module, attr_name)
                                if callable(attr):
                                    if attr_name.endswith("plugin"):
                                        plugin_func = attr
                                    elif attr_name.endswith("plugin_name"):
                                        name_func = attr
                            if plugin_func is not None:
                                plugin_name = name_func() if name_func else module_name
                                plugins.append((plugin_name, plugin_func))
                except Exception as e:
                    print(e)
            for name, func in plugins:
                try:
                    print(t("loading_plugin", name=name))
                    with gr.Tab(name):
                        func(CONFIG["settings"]["language"])
                except Exception as e:
                    print(t("error_loading_plugin", e=e))

        with gr.Tab("Extra"):
            with gr.Tab(t("manual_ensemble")):
                with gr.Row(equal_height=True):
                    input_files = gr.Files(show_label=False, type="filepath", file_types=[".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff"])
                    with gr.Column():
                        info_audios = gr.Textbox(label="", interactive=False)
                        man_method = gr.Dropdown(choices=['avg_wave', 'median_wave', 'min_wave', 'max_wave', 'avg_fft', 'median_fft', 'min_fft', 'max_fft'], value='avg_fft', label=t("method"), filterable=False)
                        weights_input = gr.Textbox(label=t("weights_input"), value="1.0,1.0")
                        output_man_format = gr.Dropdown(choices=OUTPUT_FORMATS, value="mp3", label=t("output_format"), filterable=False)
                run_man_btn = gr.Button(t("run_button"), variant="primary")
                output_man_audio = gr.Audio(label=t("results"), type="filepath", interactive=False, show_download_button=True)
                output_man_wav = gr.Text(label="Результат в WAV", interactive=False, visible=False)

            with gr.Tab(t("inverter")):
                with gr.Row():
                    audio1 = gr.Audio(label=t("main_audio"), type="filepath", elem_classes="fixed-height", sources="upload")
                    audio2 = gr.Audio(label=t("audio_to_remove"), type="filepath", elem_classes="fixed-height", sources="upload")
                invert_man_method = gr.Radio(choices=["waveform", "spectrogram"], label=t("processing_method"), value="waveform")
                output_man_i_format = gr.Dropdown(choices=OUTPUT_FORMATS, value="mp3", label=t("output_format"), filterable=False)
                invert_man_btn = gr.Button(t("invert_button"))
                with gr.Column():
                    invert_man_output = gr.Audio(label=t("results"), interactive=False, show_download_button=True, elem_classes="fixed-height2")
                    invert_man_output_wav = gr.Text(interactive=False, visible=False)

            with gr.Tab(t("model_loading")):
                dw_m_model_type = gr.Dropdown(label=t("model_type"), choices=get_model_types(), value=get_model_types()[0], interactive=True, filterable=False)
                dw_m_model_name = gr.Dropdown(label=t("model_name"), choices=get_models_by_type(get_model_types()[0]), value=get_models_by_type(get_model_types()[0])[0], interactive=True, filterable=False)
                dw_m_btn = gr.Button(t("download_model_btn"))

        with gr.Tab(t("settings_tab")):
            config_preview = gr.Code(label=t("settings_config"), value=json.dumps(CONFIG, indent=4, ensure_ascii=False))
            restart_btn = gr.Button(t("restart_btn"), variant="stop")
            with gr.Tab(t("ui_settings")):
                with gr.Column(variant="panel"):
                    local_font = gr.Dropdown(label=t("settings_select_local_font"), choices=get_font_files())
                    upload_local_font = gr.File(label=t("settings_upload_fonts"), interactive=True, file_count="multiple", file_types=[".ttf", ".otf", ".woff", ".eot"], elem_classes="fixed-height")
                    get_list_fonts = gr.Button(t("settings_get_list_fonts"))
                with gr.Column(variant="panel"):
                    google_font_info = gr.Markdown(t("settings_info_font"))
                    google_font = gr.Text(label=t("settings_google_font"))
                set_font_btn = gr.Button(t("settings_set_font"), variant="primary")
                language = gr.Radio(label=t("settings_language"), choices=LANGS, value=CONFIG["settings"]["language"])
                server_share = gr.Checkbox(label=t("settings_share"), value=CONFIG["settings"]["share"])
                server_debug = gr.Checkbox(label=t("settings_debug"), value=CONFIG["settings"]["debug"])
            with gr.Tab(t("infer_settings")):
                with gr.Column(variant="panel"):
                    current_output_dir = gr.Text(label=t("last_output_directory"), value=CONFIG["inference"]["output_dir"], interactive=False)
                    output_dir = gr.Text(label=t("infer_output_dir"), value=CONFIG["inference"]["output_dir"])
                    models_cache_dir = gr.Text(label=t("infer_models_cache_dir"), value=CONFIG["inference"]["models_cache_dir"])
                    output_bitrate = gr.Slider(label=t("infer_bitrate"), minimum=32, maximum=320, value=int(CONFIG["inference"]["output_bitrate"].replace("k", "")), step=1)
                with gr.Column(variant="panel"):
                    with gr.Row():
                        call_method_info = gr.Markdown(t("infer_call_method_info"), line_breaks=True)
                    call_method = gr.Radio(label=t("infer_call_method"), choices=CALL_METHODS, value=CONFIG["inference"]["call_method"])
    ########### EVENT HANDLERS
    def write_OUTPUT_DIR_config(val):
        config_manager.update("inference", "output_dir", val)
        return gr.update(value=val)

    def write_MODELS_CACHE_DIR_config(val):
        config_manager.update("inference", "models_cache_dir", val)
        return gr.update(value=val)

    def write_OUTPUT_BITRATE_config(val):
        config_manager.update("inference", "output_bitrate", f"{int(val)}k")
        return gr.update(value=f"{int(val)}k")

    def write_CALL_METHOD_config(val):
        config_manager.update("inference", "call_method", val)
        return gr.update(value=val)

    def write_LANG_config(val):
        config_manager.update("settings", "language", val)
        return gr.update(value=val)

    def write_SHARE_config(val):
        config_manager.update("settings", "share", val)
        return gr.update(value=val)

    def write_DEBUG_config(val):
        config_manager.update("settings", "debug", val)
        return gr.update(value=val)

    def write_FONT_config(val):
        config_manager.update("settings", "font", val)
        return gr.update(value=val)

    def add_model(model_type, model_name, stem, weight):
        global manager
        return manager.add_model(model_type, model_name, stem, weight)

    def remove_model(index):
        global manager
        return manager.remove_model(int(index) - 1)

    def clear_all_models():
        global manager
        return manager.clear_models()

    def analyze_sample_rate(files):
        sample_rates = []
        for file in files:
            try:
                info = mediainfo(file)
                sr = info.get("sample_rate", "Unknown")
                sample_rates.append(f"{os.path.basename(file)}: {sr} Hz")
            except Exception:
                sample_rates.append(f"{os.path.basename(file)}: Error reading sample rate")
        return "\n".join(sample_rates)

    def run_ensemble(input_audio, ensemble_type, out_format):
        settings = manager.get_settings()
        return ensembless(input_audio, settings, ensemble_type, out_format)

    # bindings
    output_dir.change(fn=write_OUTPUT_DIR_config, inputs=output_dir).then(lambda: gr.update(value=json.dumps(CONFIG, indent=4, ensure_ascii=False)), None, config_preview)
    models_cache_dir.change(fn=write_MODELS_CACHE_DIR_config, inputs=models_cache_dir).then(lambda: gr.update(value=json.dumps(CONFIG, indent=4, ensure_ascii=False)), None, config_preview)
    output_bitrate.change(fn=write_OUTPUT_BITRATE_config, inputs=output_bitrate).then(lambda: gr.update(value=json.dumps(CONFIG, indent=4, ensure_ascii=False)), None, config_preview)
    call_method.change(fn=write_CALL_METHOD_config, inputs=call_method).then(lambda: gr.update(value=json.dumps(CONFIG, indent=4, ensure_ascii=False)), None, config_preview)

    language.change(fn=write_LANG_config, inputs=language).then(lambda: gr.update(value=json.dumps(CONFIG, indent=4, ensure_ascii=False)), None, config_preview)
    server_share.change(fn=write_SHARE_config, inputs=server_share).then(lambda: gr.update(value=json.dumps(CONFIG, indent=4, ensure_ascii=False)), None, config_preview)
    server_debug.change(fn=write_DEBUG_config, inputs=server_debug).then(lambda: gr.update(value=json.dumps(CONFIG, indent=4, ensure_ascii=False)), None, config_preview)
    set_font_btn.click(fn=lambda x, y: write_FONT_config(y if y else x), inputs=[local_font, google_font], outputs=None).then(lambda: gr.update(value=json.dumps(CONFIG, indent=4, ensure_ascii=False)), None, config_preview)
    get_list_fonts.click(fn=lambda: gr.update(choices=get_font_files()), inputs=None, outputs=local_font)
    upload_local_font.upload(fn=write_fonts, inputs=upload_local_font, outputs=None)

    upload_btn.click(fn=lambda files: upload_plugin_list(files), inputs=upload_plugin_files)
    restart_btn.click(fn=lambda: (time.sleep(1), os.execl(sys.executable, sys.executable, *sys.argv))[1], inputs=None)

    local_check.change(
        fn=lambda x: (gr.update(visible=not x), gr.update(value=None), gr.update(value=None), gr.update(visible=x)),
        inputs=local_check,
        outputs=[upload_group, input_audio, input_audios, input_file_explorer]
    )

    batch_separation.change(
        fn=lambda x: (
            gr.update(visible=x, value=None),
            gr.update(visible=x),
            gr.update(visible=not x, value=None),
            gr.update(visible=not x)
        ),
        inputs=batch_separation,
        outputs=[input_audios, batch_separate_btn, input_audio, single_separate_btn]
    ).then(
        fn=lambda x: gr.update(file_count="multiple" if x else "single", value=None),
        inputs=batch_separation,
        outputs=input_file_explorer
    )

    model_type.change(
        fn=lambda x: gr.update(visible=(x == "vr")),
        inputs=model_type,
        outputs=vr_aggr_slider
    ).then(
        fn=lambda x: gr.update(choices=list(models_data[x].keys()), value=list(models_data[x].keys())[0]),
        inputs=model_type,
        outputs=model_name
    ).then(
        fn=lambda x: gr.update(visible=(x not in ["vr", "mdx"])),
        inputs=model_type,
        outputs=ext_inst
    )

    dw_m_model_type.change(
        fn=lambda x: gr.update(
            choices=[m for m in list(models_data[x].keys()) if (models_data[x][m].get("checkpoint_url") if x not in ["vr", "mdx"] else None) or (models_data[x][m].get("custom_vr") if x == "vr" else None)],
            value=None
        ),
        inputs=dw_m_model_type,
        outputs=dw_m_model_name
    )

    model_name.change(
        fn=lambda x, y: gr.update(choices=list(models_data[x][y]["stems"]), value=[], interactive=(models_data[x][y]["target_instrument"] is None)),
        inputs=[model_type, model_name],
        outputs=stems
    ).then(
        fn=lambda x, y: gr.update(value=(models_data[x][y]["target_instrument"] is not None)),
        inputs=[model_type, model_name],
        outputs=ext_inst
    )

    single_separate_btn.click(
        fn=lambda x: gr.update(choices=None, visible=False, value=None),
        inputs=None,
        outputs=batch_select_dir
    ).then(
        fn=lambda x: os.path.join(CONFIG["inference"]["output_dir"], f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{x}'),
        inputs=model_name,
        outputs=current_output_dir
    ).then(
        fn=mvsepless_sep_gradio,
        inputs=[input_audio, input_file_explorer, current_output_dir, model_type, model_name, ext_inst, vr_aggr_slider, output_format, template, stems, batch_separation, local_check],
        outputs=[output_info, *output_stems],
        show_progress_on=output_info
    )

    batch_separate_btn.click(
        fn=lambda: gr.update(choices=None, visible=False, value=None),
        inputs=None,
        outputs=batch_select_dir
    ).then(
        fn=lambda x: os.path.join(CONFIG["inference"]["output_dir"], f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{x}'),
        inputs=model_name,
        outputs=current_output_dir
    ).then(
        fn=mvsepless_sep_gradio,
        inputs=[input_audios, input_file_explorer, current_output_dir, model_type, model_name, ext_inst, vr_aggr_slider, output_format, template, stems, batch_separation, local_check],
        outputs=[output_info, batch_results_state],
        show_progress_on=output_info
    ).then(
        fn=batch_show_names,
        inputs=batch_results_state,
        outputs=batch_select_dir
    )

    batch_select_dir.change(
        fn=batch_show_results,
        inputs=[batch_results_state, batch_select_dir],
        outputs=output_stems
    )

    e_model_type.change(update_model_dropdown, inputs=e_model_type, outputs=e_model_name)
    e_model_name.change(update_stem_dropdown, inputs=[e_model_type, e_model_name], outputs=e_stem)
    invert_btn.click(
        fn=process_audio,
        inputs=[e_input_audio_resampled, e_output_wav, e_output_format, invert_method],
        outputs=[inverted_output_audio, inverted_wav]
    )
    input_audio.upload(resample_audio, inputs=input_audio, outputs=e_input_audio_resampled)
    resample_local_btn.click(resample_audio, inputs=input_file_explorer, outputs=e_input_audio_resampled)
    e_add_btn.click(add_model, inputs=[e_model_type, e_model_name, e_stem, e_weight], outputs=ensemble_df)
    remove_btn.click(remove_model, inputs=remove_idx, outputs=ensemble_df)
    clear_btn.click(clear_all_models, outputs=ensemble_df)
    e_run_btn.click(
        fn=lambda: (gr.update(value=None), gr.update(value=None)),
        inputs=None,
        outputs=[inverted_output_audio, inverted_wav]
    ).then(
        run_ensemble,
        inputs=[e_input_audio_resampled, ensemble_type, e_output_format],
        outputs=[e_output_audio, e_output_wav, result_source]
    )
    invert_man_btn.click(
        process_audio,
        inputs=[audio1, audio2, output_man_i_format, invert_man_method],
        outputs=[invert_man_output, invert_man_output_wav]
    )
    input_files.upload(fn=analyze_sample_rate, inputs=input_files, outputs=info_audios)
    run_man_btn.click(manual_ensem, inputs=[input_files, man_method, weights_input, output_man_format], outputs=[output_man_audio, output_man_wav])

########### MAIN
def parse_args():
    parser = argparse.ArgumentParser(description="Базовый интерфейс для разделения музыки и вокала")
    parser.add_argument("--ngrok_token", type=str, help="Аутентификация (формат: username:password)")
    parser.add_argument("--ssl-keyfile", type=str, help="Путь к SSL ключу")
    parser.add_argument("--ssl-certfile", type=str, help="Путь к SSL сертификату")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    GRADIO_SSL_KEYFILE = args.ssl_keyfile
    GRADIO_SSL_CERTFILE = args.ssl_certfile

    css = """
    .fixed-height { height: 160px !important; min-height: 160px !important; }
    .fixed-height2 { height: 250px !important; min-height: 250px !important; }
    """

    font = CONFIG["settings"]["font"]
    if font and os.path.isfile(font) and font.lower().endswith((".ttf", ".otf", ".woff", ".eot")):
        with open(font, "rb") as font_file:
            base64_font = base64.b64encode(font_file.read()).decode("utf-8")
        css += f"""
        @font-face {{
            font-family: 'CustomFont';
            src: url(data:font/truetype;charset=utf-8;base64,{base64_font}) format('truetype');
        }}
        body, .gradio-container {{ font-family: 'CustomFont', sans-serif !important; }}
        """

    with gr.Blocks(title=t("app_title"), theme=mvsepless_theme(), css=css) as MVSEPLESS_UI:
        create_mvsepless_app()

    if args.ngrok_token:
        ngrok.set_auth_token(args.ngrok_token)
        ngrok.kill()
        tunnel = ngrok.connect(CONFIG["settings"]["port"])
        print(f"Публичная ссылка: {tunnel.public_url}")

    MVSEPLESS_UI.launch(
        server_name=GRADIO_HOST,
        server_port=CONFIG["settings"]["port"],
        share=CONFIG["settings"]["share"],
        debug=CONFIG["settings"]["debug"],
        auth=CONFIG["settings"]["auth"],
        ssl_keyfile=GRADIO_SSL_KEYFILE,
        ssl_certfile=GRADIO_SSL_CERTFILE,
        max_file_size=CONFIG["settings"]["max_file_size"],
        allowed_paths=[SCRIPT_DIR, CONFIG["inference"]["output_dir"], CONFIG["inference"]["models_cache_dir"]],
        favicon_path=FAVICON_PATH
    )
