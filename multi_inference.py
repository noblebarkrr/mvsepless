import os
import time
import shutil
import sys
import argparse
from pyngrok import ngrok
import gradio as gr
from datetime import datetime
import importlib.util
import base64


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

import json
from model_list import  models_data
from utils.preedit_config import conf_editor
from utils.download_models import download_model
from assets.translations import MVSEPLESS_TRANSLATIONS as TRANSLATIONS

import pandas as pd
import tempfile
from separator.ensemble import ensemble_audio_files
from pydub.utils import mediainfo
from pydub import AudioSegment
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from separator.audio_writer import write_audio_file
from pydub.exceptions import CouldntDecodeError

########### Константы

LANGS = ["ru", "en"]
FONTS_DIR = os.sep.join([SCRIPT_DIR, "assets", "fonts"])
CONFIG_UI_PATH = os.path.join(SCRIPT_DIR, "config.json")
FAVICON_PATH = os.path.join(SCRIPT_DIR, os.path.join("assets", "mvsepless.png"))
MODELS_CACHE_DIR = os.path.join(SCRIPT_DIR, os.path.join("separator", "models_cache"))
OUTPUT_BITRATE = "320k"
OUTPUT_FORMATS = ["mp3", "wav", "flac", "ogg", "opus", "m4a", "aac", "aiff"]
OUTPUT_DIR = "/content/output"
ENSEMBLESS_OUTPUT_DIR = "/content/ensembless_output"
GRADIO_HOST = "0.0.0.0"
GRADIO_SSL_KEYFILE = None
GRADIO_SSL_CERTFILE = None
MVSEPLESS_UI = None
N_FFT = 2048
WIN_LENGTH = 2048
HOP_LENGTH = WIN_LENGTH // 4
plugins_dir = os.path.join(SCRIPT_DIR, "plugins")
os.makedirs(plugins_dir, exist_ok=True)
os.makedirs(FONTS_DIR, exist_ok=True)
CALL_METHODS = ["cli", "direct"]
CALL_METHOD = "cli"
GOOGLE_FONT = "Tektur"
GRADIO_PORT = 7860
GRADIO_SHARE = True
GRADIO_DEBUG = True
GRADIO_AUTH = None
CURRENT_LANG = "ru"
GRADIO_MAX_FILE_SIZE = "10000MB"

CONFIG = {
    "inference": {
        "output_dir": OUTPUT_DIR,
        "models_cache_dir": MODELS_CACHE_DIR,
        "output_bitrate": OUTPUT_BITRATE,
        "call_method": CALL_METHOD
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

########### Сохранить в конфиг

def write_UI_settings():
    with open(file=CONFIG_UI_PATH, mode="w") as f:
        json.dump(CONFIG, f, indent=2)

def read_UI_settings():
    global GOOGLE_FONT, GRADIO_PORT, GRADIO_SHARE, GRADIO_DEBUG, GRADIO_AUTH, CURRENT_LANG, GRADIO_MAX_FILE_SIZE, OUTPUT_DIR, MODELS_CACHE_DIR, OUTPUT_BITRATE, CALL_METHOD, CONFIG
    with open(file=CONFIG_UI_PATH, mode="r") as f:
        config = json.load(f)

    CALL_METHOD = config["inference"]["call_method"]
    OUTPUT_BITRATE = config["inference"]["output_bitrate"]
    OUTPUT_DIR = config["inference"]["output_dir"]
    MODELS_CACHE_DIR = config["inference"]["models_cache_dir"]
    GOOGLE_FONT = config["settings"]["font"]
    GRADIO_PORT = config["settings"]["port"]
    GRADIO_SHARE = config["settings"]["share"]
    GRADIO_DEBUG = config["settings"]["debug"]
    GRADIO_AUTH = config["settings"]["auth"]
    CURRENT_LANG = config["settings"]["language"]
    GRADIO_MAX_FILE_SIZE = config["settings"]["max_file_size"]
    CONFIG = {
        "inference": {
            "output_dir": OUTPUT_DIR,
            "models_cache_dir": MODELS_CACHE_DIR,
            "output_bitrate": OUTPUT_BITRATE,
            "call_method": CALL_METHOD
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


def write_FONT_config(font):
    global CONFIG
    CONFIG["settings"]["font"] = font
    write_UI_settings()

def write_SHARE_config(share):
    global CONFIG
    CONFIG["settings"]["share"] = share
    write_UI_settings()

def write_DEBUG_config(debug):
    global CONFIG
    CONFIG["settings"]["debug"] = debug
    write_UI_settings()

def write_LANG_config(lang):
    global CONFIG
    CONFIG["settings"]["language"] = lang
    write_UI_settings()

def write_OUTPUT_DIR_config(output_dir):
    global CONFIG, OUTPUT_DIR
    CONFIG["inference"]["output_dir"] = output_dir
    OUTPUT_DIR = output_dir
    write_UI_settings()

def write_MODELS_CACHE_DIR_config(cache_dir):
    global CONFIG, MODELS_CACHE_DIR
    CONFIG["inference"]["models_cache_dir"] = cache_dir
    MODELS_CACHE_DIR = cache_dir
    write_UI_settings()

def write_CALL_METHOD_config(method):
    global CONFIG, CALL_METHOD
    CONFIG["inference"]["call_method"] = method
    CALL_METHOD = method
    write_UI_settings()


def write_OUTPUT_BITRATE_config(output_bitrate):
    bitrate = f"{int(output_bitrate)}k"
    global CONFIG, OUTPUT_BITRATE
    CONFIG["inference"]["output_bitrate"] = bitrate
    OUTPUT_BITRATE = bitrate
    write_UI_settings()

def get_font_files():
    fonts = []
    for filename in os.listdir(FONTS_DIR):
        path = os.path.join(FONTS_DIR, filename)  # Нужно объединить с базовой директорией
        if os.path.isfile(path):
            if filename.lower().endswith((".ttf", ".otf", ".woff", ".eot")):
                fonts.append(os.path.abspath(path))  # Исправлено asbpath на abspath
    return fonts  # Не забываем вернуть результат

def write_fonts(files):
    for file in files:
        shutil.copy(file, os.path.join(FONTS_DIR, os.path.basename(file)))
    gr.Warning("Uploading fonts...")

########### CSS

css = None

########### Код для перевода на нужный язык

def set_language(lang):
    global CURRENT_LANG
    CURRENT_LANG = lang

def t(key, **kwargs):
    """Функция для получения перевода с подстановкой значений"""
    translation = TRANSLATIONS[CURRENT_LANG].get(key, key)
    return translation.format(**kwargs) if kwargs else translation

########### Запуск интерфейса

def load_ui(mvsepless_ui):
    mvsepless_ui.launch(        
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        share=GRADIO_SHARE,
        debug=GRADIO_DEBUG,
        auth=GRADIO_AUTH,
        ssl_keyfile=GRADIO_SSL_KEYFILE,
        ssl_certfile=GRADIO_SSL_CERTFILE,
        max_file_size=GRADIO_MAX_FILE_SIZE,
        allowed_paths=["/content", OUTPUT_DIR, MODELS_CACHE_DIR],
        favicon_path=FAVICON_PATH
    )

########### Перезапуск интерфейса

def restart_ui():
    python = sys.executable
    os.execl(python, python, *sys.argv)

########### Загрузка плагинов из списка

def upload_plugin_list(files):
    if not files:
        return 
    if files is not None:
        for file in files:
            shutil.copy(file, os.path.join(plugins_dir, os.path.basename(file)))    

        gr.Warning(t("restart_warning"))
        time.sleep(5)
        restart_ui()

########### Генерация темы для интерфейса

def mvsepless_theme(font="Tektur"):
    theme = gr.themes.Default(
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
    
    return theme

########### Загрузка модели, минуя разделение

def downloader_models(model_type, model_name):
    if model_type in ["mel_band_roformer", "bs_roformer", "mdx23c", "scnet", "htdemucs", "bandit", "bandit_v2"]:
        config_url = models_data[model_type][model_name]["config_url"]
        checkpoint_url = models_data[model_type][model_name]["checkpoint_url"]
        conf, ckpt = download_model(MODELS_CACHE_DIR, model_name, model_type, checkpoint_url, config_url)
    elif model_type == "vr" and models_data[model_type][model_name]["custom_vr"]:
        config_url = models_data[model_type][model_name]["config_url"]
        checkpoint_url = models_data[model_type][model_name]["checkpoint_url"]               
        primary_stem = models_data[model_type][model_name]["primary_stem"]
        conf, ckpt = download_model(MODELS_CACHE_DIR, model_name, model_type, checkpoint_url, config_url)

########### Получение данных из словаря моделей (Авто ансамбль)

def get_model_types():
    return list(models_data.keys())

def get_models_by_type(model_type):
    return list(models_data[model_type].keys()) if model_type in models_data else []

def get_stems_by_model(model_type, model_name):
    if model_type in models_data and model_name in models_data[model_type]:
        return models_data[model_type][model_name]['stems']
    return []

def update_model_dropdown(model_type):
    models = get_models_by_type(model_type)
    return gr.Dropdown(choices=models, value=models[0] if models else None)

def update_stem_dropdown(model_type, model_name):
    stems = get_stems_by_model(model_type, model_name)
    return gr.Dropdown(choices=stems, value=stems[0] if stems else None)

########### Отображение списка входных файлов, с которых были извлечены стемы (Пакетная обработка)

def batch_show_names(out):
    names = []
    for name, (stems) in out:
        names.append(name)
    return gr.update(choices=names, value=None, visible=True)

########### Отображение стемов в пакетной обработке

def batch_show_results(out, namefile):
        batch_names = []
        if namefile is not None:
            for name, (stems) in out:
                if name == namefile:
                    for i, (stem, output_file) in enumerate(stems[:20]):
                        batch_names.append(gr.update(
                            visible=True,
                            label=stem,
                            value=output_file
                        ))

                    while len(batch_names) < 20:
                        batch_names.append(gr.update(visible=False, label=None, value=None))
                    return tuple(batch_names)                         
        else:
            for i in range(20):
                batch_names.append(gr.update(visible=(i == 0), label=None, value=None))
            return tuple(batch_names)


########### Смена частоты дискретизации аудио (Авто-ансамбль)

def resample_audio(audio, file_path):
    if not audio and not file_path:
        gr.Warning(t("error_no_audio"))
        return None
    if isinstance(file_path, list) and not audio:
        audio = file_path[0]
    if isinstance(audio, str) and not file_path:
        audio = audio
    if isinstance(file_path, str) and not audio:
        audio = file_path

    original_name = os.path.splitext(os.path.basename(audio))[0]
    folder_path = os.path.dirname(audio)
    audio = AudioSegment.from_file(audio)
    audio_resampled = audio.set_frame_rate(44100)
    resampled_audio = os.path.join(folder_path, f"resampled_{original_name}.wav")
    audio_resampled.export(resampled_audio, format="wav")
    # gr.Warning(message=t("resample_warning"))
    return resampled_audio

########### Код для работы инвертера и ручного ансамбля

### Проверка файлов из списка на одинаковую частоту дискретизации

def analyze_sample_rate(files):
    """
    Анализирует частоту дискретизации для списка аудиофайлов
    Возвращает форматированную строку с результатами
    """
    if not files:
        return t("error_no_files")
    
    results = []
    common_rate = None
    all_same = True
    
    for file_info in files:
        try:
            audio = AudioSegment.from_file(file_info.name)
            rate = audio.frame_rate
            
            if common_rate is None:
                common_rate = rate
            elif common_rate != rate:
                all_same = False
                
            results.append(f"{file_info.name.split('/')[-1]}: {rate} Hz")
            
        except CouldntDecodeError:
            results.append(f"{file_info.name.split('/')[-1]}: {t('error_unsupported_format')}")
        except Exception as e:
            results.append(f"{file_info.name.split('/')[-1]}: {t('error_general', error=str(e))}")
    
    header = t("analyze_title") + "\n" + "-" * 50 + "\n"
    body = "\n".join(results)
    footer = "\n" + "-" * 50 + "\n"
    
    if all_same and common_rate is not None:
        footer += f"\n{t('all_same_rate', rate=common_rate)}"
    elif common_rate is not None:
        footer += f"\n{t('different_rates')}"
    
    return header + body + footer

### Инвертер

def load_audio(filepath):
    """Загрузка аудиофайла с помощью librosa"""
    if filepath is None:
        return None, None
    try:
        return librosa.load(filepath, sr=None, mono=False)
    except Exception as e:
        print(f"Ошибка загрузки аудио: {e}")
        return None, None

def process_channel(y1_ch, y2_ch, sr, method):
    """Обработка одного аудиоканала"""
    if method == "waveform":
        return y1_ch - y2_ch
    
    elif method == "spectrogram":
        S1 = librosa.stft(y1_ch, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
        S2 = librosa.stft(y2_ch, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
        
        mag1 = np.abs(S1)
        mag2 = np.abs(S2)
        
        mag_result = np.maximum(mag1 - mag2, 0)
        
        phase = np.angle(S1)
        
        S_result = mag_result * np.exp(1j * phase)
        
        return librosa.istft(
            S_result,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            length=len(y1_ch)
        )

def process_audio(audio1_path, audio2_path, out_format, method):
    y1, sr1 = load_audio(audio1_path)
    y2, sr2 = load_audio(audio2_path)
    
    if sr1 is None or sr2 is None:
        raise gr.Error(t("error_both_audio"))
    
    channels1 = 1 if y1.ndim == 1 else y1.shape[0]
    channels2 = 1 if y2.ndim == 1 else y2.shape[0]
    
    if channels1 > 1:
        y1 = y1.T  
    else:
        y1 = y1.reshape(-1, 1)
    
    if channels2 > 1:
        y2 = y2.T  
    else:
        y2 = y2.reshape(-1, 1)
    
    if sr1 != sr2:
        if channels2 > 1:
            y2_resampled = np.zeros((len(y2), channels2), dtype=np.float32)
            for c in range(channels2):
                y2_resampled[:, c] = librosa.resample(
                    y2[:, c], 
                    orig_sr=sr2, 
                    target_sr=sr1
                )
            y2 = y2_resampled
        else:
            y2 = librosa.resample(y2[:, 0], orig_sr=sr2, target_sr=sr1)
            y2 = y2.reshape(-1, 1)
        sr2 = sr1
    
    min_len = min(len(y1), len(y2))
    y1 = y1[:min_len]
    y2 = y2[:min_len]
    
    result_channels = []
    
    if channels1 == 1 and channels2 > 1:
        y2 = y2.mean(axis=1, keepdims=True)
        channels2 = 1
    
    for c in range(channels1):
        y1_ch = y1[:, c]
        
        if channels2 == 1:
            y2_ch = y2[:, 0]
        else:
            y2_ch = y2[:, min(c, channels2-1)]
        
        result_ch = process_channel(y1_ch, y2_ch, sr1, method)
        result_channels.append(result_ch)
    
    if len(result_channels) > 1:
        result = np.column_stack(result_channels)
    else:
        result = np.array(result_channels[0])
    
    if result.ndim > 1:
        for c in range(result.shape[1]):
            channel = result[:, c]
            max_val = np.max(np.abs(channel))
            if max_val > 0:
                result[:, c] = channel * 0.9 / max_val
    else:
        max_val = np.max(np.abs(result))
        if max_val > 0:
            result = result * 0.9 / max_val

    folder_path = os.path.dirname(audio2_path)
    inverted_wav = os.path.join(folder_path, "inverted.wav")
    sf.write(inverted_wav, result, sr1)
    inverted = os.path.join(folder_path, f"inverted_ensemble.{out_format}")
    write_audio_file(inverted, result.T, sr1, out_format, "320k")
    return inverted, inverted_wav

### Ручной ансамбль с автоподгоном длительности аудио

def manual_ensem(input_audios, method, weights, out_format):
    temp_dir = tempfile.mkdtemp()
    weights = [float(x) for x in weights.split(",")]
    padded_files = []

    audio_data = []
    max_length = 0
    for file in input_audios:
        
        data, sr = librosa.load(file, sr=None, mono=False)
        if data.ndim == 1:
            data = np.stack([data, data])
        elif data.shape[0] != 2:
            data = data.T
        audio_data.append([file, data])
        max_length = max(max_length, data.shape[1])
                          
    for i, [file, data] in enumerate(audio_data):
        if data.shape[1] < max_length:
            pad_width = ((0, 0), (0, max_length - data.shape[1]))
            padded_data = np.pad(data, pad_width, mode='constant')
        else:
            padded_data = data
        sf.write(f"{file}.wav", padded_data.T, sr)
        padded_files.append(f"{file}.wav")
    a1, a2 = ensemble_audio_files(padded_files, output=os.path.join(temp_dir, f"ensemble_{method}"), ensemble_type=method, weights=weights, out_format=out_format)
    return a1, a2

########### Основной инференс (только одиночная обработка)

def single_multi_inference(
    input_file, # Путь к аудио
    output_dir, # Путь к папке с выходными файлами
    model_type, # тип модели
    model_name, # Имя модели
    ext_inst, # Извлечение инструментала/инвертированного стема
    vr_aggr=5, # Агрессивность подавления на VR ARCH моделях
    output_format="wav", # Формат вывода
    output_bitrate="320k", # Битрейт (для кодеков сжатия с потерями)
    template="NAME_(STEM)_MODEL", 
    # Формат имени выходного стема
    # NAME - имя файла без расширения
    call_method="cli", 
    # Методы вызова инференса
    # "cli" - Через командную строку (чистится оперативная память после завершения работы инференса), 
    # "direct" - напрямую через функцию (Высокий риск преждевременного закрытия программы)
    selected_stems=[] # Выбранные стемы
):
    if model_type in ["mel_band_roformer", "bs_roformer", "mdx23c", "scnet", "htdemucs", "bandit", "bandit_v2"]:
        config_url = models_data[model_type][model_name]["config_url"]
        checkpoint_url = models_data[model_type][model_name]["checkpoint_url"]
        conf, ckpt = download_model(MODELS_CACHE_DIR, model_name, model_type, checkpoint_url, config_url)
        if model_type != "htdemucs":
            conf_editor(conf)
        
        if call_method == "cli":
            cmd = ["python", "-m", "separator.msst_separator", f"--input \"{input_file}\"", f"--store_dir \"{output_dir}\"", f"--model_type \"{model_type}\"", f"--model_name \"{model_name}\"", f"--config_path \"{conf}\"", f"--start_check_point \"{ckpt}\"", f"--output_format \"{output_format}\"", f"--output_bitrate \"{output_bitrate}\"", f"--template \"{template}\"", "--save_results_info",]
            
            if ext_inst:
                cmd.append("--extract_instrumental")
            if selected_stems:
                instruments = " ".join(f'"{stem}"' for stem in selected_stems)
                cmd.append(f'--selected_instruments {instruments}')
            
            command = " ".join(cmd)
            print(f"Executing command: {command}")
            os.system(command)
            
            results_path = os.path.join(output_dir, "results.json")
            if os.path.exists(results_path):
                with open(results_path) as f:
                    return json.load(f)
            else:
                return None
                    
        elif call_method == "direct":
            from separator.msst_separator import mvsep_offline
            return mvsep_offline(input_path=input_file, store_dir=output_dir, model_type=model_type, config_path=conf, start_check_point=ckpt, extract_instrumental=ext_inst, output_format=output_format, output_bitrate=output_bitrate, model_name=model_name, template=template, device_ids=0, disable_detailed_pbar=False, use_tta=False, force_cpu=False, verbose=False, selected_instruments=selected_stems, save_results_info=False)
    
    elif model_type in ["vr", "mdx"]:
    
        if model_type == "vr" and models_data[model_type][model_name]["custom_vr"]:
            config_url = models_data[model_type][model_name]["config_url"]
            checkpoint_url = models_data[model_type][model_name]["checkpoint_url"]               
            primary_stem = models_data[model_type][model_name]["primary_stem"]
            conf, ckpt = download_model(MODELS_CACHE_DIR, model_name, model_type, checkpoint_url, config_url)
            
            if call_method == "cli":
                cmd = ["python", "-m", "separator.uvr_sep custom_vr", f"--input_file \"{input_file}\"", f"--ckpt_path \"{ckpt}\"", f"--config_path \"{conf}\"", f"--bitrate \"{output_bitrate}\"", f"--model_name \"{model_name}\"", f"--template \"{template}\"", f"--output_format \"{output_format}\"", f"--primary_stem \"{primary_stem}\"", f"--aggression {vr_aggr}", f"--output_dir \"{output_dir}\"",]
                if selected_stems:
                    instruments = " ".join(f'"{stem}"' for stem in selected_stems)
                    cmd.append(f'--selected_instruments {instruments}')
                command = " ".join(cmd)
                print(f"Executing command: {command}")
                os.system(command)
                
                results_path = os.path.join(output_dir, "results.json")
                if os.path.exists(results_path):
                    with open(results_path) as f:
                        return json.load(f)
                else:
                    return None
                
            elif call_method == "direct":
                from separator.uvr_sep import custom_vr_separate
                return custom_vr_separate(input_file=input_file, ckpt_path=ckpt, config_path=conf, bitrate=output_bitrate, model_name=model_name, template=template, output_format=output_format, primary_stem=primary_stem, aggression=vr_aggr, output_dir=output_dir, selected_instruments=selected_stems)
                                    
        else:
            if call_method == "cli":
                cmd = ["python", "-m", "separator.uvr_sep uvr", f"--input_file \"{input_file}\"", f"--output_dir \"{output_dir}\"", f"--template \"{template}\"", f"--bitrate \"{output_bitrate}\"", f"--model_dir \"{MODELS_CACHE_DIR}\"",  f"--model_type \"{model_type}\"", f"--model_name \"{model_name}\"", f"--output_format \"{output_format}\"", f"--aggression {vr_aggr}",]
                if selected_stems:
                    instruments = " ".join(f'"{stem}"' for stem in selected_stems)
                    cmd.append(f'--selected_instruments {instruments}')
                
                command = " ".join(cmd)
                print(f"Executing command: {command}")
                os.system(command)
                
                results_path = os.path.join(output_dir, "results.json")
                if os.path.exists(results_path):
                    with open(results_path) as f:
                        return json.load(f)
                else:
                    return None

            elif call_method == "direct":
                from separator.uvr_sep import non_custom_uvr_inference
                return non_custom_uvr_inference(input_file=input_file, output_dir=output_dir, template=template, bitrate=output_bitrate, model_dir=MODELS_CACHE_DIR, model_type=model_type, model_name=model_name, output_format=output_format, aggression=vr_aggr, selected_instruments=selected_stems)

########### Обёртка для инференса в Gradio #1 (Одиночная и пакетная обработка)

def mvsepless(
    input_audio="test.mp3", # Путь к аудио / Список путей к аудио
    output_dir="/content/output", # Путь к папке с выходными файлами
    model_type="mel_band_roformer", # тип модели
    model_name="TEST", # Имя модели
    ext_inst=False, # Извлечение инструментала/инвертированного стема
    vr_aggr=5, # Агрессивность подавления на VR ARCH моделях
    output_format="wav", # Формат вывода
    output_bitrate="320k", # Битрейт (для кодеков сжатия с потерями)
    template="NAME_(STEM)_MODEL", 
    # Формат имени выходного стема
    # NAME - имя файла без расширения
    call_method="cli", 
    # Методы вызова инференса
    # "cli" - Через командную строку (чистится оперативная память после завершения работы инференса), 
    # "direct" - напрямую через функцию (Высокий риск преждевременного закрытия программы)
    selected_stems=[] # Выбранные стемы
):

    if not "STEM" in template:
        template = template + "_STEM"

    if not input_audio or input_audio == "":
        print(t("no_audio"))
        output_text = t("no_audio")
        return output_text, None

    if not model_type or model_type == "" or not model_name or model_name == "":
        print(t("no_model"))
        output_text = t("no_model")
        return output_text, None

    if not output_dir:
        print(t("no_output_dir"))
        output_text = t("no_output_dir")
        return output_text, None

    model_fullname = models_data[model_type][model_name]["full_name"]

    if input_audio is not None and isinstance(input_audio, str):
        results = single_multi_inference(input_audio, output_dir, model_type, model_name, ext_inst, vr_aggr, output_format, output_bitrate, template, call_method, selected_stems)
        output_text = f"""
        {t("algorithm", model_fullname=model_fullname)}
        {t("output_format_info", output_format=output_format)}
        """
        return output_text, results
            
    if input_audio is not None and isinstance(input_audio, list):
        progress = gr.Progress()
        batch_results = []
        for i, file in enumerate(input_audio):
            base_name = os.path.splitext(os.path.basename(file))[0]
            total_steps = len(input_audio)
            progress(
                (i+1, total_steps),
                desc=t("processing", base_name=base_name),
                unit=t("files")
            )             
            results = single_multi_inference(file, output_dir, model_type, model_name, ext_inst, vr_aggr, output_format, output_bitrate, template, call_method, selected_stems)
            batch_results.append((base_name, results))
        output_text = f"""
        {t("algorithm", model_fullname=model_fullname)}
        {t("output_format_info", output_format=output_format)}
        """
            
        return output_text, batch_results

########### Дополнительная обёртка для инференса в Gradio #2

def mvsepless_sep_gradio(a1, a2, b, c, d, e, f, g, h, i_stem, batch, local_check):
    if local_check == False:
        if not a1:
            text, output = mvsepless(a1, b, c, d, e, f, g, OUTPUT_BITRATE, h, CALL_METHOD, i_stem)
            if batch == True:
                return text, None
            else:
                results = []
                for i in range(20):
                    results.append(gr.update(visible=False, label=None, value=None))
                return (gr.update(value=text),) + tuple(results)
        elif a1 is not None and isinstance(a1, list):
            text, batch_separated = mvsepless(a1, b, c, d, e, f, g, OUTPUT_BITRATE, h, CALL_METHOD, i_stem)
            return text, batch_separated        
        elif a1 is not None and isinstance(a1, str):
            text, output_audio = mvsepless(a1, b, c, d, e, f, g, OUTPUT_BITRATE, h, CALL_METHOD, i_stem)
            results = []
            if output_audio is not None:
                for i, (stem, output_file) in enumerate(output_audio[:20]):
                    results.append(gr.update(
                        visible=True,
                        label=stem,
                        value=output_file
                    ))
            while len(results) < 20:
                results.append(gr.update(visible=False, label=None, value=None))
            return (gr.update(value=text),) + tuple(results)

    if local_check == True:
        if not a2:
            text, output = mvsepless(a2, b, c, d, e, f, g, "320k", h, "cli", i_stem)
            if batch == True:
                return text, None
            else:
                results = []
                for i in range(20):
                    results.append(gr.update(visible=False, label=None, value=None))
                return (gr.update(value=text),) + tuple(results)
        elif a2 is not None and isinstance(a2, list):
            text, batch_separated = mvsepless(a2, b, c, d, e, f, g, "320k", h, "cli", i_stem)
            return text, batch_separated        
        elif a2 is not None and isinstance(a2, str):
            text, output_audio = mvsepless(a2, b, c, d, e, f, g, "320k", h, "cli", i_stem)
            results = []
            if output_audio is not None:
                for i, (stem, output_file) in enumerate(output_audio[:20]):
                    results.append(gr.update(
                        visible=True,
                        label=stem,
                        value=output_file
                    ))
            while len(results) < 20:
                results.append(gr.update(visible=False, label=None, value=None))
            return (gr.update(value=text),) + tuple(results)

########### Авто-ансамбль

def ensembless(input_audio, input_settings, type, out_format):

    progress = gr.Progress()
    progress(0, desc=f"{t('process1')}...")

    base_name = os.path.splitext(os.path.basename(input_audio))[0]
    temp_dir = os.path.join(ENSEMBLESS_OUTPUT_DIR, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    source_files = []
    output_s_files = []
    output_s_weights = []
    block_count = len(input_settings)

    for i, (input_model, weight, s_stem) in enumerate(input_settings):
           
        progress(i / block_count, desc=f"{t('process2')} {i+1}/{block_count}")
    
        model_type, model_name = input_model.split(" / ")
        
        output_s_dir = os.path.join(temp_dir, f"{model_type}_{model_name}_s_stems")
        
        output = single_multi_inference(input_audio, output_s_dir, model_type, model_name, True, vr_aggr=10, output_format="wav", output_bitrate="320k", template="MODEL_STEM", call_method="cli", selected_stems=[])
        
        for stem, file in output:       
            source_files.append(file)
            if stem == s_stem:
               output_s_files.append(file)
               output_s_weights.append(weight)

    progress(0.9, desc=f"{t('process3')}...")
               
    padded_files = []

    audio_data = []
    max_length = 0
    for file in output_s_files:
        
        data, sr = sf.read(file)
        if data.ndim == 1:
            data = np.stack([data, data])
        elif data.shape[0] != 2:
            data = data.T
        audio_data.append([file, data])
        max_length = max(max_length, data.shape[1])
                          
    for i, [file, data] in enumerate(audio_data):
        if data.shape[1] < max_length:
            pad_width = ((0, 0), (0, max_length - data.shape[1]))
            padded_data = np.pad(data, pad_width, mode='constant')
        else:
            padded_data = data
        sf.write(file, padded_data.T, sr)
        padded_files.append(file)

    progress(0.95, desc=f"{t('process4')}...")
           
    output, output_wav = ensemble_audio_files(files=output_s_files, output=os.path.join(temp_dir, f"ensemble_{base_name}_{type}"), ensemble_type=type, weights=output_s_weights, out_format=out_format)

    return output, output_wav, source_files

########### Менеджер для авто-ансамбля

class EnsembleManager:
    def __init__(self):
        self.models = []
    
    def add_model(self, model_type, model_name, stem, weight):
        model_info = {
            'type': model_type,
            'name': model_name,
            'stem': stem,
            'weight': float(weight)
        }
        self.models.append(model_info)
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
            columns = ["#", t("model_type"), t("model_name"), t("stem"), t("weight")]
            return pd.DataFrame(columns=columns)
        
        data = []
        for i, model in enumerate(self.models):
            data.append([
                f"{i+1}",
                model['type'],
                model['name'],
                model['stem'],
                model['weight']
            ])
        columns = ["#", t("model_type"), t("model_name"), t("stem"), t("weight")]
        return pd.DataFrame(data, columns=columns)
    
    def get_settings(self):
        return [(f"{m['type']} / {m['name']}", m['weight'], m['stem']) for m in self.models]

manager = EnsembleManager()

def add_model(model_type, model_name, stem, weight):
    return manager.add_model(model_type, model_name, stem, weight)

def remove_model(index):
    if index >= 0:
        return manager.remove_model(index-1)  
    return manager.get_df()

def clear_all_models():
    return manager.clear_models()

def run_ensemble(input_audio, ensemble_type, output_format):
    if not manager.models:
        raise gr.Error(t("error_no_models"))
        
    if not input_audio:
        raise gr.Error(t("error_no_audio"))
    
    input_settings = manager.get_settings()
    
    output, output_wav, result_source = ensembless(
        input_audio=input_audio,
        input_settings=input_settings,
        type=ensemble_type,
        out_format=output_format,
    )
    return output, output_wav, result_source

########### Создание интерфейса в Gradio

def create_mvsepless_app(lang):
    gr.HTML(f"<h1><center> {t('app_title')} </center></h1>")
    with gr.Tabs():
        ########### Разделение
        with gr.Tab(t("separation")):
            with gr.Column():
                with gr.Group(visible=True) as upload_group:
                    input_audio = gr.Audio(show_label=False, type="filepath", interactive=True, elem_classes="fixed-height", sources="upload")
                    input_audios = gr.Files(show_label=False, type="filepath", visible=False, interactive=True, file_types=[".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff"], elem_classes="fixed-height")
                input_file_explorer = gr.FileExplorer(show_label=False, root_dir="/content", file_count="single", visible=False, elem_classes="fixed-height")
                local_check = gr.Checkbox(label=t("local_path"), value=False, interactive=True)
                batch_separation = gr.Checkbox(label=t("batch_processing"), value=False, interactive=True, info=t("batch_info"))
            ########### Основной инференс
            with gr.Tab(t("inference")):
                batch_results_state = gr.State()
                with gr.Row(equal_height=False):
                    with gr.Column(variant="panel"):
                        with gr.Group():
                            with gr.Row():
                                model_type = gr.Dropdown(label=t("model_type"), choices=list(models_data.keys()), value=list(models_data.keys())[0], interactive=True, filterable=False)
                                model_name = gr.Dropdown(label=t("model_name"), choices=list(models_data[list(models_data.keys())[0]].keys()), value=list(models_data[list(models_data.keys())[0]].keys())[0], interactive=True, filterable=False)
                            ext_inst = gr.Checkbox(label=t("extract_instrumental"), visible=True, value=True, interactive=True, info=t("extract_info"))
                            vr_aggr_slider = gr.Slider(label=t("vr_aggressiveness"), minimum=0, maximum=100, step=1, visible=False, interactive=True, value=5)
                            stems = gr.CheckboxGroup(label=t("stems_list"), choices=models_data[list(models_data.keys())[0]][list(models_data[list(models_data.keys())[0]].keys())[0]]["stems"], value=None, interactive=False, info=t("stems_info", target_instrument="vocals"))
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

                            # Создаем первую строку с видимым плеером
                            with gr.Row():
                                audio1 = gr.Audio(visible=True, interactive=False, type="filepath", show_download_button=True)
                                audio2 = gr.Audio(visible=False, interactive=False, type="filepath", show_download_button=True)
                                output_stems.extend([audio1, audio2])

                            # Создаем остальные строки со скрытыми плеерами
                            for _ in range(9):  # 9 строк (итого 10 строк = 20 элементов)
                                with gr.Row():
                                    audio1 = gr.Audio(visible=False, interactive=False, type="filepath", show_download_button=True)
                                    audio2 = gr.Audio(visible=False, interactive=False, type="filepath", show_download_button=True)
                                    output_stems.extend([audio1, audio2])



            ########### Авто-ансамбль
            with gr.Tab(t("auto_ensemble")):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(f"### {t('model_selection')}")
                        e_model_type = gr.Dropdown(
                            choices=get_model_types(),
                            label=t("model_type"),
                            value=get_model_types()[0] if get_model_types() else None,
                            filterable=False
                        )
                        e_model_name = gr.Dropdown(
                            choices=get_models_by_type(get_model_types()[0]),
                            label=t("model_name"),
                            interactive=True,
                            value=get_models_by_type(get_model_types()[0])[0],
                            filterable=False
                        )
                        e_stem = gr.Dropdown(
                            choices=get_stems_by_model(get_model_types()[0], get_models_by_type(get_model_types()[0])[0]),
                            label=t("stem_selection"),
                            interactive=True,
                            filterable=False
                        )
                        e_weight = gr.Slider(
                            label=t("weight"),
                            value=1.0,
                            minimum=0.1,
                            maximum=10.0,
                            step=0.1
                        )
                        e_add_btn = gr.Button(t("add_button"), variant="primary")
            
                    with gr.Column(scale=2):
                        gr.Markdown(f"### {t('current_ensemble')}")
                        ensemble_df = gr.Dataframe(
                            value=manager.get_df(),
                            headers=["#", t("model_type"), t("model_name"), t("stem"), t("weight")],
                            datatype=["str", "str", "str", "str", "number"],
                            interactive=False
                        )
                        
                        with gr.Row(equal_height=True):
                            remove_idx = gr.Number(
                                label=t("remove_index"),
                                precision=0,
                                minimum=1,
                                interactive=True
                            )
                            with gr.Column():
                                remove_btn = gr.Button(t("remove_button"), variant="stop")
                                clear_btn = gr.Button(t("clear_button"), variant="stop")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(f"### {t('input_audio')}")
                        resample_local_btn = gr.Button(t("resample"))
                        e_input_audio_resampled = gr.Textbox(label=t("resampled_path"), interactive=False, lines=3, max_length=25)
                        
                        gr.Markdown(f"### {t('settings')}")
                        ensemble_type = gr.Dropdown(
                            choices=['avg_wave', 'median_wave', 'min_wave', 'max_wave', 
                                     'avg_fft', 'median_fft', 'min_fft', 'max_fft'],
                            value='avg_fft',
                            label=t("method"),
                            filterable=False
                        )
                        e_output_format = gr.Dropdown(
                            choices=["wav", "mp3", "flac", "m4a", "aac", "ogg", "opus", "aiff"],
                            value="mp3",
                            label=t("output_format"),
                            filterable=False
                        )
                        e_run_btn = gr.Button(t("run_button"), variant="primary")
                    with gr.Column(scale=2):
    
                        with gr.Tab(t('results')):
                        
                            with gr.Column():
                                e_output_audio = gr.Audio(label=t("results"), type="filepath", interactive=False, show_download_button=True)
                                e_output_wav = gr.Text(label="Результат в WAV", interactive=False, visible=False)
                    
                                gr.Markdown(f"###### {t('inverted_result')}")
                    
                                invert_method = gr.Radio(
                                    choices=["waveform", "spectrogram"],
                                    label=t("invert_method"),
                                    value="waveform"
                                )
                                invert_btn = gr.Button(t("invert_button"))
                                inverted_output_audio = gr.Audio(label=t("inverted_result"), type="filepath", interactive=False, show_download_button=True)
                                inverted_wav = gr.Text(label="Инвертированный результат в WAV", interactive=False, visible=False)
    
                        with gr.Tab(t('result_source')):
                            result_source = gr.Files(interactive=False, label=t('result_source'))

        ########### Плагины
        with gr.Tab(t("plugins")):
            plugins = [] 

            with gr.Tab(t('upload')):
                with gr.Blocks():
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

                        plugin_func = None
                        name_func = None

                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if callable(attr):
                                if attr_name.endswith("plugin"):
                                    plugin_func = attr
                                elif attr_name.endswith("plugin_name"):
                                    name_func = attr

                        if plugin_func is not None:
                            plugin_name = name_func() if name_func is not None else module_name
                            plugins.append((plugin_name, plugin_func))

              except Exception as e:
                print(e)

            for name, func in plugins:
                try:
                    print(t("loading_plugin", name=name))
                    with gr.Tab(name):
                        func(lang)
                except Exception as e:
                    print(t("error_loading_plugin", e=e))
                    pass
        ########### Extra
        with gr.Tab("Extra"):
            ########### Ручной ансамбль
            with gr.Tab(t("manual_ensemble")):
                with gr.Row(equal_height=True):
                    input_files = gr.Files(show_label=False, type="filepath", file_types=[".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff"])
                    with gr.Column():
                        info_audios = gr.Textbox(label="", interactive=False)
                        man_method = gr.Dropdown(
                            choices=['avg_wave', 'median_wave', 'min_wave', 'max_wave', 
                                     'avg_fft', 'median_fft', 'min_fft', 'max_fft'],
                            value='avg_fft',
                            label=t("method"),
                            filterable=False
                        )
                        
                        weights_input = gr.Textbox(label=t("weights_input"), value="1.0,1.0")
                        
                        output_man_format = gr.Dropdown(
                            choices=["wav", "mp3", "flac", "m4a", "aac", "ogg", "opus", "aiff"],
                            value="mp3",
                            label=t("output_format"),
                            filterable=False
                        )

                run_man_btn = gr.Button(t("run_button"), variant="primary")
                        
                output_man_audio = gr.Audio(label=t("results"), type="filepath", interactive=False, show_download_button=True)
                output_man_wav = gr.Text(label="Результат в WAV", interactive=False, visible=False)
                
            ########### Инвертер
            with gr.Tab(t("inverter")):
                with gr.Row():
                    audio1 = gr.Audio(label=t("main_audio"), type="filepath", elem_classes="fixed-height", sources="upload")
                    audio2 = gr.Audio(label=t("audio_to_remove"), type="filepath", elem_classes="fixed-height", sources="upload")
                invert_man_method = gr.Radio(
                    choices=["waveform", "spectrogram"],
                    label=t("processing_method"),
                    value="waveform"
                )
                output_man_i_format = gr.Dropdown(
                    choices=["wav", "mp3", "flac", "m4a", "aac", "ogg", "opus", "aiff"],
                    value="mp3",
                    label=t("output_format"),
                    filterable=False
                )
                invert_man_btn = gr.Button(t("invert_button"))
                
                with gr.Column():
                    invert_man_output = gr.Audio(label=t("results"), interactive=False, show_download_button=True, elem_classes="fixed-height2")
                    invert_man_output_wav = gr.Text(interactive=False, visible=False)

            with gr.Tab(t("model_loading")):
                dw_m_model_type = gr.Dropdown(label=t("model_type"), choices=list(models_data.keys()), value=list(models_data.keys())[0], interactive=True, filterable=False)
                dw_m_model_name = gr.Dropdown(label=t("model_name"), choices=list(models_data[list(models_data.keys())[0]].keys()), value=list(models_data[list(models_data.keys())[0]].keys())[0], interactive=True, filterable=False)
                dw_m_btn = gr.Button(t("download_model_btn"))

        with gr.Tab(t("settings_tab")):
            config_preview = gr.Code(label=t("settings_config"), value=json.dumps(CONFIG, indent=4, ensure_ascii=False))
            restart_btn = gr.Button(t("restart_btn"), variant="stop")
            with gr.Tab("UI"):
                with gr.Column(variant="panel"):
                    local_font = gr.Dropdown(label=t("settings_select_local_font"), choices=get_font_files())
                    upload_local_font = gr.File(label=t("settings_upload_fonts"), interactive=True, file_count="multiple", file_types=[".ttf", ".otf", ".woff", ".eot"], elem_classes="fixed-height")
                    get_list_fonts = gr.Button(t("settings_get_list_fonts"))
                with gr.Column(variant="panel"):
                    google_font_info = gr.Markdown(t("settings_info_font"))
                    google_font = gr.Text(label=t("settings_google_font"))
                set_font_btn = gr.Button(t("settings_set_font"), variant="primary")
                with gr.Column(variant="panel"):
                    language = gr.Radio(label=t("settings_language"), choices=LANGS, value=CURRENT_LANG)
                    server_share = gr.Checkbox(label=t("settings_share"), value=GRADIO_SHARE)
                    server_debug = gr.Checkbox(label=t("settings_debug"), value=GRADIO_DEBUG)
            with gr.Tab("Inference_settings"):
                with gr.Column(variant="panel"):
                    current_output_dir = gr.Text(label="Current output directory", value=OUTPUT_DIR)
                    output_dir = gr.Text(label="Output Directory", value=OUTPUT_DIR)
                    models_cache_dir = gr.Text(label="Cache model dir", value=MODELS_CACHE_DIR)
                    output_bitrate = gr.Slider(label="Bitrate", minimum=32, maximum=320, value=320, step=1)
                    call_method = gr.Radio(label="Call method", choices=CALL_METHODS, value=CALL_METHOD)
                   

    ########### Обработчики событий

    output_dir.change(fn=write_OUTPUT_DIR_config, inputs=output_dir).then(fn=(lambda: gr.update(value=json.dumps(CONFIG, indent=4, ensure_ascii=False))), inputs=None, outputs=config_preview)
    models_cache_dir.change(fn=write_MODELS_CACHE_DIR_config, inputs=models_cache_dir).then(fn=(lambda: gr.update(value=json.dumps(CONFIG, indent=4, ensure_ascii=False))), inputs=None, outputs=config_preview)
    output_bitrate.change(fn=write_OUTPUT_BITRATE_config, inputs=output_bitrate).then(fn=(lambda: gr.update(value=json.dumps(CONFIG, indent=4, ensure_ascii=False))), inputs=None, outputs=config_preview)
    call_method.change(fn=write_CALL_METHOD_config, inputs=call_method).then(fn=(lambda: gr.update(value=json.dumps(CONFIG, indent=4, ensure_ascii=False))), inputs=None, outputs=config_preview)

    language.change(fn=write_LANG_config, inputs=language).then(fn=(lambda: gr.update(value=json.dumps(CONFIG, indent=4, ensure_ascii=False))), inputs=None, outputs=config_preview)
    server_share.change(fn=write_SHARE_config, inputs=server_share).then(fn=(lambda: gr.update(value=json.dumps(CONFIG, indent=4, ensure_ascii=False))), inputs=None, outputs=config_preview)
    server_debug.change(fn=write_DEBUG_config, inputs=server_debug).then(fn=(lambda: gr.update(value=json.dumps(CONFIG, indent=4, ensure_ascii=False))), inputs=None, outputs=config_preview)
    set_font_btn.click(fn=(lambda x, y: write_FONT_config(y if y != "" else x) ), inputs=[local_font, google_font], outputs=None).then(fn=(lambda: gr.update(value=json.dumps(CONFIG, indent=4, ensure_ascii=False))), inputs=None, outputs=config_preview)
    get_list_fonts.click(fn=(lambda: gr.update(choices=get_font_files())), inputs=None, outputs=local_font)
    upload_local_font.upload(fn=write_fonts, inputs=upload_local_font, outputs=None)



    upload_btn.click(fn=upload_plugin_list, inputs=upload_plugin_files)
    restart_btn.click(restart_ui)
    local_check.change(fn=(lambda x:(gr.update(visible=False if x == True else True), gr.update(value=None), gr.update(value=None), gr.update(visible=True if x == True else False))), inputs=local_check, outputs={upload_group, input_audio, input_audios, input_file_explorer})
    dw_m_btn.click(fn=downloader_models, inputs=[dw_m_model_type, dw_m_model_name], outputs=None)   
    batch_separation.change(fn=(lambda x: (gr.update(visible=True if x == True else False, value=None), gr.update(visible=True if x == True else False), gr.update(visible=False if x == True else True, value=None), gr.update(visible=False if x == True else True))), inputs=batch_separation, outputs=[input_audios, batch_separate_btn, input_audio, single_separate_btn]).then(fn=(lambda x: gr.update(file_count="multiple" if x == True else "single", value=None)), inputs=batch_separation, outputs=input_file_explorer)
    model_type.change(fn=lambda x: gr.update(visible=True if x == "vr" else False), inputs=model_type, outputs=vr_aggr_slider).then(
        fn=lambda x: gr.update(choices=list(models_data[x].keys()), value=list(models_data[x].keys())[0]), inputs=model_type, outputs=model_name).then(fn=(lambda x: gr.update(visible=False if x in ["vr", "mdx"] else True)), inputs=model_type, outputs=ext_inst)
    dw_m_model_type.change(fn=lambda x: gr.update(choices=[model for model in list(models_data[x].keys()) if (models_data[x][model]["checkpoint_url"] if x not in ["vr", "mdx"] else None) or (models_data[x][model]["custom_vr"] if x == "vr" else None)], value=None if not [model for model in list(models_data[x].keys()) if (models_data[x][model]["checkpoint_url"] if x not in ["vr", "mdx"] else None) or (models_data[x][model]["custom_vr"] if x == "vr" else None)] else [model for model in list(models_data[x].keys()) if (models_data[x][model]["checkpoint_url"] if x not in ["vr", "mdx"] else None) or (models_data[x][model]["custom_vr"] if x == "vr" else None)][0]), inputs=dw_m_model_type, outputs=dw_m_model_name)
    model_name.change(fn=lambda x, y: gr.update(choices=list(models_data[x][y]["stems"]), value=None, interactive=True if models_data[x][y]["target_instrument"] == None else False, info=t("stems_info", target_instrument=models_data[x][y]["target_instrument"]) if models_data[x][y]["target_instrument"] != None else t("stems_info2")), inputs=[model_type, model_name], outputs=stems).then(fn=(lambda x, y: gr.update(value=False if models_data[x][y]["target_instrument"] == None else True) ), inputs=[model_type, model_name], outputs=ext_inst)
    single_separate_btn.click(fn=(lambda : gr.update(choices=None, visible=False, value=None)), inputs=None, outputs=batch_select_dir).then(fn=(lambda x: os.path.join(OUTPUT_DIR, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{x}')), inputs=model_name, outputs=current_output_dir).then(fn=mvsepless_sep_gradio, inputs=[input_audio, input_file_explorer, output_dir, model_type, model_name, ext_inst, vr_aggr_slider, output_format, template, stems, batch_separation, local_check], outputs=[output_info, *output_stems])
    batch_separate_btn.click(fn=(lambda : gr.update(choices=None, visible=False, value=None)), inputs=None, outputs=batch_select_dir).then(fn=(lambda x: os.path.join(OUTPUT_DIR, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{x}')), inputs=model_name, outputs=current_output_dir).then(fn=mvsepless_sep_gradio, inputs=[input_audios, input_file_explorer, output_dir, model_type, model_name, ext_inst, vr_aggr_slider, output_format, template, stems, batch_separation, local_check], outputs=[output_info, batch_results_state]).then(fn=batch_show_names, inputs=batch_results_state, outputs=batch_select_dir)
    batch_select_dir.change(fn=batch_show_results, inputs=[batch_results_state, batch_select_dir], outputs=[*output_stems])
    e_model_type.change(update_model_dropdown, inputs=e_model_type, outputs=e_model_name)
    e_model_name.change(update_stem_dropdown, inputs=[e_model_type, e_model_name], outputs=e_stem)
    invert_btn.click(process_audio, inputs=[e_input_audio_resampled, e_output_wav, e_output_format, invert_method], outputs=[inverted_output_audio, inverted_wav])
    input_audio.upload(resample_audio, inputs=[input_audio, input_file_explorer], outputs=e_input_audio_resampled)
    resample_local_btn.click(resample_audio, inputs=[input_audio, input_file_explorer], outputs=e_input_audio_resampled)
    e_add_btn.click(add_model, inputs=[e_model_type, e_model_name, e_stem, e_weight], outputs=ensemble_df)
    remove_btn.click(remove_model, inputs=remove_idx, outputs=ensemble_df)
    clear_btn.click(clear_all_models, outputs=ensemble_df)
    e_run_btn.click(fn=(lambda: (gr.update(value=None), gr.update(value=None))), inputs=None, outputs=[inverted_output_audio, inverted_wav]).then(run_ensemble, inputs=[e_input_audio_resampled, ensemble_type, output_format], outputs=[e_output_audio, e_output_wav, result_source])
    invert_man_btn.click(process_audio, inputs=[audio1, audio2, output_man_i_format, invert_man_method], outputs=[invert_man_output, invert_man_output_wav])
    input_files.upload(fn=analyze_sample_rate, inputs=input_files, outputs=info_audios)                
    run_man_btn.click(manual_ensem, inputs=[input_files, man_method, weights_input, output_man_format], outputs=[output_man_audio, output_man_wav])

def parse_args():
    parser = argparse.ArgumentParser(description="Базовый интерфейс для разделения музыки и вокала")
    
    parser.add_argument("--ngrok_token", type=str, help="Аутентификация (формат: username:password)")
    parser.add_argument("--ssl-keyfile", type=str, help="Путь к SSL ключу")
    parser.add_argument("--ssl-certfile", type=str, help="Путь к SSL сертификату")
    
    return parser.parse_args()

if __name__ == "__main__":

    if not os.path.exists(CONFIG_UI_PATH):
        write_UI_settings()
    else:
        read_UI_settings()

    args = parse_args()

    GRADIO_SSL_KEYFILE = args.ssl_keyfile
    GRADIO_SSL_CERTFILE = args.ssl_certfile

    css = """
.fixed-height {
    height: 160px !important;  /* Фиксируем высоту */
    min-height: 160px !important; /* Запрещаем уменьшение */
}
.fixed-height2 {
    height: 250px !important;  /* Фиксируем высоту */
    min-height: 250px !important; /* Запрещаем уменьшение */
}
    """
    if GOOGLE_FONT.endswith((".ttf", ".otf", ".woff", ".eot")):

        # Кодирование шрифта в Base64
        with open(GOOGLE_FONT, "rb") as font_file:
            base64_font = base64.b64encode(font_file.read()).decode("utf-8")

        # CSS с встроенным шрифтом
        css_font = f"""
        @font-face {{
            font-family: 'CustomFont';
            src: url(data:font/truetype;charset=utf-8;base64,{base64_font}) format('truetype');
        }}
        body, .gradio-container, .input, .output, .button, .textbox, .label, .title, .description {{
            font-family: 'CustomFont', sans-serif !important;
        }}
        """
        css = css_font + css

    print(GOOGLE_FONT)

    with gr.Blocks(title="Разделение музыки и вокала", theme=mvsepless_theme(GOOGLE_FONT if GOOGLE_FONT.endswith((".ttf", ".otf", ".woff", ".eot")) == False else "Roboto"), css=css) as MVSEPLESS_UI:
        create_mvsepless_app(CURRENT_LANG)

    if args.ngrok_token:
        ngrok.set_auth_token(args.ngrok_token)
        ngrok.kill()
        tunnel = ngrok.connect(GRADIO_PORT)
        print(f"Публичная ссылка - {tunnel.public_url}")

    load_ui(MVSEPLESS_UI)
