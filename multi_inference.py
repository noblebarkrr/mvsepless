import os
import ast
import time
import shutil
import sys
import gc
import yaml
import argparse
import json
import subprocess
from datetime import datetime
from typing import Literal
import zipfile
import gradio as gr
from tqdm import tqdm
import urllib.request
import yt_dlp
import importlib.util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

class MVSEPLESS:
    def __init__(self, lang="ru"):
        self.lang = lang
        self.model_types = ["mel_band_roformer", "bs_roformer", "mdx23c", "scnet", "htdemucs", "bandit", "bandit_v2", "vr", "mdx"]
        self.output_formats = ["mp3", "wav", "flac", "ogg", "opus", "m4a", "aac", "aiff"]
        self.python_path = os.environ.get("MVSEPLESS_PYTHON_PATH", "python")
        self.history_file_path = os.environ.get("MVSEPLESS_HISTORY_PATH", os.path.join(SCRIPT_DIR, "history.json"))
        self.models_info_path = os.environ.get("MVSEPLESS_MODEL_LIST", os.path.join(SCRIPT_DIR, "models.json"))
        self.output_app_base_dir = os.environ.get("MVSEPLESS_OUTPUT_DIR", os.path.join(SCRIPT_DIR, "output"))
        self.plugins_dir = os.environ.get("MVSEPLESS_PLUGINS_DIR", os.path.join(SCRIPT_DIR, "plugins"))
        self.download_dir = os.environ.get("MVSEPLESS_DOWNLOAD_DIR", os.path.join(SCRIPT_DIR, "downloaded"))
        self.models_cache_dir = os.environ.get("MVSEPLESS_MODELS_CACHE_DIR", os.path.join(SCRIPT_DIR, "separator", "models_cache"))
        self.I18N_DATA = {
            "ru": {
                "title": "Разделение музыки и вокала",
                "starting": "Начало разделения с моделью",
                "bitrate": "Битрейт",
                "selected_stems": "Выбранные стемы",
                "batch_base_dir": "Список файлов",
                "batch_progress_files": "Файл",
                "batch_separation": "Пакетная обработка",
                "upload_label": "Входное аудио",
                "url_label": "Введите ссылку",
                "path_label": "Введите путь к аудиофайлу",
                "url_placeholder": "Ссылка на аудиофайл",
                "path_placeholder": "/путь/к/аудио/",
                "url_btn": "Ввести URL",
                "path_btn": "Ввести путь к файлу",
                "upload_cookie": "Загрузить cookie",
                "download_audio_btn": "Скачать аудио",
                "upload_btn": "Загрузить аудио",
                "not_input_file": "Входной файл не указан",
                "file_not_exists": "Указанного файла не существует",
                "model_type": "Тип модели",
                "model_id": "ID",
                "model_name": "Имя модели",
                "model_not_exist": "Данной модели не существует",
                "output_format": "Формат вывода",
                "not_supported_format": "Указанный формат не поддерживается",
                "separate": "Разделить",
                "error_no_input": "Ошибка: нет входного аудио.",
                "error_no_model": "Ошибка: не выбрана модель.",
                "error_no_models": "Нет моделей, у которых есть указанный стем.",
                "error_unsupported_model_type": "Нeподдерживаемый тип модели",
                "error_invalid_format": "Ошибка: неверный формат вывода.",
                "output_zip": "Скачать ZIP",
                "inference_tab": "Инференс",
                "results": "Результаты",
                "extract_instrumental": "Извлечь инструментал",
                "stems_list": "Список стемов",
                "target_instrument": "Целевой инструмент",
                "stems": "Стемы",
                "stems_info": "Выбор стемов недоступен\nДля извлечения второго стема включите \"Извлечь инструментал\"",
                "stems_info2": "Для получения остатка (при выбранных стемах), включите \"Извлечь инструментал\"",
                "add_settings": "Дополнительные настройки",
                "template": "Формат имени",
                "vr_aggr_slider": "Агрессивность",
                "template_help": """
> Формат имени результатов в мульти-инференсе.

> Доступные ключи для формата имени стемов:
> (изменить формат имени стемов можно здесь)
> * **NAME** - Имя входного файла
> * **STEM** - Название стема (например, vocals, drums, bass)
> * **MODEL** - Имя модели (например, Mel-Band-Roformer_Instrumental_FvX_gabox, UVR-MDX-NET-Inst_HQ_3)
> * **ID** - ID модели (например, BS-Roformer_SW: 217)

> Пример:
> * **Шаблон:** NAME_STEM_MODEL
> * **Результат:** test_vocals_Mel-Band-Roformer_Instrumental_FvX_gabox

<div style="color: red; font-weight: bold; background-color: #ffecec; padding: 10px; border-left: 3px solid red; margin: 10px 0;">

Используйте ТОЛЬКО указанные ключи (NAME, STEM, MODEL) во избежание повреждения файлов. 

НЕ добавляйте дополнительный текст или символы вне этих ключей, либо делайте это с осторожностью.

</div>
                """,
                "plugin_manager": "Менеджер плагинов",
                "upload_label_2": "Загрузить плагин",
                "upload_btn_2": "Загрузить",
                "loading_plugin": "Загрузка плагина: {name}",
                "plugin_error": "Ошибка плагина {name}: {error}",
                "plugin_not_found": "Плагин {name} не найден",
                "plugin_loaded": "Плагин {name} успешно загружен",
                "plugin_failed": "Не удалось загрузить плагин {name}",
                "install_plugins": "Установить плагины",
                "history_tab": "История",
                "history_single_tab": "Одиночная",
                "history_batch_tab": "Пакетная",
                "select_history_task": "Выберите разделение",
                "refresh_btn": "Обновить"
            },
            "en": {
                "title": "Separate music and vocals",
                "starting": "Starting separation with model",
                "bitrate": "Bitrate",
                "selected_stems": "Selected stems",
                "batch_base_dir": "List files",
                "batch_progress_files": "File",
                "batch_separation": "Batch processing",
                "upload_label": "Input audio",
                "url_label": "Enter URL",
                "path_label": "Enter path to audio",
                "url_placeholder": "Link to audio file",
                "path_placeholder": "/path/to/audio",
                "url_btn": "Input URL",
                "path_btn": "Input audio path",
                "upload_cookie": "Upload cookies",
                "download_audio_btn": "Download",
                "upload_btn": "Upload audio",
                "not_input_file": "No input audio is specified",
                "file_not_exists": "The specified audio does not exists",
                "model_type": "Model type",
                "model_id": "ID",
                "model_name": "Model name",
                "model_not_exist": "This model does not exist",
                "output_format": "Output format",
                "not_supported_format": "Specified format unsupported",
                "separate": "Separate",
                "error_no_input": "Error: No input audio.",
                "error_no_model": "Error: No model selected.",
                "error_no_models": "No models found matching the filter stems.",
                "error_unsupported_model_type": "Unsupported model type",
                "error_invalid_format": "Error: Invalid output format.",
                "output_zip": "Download ZIP",
                "inference_tab": "Inference",
                "results": "Results",
                "extract_instrumental": "Extract Instrumental",
                "stems_list": "Stems List",
                "target_instrument": "Target instrument",
                "stems": "Stems",
                "stems_info": "Stem selection unavailable\nEnable \"Extract Instrumental\" to extract the second stem",
                "stems_info2": "To extract the residual (with selected_stems), enable \"Extract Instrumental\"",
                "add_settings": "Additional settings",
                "template": "Name format",
                "vr_aggr_slider": "Aggressive",
                "template_help": """
> The format for naming results in multi-inference.

> Available keys for stem name formatting:
> (you can change the stem name format here)
> * **NAME** - Input file name  
> * **STEM** - Stem name (e.g., vocals, drums, bass)  
> * **MODEL** - Model name (e.g., Mel-Band-Roformer_Instrumental_FvX_gabox, UVR-MDX-NET-Inst_HQ_3)  
> * **ID** - Model ID (e.g., BS-Roformer_SW: 217)

> Example:  
> * **Template:** NAME_STEM_MODEL  
> * **Result:** test_vocals_Mel-Band-Roformer_Instrumental_FvX_gabox  

<div style="color: red; font-weight: bold; background-color: #ffecec; padding: 10px; border-left: 3px solid red; margin: 10px 0;">

Use ONLY the specified keys (NAME, STEM, MODEL) to avoid file corruption.  

DO NOT add extra text or symbols outside these keys, or do so with caution.  

</div>
                """,
                "plugin_manager": "Plugin Manager",
                "upload_label_2": "Upload Plugin",
                "upload_btn_2": "Upload",
                "loading_plugin": "Loading plugin: {name}",
                "plugin_error": "Plugin {name} error: {error}",
                "plugin_not_found": "Plugin {name} not found",
                "plugin_loaded": "Plugin {name} loaded successfully",
                "plugin_failed": "Failed to load plugin {name}",
                "install_plugins": "Install Plugins",
                "history_tab": "History",
                "history_single_tab": "Single",
                "history_batch_tab": "Batch",
                "select_history_task": "Select a separation",
                "refresh_btn": "Refresh"
            }
        }
        self.I18N_STEMS = {
            "ru": {
                "vocals": "Вокал",
                "Vocals": "Вокал",
                "Lead": "Ведущий вокал",
                "lead": "Ведущий вокал",
                "Back": "Бэк-вокал",
                "back": "Бэк-вокал",
                "other": "Другое",
                "Other": "Другое",
                "Instrumental": "Инструментал",
                "instrumnetal": "Инструментал",
                "instrumental +": "Инструментал +",
                "instrumental -": "Инструментал -",
                "Bleed": "Фон",
                "Guitar": "Гитара",
                "drums": "Барабаны",
                "bass": "Бас",
                "karaoke": "Караоке",
                "reverb": "Реверберация",
                "noreverb": "Без реверберации",
                "aspiration": "Придыхание",
                "dry": "Сухой звук",
                "crowd": "Звуки толпы",
                "percussions": "Перкуссия",
                "piano": "Пианино",
                "guitar": "Гитара",
                "male": "Мужской",
                "female": "Женский",
                "kick": "Кик",
                "snare": "Малый барабан",
                "toms": "Том-томы",
                "hh": "Хай-хэт",
                "ride": "Райд",
                "crash": "Крэш",
                "similarity": "Фантомный центр",
                "difference": "Стерео база",
                "inst": "Инструмент",
                "orch": "Оркестр",
                "No Woodwinds": "Без деревянных духовых",
                "Woodwinds": "Деревянные духовые",
                "No Echo": "Без эха",
                "Echo": "Эхо",
                "No Reverb": "Без реверберации",
                "Reverb": "Реверберация",
                "Noise": "Шум",
                "No Noise": "Без шума",
                "Dry": "Сухой звук",
                "No Dry": "Не сухой звук",
                "Breath": "Дыхание",
                "No Breath": "Без дыхания",
                "No Crowd": "Без звуков толпы",
                "Crowd": "Звуки толпы",
                "No Other": "Без другого",
                "Bass": "Бас",
                "No Bass": "Без баса",
                "Drums": "Барабаны",
                "No Drums": "Без барабанов",
                "speech": "Речь",
                "music": "Музыка",
                "effects": "Эффекты",
                "sfx": "Звуковые эффекты",
                "inverted +": "Инверсия +",
                "inverted -": "Инверсия -",
                "Unsupported model type": "Данный тип модели не поддерживается",
                "Error": "Ошибка",
                "Model not exist": "Данной модели не существует",
                "Input file not exist": "Указанного файла не существует",
                "Input path is none": "Не указан путь к файлу"
            },
            "en": {
                "vocals": "Vocals",
                "Vocals": "Vocals",
                "Lead": "Lead vocals",
                "lead": "Lead vocals",
                "Back": "Back vocals",
                "back": "Back vocals",
                "other": "Other",
                "Other": "Other",
                "Instrumental": "Instrumental",
                "instrumnetal": "Instrumental",
                "instrumental +": "Instrumental +",
                "instrumental -": "Instrumental -",
                "Bleed": "Bleed",
                "Guitar": "Guitar",
                "drums": "Drums",
                "bass": "Bass",
                "karaoke": "Karaoke",
                "reverb": "Reverb",
                "noreverb": "No reverb",
                "aspiration": "Aspiration",
                "dry": "Dry",
                "crowd": "Crowd",
                "percussions": "Percussions",
                "piano": "Piano",
                "guitar": "Guitar",
                "male": "Male",
                "female": "Female",
                "kick": "Kick",
                "snare": "Snare",
                "toms": "Toms",
                "hh": "Hi-hat",
                "ride": "Ride",
                "crash": "Crash",
                "similarity": "Pnahtom center",
                "difference": "Stereo base",
                "inst": "Instrument",
                "orch": "Orchestra",
                "No Woodwinds": "No woodwinds",
                "Woodwinds": "Woodwinds",
                "No Echo": "No echo",
                "Echo": "Echo",
                "No Reverb": "No reverb",
                "Reverb": "Reverb",
                "Noise": "Noise",
                "No Noise": "No noise",
                "Dry": "Dry",
                "No Dry": "No dry",
                "Breath": "Breath",
                "No Breath": "No breath",
                "No Crowd": "No crowd",
                "Crowd": "Crowd",
                "No Other": "No other",
                "Bass": "Bass",
                "No Bass": "No bass",
                "Drums": "Drums",
                "No Drums": "No drums",
                "speech": "Speech",
                "music": "Music",
                "effects": "Effects",
                "sfx": "SFX",
                "inverted +": "Inverted +",
                "inverted -": "Inverted -",
                "Unsupported model type": "Current model type unsupported",
                "Error": "Error",
                "Model not exist": "Current model is not exist",
                "Input file not exist": "Current file is not exist",
                "Input path is none": "File path not specified"
            }
        }



        class I18N_Helper:
            def __init__(self, I18N_DATA=self.I18N_DATA, I18N_STEMS=self.I18N_STEMS, lang=self.lang):
                self.I18N_DATA = I18N_DATA
                self.I18N_STEMS = I18N_STEMS
                self.language = lang

            def set_lang(self, lang):
                """Функция для установки текущего языка"""
                if lang in self.I18N_DATA:
                    self.language = lang
                else:
                    raise ValueError(f"Unsupported language: {lang}")

            def t(self, key, **kwargs):
                """Функция для получения перевода с подстановкой значений"""
                lang = self.language
                translation = self.I18N_DATA.get(lang, {}).get(key, key)
                return translation.format(**kwargs) if kwargs else translation

            def t_stem(self, key, **kwargs):
                """Функция для получения перевода с подстановкой значений"""
                lang = self.language
                translation = self.I18N_STEMS.get(lang, {}).get(key, key)
                return translation.format(**kwargs) if kwargs else translation
            
        self.I18N_helper = I18N_Helper()

        class Downloader:
            def __init__(self, output_dir=self.download_dir):
                self.output_dir = output_dir
                os.makedirs(self.output_dir, exist_ok=True)

            def dw_yt_dlp(self, url, cookie=None, output_format="mp3", output_bitrate="320", title=None):
                # Подготовка шаблона имени файла
                outtmpl = "%(title)s.%(ext)s" if title is None else f"{title}.%(ext)s"
                
                ydl_opts = {
                    "format": "bestaudio/best",
                    "outtmpl": os.path.join(self.output_dir, outtmpl),
                    "postprocessors": [
                        {
                            "key": "FFmpegExtractAudio",
                            "preferredcodec": output_format,
                            "preferredquality": output_bitrate,
                        }
                    ],
                    "noplaylist": True,  # Скачивать только одно видео, не плейлист
                    "quiet": True,  # Отключить вывод в консоль
                    "no_warnings": True,  # Скрыть предупреждения
                }

                # Добавляем cookies если указаны
                if cookie and os.path.exists(cookie):
                    ydl_opts["cookiefile"] = cookie

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    try:
                        info = ydl.extract_info(url, download=True)
                        if "_type" in info and info["_type"] == "playlist":
                            # Для плейлистов берем первое видео
                            entry = info["entries"][0]
                            filename = ydl.prepare_filename(entry)
                        else:
                            # Для одиночного видео
                            filename = ydl.prepare_filename(info)

                        # Заменяем оригинальное расширение на выбранный формат
                        base, _ = os.path.splitext(filename)
                        audio_file = base + f".{output_format}"

                        return os.path.abspath(audio_file)
                    except Exception as e:
                        print(e)
                        gr.Warning(e)
                        return url
                    
            def dw_from_url(self, url, title=None):
                try:
                    response = urllib.request.urlopen(url)
                    content_type = response.info().get_content_type()
                    if "audio" in content_type:
                        filename = os.path.join(self.output_dir, title or "downloaded_audio")
                        with open(filename, 'wb') as f:
                            f.write(response.read())
                        return filename
                    else:
                        raise ValueError("URL does not point to an audio file.")
                except Exception as e:
                    print(e)
                    gr.Warning(e)
                    return url

        self.downloader_audio = Downloader()

        class ModelManager:
            def __init__(self, models_info_path=self.models_info_path, I18N_helper=self.I18N_helper, cache_dir=self.models_cache_dir):
                self.models_cache_dir = cache_dir
                self.models_info_path = models_info_path
                self.I18N_helper = I18N_helper
                with open(self.models_info_path, "r", encoding="utf-8") as f:
                    models_info = json.load(f)
                self.models_info = models_info

            def get_mt(self):
                return list(self.models_info.keys())
            
            def get_mn(self, model_type):
                return list(self.models_info[model_type].keys())
            
            def get_stems(self, model_type, model_name):
                stems = self.models_info[model_type][model_name]["stems"]
                return stems

            def get_id(self, model_type, model_name):
                id = self.models_info[model_type][model_name]["id"]
                return id

            def get_tgt_inst(self, model_type, model_name):
                target_instrument = self.models_info[model_type][model_name]["target_instrument"]
                return target_instrument

            def display_models_info(self, filter: str = None):
                try:
                    from tabulate import tabulate
                except:
                    os.system("pip install tabulate")
                    from tabulate import tabulate
                # Собираем данные для таблицы
                table_data = []
                headers = [self.I18N_helper.t("model_type"), self.I18N_helper.t("model_id"), self.I18N_helper.t("model_name"), self.I18N_helper.t("stems"), self.I18N_helper.t("target_instrument")]
                
                for model_type, models in self.models_info.items():
                    for model_name, model_info in models.items():
                        stems_list = model_info.get('stems', [])
                        id = model_info.get('id', "No ID")
                        # Применяем фильтр (регистронезависимо)
                        if filter:
                            filter_lower = filter.lower()
                            if not any(filter_lower == s.lower() for s in stems_list):
                                continue
                        
                        # Подготавливаем данные для строки таблицы
                        row = [
                            model_type,
                            id,
                            model_name,
                            ", ".join(stems_list) or "N/A",
                            model_info.get('target_instrument', "N/A")
                        ]
                        table_data.append(row)
                
                # Выводим результат
                if table_data:
                    print(tabulate(table_data, headers=headers, tablefmt="grid"))
                else:
                    print(self.I18N_helper.t("error_no_models"))

            def download_file(self, url_model, local_path):
                class TqdmUpTo(tqdm):
                    def update_to(self, b=1, bsize=1, tsize=None):
                        if tsize is not None:
                            self.total = tsize
                        self.update(b * bsize - self.n)
                with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                            desc=os.path.basename(local_path)) as t:
                    urllib.request.urlretrieve(url_model, local_path, reporthook=t.update_to)

            def download_model(self, model_paths, model_name, model_type, ckpt_url, conf_url):
                model_dir = os.path.join(model_paths, model_type)
                os.makedirs(model_dir, exist_ok=True)

                config_path = None
                checkpoint_path = None

                if model_type == "mel_band_roformer":
                    config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
                    checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")

                elif model_type == "vr":
                    config_path = os.path.join(model_dir, f"{model_name}.json")
                    checkpoint_path = os.path.join(model_dir, f"{model_name}.pth")
            
                elif model_type == "bs_roformer":
                    config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
                    checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")
                
                elif model_type == "mdx23c":
                    config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
                    checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")
                
                elif model_type == "scnet":
                    config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
                    checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")

                elif model_type == "bandit":
                    config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
                    checkpoint_path = os.path.join(model_dir, f"{model_name}.chpt")

                elif model_type == "bandit_v2":
                    config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
                    checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")
                
                elif model_type == "htdemucs":
                    config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
                    checkpoint_path = os.path.join(model_dir, f"{model_name}.th")
                
                else:
                    raise ValueError(f"{self.I18N_helper.t('error_unsupported_model_type')}: {model_type}")

                # Проверяем, что пути заданы (на всякий случай)
                if config_path is None or checkpoint_path is None:
                    raise RuntimeError()

                # Если файлы уже есть — пропускаем загрузку
                if os.path.exists(checkpoint_path) and os.path.exists(config_path):
                    pass
                else:
                    for local_path, url_model in [(checkpoint_path, ckpt_url), (config_path, conf_url)]:
                        if not os.path.exists(local_path):

                            self.download_file(url_model, local_path)

                return config_path, checkpoint_path

            def conf_editor(self, config_path):

                class IndentDumper(yaml.Dumper):
                    def increase_indent(self, flow=False, indentless=False):
                        return super(IndentDumper, self).increase_indent(flow, False)


                def tuple_constructor(loader, node):
                    # Load the sequence of values from the YAML node
                    values = loader.construct_sequence(node)
                    # Return a tuple constructed from the sequence
                    return tuple(values)

                # Register the constructor with PyYAML  
                yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple',
            tuple_constructor)



                def conf_edit(config_path):
                    with open(config_path, 'r') as f:
                        data = yaml.load(f, Loader=yaml.SafeLoader)

                    # handle cases where 'use_amp' is missing from config:
                    if 'use_amp' not in data.keys():
                        data['training']['use_amp'] = True

                    if data['inference']['num_overlap'] != 2:
                        data['inference']['num_overlap'] = 2

                    if data['inference']['batch_size'] == 1:
                        data['inference']['batch_size'] = 2

                    print("Using custom overlap and chunk_size values:")
                    print(f"batch_size = {data['inference']['batch_size']}")


                    with open(config_path, 'w') as f:
                        yaml.dump(data, f, default_flow_style=False, sort_keys=False, Dumper=IndentDumper, allow_unicode=True)


        class History:
            def __init__(self, history_path=self.history_file_path):
                self.history = {"single": {}, "batch": {}}
                self.history_path = history_path

            def write_history(self):
                with open(self.history_path, "w") as f:
                    json.dump(self.history, f, indent=4)

            def load_history(self):
                with open(self.history_path, "r") as f:
                    return json.load(f)
                
            def add_history(self, type: Literal["single", "batch"] = "single", results: dict | list = []):
                dt = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.history[type][dt] = results
                self.write_history()

            def del_history(self, type: Literal["single", "batch"] = "single", datetime: str = ""):
                del self.history[type][datetime]
                self.write_history()

            def parse_history_list(self, type: Literal["single", "batch"] = "single"):
                sorted_limit = []
                list_separations = sorted(list(self.history[type].keys()), reverse=True)

                return list_separations

            def parse_history_results(self, type: Literal["single", "batch"] = "single", datetime: str = ""):
                return self.history[type][datetime]

            def check_and_load(self):
                if os.path.exists(self.history_path):
                    self.history = self.load_history()
                else:
                    self.write_history()
                    
        self.history_manager = History(history_path=self.history_file_path)
        self.history_manager.check_and_load()
        self.model_manager = ModelManager(models_info_path=self.models_info_path)

        class Separator:
            def __init__(
                self, 
                models_cache_dir=self.models_cache_dir, 
                model_manager=self.model_manager, 
                I18N_helper=self.I18N_helper, 
                history_manager=self.history_manager,
                formats=self.output_formats,
                model_types=self.model_types,
                p_path=self.python_path
            ):
                self.models_cache_dir = models_cache_dir
                self.model_manager = model_manager
                self.I18N_helper = I18N_helper
                self.history_manager = history_manager
                self.output_formats = formats
                self.model_types = model_types
                self.python_path = p_path

            def base(
                self,
                input_file: str = None,
                output_dir: str = None,
                model_type: Literal["mel_band_roformer", "bs_roformer", "mdx23c", "scnet", "htdemucs", "bandit", "bandit_v2", "vr", "mdx"] = "mel_band_roformer",
                model_name: str = "Mel-Band-Roformer_Vocals_kimberley_jensen",
                ext_inst: bool = False,
                vr_aggr: int = 5,
                output_format: Literal["mp3", "wav", "flac", "ogg", "opus", "m4a", "aac", "aiff"] = "mp3",
                output_bitrate: str = "320k",
                template: str = "NAME_(STEM)_MODEL",
                call_method: str = "cli",
                selected_stems: list = None
            ):

                if output_format not in self.output_formats or not output_format:
                    print(self.I18N_helper.t("not_supported_format"))
                    output_format = "flac"
                    
                if output_dir is None:
                    output_dir = os.getcwd()

                if output_dir:
                    output_dir = os.path.abspath(output_dir)
                
                if selected_stems is None:
                    selected_stems = []
                    
                if not input_file:
                    print(self.I18N_helper.t("not_input_file"))
                    return [("Input path is none", "/none/none.mp3")]
            
                if not os.path.exists(input_file):
                    print(self.I18N_helper.t("file_not_exists"))
                    return [("Input file not exist", "/none/none.mp3")]
            
                if "STEM" not in template and template is not None:
                    template = template + "_STEM_"
                if not template:
                    template = "mvsepless_NAME_(STEM)"
            
                print(f"{self.I18N_helper.t('starting')}: {model_type}/{model_name}, {self.I18N_helper.t('bitrate')}={output_bitrate}, {self.I18N_helper.t('selected_stems')}={selected_stems}")
                os.makedirs(output_dir, exist_ok=True)
            
                if model_type in ["mel_band_roformer", "bs_roformer", "mdx23c", "scnet", "htdemucs", "bandit", "bandit_v2"]:
                    try:
                        info = self.model_manager.models_info[model_type][model_name]
                    except KeyError:
                        print(self.I18N_helper.t("model_not_exist"))
                        return [("Model not exist", "/none/none.mp3")]
                    
                    id = self.model_manager.get_id(model_type, model_name)
                    conf, ckpt = self.model_manager.download_model(self.models_cache_dir, model_name, model_type, 
                                            info["checkpoint_url"], info["config_url"])
                    if model_type != "htdemucs":
                        self.model_manager.conf_editor(conf)
            
                    if call_method == "cli":
                        cmd = [self.python_path, "-m", "separator.msst_separator", f'--input "{input_file}"', 
                            f'--store_dir "{output_dir}"', f'--model_type "{model_type}"', 
                            f'--model_name "{model_name}"', f'--model_id {id}', f'--config_path "{conf}"', 
                            f'--start_check_point "{ckpt}"', f'--output_format "{output_format}"', 
                            f'--output_bitrate "{output_bitrate}"', f'--template "{template}"', 
                            "--save_results_info"]
                        if ext_inst:
                            cmd.append("--extract_instrumental")
                        if selected_stems:
                            instruments = " ".join(f'"{s}"' for s in selected_stems)
                            cmd.append(f'--selected_instruments {instruments}')
                        try:
                            subprocess.run(" ".join(cmd), shell=True, check=True)
                        except Exception as e:
                            print(e)
                            return [("Error", "/none/none.mp3")]
                            
                        results_path = os.path.join(output_dir, "results.json")
                        if os.path.exists(results_path):
                            with open(results_path, encoding="utf-8") as f:
                                return json.load(f)
                        return [("Error", "/none/none.mp3")]
            
                    elif call_method == "direct":
                        from separator.msst_separator import mvsep_offline
                        try:
                            return mvsep_offline(
                                input_path=input_file, store_dir=output_dir, model_type=model_type, 
                                config_path=conf, start_check_point=ckpt, extract_instrumental=ext_inst, 
                                output_format=output_format, output_bitrate=output_bitrate, 
                                model_name=model_name, template=template, selected_instruments=selected_stems, model_id=id
                            )
                        except Exception as e:
                            print(e)
                            return [("Error", "/none/none.mp3")]
            
                elif model_type in ["vr", "mdx"]:
                    try:
                        info = self.model_manager.models_info[model_type][model_name]
                    except KeyError:
                        print(self.I18N_helper.t("model_not_exist"))
                        return [("Model not exist", "/none/none.mp3")]
                    
                    id = self.model_manager.get_id(model_type, model_name)
                    if model_type == "vr" and info.get("custom_vr", False):
                        conf, ckpt = self.model_manager.download_model(self.models_cache_dir, model_name, model_type, 
                                                info["checkpoint_url"], info["config_url"])
                        primary_stem = info["primary_stem"]
            
                        if call_method == "cli":
                            cmd = [self.python_path, "-m", "separator.uvr_sep", "custom_vr", 
                                f'--input_file "{input_file}"', f'--ckpt_path "{ckpt}"', 
                                f'--config_path "{conf}"', f'--bitrate "{output_bitrate}"', 
                                f'--model_name "{model_name}"', f'--model_id {id}', f'--template "{template}"', 
                                f'--output_format "{output_format}"', f'--primary_stem "{primary_stem}"', 
                                f'--aggression {vr_aggr}', f'--output_dir "{output_dir}"']
                            if selected_stems:
                                instruments = " ".join(f'"{s}"' for s in selected_stems)
                                cmd.append(f'--selected_instruments {instruments}')
                            try:
                                subprocess.run(" ".join(cmd), shell=True, check=True)
                            except Exception as e:
                                print(e)
                                return [("Error", "/none/none.mp3")]
                            results_path = os.path.join(output_dir, "results.json")
                            if os.path.exists(results_path):
                                with open(results_path, encoding="utf-8") as f:
                                    return json.load(f)
                            return [("Error", "/none/none.mp3")]
            
                        elif call_method == "direct":
                            from separator.uvr_sep import custom_vr_separate
                            try:
                                return custom_vr_separate(
                                    input_file=input_file, ckpt_path=ckpt, config_path=conf, 
                                    bitrate=output_bitrate, model_name=model_name, template=template, 
                                    output_format=output_format, primary_stem=primary_stem, 
                                    aggression=vr_aggr, output_dir=output_dir, 
                                    selected_instruments=selected_stems, model_id=id
                                )
                            except Exception as e:
                                print(e)
                                return [("Error", "/none/none.mp3")]
                    else:
                        if call_method == "cli":
                            cmd = [self.python_path, "-m", "separator.uvr_sep", "uvr", 
                                f'--input_file "{input_file}"', f'--output_dir "{output_dir}"', 
                                f'--template "{template}"', f'--bitrate "{output_bitrate}"', 
                                f'--model_dir "{self.models_cache_dir}"', f'--model_type "{model_type}"', 
                                f'--model_name "{model_name}"', f'--model_id {id}', f'--output_format "{output_format}"', 
                                f'--aggression {vr_aggr}']
                            if selected_stems:
                                instruments = " ".join(f'"{s}"' for s in selected_stems)
                                cmd.append(f'--selected_instruments {instruments}')
                            try:
                                subprocess.run(" ".join(cmd), shell=True, check=True)
                            except Exception as e:
                                print(e)
                                return [("Error", "/none/none.mp3")]
            
                            results_path = os.path.join(output_dir, "results.json")
                            if os.path.exists(results_path):
                                with open(results_path, encoding="utf-8") as f:
                                    return json.load(f)
                            return [("Error", "/none/none.mp3")]
            
                        elif call_method == "direct":
                            from separator.uvr_sep import non_custom_uvr_inference
                            try:
                                return non_custom_uvr_inference(
                                    input_file=input_file, output_dir=output_dir, template=template, 
                                    bitrate=output_bitrate, model_dir=self.models_cache_dir, 
                                    model_type=model_type, model_name=model_name, 
                                    output_format=output_format, aggression=vr_aggr, 
                                    selected_instruments=selected_stems, model_id=id
                                )
                            except Exception as e:
                                print(e)
                                return [("Error", "/none/none.mp3")]
            
                print(self.I18N_helper.t("error_unsupported_model_type"))
                return [("Unsupported model type", "/none/none.mp3")]

            def id_base(
                self,
                input_file: str = None,
                output_dir: str = None,
                id: int = 217,
                ext_inst: bool = False,
                vr_aggr: int = 5,
                output_format: str = "wav",
                output_bitrate: str = "320k",
                template: str = "NAME_(STEM)_MODEL",
                call_method: str = "cli",
                selected_stems: list = None
            ):
                m_type = None
                m_name = None
                for mt in self.model_manager.get_mt():
                    for mn in self.model_manager.get_mn(mt):
                        if self.model_manager.get_id(mt, mn) == id:
                            m_type = mt
                            m_name = mn
                            break

                if m_type and m_name:
                    results = self.base(input_file=input_file, output_dir=output_dir, model_type=m_type, model_name=m_name, ext_inst=ext_inst, vr_aggr=vr_aggr, output_format=output_format, output_bitrate=output_bitrate, template=template, call_method=call_method, selected_stems=selected_stems)
                else:
                    results = self.base(input_file=input_file, output_dir=output_dir, model_type=None, model_name=None, ext_inst=ext_inst, vr_aggr=vr_aggr, output_format=output_format, output_bitrate=output_bitrate, template=template, call_method=call_method, selected_stems=selected_stems)

                return results

            def batch(
                self,
                input_list: list = [],
                output_dir: str = None,
                model_type: Literal["mel_band_roformer", "bs_roformer", "mdx23c", "scnet", "htdemucs", "bandit", "bandit_v2", "vr", "mdx"] = "mel_band_roformer",
                model_name: str = "Mel-Band-Roformer_Vocals_kimberley_jensen",
                id: int = 217,
                model_mode: Literal["name", "id"] = "name",
                ext_inst: bool = False,
                vr_aggr: int = 5,
                output_format: str = "wav",
                output_bitrate: str = "320k",
                template: str = "NAME_(STEM)_MODEL",
                call_method: str = "cli",
                selected_stems: list = None,
                progress: any = gr.Progress()
            ):
                batch = {}
                if input_list:
                    block_count = len(input_list)
                    for i, input_file in enumerate(input_list):
                        progress((i+1) / block_count, desc=f"{i+1}/{block_count}")
                        print(f"{self.I18N_helper.t('batch_progress_files')} {i+1} / {block_count}")
                        base_name = os.path.splitext(os.path.basename(input_file))[0]
                        output_dir_base = os.path.join(output_dir, base_name)
                        if model_mode == "name":
                            results = self.base(input_file=input_file, output_dir=output_dir_base, model_type=model_type, model_name=model_name, ext_inst=ext_inst, vr_aggr=vr_aggr, output_format=output_format, output_bitrate=output_bitrate, template=template, call_method=call_method)
                        elif model_mode == "id":
                            results = self.id_base(input_file=input_file, output_dir=output_dir_base, id=id, ext_inst=ext_inst, vr_aggr=vr_aggr, output_format=output_format, output_bitrate=output_bitrate, template=template, call_method=call_method)
                        batch[base_name] = results

                self.history_manager.add_history(type="batch", results=batch)

                return batch

            def single(
                self,
                input_file: str = None,
                output_dir: str = None,
                model_type: Literal["mel_band_roformer", "bs_roformer", "mdx23c", "scnet", "htdemucs", "bandit", "bandit_v2", "vr", "mdx"] = "mel_band_roformer",
                model_name: str = "Mel-Band-Roformer_Vocals_kimberley_jensen",
                id: int = 217,
                model_mode: Literal["name", "id"] = "name",
                ext_inst: bool = False,
                vr_aggr: int = 5,
                output_format: str = "wav",
                output_bitrate: str = "320k",
                template: str = "NAME_(STEM)_MODEL",
                call_method: str = "cli",
                selected_stems: list = None
            ):
                if model_mode == "name":
                    results = self.base(input_file=input_file, output_dir=output_dir, model_type=model_type, model_name=model_name, ext_inst=ext_inst, vr_aggr=vr_aggr, output_format=output_format, output_bitrate=output_bitrate, template=template, call_method=call_method)
                elif model_mode == "id":
                    results = self.id_base(input_file=input_file, output_dir=output_dir, id=id, ext_inst=ext_inst, vr_aggr=vr_aggr, output_format=output_format, output_bitrate=output_bitrate, template=template, call_method=call_method)

                self.history_manager.add_history(type="single", results=results)

                return results

        self.separator = Separator()

        class GradioHelper:
            def __init__(
                self, 
                I18N_helper=self.I18N_helper, 
                separator=self.separator, 
                output_base_dir=self.output_app_base_dir, 
                history_manager=self.history_manager,
                dw=self.downloader_audio
            ):
                self.BATCH_STATE = {}
                self.I18N_helper = I18N_helper
                self.history_manager = history_manager
                self.separator = separator
                self.output_base_dir = output_base_dir
                self.downloader_audio = dw

            def download_wrapper(self, url, cookie):
                t = self.downloader_audio.dw_yt_dlp(url, cookie)
                return (
                    gr.update(value=t),
                    gr.update(value=t),
                    gr.update(visible=True),
                    gr.update(visible=False),
                )

            def download_wrapper_batch(self, url, cookie, input_files: str = ""):
                input_files = ast.literal_eval(input_files)
                t = self.downloader_audio.dw_yt_dlp(url, cookie)
                input_files.append(t)
                return (
                    gr.update(value=input_files),
                    gr.update(value=str(input_files)),
                    gr.update(visible=True),
                    gr.update(visible=False),
                )

            def create_zip(self, output_audio, output_dir=".", zip_name="output.zip"):
                zip_path = os.path.join(output_dir, zip_name)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)

                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for stem_name, audio_file in output_audio:
                        ext = os.path.splitext(audio_file)[1]
                        arcname = f"{stem_name}{ext}"
                        if os.path.exists(audio_file):
                            zf.write(audio_file, arcname=arcname)

                return zip_path

            def run_inference(
                self,
                input_file: str = None,
                model_type: str = "mel_band_roformer",
                model_name: str = "Mel-Band-Roformer_Vocals_kimberley_jensen",
                ext_inst: bool = False,
                vr_aggr: int = 5,
                output_format: str = "wav",
                template: str = "NAME_(STEM)_MODEL",
                selected_stems: list = [],
            ):
                """Функция для запуска инференса"""

                temp_dir = os.path.join(
                    self.output_base_dir, datetime.now().strftime("%Y%m%d_%H%M%S")
                )

                output_audio = self.separator.single(
                    input_file=input_file,
                    output_dir=temp_dir,
                    model_type=model_type,
                    model_name=model_name,
                    model_mode="name",
                    output_format=output_format,
                    ext_inst=ext_inst,
                    call_method="cli",
                    vr_aggr=vr_aggr,
                    template=template,
                    selected_stems=selected_stems,
                )

                output_audio_2 = []

                for stem, path in output_audio:
                    output_audio_2.append((f"{os.path.splitext(os.path.basename(str(input_file)))[0]} - {stem}", path))

                audio_updates = [
                    gr.update(
                        label=self.I18N_helper.t_stem(output_audio[i][0]) if i < len(output_audio) else "",
                        value=output_audio[i][1] if i < len(output_audio) else None,
                        visible=i < len(output_audio),
                    )
                    for i in range(64)
                ]
                zip_update = gr.update(
                    value=self.create_zip(
                        output_audio_2,
                        output_dir=temp_dir,
                        zip_name=f'mvsepless_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip',
                    ),
                    visible=True,
                )
                return audio_updates + [zip_update]

            def run_inference_batch(
                self,
                input_file: str = None,
                model_type: str = "mel_band_roformer",
                model_name: str = "Mel-Band-Roformer_Vocals_kimberley_jensen",
                ext_inst: bool = False,
                vr_aggr: int = 5,
                output_format: str = "wav",
                template: str = "NAME_(STEM)_MODEL",
                selected_stems: list = [],
            ):
                """Функция для запуска инференса"""

                self.BATCH_STATE = {}

                temp_dir = os.path.join(
                    self.output_base_dir, datetime.now().strftime("%Y%m%d_%H%M%S")
                )

                try:
                    input_list = ast.literal_eval(input_file)
                except:
                    input_list = []

                output_audio = self.separator.batch(
                    input_list=input_list,
                    output_dir=temp_dir,
                    model_type=model_type,
                    model_name=model_name,
                    model_mode="name",
                    output_format=output_format,
                    ext_inst=ext_inst,
                    call_method="cli",
                    vr_aggr=vr_aggr,
                    template=template,
                    selected_stems=selected_stems,
                )

                self.BATCH_STATE = output_audio

                list_dirs = []
                output_audio_2 = []

                for dir in list(self.BATCH_STATE.keys()):
                    list_dirs.append(dir)
                    for stem, path in self.BATCH_STATE[dir]:
                        output_audio_2.append((f"{dir} - {stem}", path))
                    
                zip_update = gr.update(
                    value=self.create_zip(
                        output_audio_2,
                        output_dir=temp_dir,
                        zip_name=f'mvsepless_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip',
                    ),
                    visible=True,
                )
                return gr.update(choices=list_dirs, value=None, visible=True), zip_update

            def batch_players(self, dir):
                list_dirs = []
                if self.BATCH_STATE:
                    for dir_2 in list(self.BATCH_STATE.keys()):
                        list_dirs.append(dir_2)
                if dir in list_dirs:
                    output_audio = self.BATCH_STATE[dir]
                else:
                    output_audio = None
                if output_audio:
                    audio_updates = [
                        gr.update(
                            label=self.I18N_helper.t_stem(output_audio[i][0]) if i < len(output_audio) else "",
                            value=output_audio[i][1] if i < len(output_audio) else None,
                           	visible=i < len(output_audio),
                		   )
                		   for i in range(64)
                	   ]
                else:
                    audio_updates = [
                        gr.update(
                            label=None,
                            value=None,
                           	visible=False,
                		    )
                		    for i in range(64)
                	  ]
                return audio_updates

            def history_send_player(self, type, dt, name=None, player_batch=False):

                try:
                    history_results = self.history_manager.parse_history_results(type=type, datetime=dt)
                except KeyError:
                    if type == "single":
                        history_results = None
                    elif type == "batch":
                        history_results =  None
                if type == "single":

                    if history_results:

                        audio_updates = [
                            gr.update(
                                label=self.I18N_helper.t_stem(history_results[i][0]) if i < len(history_results) else "",
                                value=history_results[i][1] if i < len(history_results) else None,
                                visible=i < len(history_results),
                            )
                            for i in range(64)
                        ]

                    elif not history_results:

                        audio_updates = [
                            gr.update(
                                label=None,
                                value=None,
                                visible=False,
                            )
                            for i in range(64)
                        ]

                    return audio_updates
                
                if type == "batch":

                    if player_batch == True:

                        try:
                            out_audio = history_results[name]
                        except:
                            out_audio = None

                        if out_audio:
                            audio_updates = [
                                gr.update(
                                    label=self.I18N_helper.t_stem(out_audio[i][0]) if i < len(out_audio) else "",
                                    value=out_audio[i][1] if i < len(out_audio) else None,
                                    visible=i < len(out_audio),
                                )
                                for i in range(64)
                            ]

                        elif not out_audio:

                            audio_updates = [
                                gr.update(
                                    label=None,
                                    value=None,
                                    visible=False,
                                )
                                for i in range(64)
                            ]

                        return audio_updates
                    else:
                        try:
                            list_dirs = list(history_results.keys())
                        except:
                            list_dirs = []
                        return gr.update(choices=list_dirs, value=None)

            def clear_players(self):
                return [gr.update(choices=[], visible=False, value=None)] + [gr.update(value=None, visible=False)] + [gr.update(visible=False, value=None, label=None) for i in range(64)]

            def history_player(self, type, dt, name=None):

                try:
                    history_results = self.history_manager.parse_history_results(type=type, datetime=dt)
                except KeyError:
                    if type == "single":
                        history_results = [(None, None)]
                    elif type == "multiple":
                        history_results =  {None: [(None, None)]}
                if type == "single":

                    audio_updates = [
                        gr.update(
                            label=self.I18N_helper.t_stem(history_results[i][0]) if i < len(history_results) else "",
                            value=history_results[i][1] if i < len(history_results) else None,
                            visible=i < len(history_results),
                        )
                        for i in range(64)
                    ]

                    return audio_updates
                
                if type == "multiple":

                    if name != "" or name is not None:

                        try:
                            out_audio = history_results[name]
                        except KeyError:
                            out_audio = [(None, None)]

                        audio_updates = [
                            gr.update(
                                label=self.I18N_helper.t_stem(out_audio[i][0]) if i < len(out_audio) else "",
                                value=out_audio[i][1] if i < len(out_audio) else None,
                                visible=i < len(out_audio),
                            )
                            for i in range(64)
                        ]

                        return audio_updates
                    else:
                        list_dirs = list(history_results.keys())
                        return gr.update(choices=list_dirs, value=None)

        self.gradio_helper = GradioHelper()

        class GradioPluginHelper:
            def __init__(self, plugins_dir=self.plugins_dir, I18N_helper=self.I18N_helper):
                self.plugins_dir = plugins_dir
                self.I18N_helper = I18N_helper
            def restart(self):
                python = sys.executable
                subprocess.Popen([python] + sys.argv)
                os._exit(0)

            def upload_plugin_list(self, files):
                if not files:
                    return
                for file in files:
                    try:
                        shutil.copy(file, os.path.join(self.plugins_dir, os.path.basename(file)))
                    except Exception as e:
                        print(f"Error copying plugin: {e}")
                time.sleep(2)
                self.restart()

            def load_plugins(self):
                plugins = []
                if os.path.exists(self.plugins_dir) and os.path.isdir(self.plugins_dir):
                    try:
                        for filename in os.listdir(self.plugins_dir):
                            if filename.endswith(".py") and filename != "__init__.py":
                                file_path = os.path.join(self.plugins_dir, filename)
                                module_name = filename[:-3]
                                spec = importlib.util.spec_from_file_location(
                                    module_name, file_path
                                )
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
                                    plugin_name = (
                                        name_func()
                                        if name_func is not None
                                        else module_name
                                    )
                                    plugins.append((plugin_name, plugin_func))

                    except Exception as e:
                        print(e)
                return plugins

            def load_plugin_ui(self, lang="ru"):
                plugins = self.load_plugins()
                with gr.Tab(self.I18N_helper.t("plugin_manager")):

                    with gr.Tab(self.I18N_helper.t("install_plugins")):
                        with gr.Blocks():
                            upload_plugin_files = gr.Files(
                                label=self.I18N_helper.t("upload_label_2"), file_types=[".py"], interactive=True
                            )
                            upload_btn = gr.Button(self.I18N_helper.t("upload_btn_2"), interactive=True)
                            upload_btn.click(fn=self.upload_plugin_list, inputs=upload_plugin_files)
                    if plugins:
                        for name, func in plugins:
                            try:
                                print(self.I18N_helper.t("loading_plugin", name=name))
                                with gr.Tab(name):
                                    func(lang)
                            except Exception as e:
                                print(self.I18N_helper.t("plugin_error", name=name, error=str(e)))
                                pass

        self.gradio_plugin_helper = GradioPluginHelper()

        class GradioApp:
            def __init__(self, 
                I18N_helper=self.I18N_helper, 
                formats=self.output_formats,
                plugin_manager=self.gradio_plugin_helper,
                separator=self.separator,
                ui_helper=self.gradio_helper,
                model_manager=self.model_manager,
                history_manager=self.history_manager,
                plugins=True,
                port=7860, 
                share=False, 
                debug=True,
                inline=False,
                theme: Literal["gamma", "beta", "alpha", "mvsep", "vbachgen"] = "gamma"
            ):
                self.I18N_helper = I18N_helper
                self.formats = formats
                self.port = port
                self.share = share
                self.debug = debug
                self.inline = inline
                self.gradio_plugin_helper = plugin_manager
                self.gradio_helper = ui_helper
                self.separator = separator
                self.history_manager = history_manager
                self.model_manager = model_manager
                self.plugins = plugins
                self.themes = {
                    "gamma": gr.themes.Citrus(
                    primary_hue="teal",
                    secondary_hue="blue",
                    neutral_hue="blue",
                    spacing_size="sm",
                    font=[gr.themes.GoogleFont('Montserrat'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
                    ),

                    "beta": gr.themes.Default(
                    primary_hue="violet",
                    secondary_hue="cyan",
                    neutral_hue="blue",
                    spacing_size="sm",
                    text_size="sm",
                    font=[gr.themes.GoogleFont("Tektur"), "ui-sans-serif", "system-ui", "sans-serif"],
                    ).set(
                    body_text_color="*neutral_950",
                    body_text_color_subdued="*neutral_500",
                    background_fill_primary="*neutral_200",
                    background_fill_primary_dark="*neutral_800",
                    border_color_accent="*primary_950",
                    border_color_accent_dark="*neutral_700",
                    border_color_accent_subdued="*primary_500",
                    border_color_primary="*primary_800",
                    border_color_primary_dark="*neutral_400",
                    color_accent_soft="*primary_100",
                    color_accent_soft_dark="*neutral_800",
                    link_text_color="*secondary_700",
                    link_text_color_active="*secondary_700",
                    link_text_color_hover="*secondary_800",
                    link_text_color_visited="*secondary_600",
                    link_text_color_visited_dark="*secondary_700",
                    block_background_fill="*background_fill_secondary",
                    block_background_fill_dark="*neutral_950",
                    block_label_background_fill="*secondary_400",
                    block_label_text_color="*neutral_800",
                    panel_background_fill="*background_fill_primary",
                    checkbox_background_color="*background_fill_secondary",
                    checkbox_label_background_fill_dark="*neutral_900",
                    input_background_fill_dark="*neutral_900",
                    input_background_fill_focus="*neutral_100",
                    input_background_fill_focus_dark="*neutral_950",
                    button_small_radius="*radius_sm",
                    button_secondary_background_fill="*neutral_400",
                    button_secondary_background_fill_dark="*neutral_500",
                    button_secondary_background_fill_hover_dark="*neutral_950",
                    ),

                    "alpha": gr.themes.Base(
                    primary_hue="blue",
                    secondary_hue="gray",
                    neutral_hue="slate",
                    font=[gr.themes.GoogleFont("Montserrat"), "Arial", "sans-serif"],
                    font_mono=[gr.themes.GoogleFont("Roboto Mono"), "Courier New", "monospace"]
                    ).set(
                    button_primary_background_fill="#3a7bd5",
                    button_primary_background_fill_hover="#2c65c0",
                    button_primary_text_color="#ffffff",
                    input_background_fill="#ffffff",
                    input_border_color="#d0d0d6",
                    block_background_fill="#ffffff",
                    border_color_primary="#d0d0d6"
                    ),

                    "vbachgen": gr.themes.Base(
                    primary_hue="rose",
                    spacing_size="sm",
                    font=[gr.themes.GoogleFont('Tektur')],
                    ),

                    "mvsep": gr.themes.Base( # Тема соответствующая цветовой стилистике MVSep.com
                    primary_hue="blue",
                    secondary_hue="gray",
                    neutral_hue="slate",
                    font=[gr.themes.GoogleFont("Montserrat"), "Arial", "sans-serif"],
                    font_mono=[gr.themes.GoogleFont("Roboto Mono"), "Courier New", "monospace"]
                    ).set(
                    button_primary_background_fill="#3a7bd5",
                    button_primary_background_fill_hover="#2c65c0",
                    button_primary_text_color="#ffffff",
                    input_background_fill="#ffffff",
                    input_border_color="#d0d0d6",
                    block_background_fill="#ffffff",
                    border_color_primary="#d0d0d6"
                    )
                }
                self.theme = theme

            def ui(self):

                with gr.Blocks(theme=self.themes[self.theme], title=self.I18N_helper.t("title")) as lite_app:
                    with gr.Tab(self.I18N_helper.t("inference_tab")):
                        with gr.Row():
                            with gr.Column():
                                with gr.Column() as single_separation:
                                    with gr.Group() as local:
                                        input_audio = gr.Audio(
                                            label=self.I18N_helper.t("upload_label"), type="filepath", interactive=True
                                        )
                                        with gr.Row(equal_height=True):
                                            path_0_btn = gr.Button(self.I18N_helper.t("path_btn"), interactive=True)
                                            url_0_btn = gr.Button(self.I18N_helper.t("url_btn"), interactive=True)
                                    with gr.Group(visible=False) as url:
                                        with gr.Column(variant="compact"):
                                            with gr.Row(equal_height=True):
                                                upload_cookie = gr.UploadButton(
                                                    label=self.I18N_helper.t("upload_cookie"),
                                                    file_types=[".txt"],
                                                    file_count="single",
                                                    scale=1,
                                                    variant="primary",
                                                    interactive=True
                                                )
                                                input_link = gr.Textbox(
                                                    label=self.I18N_helper.t("url_label"),
                                                    placeholder=self.I18N_helper.t("url_placeholder"),
                                                    interactive=True,
                                                    scale=10,
                                                )
                                                download_audio_btn = gr.Button(
                                                    self.I18N_helper.t("download_audio_btn"), scale=1, variant="stop", interactive=True
                                                )
                                        with gr.Row(equal_height=True):
                                            path_1_btn = gr.Button(self.I18N_helper.t("path_btn"), interactive=True)
                                            upload_0_btn = gr.Button(self.I18N_helper.t("upload_btn"), variant="primary",interactive=True)
                                    with gr.Group(visible=False) as path:
                                        input_path = gr.Textbox(
                                            label=self.I18N_helper.t("path_label"),
                                            placeholder=self.I18N_helper.t("path_placeholder"),
                                            interactive=True,
                                        )
                                        with gr.Row(equal_height=True):
                                            upload_1_btn = gr.Button(self.I18N_helper.t("upload_btn"), variant="primary", interactive=True)
                                            url_1_btn = gr.Button(self.I18N_helper.t("url_btn"), interactive=True)

                                with gr.Column(visible=False) as batch_separation:
                                    with gr.Group() as b_local:
                                        b_input_audio = gr.File(
                                            label=self.I18N_helper.t("upload_label"), type="filepath", interactive=True, file_count="multiple"
                                        )
                                        with gr.Row(equal_height=True):
                                            b_path_0_btn = gr.Button(self.I18N_helper.t("path_btn"), interactive=True)
                                            b_url_0_btn = gr.Button(self.I18N_helper.t("url_btn"), interactive=True)
                                    with gr.Group(visible=False) as b_url:
                                        with gr.Column(variant="compact"):
                                            with gr.Row(equal_height=True):
                                                b_upload_cookie = gr.UploadButton(
                                                    label=self.I18N_helper.t("upload_cookie"),
                                                    file_types=[".txt"],
                                                    file_count="single",
                                                    scale=1,
                                                    variant="primary",
                                                    interactive=True
                                                )
                                                b_input_link = gr.Textbox(
                                                    label=self.I18N_helper.t("url_label"),
                                                    placeholder=self.I18N_helper.t("url_placeholder"),
                                                    interactive=True,
                                                    scale=10,
                                                )
                                                b_download_audio_btn = gr.Button(
                                                    self.I18N_helper.t("download_audio_btn"), scale=1, variant="stop", interactive=True
                                                )
                                        with gr.Row(equal_height=True):
                                            b_path_1_btn = gr.Button(self.I18N_helper.t("path_btn"),interactive=True)
                                            b_upload_0_btn = gr.Button(self.I18N_helper.t("upload_btn"), variant="primary",interactive=True)
                                    with gr.Group(visible=False) as b_path:
                                        b_input_path = gr.Textbox(
                                            label=self.I18N_helper.t("path_label"),
                                            placeholder=self.I18N_helper.t("path_placeholder"),
                                            interactive=True,
                                        )
                                        with gr.Row(equal_height=True):
                                            b_upload_1_btn = gr.Button(self.I18N_helper.t("upload_btn"), variant="primary", interactive=True)
                                            b_url_1_btn = gr.Button(self.I18N_helper.t("url_btn"), interactive=True)

                                batch_toggle = gr.Checkbox(label=self.I18N_helper.t("batch_separation"), interactive=True)

                            with gr.Column():
                                with gr.Group():
                                    model_type = gr.Dropdown(
                                        label=self.I18N_helper.t("model_type"),
                                        choices=[],
                                        filterable=False,
                                        value=None,
                                        interactive=True
                                    )
                                    model_name = gr.Dropdown(
                                        label=self.I18N_helper.t("model_name"),
                                        choices=[],
                                        value=None,
                                        filterable=False,
                                        interactive=True
                                    )
                                    vr_aggr_slider = gr.Slider(
                                        label=self.I18N_helper.t("vr_aggr_slider"),
                                        minimum=0,
                                        maximum=100,
                                        step=1,
                                        value=5,
                                        visible=False,
                                        interactive=True
                                    )
                                    with gr.Accordion(label=self.I18N_helper.t("add_settings"), open=False):
                                        extract_instrumental = gr.Checkbox(
                                            label=self.I18N_helper.t("extract_instrumental"),
                                            value=True,
                                            interactive=True,
                                        )
                                        target_instrument = gr.Textbox(
                                            label=self.I18N_helper.t("target_instrument"),
                                            value=None,
                                            interactive=False,
                                        )
                                        stems_list = gr.CheckboxGroup(
                                            label=self.I18N_helper.t("stems_list"),
                                            info="",
                                            choices=[],
                                            value=None,
                                            interactive=False,
                                        )
                                        with gr.Accordion(label=self.I18N_helper.t("template"), open=False):
                                            gr.Markdown(self.I18N_helper.t("template_help"))
                                            template = gr.Textbox(
                                                label=self.I18N_helper.t("template"), value="NAME_MODEL_STEM", interactive=True
                                            )
                                    output_format = gr.Dropdown(
                                        label=self.I18N_helper.t("output_format"),
                                        choices=self.formats,
                                        value="mp3",
                                        filterable=False,
                                        interactive=True
                                    )
                                    separate_btn = gr.Button(self.I18N_helper.t("separate"), variant="primary", interactive=True)
                                    b_separate_btn = gr.Button(self.I18N_helper.t("separate"), variant="primary", visible=False, interactive=True)

                        with gr.Group():
                            batch_base_dir = gr.Dropdown(label=self.I18N_helper.t("batch_base_dir"), choices=[], interactive=True, visible=False)
                            output_audio = [
                                gr.Audio(
                                    interactive=False,
                                    type="filepath",
                                    visible=False,
                                    show_download_button=True,
                                )
                                for _ in range(64)
                            ]
                            output_zip = gr.DownloadButton(label=self.I18N_helper.t("output_zip"), visible=False, interactive=True)
                    with gr.Tab(self.I18N_helper.t("history_tab")):
                        with gr.Tab(self.I18N_helper.t("history_single_tab")):

                            with gr.Group():
                                h_refresh_btn = gr.Button(value=self.I18N_helper.t("refresh_btn"), interactive=True)
                                history_list = gr.Dropdown(label=self.I18N_helper.t("select_history_task"), choices=[], value=None, interactive=True)
                                hs_output_audio = [
                                    gr.Audio(
                                        interactive=False,
                                        type="filepath",
                                        visible=False,
                                        show_download_button=True,
                                    )
                                    for _ in range(64)
                                ]

                        with gr.Tab(self.I18N_helper.t("history_batch_tab")):
                            with gr.Group():
                                hb_refresh_btn = gr.Button(value=self.I18N_helper.t("refresh_btn"), interactive=True)
                                historyb_list = gr.Dropdown(label=self.I18N_helper.t("select_history_task"), choices=[], value=None, interactive=True)
                                historyb_base_dir = gr.Dropdown(label=self.I18N_helper.t("batch_base_dir"), choices=[], value=None, interactive=True)
                                hb_output_audio = [
                                    gr.Audio(
                                        interactive=False,
                                        type="filepath",
                                        visible=False,
                                        show_download_button=True,
                                    )
                                    for _ in range(64)
                                ]

                    if self.plugins:
                        self.gradio_plugin_helper.load_plugin_ui(lang=self.I18N_helper.language)

                    h_refresh_btn.click(
                        lambda x: gr.update(choices=self.history_manager.parse_history_list(type=x), value=None),
                        inputs=[gr.State("single")],
                        outputs=history_list
                    )

                    history_list.change(
                        self.gradio_helper.history_send_player,
                        inputs=[gr.State("single"), history_list],
                        outputs=[*hs_output_audio]
                    )

                    hb_refresh_btn.click(
                        lambda x: gr.update(choices=self.history_manager.parse_history_list(type=x), value=None),
                        inputs=[gr.State("batch")],
                        outputs=historyb_list
                    )

                    historyb_list.change(
                        self.gradio_helper.history_send_player,
                        inputs=[gr.State("batch"), historyb_list, gr.State(None), gr.State(False)],
                        outputs=[historyb_base_dir]
                    )

                    historyb_base_dir.change(
                        self.gradio_helper.history_send_player,
                        inputs=[gr.State("batch"), historyb_list, historyb_base_dir, gr.State(True)],
                        outputs=[*hb_output_audio]
                    )

                    input_audio.change(
                        lambda x: gr.update(value=x), inputs=input_audio, outputs=input_path
                    )

                    path_0_btn.click(
                        lambda: (gr.update(visible=False), gr.update(visible=True)),
                        outputs=[local, path],
                    )

                    path_1_btn.click(
                        lambda: (gr.update(visible=False), gr.update(visible=True)),
                        outputs=[url, path],
                    )

                    url_0_btn.click(
                        lambda: (gr.update(visible=False), gr.update(visible=True)),
                        outputs=[local, url],
                    )

                    url_1_btn.click(
                        lambda: (gr.update(visible=False), gr.update(visible=True)),
                        outputs=[path, url],
                    )

                    upload_0_btn.click(
                        lambda: (gr.update(visible=True), gr.update(visible=False)),
                        outputs=[local, url],
                    )

                    upload_1_btn.click(
                        lambda: (gr.update(visible=True), gr.update(visible=False)),
                        outputs=[local, path],
                    )

                    download_audio_btn.click(
                        self.gradio_helper.download_wrapper,
                        inputs=[input_link, upload_cookie],
                        outputs=[input_audio, input_path, local, url],
                        show_progress=True,
                    )

                    b_input_audio.change(
                        lambda x: gr.update(value=x), inputs=b_input_audio, outputs=b_input_path
                    )

                    b_path_0_btn.click(
                        lambda: (gr.update(visible=False), gr.update(visible=True)),
                        outputs=[b_local, b_path],
                    )

                    b_path_1_btn.click(
                        lambda: (gr.update(visible=False), gr.update(visible=True)),
                        outputs=[b_url, b_path],
                    )

                    b_url_0_btn.click(
                        lambda: (gr.update(visible=False), gr.update(visible=True)),
                        outputs=[b_local, b_url],
                    )

                    b_url_1_btn.click(
                        lambda: (gr.update(visible=False), gr.update(visible=True)),
                        outputs=[b_path, b_url],
                    )

                    b_upload_0_btn.click(
                        lambda: (gr.update(visible=True), gr.update(visible=False)),
                        outputs=[b_local, b_url],
                    )

                    b_upload_1_btn.click(
                        lambda: (gr.update(visible=True), gr.update(visible=False)),
                        outputs=[b_local, b_path],
                    )

                    b_download_audio_btn.click(
                        self.gradio_helper.download_wrapper_batch,
                        inputs=[b_input_link, b_upload_cookie, b_input_path],
                        outputs=[b_input_audio, b_input_path, b_local, b_url],
                        show_progress=True,
                    )

                    model_type.change(
                        fn=(
                            lambda x: gr.update(
                                choices=self.model_manager.get_mn(x), value=self.model_manager.get_mn(x)[0]
                            )
                        ),
                        inputs=model_type,
                        outputs=model_name,
                    ).then(
                        fn=(
                            lambda x: (
                                gr.update(visible=False if x in ["vr", "mdx"] else True),
                                gr.update(visible=True if x == "vr" else False),
                            )
                        ),
                        inputs=model_type,
                        outputs=[extract_instrumental, vr_aggr_slider],
                    )
                    model_name.change(
                        fn=(lambda x, y: gr.update(choices=self.model_manager.get_stems(x, y), value=None)),
                        inputs=[model_type, model_name],
                        outputs=stems_list,
                    ).then(
                        fn=(
                            lambda x, y: (
                                gr.update(info=f"ID: {self.model_manager.get_id(x, y)}"),
                                gr.update(
                                    interactive=(
                                        True if self.model_manager.get_tgt_inst(x, y) == None else None
                                    ),
                                    info=(
                                        self.I18N_helper.t(
                                            "stems_info",
                                            target_instrument=self.model_manager.get_tgt_inst(x, y),
                                        )
                                        if self.model_manager.get_tgt_inst(x, y) is not None
                                        else self.I18N_helper.t("stems_info2")
                                    ),
                                ),
                                gr.update(value=self.model_manager.get_tgt_inst(x, y)),
                                gr.update(
                                    value=(
                                        True if self.model_manager.get_tgt_inst(x, y) is not None else False
                                    )
                                ),
                            )
                        ),
                        inputs=[model_type, model_name],
                        outputs=[model_name, stems_list, target_instrument, extract_instrumental],
                    )

                    b_separate_btn.click(
                        self.gradio_helper.clear_players,
                        outputs=[batch_base_dir, output_zip, *output_audio]
                    ).then(
                        self.gradio_helper.run_inference_batch,
                        inputs=[
                            b_input_path,
                            model_type,
                            model_name,
                            extract_instrumental,
                            vr_aggr_slider,
                            output_format,
                            template,
                            stems_list,
                        ],
                        outputs=[batch_base_dir, output_zip],
                        show_progress_on=[b_input_audio, b_input_path],
                    )

                    separate_btn.click(
                        self.gradio_helper.clear_players,
                        outputs=[batch_base_dir, output_zip, *output_audio]
                    ).then(
                        self.gradio_helper.run_inference,
                        inputs=[
                            input_path,
                            model_type,
                            model_name,
                            extract_instrumental,
                            vr_aggr_slider,
                            output_format,
                            template,
                            stems_list,
                        ],
                        outputs=[*output_audio, output_zip],
                        show_progress_on=[input_audio, input_path],
                    )

                    batch_base_dir.change(
                        self.gradio_helper.batch_players,
                        inputs=batch_base_dir,
                        outputs=output_audio
                    )

                    batch_toggle.change(
                        fn=(
                            lambda x: (
                                gr.update(visible=False if x == True else True),
                                gr.update(visible=True if x == True else False),
                                gr.update(visible=False if x == True else True),
                                gr.update(visible=True if x == True else False)
                            )
                        ),
                        inputs=batch_toggle,
                        outputs=[single_separation, batch_separation, separate_btn, b_separate_btn]
                    )

                    gr.on(
                        fn=(
                            lambda: gr.update(
                                choices=self.model_manager.get_mt(), value=self.model_manager.get_mt()[0]
                            )
                        ),
                        outputs=model_type,
                    ).then(
                        fn=(
                            lambda x: gr.update(
                                choices=self.model_manager.get_mn(x), value=self.model_manager.get_mn(x)[0]
                            )
                        ),
                        inputs=model_type,
                        outputs=model_name,
                    ).then(
                        fn=(
                            lambda x: (
                                gr.update(visible=False if x in ["vr", "mdx"] else True),
                                gr.update(visible=True if x == "vr" else False),
                            )
                        ),
                        inputs=model_type,
                        outputs=[extract_instrumental, vr_aggr_slider],
                    ).then(
                        fn=(lambda x, y: gr.update(choices=self.model_manager.get_stems(x, y), value=None)),
                        inputs=[model_type, model_name],
                        outputs=stems_list,
                    ).then(
                        fn=(
                            lambda x, y: (
                                gr.update(info=f"ID: {self.model_manager.get_id(x, y)}"),
                                gr.update(
                                    interactive=(
                                        True if self.model_manager.get_tgt_inst(x, y) == None else None
                                    ),
                                    info=(
                                        self.I18N_helper.t(
                                            "stems_info",
                                            target_instrument=self.model_manager.get_tgt_inst(x, y),
                                        )
                                        if self.model_manager.get_tgt_inst(x, y) is not None
                                        else self.I18N_helper.t("stems_info2")
                                    ),
                                ),
                                gr.update(value=self.model_manager.get_tgt_inst(x, y)),
                                gr.update(
                                    value=(
                                        True if self.model_manager.get_tgt_inst(x, y) is not None else False
                                    )
                                ),
                            )
                        ),
                        inputs=[model_type, model_name],
                        outputs=[model_name, stems_list, target_instrument, extract_instrumental],
                    )
                return lite_app
            
            def launch(self):
                self.app = self.ui()
                self.app.launch(
                    server_name="0.0.0.0",
                    server_port=self.port,
                    share=self.share,
                    debug=self.debug,
                    inline=self.inline,
                    favicon_path=os.path.join(SCRIPT_DIR, "assets", "mvsepless.png"),
                    allowed_paths=[os.path.join(os.path.abspath(os.sep), "none"), SCRIPT_DIR, os.path.join(os.path.abspath(os.sep), "sdcard"), os.path.join(os.path.abspath(os.sep), "content"), os.path.join(os.path.abspath(os.sep), "root"), os.path.join(os.path.abspath(os.sep), "home")],
                )
        self.gradio_app = GradioApp

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-inference for separation audio in Google Colab")
    parser.add_argument(
        "--lang", type=str, default="ru", help="Language for translations (default: ru)"
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-command help')

    list_models = subparsers.add_parser('list', help='List of exist models')
    list_models.add_argument("-l_filter", "--list_filter", type=str, default=None, help="Show models in list only with specified stem")

    separate = subparsers.add_parser('separate', help='Separate I/O params')
    separate.add_argument("-i", "--input", type=str, required=True, help="Input file or directory")
    separate.add_argument("-o", "--output", type=str, required=True, help="Output directory")
    separate.add_argument("-mt", "--model_type", type=str, required=True, help="Model type")
    separate.add_argument("-mn", "--model_name", type=str, required=True, help="Model name")
    separate.add_argument("-inst", "--instrumental", action='store_true', help="Extract instrumental")
    separate.add_argument("-stems", "--stems", nargs="+", help="Select output stems")
    separate.add_argument("-bitrate", "--bitrate", type=str, default="320k", help="Output bitrate")
    separate.add_argument("-of", "--format", type=str, default="mp3", help="Output format")
    separate.add_argument("-vr_aggr", "--vr_arch_aggressive", type=int, default=5, help="Aggression for VR ARCH models")
    separate.add_argument('--template', type=str, default='NAME_STEM', help='Template naming of output files')
    separate.add_argument("-l_out", "--list_output", action='store_true', help="Show list output files")

    separate_id = subparsers.add_parser('id_separate', help='Separate I/O params (ID)')
    separate_id.add_argument("-i", "--input", type=str, required=True, help="Input file or directory")
    separate_id.add_argument("-o", "--output", type=str, required=True, help="Output directory")
    separate_id.add_argument("-m_id", "--model_id", type=int, required=True, help="Model ID")
    separate_id.add_argument("-inst", "--instrumental", action='store_true', help="Extract instrumental")
    separate_id.add_argument("-stems", "--stems", nargs="+", help="Select output stems")
    separate_id.add_argument("-bitrate", "--bitrate", type=str, default="320k", help="Output bitrate")
    separate_id.add_argument("-of", "--format", type=str, default="mp3", help="Output format")
    separate_id.add_argument("-vr_aggr", "--vr_arch_aggressive", type=int, default=5, help="Aggression for VR ARCH models")
    separate_id.add_argument('--template', type=str, default='NAME_STEM', help='Template naming of output files')
    separate_id.add_argument("-l_out", "--list_output", action='store_true', help="Show list output files")

    app = subparsers.add_parser('app', help='App')
    app.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio app on (default: 7860)",
    )
    app.add_argument(
        "--share", action="store_true", help="Share the Gradio app publicly"
    )
    app.add_argument("--debug", action="store_true", help="Run in debug mode")
    app.add_argument(
        "--theme",
        type=str,
        default="gamma",
        help="Theme for the Gradio app (default: default)",
    )
    app.add_argument("--plugins", action="store_true", help="Enable plugin support")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    mvsepless = MVSEPLESS(lang=args.lang)
    
    if args.command == 'list':
        mvsepless.model_manager.display_models_info(args.list_filter)
        
    elif args.command == 'separate':
        if os.path.isfile(args.input):
            results = mvsepless.separator.single(
                input_file=args.input,
                output_dir=args.output,
                model_type=args.model_type,
                model_name=args.model_name,
                model_mode="name",
                ext_inst=args.instrumental,
                vr_aggr=args.vr_arch_aggressive,
                output_format=args.format,
                output_bitrate=args.bitrate,
                template=args.template,
                call_method="cli",
                selected_stems=args.stems
            )
            if args.list_output:
                print("Results\n")
                for stem, path in results:
                    print(f"Stem - {stem}\nPath - {path}\n")
                    
        elif os.path.isdir(args.input):
            list_files = []
            for file in os.listdir(args.input):
                abs_path_file = os.path.join(args.input, file)
                if os.path.isfile(abs_path_file):
                    list_files.append(abs_path_file)
            results = mvsepless.separator.batch(
                input_list=list_files,
                output_dir=args.output,
                model_type=args.model_type,
                model_name=args.model_name,
                model_mode="name",
                ext_inst=args.instrumental,
                vr_aggr=args.vr_arch_aggressive,
                output_format=args.format,
                output_bitrate=args.bitrate,
                template=args.template,
                call_method="cli",
                selected_stems=args.stems
            )

            if args.list_output:
                print(" \nResults\n \n")
                for name in list(results.keys()):
                    print(f"Name - {name}")
                    for stem, path in results[name]:
                        print(f"  Stem - {stem}\n  Path - {path}\n")

    elif args.command == 'id_separate':
        if os.path.isfile(args.input):
            results = mvsepless.separator.single(
                input_file=args.input,
                output_dir=args.output,
                id=args.model_id,
                model_mode="id",
                ext_inst=args.instrumental,
                vr_aggr=args.vr_arch_aggressive,
                output_format=args.format,
                output_bitrate=args.bitrate,
                template=args.template,
                call_method="cli",
                selected_stems=args.stems
            )
            if args.list_output:
                print("Results\n")
                for stem, path in results:
                    print(f"Stem - {stem}\nPath - {path}\n")
                    
        elif os.path.isdir(args.input):
            list_files = []
            for file in os.listdir(args.input):
                abs_path_file = os.path.join(args.input, file)
                if os.path.isfile(abs_path_file):
                    list_files.append(abs_path_file)
            results = mvsepless.separator.batch(
                input_list=list_files,
                output_dir=args.output,
                id=args.model_id,
                model_mode="id",
                ext_inst=args.instrumental,
                vr_aggr=args.vr_arch_aggressive,
                output_format=args.format,
                output_bitrate=args.bitrate,
                template=args.template,
                call_method="cli",
                selected_stems=args.stems
            )
                    
            if args.list_output:
                print(" \nResults\n \n")
                for name in list(results.keys()):
                    print(f"Name - {name}")
                    for stem, path in results[name]:
                        print(f"  Stem - {stem}\n  Path - {path}\n")

    elif args.command == 'app':

        gradio_app = mvsepless.gradio_app(port=args.port, share=args.share, debug=args.debug, plugins=args.plugins, theme=args.theme)
        gradio_app.launch()
