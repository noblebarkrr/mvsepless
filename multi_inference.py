import os
import sys
import argparse
from pyngrok import ngrok
import gradio as gr
from datetime import datetime
import importlib.util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

import json
from model_list import models_data, medley_vox_models
from utils.preedit_config import conf_editor
from utils.download_models import download_model
from assets.translations import MVSEPLESS_TRANSLATIONS as TRANSLATIONS

MODELS_CACHE_DIR = os.path.join(SCRIPT_DIR, os.path.join("separator", "models_cache"))
OUTPUT_FORMATS = ["mp3", "wav", "flac", "ogg", "opus", "m4a", "aac", "aiff"]
OUTPUT_DIR = "/content/output"


# Глобальная переменная для текущего языка
CURRENT_LANG = "ru"

def set_language(lang):
    global CURRENT_LANG
    CURRENT_LANG = lang

def t(key, **kwargs):
    """Функция для получения перевода с подстановкой значений"""
    translation = TRANSLATIONS[CURRENT_LANG].get(key, key)
    return translation.format(**kwargs) if kwargs else translation

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
            # Создаем базовую команду
            cmd = [
                "python",
                "-m",
                "separator.msst_separator",
                f"--input \"{input_file}\"",
                f"--store_dir \"{output_dir}\"",
                f"--model_type \"{model_type}\"",
                f"--model_name \"{model_name}\"",
                f"--config_path \"{conf}\"",
                f"--start_check_point \"{ckpt}\"",
                f"--output_format \"{output_format}\"",
                f"--output_bitrate \"{output_bitrate}\"",
                f"--template \"{template}\"",
                "--save_results_info"
            ]
            
            # Добавляем опциональные аргументы
            if ext_inst:
                cmd.append("--extract_instrumental")
            if selected_stems:
                instruments = " ".join(f'"{stem}"' for stem in selected_stems)
                cmd.append(f'--selected_instruments {instruments}')
            
            # Преобразуем список в строку и выполняем
            command = " ".join(cmd)
            print(f"Executing command: {command}")  # Для отладки
            os.system(command)
            
            # Проверяем результаты
            results_path = os.path.join(output_dir, "results.json")
            if os.path.exists(results_path):
                with open(results_path) as f:
                    return json.load(f)
            else:
                return None
                    
        elif call_method == "direct":
            from separator.msst_separator import mvsep_offline
            return mvsep_offline(
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
                device_ids=0,
                disable_detailed_pbar=False,
                use_tta=False,
                force_cpu=False,
                verbose=False,
                selected_instruments=selected_stems,
                save_results_info=False,
            )
    
    elif model_type in ["vr", "mdx"]:
    
        if model_type == "vr" and models_data[model_type][model_name]["custom_vr"]:
            config_url = models_data[model_type][model_name]["config_url"]
            checkpoint_url = models_data[model_type][model_name]["checkpoint_url"]               
            primary_stem = models_data[model_type][model_name]["primary_stem"]
            conf, ckpt = download_model(MODELS_CACHE_DIR, model_name, model_type, checkpoint_url, config_url)
            
            if call_method == "cli":
                # Создаем команду для custom VR через CLI
                cmd = [
                    "python",
                    "-m",
                    "separator.uvr_sep custom_vr",
                    f"--input_file \"{input_file}\"",
                    f"--ckpt_path \"{ckpt}\"",
                    f"--config_path \"{conf}\"",
                    f"--bitrate \"{output_bitrate}\"",
                    f"--model_name \"{model_name}\"",
                    f"--template \"{template}\"",
                    f"--output_format \"{output_format}\"",
                    f"--primary_stem \"{primary_stem}\"",
                    f"--aggression {vr_aggr}",
                    f"--output_dir \"{output_dir}\"",
                ]
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
                return custom_vr_separate(
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
                                    
        else:
            if call_method == "cli":
                # Создаем команду для non-custom UVR через CLI
                cmd = [
                    "python",
                    "-m",
                    "separator.uvr_sep uvr",
                    f"--input_file \"{input_file}\"",
                    f"--output_dir \"{output_dir}\"",
                    f"--template \"{template}\"",
                    f"--bitrate \"{output_bitrate}\"",
                    f"--model_dir \"{MODELS_CACHE_DIR}\"",
                    f"--model_type \"{model_type}\"",
                    f"--model_name \"{model_name}\"",
                    f"--output_format \"{output_format}\"",
                    f"--aggression {vr_aggr}",
                ]
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
                return non_custom_uvr_inference(
                    input_file=input_file, 
                    output_dir=output_dir, 
                    template=template, 
                    bitrate=output_bitrate, 
                    model_dir=MODELS_CACHE_DIR, 
                    model_type=model_type, 
                    model_name=model_name, 
                    output_format=output_format, 
                    aggression=vr_aggr, 
                    selected_instruments=selected_stems
                )


##############


def medley_voxer(input, output, model_name, output_format, stereo_mode):
    config_url = medley_vox_models[model_name]["config_url"]
    checkpoint_url = medley_vox_models[model_name]["checkpoint_url"]
    medley_vox_model_dir = download_model(MODELS_CACHE_DIR, model_name, "medley_vox", checkpoint_url, config_url)
    command = (
        f"python -m separator.medley_vox.svs.inference "
        f"--inference_data_dir '{input}' "
        f"--results_save_dir '{output}' "
        f"--model_dir '{medley_vox_model_dir}' "
        f"--exp_name {model_name} "
        f"--use_overlapadd=ola "
        f"--stereo '{stereo_mode}' "
        f"--output_format {output_format} "
    )
    os.system(command)
    results_path = os.path.join(output, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return []

def medley_voxer_gradio(input, output, model_name, output_format, stereo_mode):
    output_audio = medley_voxer(input, output, model_name, output_format, stereo_mode)
    results = []
    if output_audio is not None:
        for i, (stem, output_file) in enumerate(output_audio[:2]):
            results.append(gr.update(
                visible=True,
                label=stem,
                value=output_file
            ))
        return tuple(results)


##############
   
    
def multi_voxer(input, output, model_name, output_format, stereo_mode, stems):
    output_audio = medley_voxer(input, output, model_name, output_format, stereo_mode) # primary stems
    results = [] 
    if stems == 2:
        return output_audio
    
    if stems == 4: 
        for stem, file in output_audio:
            voxes = medley_voxer(file, output, model_name, output_format, stereo_mode)
            results.extend(voxes)
        print(results)
        return results

    if stems == 8: 
        for stem, file in output_audio:
            voxes = medley_voxer(file, output, model_name, output_format, stereo_mode)
            for stem2, file2 in voxes:
                voxes2 = medley_voxer(file2, output, model_name, output_format, stereo_mode)
                results.extend(voxes2)
        print(results)
        return results
                    
    if stems == 16: 
        for stem, file in output_audio:
            voxes = medley_voxer(file, output, model_name, output_format, stereo_mode)
            for stem2, file2 in voxes:
                voxes2 = medley_voxer(file2, output, model_name, output_format, stereo_mode)
                for stem3, file3 in voxes2:
                    voxes3 = medley_voxer(file3, output, model_name, output_format, stereo_mode)    
                    results.extend(voxes3)
        print(results)
        return results


##############

def multi_voxer_gradio(input, output, model_name, output_format, stereo_mode, stems):

    output_audio = multi_voxer(input, output, model_name, output_format, stereo_mode, stems)
    batch_names = []
    if output_audio is not None:
        for i, (stem, output_file) in enumerate(output_audio[:20]):
            batch_names.append(gr.update(
                visible=True,
                label=stem,
                value=output_file
            ))
        # Заполняем оставшиеся слоты невидимыми элементами
        while len(batch_names) < 20:
            batch_names.append(gr.update(visible=False, label=None, value=None))
        return tuple(batch_names)                         

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

##############

def mvsepless_sep_gradio(a, b, c, d, e, f, g, h, i_stem, batch):
    if not a:
        text, output = mvsepless(a, b, c, d, e, f, g, "320k", h, "cli", i_stem)
        if batch == True:
            return text, None
        else:
            results = []
            for i in range(20):
                results.append(gr.update(visible=False, label=None, value=None))
            return (gr.update(value=text),) + tuple(results)
    elif a is not None and isinstance(a, list):
        text, batch_separated = mvsepless(a, b, c, d, e, f, g, "320k", h, "cli", i_stem)
        return text, batch_separated        
    elif a is not None and isinstance(a, str):
        text, output_audio = mvsepless(a, b, c, d, e, f, g, "320k", h, "cli", i_stem)
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

##############
        
def batch_show_names(out):
    names = []
    for name, (stems) in out:
        names.append(name)
    return gr.update(choices=names, value=None, visible=True)
    
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
                    # Заполняем оставшиеся слоты невидимыми элементами
                    while len(batch_names) < 20:
                        batch_names.append(gr.update(visible=False, label=None, value=None))
                    return tuple(batch_names)                         
        else:
            for i in range(20):
                batch_names.append(gr.update(visible=(i == 0), label=None, value=None))
            return tuple(batch_names)

##############

def mvsepless_theme(font="Tektur"):
    theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont(font), "Arial", "sans-serif"],
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

    return theme


def create_mvsepless_app(lang):
    # Добавляем переключатель языка
    gr.HTML(f"<h1><center> {t('app_title')} </center></h1>")
    with gr.Tabs():
        with gr.Tab(t("separation")):
            with gr.Tab("MVSEPLESS"):
                with gr.Tab(t("inference")):
                    output_dir = gr.Text(value="/content/output/", visible=False)
                    batch_results_state = gr.State()
                    with gr.Row(equal_height=False):
                        with gr.Column():
                            input_audio = gr.Audio(show_label=False, type="filepath", interactive=True)
                            input_audios = gr.Files(show_label=False, type="filepath", visible=False, interactive=True, file_types=[".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff"])
                            batch_separation = gr.Checkbox(label=t("batch_processing"), value=False, interactive=True, info=t("batch_info"))
                        with gr.Column():
                            with gr.Row():
                                model_type = gr.Dropdown(label=t("model_type"), choices=list(models_data.keys()), value=list(models_data.keys())[0], interactive=True, filterable=False)
                                model_name = gr.Dropdown(label=t("model_name"), choices=list(models_data[list(models_data.keys())[0]].keys()), value=list(models_data[list(models_data.keys())[0]].keys())[0], interactive=True, filterable=False)
                            ext_inst = gr.Checkbox(label=t("extract_instrumental"), visible=True, value=True, interactive=True, info=t("extract_info"))
                            vr_aggr_slider = gr.Slider(label=t("vr_aggressiveness"), minimum=0, maximum=100, step=1, visible=False, interactive=True, value=5)
                            stems = gr.CheckboxGroup(label=t("stems_list"), choices=models_data[list(models_data.keys())[0]][list(models_data[list(models_data.keys())[0]].keys())[0]]["stems"], value=None, interactive=False, info=t("stems_info", target_instrument="vocals"))
                            with gr.Row():
                                template = gr.Text(label=t("template"), value="NAME_(STEM)_MODEL", interactive=True, info=t("template_info"))
                                output_format = gr.Dropdown(label=t("output_format"), choices=OUTPUT_FORMATS, value="mp3", interactive=True, filterable=False)
                            single_separate_btn = gr.Button(t("separate_btn"), variant="primary", interactive=True, size="lg")
                            batch_separate_btn = gr.Button(t("separate_btn"), variant="primary", visible=False, interactive=True, size="lg")
                with gr.Tab(t("model_loading")):
                    dw_m_model_type = gr.Dropdown(label=t("model_type"), choices=list(models_data.keys()), value=list(models_data.keys())[0], interactive=True, filterable=False)
                    dw_m_model_name = gr.Dropdown(label=t("model_name"), choices=list(models_data[list(models_data.keys())[0]].keys()), value=list(models_data[list(models_data.keys())[0]].keys())[0], interactive=True, filterable=False)
                    dw_m_btn = gr.Button(t("download_model_btn"))

                with gr.Tab(t("results")):
                    output_info = gr.Textbox(label=t("separation_info"))
                    batch_select_dir = gr.Dropdown(label=t("select_file"), visible=False, interactive=True, filterable=False)
                    output_stems = [gr.Audio(visible=(i == 0), interactive=False, type="filepath", show_download_button=True) for i in range(20)]
                    
            with gr.Tab("Medley-Vox"):
                with gr.Tab(t("inference")):
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            input_voice = gr.Audio(show_label=False, type="filepath", interactive=True)
                        with gr.Column():
                            vox_model_name = gr.Dropdown(label=t("vox_model_name"), choices=list(medley_vox_models.keys()), value=list(medley_vox_models.keys())[0], interactive=True, filterable=False)
                            stereo_mode = gr.Dropdown(label=t("vox_stereo_mode"), choices=["mono", "full"], value="mono", interactive=True, filterable=False)
                            output_vox_format = gr.Dropdown(label=t("vox_output_format"), choices=list(filter(lambda fmt: fmt != "ogg", OUTPUT_FORMATS)), value="mp3", interactive=True, filterable=False)
                            separate_vox_btn = gr.Button(t("separate_vocals_btn"), variant="primary")
                    output_voxes = [gr.Audio(visible=(i == 0), interactive=False, type="filepath", show_download_button=True) for i in range(2)]

                with gr.Tab(t("vocal_multi_separation")):
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            input_vox = gr.Audio(show_label=False, type="filepath", interactive=True)
                        with gr.Column():
                            vox_m_model_name = gr.Dropdown(label=t("vox_model_name"), choices=list(medley_vox_models.keys()), value=list(medley_vox_models.keys())[0], interactive=True, filterable=False)
                            with gr.Row():
                                stereo_m_mode = gr.Dropdown(label=t("vox_stereo_mode"), choices=["mono", "full"], value="mono", interactive=True, filterable=False)
                                count_stems = gr.Dropdown(label=t("vox_count_stems"), choices=[2, 4, 8, 16], value=2, interactive=True, filterable=False)
                            output_m_vox_format = gr.Dropdown(label=t("vox_output_format"), choices=list(filter(lambda fmt: fmt != "ogg", OUTPUT_FORMATS)), value="mp3", interactive=True, filterable=False)
                            separate_m_vox_btn = gr.Button(t("vox_multi_separate_btn"), variant="primary")
                    output_m_voxes = [gr.Audio(visible=(i == 0), interactive=False, type="filepath", show_download_button=True) for i in range(20)]

        with gr.Tab(t("ensemble")):
            from ensembless import create_ensembless_app as ensem_tab
            ensem_tab(lang)
            
        try:
            from vbach.demo.app import create_demo as vbach_ui
            with gr.Tab(t("transform")):
                vbach_ui(lang)
        except ImportError:
            pass

        with gr.Tab(t("plugins")):

            # Путь к папке с плагинами
            plugins_dir = "plugins"

            # Словарь для хранения найденных плагинов
            plugins = {}
            plugin_name = {}

            # Проверяем существование директории
            if os.path.exists(plugins_dir) and os.path.isdir(plugins_dir):
                # Перебираем файлы в директории
                for filename in os.listdir(plugins_dir):
                    # Фильтруем .py файлы (исключаем __init__.py)
                    if filename.endswith(".py") and filename != "__init__.py":
                        # Формируем полный путь к файлу
                        file_path = os.path.join(plugins_dir, filename)
                        # Получаем имя модуля из имени файла
                        module_name = filename[:-3]  # Удаляем расширение .py
                        
                        # Загружаем модуль
                        spec = importlib.util.spec_from_file_location(module_name, file_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Ищем функции с суффиксом 'plugin'
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            # Проверяем что это функция и имя заканчивается на 'plugin'
                            if callable(attr) and attr_name.endswith("plugin"):
                                # Сохраняем в словарь
                                plugins[f"{module_name}.{attr_name}"] = attr

                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            # Проверяем что это функция и имя заканчивается на 'plugin'
                            if callable(attr) and attr_name.endswith("plugin_name"):
                                # Сохраняем в словарь
                                plugin_name[f"{module_name}.{attr_name}"] = attr

            # Пример использования
            for func_name, func in plugins.items():
                for f_name, pl_name in plugin_name.items():
                   custom_plugin_name = pl_name()
                print(f"Загружен плагин: {func_name}")
                with gr.Tab(custom_plugin_name):
                    func(lang)  # Можно вызвать функцию
                        
            
 
    separate_vox_btn.click(fn=(lambda : os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))), inputs=None, outputs=output_dir).then(fn=medley_voxer_gradio, inputs=[input_voice, output_dir, vox_model_name, output_vox_format, stereo_mode], outputs=output_voxes)

    dw_m_btn.click(fn=downloader_models, inputs=[dw_m_model_type, dw_m_model_name], outputs=None)
    
    separate_m_vox_btn.click(fn=(lambda : os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))), inputs=None, outputs=output_dir).then(fn=multi_voxer_gradio, inputs=[input_vox, output_dir, vox_m_model_name, output_m_vox_format, stereo_m_mode, count_stems], outputs=[*output_m_voxes])
    
    batch_separation.change(fn=(lambda x: (gr.update(visible=True if x == True else False), gr.update(visible=True if x == True else False), gr.update(visible=False if x == True else True), gr.update(visible=False if x == True else True))), inputs=batch_separation, outputs=[input_audios, batch_separate_btn, input_audio, single_separate_btn])

    model_type.change(fn=lambda x: gr.update(visible=True if x == "vr" else False), inputs=model_type, outputs=vr_aggr_slider).then(
        fn=lambda x: gr.update(choices=list(models_data[x].keys()), value=list(models_data[x].keys())[0]), inputs=model_type, outputs=model_name).then(fn=(lambda x: gr.update(visible=False if x in ["vr", "mdx"] else True)), inputs=model_type, outputs=ext_inst)

    dw_m_model_type.change(fn=lambda x: gr.update(choices=[model for model in list(models_data[x].keys()) if (models_data[x][model]["checkpoint_url"] if x not in ["vr", "mdx"] else None) or (models_data[x][model]["custom_vr"] if x == "vr" else None)], value=None if not [model for model in list(models_data[x].keys()) if (models_data[x][model]["checkpoint_url"] if x not in ["vr", "mdx"] else None) or (models_data[x][model]["custom_vr"] if x == "vr" else None)] else [model for model in list(models_data[x].keys()) if (models_data[x][model]["checkpoint_url"] if x not in ["vr", "mdx"] else None) or (models_data[x][model]["custom_vr"] if x == "vr" else None)][0]), inputs=dw_m_model_type, outputs=dw_m_model_name)

    model_name.change(fn=lambda x, y: gr.update(choices=list(models_data[x][y]["stems"]), value=None, interactive=True if models_data[x][y]["target_instrument"] == None else False, info=t("stems_info", target_instrument=models_data[x][y]["target_instrument"]) if models_data[x][y]["target_instrument"] != None else t("stems_info2")), inputs=[model_type, model_name], outputs=stems).then(fn=(lambda x, y: gr.update(value=False if models_data[x][y]["target_instrument"] == None else True) ), inputs=[model_type, model_name], outputs=ext_inst)

    single_separate_btn.click(fn=(lambda : gr.update(choices=None, visible=False, value=None)), inputs=None, outputs=batch_select_dir).then(fn=(lambda : os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))), inputs=None, outputs=output_dir).then(fn=mvsepless_sep_gradio, inputs=[input_audio, output_dir, model_type, model_name, ext_inst, vr_aggr_slider, output_format, template, stems, batch_separation], outputs=[output_info, *output_stems])
    
    batch_separate_btn.click(fn=(lambda : gr.update(choices=None, visible=False, value=None)), inputs=None, outputs=batch_select_dir).then(fn=(lambda : os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))), inputs=None, outputs=output_dir).then(fn=mvsepless_sep_gradio, inputs=[input_audios, output_dir, model_type, model_name, ext_inst, vr_aggr_slider, output_format, template, stems, batch_separation], outputs=[output_info, batch_results_state]).then(fn=batch_show_names, inputs=batch_results_state, outputs=batch_select_dir)
    
    batch_select_dir.change(fn=batch_show_results, inputs=[batch_results_state, batch_select_dir], outputs=[*output_stems])
    


def parse_args():
    parser = argparse.ArgumentParser(description="Базовый интерфейс для разделения музыки и вокала")
    
    # Основные параметры запуска
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP-адрес (по умолчанию: 0.0.0.0)")
    parser.add_argument("--server_port", type=int, default=7860, help="Порт (по умолчанию: 7860)")
    parser.add_argument("--share", action="store_true", help="")
    parser.add_argument("--debug", action="store_true", help="Включить отладку")
    parser.add_argument("--ngrok_token", type=str, help="Аутентификация (формат: username:password)")
    # Настройки безопасности
    parser.add_argument("--auth", type=str, help="Аутентификация (формат: username:password)")
    parser.add_argument("--ssl-keyfile", type=str, help="Путь к SSL ключу")
    parser.add_argument("--ssl-certfile", type=str, help="Путь к SSL сертификату")
    
    # Производительность
    parser.add_argument("--max-file-size", type=str, default="10000MB", help="Максимальный лимит загрузки файлов в интерфейс")
    
    # Пути
    parser.add_argument("--output-dir", type=str, default="/content/output", help="Путь к директории вывода")
    parser.add_argument("--models-cache-dir", type=str, default=None, help="Путь к кэшу моделей")
    parser.add_argument("--language", type=str, choices=["ru", "en"], default="ru", help="Язык интерфейса")
    
    # Шрифт в интерфейсе
    parser.add_argument("--google_font", type=str, default="Montserrat", help="Шрифт в интерфейсе")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Переопределяем пути, если указаны в аргументах
    if args.models_cache_dir:
        MODELS_CACHE_DIR = args.models_cache_dir
    if args.output_dir:
        OUTPUT_DIR = args.output_dir

    set_language(args.language)

    with gr.Blocks(title="Разделение музыки и вокала", theme=mvsepless_theme(args.google_font)) as demo:
        create_mvsepless_app(args.language)
    # Запуск Gradio с парсированными аргументами
    if args.ngrok_token:
        ngrok.set_auth_token(args.ngrok_token)
        ngrok.kill()
        tunnel = ngrok.connect(args.server_port)
        print(f"Публичная ссылка - {tunnel.public_url}")
    
    demo.launch(
        server_name=args.host,
        server_port=args.server_port,
        share=args.share,
        debug=args.debug,
        auth=args.auth.split(":") if args.auth else None,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        max_file_size=args.max_file_size,
        allowed_paths=["/content", OUTPUT_DIR, MODELS_CACHE_DIR]  # Добавляем разрешенные пути
    )
