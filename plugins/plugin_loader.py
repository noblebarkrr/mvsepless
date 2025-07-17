import gradio as gr
import os
import shutil

TRANSLATIONS = {
    "ru": {
        "upload": "Загрузка плагинов (.py)",
        "upload_btn": "Загрузить",
        "restart_warning": "Чтобы загруженные плагины отобразились в интерфейсе, перезапустите Gradio"
    },
    "en": {
        "upload": "Upload plugins (.py)",
        "upload_btn": "Upload",
        "restart_warning": "For loaded plugins to appear in the interface, restart Gradio"
    }
}

CURRENT_LANG = "ru"

def set_language(lang):
    global CURRENT_LANG
    CURRENT_LANG = lang

def t(key, **kwargs):
    """Функция для получения перевода с подстановкой значений"""
    translation = TRANSLATIONS[CURRENT_LANG].get(key, key)
    return translation.format(**kwargs) if kwargs else translation

plugins_dir = "plugins"

def upload_plugin_list(files):
    if not files:
        for file in files:
            shutil(file, os.path.join(plugins_dir, os.path.basename(file)))    

        gr.Warning(t("restart_warning"))

def test_upload_plugin_name():
    return "Uploader Plugins"

def test_upload_plugin(lang):
    set_language(lang)
    with gr.Blocks():
        upload_plugin_files = gr.Files(label=t('upload'), file_types=[".py"])
        upload_btn = gr.Button(t('upload_btn'))
