# plugins

import os
import sys
import time
import subprocess
import shutil
import importlib.util
import gradio as gr

CURRENT_LANG = "ru"

plugins_dir = os.path.join(os.getcwd(), "plugins")
os.makedirs(plugins_dir, exist_ok=True)

TRANSLATIONS = {
    "ru": {
        "plugin_manager": "Менеджер плагинов",
        "upload_label": "Загрузить плагин",
        "upload_btn": "Загрузить",
        "loading_plugin": "Загрузка плагина: {name}",
        "plugin_error": "Ошибка плагина {name}: {error}",
        "plugin_not_found": "Плагин {name} не найден",
        "plugin_loaded": "Плагин {name} успешно загружен",
        "plugin_failed": "Не удалось загрузить плагин {name}",
        "install_plugins": "Установить плагины",
    },
    "en": {
        "plugin_manager": "Plugin Manager",
        "upload_label": "Upload Plugin",
        "upload_btn": "Upload",
        "loading_plugin": "Loading plugin: {name}",
        "plugin_error": "Plugin {name} error: {error}",
        "plugin_not_found": "Plugin {name} not found",
        "plugin_loaded": "Plugin {name} loaded successfully",
        "plugin_failed": "Failed to load plugin {name}",
        "install_plugins": "Install Plugins",
    }
}

def t(key, **kwargs): 
    """Функция для получения перевода с подстановкой значений"""
    lang = CURRENT_LANG
    translation = TRANSLATIONS.get(lang, {}).get(key, key)
    return translation.format(**kwargs) if kwargs else translation

def set_lang(lang):
    """Функция для установки текущего языка"""
    global CURRENT_LANG
    if lang in TRANSLATIONS:
        CURRENT_LANG = lang
    else:
        raise ValueError(f"Unsupported language: {lang}")

def restart():
    python = sys.executable
    subprocess.Popen([python] + sys.argv)
    os._exit(0)

def upload_plugin_list(files):
    if not files:
        return
    for file in files:
        try:
            shutil.copy(file, os.path.join(plugins_dir, os.path.basename(file)))
        except Exception as e:
            print(f"Error copying plugin: {e}")
    time.sleep(2)
    restart()

def load_plugins():
    plugins = []
    if os.path.exists(plugins_dir) and os.path.isdir(plugins_dir):
        try:
            for filename in os.listdir(plugins_dir):
                if filename.endswith(".py") and filename != "__init__.py":
                    file_path = os.path.join(plugins_dir, filename)
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

def load_plugin_ui(lang="ru"):
    plugins = load_plugins()
    set_lang(lang)
    with gr.Tab(t("plugin_manager")):

        with gr.Tab(t("install_plugins")):
            with gr.Blocks():
                upload_plugin_files = gr.Files(
                    label=t("upload_label"), file_types=[".py"], interactive=True
                )
                upload_btn = gr.Button(t("upload_btn"), interactive=True)
                upload_btn.click(fn=upload_plugin_list, inputs=upload_plugin_files)
        if plugins:
            for name, func in plugins:
                try:
                    print(t("loading_plugin", name=name))
                    with gr.Tab(name):
                        func(lang)
                except Exception as e:
                    print(t("plugin_error", name=name, error=str(e)))
                    pass