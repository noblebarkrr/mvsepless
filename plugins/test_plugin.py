import gradio as gr

MVSEPLESS_TRANSLATIONS = {
    "ru": {
        "test": "Сейчас интерфейс на русском языке"
    },
    "en": {
        "test": "Now interface on English"
    }

CURRENT_LANG = "ru"

def set_language(lang):
    global CURRENT_LANG
    CURRENT_LANG = lang

def t(key, **kwargs):
    """Функция для получения перевода с подстановкой значений"""
    translation = TRANSLATIONS[CURRENT_LANG].get(key, key)
    return translation.format(**kwargs) if kwargs else translation

def test_plugin_name():
    return "Test"

def test_plugin(lang):
    set_language(lang)
    with gr.Blocks():
        gr.Markdown(t('test'))