import os
import gradio as gr
from utils.inverter import Inverter

inverter = Inverter()

CURRENT_LANG = "ru"

def set_language(lang):
    global CURRENT_LANG
    CURRENT_LANG = lang

TRANSLATIONS = {
    "ru": {
        "invert_button": "Инвертировать аудио",
        "main_audio": "Основное аудио",
        "audio_to_remove": "Аудио для удаления",
        "processing_method": "Метод обработки",
        "output_format": "Формат выходного файла",
        "results": "Результат",
        "waveform": "Волновая форма",
        "spectrogram": "Спектрограмма",
        "plugin_title": "Аудио Инвертор",
        "plugin_description": "Инструмент для удаления одного аудио из другого"
    },
    "en": {
        "invert_button": "Invert audio",
        "main_audio": "Main audio",
        "audio_to_remove": "Audio to remove",
        "processing_method": "Processing method",
        "output_format": "Output format",
        "results": "Results",
        "waveform": "Waveform",
        "spectrogram": "Spectrogram",
        "plugin_title": "Audio Inverter",
        "plugin_description": "Tool to remove one audio from another"
    }
}

def t(key, **kwargs):
    """Функция для получения перевода с подстановкой значений"""
    translation = TRANSLATIONS[CURRENT_LANG].get(key, key)
    return translation.format(**kwargs) if kwargs else translation

def plugin_name():
    return "Inverter"

def plugin(lang="ru"):
    set_language(lang)
        
    with gr.Row():
        audio1 = gr.Audio(label=t("main_audio"), type="filepath")
        audio2 = gr.Audio(label=t("audio_to_remove"), type="filepath")
    
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
        invert_man_output = gr.Audio(label=t("results"), interactive=False, show_download_button=True)
        invert_man_output_wav = gr.Text(interactive=False, visible=False)
    
    invert_man_btn.click(
        inverter.process_audio,
        inputs=[audio1, audio2, output_man_i_format, invert_man_method],
        outputs=[invert_man_output, invert_man_output_wav]               
    )
