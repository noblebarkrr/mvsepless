import os
import gradio as gr
from utils.inverter import Inverter
from multi_inference import MVSEPLESS
from separator.audio_utils import Audio

output_formats = Audio().output_formats
input_formats = Audio().input_formats

inverter = Inverter(save_to_temp=True)

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
        "plugin_description": "Инструмент для удаления одного аудио из другого",
        "overlap": "Перекрытие",
        "window_size": "Размер окна",
        "window_type": "Тип окна",
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
        "plugin_description": "Tool to remove one audio from another",
        "overlap": "Overlap",
        "window_size": "Window size",
        "window_type": "Window type",
    },
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
        inverter_audio1 = gr.File(label=t("main_audio"), type="filepath", file_types=[f".{of}" for of in input_formats])
        inverter_audio2 = gr.File(label=t("audio_to_remove"), type="filepath", file_types=[f".{of}" for of in input_formats])

    inverter_method = gr.Radio(
        choices=["waveform", "spectrogram"],
        label=t("processing_method"),
        value="waveform",
        interactive=True
    )

    with gr.Group(visible=False) as inverter_spec_settings:
        inverter_wt = gr.Dropdown(
            inverter.w_types,
            value=inverter.w_types[4],
            filterable=False,
            interactive=True,
            label=t("window_type"),
            visible=True
        )

        inverter_ovl = gr.Slider(
            2,
            30,
            value=4,
            step=1,
            label=t("overlap"),
            interactive=True,
            visible=True,
        )
        inverter_ws = gr.Number(
            label=t("window_size"),
            interactive=True,
            visible=True,
            minimum=32,
            maximum=882000,
            precision=1,
            value=2048,
        )

    inverter_output_format = gr.Dropdown(
        choices=output_formats, value="mp3", label=t("output_format"), filterable=False
    )

    inverter_run_btn = gr.Button(t("invert_button"))

    with gr.Column():
        inverter_output = gr.Audio(
            label=t("results"), interactive=False, show_download_button=True
        )
        inverter_output_wav = gr.Text(interactive=False, visible=False)

    inverter_method.change(
        lambda x: gr.update(visible=True if x == "spectrogram" else False),
        inputs=inverter_method,
        outputs=inverter_spec_settings
    )

    inverter_run_btn.click(
        inverter.process_audio,
        inputs=[inverter_audio1, inverter_audio2, inverter_output_format, inverter_method, inverter_ws, inverter_ovl, inverter_wt],
        outputs=[inverter_output, inverter_output_wav],
    )
