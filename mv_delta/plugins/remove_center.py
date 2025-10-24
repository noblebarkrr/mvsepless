import os
import gradio as gr
from separator.audio_utils import Audio

output_formats = Audio().output_formats

from utils.remove_center import PhantomCenterExtractor

TRANSLATIONS = {
    "ru": {
        "input_audio": "Входное аудио (только стерео)",
        "output_format": "Формат вывода",
        "reduction": "Сила подавления",
        "overlap": "Перекрытие",
        "window_size": "Размер окна",
        "window_type": "Тип окна",
        "stereo_mode": "Режим",
        "separate_btn": "Разделить",
        "phantom_center": "Фантомный центр",
        "stereo_base": "Стерео-база",
    },
    "en": {
        "input_audio": "Input audio (stereo only)",
        "output_format": "Output format",
        "reduction": "Reduction factor",
        "overlap": "Overlap",
        "window_size": "Window size",
        "window_type": "Window type",
        "stereo_mode": "Mode",
        "separate_btn": "Separate",
        "phantom_center": "Phantom center",
        "stereo_base": "Stereo base",
    },
}

# Глобальная переменная для текущего языка
CURRENT_LANG = "ru"


def set_language(lang):
    global CURRENT_LANG
    CURRENT_LANG = lang


def t(key, **kwargs):
    """Функция для получения перевода с подстановкой значений"""
    translation = TRANSLATIONS[CURRENT_LANG].get(key, key)
    return translation.format(**kwargs) if kwargs else translation

rmv_center = PhantomCenterExtractor()


def plugin_name():
    return "Remove Center"

def plugin(lang):
    set_language(lang)
    with gr.Row():
        rmv_center_ui_input_audio = gr.Audio(
            label=t("input_audio"), type="filepath", interactive=True
        )
        with gr.Group():
            with gr.Row():
                rmv_center_ui_reduction_f = gr.Slider(
                    0.1,
                    10,
                    value=1,
                    step=0.1,
                    label=t("reduction"),
                    interactive=True,
                    visible=False,
                )
                rmv_center_ui_overlap = gr.Slider(
                    2,
                    30,
                    value=2,
                    step=1,
                    label=t("overlap"),
                    interactive=True,
                    visible=True,
                )
                rmv_center_ui_window_size = gr.Number(
                    label=t("window_size"),
                    interactive=True,
                    visible=True,
                    minimum=32,
                    maximum=882000,
                    precision=1,
                    value=2048,
                )
            with gr.Row():
                rmv_center_ui_format = gr.Dropdown(
                    output_formats,
                    value=output_formats[0],
                    filterable=False,
                    label=t("output_format"),
                )
                rmv_center_ui_window_types = gr.Dropdown(
                    rmv_center.w_types,
                    value=rmv_center.w_types[4],
                    filterable=False,
                    label=t("window_type"),
                )

            rmv_center_ui_mono_mode = gr.Dropdown(
                ["mono", "stereo"],
                value="mono",
                filterable=False,
                label=t("stereo_mode"),
            )
            rmv_center_ui_extract_btn = gr.Button(t("separate_btn"))
    with gr.Group():
        with gr.Column():
            with gr.Row():
                rmv_center_ui_mid = gr.Audio(
                    type="filepath",
                    interactive=False,
                    label=t("phantom_center"),
                    visible=True,
                    show_download_button=True,
                )
                rmv_center_ui_side = gr.Audio(
                    type="filepath",
                    interactive=False,
                    label=t("stereo_base"),
                    visible=True,
                    show_download_button=True,
                )
    rmv_center_ui_extract_btn.click(
        fn=rmv_center.remove_center,
        inputs=[
            rmv_center_ui_input_audio,
            rmv_center_ui_format,
            rmv_center_ui_reduction_f,
            rmv_center_ui_window_size,
            rmv_center_ui_overlap,
            rmv_center_ui_window_types,
            rmv_center_ui_mono_mode,
        ],
        outputs=[rmv_center_ui_side, rmv_center_ui_mid],
    )
