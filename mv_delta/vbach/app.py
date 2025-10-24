import os
import sys
import requests
import urllib.request
import zipfile
import shutil
import gradio as gr
import tempfile
from datetime import datetime
import logging
import argparse

from vbach.cli.vbach import voice_conversion, output_file_template, clean_filename
from vbach.utils.model_manager import model_manager
from vbach.utils.audio_utils import Audio

input_formats = Audio().input_formats
output_formats = Audio().output_formats

logging.basicConfig(level=logging.WARNING)

def round_nearest_multiple_of_12(x):
    if x == 0:
        return 0
    return round(x / 12) * 12

def convert_voice(
    input_path: str = None,
    template: str = "NAME_MODEL_F0METHOD_PITCH",
    model_name: str = "",
    input_file: str = None,
    index_rate: float = 0,
    output_format: str = "wav",
    output_bitrate: int = 320,
    stereo_mode: str = "mono",
    method_pitch: str = "rmvpe+",
    pitch: float = 0,
    hop_length: int = 128,
    filter_radius: int = 3,
    rms: float = 0.25,
    protect: float = 0.33,
    f0_min: int = 50,
    f0_max: int = 1100
):

    output_dir = tempfile.mkdtemp(prefix="converted_voice_")
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(output_dir, f"{output_file_template(template, base_name, pitch, method_pitch, model_name)}.{output_format}")

    try:
        output_path = voice_conversion(
            voice_model=model_name,
            vocals_path=input_file,
            output_path=output_path,
            pitch=pitch,
            f0_method=method_pitch,
            index_rate=index_rate,
            filter_radius=filter_radius,
            volume_envelope=rms,
            protect=protect,
            hop_length=hop_length,
            f0_min=f0_min,
            f0_max=f0_max,
            format_output=output_format,
            output_bitrate=f"{int(output_bitrate)}k",
            stereo_mode=stereo_mode
        )
    except Exception as e:
        print(e)
    finally:
        return output_path

def process_audio(
    input_audio: str = None,
    template: str = "NAME_MODEL_F0METHOD_PITCH",
    model: str = "",
    index_rate: float = 0,
    output_format: str = "wav",
    output_bitrate: int = 320,
    stereo_mode: str = "mono",
    method_pitch: str = "rmvpe+",
    pitch: float = 0,
    hop_length: int = 128,
    filter_radius: int = 3,
    rms: float = 0.25,
    protect: float = 0.33,
    f0_min: int = 50,
    f0_max: int = 1100
):
    if not input_audio:
        raise gr.Error()

    if not model:
        raise gr.Error()

    if template is None or template == "":
        template = "Vbach_NAME_MODEL_F0METHOD_PITCH"

    if input_audio:
        if isinstance(input_audio, list):
            if "NAME" not in template:
                template = f"{template}_NAME"
            input_files = [f for f in input_audio if os.path.exists(f)]
            if not input_files:
                raise gr.Error("В списке нет выходных файлов")
            return [convert_voice(
                input_file=f,
                template=template,
                model_name=model,
                index_rate=index_rate,
                output_format=output_format,
                output_bitrate=output_bitrate,
                stereo_mode=stereo_mode,
                method_pitch=method_pitch,
                pitch=pitch,
                hop_length=hop_length,
                filter_radius=filter_radius,
                rms=rms,
                protect=protect,
                f0_min=f0_min,
                f0_max=f0_max
            ) for f in input_files]
        else:
            return convert_voice(
                input_file=input_audio,
                template=template,
                model_name=model,
                index_rate=index_rate,
                output_format=output_format,
                output_bitrate=output_bitrate,
                stereo_mode=stereo_mode,
                method_pitch=method_pitch,
                pitch=pitch,
                hop_length=hop_length,
                filter_radius=filter_radius,
                rms=rms,
                protect=protect,
                f0_min=f0_min,
                f0_max=f0_max
            )

def vbach_plugin_name():
    return "VBach"

def vbach_app():
    with gr.TabItem("Инференс"):
        with gr.Column():
            with gr.Row():
                with gr.Column(variant="panel"):
                    with gr.Group():
                        with gr.Row(equal_height=True):
                            vbach_ui_gr_model = gr.Dropdown(
                                choices=model_manager.parse_voice_models(),
                                label="Имя модели",
                                multiselect=False,
                                interactive=True,
                                scale=4
                            )
                            vbach_ui_gr_model_update_btn = gr.Button(
                                "Обновить", variant="secondary", size="sm", scale=1, min_width=150
                            )
                        vbach_ui_gr_pitch = gr.Slider(
                            minimum=-48,
                            maximum=48,
                            step=1,
                            value=0,
                            label="Высота тона",
                            interactive=True,
                        )
                        vbach_ui_gr_pitch_step = gr.Checkbox(
                            label="Менять только октавы",
                            value=False,
                            interactive=True,
                        )

                with gr.Group():
                    vbach_ui_gr_input_audio = gr.File(
                        label="Входное аудио",
                        file_types=["audio"],
                        file_count="single",
                        interactive=True,
                    )
                    vbach_ui_gr_batch_upload_check = gr.Checkbox(
                        label="Пакетная обработка",
                        value=False,
                        interactive=True,
                    )
            with gr.Group():
                with gr.Row(equal_height=True):
                    vbach_ui_gr_output_format = gr.Dropdown(
                        choices=output_formats,
                        label="Формат вывода",
                        value="mp3",
                        multiselect=False,
                        interactive=True,
                        scale=1
                    )
                    vbach_ui_gr_generated_voice = gr.Audio(
                        label="Сгенерированный голос",
                        type="filepath",
                        interactive=False,
                        show_download_button=True,
                        scale=8
                    )
                    vbach_ui_gr_generated_voices = gr.Files(
                        label="Сгенерированные голоса",
                        type="filepath",
                        interactive=False,
                        visible=False,
                        scale=8
                    )
                    vbach_ui_gr_generate_btn = gr.Button(
                        "Конвертировать один вокал",
                        variant="primary",
                        size="md",
                        scale=1,
                        min_width=150
                    )
                    vbach_ui_gr_generate_batch_btn = gr.Button(
                        "Конвертировать несколько вокалов",
                        variant="primary",
                        size="md",
                        scale=1,
                        min_width=150,
                        visible=False
                    )
            with gr.Group():
                vbach_ui_gr_method_pitch = gr.Radio(
                    choices=["mangio-crepe", "rmvpe+", "fcpe"],
                    label="Метод извлечения тона",
                    value="rmvpe+",
                    interactive=True,
                    scale=1
                )
                vbach_ui_gr_hop_length = gr.Slider(
                    minimum=8,
                    maximum=512,
                    step=8,
                    value=128,
                    label="Длина шага",
                    info="Длина шага влияет на точность передачи высоты тона\nЧем меньше длина шага - тем точнее будет передана высота тона",
                    interactive=True,
                    visible=False
                )
            with gr.Accordion(
                "Настройка преобразования",
                open=False,
            ):
                with gr.Accordion(
                    "Стандартные настройки",
                    open=False,
                ):
                    with gr.Group():
                        vbach_ui_gr_stereo_mode = gr.Radio(
                            choices=["mono", "left/right", "sim/dif"],
                            label="Стерео режим",
                            info="mono - монофоническая обработка аудио, \nleft/right - обработка левого и правого каналов отдельно, \nsim/dif - обработка фантомного центра и стерео-базы, разделенную на левый и правый каналы",
                            value="mono",
                            interactive=True,
                            scale=1
                        )

                        vbach_ui_gr_index_rate = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0,
                            label="Влияние индекса",
                            info="Чем ниже значение, тем больше голос похож на исходный; чем выше, тем ближе к модели",
                            interactive=True
                        )

                        vbach_ui_gr_filter_radius = gr.Slider(
                            minimum=1,
                            maximum=7,
                            step=1,
                            value=3,
                            label="Радиус медианного фильтра",
                            info="Сглаживает результаты извлечения тона\nМожет снизить дыхание и шумы на выходе",
                            interactive=True
                        )
                        vbach_ui_gr_rms = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.25,
                            label="Соотношение огибающих громкости",
                            info="Значение 0 - огибающая громкости как у входного аудио, 1 - как у выходного сигнала",
                            interactive=True
                        )
                        vbach_ui_gr_protect = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            step=0.01,
                            value=0.33,
                            label="Защита согласных",
                            info="Предовращает роботизацию дыхания и согласных (Может влиять на четкость речи)\nЗначение 0.5 - выключает защиту, 0 - максимальная защита",
                            interactive=True
                        )
                with gr.Accordion(
                    "Дополнительные настройки",
                    open=False,
                ):
                    with gr.Row(equal_height=True):    
                        vbach_ui_gr_f0_min = gr.Slider(
                            minimum=1,
                            maximum=120,
                            step=1,
                            value=50,
                            label="Нижний предел диапазона определения высоты тона",
                            interactive=True
                        )
                        vbach_ui_gr_f0_max = gr.Slider(
                            minimum=380,
                            maximum=16000,
                            step=1,
                            value=1100,
                            label="Верхний предел диапазона определения высоты тона",
                            interactive=True

                        )

                with gr.Accordion(
                    "Настройки экспорта",
                    open=False,
                ):
                    with gr.Group():
                        with gr.Accordion(
                            "Выходное имя файла",
                            open=False,
                        ):
                            gr.Markdown("""
> Доступные ключи для формата имени вывода:
> (изменить формат имени вывода можно здесь)

> * **NAME** - Имя входного файла
> * **DATETIME** - Дата и время создания результатов
> * **MODEL** - Имя голосовой модели
> * **F0METHOD** - Метод извлечения тона
> * **PITCH** - Высота тона

> Пример:
> * **Шаблон:** NAME_MODEL_F0METHOD_PITCH
> * **Результат:** name_your-model_rmvpe+_12

<div style="color: red; font-weight: bold; background-color: #ffecec; padding: 10px; border-left: 3px solid red; margin: 10px 0;">

Используйте ТОЛЬКО указанные ключи (NAME, DATETIME, MODEL, F0METHOD, PITCH) для избежания повреждения файла.

НЕ добавляйте дополнительный текст или символы вне этих ключей, либо делайте это с осторожностью.
Шаблон автоматически очищается от дубликатов ключей, запрещенных символов и укорачивается в процессе экспорта. 
А также укорачивается имя входного файла, если длина выходного имени превышает допустимый лимит

</div>
        """, line_breaks=True)
                            vbach_ui_gr_template = gr.Textbox(
                                label="Шаблон",
                                value="NAME_MODEL_F0METHOD_PITCH",
                                lines=1,
                                interactive=True,
                                placeholder="NAME_MODEL_F0METHOD_PITCH"
                            )

                        vbach_ui_gr_output_bitrate = gr.Slider(
                            minimum=32,
                            maximum=320,
                            step=8,
                            value=320,
                            label="Битрейт (в кбит/сек)",
                            interactive=True
                        )

    vbach_ui_gr_batch_upload_check.change(
        lambda x: (
            gr.update(file_count="multiple" if x == True else "single", value=None),
            gr.update(visible=False if x == True else True), 
            gr.update(visible=True if x == True else False),
            gr.update(visible=False if x == True else True), 
            gr.update(visible=True if x == True else False)
        ),
        inputs=vbach_ui_gr_batch_upload_check,
        outputs=[vbach_ui_gr_input_audio, vbach_ui_gr_generated_voice, vbach_ui_gr_generated_voices, vbach_ui_gr_generate_btn, vbach_ui_gr_generate_batch_btn]
    )

    vbach_ui_gr_model_update_btn.click(
        lambda : gr.update(choices=model_manager.parse_voice_models()),
        outputs=vbach_ui_gr_model
    )
    vbach_ui_gr_pitch_step.change(
        lambda x, y: gr.update(step=12 if x == True else 1, value=round_nearest_multiple_of_12(y) if x == True else y),
        inputs=[vbach_ui_gr_pitch_step, vbach_ui_gr_pitch],
        outputs=vbach_ui_gr_pitch
    )
    vbach_ui_gr_method_pitch.change(
        lambda x: gr.update(visible=True if x == "mangio-crepe" else False),
        inputs=vbach_ui_gr_method_pitch,
        outputs=vbach_ui_gr_hop_length
    )
    vbach_ui_gr_generate_btn.click(
        fn=process_audio,
        inputs=[
            vbach_ui_gr_input_audio, vbach_ui_gr_template, vbach_ui_gr_model, vbach_ui_gr_index_rate, vbach_ui_gr_output_format, vbach_ui_gr_output_bitrate,
            vbach_ui_gr_stereo_mode, vbach_ui_gr_method_pitch, vbach_ui_gr_pitch, vbach_ui_gr_hop_length, vbach_ui_gr_filter_radius, vbach_ui_gr_rms, vbach_ui_gr_protect,
            vbach_ui_gr_f0_min, vbach_ui_gr_f0_max
        ],
        outputs=[vbach_ui_gr_generated_voice]
    )
    vbach_ui_gr_generate_batch_btn.click(
        fn=process_audio,
        inputs=[
            vbach_ui_gr_input_audio, vbach_ui_gr_template, vbach_ui_gr_model, vbach_ui_gr_index_rate, vbach_ui_gr_output_format, vbach_ui_gr_output_bitrate,
            vbach_ui_gr_stereo_mode, vbach_ui_gr_method_pitch, vbach_ui_gr_pitch, vbach_ui_gr_hop_length, vbach_ui_gr_filter_radius, vbach_ui_gr_rms, vbach_ui_gr_protect,
            vbach_ui_gr_f0_min, vbach_ui_gr_f0_max
        ],
        outputs=[vbach_ui_gr_generated_voices]
    )

    with gr.TabItem("Менеджер"):
        with gr.TabItem("Загрузить по ссылке"):
            with gr.TabItem("Через zip файл"):
                with gr.Row():
                    with gr.Column(variant="panel"):
                        vbach_ui_gr_model_zip_link = gr.Text(label="Ссылка на zip файл")
                        with gr.Group():
                            vbach_ui_gr_zip_model_name = gr.Text(
                                label="Имя модели",
                            )
                            vbach_ui_gr_download_btn = gr.Button("Загрузить", variant="primary")

                        vbach_ui_gr_dl_output_message = gr.Text(label="Статус", interactive=False, lines=5)
                        vbach_ui_gr_download_btn.click(
                            (lambda x, y: model_manager.install_model_zip(x, clean_filename(y, length=40), "url")),
                            inputs=[vbach_ui_gr_model_zip_link, vbach_ui_gr_zip_model_name],
                            outputs=vbach_ui_gr_dl_output_message,
                        )


            with gr.TabItem("Через отдельные файлы"):
                with gr.Row():
                    with gr.Column(variant="panel"):
                        vbach_ui_gr_model_pth_link = gr.Text(label="Ссылка на *.pth файл")
                        vbach_ui_gr_model_index_link = gr.Text(label="Ссылка на *.index файл (необязательно)")
                        with gr.Group():
                            vbach_ui_gr_install_model_name = gr.Text(
                                label="Имя модели",
                            )
                            vbach_ui_gr_download_2_btn = gr.Button("Загрузить", variant="primary")

                        vbach_ui_gr_dl_2_output_message = gr.Text(label="Статус", interactive=False, lines=5)
                        vbach_ui_gr_download_2_btn.click(
                            (lambda x, y, z: model_manager.install_model_files(x, y, clean_filename(z, length=40), "url")),
                            inputs=[vbach_ui_gr_model_index_link, vbach_ui_gr_model_pth_link, vbach_ui_gr_install_model_name],
                            outputs=vbach_ui_gr_dl_2_output_message,
                        )

        with gr.Tab("Загрузить с устройства"):
            with gr.Tab("Через zip файл"):
                with gr.Row():
                    with gr.Column():
                        vbach_ui_gr_zip_file = gr.File(
                            label="zip файл", file_types=[".zip"], file_count="single"
                        )
                    with gr.Column(variant="panel"):
                        with gr.Group():
                            vbach_ui_gr_local_model_name = gr.Text(
                                label="Имя модели",
                            )
                            vbach_ui_gr_model_upload_button = gr.Button("Загрузить", variant="primary")

                        vbach_ui_gr_local_upload_output_message = gr.Text(label="Статус", interactive=False, lines=5)
                        vbach_ui_gr_model_upload_button.click(
                            (lambda x, y: model_manager.install_model_zip(x, clean_filename(y, length=40), "local")),
                            inputs=[vbach_ui_gr_zip_file, vbach_ui_gr_local_model_name],
                            outputs=vbach_ui_gr_local_upload_output_message,
                        )

            with gr.TabItem("Через отдельные файлы"):
                with gr.Group():
                    with gr.Row():
                        vbach_ui_gr_pth_file = gr.File(
                            label="*.pth файл", file_types=[".pth"], file_count="single"
                        )
                        vbach_ui_gr_index_file = gr.File(
                            label="*.index файл (необязательно)", file_types=[".index"], file_count="single"
                        )
                    with gr.Column(variant="panel"):
                        with gr.Group():
                            vbach_ui_gr_separate_model_name = gr.Text(
                                label="Имя модели",
                            )
                            vbach_ui_gr_separate_upload_button = gr.Button("Загрузить", variant="primary")

                        vbach_ui_gr_separate_upload_output_message = gr.Text(
                            label="Статус", interactive=False
                        )
                        vbach_ui_gr_separate_upload_button.click(
                            (lambda x, y, z: model_manager.install_model_files(x, y, clean_filename(z, length=40), "local")),
                            inputs=[vbach_ui_gr_index_file, vbach_ui_gr_pth_file, vbach_ui_gr_separate_model_name],
                            outputs=vbach_ui_gr_separate_upload_output_message,
                        )

            with gr.TabItem("Удалить модель"):
              with gr.Column(variant="panel"):
                with gr.Group():
                  vbach_ui_gr_delete_voicemodel_name = gr.Dropdown(
                    label="Имя модели",
                    choices=model_manager.parse_voice_models(),
                    interactive=True,
                    filterable=False
                  )
                  vbach_ui_gr_refresh_delete_btn = gr.Button("Обновить")
                  vbach_ui_gr_refresh_delete_btn.click(fn=(lambda : gr.update(choices=model_manager.parse_voice_models())), inputs=None, outputs=vbach_ui_gr_delete_voicemodel_name)
                  vbach_ui_gr_delete_model_output_message = gr.Text(
                    label="Статус", interactive=False, lines=5
                  )
                  vbach_ui_gr_delete_model_btn = gr.Button("Удалить")
                  vbach_ui_gr_delete_model_btn.click(
                    fn=model_manager.del_voice_model,
                    inputs=vbach_ui_gr_delete_voicemodel_name,
                    outputs=vbach_ui_gr_delete_model_output_message
                  )

        gr.on(fn=lambda: gr.update(choices=model_manager.parse_voice_models()), inputs=None, outputs=vbach_ui_gr_delete_voicemodel_name)
        gr.on(fn=lambda: gr.update(choices=model_manager.parse_voice_models()), inputs=None, outputs=vbach_ui_gr_model)

if __name__ == "__main__": 
    theme = gr.themes.Citrus(
        primary_hue="teal",
        secondary_hue="blue",
        neutral_hue="blue",
        spacing_size="sm",
        font=[
            gr.themes.GoogleFont("Montserrat"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ]
    )
    app = argparse.ArgumentParser(description='Vbach APP')
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
    args = app.parse_args()
    with gr.Blocks(theme=theme) as vbach_app_2:
        vbach_app()
    vbach_app_2.launch(server_port=args.port, share=args.share, debug=args.debug, allowed_paths=[
                        os.path.join(os.path.abspath(os.sep), "none"),
                        os.getcwd(),
                        os.path.expanduser('~'),
                        os.path.join(os.path.abspath(os.sep), "sdcard"),
                        os.path.join(os.path.abspath(os.sep), "content"),
                    ])
