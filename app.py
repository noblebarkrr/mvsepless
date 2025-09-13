import gradio as gr
import os
import zipfile
from datetime import datetime
from multi_inference import MVSEPLESS, OUTPUT_FORMATS
from utils.download_audio import Downloader

from assets.translations import TRANSLATIONS, TRANSLATIONS_STEMS
from assets.themes import THEMES
from utils.plugin_manager import load_plugin_ui
import argparse

mvsepless = MVSEPLESS()
downloader = Downloader()


CURRENT_LANG = "ru"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def set_lang(lang):
    """Функция для установки текущего языка"""
    global CURRENT_LANG
    if lang in TRANSLATIONS:
        CURRENT_LANG = lang
    else:
        raise ValueError(f"Unsupported language: {lang}")


def t(key, **kwargs):
    """Функция для получения перевода с подстановкой значений"""
    lang = CURRENT_LANG
    translation = TRANSLATIONS.get(lang, {}).get(key, key)
    return translation.format(**kwargs) if kwargs else translation


def t_stem(key, **kwargs):
    """Функция для получения перевода с подстановкой значений"""
    lang = CURRENT_LANG
    translation = TRANSLATIONS_STEMS.get(lang, {}).get(key, key)
    return translation.format(**kwargs) if kwargs else translation


def create_zip(output_audio, output_dir=".", zip_name="output.zip"):
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


def download_wrapper(url, cookie):
    t = downloader.dw_yt_dlp(url, cookie)
    return (
        gr.update(value=t),
        gr.update(value=t),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def run_inference(
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
        CURRENT_DIR, "output", datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    output_audio = mvsepless.separator(
        input_file=input_file,
        output_dir=temp_dir,
        model_type=model_type,
        model_name=model_name,
        output_format=output_format,
        ext_inst=ext_inst,
        call_method="cli",
        vr_aggr=vr_aggr,
        template=template,
        selected_stems=selected_stems,
    )
    audio_updates = [
        gr.update(
            label=t_stem(output_audio[i][0]) if i < len(output_audio) else "",
            value=output_audio[i][1] if i < len(output_audio) else None,
            visible=i < len(output_audio),
        )
        for i in range(64)
    ]
    zip_update = gr.update(
        value=create_zip(
            output_audio,
            output_dir=temp_dir,
            zip_name=f'mvsepless_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip',
        ),
        visible=True,
    )
    return audio_updates + [zip_update]


def parse_args():

    parser = argparse.ArgumentParser(description="Audio separation tool")
    parser.add_argument(
        "--lang", type=str, default="ru", help="Language for translations (default: ru)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio app on (default: 7860)",
    )
    parser.add_argument(
        "--share", action="store_true", help="Share the Gradio app publicly"
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--theme",
        type=str,
        default="gamma",
        choices=THEMES.keys(),
        help="Theme for the Gradio app (default: default)",
    )
    parser.add_argument("--plugins", action="store_true", help="Enable plugin support")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_lang(args.lang)
    theme = THEMES[args.theme]

    with gr.Blocks(theme=theme) as lite_app:
        with gr.Tab(t("inference_tab")):
            with gr.Row():
                with gr.Column():
                    with gr.Group() as local:
                        input_audio = gr.Audio(
                            label=t("upload_label"), type="filepath", interactive=True
                        )
                        with gr.Row(equal_height=True):
                            path_0_btn = gr.Button(t("path_btn"))
                            url_0_btn = gr.Button(t("url_btn"))
                    with gr.Group(visible=False) as url:
                        with gr.Column(variant="compact"):
                            with gr.Row(equal_height=True):
                                upload_cookie = gr.UploadButton(
                                    label=t("upload_cookie"),
                                    file_types=[".txt"],
                                    file_count="single",
                                    scale=1,
                                    variant="primary",
                                )
                                input_link = gr.Textbox(
                                    label=t("url_label"),
                                    placeholder=t("url_placeholder"),
                                    interactive=True,
                                    scale=10,
                                )
                                download_audio_btn = gr.Button(
                                    t("download_audio_btn"), scale=1, variant="stop"
                                )
                        with gr.Row(equal_height=True):
                            path_1_btn = gr.Button(t("path_btn"))
                            upload_0_btn = gr.Button(t("upload_btn"), variant="primary")
                    with gr.Group(visible=False) as path:
                        input_path = gr.Textbox(
                            label=t("path_label"),
                            placeholder=t("path_placeholder"),
                            interactive=True,
                        )
                        with gr.Row(equal_height=True):
                            upload_1_btn = gr.Button(t("upload_btn"), variant="primary")
                            url_1_btn = gr.Button(t("url_btn"))

                with gr.Column():
                    with gr.Group():
                        model_type = gr.Dropdown(
                            label=t("model_type"),
                            choices=[],
                            filterable=False,
                            value=None,
                        )
                        model_name = gr.Dropdown(
                            label=t("model_name"),
                            choices=[],
                            value=None,
                            filterable=False,
                        )
                        vr_aggr_slider = gr.Slider(
                            label=t("vr_aggr_slider"),
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=5,
                            visible=False,
                        )
                        with gr.Accordion(label=t("add_settings"), open=False):
                            extract_instrumental = gr.Checkbox(
                                label=t("extract_instrumental"),
                                value=True,
                                interactive=True,
                            )
                            target_instrument = gr.Textbox(
                                label=t("target_instrument"),
                                value=None,
                                interactive=False,
                            )
                            stems_list = gr.CheckboxGroup(
                                label=t("stems_list"),
                                info="",
                                choices=[],
                                value=None,
                                interactive=False,
                            )
                            with gr.Accordion(label=t("template"), open=False):
                                gr.Markdown(t("template_help"))
                                template = gr.Textbox(
                                    label=t("template"), value="NAME_MODEL_STEM"
                                )
                        output_format = gr.Dropdown(
                            label=t("output_format"),
                            choices=OUTPUT_FORMATS,
                            value="mp3",
                            filterable=False,
                        )
                        separate_btn = gr.Button(t("separate"), variant="primary")

            with gr.Group():
                output_audio = [
                    gr.Audio(
                        interactive=False,
                        type="filepath",
                        visible=False,
                        show_download_button=True,
                    )
                    for _ in range(64)
                ]
                output_zip = gr.DownloadButton(label=t("output_zip"), visible=False)

        if args.plugins:
            load_plugin_ui(lang=args.lang)

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
            download_wrapper,
            inputs=[input_link, upload_cookie],
            outputs=[input_audio, input_path, local, url],
            show_progress=True,
        )

        model_type.change(
            fn=(
                lambda x: gr.update(
                    choices=mvsepless.get_mn(x), value=mvsepless.get_mn(x)[0]
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
            fn=(lambda x, y: gr.update(choices=mvsepless.get_stems(x, y), value=None)),
            inputs=[model_type, model_name],
            outputs=stems_list,
        ).then(
            fn=(
                lambda x, y: (
                    gr.update(
                        interactive=(
                            True if mvsepless.get_tgt_inst(x, y) == None else None
                        ),
                        info=(
                            t(
                                "stems_info",
                                target_instrument=mvsepless.get_tgt_inst(x, y),
                            )
                            if mvsepless.get_tgt_inst(x, y) is not None
                            else t("stems_info2")
                        ),
                    ),
                    gr.update(value=mvsepless.get_tgt_inst(x, y)),
                    gr.update(
                        value=(
                            True if mvsepless.get_tgt_inst(x, y) is not None else False
                        )
                    ),
                )
            ),
            inputs=[model_type, model_name],
            outputs=[stems_list, target_instrument, extract_instrumental],
        )

        separate_btn.click(
            run_inference,
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

        gr.on(
            fn=(
                lambda: gr.update(
                    choices=mvsepless.get_mt(), value=mvsepless.get_mt()[0]
                )
            ),
            outputs=model_type,
        ).then(
            fn=(
                lambda x: gr.update(
                    choices=mvsepless.get_mn(x), value=mvsepless.get_mn(x)[0]
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
            fn=(lambda x, y: gr.update(choices=mvsepless.get_stems(x, y), value=None)),
            inputs=[model_type, model_name],
            outputs=stems_list,
        ).then(
            fn=(
                lambda x, y: (
                    gr.update(
                        interactive=(
                            True if mvsepless.get_tgt_inst(x, y) == None else None
                        ),
                        info=(
                            t(
                                "stems_info",
                                target_instrument=mvsepless.get_tgt_inst(x, y),
                            )
                            if mvsepless.get_tgt_inst(x, y) is not None
                            else t("stems_info2")
                        ),
                    ),
                    gr.update(value=mvsepless.get_tgt_inst(x, y)),
                    gr.update(
                        value=(
                            True if mvsepless.get_tgt_inst(x, y) is not None else False
                        )
                    ),
                )
            ),
            inputs=[model_type, model_name],
            outputs=[stems_list, target_instrument, extract_instrumental],
        )

    lite_app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        favicon_path=os.path.join(CURRENT_DIR, "assets", "mvsepless.png"),
        allowed_paths=["/none", CURRENT_DIR],
    )
