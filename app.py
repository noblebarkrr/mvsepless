import os
import sys
import time
import base64

try:
    import yt_dlp
    import validators
except ImportError:
    os.system("pip install validators")
    os.system("pip install yt-dlp")
    import validators
    import yt_dlp

import shutil
import argparse
from datetime import datetime
import gradio as gr
from pyngrok import ngrok
import importlib.util
from multi_inference import MVSEPLESS, OUTPUT_FORMATS
from assets.translations import TRANSLATIONS, TRANSLATIONS_STEMS


def restart_ui():
    python = sys.executable
    os.execl(python, python, *sys.argv)


COOKIE_FILE = None
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloaded")
plugins_dir = os.path.join(os.getcwd(), "plugins")
os.makedirs(plugins_dir, exist_ok=True)


def load_cookie(file):
    global COOKIE_FILE
    COOKIE_FILE = file
    gr.Warning(t("cookie_loaded"))


def upload_plugin_list(files):
    if not files:
        return
    for file in files:
        try:
            shutil.copy(file, os.path.join(plugins_dir, os.path.basename(file)))
        except Exception as e:
            print(f"Error copying plugin: {e}")
    time.sleep(2)
    restart_ui()


def download_file(url, output_format="mp3", output_bitrate="320", title=None):
    """
    Скачивает аудио из URL с возможностью выбора формата, битрейта и названия файла.
    
    Args:
        url (str): URL видео или аудио для скачивания
        output_format (str): Желаемый выходной формат (по умолчанию 'mp3')
        output_bitrate (str): Желаемый битрейт (по умолчанию '320')
        title (str): Желаемое название файла (без расширения). Если None, будет использовано оригинальное название.
    
    Returns:
        str: Абсолютный путь к скачанному файлу или оригинальный URL если скачивание не удалось
    """
    if validators.url(url) and not os.path.exists(url):
        # Подготовка шаблона имени файла
        outtmpl = "%(title)s.%(ext)s" if title is None else f"{title}.%(ext)s"
        
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(DOWNLOAD_DIR, outtmpl),
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
        if COOKIE_FILE and os.path.exists(COOKIE_FILE):
            ydl_opts["cookiefile"] = COOKIE_FILE

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
                gr.Warning(e)
                return url


    return url


CURRENT_LANG = "ru"


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


def gen_out_dir():
    return os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))


mvsepless = MVSEPLESS()


def sep_wrapper(a, b, c, d, e, f, g, h, i):

    if not g:
        g = 128
    results = mvsepless.separator(
        input_file=a,
        output_dir=gen_out_dir(),
        model_type=b,
        model_name=c,
        ext_inst=d,
        vr_aggr=e,
        output_format=f,
        output_bitrate=f"{g}k",
        call_method="cli",
        selected_stems=h,
        template=i,
    )
    stems = []
    if results:
        for i, (stem, output_file) in enumerate(results[:20]):
            stems.append(gr.update(visible=True, label=t_stem(stem), value=output_file))

    while len(stems) < 20:
        stems.append(gr.update(visible=False, label=None, value=None))

    return tuple(stems)


theme = gr.themes.Default(
    primary_hue="violet",
    secondary_hue="cyan",
    neutral_hue="blue",
    spacing_size="sm",
    text_size="sm",
    font=[gr.themes.GoogleFont("Rubik"), "ui-sans-serif", "system-ui", "sans-serif"],
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
)


def create_app(css=None, theme=None):
    with gr.Blocks(theme=theme, css=css) as app:
        with gr.Tab(t("separation")):
            with gr.Column(scale=4):
                with gr.Group():
                    input_audio = gr.File(
                        label=t("select_file"),
                        interactive=True,
                        type="filepath",
                        file_count="single",
                        file_types=["audio"],
                    )

                    with gr.Group():
                        with gr.Row(equal_height=True):

                            input_audio_path = gr.Textbox(
                                show_label=False,
                                placeholder=t("audio_path"),
                                interactive=True,
                                lines=2,
                                scale=20,
                                container=False,
                                show_copy_button=True,
                            )
                            dwn_audio_btn = gr.Button(
                                "↓", variant="primary", min_width=60, scale=1
                            )
                with gr.Group():
                    with gr.Row():
                        model_type = gr.Dropdown(
                            label=t("model_type"), interactive=True, filterable=False
                        )
                        model_name = gr.Dropdown(
                            label=t("model_name"), interactive=True, filterable=False
                        )
                with gr.Accordion(label=t("add_settings"), open=False):
                    with gr.Column(variant="compact"):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=2):
                                target_instrument = gr.Textbox(
                                    label=t("target_instrument"), interactive=False
                                )
                                vr_aggr = gr.Slider(
                                    0,
                                    100,
                                    step=1,
                                    label=t("vr_aggressiveness"),
                                    visible=False,
                                    value=5,
                                    interactive=True,
                                )
                                extract_instrumental = gr.Checkbox(
                                    label=t("extract_instrumental"),
                                    value=False,
                                    interactive=True,
                                )
                            stems_list = gr.CheckboxGroup(
                                label=t("stems_list"),
                                value=None,
                                interactive=False,
                                scale=4,
                            )
                        with gr.Accordion(label=t("encoder_settings"), open=False):
                            with gr.Accordion(label=t("template"), open=False):
                                gr.Markdown(t("template_help"))
                                template = gr.Textbox(
                                    label=t("template"), value="NAME_MODEL_STEM"
                                )
                            with gr.Row():
                                output_format, output_bitrate = gr.Dropdown(
                                    label=t("output_format"),
                                    choices=OUTPUT_FORMATS,
                                    value="mp3",
                                    interactive=True,
                                    filterable=False,
                                ), gr.Slider(
                                    32,
                                    320,
                                    step=1,
                                    label=t("bitrate"),
                                    value=320,
                                    interactive=True,
                                )

                        with gr.Accordion(label=t("downloader_settings"), open=False):
                            with gr.Row(equal_height=True):
                                dw_output_format, dw_output_bitrate = gr.Dropdown(
                                    label=t("output_format"),
                                    choices=OUTPUT_FORMATS,
                                    value="mp3",
                                    interactive=True,
                                    filterable=False,
                                    scale=4,
                                ), gr.Slider(
                                    32,
                                    320,
                                    step=1,
                                    label=t("bitrate"),
                                    value=320,
                                    interactive=True,
                                    scale=4,
                                )
                                use_cookies = gr.UploadButton(
                                    label=t("use_cookies"),
                                    file_count="single",
                                    file_types=[".txt"],
                                    scale=2
                                )
                separate_btn = gr.Button(
                    t("separate_btn"), variant="primary", interactive=True
                )
                with gr.Column(variant="compact"):
                    with gr.Group():
                        with gr.Column(variant="compact"):
                            gr.Markdown(t("results"))
                        output_stems = []
                        for _ in range(10):
                            with gr.Row(equal_height=True):
                                audio1 = gr.Audio(
                                    visible=False,
                                    interactive=False,
                                    type="filepath",
                                    show_download_button=True,
                                )
                                audio2 = gr.Audio(
                                    visible=False,
                                    interactive=False,
                                    type="filepath",
                                    show_download_button=True,
                                )
                                output_stems.extend([audio1, audio2])

        with gr.Tab(t("plugins")):
            plugins = []

            with gr.Tab(t("upload")):
                with gr.Blocks():
                    upload_plugin_files = gr.Files(
                        label=t("upload"), file_types=[".py"], interactive=True
                    )
                    upload_btn = gr.Button(t("upload_btn"), interactive=True)

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

            for name, func in plugins:
                try:
                    print(t("loading_plugin", name=name))
                    with gr.Tab(name):
                        func(CURRENT_LANG)
                except Exception as e:
                    print(t("error_loading_plugin", e=e))
                    pass

        output_format.change(
            lambda x: gr.update(
                visible=False if x in ["flac", "wav", "aiff"] else True
            ),
            inputs=output_format,
            outputs=output_bitrate,
        )
        dw_output_format.change(
            lambda x: gr.update(
                visible=False if x in ["flac", "wav", "aiff"] else True
            ),
            inputs=dw_output_format,
            outputs=dw_output_bitrate,
        )
        dwn_audio_btn.click(
            download_file, inputs=[input_audio_path, dw_output_format, dw_output_bitrate], outputs=input_audio_path
        )
        app.load(fn=mvsepless.update_models).then(
            fn=(
                lambda: gr.update(
                    choices=mvsepless.get_mt(), value=mvsepless.get_mt()[0]
                )
            ),
            inputs=None,
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
            outputs=[extract_instrumental, vr_aggr],
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
        input_audio.upload(
            fn=(lambda x: gr.update(value=x)),
            inputs=input_audio,
            outputs=input_audio_path,
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
            outputs=[extract_instrumental, vr_aggr],
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
            fn=sep_wrapper,
            inputs=[
                input_audio_path,
                model_type,
                model_name,
                extract_instrumental,
                vr_aggr,
                output_format,
                output_bitrate,
                stems_list,
                template,
            ],
            outputs=output_stems,
            show_progress_on=input_audio,
        )
        use_cookies.upload(fn=load_cookie, inputs=use_cookies)
        upload_btn.click(fn=upload_plugin_list, inputs=upload_plugin_files)
    return app


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngrok_token", type=str)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--font", type=str)
    parser.add_argument("--lang", type=str, default="ru", choices=["ru", "en"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CURRENT_LANG = args.lang
    css = """
    .fixed-height { height: 160px !important; min-height: 160px !important; }
    .fixed-height2 { height: 250px !important; min-height: 250px !important; }
    """
    font = args.font
    if (
        font
        and os.path.isfile(font)
        and font.lower().endswith((".ttf", ".otf", ".woff", ".eot"))
    ):
        with open(font, "rb") as font_file:
            base64_font = base64.b64encode(font_file.read()).decode("utf-8")
        css += f"""
        @font-face {{
            font-family: 'CustomFont';
            src: url(data:font/truetype;charset=utf-8;base64,{base64_font}) format('truetype');
        }}
        body, .gradio-container {{ font-family: 'CustomFont', sans-serif !important; }}
        """

    app = create_app(css=css, theme=theme)

    if args.ngrok_token:
        ngrok.set_auth_token(args.ngrok_token)
        ngrok.kill()
        tunnel = ngrok.connect(args.port)
        print(f"Публичная ссылка: {tunnel.public_url}")

    app.launch(
        allowed_paths=["/"],
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
