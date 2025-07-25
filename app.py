import os
import sys
import argparse
from datetime import datetime
import gradio as gr
from pyngrok import ngrok
from multi_inference import MVSEPLESS, OUTPUT_FORMATS
from assets.translations import TRANSLATIONS, TRANSLATIONS_STEMS

def restart_ui():
    python = sys.executable
    os.execl(python, python, *sys.argv)

OUTPUT_DIR = os.path.join(os.getcwd(), "output")
plugins_dir = os.path.join(os.getcwd(), "plugins")
os.makedirs(plugins_dir, exist_ok=True)

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

def sep_wrapper(a, b, c, d, e, f, g, h):
    results = mvsepless.separator(input_file=a, output_dir=gen_out_dir(), model_type=b, model_name=c, ext_inst=d, vr_aggr=e, output_format=f, output_bitrate=f'{g}k', call_method="cli", selected_stems=h)
    stems = []
    if results:
        for i, (stem, output_file) in enumerate(results[:20]):
            stems.append(gr.update(
                visible=True,
                label=t_stem(stem),
                value=output_file
            ))
    
    while len(stems) < 20:
        stems.append(gr.update(visible=False, label=None, value=None))
        
    return tuple(stems)


theme = gr.themes.Default(
        primary_hue="violet",
        secondary_hue="cyan",
        neutral_hue="blue",
        spacing_size="sm",
        font=[gr.themes.GoogleFont("Tektur"), 'ui-sans-serif', 'system-ui', 'sans-serif'],
    ).set(
        body_text_color='*neutral_950',
        body_text_color_subdued='*neutral_500',
        background_fill_primary='*neutral_200',
        background_fill_primary_dark='*neutral_800',
        border_color_accent='*primary_950',
        border_color_accent_dark='*neutral_700',
        border_color_accent_subdued='*primary_500',
        border_color_primary='*primary_800',
        border_color_primary_dark='*neutral_400',
        color_accent_soft='*primary_100',
        color_accent_soft_dark='*neutral_800',
        link_text_color='*secondary_700',
        link_text_color_active='*secondary_700',
        link_text_color_hover='*secondary_800',
        link_text_color_visited='*secondary_600',
        link_text_color_visited_dark='*secondary_700',
        block_background_fill='*background_fill_secondary',
        block_background_fill_dark='*neutral_950',
        block_label_background_fill='*secondary_400',
        block_label_text_color='*neutral_800',
        panel_background_fill='*background_fill_primary',
        checkbox_background_color='*background_fill_secondary',
        checkbox_label_background_fill_dark='*neutral_900',
        input_background_fill_dark='*neutral_900',
        input_background_fill_focus='*neutral_100',
        input_background_fill_focus_dark='*neutral_950',
        button_small_radius='*radius_sm',
        button_secondary_background_fill='*neutral_400',
        button_secondary_background_fill_dark='*neutral_500',
        button_secondary_background_fill_hover_dark='*neutral_950'
    )


def create_app():
    with gr.Tab(t("separation")):
        with gr.Row():
            with gr.Column():
                input_audio = gr.Audio(label=t("select_file"), interactive=True, type="filepath")
                input_audio_path = gr.Textbox(label=t("audio_path"), info=t("audio_path_info"), interactive=True)
            with gr.Column():
                with gr.Row():
                    model_type = gr.Dropdown(label=t("model_type"), choices=mvsepless.get_mt(), value=mvsepless.get_mt()[0], interactive=True)
                    model_name = gr.Dropdown(label=t("model_name"), choices=mvsepless.get_mn(mvsepless.get_mt()[0]), value=mvsepless.get_mn(mvsepless.get_mt()[0])[0], interactive=True)
                target_instrument = gr.Textbox(label=t("target_instrument"), value=mvsepless.get_tgt_inst(mvsepless.get_mt()[0], mvsepless.get_mn(mvsepless.get_mt()[0])[0]), interactive=False)
                vr_aggr = gr.Slider(0, 100, step=1, label=t("vr_aggressiveness"), visible=False, value=5, interactive=True)
                extract_instrumental = gr.Checkbox(label=t("extract_instrumental"), value=True, interactive=True)
                stems_list = gr.CheckboxGroup(label=t("stems_list"), info=t("stems_info", target_instrument="vocals"), choices=mvsepless.get_stems(mvsepless.get_mt()[0], mvsepless.get_mn(mvsepless.get_mt()[0])[0]), value=None, interactive=False)
                with gr.Row():
                    output_format, output_bitrate = gr.Dropdown(label=t("output_format"), choices=OUTPUT_FORMATS, value="mp3", interactive=True), gr.Slider(32, 320, step=1, label=t("bitrate"), value=320, interactive=True)
                separate_btn = gr.Button(t("separate_btn"), variant="primary", interactive=True)
        download_via_zip_btn = gr.DownloadButton(label="Download via zip", visible=False, interactive=True)
        output_stems = []
        for _ in range(10):
            with gr.Row():
                audio1 = gr.Audio(visible=False, interactive=False, type="filepath", show_download_button=True)
                audio2 = gr.Audio(visible=False, interactive=False, type="filepath", show_download_button=True)
                output_stems.extend([audio1, audio2])

    with gr.Tab(t("plugins")):
        plugins = [] 

        with gr.Tab(t('upload')):
            with gr.Blocks():
                upload_plugin_files = gr.Files(label=t('upload'), file_types=[".py"], interactive=True)
                upload_btn = gr.Button(t('upload_btn'), interactive=True)

        if os.path.exists(plugins_dir) and os.path.isdir(plugins_dir):
          try:
            for filename in os.listdir(plugins_dir):
                if filename.endswith(".py") and filename != "__init__.py":
                    file_path = os.path.join(plugins_dir, filename)
                    module_name = filename[:-3]
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
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
                        plugin_name = name_func() if name_func is not None else module_name
                        plugins.append((plugin_name, plugin_func))

          except Exception as e:
            print(e)

        for name, func in plugins:
            try:
                print(t("loading_plugin", name=name))
                with gr.Tab(name):
                    func(lang)
            except Exception as e:
                print(t("error_loading_plugin", e=e))
                pass

    input_audio.upload(fn=(lambda x: gr.update(value=x)), inputs=input_audio, outputs=input_audio_path)
    model_type.change(fn=(lambda x: gr.update(choices=mvsepless.get_mn(x), value=mvsepless.get_mn(x)[0])), inputs=model_type, outputs=model_name).then(fn=(lambda x: (gr.update(visible=False if x in ["vr", "mdx"] else True), gr.update(visible=True if x == "vr" else False))), inputs=model_type, outputs=[extract_instrumental, vr_aggr])
    model_name.change(fn=(lambda x, y: gr.update(choices=mvsepless.get_stems(x, y), value=None)), inputs=[model_type, model_name], outputs=stems_list).then(fn=(lambda x, y: (gr.update(interactive=True if mvsepless.get_tgt_inst(x, y) == None else None, info=t("stems_info", target_instrument=mvsepless.get_tgt_inst(x, y)) if mvsepless.get_tgt_inst(x, y) is not None else t("stems_info2")), gr.update(value=mvsepless.get_tgt_inst(x, y)), gr.update(value=True if mvsepless.get_tgt_inst(x, y) is not None else False))), inputs=[model_type, model_name], outputs=[stems_list, target_instrument, extract_instrumental])
    separate_btn.click(fn=sep_wrapper, inputs=[input_audio_path, model_type, model_name, extract_instrumental, vr_aggr, output_format, output_bitrate, stems_list], outputs=output_stems, show_progress_on=input_audio)

    upload_btn.click(fn=upload_plugin_list, inputs=upload_plugin_files)

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
    if font and os.path.isfile(font) and font.lower().endswith((".ttf", ".otf", ".woff", ".eot")):
        with open(font, "rb") as font_file:
            base64_font = base64.b64encode(font_file.read()).decode("utf-8")
        css += f"""
        @font-face {{
            font-family: 'CustomFont';
            src: url(data:font/truetype;charset=utf-8;base64,{base64_font}) format('truetype');
        }}
        body, .gradio-container {{ font-family: 'CustomFont', sans-serif !important; }}
        """

    with gr.Blocks(theme=theme, css=css) as app:
        create_app()

    if args.ngrok_token:
        ngrok.set_auth_token(args.ngrok_token)
        ngrok.kill()
        tunnel = ngrok.connect(CONFIG["settings"]["port"])
        print(f"Публичная ссылка: {tunnel.public_url}")

    app.launch(allowed_paths=["/"], server_port=args.port, share=args.share)
