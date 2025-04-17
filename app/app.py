import gradio as gr
from tabs.test import mvsep_sep
from tabs.ensemble import pre_ensem

from tabs.conversion import conversion
from tabs.download_models import url_download, zip_upload, files_upload

theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Montserrat"), "Arial", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Roboto Mono"), "Courier New", "monospace"]
).set(

    button_primary_background_fill="#3a7bd5",
    button_primary_background_fill_hover="#2c65c0",
    button_primary_text_color="#ffffff",
    input_background_fill="#ffffff",
    input_border_color="#d0d0d6",
    block_background_fill="#ffffff",
    border_color_primary="#d0d0d6"
)


with gr.Blocks(title="Music & Voice Separation", theme=theme) as demo:
    gr.HTML("<h1><center> MVSEPLESS </center></h1>")

    with gr.Tabs():
        with gr.TabItem("Separate"):
            mvsep_sep()
        with gr.TabItem("Ensemble"):
            pre_ensem()
        with gr.TabItem("Conversion"):
            with gr.TabItem("Inference"):
                conversion()
            with gr.TabItem("Download models"):
                url_download()
                zip_upload()
                files_upload()
if __name__ == "__main__": 
    demo.launch(share=True, allowed_paths=["/content"])