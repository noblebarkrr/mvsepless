import gradio as gr
from tabs.separate import separate_ui
from tabs.history_separation import history_separations
from tabs.conversion import conversion
from tabs.download_models import url_download, zip_upload, files_upload

theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Montserrat"), "Arial", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Roboto Mono"), "Courier New", "monospace"]
).set(
    # Только базовые параметры, которые работают во всех версиях:
    button_primary_background_fill="#3a7bd5",
    button_primary_background_fill_hover="#2c65c0",
    button_primary_text_color="#ffffff",
    input_background_fill="#ffffff",
    input_border_color="#d0d0d6",
    block_background_fill="#ffffff",
    border_color_primary="#d0d0d6"
)


with gr.Blocks(title="Разделение музыки и голоса", theme=theme) as demo:
    gr.HTML("<h1><center> MVSEPLESS </center></h1>")

    with gr.Tabs():
        with gr.TabItem("Разделение вокала"):
            separate_ui()
        with gr.TabItem("История"):
            history_separations()
        with gr.TabItem("Замена вокала"):
            conversion()
        with gr.Tab("Загрузка модели"):
            url_download()
            zip_upload()
            files_upload()

if __name__ == "__main__": 
    demo.launch(share=True, allowed_paths=["/content"])