import gradio as gr
import os
import subprocess
from datetime import datetime
from rvc.scripts.voice_conversion import voice_pipeline

rvc_models_dir = "voice_models"

def get_models_list():
    """Получение списка моделей."""
    models = []
    if os.path.exists(rvc_models_dir):
        models = [d for d in os.listdir(rvc_models_dir) if os.path.isdir(os.path.join(rvc_models_dir, d))]
    return models

def conversion():
    with gr.Column() as conversion_group:
        file_input = gr.Audio(label="Upload audio", type="filepath")
        voicemodel_name = gr.Dropdown(
            choices=list(get_models_list()), 
            label="Model name", 
            value="senko",
            interactive=True,
            filterable=False
        )
        refresh_btn = gr.Button("Refresh")
        pitch_vocal = gr.Slider(-48, 48, value=0, step=12, label="Pitch", interactive=True)
        method_pitch = gr.Dropdown(
            label="F0 method", 
            choices=["rmvpe+", "mangio-crepe", "fcpe"], 
            value="rmvpe+",
            interactive=True,
            filterable=False
        )
        hop_length = gr.Slider(0, 255, value=73, step=1, label="Hop length (mangio-crepe only)", interactive=True)
        index_rate = gr.Slider(0, 1, value=1, step=0.05, label="Index rate", interactive=True)
        filter_radius = gr.Slider(0, 7, value=7, step=1, label="Filter radius", interactive=True)
        rms = gr.Slider(0, 1, value=0, step=0.1, label="RMS", interactive=True)
        protect = gr.Slider(0, 0.5, value=0.35, step=0.05, label="Protect", interactive=True)
        f0_max = gr.Slider(1100, 2700, value=1100, step=50, label="F0 Max", interactive=True)
        output_format_rvc = gr.Dropdown(
            label="Format export", 
            choices=["wav", "mp3", "flac"], 
            interactive=True,
            filterable=False
        )
        # Add hidden components for constant values
        constant_value = gr.Number(value=50, visible=False)
        output_dir = gr.Text(value="/content/voice_output", visible=False)
        convert_btn = gr.Button("Convert", variant="primary")
        
    with gr.Column() as output_voice_group:
        converted_voice = gr.Audio(type="filepath", interactive=False, visible=True)
        output_filename = gr.Text(visible=False)

    refresh_btn.click(
        fn=lambda: gr.update(choices=get_models_list()),
        outputs=voicemodel_name
    )
    
    def create_output_filename(model, method, pitch):
        return f"converted_voice_{model}_{method}_{pitch}"
    
    convert_btn.click(
        fn=create_output_filename,
        inputs=[voicemodel_name, method_pitch, pitch_vocal],
        outputs=output_filename
    ).then(
        fn=voice_pipeline,
        inputs=[file_input, voicemodel_name, pitch_vocal, index_rate, 
                filter_radius, rms, method_pitch, hop_length, 
                protect, output_format_rvc, constant_value, f0_max, 
                output_dir, output_filename],
        outputs=converted_voice
    )