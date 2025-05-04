import os
import yaml
import shutil
import tempfile
import gradio as gr

from multi_infer import audio_separation, theme
from ensem import ensemble_audio_files
from infer_utils.audio_processing.invert import invert_and_overlay_wav


# Import options from preset


def import_preset(preset_file):

    with open(preset_file, 'r') as file:
        preset_data = yaml.safe_load(file)

    weights = preset_data['preset']['weights']
    list_models = preset_data['preset']['list_models']
    type = preset_data['preset']['type']
    ext_inst = preset_data['preset']['ext_inst']

    return weights, list_models, type, ext_inst


# Ensemble Inference


def ensem_infer(input_file, preset_file, output_dir):

    temp_files = tempfile.mkdtemp(prefix="ensem_temp_")
    os.makedirs(output_dir, exist_ok=True)

    weights, list_models, type, ext_inst = import_preset(preset_file)
    for mt, mn, slctd_stms in list_models:
        audio_separation(
            input_dir=input_file,
            output_dir=temp_files,
            instrum=ext_inst,
            model_name=mn,
            model_type=mt,
            output_format='wav',
            use_tta=False,
            batch=False,
            template=f"MODEL_STEM",
            selected_instruments=slctd_stms,
            gradio=False
        )

    output_file = os.path.join(output_dir, "output")

    inverted_file = os.path.join(output_dir, "inverted")

    temp_ensem_files = [os.path.abspath(os.path.join(temp_files, f)) 
         for f in os.listdir(temp_files) 
         if os.path.isfile(os.path.join(temp_files, f))]

    ensemble_audio_files(files=temp_ensem_files, output=f"{output_file}.wav", ensemble_type=type, weights=weights)

    invert_and_overlay_wav(f"{output_file}.wav", input_file, f"{inverted_file}.wav")

    return f"{output_file}.wav", f"{inverted_file}.wav"



# Ensemble NON-CLI



def get_preset_files():
    # Путь к папке с пресетами (измените на ваш реальный путь)
    presets_dir = "ensem_presets"
    
    # Получаем список файлов в папке с полными путями
    if os.path.exists(presets_dir):
        preset_files = [
            os.path.join(presets_dir, f) 
            for f in os.listdir(presets_dir) 
            if f.endswith('.yaml') or f.endswith('.txt')
        ]
        return preset_files
    return []

def pre_ensem():

    preset_files = get_preset_files()
    
    with gr.Blocks() as demo:
        with gr.Row():
            input_file = gr.Audio(label="Upload audio", type="filepath", visible=True)
        
        # Добавляем выпадающий список для выбора пресета
        preset = gr.Dropdown(
            label="Select preset",
            choices=preset_files,
            value=preset_files[0] if preset_files else None,
            visible=bool(preset_files)
        )
        output = gr.Text(value="ensem_output", visible=False)

        separate_btn = gr.Button("Separate", variant="primary", visible=True)

        result = gr.Audio(type="filepath", interactive=False, visible=True)
        invert = gr.Audio(type="filepath", interactive=False, visible=True)

        separate_btn.click(
            fn=ensem_infer,
            inputs=[input_file, preset, output],
            outputs=[result, invert]
        )

if __name__ == "__main__": 
    with gr.Blocks(title="Music & Voice Separation", theme=theme) as demo:
        gr.HTML("<h1><center> Ensemble Inference </center></h1>")

        pre_ensem()
        
    demo.launch(share=True, allowed_paths=["/content"])