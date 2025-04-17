import gradio as gr
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root) 

from ensemble_infer import ensemble_infer

def get_preset_files():
    # Путь к папке с пресетами (измените на ваш реальный путь)
    presets_dir = os.path.join(project_root, "ensemble_presets")
    
    # Получаем список файлов в папке
    if os.path.exists(presets_dir):
        preset_files = [f for f in os.listdir(presets_dir) if f.endswith('.json') or f.endswith('.py')]
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
        
        separate_btn = gr.Button("Separate", variant="primary", visible=True)

        result = gr.Audio(type="filepath", interactive=False, visible=True)
        invert = gr.Audio(type="filepath", interactive=False, visible=True)

        separate_btn.click(
            fn=ensemble_infer,
            inputs=[input_file, preset],
            outputs=[result, invert]
        )
    
    return demo