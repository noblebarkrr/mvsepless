import gradio as gr
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root) 

from model_list import models_data
from multi_infer_test import audio_separation

def update_model_names(model_type):
    if model_type in models_data:
        return gr.Dropdown(choices=list(models_data[model_type].keys())), gr.Dropdown(choices=[], interactive=False)
    return gr.Dropdown(choices=[]), gr.Dropdown(choices=[], interactive=True)

def get_stems_from_model(model_type, model_name):
    if not model_type or not model_name:
        return []
    
    model_info = models_data.get(model_type, {}).get(model_name, {})
    return model_info.get("stems", [])

def update_stems_ui(model_type, model_name, extract_checked):

    stems = get_stems_from_model(model_type, model_name)
    model_info = models_data.get(model_type, {}).get(model_name, {})
    target_instrument = model_info.get("target_instrument", "No")
    
    if target_instrument != "No":
        return gr.CheckboxGroup(
            label=f"Target instrument is {target_instrument}, stems selection ignored",
            choices=stems,
            interactive=False
        )
    elif extract_checked:
        return gr.CheckboxGroup(
            label="Instrumental extraction enabled",
            choices=stems,
            interactive=True
        )
    else:
        return gr.CheckboxGroup(
            choices=stems,
            label="Available stems",
            interactive=bool(stems) 
        )

def on_extract_change(checked, model_type, model_name):
    print(f"Extract Instrumental: {checked}")
    return update_stems_ui(model_type, model_name, checked), checked

def mvsep_sep():
    with gr.Blocks() as demo:
        extract_state = gr.State(False)
        false_state = gr.State(False)
        true_state = gr.State(True)

        with gr.Row():
            input_file = gr.Audio(label="Upload audio", type="filepath", visible=True)

        with gr.Row():
            model_type_dropdown = gr.Dropdown(
                choices=list(models_data.keys()),
                label="Select Model Type",
                interactive=True,
                filterable=False
            )
            model_name_dropdown = gr.Dropdown(
                choices=list(models_data["mel_band_roformer"].keys()),
                label="Select Model Name",
                value="aname_4_stems_xl",
                interactive=True,
                filterable=False
            )
        
        # Инициализируем стемы из начальной модели
        initial_stems = get_stems_from_model("mel_band_roformer", "aname_4_stems_xl")
        stems_checkbox = gr.CheckboxGroup(
            label="Available stems",
            interactive=True,
            choices=initial_stems
        )

        extract_checkbox = gr.Checkbox(
            label="Extract Instrumental",
            value=False
        )

        with gr.Row():
            output_format = gr.Radio(
                label="Format export",
                choices=["wav", "mp3", "flac"],
                value="flac",
                visible=True
            )

        template = gr.Text(value="NAME_STEM_MODEL", visible=False)
        output = gr.Text(value="output_sep", visible=False)

        separate_btn = gr.Button("Separate", variant="primary", visible=True)

        stems = [gr.Audio(visible=False) for _ in range(7)]

        separate_btn.click(
            fn=audio_separation,
            inputs=[input_file, output, extract_state, model_name_dropdown, model_type_dropdown, output_format, false_state, false_state, template, stems_checkbox, true_state],
            outputs=[*stems]
        )

        stems_checkbox.change(
            lambda x: print(f"Stems selected: {x}"),
            inputs=stems_checkbox,
            outputs=None
        )

        extract_checkbox.change(
            fn=on_extract_change,
            inputs=[extract_checkbox, model_type_dropdown, model_name_dropdown],
            outputs=[stems_checkbox, extract_state]
        )

        model_type_dropdown.change(
            update_model_names,
            inputs=model_type_dropdown,
            outputs=[model_name_dropdown, stems_checkbox]
        )

        model_name_dropdown.change(
            lambda model_type, model_name, extract_checked: update_stems_ui(model_type, model_name, extract_checked),
            inputs=[model_type_dropdown, model_name_dropdown, extract_checkbox],
            outputs=stems_checkbox
        )