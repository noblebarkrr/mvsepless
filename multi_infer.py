import os
import subprocess
import shutil
import argparse
import subprocess
import tempfile
import gradio as gr


# audio-separator
from audio_separator.separator import Separator

# Modded Music-Source-Separation-Training
from inference import mvsep_offline

# Models list in inference
from model_list import models_data

# Model downloader
from infer_utils.download_models import download_model

# Renamer stems for audio-separator
from infer_utils.uvr_rename_stems import rename_stems

# Pre-edit config for Music-Source-Separation-Training
from infer_utils.preedit_config import conf_editor

# Add Vbach in app:
def add_vbach(vbach):
    if vbach:
        from vbach import conversion, url_download, zip_upload, files_upload
        with gr.TabItem("Voice to voice"):
            with gr.TabItem("Inference"):
                conversion()
            with gr.TabItem("Download models"):
                url_download()                            
                zip_upload()
                files_upload()



# MVSEPLESS NON-CLI FUNCTIONS



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
            label=f"Target instrument is {target_instrument}, for instrumental extraction, enable the 'Extract Instrumental' option.",
            choices=stems,
            value=None,
            interactive=False
        )
    elif extract_checked:
        return gr.CheckboxGroup(
            label="Instrumental extraction enabled. If one or more stems are selected, an 'inverted' stem is added.",
            choices=stems,
            value=None,
            interactive=True
        )
    else:
        return gr.CheckboxGroup(
            choices=stems,
            value=None,
            label="Available stems",
            interactive=bool(stems) 
        )

def on_extract_change(checked, model_type, model_name):
    print(f"Extract Instrumental: {checked}")
    return update_stems_ui(model_type, model_name, checked), checked


# Medley-Vox Wrapper
def medley_inference(input, output, model_dir, model_name, output_format, batch): 
    medley_infer = ("python", "-m", "models.medley_vox.svs.inference", "--inference_data_dir", str(input), "--results_save_dir", str(output), "--model_dir", str(model_dir), "--exp_name", str(model_name), "--use_overlapadd=ola", "--output_format", str(output_format), ('--batch' if batch else ''))
    subprocess.run(medley_infer, check=True) 
    return







# MVSEPLESS CLI FUNCTION FOR USING IN ANY PROJECTS



def audio_separation(input_dir, output_dir="", instrum=False, model_name="", model_type="", output_format='wav', use_tta=False, batch=False, template=None, selected_instruments=None, gradio=False, progress=gr.Progress(track_tqdm=True)):
    os.makedirs(output_dir, exist_ok=True)
    
    if gradio:
        
        output_dir = tempfile.mkdtemp(prefix="mvsepless_")
        progress(0, desc="Start separation...")
    # Separate audio in Music-Source-Separation-Training

    if model_type == "vr_arch" or model_type == "mdx_net":

            separator = Separator(
                    output_dir=output_dir,
                    output_format=output_format,
                    output_single_stem=(selected_instruments[0] if len(selected_instruments) == 1 else None),
                    vr_params={"batch_size": 1, "window_size": 512, "aggression": 100, "enable_tta": use_tta, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False},
                    mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": True}
            )
            if gradio:
                progress(0.2, desc="Loading model...")
            separator.load_model(model_filename=model_name)
            if batch:
                for filename in os.listdir(input_dir):
                    input_file = os.path.join(input_dir, filename)
                    if os.path.isfile(input_file):
                        output_names = rename_stems(input_file, template, model_name)
                        uvr_output = separator.separate(input_file, output_names)
            else:
                output_names = rename_stems(input_dir, template, model_name)
                if gradio:
                    progress(0.5, desc="Separating audio...")
                uvr_output = separator.separate(input_dir, output_names)


    elif model_type == "mel_band_roformer" or model_type == "bs_roformer" or model_type == "mdx23c" or model_type == "scnet" or model_type == "htdemucs":

        model_paths = "ckpts"
        config_url = models_data[model_type][model_name]["config_url"]

        checkpoint_url = models_data[model_type][model_name]["checkpoint_url"]
        if gradio:
            progress(0.2, desc="Loading model")
        conf, ckpt = download_model(model_paths, model_name, model_type, checkpoint_url, config_url)
   
        print(selected_instruments)
        
        if model_type != "htdemucs":
            conf_editor(conf)
        if gradio:
            progress(0.5, desc="Separating audio...")
        mvsep_offline(input_dir, output_dir, model_type, conf, ckpt, instrum, output_format, model_name, template, 0, use_tta=use_tta, batch=batch, selected_instruments=selected_instruments) 

    elif model_type == "medley_vox":

        model_paths = "ckpts"
        config_url = models_data[model_type][model_name]["config_url"]

        checkpoint_url = models_data[model_type][model_name]["checkpoint_url"]
        if gradio:
            progress(0.2, desc="Loading model")
        medley_vox_model_dir = download_model(model_paths, model_name, model_type, checkpoint_url, config_url)
        if gradio:
            progress(0.5, desc="Separating audio...")
        medley_inference(input_dir, output_dir, medley_vox_model_dir, model_name, output_format, batch)

    if gradio: # Show output results in Gradio
        audio_folder = output_dir
        audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith((".wav", ".mp3", ".flac"))][:7]
    
        results = []
        for i in range(7):
            visible = i < len(audio_files)
            results.append(gr.update(visible=visible, value=audio_files[i] if visible else None))
        return tuple(results)



# MVSEPLESS NON-CLI



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

def mvsepless_non_cli(vbach):

    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("Separate"):
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

                template = gr.Text(label="Template for output file", value="NAME_STEM_MODEL", visible=True)
                output = gr.Text(value="output_sep", visible=False)

                separate_btn = gr.Button("Separate", variant="primary", visible=True)

                stems = [gr.Audio(visible=(i == 0)) for i in range(7)]

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


            # Vbach NON-CLI

            add_vbach(vbach)
                    

def code_infer():
    parser = argparse.ArgumentParser(description="Multi-inference fo separate audio")
    parser.add_argument("-i", "--input", type=str, help="Input file/dir path")
    parser.add_argument("-o", "--output", default="", type=str, help="Output file/dir path")
    parser.add_argument("-inst", "--instrum", action='store_true', help="Extract instrumental/Inverse all selected stems")
    
    parser.add_argument("-mn", "--model_name", type=str, help="Short name of model")
    parser.add_argument("-mt", "--model_type", type=str, help="Model type")
    
    parser.add_argument("-of", "--output_format", type=str, choices=['mp3', 'wav', 'flac'], default='wav', help="Format export")
    parser.add_argument("-tta", "--use_tta", action='store_true', help="Use TTA")
    parser.add_argument("-b", "--batch", action='store_true', help="Batch separation")
    parser.add_argument("-tmpl", "--template", type=str, default='NAME_MODEL_STEM', help="Template name output files")
    parser.add_argument("--select", nargs='+', help="Select stems")
    parser.add_argument("-gr", "--gradio", action='store_true', help="Use Gradio")
    parser.add_argument("-grvc", "--gradiovbach", action='store_true', help="Use Gradio with Vbach")

    args = parser.parse_args()

    if args.model_name and not args.model_type:
        parser.error("При указании --model_name необходимо указать и --model_arch")

    if args.gradio or args.gradiovbach:

        vbach = args.gradiovbach
        with gr.Blocks(title="Music & Voice Separation", theme=theme) as demo:
            gr.HTML("<h1><center> MVSEPLESS </center></h1>")
            mvsepless_non_cli(vbach)
        demo.launch(share=True, allowed_paths=["/content"])
    else:

        audio_separation(
            input_dir=args.input,
            output_dir=args.output,
            instrum=args.instrum,
            model_name=args.model_name,
            model_type=args.model_type,
            output_format=args.output_format,
            use_tta=args.use_tta,
            batch=args.batch,
            template=args.template,
            selected_instruments=args.select
        )

if __name__ == "__main__":
    code_infer()