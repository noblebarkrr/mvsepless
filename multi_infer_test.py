import os
import shutil
import argparse
import subprocess
import gradio as gr

from audio_separator.separator import Separator
from inference import mvsep_offline
from model_list import models_data
from infer_utils.download_models import download_model
from infer_utils.uvr_rename_stems import rename_stems
from infer_utils.preedit_config import conf_editor

def audio_separation(input_dir, output_dir="", instrum=False, model_name="", model_type="", output_format='wav', use_tta=False, batch=False, template=None, selected_instruments=None, gradio=False):
    os.makedirs(output_dir, exist_ok=True)
    
    if gradio:
        print("Inference loaded")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
    # Separate audio in Music-Source-Separation-Training

    if model_type == "vr_arch" or model_type == "mdx_net":

            separator = Separator(
                    output_dir=output_dir,
                    output_format=output_format,
                    output_single_stem=(selected_instruments[0] if len(selected_instruments) == 1 else None),
                    vr_params={"batch_size": 1, "window_size": 512, "aggression": 100, "enable_tta": use_tta, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False},
                    mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": True}
            )
            separator.load_model(model_filename=model_name)
            if batch:
                for filename in os.listdir(input_dir):
                    input_file = os.path.join(input_dir, filename)
                    if os.path.isfile(input_file):
                        output_names = rename_stems(input_file, template, model_name)
                        uvr_output = separator.separate(input_file, output_names)
            else:
                output_names = rename_stems(input_dir, template, model_name)
                uvr_output = separator.separate(input_dir, output_names)


    elif model_type == "mel_band_roformer" or model_type == "bs_roformer" or model_type == "mdx23c" or model_type == "scnet":

        model_paths = "ckpts"
        config_url = models_data[model_type][model_name]["config_url"]

        checkpoint_url = models_data[model_type][model_name]["checkpoint_url"]
    
        conf, ckpt = download_model(model_paths, model_name, model_type, checkpoint_url, config_url)
   
        print(selected_instruments)

        conf_editor(conf)

        mvsep_offline(input_dir, output_dir, model_type, conf, ckpt, instrum, output_format, model_name, template, 0, use_tta=use_tta, batch=batch, selected_instruments=selected_instruments) 

    if gradio: # Show output results in Gradio
        audio_folder = output_dir
        audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith((".wav", ".mp3", ".flac"))][:7]
    
        results = []
        for i in range(7):
            visible = i < len(audio_files)
            results.append(gr.update(visible=visible, value=audio_files[i] if visible else None))
        return tuple(results)



def code_infer():
    parser = argparse.ArgumentParser(description="Multi-inference fo separate audio")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input file/dir path")
    parser.add_argument("-o", "--output", default="", type=str, help="Output file/dir path")
    parser.add_argument("-inst", "--instrum", action='store_true', help="Extract instrumental/Inverse all selected stems")
    
    parser.add_argument("-mn", "--model_name", type=str, help="Short name of model")
    parser.add_argument("-mt", "--model_type", type=str, help="Model type")
    
    parser.add_argument("-of", "--output_format", type=str, choices=['mp3', 'wav', 'flac'], default='wav', help="Format export")
    parser.add_argument("-tta", "--use_tta", action='store_true', help="Use TTA")
    parser.add_argument("-b", "--batch", action='store_true', help="Batch separation")
    parser.add_argument("-tmpl", "--template", type=str, default='NAME_MODEL_STEM', help="Template name output files")
    parser.add_argument("--select", nargs='+', help="Select stems")

    args = parser.parse_args()

    if args.model_name and not args.model_type:
        parser.error("При указании --model_name необходимо указать и --model_arch")

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