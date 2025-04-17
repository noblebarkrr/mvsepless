import os
import importlib
import glob
import shutil

from multi_infer_test import audio_separation
from ensemble_test import ensemble_audio_files
from infer_utils.audio_processing.invert import invert_and_overlay_wav

def ensemble_infer(input_file, name_preset):
    if os.path.exists("output"):
        shutil.rmtree("output")
    name_preset = os.path.splitext(name_preset)[0]
    
    base_name = os.path.splitext(name_preset)[0]
    output_file = f"output/{base_name}.wav"
    inverted_file = f"output/inverted.wav"

    module = importlib.import_module(f"ensemble_presets.{name_preset}")
    preset = module.preset
    output_dir = os.path.join("output", base_name)

    temp_files = os.path.join(output_dir, "temp")
    target_stem, weights, list_models, type, ext_inst = preset()
    for mt, mn in list_models:
        audio_separation(
            input_dir=input_file,
            output_dir=output_dir,
            instrum=ext_inst,
            model_name=mn,
            model_type=mt,
            output_format='wav',
            use_tta=False,
            batch=False,
            template=f"NAME_MODEL_{target_stem}",
            selected_instruments=target_stem,
            gradio=False
        )
    
    ensemble_audio_files(files=(glob.glob(f"{output_dir}/*_{target_stem}.wav")), output=output_file, ensemble_type=type, weights=weights)    

    invert_and_overlay_wav(output_file, input_file, inverted_file)

    return output_file, inverted_file