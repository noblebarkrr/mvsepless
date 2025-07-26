import os
import argparse
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
import json

from audio_separator.separator import Separator
from renamer_stems import audio_separator_rename_stems

def give_vr_params(file):
    path, filename = os.path.split(file)
    name_without_ext = os.path.splitext(filename)[0]
    vr_param = os.path.join(path, name_without_ext)
    return vr_param

def custom_vr_separate(
    input_file, 
    ckpt_path, 
    config_path,
    bitrate,
    model_name,
    template,
    output_format,
    primary_stem="Vocals", 
    aggression=5,
    output_dir="./",
    selected_instruments=[]
):
    
    separator = Separator(
        output_dir=output_dir,
        output_bitrate=bitrate,
        use_soundfile=False,
        output_format=output_format,
        output_single_stem=(selected_instruments[0] if len(selected_instruments) == 1 else None)
    )
    output_names = audio_separator_rename_stems(input_file, template, model_name)

    separator.load_custom_vr_model(
        model_path=ckpt_path,
        config_path=config_path,
        params={"primary_stem": primary_stem, "vr_model_param" : give_vr_params(config_path), "window_size" : 512, "aggression": aggression},
    )

    output_files = separator.separate(input_file, output_names)

    return output_files

def give_full_model_name(model_type, model_name):
    if model_type == "mdx":
        return f"{model_name}.onnx"
    elif model_type == "vr":
        return f"{model_name}.pth"


def non_custom_uvr_inference(input_file, output_dir, template, bitrate, model_dir, model_type, model_name, output_format, aggression, selected_instruments=[]):
       
    separator = Separator(
        output_dir=output_dir,
        output_bitrate=bitrate,
        model_file_dir=model_dir,
        use_soundfile=False,
        output_format=output_format,
        output_single_stem=(selected_instruments[0] if len(selected_instruments) == 1 else None),
        vr_params={"batch_size": 1, "window_size": 512, "aggression": aggression, "enable_tta": False, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False},
        mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": True}
        )
    separator.load_model(model_filename=give_full_model_name(model_type, model_name))
    
    output_names = audio_separator_rename_stems(input_file, template, model_name)
    
    output_files = separator.separate(input_file, output_names)
    
    return output_files











def main():
    parser = argparse.ArgumentParser(description='Audio separation tool')
    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-command help')

    # Парсер для custom VR separation
    custom_parser = subparsers.add_parser('custom_vr', help='Custom VR model separation')
    custom_parser.add_argument('--input_file', required=True, help='Input audio file path')
    custom_parser.add_argument('--ckpt_path', required=True, help='Path to model checkpoint (.pth file)')
    custom_parser.add_argument('--config_path', required=True, help='Path to model config file')
    custom_parser.add_argument('--bitrate', type=str, default="320k", help='Output bitrate')
    custom_parser.add_argument('--model_name', required=True, help='Name of the model')
    custom_parser.add_argument('--template', default="{track_name}_{stem}_{model_name}", help='Output filename template')
    custom_parser.add_argument('--output_format', default="mp3", help='Output audio format')
    custom_parser.add_argument('--primary_stem', default="Vocals", help='Primary stem to separate')
    custom_parser.add_argument('--aggression', type=int, default=5, help='Separation aggression level')
    custom_parser.add_argument('--output_dir', default="./", help='Output directory')
    custom_parser.add_argument('--selected_instruments', nargs='*', default=[], help='List of instruments to separate')

    # Парсер для non-custom UVR separation
    uvr_parser = subparsers.add_parser('uvr', help='Non-custom UVR separation')
    uvr_parser.add_argument('--input_file', required=True, help='Input audio file path')
    uvr_parser.add_argument('--output_dir', default="./", help='Output directory')
    uvr_parser.add_argument('--template', default="{track_name}_{stem}_{model_name}", help='Output filename template')
    uvr_parser.add_argument('--bitrate', type=str, default="320k", help='Output bitrate')
    uvr_parser.add_argument('--model_dir', required=True, help='Directory containing model files')
    uvr_parser.add_argument('--model_type', required=True, choices=['mdx', 'vr'], help='Model type (mdx or vr)')
    uvr_parser.add_argument('--model_name', required=True, help='Name of the model')
    uvr_parser.add_argument('--output_format', default="mp3", help='Output audio format')
    uvr_parser.add_argument('--aggression', type=int, default=5, help='Separation aggression level (for VR models)')
    uvr_parser.add_argument('--selected_instruments', nargs='*', default=[], help='List of instruments to separate')

    args = parser.parse_args()

    if args.command == 'custom_vr':
        # Запуск custom VR separation
        results = custom_vr_separate(
            input_file=args.input_file,
            ckpt_path=args.ckpt_path,
            config_path=args.config_path,
            bitrate=args.bitrate,
            model_name=args.model_name,
            template=args.template,
            output_format=args.output_format,
            primary_stem=args.primary_stem,
            aggression=args.aggression,
            output_dir=args.output_dir,
            selected_instruments=args.selected_instruments
        )
        with open((os.path.join(args.output_dir, "results.json")), 'w') as f:
            json.dump(results, f)

    elif args.command == 'uvr':
        # Запуск non-custom UVR separation
        results = non_custom_uvr_inference(
            input_file=args.input_file,
            output_dir=args.output_dir,
            template=args.template,
            bitrate=args.bitrate,
            model_dir=args.model_dir,
            model_type=args.model_type,
            model_name=args.model_name,
            output_format=args.output_format,
            aggression=args.aggression,
            selected_instruments=args.selected_instruments
        )
        with open((os.path.join(args.output_dir, "results.json")), 'w') as f:
            json.dump(results, f)

if __name__ == "__main__":
    main()
