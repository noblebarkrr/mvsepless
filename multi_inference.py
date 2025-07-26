import os
import time
import shutil
import sys
import gc
import argparse
import json
import subprocess
from datetime import datetime
from tabulate import tabulate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from model_list import models_data
from utils.preedit_config import conf_editor
from utils.download_models import download_model

MODELS_CACHE_DIR = os.path.join(SCRIPT_DIR, "separator", "models_cache")
MODEL_TYPES = ["mel_band_roformer", "bs_roformer", "mdx23c", "scnet", "htdemucs", "bandit", "bandit_v2", "vr", "mdx"]
OUTPUT_FORMATS = ["mp3", "wav", "flac", "ogg", "opus", "m4a", "aac", "aiff"]

class MVSEPLESS:
    def __init__(self):
        self.models_cache_dir = os.path.join(SCRIPT_DIR, "separator", "models_cache")
        self.model_types = MODEL_TYPES
        self.output_formats = OUTPUT_FORMATS

    def get_mt(self):
        return list(models_data.keys())
       
    def get_mn(self, model_type):
        return list(models_data[model_type].keys())
       
    def get_stems(self, model_type, model_name):
        stems = models_data[model_type][model_name]["stems"]
        return stems
    
    def get_tgt_inst(self, model_type, model_name):
        target_instrument = models_data[model_type][model_name]["target_instrument"]
        return target_instrument

    def display_models_info(self, filter: str = None):
        print("\nAvailable Models Information:")
        print("=" * 50)
        
        for model_type in models_data:
            print(f"\nModel Type: {model_type.upper()}")
            print("-" * 50)
            
            table_data = []
            headers = ["Model Name", "Stems", "Target Instrument", "Primary Stem"]
            
            for model_name in models_data[model_type]:
                model_info = models_data[model_type][model_name]
                
                if filter and filter not in model_info.get('stems', []):
                    continue
                    
                stems = "\n".join(model_info.get('stems', [])) if 'stems' in model_info else "N/A"
                target = model_info.get('target_instrument', "N/A")
                primary = model_info.get('primary_stem', "N/A")
                
                table_data.append([model_name, stems, target, primary])
            
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            print()
            
    def separator(
        self,
        input_file: str = None,
        output_dir: str = None,
        model_type: str = "mel_band_roformer",
        model_name: str = "Mel-Band-Roformer_Vocals_kimberley_jensen",
        ext_inst: bool = False,
        vr_aggr: int = 5,
        output_format: str = "wav",
        output_bitrate: str = "320k",
        template: str = "NAME_(STEM)_MODEL",
        call_method: str = "cli",
        selected_stems: list = None
    ):
        if selected_stems is None:
            selected_stems = []
            
        if not input_file:
            print("Please, input path to input file")
            return [("None", "/none/none.mp3")]
    
        if not os.path.exists(input_file):
            print("Input file not exist")
            return [("None", "/none/none.mp3")]
    
        if "STEM" not in template:
            template = template + "_STEM"
    
        print(f"Starting inference: {model_type}/{model_name}, bitrate={output_bitrate}, method={call_method}, stems={selected_stems}")
        os.makedirs(output_dir, exist_ok=True)
    
        if model_type in ["mel_band_roformer", "bs_roformer", "mdx23c", "scnet", "htdemucs", "bandit", "bandit_v2"]:
            try:
                info = models_data[model_type][model_name]
            except KeyError:
                print("Model not exist")
                return [("None", "/none/none.mp3")]
                
            conf, ckpt = download_model(self.models_cache_dir, model_name, model_type, 
                                      info["checkpoint_url"], info["config_url"])
            if model_type != "htdemucs":
                conf_editor(conf)
    
            if call_method == "cli":
                cmd = ["python", "-m", "separator.msst_separator", f'--input "{input_file}"', 
                      f'--store_dir "{output_dir}"', f'--model_type "{model_type}"', 
                      f'--model_name "{model_name}"', f'--config_path "{conf}"', 
                      f'--start_check_point "{ckpt}"', f'--output_format "{output_format}"', 
                      f'--output_bitrate "{output_bitrate}"', f'--template "{template}"', 
                      "--save_results_info"]
                if ext_inst:
                    cmd.append("--extract_instrumental")
                if selected_stems:
                    instruments = " ".join(f'"{s}"' for s in selected_stems)
                    cmd.append(f'--selected_instruments {instruments}')
                subprocess.run(" ".join(cmd), shell=True, check=True)
    
                results_path = os.path.join(output_dir, "results.json")
                if os.path.exists(results_path):
                    with open(results_path, encoding="utf-8") as f:
                        return json.load(f)
                return [("None", "/none/none.mp3")]
    
            elif call_method == "direct":
                from separator.msst_separator import mvsep_offline
                try:
                    return mvsep_offline(
                        input_path=input_file, store_dir=output_dir, model_type=model_type, 
                        config_path=conf, start_check_point=ckpt, extract_instrumental=ext_inst, 
                        output_format=output_format, output_bitrate=output_bitrate, 
                        model_name=model_name, template=template, selected_instruments=selected_stems
                    )
                except Exception as e:
                    print(e)
                    return [("None", "/none/none.mp3")]
    
        elif model_type in ["vr", "mdx"]:
            try:
                info = models_data[model_type][model_name]
            except KeyError:
                print("Model not exist")
                return [("None", "/none/none.mp3")]
                
            if model_type == "vr" and info.get("custom_vr", False):
                conf, ckpt = download_model(self.models_cache_dir, model_name, model_type, 
                                          info["checkpoint_url"], info["config_url"])
                primary_stem = info["primary_stem"]
    
                if call_method == "cli":
                    cmd = ["python", "-m", "separator.uvr_sep", "custom_vr", 
                          f'--input_file "{input_file}"', f'--ckpt_path "{ckpt}"', 
                          f'--config_path "{conf}"', f'--bitrate "{output_bitrate}"', 
                          f'--model_name "{model_name}"', f'--template "{template}"', 
                          f'--output_format "{output_format}"', f'--primary_stem "{primary_stem}"', 
                          f'--aggression {vr_aggr}', f'--output_dir "{output_dir}"']
                    if selected_stems:
                        instruments = " ".join(f'"{s}"' for s in selected_stems)
                        cmd.append(f'--selected_instruments {instruments}')
                    subprocess.run(" ".join(cmd), shell=True, check=True)
    
                    results_path = os.path.join(output_dir, "results.json")
                    if os.path.exists(results_path):
                        with open(results_path, encoding="utf-8") as f:
                            return json.load(f)
                    return [("None", "/none/none.mp3")]
    
                elif call_method == "direct":
                    from separator.uvr_sep import custom_vr_separate
                    try:
                        return custom_vr_separate(
                            input_file=input_file, ckpt_path=ckpt, config_path=conf, 
                            bitrate=output_bitrate, model_name=model_name, template=template, 
                            output_format=output_format, primary_stem=primary_stem, 
                            aggression=vr_aggr, output_dir=output_dir, 
                            selected_instruments=selected_stems
                        )
                    except Exception as e:
                        print(e)
                        return [("None", "/none/none.mp3")]
            else:
                if call_method == "cli":
                    cmd = ["python", "-m", "separator.uvr_sep", "uvr", 
                          f'--input_file "{input_file}"', f'--output_dir "{output_dir}"', 
                          f'--template "{template}"', f'--bitrate "{output_bitrate}"', 
                          f'--model_dir "{self.models_cache_dir}"', f'--model_type "{model_type}"', 
                          f'--model_name "{model_name}"', f'--output_format "{output_format}"', 
                          f'--aggression {vr_aggr}']
                    if selected_stems:
                        instruments = " ".join(f'"{s}"' for s in selected_stems)
                        cmd.append(f'--selected_instruments {instruments}')
                    subprocess.run(" ".join(cmd), shell=True, check=True)
    
                    results_path = os.path.join(output_dir, "results.json")
                    if os.path.exists(results_path):
                        with open(results_path, encoding="utf-8") as f:
                            return json.load(f)
                    return [("None", "/none/none.mp3")]
    
                elif call_method == "direct":
                    from separator.uvr_sep import non_custom_uvr_inference
                    try:
                        return non_custom_uvr_inference(
                            input_file=input_file, output_dir=output_dir, template=template, 
                            bitrate=output_bitrate, model_dir=self.models_cache_dir, 
                            model_type=model_type, model_name=model_name, 
                            output_format=output_format, aggression=vr_aggr, 
                            selected_instruments=selected_stems
                        )
                    except Exception as e:
                        print(e)
                        return [("None", "/none/none.mp3")]
    
        print("Unsupported model type")
        return [("None", "/none/none.mp3")]

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-inference for separation audio in Google Colab")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-command help')
    
    list_models = subparsers.add_parser('list', help='List of exist models')
    list_models.add_argument("-l_filter", "--list_filter", type=str, default=None, help="Show models in list only with specified stem")

    separate = subparsers.add_parser('separate', help='Separate I/O params')
    separate.add_argument("-i", "--input", type=str, required=True, help="Input file or directory")
    separate.add_argument("-o", "--output", type=str, required=True, help="Output directory")
    separate.add_argument("-mt", "--model_type", type=str, required=True, choices=MODEL_TYPES, help="Model type")
    separate.add_argument("-mn", "--model_name", type=str, required=True, help="Model name")
    separate.add_argument("-inst", "--instrumental", action='store_true', help="Extract instrumental")
    separate.add_argument("-stems", "--stems", nargs="+", help="Select output stems")
    separate.add_argument("-bitrate", "--bitrate", type=str, default="320k", help="Output bitrate")
    separate.add_argument("-of", "--format", type=str, default="mp3", help="Output format")
    separate.add_argument("-vr_aggr", "--vr_arch_aggressive", type=int, default=5, help="Aggression for VR ARCH models")
    separate.add_argument('--template', type=str, default='NAME_STEM', help='Template naming of output files')
    separate.add_argument("-l_out", "--list_output", action='store_true', help="Show list output files")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    mvsepless = MVSEPLESS()
    
    if args.command == 'list':
        mvsepless.display_models_info(args.list_filter)
        
    elif args.command == 'separate':
        if os.path.isfile(args.input):
            results = mvsepless.separator(
                input_file=args.input,
                output_dir=args.output,
                model_type=args.model_type,
                model_name=args.model_name,
                ext_inst=args.instrumental,
                vr_aggr=args.vr_arch_aggressive,
                output_format=args.format,
                output_bitrate=args.bitrate,
                template=args.template,
                call_method="cli",
                selected_stems=args.stems
            )
            if args.list_output:
                print("Results\n")
                for stem, path in results:
                    print(f"Stem - {stem}\nPath - {path}\n")
                    
        elif os.path.isdir(args.input):
            batch_results = []
            for file in os.listdir(args.input):
                abs_path_file = os.path.join(args.input, file)
                if os.path.isfile(abs_path_file):
                    base_name = os.path.splitext(os.path.basename(abs_path_file))[0]
                    output_subdir = os.path.join(args.output, base_name)
                    
                    results = mvsepless.separator(
                        input_file=abs_path_file,
                        output_dir=output_subdir,
                        model_type=args.model_type,
                        model_name=args.model_name,
                        ext_inst=args.instrumental,
                        vr_aggr=args.vr_arch_aggressive,
                        output_format=args.format,
                        output_bitrate=args.bitrate,
                        template=args.template,
                        call_method="cli",
                        selected_stems=args.stems
                    )
                    batch_results.append((base_name, results))
                    
            if args.list_output:
                print("Results\n")
                for name, stems in batch_results:
                    print(f"Name - {name}")
                    for stem, path in stems:
                        print(f"  Stem - {stem}\n  Path - {path}\n")

