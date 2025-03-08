import os
import argparse
import subprocess
from inference import mvsep_offline
from config import conf_editor


def code_infer():
    parser = argparse.ArgumentParser(description="Обработка входных и выходных директорий и кода модели.")
    parser.add_argument("-i", "--input", type=str, help="folder with mixtures to process")
    parser.add_argument("-o", "--output", default="", type=str, help="path to store results")
    parser.add_argument("-inst", "--instrum", action='store_true', help="invert vocals")
    parser.add_argument("-mcode", "-mc", "--modelcode", dest='modelcode', type=int, required=True, help="Код модели")
    parser.add_argument("-of", "--output_format", type=str, choices=['mp3', 'wav', 'flac'], default='wav', help="Format of output files")
    parser.add_argument("-tta", "--use_tta", action='store_true', help="Flag adds test time augmentation during inference (polarity and channel inverse). While this triples the runtime, it reduces noise and slightly improves prediction quality.")
    args = parser.parse_args()

    model_dir = f"model/{args.modelcode}"
    os.makedirs(model_dir, exist_ok=True)
    
    from models_list import get_model_config
    config = get_model_config(args.modelcode)

    if config:
        model_name = config["model_name"]
        arch = config["arch"]
        ckpt_url = config["ckpt_url"]
        conf_url = config["conf_url"]

        print(f"Model Name: {model_name}")
        print(f"Architecture: {arch}")
        print(f"Checkpoint URL: {ckpt_url}")
        print(f"Config URL: {conf_url}")
    else:
        print(f"Model with code {model_code} not found.")

    if arch == "mel_band_roformer" or arch == "bs_roformer" or arch == "mdx23c" or arch == "scnet":
        conf = f"{model_dir}/config.yaml"
        ckpt = f"{model_dir}/model.ckpt"
        infer = "mss"

    elif arch == "medleyvox":
        conf = f"{model_dir}/vocals.json"
        ckpt = f"{model_dir}/vocals.pth"
        infer = "medley_vox"

    for local_path, url_model in [(ckpt, ckpt_url), (conf, conf_url)]:
        download_ckpt = ("wget", "-O", str(local_path), str(url_model))
        subprocess.run(download_ckpt, check=True)

    if infer == "mss":
        conf_editor(conf)
        output_format = args.output_format
        mvsep_offline(
            args.input,
            args.output,
            arch,
            conf,
            ckpt,
            0,
            args.instrum,
            False,
            args.output_format,
            args.use_tta,
            False, False, args.modelcode)
    elif infer == "medley_vox":
        # coming_soon m_vox(input, output, args.modelcode, args.output_format)
        print(infer)

        

if __name__ == "__main__":
    code_infer()