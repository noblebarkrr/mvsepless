import os
import argparse
import subprocess
from inference import mvsep_offline
from config import conf_editor
from audio_separator.separator import Separator
from audio_sep import uvr

def code_infer():
    parser = argparse.ArgumentParser(description="Универсальный инференс для разделения аудио на музыкальную и вокальную части")
    parser.add_argument("-i", "--input", type=str, help="Директория входа")
    parser.add_argument("-o", "--output", default="", type=str, help="Директория вывода")
    parser.add_argument("-inst", "--instrum", action='store_true', help="Сохранение инструментала")
    parser.add_argument("-mcode", "-mc", "--modelcode", dest='modelcode', type=int, required=True, help="Код модели")
    parser.add_argument("-of", "--output_format", type=str, choices=['mp3', 'wav', 'flac'], default='wav', help="Формат вывода")
    parser.add_argument("-tta", "--use_tta", action='store_true', help="Повышение качества разделения за счет смены полярности и инвертирования каналов в три прохода")
    args = parser.parse_args()

    model_dir = f"model/{args.modelcode}"
    os.makedirs(model_dir, exist_ok=True)
    
    from models_list import get_model_config
    from models_list import get_uvr_model_config
    config = get_model_config(args.modelcode)

    if config:
        model_name = config["model_name"]
        arch = config["arch"]
        ckpt_url = config["ckpt_url"]
        conf_url = config["conf_url"]

        print(f"Название модели: {model_name}")
        print(f"Архитектура: {arch}")
        print(f"Ссылка на модель: {ckpt_url}")
        print(f"Ссылка на конфигурационный файл: {conf_url}")
    else:
        print(f"Модель с кодом {model_code} не найдена.")
    config = get_uvr_model_config(args.modelcode)
    if config:
        arch = config["arch"]
        model_name = config["ckpt_url"]

        print(f"Название модели: {model_name}")
        print(f"Архитектура: {arch}")
    else:
        print(f"Модель с кодом {model_code} не найдена.")
    if arch == "mel_band_roformer" or arch == "bs_roformer" or arch == "mdx23c" or arch == "scnet":
        conf = f"{model_dir}/config.yaml"
        ckpt = f"{model_dir}/model.ckpt"
        infer = "mss"

    elif arch == "medleyvox":
        conf = f"{model_dir}/vocals.json"
        ckpt = f"{model_dir}/vocals.pth"
        infer = "medley_vox"
    elif arch == "vr_arch" or arch == "mdx-net" or arch == "demucs":
        infer = "uvr"

    if infer == "mss":
        for local_path, url_model in [(ckpt, ckpt_url), (conf, conf_url)]:
            download_ckpt = ("wget", "-O", str(local_path), str(url_model))
            subprocess.run(download_ckpt, check=True)
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
    elif infer == "uvr":
        uvr(args.input, model_name)

        

if __name__ == "__main__":
    code_infer()