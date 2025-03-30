import os
import argparse
import subprocess
from inference import mvsep_offline
from config import conf_editor
from audio_separator.separator import Separator

def code_infer():
    parser = argparse.ArgumentParser(description="Универсальный инференс для разделения аудио на музыкальную и вокальную части")
    parser.add_argument("-i", "--input", type=str, help="Директория входа")
    parser.add_argument("-o", "--output", default="", type=str, help="Директория вывода")
    parser.add_argument("-inst", "--instrum", action='store_true', help="Сохранение инструментала")
    parser.add_argument("-mcode", "-mc", "--modelcode", dest='modelcode', type=int, required=True, help="Код модели")
    parser.add_argument("-of", "--output_format", type=str, choices=['mp3', 'wav', 'flac'], default='wav', help="Формат вывода")
    parser.add_argument("-tta", "--use_tta", action='store_true', help="Повышение качества разделения за счет смены полярности и инвертирования каналов в три прохода")
    parser.add_argument("-b", "--batch", action='store_true', help="Пакетная обработка")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    model_dir = f"model/{args.modelcode}"
    model_dir_medley_vox = f"model"
    os.makedirs(model_dir, exist_ok=True)
    
    from models_list import get_model_config


    config = get_model_config(args.modelcode)

    if config:
        archr = config["arch"]
        if archr == "vr_arch" or archr == "mdx-net" or archr == "demucs":
            model_name = config["ckpt_url"]
            print(f"Архитектура: {archr}")
            print(f"Название модели: {model_name}")
            
        else:
            model_name = config["model_name"]
            ckpt_url = config["ckpt_url"]
            conf_url = config["conf_url"]

            print(f"Название модели: {model_name}")
            print(f"Архитектура: {archr}")
            print(f"Ссылка на модель: {ckpt_url}")
            print(f"Ссылка на конфигурационный файл: {conf_url}")

    else:
        raise("no no no")

    if archr == "mel_band_roformer" or archr == "bs_roformer" or archr == "mdx23c" or archr == "scnet":
        conf = f"{model_dir}/config.yaml"
        ckpt = f"{model_dir}/model.ckpt"
        infer = "mss"
        for local_path, url_model in [(ckpt, ckpt_url), (conf, conf_url)]:
            download_ckpt = ("wget", "-O", str(local_path), str(url_model))
            subprocess.run(download_ckpt, check=True)

    elif archr == "medleyvox":
        conf = f"{model_dir}/vocals.json"
        ckpt = f"{model_dir}/vocals.pth"
        for local_path, url_model in [(ckpt, ckpt_url), (conf, conf_url)]:
            download_ckpt = ("wget", "-O", str(local_path), str(url_model))
            subprocess.run(download_ckpt, check=True)
        infer = "medley_vox"
    elif archr == "vr_arch" or archr == "mdx-net" or archr == "demucs":
        infer = "uvr"

    if infer == "mss":
        conf_editor(conf)
        output_format = args.output_format
        mvsep_offline(
            args.input,
            args.output,
            archr,
            conf,
            ckpt,
            0,
            args.instrum,
            False,
            args.output_format,
            args.use_tta,
            False, False, args.modelcode, args.batch)
    elif infer == "medley_vox":
        medley_infer = ("python", "-m", "models.medley_vox.svs.inference", "--inference_data_dir", str(args.input), "--results_save_dir", str(args.output), "--model_dir", str(model_dir_medley_vox), "--exp_name", str(args.modelcode), "--use_overlapadd=ola", "--output_format", str(args.output_format), ('--batch' if args.batch else '') )
        subprocess.run(medley_infer, check=True) 
        print(infer)
    elif infer == "uvr":
        if archr == "vr_arch":
            separator = Separator(use_autocast=True, output_dir=args.output, output_format=args.output_format, vr_params={"batch_size": 1, "window_size": 512, "aggression": 100, "enable_tta": args.use_tta, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False})

        elif archr == "mdx-net":
            separator = Separator(use_autocast=True, output_dir=args.output, output_format=args.output_format, mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": True})

        elif archr == "demucs":
            separator = Separator(use_autocast=True, output_dir=args.output, output_format=args.output_format, demucs_params={"segment_size": "Default", "shifts": 2, "overlap": 0.25, "segments_enabled": True})

        separator.load_model(model_filename=model_name)
        if args.batch:
            for filename in os.listdir(args.input):
                input_file = os.path.join(args.input, filename)
                if os.path.isfile(input_file):
                    uvr_sep = separator.separate(input_file)
                    #print(f"Обработан файл: {filename}")
        else:
            uvr_sep = separator.separate(args.input)
            #print(f"Обработан файл: {filename}")

        

if __name__ == "__main__":
    code_infer()
