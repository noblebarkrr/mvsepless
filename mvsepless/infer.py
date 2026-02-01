import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import json
import argparse
import time
from datetime import datetime
import gc
import glob
import yaml
import torch
import numpy as np
import soundfile as sf
import torch.nn as nn

from typing import Literal

from audio import read, multiwrite, output_formats, substractor
from namer import Namer

namer = Namer()

from infer_utils import prefer_target_instrument, demix, get_model_from_config


def normalize_peak(audio, peak):
    current_peak = np.max(np.abs(audio))
    if current_peak == 0:
        return audio
    scale_factor = peak / current_peak
    return audio * scale_factor


def create_output_path(input_path, stem_name, model_name, model_id, output_format, store_dir, template):
    file_name = os.path.splitext(os.path.basename(input_path))[0]
    file_name_shorted = namer.short_input_name_template(
        template, STEM=stem_name, MODEL=model_name, ID=model_id, NAME=file_name
    )
    custom_name = namer.template(
        template,
        STEM=stem_name,
        MODEL=model_name,
        ID=model_id,
        NAME=file_name_shorted,
    )
    return os.path.join(store_dir, f"{custom_name}.{output_format}")


gc.enable()


def cleanup_model(model):
    try:
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        model.to("cpu")

        for name, param in list(model.named_parameters()):
            del param
        for name, buf in list(model.named_buffers()):
            del buf

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        gc.collect()
    except Exception as e:
        pass

def once_inference(
    path: str = None,
    model: any = None,
    config: any = None,
    device: any = None,
    model_type: str = None,
    extract_instrumental: bool = False,
    detailed_pbar: bool = False,
    output_format: Literal[
        "mp3", "wav", "flac", "ogg", "opus", "m4a", "aac", "aiff"
    ] = "mp3",
    output_bitrate: str = "320k",
    model_name: str = None,
    sample_rate: int = 44100,
    instruments: list = [],
    store_dir: str = None,
    template: str = None,
    selected_instruments: list = [],
    model_id: int = 0,
    spec_invert_target_instrument: bool = False
):
    results = []
    sys.stdout.write(json.dumps({"reading": path}, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    sys.stdout.write(
        json.dumps({"selected_stems": selected_instruments}, ensure_ascii=False) + "\n"
    )
    sys.stdout.flush()

    output_instruments = []
    output_waveforms = {}

    mono_bool = False
    if hasattr(config.model, "stereo"):
        mono_bool = False if config.model.stereo else True
    try:
        mix, sr = read(path=path, sr=sample_rate, mono=mono_bool)
    except Exception as e:
        error_msg = f"Не удалось прочитать аудио: {path}\nОшибка: {e}"
        sys.stdout.write(json.dumps({"error": error_msg}, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return results

    mix_orig = mix.copy()

    mean = std = None
    if config.inference.get("normalize", False):
        mono = mix.mean(0)
        mean = mono.mean()
        std = mono.std()
        mix = (mix - mean) / std

    waveforms = {}

    try:
        waveforms = demix(
            config, model, mix_orig, device, pbar=detailed_pbar, model_type=model_type
        )
    except Exception as e:
        sys.stdout.write(
            json.dumps({"error": f"Ошибка при демиксе: {e}"}, ensure_ascii=False)
            + "\n"
        )
        sys.stdout.flush()
    gc.collect()

    if not waveforms:
        sys.stdout.write(
            json.dumps({"error": "Пустой результат демикса."}, ensure_ascii=False)
            + "\n"
        )
        sys.stdout.flush()
        return results

    if config.training.target_instrument is not None and not selected_instruments:
        output_waveforms[config.training.target_instrument] = waveforms[config.training.target_instrument]
        second_stem = None
        for instr_ in instruments:
            if instr_ != config.training.target_instrument:
                second_stem = instr_
                break
        if second_stem:
            output_waveforms[second_stem] = substractor(mix_orig, waveforms[config.training.target_instrument], sample_rate, sample_rate, spectrogram=spec_invert_target_instrument)[0]
    elif config.training.target_instrument is not None and selected_instruments:
        if config.training.target_instrument in selected_instruments:
            output_waveforms[config.training.target_instrument] = waveforms[config.training.target_instrument]
        second_stem = None
        for instr_ in instruments:
            if instr_ != config.training.target_instrument:
                second_stem = instr_
                break
        if second_stem:
            if second_stem in selected_instruments:
                output_waveforms[second_stem] = substractor(mix_orig, waveforms[config.training.target_instrument], sample_rate, sample_rate, spectrogram=spec_invert_target_instrument)[0]

    elif config.training.target_instrument is None and not selected_instruments and not extract_instrumental:
        for instr in waveforms:
            output_waveforms[instr] = waveforms[instr]
    elif config.training.target_instrument is None and selected_instruments and not extract_instrumental:
        for instr in waveforms:
            if instr in selected_instruments:
                output_waveforms[instr] = waveforms[instr]
    elif config.training.target_instrument is None and selected_instruments and extract_instrumental:
        for instr in waveforms:
            if instr in selected_instruments:
                output_waveforms[instr] = waveforms[instr]
        if len(instruments) > 2:
            output_waveforms["inverted -"] = mix_orig.copy()
            for instr_ in selected_instruments:
                if instr_ in waveforms:
                    output_waveforms["inverted -"] = substractor(output_waveforms["inverted -"], waveforms[instr_], sample_rate, sample_rate, spectrogram=spec_invert_target_instrument)[0]

            unselected_stems = [
                s for s in instruments if s not in selected_instruments
            ]
            if unselected_stems:
                output_waveforms["inverted +"] = np.zeros_like(mix_orig)
                for stem in unselected_stems:
                    if stem in waveforms:
                        output_waveforms["inverted +"] += waveforms[stem]
                if "inverted +" not in instruments:
                    instruments.append("inverted +")

            peak = np.max(np.abs(output_waveforms["inverted -"]))
            output_waveforms["inverted +"] = normalize_peak(output_waveforms["inverted +"], peak)

    elif (
        extract_instrumental
        and not selected_instruments
        and config.training.target_instrument is None
        and (
            all(
                instr in instruments
                for instr in ["bass", "drums", "other", "vocals"]
            )
            or all(
                instr in instruments
                for instr in ["bass", "drums", "other", "vocals", "piano", "guitar"]
            )
        )
    ):
        for instr in waveforms:
            output_waveforms[instr] = waveforms[instr]
        output_waveforms["instrumental -"] = mix_orig.copy()
        output_waveforms["instrumental -"] = substractor(output_waveforms["instrumental -"], waveforms["vocals"], sample_rate, sample_rate, spectrogram=spec_invert_target_instrument)[0]

        non_vocal_stems = [s for s in instruments if s not in ["vocals"]]
        if non_vocal_stems:
            output_waveforms["instrumental +"] = np.zeros_like(mix_orig)
            for stem in non_vocal_stems:
                if stem in waveforms:
                    output_waveforms["instrumental +"] += waveforms[stem]

        peak = np.max(np.abs(output_waveforms["instrumental -"]))
        output_waveforms["instrumental +"] = normalize_peak(output_waveforms["instrumental +"], peak)

    output_instruments = [instr__ for instr__ in output_waveforms]

    template = namer.sanitize(template)
    template = namer.dedup_template(template, keys=["NAME", "MODEL", "STEM", "ID"])
    template = namer.short(template, length=40)

    output_paths = [create_output_path(path, instr, model_name, model_id, output_format, store_dir, template) for instr in output_instruments]
    if mean is not None and std is not None:
        output_arrays = [output_waveforms[instr] * std + mean for instr in output_instruments]
    else:
        output_arrays = [output_waveforms[instr] for instr in output_instruments]
    output_sample_rates = [sample_rate for _ in range(len(output_instruments))]

    def flush_writing_file(file):
        sys.stdout.write(
            json.dumps({"writing": file}, ensure_ascii=False) + "\n"
        )
        sys.stdout.flush()

    try:
        writed_files = multiwrite(output_arrays, output_sample_rates, [namer.iter(output_path_) for output_path_ in output_paths], output_bitrate, callable_func=flush_writing_file, strict=True)
    except Exception as e:
        sys.stdout.write(
            json.dumps(
                {"error": f"Ошибка при обработке: {e}"}, ensure_ascii=False
            )
            + "\n"
        )
        sys.stdout.flush()
    gc.collect()

    results = tuple(zip(output_instruments, writed_files))

    del mix, mix_orig, waveforms, output_arrays
    gc.collect()

    return results


def run_inference(
    model: any = None,
    config: any = None,
    input_path: str = None,
    store_dir: str = None,
    device: any = None,
    model_type: str = None,
    extract_instrumental: bool = False,
    disable_detailed_pbar: bool = False,
    output_format: Literal[
        "mp3", "wav", "flac", "ogg", "opus", "m4a", "aac", "aiff"
    ] = "mp3",
    output_bitrate: str = "320k",
    model_name: str = None,
    template: str = "NAME_STEM",
    selected_instruments: list = [],
    model_id: int = 0,
    spec_invert_target_instrument: bool = False
):
    start_time = time.time()
    if model_type != "vr":
        model.eval()
    sample_rate = 44100
    if "sample_rate" in config.audio:
        sample_rate = config.audio["sample_rate"]

    instruments = config.training.instruments

    os.makedirs(store_dir, exist_ok=True)

    detailed_pbar = not disable_detailed_pbar

    results = once_inference(
        path=input_path,
        model=model,
        config=config,
        device=device,
        model_type=model_type,
        extract_instrumental=extract_instrumental,
        detailed_pbar=detailed_pbar,
        output_format=output_format,
        output_bitrate=output_bitrate,
        model_name=model_name,
        sample_rate=sample_rate,
        instruments=instruments,
        store_dir=store_dir,
        template=template,
        selected_instruments=selected_instruments,
        model_id=model_id,
        spec_invert_target_instrument=spec_invert_target_instrument
    )

    time.sleep(1)
    time_taken = time.time() - start_time
    sys.stdout.write(
        json.dumps({"time": f"{time_taken:.2f} сек."}, ensure_ascii=False) + "\n"
    )
    sys.stdout.flush()
    sys.stdout.write(json.dumps({"done": results}, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return results


def load_model(model_type, config_path, start_check_point, device: str):
    sys.stdout.write(json.dumps({"device": device}, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    
    # Определяем тип устройства
    if "cuda" in device.lower():
        # Извлекаем ID устройств для CUDA
        if ":" in device:
            device_spec = device.split(":")[1]
            device_ids = [int(id) for id in device_spec.split(",") if id.isdigit()]
        else:
            # Если указано просто "cuda", используем все доступные GPU
            device_ids = list(range(torch.cuda.device_count()))
        torch_device = torch.device("cuda" if not device_ids else f"cuda:{device_ids[0]}")
    elif "mps" in device.lower():
        device_ids = None
        torch_device = torch.device("mps")
    else:
        # CPU
        device_ids = None
        torch_device = torch.device("cpu")
    
    model_load_start_time = time.time()
    
    # Устанавливаем оптимизации только для CUDA
    if torch_device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    model, config = get_model_from_config(model_type, config_path)

    if model_type == "vr":
        model.load_checkpoint(start_check_point, torch_device)
        model.settings(
            enable_post_process=False,
            post_process_threshold=config.inference.post_process_threshold,
            batch_size=config.inference.batch_size,
            window_size=config.inference.window_size,
            high_end_process=config.inference.high_end_process,
            primary_stem=config.training.instruments[0],
            secondary_stem=config.training.instruments[1],
        )
        return model, config, torch_device

    elif model_type == "mdxnet":
        if start_check_point != "":
            sys.stdout.write(json.dumps({"checkpoint": start_check_point}) + "\n")
            sys.stdout.flush()
            model.init_onnx_session(start_check_point, torch_device, device_ids)
        return model, config, torch_device

    else:
        if start_check_point != "":
            sys.stdout.write(json.dumps({"checkpoint": start_check_point}) + "\n")
            sys.stdout.flush()

            if model_type in ["htdemucs", "apollo"]:
                state_dict = torch.load(
                    start_check_point, map_location=torch_device, weights_only=False
                )
                if "state" in state_dict:
                    state_dict = state_dict["state"]
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
            else:
                if hasattr(config, "fno"):
                    with torch.serialization.safe_globals([torch._C._nn.gelu]):
                        state_dict = torch.load(
                            start_check_point, map_location=torch_device, weights_only=True
                        )
                else:
                    try:
                        state_dict = torch.load(
                            start_check_point, map_location=torch_device, weights_only=True
                        )
                    except torch.serialization.pickle.UnpicklingError:
                        state_dict = torch.load(
                            start_check_point, map_location=torch_device, weights_only=False
                        )
            
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                print(f"Warning: Error loading state dict: {e}")
                model.load_state_dict(state_dict, strict=False)

        sys.stdout.write(
            json.dumps({"stems": list(config.training.instruments)}, ensure_ascii=False)
            + "\n"
        )
        sys.stdout.flush()
        
        # Перемещаем модель на устройство
        model = model.to(torch_device)
        
        # Используем DataParallel только если есть несколько GPU и это не MPS
        if torch_device.type == "cuda" and len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
            print(f"Using DataParallel on devices: {device_ids}")
        
        load_time = time.time() - model_load_start_time

        sys.stdout.write(
            json.dumps({"model_load_time": f"{load_time:.2f} сек."}, ensure_ascii=False)
            + "\n"
        )
        sys.stdout.flush()

        return model, config, torch_device


def mvsep_offline(
    input_path,
    store_dir,
    model_type,
    config_path,
    start_check_point,
    extract_instrumental,
    output_format,
    output_bitrate,
    model_name,
    template,
    device="cpu",
    disable_detailed_pbar=False,
    selected_instruments=None,
    model_id=0,
    spec_invert_target_instrument=False
):
    model, config, device = load_model(
        model_type, config_path, start_check_point, device
    )

    results = run_inference(
        model=model,
        config=config,
        input_path=input_path,
        store_dir=store_dir,
        device=device,
        model_type=model_type,
        extract_instrumental=extract_instrumental,
        disable_detailed_pbar=disable_detailed_pbar,
        output_format=output_format,
        output_bitrate=output_bitrate,
        model_name=model_name,
        template=template,
        selected_instruments=selected_instruments,
        model_id=model_id,
        spec_invert_target_instrument=spec_invert_target_instrument
    )

    if model_type != "vr":
        cleanup_model(model)
    del config
    gc.collect()
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Модифицированный Music-Source-Separation-Training для разделения аудио на источники"
    )

    parser.add_argument("--input", type=str, help="Путь к входному файлу или папке")
    parser.add_argument(
        "--store_dir", type=str, required=True, help="Путь для сохранения результатов"
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="htdemucs",
        choices=[
            "mel_band_roformer",
            "bs_roformer",
            "mdx23c",
            "scnet",
            "scnet_masked",
            "scnet_tran",
            "htdemucs",
            "bandit",
            "bandit_v2",
            "mdxnet",
            "vr",
        ],
        help="Тип модели (по умолчанию: htdemucs)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Путь к конфигурационному файлу модели",
    )
    parser.add_argument(
        "--start_check_point", type=str, required=True, help="Путь к чекпоинту модели"
    )

    parser.add_argument(
        "--output_format",
        type=str,
        default="wav",
        choices=output_formats,
        help="Формат выходных файлов",
    )
    parser.add_argument(
        "--output_bitrate", type=str, required=True, help="Битрейт выходного файла"
    )

    parser.add_argument(
        "--selected_instruments",
        nargs="+",
        help="Список стемов для сохранения (например: vocals drums)",
    )
    parser.add_argument(
        "--extract_instrumental",
        action="store_true",
        help="Извлечь инструментальную версию",
    )
    parser.add_argument(
        "--use_spec_invert",
        action="store_true",
        help="При извлечь инструментальной версии, использовать спектрограмму",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="NAME_STEM",
        help="Шаблон для имен выходных файлов",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="model",
        help="Имя модели для шаблона имен файлов",
    )
    parser.add_argument("-m_id", "--model_id", type=int, required=True, help="Model ID")
    parser.add_argument(
        "--device", type=str, help="Какой девайс используется для разделения", default="cuda:0"
    )
    parser.add_argument(
        "--use_tta", action="store_true", help="Использовать тестовую аугментацию"
    )
    parser.add_argument(
        "--disable_detailed_pbar",
        action="store_true",
        help="Отключить детальный прогресс-бар",
    )
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод")

    return parser.parse_args()


def main():
    args = parse_args()

    results = mvsep_offline(
        input_path=args.input,
        store_dir=args.store_dir,
        model_type=args.model_type,
        config_path=args.config_path,
        start_check_point=args.start_check_point,
        extract_instrumental=args.extract_instrumental,
        output_format=args.output_format,
        output_bitrate=args.output_bitrate,
        model_name=args.model_name,
        template=args.template,
        device=args.device,
        disable_detailed_pbar=args.disable_detailed_pbar,
        selected_instruments=args.selected_instruments,
        model_id=args.model_id,
        spec_invert_target_instrument=args.use_spec_invert
    )

if __name__ == "__main__":
    main()
