import os
import sys
sys.stdout.reconfigure(encoding='utf-8') # Фикс для Windows
sys.stderr.reconfigure(encoding='utf-8') # Фикс для Windows
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

from audio import read, write, output_formats
from namer import Namer

namer = Namer()

from infer_utils import prefer_target_instrument, demix, get_model_from_config


def normalize_peak(audio, peak):
    current_peak = np.max(np.abs(audio))
    if current_peak == 0:
        return audio
    scale_factor = peak / current_peak
    return audio * scale_factor


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
    use_tta: bool = False,
    verbose: bool = False,
    model_name: str = None,
    sample_rate: int = 44100,
    instruments: list = [],
    store_dir: str = None,
    template: str = None,
    selected_instruments: list = [],
    model_id: int = 0,
):
    results = []
    sys.stdout.write(json.dumps({"reading": path}, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    sys.stdout.write(
        json.dumps({"selected_stems": selected_instruments}, ensure_ascii=False) + "\n"
    )
    sys.stdout.flush()

    if config.training.target_instrument is not None:
        sys.stdout.write(
            json.dumps(
                {"target_instrument": config.training.target_instrument},
                ensure_ascii=False,
            )
            + "\n"
        )
        sys.stdout.flush()

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

    if use_tta:
        track_proc_list = [mix.copy(), mix[::-1].copy(), -1.0 * mix.copy()]
    else:
        track_proc_list = [mix.copy()]
    full_result = []
    for m in track_proc_list:
        try:
            waveforms = demix(
                config, model, m, device, pbar=detailed_pbar, model_type=model_type
            )

            full_result.append(waveforms)
        except Exception as e:
            sys.stdout.write(
                json.dumps({"error": f"Ошибка при демиксе: {e}"}, ensure_ascii=False)
                + "\n"
            )
            sys.stdout.flush()
        del m
        gc.collect()

    if not full_result:
        sys.stdout.write(
            json.dumps({"error": "Пустой результат демикса."}, ensure_ascii=False)
            + "\n"
        )
        sys.stdout.flush()
        return results

    waveforms = full_result[0]
    for i in range(1, len(full_result)):
        d = full_result[i]
        for el in d:
            if i == 2:
                waveforms[el] += -1.0 * d[el]
            elif i == 1:
                waveforms[el] += d[el][::-1].copy()
            else:
                waveforms[el] += d[el]
    for el in waveforms:
        waveforms[el] /= len(full_result)

    if extract_instrumental and config.training.target_instrument is not None:
        second_stem = [
            s
            for s in config.training.instruments
            if s != config.training.target_instrument
        ]
        if second_stem:
            second_stem_key = second_stem[0]
            if second_stem_key not in instruments:
                instruments.append(second_stem_key)
            waveforms[second_stem_key] = mix_orig - waveforms[instruments[0]]

    elif (
        extract_instrumental
        and selected_instruments
        and config.training.target_instrument is None
    ):

        all_instruments = config.training.instruments
        if len(all_instruments) > 2:

            waveforms["inverted -"] = mix_orig.copy()
            for instr in instruments:
                if instr in waveforms:
                    waveforms["inverted -"] -= waveforms[instr]

            if "inverted -" not in instruments:
                instruments.append("inverted -")

            unselected_stems = [
                s for s in all_instruments if s not in selected_instruments
            ]
            if unselected_stems:
                waveforms["inverted +"] = np.zeros_like(mix_orig)
                for stem in unselected_stems:
                    if stem in waveforms:
                        waveforms["inverted +"] += waveforms[stem]
                if "inverted +" not in instruments:
                    instruments.append("inverted +")

            peak = np.max(np.abs(waveforms["inverted -"]))
            waveforms["inverted +"] = normalize_peak(waveforms["inverted +"], peak)

    elif (
        extract_instrumental
        and not selected_instruments
        and config.training.target_instrument is None
        and (
            all(
                instr in config.training.instruments
                for instr in ["bass", "drums", "other", "vocals"]
            )
            or all(
                instr in config.training.instruments
                for instr in ["bass", "drums", "other", "vocals", "piano", "guitar"]
            )
        )
    ):

        waveforms["instrumental -"] = mix_orig.copy()
        waveforms["instrumental -"] -= waveforms["vocals"]

        if "instrumental -" not in instruments:
            instruments.append("instrumental -")

        all_instruments = config.training.instruments
        non_vocal_stems = [s for s in all_instruments if s not in ["vocals"]]
        if non_vocal_stems:
            waveforms["instrumental +"] = np.zeros_like(mix_orig)
            for stem in non_vocal_stems:
                if stem in waveforms:
                    waveforms["instrumental +"] += waveforms[stem]
            if "instrumental +" not in instruments:
                instruments.append("instrumental +")

        peak = np.max(np.abs(waveforms["instrumental -"]))
        waveforms["instrumental +"] = normalize_peak(waveforms["instrumental +"], peak)

    template = namer.sanitize(template)
    template = namer.dedup_template(template, keys=["NAME", "MODEL", "STEM", "ID"])
    template = namer.short(template, length=40)

    for instr in instruments:
        try:
            estimates = waveforms[instr].T
            if mean is not None and std is not None:
                estimates = estimates * std + mean

            file_name = os.path.splitext(os.path.basename(path))[0]
            file_name_shorted = namer.short_input_name_template(
                template, STEM=instr, MODEL=model_name, ID=model_id, NAME=file_name
            )
            custom_name = namer.template(
                template,
                STEM=instr,
                MODEL=model_name,
                ID=model_id,
                NAME=file_name_shorted,
            )
            output_path = os.path.join(store_dir, f"{custom_name}.{output_format}")

            sys.stdout.write(
                json.dumps({"writing": output_path}, ensure_ascii=False) + "\n"
            )
            sys.stdout.flush()

            output_path = write(
                namer.iter(output_path),
                estimates,
                sr,
                output_bitrate,
            )

            results.append((instr, output_path))
            del estimates
        except Exception as e:
            sys.stdout.write(
                json.dumps(
                    {"error": f"Ошибка при обработке {instr}: {e}"}, ensure_ascii=False
                )
                + "\n"
            )
            sys.stdout.flush()
        gc.collect()

    del mix, mix_orig, waveforms, full_result
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
    use_tta: bool = False,
    verbose: bool = False,
    model_name: str = None,
    template: str = "NAME_STEM",
    selected_instruments: list = [],
    model_id: int = 0,
):
    start_time = time.time()
    if model_type != "vr":
        model.eval()
    sample_rate = 44100
    if "sample_rate" in config.audio:
        sample_rate = config.audio["sample_rate"]

    instruments = prefer_target_instrument(config)

    if config.training.target_instrument is not None:
        sys.stdout.write(
            json.dumps(
                {
                    "info": "Целевой инструмент найден в конфигурации модели. Выбранные стемы будут проигнорированы."
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        sys.stdout.flush()
    else:
        if selected_instruments is not None and selected_instruments != []:
            instruments = [
                instr for instr in instruments if instr in selected_instruments
            ]
            if verbose:
                sys.stdout.write(
                    json.dumps({"selected_stems": instruments}, ensure_ascii=False)
                    + "\n"
                )
                sys.stdout.flush()

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
        use_tta=use_tta,
        verbose=verbose,
        model_name=model_name,
        sample_rate=sample_rate,
        instruments=instruments,
        store_dir=store_dir,
        template=template,
        selected_instruments=selected_instruments,
        model_id=model_id,
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
    use_tta=False,
    verbose=False,
    selected_instruments=None,
    model_id=0,
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
        use_tta=use_tta,
        verbose=verbose,
        model_name=model_name,
        template=template,
        selected_instruments=selected_instruments,
        model_id=model_id,
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
        use_tta=args.use_tta,
        verbose=args.verbose,
        selected_instruments=args.selected_instruments,
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()

