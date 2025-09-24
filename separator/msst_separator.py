import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

import argparse
import gradio as gr
import time
import librosa
from datetime import datetime
from tqdm.auto import tqdm
import json
import gc
import glob
import yaml
import torch
import numpy as np
import soundfile as sf
import torch.nn as nn
from audio_writer import write_audio_file
from renamer_stems import output_file_template
from typing import Literal

from msst_utils import prefer_target_instrument, demix, get_model_from_config, demix_demucs

def normalize_peak(audio, peak):
    current_peak = np.max(np.abs(audio))
    if current_peak == 0:
        return audio  # избегаем деления на ноль
    scale_factor = peak / current_peak
    return audio * scale_factor

gc.enable()

def cleanup_model(model):
    try:
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        
        model.to('cpu')
        
        for name, param in list(model.named_parameters()):
            del param
        for name, buf in list(model.named_buffers()):
            del buf
        
        del model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        gc.collect()
        print("Модель выгружена из памяти")
    except Exception as e:
        print(f"Ошибка при выгрузке модели: {str(e)}")

def once_inference(
    path: str = None, 
    model: any = None, 
    config: any = None, 
    device: any = None, 
    model_type: str = None, 
    extract_instrumental: bool = False,
    detailed_pbar: bool = False, 
    output_format: Literal["mp3", "wav", "flac", "ogg", "opus", "m4a", "aac", "aiff"] = "mp3", 
    output_bitrate: str = "320k",
    use_tta: bool = False, 
    verbose: bool = False, 
    model_name: str = None,
    sample_rate: int = 44100, 
    instruments: list = [], 
    store_dir: str = None, 
    template: str = None, 
    selected_instruments: list = [],
    model_id: int = 0
):
    results = []
    progress = gr.Progress(track_tqdm=True)
    print("Выбранное аудио:", path)
    print("Выбранные стемы:", selected_instruments)
    print("Стемы, которые будут сохранены:", instruments)

    try:
        mix, sr = librosa.load(path, sr=sample_rate, mono=False)
        if mix.ndim == 1:
            mix = np.stack([mix, mix], axis=0)
    except Exception as e:
        print(f"Не удалось прочитать аудио: {path}\nОшибка: {e}")
        return results

    mix_orig = mix.copy()

    mean = std = None
    if config.inference.get('normalize', False):
        mono = mix.mean(0)
        mean = mono.mean()
        std = mono.std()
        mix = (mix - mean) / std

    if use_tta:
        track_proc_list = [mix.copy(), mix[::-1].copy(), -1. * mix.copy()]
    else:
        track_proc_list = [mix.copy()]

    full_result = []
    for m in track_proc_list:
        try:
            if model_type != "htdemucs":
                waveforms = demix(config, model, m, device, pbar=detailed_pbar, model_type=model_type)
            elif model_type == "htdemucs":
                waveforms = demix_demucs(config, model, m, device, pbar=detailed_pbar, model_type=model_type)

            full_result.append(waveforms)
        except Exception as e:
            print(f"Ошибка при демиксе: {e}")
        del m
        gc.collect()

    if not full_result:
        print("Пустой результат демикса.")
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

    if extract_instrumental and config.training.target_instrument is not None: # Если включен "Extract Instrumental / Извлечь инструментал" и найден целевой инструмент
        second_stem = [s for s in config.training.instruments if s != config.training.target_instrument]
        if second_stem:
            second_stem_key = second_stem[0]
            if second_stem_key not in instruments:
                instruments.append(second_stem_key)
            waveforms[second_stem_key] = mix_orig - waveforms[instruments[0]]

    elif extract_instrumental and selected_instruments and config.training.target_instrument is None: # Если включен "Extract Instrumental / Извлечь инструментал" и выбраны инструменты, то создаются стемы "inverted -" и "inverted +" (если не найден целевого инструмент)
        waveforms['inverted -'] = mix_orig.copy()
        for instr in instruments:
            if instr in waveforms:
                waveforms['inverted -'] -= waveforms[instr]   # стем "inverted -": вычитание выбранного стема из оригинального сигнала (не всегда хорошо)

        if 'inverted -' not in instruments:
            instruments.append('inverted -')

        all_instruments = config.training.instruments
        unselected_stems = [s for s in all_instruments if s not in selected_instruments]
        if unselected_stems:
            waveforms['inverted +'] = np.zeros_like(mix_orig)
            for stem in unselected_stems:
                if stem in waveforms:
                    waveforms['inverted +'] += waveforms[stem]   # стем "inverted +": сложение не выбранных инструментов в один стем
            if 'inverted +' not in instruments:
                instruments.append('inverted +')

        peak = np.max(np.abs(waveforms['inverted -']))
        waveforms['inverted +'] = normalize_peak(waveforms['inverted +'], peak)

    elif (extract_instrumental and not selected_instruments and config.training.target_instrument is None and 
      (all(instr in config.training.instruments for instr in ["bass", "drums", "other", "vocals"]) or
       all(instr in config.training.instruments for instr in ["bass", "drums", "other", "vocals", "piano", "guitar"]))):

        waveforms['instrumental -'] = mix_orig.copy()
        waveforms['instrumental -'] -= waveforms["vocals"]   # стем "inverted -": вычитание выбранного стема из оригинального сигнала (не всегда хорошо)

        if 'instrumental -' not in instruments:
            instruments.append('instrumental -')

        all_instruments = config.training.instruments
        non_vocal_stems = [s for s in all_instruments if s not in ["vocals"]]
        if non_vocal_stems:
            waveforms['instrumental +'] = np.zeros_like(mix_orig)
            for stem in non_vocal_stems:
                if stem in waveforms:
                    waveforms['instrumental +'] += waveforms[stem]   # стем "inverted +": сложение не выбранных инструментов в один стем
            if 'instrumental +' not in instruments:
                instruments.append('instrumental +')

        peak = np.max(np.abs(waveforms['instrumental -']))
        waveforms['instrumental +'] = normalize_peak(waveforms['instrumental +'], peak)

    for instr in instruments:
        try:
            estimates = waveforms[instr].T
            if mean is not None and std is not None:
                estimates = estimates * std + mean

            file_name = os.path.splitext(os.path.basename(path))[0]
            custom_name = output_file_template(template, file_name, instr, model_name, model_id)
            output_path = os.path.join(store_dir, f"{custom_name}.{output_format}")
            
            write_audio_file(output_path, estimates, sr, output_format, output_bitrate) # запись стема в аудио файл с помощью универсальной функции

            results.append((instr, output_path)) # запись информации о разделении: (название стема, путь к файлу)
            del estimates
        except Exception as e:
            print(f"Ошибка при обработке {instr}: {e}")
        gc.collect()

    del mix, mix_orig, waveforms, full_result
    librosa.cache.clear()
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
    output_format: Literal["mp3", "wav", "flac", "ogg", "opus", "m4a", "aac", "aiff"] = "mp3", 
    output_bitrate: str = "320k", 
    use_tta: bool = False, 
    verbose: bool = False, 
    model_name: str = None, 
    template: str = 'NAME_STEM', 
    selected_instruments: list = [],
    model_id: int = 0
):
    start_time = time.time()
    model.eval()
    sample_rate = 44100
    if 'sample_rate' in config.audio:
        sample_rate = config.audio['sample_rate']
    
    instruments = prefer_target_instrument(config)
    
    if config.training.target_instrument is not None:
        print("Целевой инструмент найден в конфигурации модели. Выбранные стемы будут проигнорированы.")
    else:
        if selected_instruments is not None and selected_instruments != []:
            instruments = [instr for instr in instruments if instr in selected_instruments]
            if verbose:
                print(f"Выбранные стемы: {instruments}")

    os.makedirs(store_dir, exist_ok=True)

    detailed_pbar = not disable_detailed_pbar

    results = once_inference(
        path=input_path, model=model, config=config, device=device, model_type=model_type, extract_instrumental=extract_instrumental,
        detailed_pbar=detailed_pbar, output_format=output_format, output_bitrate=output_bitrate, use_tta=use_tta, verbose=verbose,
        model_name=model_name, sample_rate=sample_rate, instruments=instruments, store_dir=store_dir, template=template, selected_instruments=selected_instruments, model_id=model_id
    )

    time.sleep(1)
    print(f"Потрачено времени: {time.time() - start_time:.2f} сек.")
    return results

def load_model(model_type, config_path, start_check_point, device_ids, force_cpu=False):
    device = "cpu"
    if force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        print('Разделение выполняется на ядрах CUDA. Для выполнения на процессоре установите force_cpu=True.')
        device = "cuda"

        if device_ids is None:
            device = "cuda:0"
        elif isinstance(device_ids, (list, tuple)):
            device = f'cuda:{device_ids[0]}' if device_ids else 'cuda:0'
        elif isinstance(device_ids, bool):
            device = "cuda:0"
        else:
            device = f'cuda:{int(device_ids)}'
    elif torch.backends.mps.is_available():
        device = "mps"

    print(f"Используется устройство: {device}")

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(model_type, config_path)
    if start_check_point != '':
        print(f'Выбранный чекпоинт: {start_check_point}')
        if model_type in ['htdemucs', 'apollo']:
            state_dict = torch.load(start_check_point, map_location=device, weights_only=False)
            if 'state' in state_dict:
                state_dict = state_dict['state']
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        else:
            try:
                state_dict = torch.load(start_check_point, map_location=device, weights_only=True)
            except torch.serialization.pickle.UnpicklingError:
                with torch.serialization.safe_globals([torch._C._nn.gelu]):
                    state_dict = torch.load(start_check_point, map_location=device, weights_only=True)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)
    print(f"Стемы: {config.training.instruments}")

    if isinstance(device_ids, (list, tuple)) and len(device_ids) > 1 and not force_cpu and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=[int(d) for d in device_ids])

    model = model.to(device)

    print(f"Потрачено времени на загрузку модели: {time.time() - model_load_start_time:.2f} сек.")

    return model, config, device

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
    device_ids=None, 
    disable_detailed_pbar=False, 
    use_tta=False, 
    force_cpu=False, 
    verbose=False,
    selected_instruments=None,
    save_results_info=False,
    model_id=0
):
    model, config, device = load_model(model_type, config_path, start_check_point, device_ids, force_cpu)

    results = run_inference(
        model=model, config=config, input_path=input_path, store_dir=store_dir, device=device, model_type=model_type, extract_instrumental=extract_instrumental,
        disable_detailed_pbar=disable_detailed_pbar, output_format=output_format, output_bitrate=output_bitrate, use_tta=use_tta, verbose=verbose,
        model_name=model_name, template=template, selected_instruments=selected_instruments, model_id=model_id
    )

    if save_results_info:

        with open(os.path.join(store_dir, "results.json"), 'w') as f:
            json.dump(results, f)

    cleanup_model(model)
    del config
    gc.collect()
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Модифицированный Music-Source-Separation-Training для разделения аудио на источники')
    
    # Обязательные аргументы
    parser.add_argument('--input', type=str, help='Путь к входному файлу или папке')
    parser.add_argument('--input_list', nargs='+', help='Список с путями к входным файлам')
    parser.add_argument('--store_dir', type=str, required=True, help='Путь для сохранения результатов')
    
    # Основные параметры модели
    parser.add_argument('--model_type', type=str, default='htdemucs', choices=["mel_band_roformer", "bs_roformer", "mdx23c", "scnet", "htdemucs", "bandit", "bandit_v2"], help='Тип модели (по умолчанию: htdemucs)')
    parser.add_argument('--config_path', type=str, required=True, help='Путь к конфигурационному файлу модели')
    parser.add_argument('--start_check_point', type=str, required=True, help='Путь к чекпоинту модели')
    
    # Параметры вывода
    parser.add_argument('--output_format', type=str, default='wav', choices=["wav", "mp3", "flac", "m4a", "aac", "aiff", "ogg", "opus"], help='Формат выходных файлов')
    parser.add_argument('--output_bitrate', type=str, required=True, help='Битрейт выходного файла')
    
    # Опциональные параметры
    parser.add_argument('--batch', action='store_true', help='Обработать все файлы в папке')
    parser.add_argument('--batch_list', action='store_true', help='Обработать все файлы в списке')
    parser.add_argument('--selected_instruments', nargs='+', help='Список стемов для сохранения (например: vocals drums)')
    parser.add_argument('--extract_instrumental', action='store_true', help='Извлечь инструментальную версию')
    parser.add_argument('--template', type=str, default='NAME_STEM', help='Шаблон для имен выходных файлов')
    parser.add_argument('--model_name', type=str, default='model', help='Имя модели для шаблона имен файлов')
    parser.add_argument("-m_id", "--model_id", type=int, required=True, help="Model ID")
    parser.add_argument('--device_ids', nargs='+', help='ID GPU устройств для использования')
    parser.add_argument('--force_cpu', action='store_true', help='Принудительно использовать CPU')
    parser.add_argument('--use_tta', action='store_true', help='Использовать тестовую аугментацию')
    parser.add_argument('--disable_detailed_pbar', action='store_true', help='Отключить детальный прогресс-бар')
    parser.add_argument('--verbose', action='store_true', help='Подробный вывод')
    parser.add_argument('--save_results_info', action='store_true', help='Сохранить данные разделения в {args.store_dir}/results.json для отображения в интерфейсе')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    device_ids = None
    if args.device_ids:
        device_ids = [int(x) for x in args.device_ids]

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
        device_ids=device_ids,
        disable_detailed_pbar=args.disable_detailed_pbar,
        use_tta=args.use_tta,
        force_cpu=args.force_cpu,
        verbose=args.verbose,
        selected_instruments=args.selected_instruments,
        save_results_info=args.save_results_info,
        model_id=args.model_id
    )

if __name__ == "__main__":
    main()
