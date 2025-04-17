import argparse
import time
import librosa
from tqdm.auto import tqdm
import sys
import os
import glob
import yaml
import torch
import numpy as np
import soundfile as sf
import torch.nn as nn
from pydub import AudioSegment

from utils import prefer_target_instrument, demix, get_model_from_config

def output_file_template(template, input_file_name, stem, model_named):
    template_name = (
        template
        .replace("NAME", f"{input_file_name}")
        .replace("MODEL", f"{model_named}")
        .replace("STEM", f"{stem}")
    )
    output_name = f"{template_name}"
    return output_name

def once_inference(path, model, config, device, model_type, extract_instrumental, detailed_pbar, output_format, use_tta, verbose, model_name, sample_rate, instruments, store_dir, template, selected_stems):
    print("Starting processing track: ", path)
    print("Selected Stems")
    print(selected_stems)
    print("Output Stems")
    print(instruments)
    try:
        mix, sr = librosa.load(path, sr=sample_rate, mono=False)
    except Exception as e:
        print('Cannot read track: {}'.format(path))
        print('Error message: {}'.format(str(e)))
        return

    # Convert mono to stereo if needed
    if len(mix.shape) == 1:
        mix = np.stack([mix, mix], axis=0)

    mix_orig = mix.copy()
    if 'normalize' in config.inference:
        if config.inference['normalize'] is True:
            mono = mix.mean(0)
            mean = mono.mean()
            std = mono.std()
            mix = (mix - mean) / std

    if use_tta:
        print("USE TTA")
        print("USE TTA")
        print("USE TTA")
        print("USE TTA")
        # orig, channel inverse, polarity inverse
        track_proc_list = [mix.copy(), mix[::-1].copy(), -1. * mix.copy()]
    else:
        track_proc_list = [mix.copy()]

    full_result = []
    for mix in track_proc_list:
        waveforms = demix(config, model, mix, device, pbar=detailed_pbar, model_type=model_type)
        full_result.append(waveforms)

    # Average all values in single dict
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
        waveforms[el] = waveforms[el] / len(full_result)
        # Create a new `instr` in instruments list, 'instrumental' 

    if extract_instrumental and config.training.target_instrument is not None:
        # Create a list of instruments excluding the target instrument
        second_stem = [s for s in config.training.instruments if s != config.training.target_instrument]
        
        # Choose a single instrument from second_stem
        if second_stem:
            # If there are elements in second_stem, use the first one (or handle according to your logic)
            second_stem_key = second_stem[0]  
            
            if second_stem_key not in instruments:
                instruments.append(second_stem_key)
            
            # Output "instrumental", which is an inverse of 'vocals' or the first stem in list if 'vocals' absent
            waveforms[second_stem_key] = mix_orig - waveforms[instruments[0]]
    elif extract_instrumental and selected_stems is not None and config.training.target_instrument == None:
        waveforms['inverted'] = mix_orig.copy()
        for instr in instruments:
            if instr in waveforms:
                waveforms['inverted'] -= waveforms[instr]
        if 'inverted' not in instruments:
            instruments.append('inverted')



    for instr in instruments:
        estimates = waveforms[instr].T
        if 'normalize' in config.inference:
            if config.inference['normalize'] is True:
                estimates = estimates * std + mean
        file_name, _ = os.path.splitext(os.path.basename(path))
        
        # Исправлено: используем переданный параметр model_name вместо несуществующей переменной modelname
        custom_name = output_file_template(template, file_name, instr, f"{model_type}_{model_name}")

        if output_format == "flac":
            output_file_path = os.path.join(store_dir, f"{custom_name}.flac")
            sf.write(output_file_path, estimates, sr, subtype='PCM_16')
        elif output_format == "mp3":
            output_file_path = os.path.join(store_dir, f"{custom_name}.mp3")
    
        # Проверяем, что данные в float и нормализуем их
            if estimates.dtype == np.float32 or estimates.dtype == np.float64:
        # Нормализуем до [-1, 1] если значения выходят за эти границы
                estimates = np.clip(estimates, -1.0, 1.0)
        # Конвертируем в int16
                estimates = (estimates * 32767).astype(np.int16)
            elif estimates.dtype != np.int16:
        # Если тип не float и не int16, конвертируем в int16
                estimates = estimates.astype(np.int16)
    
            audio_segment = AudioSegment(
                estimates.tobytes(),
                frame_rate=sample_rate,
                sample_width=estimates.dtype.itemsize,
                channels=2,
            )
            audio_segment.export(output_file_path, format="mp3", bitrate="320k")
        elif output_format == "wav":
            output_file_path = os.path.join(store_dir, f"{custom_name}.wav")
            sf.write(output_file_path, estimates, sr, subtype='PCM_16')

def run_inference(model, config, input, store_dir, device, model_type, extract_instrumental, disable_detailed_pbar, output_format, use_tta, verbose, model_name, batch, template, selected_instruments):
    start_time = time.time()
    model.eval()
    sample_rate = 44100
    if 'sample_rate' in config.audio:
        sample_rate = config.audio['sample_rate']
    
    instruments = prefer_target_instrument(config)
    
    if config.training.target_instrument is not None:
        print ("Target instrument found. Selected stems ignored.")
    else:
        if selected_instruments is not None:
            instruments = [instr for instr in instruments if instr in selected_instruments]
            if verbose:
                print(f"Selected instruments: {instruments}")

    os.makedirs(store_dir, exist_ok=True)

    if disable_detailed_pbar:
        detailed_pbar = False
    else:
        detailed_pbar = True

    if batch:
        all_mixtures_path = glob.glob(input + '/*.*')
        all_mixtures_path.sort()
        if not verbose:
            # Используем tqdm только если verbose=False
            all_mixtures_path = tqdm(all_mixtures_path, desc="Total progress")
        print('Total files found: {} Use sample rate: {}'.format(len(all_mixtures_path), sample_rate))
        for path in all_mixtures_path:
            once_inference(path, model, config, device, model_type, extract_instrumental, detailed_pbar, output_format, use_tta, verbose, model_name, sample_rate, instruments, store_dir, template, selected_instruments)
    else:
        # Если batch=False, обрабатываем только один файл (input)
        once_inference(input, model, config, device, model_type, extract_instrumental, detailed_pbar, output_format, use_tta, verbose, model_name, sample_rate, instruments, store_dir, template, selected_instruments)

    time.sleep(1)
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))    


def load_model(model_type, config_path, start_check_point, device_ids, force_cpu=False):
    # Determine device
    device = "cpu"
    if force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        print('CUDA is available, use --force_cpu to disable it.')
        device = "cuda"
        # Handle device_ids properly
        if device_ids is None:
            device = "cuda:0"
        elif isinstance(device_ids, (list, tuple)):
            device = f'cuda:{device_ids[0]}' if device_ids else 'cuda:0'
        elif isinstance(device_ids, bool):
            device = "cuda:0"  # Default to first GPU if boolean value is passed
        else:
            device = f'cuda:{int(device_ids)}'  # Convert to int to ensure valid device string
    elif torch.backends.mps.is_available():
        device = "mps"

    print("Using device: ", device)

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(model_type, config_path)
    if start_check_point != '':
        print('Start from checkpoint: {}'.format(start_check_point))
        if model_type in ['htdemucs', 'apollo']:
            state_dict = torch.load(start_check_point, map_location=device, weights_only=False)
            # Fix for htdemucs pretrained models
            if 'state' in state_dict:
                state_dict = state_dict['state']
            # Fix for apollo pretrained models
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        else:
            state_dict = torch.load(start_check_point, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    print("Instruments: {}".format(config.training.instruments))

    # Handle multiple GPUs if specified
    if isinstance(device_ids, (list, tuple)) and len(device_ids) > 1 and not force_cpu and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=[int(d) for d in device_ids])

    model = model.to(device)

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

    return model, config, device

def mvsep_offline(input, store_dir, model_type, config_path, start_check_point, extract_instrumental, output_format, model_name, template, device_ids, disable_detailed_pbar=False, use_tta=False, force_cpu=False, verbose=False, batch=None, selected_instruments=None):
    
    
    model, config, device = load_model(model_type, config_path, start_check_point, device_ids, force_cpu)
    run_inference(model, config, input, store_dir, device, model_type, extract_instrumental, disable_detailed_pbar, output_format, use_tta, verbose, model_name, batch, template, selected_instruments)

if __name__ == "__main__":
    # Example: save only vocals and drums
    selected_stems = ['vocals', 'drums']
    mvsep_offline(
        input="/content/input",
        store_dir="/content/output",
        batch=True,
        selected_instruments=selected_stems
    )