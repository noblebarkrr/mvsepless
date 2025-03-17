import argparse
import time
import librosa
from tqdm.auto import tqdm
import sys
import os
import glob
import torch
import numpy as np
import soundfile as sf
import torch.nn as nn
from utils import prefer_target_instrument, demix, get_model_from_config

def once_inference(path, model, config, device, model_type, extract_instrumental, detailed_pbar, output_format, use_tta, verbose, modelcode, sample_rate, instruments):
    print("Starting processing track: ", path)
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

    for instr in instruments:
        estimates = waveforms[instr].T
        if 'normalize' in config.inference:
            if config.inference['normalize'] is True:
                estimates = estimates * std + mean
        file_name, _ = os.path.splitext(os.path.basename(path))
        from models_list import get_model_config
        config_models_list = get_model_config(modelcode)
        model_name = config_models_list["model_name"]
        if use_tta:
            custom_name = f"{file_name}_tta_mt-{model_type}_{model_name}_{instr}"
        else:
            custom_name = f"{file_name}_mt-{model_type}_{model_name}_{instr}"
        if output_format == "flac":
            output_file = os.path.join(store_dir, f"{custom_name}.flac")
            sf.write(output_file, estimates, sr, subtype='PCM_16')
        elif output_format == "mp3":
            from pydub import AudioSegment
            temp_wav_dir = "/tmp/wav_msst"
            os.makedirs(temp_wav_dir, exist_ok=True)
            output_file = os.path.join(store_dir, f"{custom_name}.mp3")
            temp_wav = os.path.join(temp_wav_dir, f"{custom_name}.wav")
            sf.write(temp_wav, estimates, sr, subtype='PCM_16')  # Временный WAV
    
            # Конвертация в MP3
            audio = AudioSegment.from_wav(temp_wav)
            audio.export(output_file, format="mp3", bitrate="320k")
            os.remove(temp_wav)  # Удаление временного файла
        elif output_format == "wav":
            output_file = os.path.join(store_dir, f"{custom_name}.wav")
            sf.write(output_file, estimates, sr, subtype='PCM_16')

def run_inference(model, config, input, store_dir, device, model_type, extract_instrumental, disable_detailed_pbar, output_format, use_tta, verbose, modelcode, batch):
    start_time = time.time()
    model.eval()
    sample_rate = 44100
    if 'sample_rate' in config.audio:
        sample_rate = config.audio['sample_rate']
    
    all_mixtures_path = glob.glob(input + '/*.*')
    all_mixtures_path.sort()
    print('Total files found: {} Use sample rate: {}'.format(len(all_mixtures_path), sample_rate))

    instruments = prefer_target_instrument(config)

    os.makedirs(store_dir, exist_ok=True)

    if not verbose:
        all_mixtures_path = tqdm(all_mixtures_path, desc="Total progress")

    if disable_detailed_pbar:
        detailed_pbar = False
    else:
        detailed_pbar = True
    if batch:
        for path in all_mixtures_path:
            once_inference(path, model, config, device, model_type, extract_instrumental, detailed_pbar, output_format, use_tta, verbose, modelcode, sample_rate, instruments)
    else:
        once_inference(input, model, config, device, model_type, extract_instrumental, detailed_pbar, output_format, use_tta, verbose, modelcode, sample_rate, instruments)

    time.sleep(1)
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))

def load_model(model_type, config_path, start_check_point, device_ids, force_cpu=False):
    device = "cpu"
    if force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        print('CUDA is available, use --force_cpu to disable it.')
        device = "cuda"
        device = f'cuda:{device_ids[0]}' if type(device_ids) == list else f'cuda:{device_ids}'
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

    # in case multiple CUDA GPUs are used and --device_ids arg is passed
    if type(device_ids) == list and len(device_ids) > 1 and not force_cpu:
        model = nn.DataParallel(model, device_ids = device_ids)

    model = model.to(device)

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

    return model, config, device

def mvsep_offline(input, store_dir, model_type, config_path, start_check_point, device_ids, extract_instrumental, disable_detailed_pbar, output_format, use_tta, force_cpu, verbose, modelcode, batch):
    model, config, device = load_model(model_type, config_path, start_check_point, device_ids, force_cpu)
    run_inference(model, config, input, store_dir, device, model_type, extract_instrumental, disable_detailed_pbar, output_format, use_tta, verbose, modelcode, batch)

if __name__ == "__main__":
    mvsep_offline(input="/content/input", store_dir="/content/output", batch=True)
