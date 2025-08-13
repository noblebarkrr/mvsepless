# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import os
import sys
import librosa
import tempfile
import soundfile as sf
import numpy as np
import argparse
from separator.audio_writer import write_audio_file


def stft(wave, nfft, hl):
    wave_left = np.asfortranarray(wave[0])
    wave_right = np.asfortranarray(wave[1])
    spec_left = librosa.stft(wave_left, n_fft=nfft, hop_length=hl)
    spec_right = librosa.stft(wave_right, n_fft=nfft, hop_length=hl)
    spec = np.asfortranarray([spec_left, spec_right])
    return spec


def istft(spec, hl, length):
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])
    wave_left = librosa.istft(spec_left, hop_length=hl, length=length)
    wave_right = librosa.istft(spec_right, hop_length=hl, length=length)
    wave = np.asfortranarray([wave_left, wave_right])
    return wave


def absmax(a, *, axis):
    dims = list(a.shape)
    dims.pop(axis)
    indices = np.ogrid[tuple(slice(0, d) for d in dims)]
    argmax = np.abs(a).argmax(axis=axis)
    # Convert indices to list before insertion
    indices = list(indices)
    indices.insert(axis % len(a.shape), argmax)
    return a[tuple(indices)]


def absmin(a, *, axis):
    dims = list(a.shape)
    dims.pop(axis)
    indices = np.ogrid[tuple(slice(0, d) for d in dims)]
    argmax = np.abs(a).argmin(axis=axis)
    indices.insert((len(a.shape) + axis) % len(a.shape), argmax)
    return a[tuple(indices)]


def lambda_max(arr, axis=None, key=None, keepdims=False):
    idxs = np.argmax(key(arr), axis)
    if axis is not None:
        idxs = np.expand_dims(idxs, axis)
        result = np.take_along_axis(arr, idxs, axis)
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result
    else:
        return arr.flatten()[idxs]


def lambda_min(arr, axis=None, key=None, keepdims=False):
    idxs = np.argmin(key(arr), axis)
    if axis is not None:
        idxs = np.expand_dims(idxs, axis)
        result = np.take_along_axis(arr, idxs, axis)
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result
    else:
        return arr.flatten()[idxs]


def average_waveforms(pred_track, weights, algorithm):
    """
    :param pred_track: shape = (num, channels, length)
    :param weights: shape = (num, )
    :param algorithm: One of avg_wave, median_wave, min_wave, max_wave, avg_fft, median_fft, min_fft, max_fft
    :return: averaged waveform in shape (channels, length)
    """

    pred_track = np.array(pred_track)
    final_length = pred_track.shape[-1]

    mod_track = []
    for i in range(pred_track.shape[0]):
        if algorithm == 'avg_wave':
            mod_track.append(pred_track[i] * weights[i])
        elif algorithm in ['median_wave', 'min_wave', 'max_wave']:
            mod_track.append(pred_track[i])
        elif algorithm in ['avg_fft', 'min_fft', 'max_fft', 'median_fft']:
            spec = stft(pred_track[i], nfft=2048, hl=1024)
            if algorithm in ['avg_fft']:
                mod_track.append(spec * weights[i])
            else:
                mod_track.append(spec)
    pred_track = np.array(mod_track)

    if algorithm in ['avg_wave']:
        pred_track = pred_track.sum(axis=0)
        pred_track /= np.array(weights).sum().T
    elif algorithm in ['median_wave']:
        pred_track = np.median(pred_track, axis=0)
    elif algorithm in ['min_wave']:
        pred_track = np.array(pred_track)
        pred_track = lambda_min(pred_track, axis=0, key=np.abs)
    elif algorithm in ['max_wave']:
        pred_track = np.array(pred_track)
        pred_track = lambda_max(pred_track, axis=0, key=np.abs)
    elif algorithm in ['avg_fft']:
        pred_track = pred_track.sum(axis=0)
        pred_track /= np.array(weights).sum()
        pred_track = istft(pred_track, 1024, final_length)
    elif algorithm in ['min_fft']:
        pred_track = np.array(pred_track)
        pred_track = lambda_min(pred_track, axis=0, key=np.abs)
        pred_track = istft(pred_track, 1024, final_length)
    elif algorithm in ['max_fft']:
        pred_track = np.array(pred_track)
        pred_track = absmax(pred_track, axis=0)
        pred_track = istft(pred_track, 1024, final_length)
    elif algorithm in ['median_fft']:
        pred_track = np.array(pred_track)
        pred_track = np.median(pred_track, axis=0)
        pred_track = istft(pred_track, 1024, final_length)
    return pred_track


def ensemble_audio_files(files, output="res.wav", ensemble_type='avg_wave', weights=None, out_format="wav"):
    """
    Основная функция для объединения аудиофайлов
    
    :param files: список путей к аудиофайлам
    :param output: путь для сохранения результата
    :param ensemble_type: метод объединения (avg_wave, median_wave, min_wave, max_wave, avg_fft, median_fft, min_fft, max_fft)
    :param weights: список весов для каждого файла (None для равных весов)
    :return: None
    """
    print('Ensemble type: {}'.format(ensemble_type))
    print('Number of input files: {}'.format(len(files)))
    if weights is not None:
        weights = np.array(weights)
    else:
        weights = np.ones(len(files))
    print('Weights: {}'.format(weights))
    print('Output file: {}'.format(output))
    
    data = []
    sr = None
    max_length = 0
    max_channels = 0
    
    # Первый проход: определяем максимальную длину и количество каналов
    for f in files:
        if not os.path.isfile(f):
            print('Error. Can\'t find file: {}. Check paths.'.format(f))
            exit()
        print('Reading file: {}'.format(f))
        wav, current_sr = librosa.load(f, sr=None, mono=False)
        if sr is None:
            sr = current_sr
        elif sr != current_sr:
            print('Error: Sample rates must be equal for all files')
            exit()
        
        # Определяем количество каналов
        if wav.ndim == 1:
            channels = 1
            length = len(wav)
        else:
            channels = wav.shape[0]
            length = wav.shape[1]
            
        max_length = max(max_length, length)
        max_channels = max(max_channels, channels)
        print("Waveform shape: {} sample rate: {}".format(wav.shape, sr))
    
    # Второй проход: обработка и выравнивание файлов
    for f in files:
        wav, current_sr = librosa.load(f, sr=None, mono=False)
        
        # Обработка каналов
        if wav.ndim == 1:
            # Моно -> стерео
            wav = np.vstack([wav, wav])
        elif wav.shape[0] == 1:
            # Один канал -> стерео
            wav = np.vstack([wav[0], wav[0]])
        elif wav.shape[0] > 2:
            # Более 2 каналов -> берем первые два
            wav = wav[:2, :]
        
        # Выравнивание длины
        if wav.shape[1] < max_length:
            pad_width = ((0, 0), (0, max_length - wav.shape[1]))
            wav = np.pad(wav, pad_width, mode='constant')
        elif wav.shape[1] > max_length:
            wav = wav[:, :max_length]
        
        data.append(wav)
    
    data = np.array(data)
    res = average_waveforms(data, weights, ensemble_type)
    print('Result shape: {}'.format(res.shape))

    output_wav = f"{output}_orig.wav"
    output = f"{output}.{out_format}"
    
    if out_format in ["wav", "flac"]:
        sf.write(output, res.T, sr, subtype='PCM_16')
        sf.write(output_wav, res.T, sr, subtype='PCM_16')

    elif out_format in ["mp3", "m4a", "aac", "ogg", "opus", "aiff"]:
        write_audio_file(output, res.T, sr, out_format, "320k")
        sf.write(output_wav, res.T, sr, subtype='PCM_16')

    return output, output_wav
