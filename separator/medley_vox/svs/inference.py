import os
import json
import argparse
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

import glob
from datetime import datetime
import numpy as np
import soundfile as sf
import librosa
import torch
import pyloudnorm as pyln
from audio_writer import write_audio_file

from .models import load_model_with_args
from .functions import load_ola_func_with_args
from .utils import loudnorm, str2bool, db2linear

def output_file_template(template, input_file_name, stem, model_named):
    template_name = (
        template
        .replace("NAME", f"{input_file_name}")
        .replace("MODEL", f"{model_named}")
        .replace("STEM", f"{stem}")
    )
    output_name = f"{template_name}"
    return output_name

def once_infer(data_path, device, args, meter, model, continuous_nnet=None):
    song_name = os.path.splitext(os.path.basename(data_path))[0]
    print(f"Сейчас обрабатывается трек - {song_name}")

# Load audio file
    if args.stereo == "full":
        stereo_mixture, sr = librosa.load(
            data_path,
            sr=args.sample_rate,
            mono=False,
            duration=args.read_length,
            dtype=np.float32,
        )
    
        left_channel = stereo_mixture[0, :]
        right_channel = stereo_mixture[1, :]
    if args.stereo == "mono":
        mono_channel, sr = librosa.load(
            data_path,
            sr=args.sample_rate,
            mono=True,
            duration=args.read_length,
            dtype=np.float32,
        )
    if args.stereo == "full":

        left_channel, adjusted_gain = loudnorm(left_channel, -24.0, meter)
        right_channel, adjusted_gain = loudnorm(right_channel, -24.0, meter)
        
    if args.stereo == "mono":

        mono_channel, adjusted_gain = loudnorm(mono_channel, -24.0, meter)

    if args.stereo == "full":

        left_channel = np.expand_dims(left_channel, axis=0)
        left_channel = left_channel.reshape(1, 1, -1)
        left_channel = torch.as_tensor(left_channel, dtype=torch.float32).to(device)


        right_channel = np.expand_dims(right_channel, axis=0)
        right_channel = right_channel.reshape(1, 1, -1)
        right_channel = torch.as_tensor(right_channel, dtype=torch.float32).to(device)

    if args.stereo == "mono":
        mono_channel = np.expand_dims(mono_channel, axis=0)
        mono_channel = mono_channel.reshape(1, 1, -1)
        mono_channel = torch.as_tensor(mono_channel, dtype=torch.float32).to(device)
    
    
    # Separate audio
    if args.use_overlapadd and continuous_nnet is not None:
        if args.stereo == "full":
            left_out_wavs = continuous_nnet.forward(left_channel)
            right_out_wavs = continuous_nnet.forward(right_channel)
        if args.stereo == "mono":
            mono_out_wavs = continuous_nnet.forward(mono_channel)
        
    else:
        if args.stereo == "full":
            left_out_wavs = model.separate(left_channel)
            right_out_wavs = model.separate(right_channel)
        if args.stereo == "mono":
            mono_out_wavs = model.separate(mono_channel)

    if args.stereo == "full":

        left_out_wav_1 = left_out_wavs[0, 0, :].cpu().detach().numpy() if args.use_gpu else left_out_wavs[0, 0, :]
        left_out_wav_2 = left_out_wavs[0, 1, :].cpu().detach().numpy() if args.use_gpu else left_out_wavs[0, 1, :]
        right_out_wav_1 = right_out_wavs[0, 0, :].cpu().detach().numpy() if args.use_gpu else right_out_wavs[0, 0, :]
        right_out_wav_2 = right_out_wavs[0, 1, :].cpu().detach().numpy() if args.use_gpu else right_out_wavs[0, 1, :]
        
    if args.stereo == "mono":
    
        mono_out_wav_1 = mono_out_wavs[0, 0, :].cpu().detach().numpy() if args.use_gpu else mono_out_wavs[0, 0, :]
        mono_out_wav_2 = mono_out_wavs[0, 1, :].cpu().detach().numpy() if args.use_gpu else mono_out_wavs[0, 1, :]
    
    if args.stereo == "full":

        left_out_wav_1 *= db2linear(-adjusted_gain)
        left_out_wav_2 *= db2linear(-adjusted_gain)
        right_out_wav_1 *= db2linear(-adjusted_gain)
        right_out_wav_2 *= db2linear(-adjusted_gain)
        
    if args.stereo == "mono":
        mono_out_wav_1 *= db2linear(-adjusted_gain)
        mono_out_wav_2 *= db2linear(-adjusted_gain)
    
    if args.stereo == "full":

        stereo_out_wav_1 = np.stack((left_out_wav_1, right_out_wav_1), axis=0)
        stereo_out_wav_2 = np.stack((left_out_wav_2, right_out_wav_2), axis=0)

    # Save results
    os.makedirs(args.results_save_dir, exist_ok=True)

    base_path = f"{args.results_save_dir}"

    results = []
    
    if args.stereo == "full":
    
        stereo_vox1_name = output_file_template(args.template, song_name, "vox1_stereo", args.exp_name)
        stereo_vox1_path = f"{os.path.join(base_path, stereo_vox1_name)}.{args.output_format}"
        stereo_vox2_name = output_file_template(args.template, song_name, "vox2_stereo", args.exp_name)
        stereo_vox2_path = f"{os.path.join(base_path, stereo_vox2_name)}.{args.output_format}"

    if args.stereo == "mono":

        mono_vox1_name = output_file_template(args.template, song_name, "vox1_mono", args.exp_name)
        mono_vox1_path = f"{os.path.join(base_path, mono_vox1_name)}.{args.output_format}"
        mono_vox2_name = output_file_template(args.template, song_name, "vox2_mono", args.exp_name)
        mono_vox2_path = f"{os.path.join(base_path, mono_vox2_name)}.{args.output_format}"
    
    if args.stereo == "mono":

        results.append(("vox1_mono", mono_vox1_path))
        results.append(("vox2_mono", mono_vox2_path))
    
    if args.stereo == "full":

        results.append(("vox1_stereo", stereo_vox1_path))
        results.append(("vox2_stereo", stereo_vox2_path))


    # Save in specified format
    if args.output_format in ("flac", "wav", "mp3", "ogg", "aiff", "opus", "m4a", "aac"):
        if args.stereo == "full":
            write_audio_file(stereo_vox1_path, stereo_out_wav_1, int(args.sample_rate), str(args.output_format), args.output_bitrate)
            write_audio_file(stereo_vox2_path, stereo_out_wav_2, int(args.sample_rate), str(args.output_format), args.output_bitrate)
        if args.stereo == "mono":
            write_audio_file(mono_vox1_path, mono_out_wav_1, int(args.sample_rate), str(args.output_format), args.output_bitrate)
            write_audio_file(mono_vox2_path, mono_out_wav_2, int(args.sample_rate), str(args.output_format), args.output_bitrate)
            
    return results

def main():
    parser = argparse.ArgumentParser(description="medley_vox")
    parser.add_argument("--output_format", type=str, 
                   choices=['mp3', 'wav', 'flac', 'aac', 'm4a', 'opus', 'ogg', 'aiff'], 
                   default='wav', help="Output format")
    parser.add_argument("--batch", action='store_true', help="Batch processing")
    parser.add_argument("--target", type=str, default="vocals")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--template", type=str, default="NAME_STEM")
    parser.add_argument("--suffix_name", type=str, default="")
    parser.add_argument("--model_dir", type=str, default="/path/to/results/singing_sep")
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--use_overlapadd", type=str, default=None,
                      choices=[None, "ola", "ola_norm", "w2v", "w2v_chunk", "sf_chunk"])
    parser.add_argument("--vad_method", type=str, default="spec",
                      choices=["spec", "webrtc"])
    parser.add_argument("--spectral_features", type=str, default="mfcc",
                      choices=["mfcc", "spectral_centroid"])
    parser.add_argument("--w2v_ckpt_dir", type=str, default="./pretrained_models")
    parser.add_argument("--w2v_nth_layer_output", nargs="+", type=int, default=[0])
    parser.add_argument("--ola_window_len", type=float, default=None)
    parser.add_argument("--ola_hop_len", type=float, default=None)
    parser.add_argument("--use_ema_model", type=str2bool, default=True)
    parser.add_argument("--ema", type=str2bool, default=False)
    parser.add_argument("--output_bitrate", type=str, default="320k")
    parser.add_argument("--stereo", type=str, default="full",
                      choices=["mono", "full"])
    parser.add_argument("--read_length", type=float, default=None)
    parser.add_argument("--mix_consistent_out", type=str2bool, default=True)
    parser.add_argument("--reorder_chunks", type=str2bool, default=True)
    parser.add_argument("--inference_data_dir", type=str, default="./segments/24k")
    parser.add_argument("--results_save_dir", type=str, default="./my_sep_results")

    args, _ = parser.parse_known_args()

    args.exp_result_dir = f"{args.model_dir}/{args.exp_name}"

    with open(f"{args.exp_result_dir}/{args.target}.json", "r") as f:
        args_dict = json.load(f)

    for key, value in args_dict["args"].items():
        setattr(args, key, value)

    # Load model
    model = load_model_with_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    target_model_path = f"{args.exp_result_dir}/{args.target}.pth"
    
    # Load checkpoint with weights_only=True for security
    checkpoint = torch.load(target_model_path, map_location=device, weights_only=True)
    
    if args.ema and args.use_ema_model:
        model_dict = model.state_dict()
        checkpoint = {k.replace("ema_model.module.", ""): v for k, v in checkpoint.items()
                     if k.replace("ema_model.module.", "") in model_dict}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    elif args.ema and not args.use_ema_model:
        model_dict = model.state_dict()
        checkpoint = {k.replace("online_model.module.", ""): v for k, v in checkpoint.items()
                     if k.replace("online_model.module.", "") in model_dict}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    meter = pyln.Meter(args.sample_rate)

    if args.use_overlapadd:
        continuous_nnet = load_ola_func_with_args(args, model, device, meter)

    os.makedirs(args.results_save_dir, exist_ok=True)
    
    results = once_infer(args.inference_data_dir, device, args, meter, model,
                  continuous_nnet if args.use_overlapadd else None)

    with open((os.path.join(args.results_save_dir, "results.json")), 'w') as f:
        json.dump(results, f)




if __name__ == "__main__":
    main()