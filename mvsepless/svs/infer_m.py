import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import pyloudnorm as pyln
from namer import Namer
import numpy as np
import argparse
import json
import torch
import gc
from audio import read, multi_channel_array_from_arrays, split_channels, multiwrite, output_formats, check, get_duration_from_array, convert_to_dtype
namer = Namer()
stereo_modes = ("mono", "left/right")
spectral_features = ("mfcc", "spectral_centroid")
vad_methods = ("spec", "webrtc")
overlap_add_methods = (None, "ola", "sf_chunk")
stems = ["vox_1", "vox_2"]

def get_windowing_array(window_size: int, fade_size: int) -> torch.Tensor:
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window

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

def main():
    from svs.utils import loudnorm, str2bool, db2linear
    from svs.models import load_model_with_args
    from svs.functions import load_ola_func_with_args
    parser = argparse.ArgumentParser(description="Адаптированный инференс Medley-Vox для MVSepLess Epsilon")
    parser.add_argument("--input", type=str, required=True, help="Путь к входному файлу или папке")
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
    parser.add_argument(
        "--output_format",
        type=str,
        default="wav",
        choices=output_formats,
        help="Формат выходных файлов",
    )
    parser.add_argument("-m_id", "--model_id", type=int, required=True, help="Model ID")
    parser.add_argument(
        "--device", type=str, help="Какой девайс используется для разделения", default="cuda:0"
    )
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Путь к чекпоинту модели")
    parser.add_argument("--json_path", type=str, required=True, help="Путь к конфигурационному файлу модели")
    parser.add_argument(
        "--use_overlapadd",
        type=str,
        default=None,
        choices=overlap_add_methods,
        help="use overlapadd functions, ola, ola_norm will work with ola_window_len, ola_hop_len argugments. sf_chunk is chunk-wise processing based on VAD, so you have to specify the vad_method args. If you use sf_chunk (spectral_featrues_chunk), you also need to specify spectral_features.",
    )
    parser.add_argument(
        "--vad_method",
        type=str,
        default=vad_methods[1],
        choices=vad_methods,
        help="what method do you want to use for 'voice activity detection (vad) -- split chunks -- processing. Only valid when 'w2v_chunk' or 'sf_chunk' for args.use_overlapadd.",
    )
    parser.add_argument(
        "--spectral_features",
        type=str,
        default=spectral_features[0],
        choices=spectral_features,
        help="what spectral feature do you want to use in correlation calc in speaker assignment (only valid when using sf_chunk)",
    )
    parser.add_argument(
        "--w2v_ckpt_path",
        type=str,
        required=False,
        help="only valid when use_overlapadd is 'w2v' or 'w2v_chunk'.",
    )
    parser.add_argument(
        "--w2v_nth_layer_output",
        nargs="+",
        type=int,
        default=[0],
        help="wav2vec nth layer output",
    )
    parser.add_argument(
        "--ola_window_len",
        type=float,
        default=None,
        help="ola window size in [sec]",
    )
    parser.add_argument(
        "--ola_hop_len",
        type=float,
        default=None,
        help="ola hop size in [sec]",
    )
    parser.add_argument(
        "--use_ema_model",
        type=str2bool,
        default=True,
        help="use ema model or online model? only vaind when args.ema it True (model trained with ema)",
    )
    parser.add_argument(
        "--stereo",
        type=str,
        choices=stereo_modes,
        default="mono",
        help='',
    )
    parser.add_argument(
        "--mix_consistent_out",
        type=str2bool,
        default=True,
        help="only valid when the model is trained with mixture_consistency loss. Default is True.",
    )
    parser.add_argument(
        "--reorder_chunks",
        type=str2bool,
        default=True,
        help="ola reorder chunks",
    )
    parser.add_argument("--results_save_dir", type=str, default="./my_sep_results", help="Путь для сохранения результатов")
    args, _ = parser.parse_known_args()
    device = args.device
    processed_mixtures_1 = []
    processed_mixtures_2 = []
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
    with open(args.json_path, "r") as f:
        args_dict = json.load(f)
    for key, value in args_dict["args"].items():
        setattr(args, key, value)

    model = load_model_with_args(args)

    device = torch_device
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if args.ema and args.use_ema_model:
        print("use ema model")
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        checkpoint = {
            k.replace("ema_model.module.", ""): v
            for k, v in checkpoint.items()
            if k.replace("ema_model.module.", "") in model_dict
        }
        # 2. overwrite entries in the existing state dict
        model_dict.update(checkpoint)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    elif args.ema and not args.use_ema_model:
        print("use ema online model")
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        checkpoint = {
            k.replace("online_model.module.", ""): v
            for k, v in checkpoint.items()
            if k.replace("online_model.module.", "") in model_dict
        }
        # 2. overwrite entries in the existing state dict
        model_dict.update(checkpoint)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    meter = pyln.Meter(args.sample_rate)
    if args.use_overlapadd:
        gc.collect()
        torch.cuda.empty_cache()
        continuous_nnet = load_ola_func_with_args(args, model, device, meter)

    os.makedirs(args.results_save_dir, exist_ok=True)

    if args.stereo == "left/right":
        mixture, sr = read(
            args.input,
            sr=args.sample_rate,
            mono=False
        )
        mixtures = split_channels(mixture)
    elif args.stereo == "mono":
        mixture, sr = read(
            args.input,
            sr=args.sample_rate,
            mono=True, flatten=True
        )
        mixtures = (mixture,)

    for num_mixture, mixture_ in enumerate(mixtures, start=1):
        mixture_d, adjusted_gain = loudnorm(mixture_, -24.0, meter)
        length_init = len(mixture_d)
        
        if args.use_overlapadd:
            # Оставляем существующую логику OLA/VAD если она включена
            mixture_tensor = mixture_d.reshape(1, 1, length_init)
            mixture_tensor = torch.as_tensor(mixture_tensor, dtype=torch.float32).to(device)
            out_wavs = continuous_nnet.forward(mixture_tensor, num_mixture)
        else:
            mix_tensor = torch.tensor(mixture_d, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            samplerate = args.sample_rate
            segment_sec = args.seq_dur
            chunk_size = int(samplerate * segment_sec)
            overlap = 2 
            step = chunk_size // overlap
            fade_size = chunk_size // 10
            
            windowing_array = get_windowing_array(chunk_size, fade_size).to(device)
            result = torch.zeros((1, 2, length_init), dtype=torch.float32, device=device)
            counter = torch.zeros((1, 2, length_init), dtype=torch.float32, device=device)

            i = 0
            while i < length_init:
                part = mix_tensor[..., i : i + chunk_size]
                cur_chunk_len = part.shape[-1]
                
                if cur_chunk_len < chunk_size:
                    part = torch.nn.functional.pad(part, (0, chunk_size - cur_chunk_len), mode='constant', value=0)
                
                with torch.no_grad():
                    out_chunk = model.separate(part) 
                
                window = windowing_array.clone()
                if i == 0: window[:fade_size] = 1
                if i + chunk_size >= length_init: window[-fade_size:] = 1
                
                result[..., i : i + cur_chunk_len] += out_chunk[..., :cur_chunk_len] * window[:cur_chunk_len]
                counter[..., i : i + cur_chunk_len] += window[:cur_chunk_len]
                
                i += step

                # ОТПРАВКА ПРОГРЕССА В JSON
                progress_data = {
                    "processing": {
                        "processed": min(i, length_init),
                        "total": length_init,
                        "unit": "сэмплов",
                        "mixture": num_mixture
                    }
                }
                sys.stdout.write(json.dumps(progress_data, ensure_ascii=False) + "\n")
                sys.stdout.flush()

            out_wavs = result / counter

        if device.type == "cuda":
            out_wav_1 = out_wavs[0, 0, :].cpu().detach().numpy()
            out_wav_2 = out_wavs[0, 1, :].cpu().detach().numpy()
        else:
            out_wav_1 = out_wavs[0, 0, :].numpy() if torch.is_tensor(out_wavs) else out_wavs[0, 0, :]
            out_wav_2 = out_wavs[0, 1, :].numpy() if torch.is_tensor(out_wavs) else out_wavs[0, 1, :]

        out_wav_1 = out_wav_1 * db2linear(-adjusted_gain)
        out_wav_2 = out_wav_2 * db2linear(-adjusted_gain)

        processed_mixtures_1.append(out_wav_1)
        processed_mixtures_2.append(out_wav_2)

        out_wav_1 = out_wav_1 * db2linear(-adjusted_gain)
        out_wav_2 = out_wav_2 * db2linear(-adjusted_gain)

        processed_mixtures_1.append(out_wav_1)
        processed_mixtures_2.append(mixture_ + -out_wav_1 if args.use_overlapadd == "sf_chunk" else out_wav_2)

    vox_1 = multi_channel_array_from_arrays(*processed_mixtures_1, index=1, dtype=np.float32)
    vox_2 = multi_channel_array_from_arrays(*processed_mixtures_2, index=1, dtype=np.float32)
    output_paths = [create_output_path(args.input, stem, args.model_name, args.model_id, args.output_format, args.results_save_dir, args.template) for stem in stems]
    output_arrays = [vox_1, vox_2]       
    writed_files = multiwrite(output_arrays, [sr for __a in range(len(output_arrays))], [namer.iter(output_path_) for output_path_ in output_paths], 180 if args.output_format == "ogg" else 320, strict=True)
    sys.stdout.write(json.dumps({"done": writed_files}, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return writed_files

if __name__ == "__main__":
    main()