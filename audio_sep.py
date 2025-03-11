from audio_separator.separator import Separator

def uvr(input_dir, model_name):
    if arch == "vr_arch":
        separator = Separator(use_autocast=True, output_dir=args.output, output_format=args.of, vr_params={"batch_size": 1, "window_size": 512, "aggression": 100, "enable_tta": args.use_tta, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False})

    if arch == "mdx-net":
        separator = Separator(use_autocast=True, output_dir=args.output, output_format=args.of, mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": True}

    if arch == "demucs":
        separator = Separator(use_autocast=True, output_dir=args.output, output_format=args.of, demucs_params={"segment_size": "Default", "shifts": 2, "overlap": 0.25, "segments_enabled": True}

    separator.load_model(model_filename=model_name)

    for filename in os.listdir(args.input):
        input_file = os.path.join(folder_path, filename)
        if os.path.isfile(input_file):
            uvr_sep = separator.separate(input_file)
            print(f"Обработан файл: {filename}")