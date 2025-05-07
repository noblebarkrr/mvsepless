import os

def download_model(model_paths, model_name, model_type, ckpt_url, conf_url):
    model_dir = os.path.join(model_paths, model_type)
    os.makedirs(model_dir, exist_ok=True)

    # Инициализация переменных (на случай, если ни одно условие не сработает)
    config_path = None
    checkpoint_path = None

    if model_type == "mel_band_roformer":
        config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
        checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")
    
    elif model_type == "bs_roformer":
        config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
        checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")
    
    elif model_type == "mdx23c":
        config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
        checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")
    
    elif model_type == "scnet":
        config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
        checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")
    
    elif model_type == "htdemucs":
        config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
        checkpoint_path = os.path.join(model_dir, f"{model_name}.th")

    elif model_type == "medley_vox":
        medley_vox_model_dir = os.path.join(model_dir, model_name)
        os.makedirs(medley_vox_model_dir, exist_ok=True)
        config_path = os.path.join(medley_vox_model_dir, f"vocals.json")
        checkpoint_path = os.path.join(medley_vox_model_dir, f"vocals.pth")
    
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Проверяем, что пути заданы (на всякий случай)
    if config_path is None or checkpoint_path is None:
        raise RuntimeError("Failed to set model paths!")

    # Если файлы уже есть — пропускаем загрузку
    if os.path.exists(checkpoint_path) and os.path.exists(config_path):
        print("Model already downloaded")
    else:
        for local_path, url_model in [(checkpoint_path, ckpt_url), (config_path, conf_url)]:
            download_cmd = f"wget -O {local_path} {url_model}"
            os.system(download_cmd)

    if model_type == "medley_vox":
        return model_dir
    else:
        return config_path, checkpoint_path