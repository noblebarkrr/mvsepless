import os
import urllib.request

def download_model(model_paths, model_name, model_type, ckpt_url, conf_url):
    model_dir = os.path.join(model_paths, model_type)
    os.makedirs(model_dir, exist_ok=True)

    config_path = None
    checkpoint_path = None

    if model_type == "mel_band_roformer":
        config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
        checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")

    elif model_type == "vr":
        config_path = os.path.join(model_dir, f"{model_name}.json")
        checkpoint_path = os.path.join(model_dir, f"{model_name}.pth")
  
    elif model_type == "bs_roformer":
        config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
        checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")
    
    elif model_type == "mdx23c":
        config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
        checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")
    
    elif model_type == "scnet":
        config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
        checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")

    elif model_type == "bandit":
        config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
        checkpoint_path = os.path.join(model_dir, f"{model_name}.chpt")

    elif model_type == "bandit_v2":
        config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
        checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")
    
    elif model_type == "htdemucs":
        config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
        checkpoint_path = os.path.join(model_dir, f"{model_name}.th")
    
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
            if not os.path.exists(local_path):
                print(f"Downloading {local_path} from {url_model}")

                urllib.request.urlretrieve(url_model, local_path)

                print(f"Downloaded to {local_path}")

    return config_path, checkpoint_path