import os
import subprocess

def download_model(model_paths, model_name, model_type, ckpt_url, conf_url):
    model_dir = os.path.join (model_paths, model_type)

    os.makedirs(model_dir, exist_ok=True)
    if model_type == "mel_band_roformer":
        config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
        checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")
        if os.path.exists(checkpoint_path) and os.path.exists(config_path):
            print("Model already downloaded")
        else:
            for local_path, url_model in [(checkpoint_path, ckpt_url), (config_path, conf_url)]:
                download_ckpt = ("wget", "-O", str(local_path), str(url_model))
                subprocess.run(download_ckpt, check=True)




    if model_type == "bs_roformer":
        config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
        checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")
        if os.path.exists(checkpoint_path) and os.path.exists(config_path):
            print("Model already downloaded")
        else:
            for local_path, url_model in [(checkpoint_path, ckpt_url), (config_path, conf_url)]:
                download_ckpt = ("wget", "-O", str(local_path), str(url_model))
                subprocess.run(download_ckpt, check=True)


    if model_type == "mdx23c":
        config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
        checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")
        if os.path.exists(checkpoint_path) and os.path.exists(config_path):
            print("Model already downloaded")
        else:
            for local_path, url_model in [(checkpoint_path, ckpt_url), (config_path, conf_url)]:
                download_ckpt = ("wget", "-O", str(local_path), str(url_model))
                subprocess.run(download_ckpt, check=True)




    if model_type == "scnet":
        config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
        checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")
        if os.path.exists(checkpoint_path) and os.path.exists(config_path):
            print("Model already downloaded")
        else:
            for local_path, url_model in [(checkpoint_path, ckpt_url), (config_path, conf_url)]:
                download_ckpt = ("wget", "-O", str(local_path), str(url_model))
                subprocess.run(download_ckpt, check=True)
    return config_path, checkpoint_path