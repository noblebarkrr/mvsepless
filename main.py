# main.py
mvsepless_dir = "/content/MVSEPLESS"
import os
import yaml
from urllib.parse import quote
import glob
import subprocess
import argparse
from uvr_models import get_model_by_code
from mdx_models import get_mdx_code
# Импортируем только функцию

# Определяем расширения файлов
extensions = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aiff", ".ac3")

# Функция для обработки аудиофайлов
def uvr_cli(audio_input, output_folder, extensions, output_format, model, window_size, aggression, tta, high_end_process, batch_size, use_autocast):
    found_files = []

    # Поиск файлов с нужными расширениями
    for audio_files in os.listdir(audio_input):
        if audio_files.endswith(extensions):
            found_files.append(audio_files)

    total_files = len(found_files)

    if total_files == 0:
        print("No valid audio files found.")
    else:
        print(f"{total_files} audio files found")

        found_files.sort()

        # Обработка каждого файла
        for audio_files in found_files:
            file_path = os.path.join(audio_input, audio_files)
            
            # Формируем команду как список аргументов
            command = [
                f"{mvsepless_dir}/uvr/venv/bin/audio-separator",
                file_path,
                "--model_filename", model,
                "--output_dir", output_folder,
                "--output_format", output_format,
                "--vr_window_size", str(window_size),
                "--vr_aggression", str(aggression),
                "--vr_batch_size", str(batch_size),
                "--model_file_dir", "./models"
            ]
            
            # Добавляем опциональные аргументы
            if tta:
                command.append("--vr_enable_tta")
            if high_end_process:
                command.append("--vr_high_end_process")
            if use_autocast:
                command.append("--use_autocast")
            
            # Выполняем команду
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Ошибка при выполнении команды: {e}")
            except FileNotFoundError as e:
                print(f"Файл или команда не найдены: {e}")

def mdx_cli(audio_input, output_folder, extensions, output_format, model, hop_length, segment_size, denoise, overlap, batch_size, use_autocast):
    found_files = []

    # Поиск файлов с нужными расширениями
    for audio_files in os.listdir(audio_input):
        if audio_files.endswith(extensions):
            found_files.append(audio_files)

    total_files = len(found_files)

    if total_files == 0:
        print("No valid audio files found.")
    else:
        print(f"{total_files} audio files found")

        found_files.sort()

        # Обработка каждого файла
        for audio_files in found_files:
            file_path = os.path.join(audio_input, audio_files)
            
            # Формируем команду как список аргументов
            command = [
                f"{mvsepless_dir}/uvr/venv/bin/audio-separator",
                file_path,
                "--model_filename", model,
                "--output_dir", output_folder,
                "--output_format", output_format,
                "--mdx_hop_length", str(hop_length),
                "--mdx_segment_size", str(segment_size),
                "--mdx_overlap", str(overlap),
                "--mdx_batch_size", str(batch_size),
                "--model_file_dir", "./models"
            ]
            
            # Добавляем опциональные аргументы
            if denoise:
                command.append("--mdx_enable_denoise")
            if use_autocast:
                command.append("--use_autocast")
            
            # Выполняем команду
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Ошибка при выполнении команды: {e}")
            except FileNotFoundError as e:
                print(f"Файл или команда не найдены: {e}")


# Основная функция
def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Обработка входных и выходных директорий и кода модели.")
    parser.add_argument('-i', '--input', dest='input_directory', required=True, help="Путь к входной директории")
    parser.add_argument('-o', '--output', dest='output_directory', required=True, help="Путь к выходной директории")
    parser.add_argument('--modelcode', dest='modelcode', type=int, required=True, help="Код модели")
    parser.add_argument("--tta", action='store_true', help="TTA")
    args = parser.parse_args()

    # Используем аргументы
    input_dir = args.input_directory
    output_dir = args.output_directory    
    model_code = args.modelcode

    if args.tta:
      tta = True
    else:
      tta = False
    
    msst_directory = "/content/MVSEPLESS/models/checkpoint/" + f"{model_code}"
    os.makedirs(msst_directory, exist_ok=True)
    config_path = msst_directory + "/config.yaml" 

    ckpt_path = msst_directory + "/checkpoint.ckpt"
    
    json_path = msst_directory + "/vocals.json" 

    pth_path = msst_directory + "/vocals.pth"
    



    new_line = f'model_code = {model_code}\n'

    class IndentDumper(yaml.Dumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(IndentDumper, self).increase_indent(flow, False)


    def tuple_constructor(loader, node):
        # Load the sequence of values from the YAML node
        values = loader.construct_sequence(node)
        # Return a tuple constructed from the sequence
        return tuple(values)

    # Register the constructor with PyYAML
    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple',
    tuple_constructor)

    def conf_edit(config_path, chunk_size, overlap):
        with open(config_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)

        # handle cases where 'use_amp' is missing from config:
        if 'use_amp' not in data.keys():
          data['training']['use_amp'] = True

        data['audio']['chunk_size'] = chunk_size
        data['inference']['num_overlap'] = overlap
        if 300 <= model_code <= 399:
          if data['inference']['batch_size'] == 1:
            data['inference']['batch_size'] = 2

        print("Using custom overlap and chunk_size values for roformer model:")
        print(f"overlap = {data['inference']['num_overlap']}")
        print(f"chunk_size = {data['audio']['chunk_size']}")
        print(f"batch_size = {data['inference']['batch_size']}")


        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, Dumper=IndentDumper, allow_unicode=True)

    def download_model():
        # Определите текст для первой строки
    
    # Прочитайте содержимое файла, если он существует
        try:
            with open('MVSEPLESS/msst_download.py', 'r+') as file:
                lines = file.readlines()
        except FileNotFoundError:
            lines = []

    # Обновите первую строку или добавьте новую
        if lines:
            lines[0] = new_line
        else:
            lines.append(new_line)

    # Запишите обновленные строки обратно в файл
        with open('MVSEPLESS/msst_download.py', 'w') as file:
            file.writelines(lines)

        from msst_download import ckpt_url 
        from msst_download import conf_url
        
        download_ckpt = ("wget", "-O", str(ckpt_path), str(ckpt_url))
        download_conf = ("wget", "-O", str(config_path), str(conf_url))
        subprocess.run(download_ckpt, check=True)
        subprocess.run(download_conf, check=True)

        if 100 <= model_code <= 299:
            overlap = 2
            chunk_size = 485100
            conf_edit(config_path, int(chunk_size), overlap)

        elif model_code == 304:
            overlap = 4
            chunk_size = 261120
            conf_edit(config_path, int(chunk_size), overlap)
        elif model_code == 303:
            overlap = 8
            chunk_size = 130560
            conf_edit(config_path, int(chunk_size), overlap)
        elif model_code == 302:
            overlap = 4
            chunk_size = 261120
            conf_edit(config_path, int(chunk_size), overlap)
        elif model_code == 301:
            chunk_size = 130560
            overlap = 4
            conf_edit(config_path, int(chunk_size), overlap)
        elif model_code == 300:
            overlap = 4
            chunk_size = 261120
            conf_edit(config_path, int(chunk_size), overlap)

    def download_model_medley_vox():
        # Определите текст для первой строки
    
    # Прочитайте содержимое файла, если он существует
        try:
            with open('MVSEPLESS/medley_vox_models.py', 'r+') as file:
                lines = file.readlines()
        except FileNotFoundError:
            lines = []

    # Обновите первую строку или добавьте новую
        if lines:
            lines[0] = new_line
        else:
            lines.append(new_line)

    # Запишите обновленные строки обратно в файл
        with open('MVSEPLESS/medley_vox_models.py', 'w') as file:
            file.writelines(lines)

        from medley_vox_models import pth_url 
        from medley_vox_models import json_url
        
        download_pth = ("wget", "-O", str(pth_path), str(pth_url))
        download_json = ("wget", "-O", str(json_path), str(json_url))
        subprocess.run(download_pth, check=True)
        subprocess.run(download_json, check=True)

    print(f"Входная директория: {input_dir}")
    print(f"Выходная директория: {output_dir}")
    print(f"Код модели: {model_code}")

    # Определение типа модели
    if 100 <= model_code <= 199:
        model_type = "mel_band_roformer"
        new = True
        inference = "msst"
        download_model()
    elif 200 <= model_code <= 299:
        model_type = "bs_roformer"
        new = True
        inference = "msst"
        download_model()
    elif 300 <= model_code <= 399:
        model_type = "mdx23c"
        new = False
        inference = "msst"
        download_model()
    elif 400 <= model_code <= 499:
        model_type = "mdx-net"
        inference = "vocalremover"
        audio_input = input_dir
        output_folder = output_dir
        selected_model = get_mdx_code(model_code)
    elif 500 <= model_code <= 599:
        model_type = "vr_arch"
        inference = "vocalremover"
        audio_input = input_dir
        output_folder = output_dir
        selected_model = get_model_by_code(model_code)
        print(f"Модель для кода {model_code}: {selected_model}")
    elif 600 <= model_code <= 699:
        model_type = "medley_vox"
        inference = "medley"
        download_model_medley_vox()
    elif 700 <= model_code <= 799:
        model_type = "htdemucs"
        inference = "vocalremover"
    elif 800 <= model_code <= 899:
        model_type = "scnet"
        new = True
        inference = "msst"
        download_model()
    else:
        print("Неверный код модели")
        return

    print(f"Тип модели: {model_type}")

    # Выполнение в зависимости от типа модели
    if inference == "msst":
        extract_instrumental = True
        flac_file = True
        infer = f"{mvsepless_dir}/msstnew/inference.py" if new else f"{mvsepless_dir}/msst/inference.py"
        msst_separate = [
            f"{mvsepless_dir}/msst/venv/bin/python", infer,
            "--model_type", str(model_type),
            "--config_path", str(config_path),
            "--start_check_point", str(ckpt_path),
            "--input_folder", str(input_dir),
            "--store_dir", str(output_dir),
        ]
        if tta:
            msst_separate.append("--use_tta")
        if extract_instrumental:
            msst_separate.append("--extract_instrumental")
        subprocess.run(msst_separate)
    elif inference == "vocalremover":
        print("VOCALREMOVER")
        if model_type == "vr_arch":
            window_size = 512
            aggression = 50
            batch_size = 1
            high_end_process = False
            use_autocast = True
            uvr_cli(input_dir, output_dir, extensions, "wav", selected_model, window_size, aggression, tta, high_end_process, batch_size, use_autocast)
        elif model_type == "mdx-net":
            hop_length = 1024 
            segment_size = 256 
            overlap = 0.25
            batch_size = 1
            denoise = True
            use_autocast = True
            mdx_cli(input_dir, output_dir, extensions, "wav", selected_model, hop_length, segment_size, denoise, overlap, batch_size, use_autocast)
    elif inference == "medley":
        print("MEDLEY-Vox")
        medley_infer = [
            f"{mvsepless_dir}/medleyvox/venv/bin/python", "-m",
            "MVSEPLESS.medleyvox.svs.inference", "--exp_name", str(model_code),
            "--model_dir=/content/MVSEPLESS/models", "--inference_data_dir", str(input_dir),
            "--results_save_dir", str(output_dir), "--use_overlapadd=ola"
        ]
        subprocess.run(medley_infer)

# Точка входа
if __name__ == "__main__":
    main()