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
from demucs_models import get_demucs_code
# Импортируем только функцию

# Определяем расширения файлов
extensions = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aiff", ".ac3")

def find_medley_out_directory(base_path="/tmp/medley_vox_out"):
    # Проверяем, существует ли базовая директория
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Директория {base_path} не найдена")

    # Ищем поддиректории в базовой директории
    for root, dirs, files in os.walk(base_path):
        if dirs:  # Если есть поддиректории
            medley_out = os.path.join(root, dirs[0])  # Берем первую найденную директорию
            return medley_out

    # Если директории не найдены
    raise FileNotFoundError(f"В директории {base_path} не найдено поддиректорий")

def convert_mp3(directory):
    """
    Конвертирует все WAV-файлы в директории в MP3 с битрейтом 320 kbit/sec
    и удаляет исходные WAV-файлы.
    
    :param directory: Путь к директории с WAV-файлами.
    """
    # Проверяем, существует ли директория
    if not os.path.exists(directory):
        print(f"Директория {directory} не существует.")
        return

    # Проходим по всем файлам в директории
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            # Полный путь к WAV-файлу
            wav_path = os.path.join(directory, filename)
            
            # Создаем имя для MP3-файла
            mp3_filename = filename.replace(".wav", ".mp3")
            mp3_path = os.path.join(directory, mp3_filename)
            
            # Команда для конвертации через ffmpeg
            command = [
                "ffmpeg",
                "-i", wav_path,  # Входной файл
                "-b:a", "320k",  # Битрейт 320 kbit/sec
                mp3_path        # Выходной файл
            ]
            
            # Выполняем команду
            try:
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"Конвертирован {wav_path} в {mp3_path}.")
                
                # Удаляем исходный WAV-файл
                os.remove(wav_path)
                print(f"Удален исходный WAV-файл: {wav_path}.")
                
            except subprocess.CalledProcessError as e:
                print(f"Ошибка при конвертации {wav_path}: {e.stderr.decode('utf-8')}")

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

import os
import subprocess

def demucs_cli(audio_input, output_folder, extensions, output_format, model, shifts, segment_size, overlap, use_autocast):
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
            
            if model == "drumsep":
                print(f"Processing file: {audio_files}")
                command = [
                    f"{mvsepless_dir}/uvr/venv/bin/demucs",
                    "--repo", "/content/MVSEPLESS",
                    "--shifts", str(shifts),
                    "--overlap", str(overlap),
                    "-o", output_folder,
                    "-n", "drumsep",
                    file_path
                ]
            else:
                command = [
                    f"{mvsepless_dir}/uvr/venv/bin/audio-separator",
                    file_path,
                    "--model_filename", model,
                    "--output_dir", output_folder,
                    "--output_format", output_format,
                    "--demucs_shifts", str(shifts),
                    "--demucs_overlap", str(overlap),
                    "--demucs_segment_size", str(segment_size),
                    "--model_file_dir", "./models"
                ]
                if use_autocast:
                    command.append("--use_autocast")
            
            # Выполняем команду
            try:
                subprocess.run(command, check=True)
                print(f"File: {audio_files} processed!")
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
    parser.add_argument("--output_format", type=str, choices=['mp3', 'wav', 'flac'], default='wav', help="Output format")
    args = parser.parse_args()

    # Используем аргументы
    input_dir = args.input_directory
    output_dir = args.output_directory    
    model_code = args.modelcode

    os.makedirs(output_dir, exist_ok=True)

    if args.tta:
      tta = True
    else:
      tta = False
    
    if args.output_format == "mp3":
      format_audio = "mp3"
    elif args.output_format == "flac":
      format_audio = "flac"
    elif args.output_format == "wav":
      format_audio = "wav"

    def convert_medley(medley_out, output_dir):
        if not os.path.exists(medley_out):
            print(f"Директория не существует.")
            return

    # Проходим по всем файлам в директории
        for filename in os.listdir(medley_out):
            if filename.endswith(".wav"):
            # Полный путь к WAV-файлу
                wav_path = os.path.join(medley_out, filename)
            
            # Создаем имя для MP3-файла
                if format_audio == "mp3":
                    mp3_filename = filename.replace(".wav", ".mp3")
                elif format_audio == "flac":
                    mp3_filename = filename.replace(".wav", ".flac")
                elif format_audio == "wav":
                    mp3_filename = filename.replace(".wav", ".wav")
                mp3_path = os.path.join(output_dir, mp3_filename)
            
            # Команда для конвертации через ffmpeg
                if format_audio == "wav":
                    command = [
                        "cp",
                        "-r", wav_path,  # Входной файл
                        mp3_path        # Выходной файл
                    ]
                if format_audio == "flac":
                    command = [
                        "ffmpeg",
                        "-i", wav_path,  # Входной файл
                        "-af", "aformat=s16:44100",  # Битрейт 320 kbit/sec
                        mp3_path        # Выходной файл
                    ]            
                else:
                    command = [
                        "ffmpeg",
                        "-i", wav_path,  # Входной файл
                        "-b:a", "320k",  # Битрейт 320 kbit/sec
                        mp3_path        # Выходной файл
                    ]
            
                # Выполняем команду
                try:
                    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print(f"Конвертирован {wav_path} в {mp3_path}.")
                
                    # Удаляем исходный WAV-файл
                    os.remove(wav_path)
                    print(f"Удален исходный WAV-файл: {wav_path}.")
                    
                except subprocess.CalledProcessError as e:
                    print(f"Ошибка при конвертации {wav_path}: {e.stderr.decode('utf-8')}")



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

        if 200 <= model_code <= 299:
            overlap = 2
            chunk_size = 485100
            conf_edit(config_path, int(chunk_size), overlap)

        if 1000 <= model_code <= 1999:
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
    if 1000 <= model_code <= 1999:
        model_type = "mel_band_roformer"
        inference = "msst"
        new = True
        download_model()
    elif 200 <= model_code <= 299:
        model_type = "bs_roformer"
        new = True
        inference = "msst"
        if model_code == 200:
          raise("Unsupported model for inference. Wait fix...")
        else:
          new = True
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
        selected_model = get_demucs_code(model_code)
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
        if format_audio == "flac":
            msst_separate.append("--flac_file")
        subprocess.run(msst_separate)
        if format_audio == "mp3":
            convert_mp3(output_dir)   
    elif inference == "vocalremover":
        print("VOCALREMOVER")
        if model_type == "vr_arch":
            window_size = 512
            aggression = 50
            batch_size = 1
            high_end_process = False
            use_autocast = True
            uvr_cli(input_dir, output_dir, extensions, format_audio, selected_model, window_size, aggression, tta, high_end_process, batch_size, use_autocast)
        elif model_type == "mdx-net":
            hop_length = 1024 
            segment_size = 256 
            overlap = 0.25
            batch_size = 1
            denoise = True
            use_autocast = True
            mdx_cli(input_dir, output_dir, extensions, format_audio, selected_model, hop_length, segment_size, denoise, overlap, batch_size, use_autocast)
        elif model_type == "htdemucs":
            shifts = 2
            overlap = 0.025
            segment_size = 40
            use_autocast = True
            demucs_cli(input_dir, output_dir, extensions, format_audio, selected_model, shifts, segment_size, overlap, use_autocast)
    elif inference == "medley":
        print("MEDLEY-Vox")
        temp_voxes = "/tmp/medley_vox_out"
        os.makedirs(temp_voxes, exist_ok=True)
        medley_infer = [
            f"{mvsepless_dir}/medleyvox/venv/bin/python", "-m",
            "MVSEPLESS.medleyvox.svs.inference", "--exp_name", str(model_code),
            "--model_dir=/content/MVSEPLESS/models", "--inference_data_dir", str(input_dir),
            "--results_save_dir", "/tmp/medley_vox_out", "--use_overlapadd=ola"
        ]
        subprocess.run(medley_infer)
        medley_out = find_medley_out_directory()
        convert_medley(medley_out, output_dir)
        rm_temp = ("rm", "-rf", "/tmp/medley_vox_out")
        subprocess.run(rm_temp)
# Точка входа
if __name__ == "__main__":
    main()
