
import gc
import os
import re
import datetime
import gradio as gr
import torch
import librosa
import tempfile
from datetime import datetime
import argparse
from vbach.infer.infer import Config, load_hubert, get_vc, rvc_infer
from vbach.utils.model_manager import model_manager
from vbach.utils.audio_utils import Audio

MAX_LENGTH_NAME = 255

audio = Audio()

OUTPUT_FORMAT = audio.output_formats
input_formats = audio.input_formats

def check_audio_file(file_path):
    if file_path.endswith(tuple([f".{of}" for of in input_formats])):
        streams = audio.get_info(file_path)
        if len(list(streams.keys())) == 0:
            return False
        else:
            return True
    else:
        return False

def clean_filename(filename, length=240):
    # Список символов, запрещенных в обеих системах
    universal_forbidden = r"\\/*?:<>|"

    # Дополнительные символы, запрещенные в Linux
    linux_forbidden = r"&;~\'`()[]$#^%!"

    # Создаем набор всех запрещенных символов
    forbidden_chars = set(universal_forbidden + linux_forbidden)

    # Удаляем запрещенные символы
    cleaned = "".join(c for c in filename if c not in forbidden_chars)

    # Удаляем пробелы в начале и конце
    cleaned = cleaned.strip()

    # Проверяем на зарезервированные имена Windows
    reserved_windows = {
        "CON",
        "AUX",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "LPT1",
        "LPT2",
        "LPT3",
        "PRN",
        "NUL",
    }

    # Если имя файла зарезервировано, добавляем префикс
    if cleaned.upper() in reserved_windows:
        cleaned = f"file_{cleaned}"
    if len(cleaned) > length:
        return f"{cleaned[:length // 2]}...{cleaned[-(length // 3):]}"
    return cleaned

def remove_duplicate_keys(input_str, keys=("NAME", "MODEL", "PITCH", "F0METHOD", "DATETIME")):
    # Создаем множество для отслеживания найденных ключей
    seen = set()
    # Шаблон для поиска любого из ключей
    pattern = r"({})".format("|".join(re.escape(key) for key in keys))

    def replace(match):
        key = match.group(1)
        if key in seen:
            return ""  # Удаляем дубликат
        seen.add(key)
        return key  # Оставляем первое вхождение

    # Заменяем дубликаты на пустую строку
    result = re.sub(pattern, replace, input_str)
    return result


def shorter_name(template, file_name, pitch, method_pitch, model_name, date_time):
    # Удаляем дубликаты ключей в шаблоне перед расчетами
    clean_template = remove_duplicate_keys(template)

    str_pitch = str(pitch)
    template_no_keys_length = len(
        clean_template.replace("NAME", "")
        .replace("MODEL", "")
        .replace("PITCH", "")
        .replace("F0METHOD", "")
        .replace("DATETIME", "")
    )
    key_values_length = (
        len(model_name)
        if "MODEL" in clean_template
        else (
            0 + len(method_pitch)
            if "F0METHOD" in clean_template
            else 0 + len(str_pitch) if "PITCH" in clean_template else 0 + len(date_time) if "DATETIME" in clean_template else 0
        )
    )
    free_length = MAX_LENGTH_NAME - (template_no_keys_length + key_values_length)
    if len(file_name) > (free_length - 7):
        shorted_name = f"{file_name[:(free_length // 2)]}...{file_name[-((free_length // 2) - 7):]}"
        return shorted_name
    else:
        return file_name

def output_file_template(template, file_name, pitch, method_pitch, model_name):
    
    clean_template = remove_duplicate_keys(template)

    time_create_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    input_file_name = shorter_name(
        clean_template, file_name, pitch, method_pitch, model_name, time_create_file
    )
    template_name = (
        clean_template.replace("MODEL", f"{model_name}")
        .replace("DATETIME", f"{time_create_file}")
        .replace("PITCH", f"{pitch}")
        .replace("F0METHOD", f"{method_pitch}")
        .replace("NAME", f"{input_file_name}")
    )
    output_name = f"{template_name}"
    return output_name

def short_name(name):
    if len(name) > (MAX_LENGTH_NAME - 15):
        return f"{name[:150]}...{name[-20:]}"
    else:
        return name

def load_rvc_model(voice_model):

    if voice_model in model_manager.parse_voice_models():
        rvc_model_path, rvc_index_path = model_manager.parse_pth_and_index(voice_model)

        if not rvc_model_path:
            raise ValueError(
                f"[91mФайла для модели {voice_model} не существует. "
                "Возможно, вы неправильно её установили.[0m"
            )

    else:
        raise ValueError(
            f"[91mМодели {voice_model} не существует. "
            "Возможно, вы неправильно ввели имя.[0m"
        )

    return rvc_model_path, rvc_index_path

def voice_conversion(
    voice_model,
    vocals_path,
    output_path,
    pitch,
    f0_method,
    index_rate,
    filter_radius,
    volume_envelope,
    protect,
    hop_length,
    f0_min,
    f0_max,
    format_output,
    output_bitrate,
    stereo_mode,
    hubert_path=None
):
    rvc_model_path, rvc_index_path = load_rvc_model(voice_model)

    config = Config()
    hubert_model = load_hubert(config.device, config.is_half, hubert_path if hubert_path else model_manager.hubert_path)
    cpt, version, net_g, tgt_sr, vc = get_vc(
        config.device, config.is_half, config, rvc_model_path
    )

    output_audio = rvc_infer(
        rvc_index_path,
        index_rate,
        vocals_path,
        output_path,
        pitch,
        f0_method,
        cpt,
        version,
        net_g,
        filter_radius,
        tgt_sr,
        volume_envelope,
        protect,
        hop_length,
        vc,
        hubert_model,
        f0_min,
        f0_max,
        format_output,
        output_bitrate,
        stereo_mode
    )

    del hubert_model, cpt, net_g, vc
    gc.collect()
    torch.cuda.empty_cache()
    return output_audio

def cli_conversion(input_audios, template="NAME_MODEL_F0METHOD_PITCH", output_dir="output", model_name="", index_rate=0, output_format="wav", stereo_mode="mono", method_pitch="rmvpe+", pitch=0, hop_length=128, filter_radius=3, rms=0.25, protect=0.33, f0_min=50, f0_max=1100, hubert_path=None):
    if not input_audios:
        raise ValueError(
            "Не удалось найти аудиофайл(ы). "
            "Убедитесь, что файл загрузился или проверьте правильность пути к нему."
        )
    if not model_name:
        raise ValueError("Выберите модель голоса для преобразования.")
    if not os.path.exists(input_audios):
        raise ValueError(f"Файл {input_audios} не найден.")

    if not os.path.exists(input_audios):
        raise FileNotFoundError(f"Ошибка: '{input_audios}' не существует.")

    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(input_audios):
        if not check_audio_file(input_audios):
            raise ValueError(f"Ошибка: '{input_audios}' не является аудиофайлом.")
        print(f"Найден аудиофайл: {input_audios}")

        try:
            file_name = os.path.basename(input_audios)
            namefile = os.path.splitext(file_name)[0]
            time_create_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            template = clean_filename(template, length=200)
            output_name = template
            output_path = os.path.join(output_dir, f"{output_name}.{output_format}")
            voice_conversion(voice_model=model_name, 
                             vocals_path=input_audios, 
                             output_path=output_path, 
                             pitch=pitch, 
                             f0_method=method_pitch, 
                             index_rate=index_rate, 
                             filter_radius=filter_radius, 
                             volume_envelope=rms, 
                             protect=protect, 
                             hop_length=hop_length, 
                             f0_min=f0_min, 
                             f0_max=f0_max, 
                             format_output=output_format, 
                             output_bitrate="320k", 
                             stereo_mode=stereo_mode, 
                             hubert_path=hubert_path)
        finally:
            if os.path.exists(output_path):
                print("Вокал успешно преобразован")  
    
    elif os.path.isdir(input_audios):
        # Ищем аудиофайлы в папке
        list_files = []
        for file in os.listdir(input_audios):
            abs_path_file = os.path.join(input_audios, file)
            if os.path.isfile(abs_path_file) and check_audio_file(abs_path_file):
                list_files.append(abs_path_file)

        if not list_files:
            raise FileNotFoundError(f"Ошибка: в папке '{input_audios}' нет аудиофайлов.")

        print(f"Найдены аудиофайлы: {list_files}")

        try:
            output_paths = []
            for i, file in enumerate(list_files):
                print(f"Файл {i + 1}/{len(list_files)}")
                file_name = os.path.basename(file)
                namefile = os.path.splitext(file_name)[0]
                template = clean_filename(template, length=50)
                if "NAME" not in template:
                    template = f"{template}_NAME"
                output_name = output_file_template(template, namefile, pitch, method_pitch, model_name)
                output_path = os.path.join(output_dir, f"{output_name}.{output_format}")
                voice_conversion(voice_model=model_name, 
                                vocals_path=file, 
                                output_path=output_path, 
                                pitch=pitch, 
                                f0_method=method_pitch, 
                                index_rate=index_rate, 
                                filter_radius=filter_radius, 
                                volume_envelope=rms, 
                                protect=protect, 
                                hop_length=hop_length, 
                                f0_min=f0_min, 
                                f0_max=f0_max, 
                                format_output=output_format, 
                                output_bitrate="320k", 
                                stereo_mode=stereo_mode, 
                                hubert_path=hubert_path)
                if os.path.exists(output_path):
                    output_paths.append(output_path)
                
        finally:
            print("Вокалы успешно преобразованы")     
    else:
        raise ValueError(f"Ошибка: '{input_audios}' не является ни файлом, ни папкой.")

def setup_args():
    parser = argparse.ArgumentParser(description='Vbach CLI')
    
    # Обязательные аргументы
    parser.add_argument(
        'input_audios',
        type=str,
        help='Путь к аудиофайлу или папке с аудиофайлами для обработки'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Папка для сохранения результатов конвертации'
    )
    parser.add_argument(
        'model_name',
        type=str,
        help='Название голосовой модели RVC для преобразования'
    )
    
    # Необязательные аргументы с значениями по умолчанию
    parser.add_argument(
        '--output_name',
        type=str,
        default="",
        help='Имя выходного файла (доступные замены: DATETIME, NAME, MODEL, F0METHOD, PITCH)'
    )
    parser.add_argument(
        '--index_rate',
        type=float,
        default=0,
        help='Интенсивность использования индексного файла (от 0.0 до 1.0)',
        metavar='[0.0-1.0]'
    )
    parser.add_argument(
        '--output_format',
        type=str,
        default="wav",
        choices=OUTPUT_FORMAT,
        help='Формат выходного аудиофайла'
    )
    parser.add_argument(
        '--stereo_mode',
        type=str,
        default="mono",
        choices=["mono", "left/right", "sim/dif"],
        help='Режим каналов: моно или стерео'
    )
    parser.add_argument(
        '--method_pitch',
        type=str,
        default="rmvpe+",
        help='Метод извлечения pitch (тона)'
    )
    parser.add_argument(
        '--pitch',
        type=int,
        default=0,
        help='Корректировка тона в полутонах'
    )
    parser.add_argument(
        '--hop_length',
        type=int,
        default=128,
        help='Длина hop (в семплах) для обработки'
    )
    parser.add_argument(
        '--filter_radius',
        type=int,
        default=3,
        help='Радиус фильтра для сглаживания'
    )
    parser.add_argument(
        '--rms',
        type=float,
        default=0.25,
        help='Масштабирование огибающей громкости (RMS)'
    )
    parser.add_argument(
        '--protect',
        type=float,
        default=0.33,
        help='Защита для глухих согласных звуков'
    )
    parser.add_argument(
        '--f0_min',
        type=int,
        default=50,
        help='Минимальная частота pitch (F0) в Hz'
    )
    parser.add_argument(
        '--f0_max',
        type=int,
        default=1100,
        help='Максимальная частота pitch (F0) в Hz'
    )
    parser.add_argument(
        '--hubert_path',
        type=str,
        default="",
        help='Путь к hubert'
    )
    return parser.parse_args()

# Пример использования:
if __name__ == "__main__":
    args = setup_args()
    cli_conversion(
        input_audios=args.input_audios,
        output_dir=args.output_dir,
        model_name=args.model_name,
        template=args.output_name,
        index_rate=args.index_rate,
        output_format=args.output_format,
        stereo_mode=args.stereo_mode,
        method_pitch=args.method_pitch,
        pitch=args.pitch,
        hop_length=args.hop_length,
        filter_radius=args.filter_radius,
        rms=args.rms,
        protect=args.protect,
        f0_min=args.f0_min,
        f0_max=args.f0_max,
        hubert_path=args.hubert_path
    )



