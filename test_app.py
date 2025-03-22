import gradio as gr
import os
import subprocess
import shutil
from datetime import datetime

rvc_models_dir = "voice_models"

output_dir_rvc = "/content/voice_output"
output_dir_uvr = "/content/output"

def conversion_vocals(input_file, pitch, model_name, index_rate, filter_radius, rms, protect, output_format, hop_length, method_pitch):
    # Создаем директорию для входных файлов
    # Получаем путь к временному файлу
    temp_path = input_file  # Gradio возвращает путь как строку
    input_filename = os.path.basename(temp_path)
    output_dir = "/content/voice_output"
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    custom_name = f"converted_voice_{model_name}_{method_pitch}_{pitch}"
    output_name = os.path.join(output_dir, f"{custom_name}.{output_format}")
    # Формируем команду
    command = [
        "python", "-m", "rvc.cli.rvc_cli",
        "-i", str(temp_path), "--o", str(output_dir),
        "-m", str(model_name), "--format", 
        str(output_format), "--custom_name", str(custom_name),
        "-ir", str(index_rate), "-fr", str(filter_radius),
        "-rms", str(rms), "-p", str(pitch),
        "-pro", str(protect), "-hop", str(hop_length),
        "-f0", str(method_pitch)
    ]
    subprocess.run(command)
    return output_name
    

def update_audio_players(history_list):
    # Получаем список аудиофайлов в папке
    audio_folder = os.path.join(f"{output_dir_uvr}", f"{history_list}")
    audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith((".wav", ".mp3", ".flac"))]
    
    # Ограничиваем количество файлов до 7 (или любого другого числа)
    audio_files = audio_files[:7]
    # Список для хранения видимости каждой строки
    visibility = [False] * 7
    audio_paths = [None] * 7  # Список для хранения путей к аудиофайлам
    
    # Проверяем, сколько аудиофайлов передано
    for i in range(len(audio_files)):
        if i < 7:  # Убедимся, что не выходим за пределы количества строк
            visibility[i] = True
            audio_paths[i] = audio_files[i]  # Используем путь к файлу напрямую
    
    # Возвращаем обновленные параметры для каждого компонента
    return (
        gr.update(visible=visibility[0], value=audio_paths[0]),
        gr.update(visible=visibility[1], value=audio_paths[1]),
        gr.update(visible=visibility[2], value=audio_paths[2]),
        gr.update(visible=visibility[3], value=audio_paths[3]),
        gr.update(visible=visibility[4], value=audio_paths[4]),
        gr.update(visible=visibility[5], value=audio_paths[5]),
        gr.update(visible=visibility[6], value=audio_paths[6]),
    )

# Функция для выполнения команды разделения
def separate_audio(input_file, separation_type, model, output_format):
    # Создаем директорию для входных файлов
    # Получаем путь к временному файлу
    temp_path = input_file.name  # Gradio возвращает путь как строку
    input_filename = os.path.basename(temp_path)
    output_dir = os.path.join("/content/output", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    # Получаем код модели
    model_code = model_mapping[separation_type][model]
    
    # Формируем команду
    command = [
        "python", "code_infer.py",
        "--input", str(temp_path), "--output", str(output_dir),
        "--modelcode", str(model_code), "--output_format", str(output_format), "-inst"
    ]
    subprocess.run(command)
    
    audio_folder = output_dir
    audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith((".wav", ".mp3", ".flac"))]
    
    # Ограничиваем количество файлов до 7 (или любого другого числа)
    audio_files = audio_files[:7]
    # Список для хранения видимости каждой строки
    visibility = [False] * 7
    audio_paths = [None] * 7  # Список для хранения путей к аудиофайлам
    
    # Проверяем, сколько аудиофайлов передано
    for i in range(len(audio_files)):
        if i < 7:  # Убедимся, что не выходим за пределы количества строк
            visibility[i] = True
            audio_paths[i] = audio_files[i]  # Используем путь к файлу напрямую
    
    # Возвращаем обновленные параметры для каждого компонента
    return (
        gr.update(visible=visibility[0], value=audio_paths[0]),
        gr.update(visible=visibility[1], value=audio_paths[1]),
        gr.update(visible=visibility[2], value=audio_paths[2]),
        gr.update(visible=visibility[3], value=audio_paths[3]),
        gr.update(visible=visibility[4], value=audio_paths[4]),
        gr.update(visible=visibility[5], value=audio_paths[5]),
        gr.update(visible=visibility[6], value=audio_paths[6]),
        [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(output_format)]
    )

    
def separate_backs():
    # Создаем директорию для входных файлов
    print("Скоро...")
    return input_file

def get_models_list():
    """Получение списка моделей."""
    models = []
    if os.path.exists(rvc_models_dir):
        models = [d for d in os.listdir(rvc_models_dir) if os.path.isdir(os.path.join(rvc_models_dir, d))]
    return models

def get_history_list(output_dir):
    """Получение списка моделей."""
    history = []
    if os.path.exists(output_dir):
        history = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    return history

# Словарь для сопоставления моделей и их кодов
model_mapping = {
    "Инструментал/Вокал": {
        "MelBand Roformer / GaboxR67 / Instrumental Fullness v1": 1100,
        "MelBand Roformer / GaboxR67 / Instrumental Fullness v2": 1101,
        "MelBand Roformer / GaboxR67 / Instrumental Fullness v3": 1102,
        "MelBand Roformer / GaboxR67 / Instrumental Fullness v4 Noise": 1103,
        "MelBand Roformer / GaboxR67 / Instrumental Fullness v5": 1104,
        "MelBand Roformer / GaboxR67 / Instrumental Fullness v5 Noise": 1105,
        "MelBand Roformer / GaboxR67 / Instrumental Fullness v6": 1106,
        "MelBand Roformer / GaboxR67 / Instrumental Fullness v6 Noise": 1107,
        "MelBand Roformer / GaboxR67 / Instrumental Fullness v7": 1108,
        "MelBand Roformer / GaboxR67 / Instrumental Fullness v7 Noise": 1109,
        "MelBand Roformer / GaboxR67 / Instrumental Fullness v8": 1110,
        "MelBand Roformer / GaboxR67 / Instrumental Fullness v9": 1111,
        "MelBand Roformer / GaboxR67 / Instrumental Fullness v10": 1112,
        "MelBand Roformer / GaboxR67 / Instrumental Fullness vX": 1130,
        "MelBand Roformer / GaboxR67 / Instrumental Bleedless v1": 1150,
        "MelBand Roformer / GaboxR67 / Instrumental Bleedless v2": 1151,
        "MelBand Roformer / GaboxR67 / Vocals Fullness v1": 1170,
        "MelBand Roformer / GaboxR67 / Vocals Fullness v2": 1171,
        "MelBand Roformer / GaboxR67 / Vocals Fullness v3": 1172,
        "MelBand Roformer / GaboxR67 / Vocals Fullness v4": 1173,
        "MelBand Roformer / Becruily / Instrumental": 1016,
        "MelBand Roformer / Becruily / Vocals": 1017,
        "MelBand Roformer / Unwa / Instrumental v1": 1013,
        "MelBand Roformer / Unwa / Instrumental v1e": 1002,
        "MelBand Roformer / Unwa / Instrumental v2": 1014,
        "MelBand Roformer / Unwa / Big Beta v1": 1008,
        "MelBand Roformer / Unwa / Big Beta v2": 1009,
        "MelBand Roformer / Unwa / Big Beta v3": 1010,
        "MelBand Roformer / Unwa / Big Beta v4": 1011,
        "MelBand Roformer / Unwa / Big Beta v5e": 1003,
        "MelBand Roformer / Unwa / Kim FT v1": 1001,
        "MelBand Roformer / Unwa / Kim FT v2": 1006,
        "MelBand Roformer / Unwa / Kim FT v2 Bleedless": 1007,
        "MelBand Roformer / Unwa / Small v1": 1015,
        "MelBand Roformer / Unwa / Instrumental & Vocals Duality v1": 1004,
        "MelBand Roformer / Unwa / Instrumental & Vocals Duality v2": 1005,
        "MelBand Roformer / KimberlyJSN / Vocals": 1000,
        "MelBand Roformer / SYH99999 / SYHFT v1": 1018,
        "MelBand Roformer / SYH99999 / SYHFT v2": 1019,
        "MelBand Roformer / SYH99999 / SYHFT v2.5": 1020,
        "MelBand Roformer / SYH99999 / SYHFT v3": 1021,
        "MelBand Roformer / SYH99999 / SYHFT Big v1 FAST Vocals": 1022,
        "MelBand Roformer / SYH99999 / SYHFT Merged Beta v1 Vocals": 1023,
        "MelBand Roformer / SYH99999 / SYHFT B1 Vocals": 1024,
        "MelBand Roformer / ViperX / Vocals": 1054,
        "BS Roformer / ViperX / Vocals": 202,
        "BS Roformer / GaboxR67 / Vocals": 203,
        "MDX23C / INST-VOC HQ": 300,
        "MDX-NET / INST-FULL": 400,
        "MDX-NET / INST 187 beta": 401,
        "MDX-NET / INST 82 beta": 402,
        "MDX-NET / INST 90 beta": 403,
        "MDX-NET / MAIN 340": 404,
        "MDX-NET / MAIN 390": 405,
        "MDX-NET / MAIN 406": 406,
        "MDX-NET / MAIN 427": 407,
        "MDX-NET / MAIN 438": 408,
        "MDX-NET / INST-HQ 1": 409,
        "MDX-NET / INST-HQ 2": 410,
        "MDX-NET / INST-HQ 3": 411,
        "MDX-NET / INST-HQ 4": 412,
        "MDX-NET / INST-HQ 5": 413,
        "MDX-NET / MAIN": 414,
        "MDX-NET / 1": 416,
        "MDX-NET / 2": 417,
        "MDX-NET / 3": 418,
        "MDX-NET / INST 1": 419,
        "MDX-NET / INST 2": 420,
        "MDX-NET / INST 3": 421,
        "MDX-NET / VOC FT": 425,
        "MDX-NET / KIM VOCAL 1": 426,
        "MDX-NET / KIM VOCAL 2": 427,
        "MDX-NET / KIM INST": 428,
        "VR ARCH / 1_HP": 500,
        "VR ARCH / 2_HP": 501,
        "VR ARCH / 3_HP_Vocal": 502,
        "VR ARCH / 4_HP_Vocal": 503,
        "VR ARCH / 7_HP2": 506,
        "VR ARCH / 8_HP2": 507,
        "VR ARCH / 9_HP2": 508,
        "VR ARCH / 12_SP": 511,
        "VR ARCH / 13_SP": 512,
        "VR ARCH / 14_SP": 513,
        "VR ARCH / 15_SP_MID": 514,
        "VR ARCH / 16_SP_MID": 515,
    },
    "Лид/Бэки": {
        "MelBand Roformer / GaboxR67 / Karaoke 25.02.2024": 1190,
        "MelBand Roformer / GaboxR67 / Karaoke 28.02.2024": 1191,
        "MelBand Roformer / Aufr33 & ViperX / Karaoke": 1054,
        "MelBand Roformer / Sucial / Male-Female 146 epoch": 206,
        "MelBand Roformer / Sucial / Male-Female 267 epoch": 207,
        "MelBand Roformer / Aufr33 / Male-Female": 208,
        "MDX23C / Wesleyr36 / Mid-Side": 303,
        "MDX-NET / Karaoke v1": 422,
        "MDX-NET / Karaoke v2": 423,
        "VR ARCH / Karaoke v1 (5_HP)": 504,
        "VR ARCH / Karaoke v2 (6_HP)": 505,
        "VR ARCH / BVE": 524,
        "Medley-Vox / Vocals 238 epochs": 609,
        "Medley-Vox / Multi Singing Librispeech 138 epochs": 601,
        "Medley-Vox / Singing Librispeech FT iSRNet": 602,
    },
    "Мультитрек": {
        "MelBand Roformer / SYH99999 / 4 Stems FT Large": 1025,
        "MelBand Roformer / SYH99999 / 4 Stems FT Large v2": 1026,
        "BS Roformer / SYH99999 / 4 Stems FT": 205,
        "BS Roformer / ZFTurbo / 4 Stems MUSDB18": 204,
        "MDX23C / ZFTurbo / 4 Stems MUSDB18": 304,
        "MDX23C / Aufr33 & Jarredou / Drumsep 6 Stems": 301,
        "SCNET / ZFTurbo / 4 Stems MUSDB18": 800,
        "SCNET / Starrytong / 4 Stems MUSDB18 XL": 801,
        "SCNET / ZFTurbo / 4 Stems MUSDB18 Large": 802,
        "HT DEMUCS / FT 4 Stems": 90,
        "HT DEMUCS / 4 Stems": 91,
        "HT DEMUCS / 6 Stems": 93,
    },
    "Реверб & Эхо": {
        "MelBand Roformer / Sucial / Dereverb Deecho": 1041,
        "MelBand Roformer / Sucial / Big Dereverb": 1042,
        "MelBand Roformer / Sucial / Super Big Dereverb": 1043,
        "MelBand Roformer / Sucial / Dereverb Deecho MBR Fused v1": 1044,
        "MelBand Roformer / Sucial / Dereverb Deecho MBR v2": 1045,
        "MelBand Roformer / Anvuew / Dereverb": 1030,
        "MelBand Roformer / Anvuew / Dereverb Aggressive": 1031,
        "MelBand Roformer / Anvuew / Dereverb Mono": 1032,
        "BS Roformer / Anvuew / Dereverb 256 Dim 8 Depth": 209,
        "BS Roformer / Anvuew / Dereverb 384 Dim 10 Depth": 210,
        "MDX23C / Aufr33 & Jarredou / Dereverb": 302,
        "MDX-NET / FoxJoy / Dereverb": 429,
        "VR ARCH / Deecho Normal": 518,
        "VR ARCH / Deecho Aggressive": 517,
        "VR ARCH / Deecho Dereverb": 519,
        "VR ARCH / Aufr33 & Jarredou / Dereverb": 520,
    },
    "Шумы": {
        "MelBand Roformer / Aufr33 & ViperX / Decrowd": 1053,
        "MelBand Roformer / Aufr33 / Denoise": 1052,
        "MelBand Roformer / GaboxR67 / Denoise Debleed": 1199,
        "MDX-NET / Decrowd": 430,
        "VR ARCH / Denoise Lite": 522,
        "VR ARCH / Denoise": 523,
    },
}

# Функция для обновления списка моделей в зависимости от типа разделения
# ... (остальной код без изменений, кроме функции update_models)

# Функция для обновления списка моделей в зависимости от типа разделения
def update_models(separation_type):
    return gr.Dropdown(choices=list(model_mapping[separation_type].keys()))
# Создаем интерфейс Gradio с правильным обновлением моделей
with gr.Blocks(title="Разделение музыки и голоса", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# MVSEPLESS")

    with gr.Tabs():
        # Первая вкладка
        with gr.TabItem("Разделение вокала"):
            with gr.Row():
                input_file = gr.File(label="Перетащите, чтобы загрузить файл на сервер или выберите файл на устройстве", file_types=[".wav", ".mp3", ".flac"])

            with gr.Row():
                separation_type = gr.Dropdown(
                    label="Тип разделения", 
                    choices=list(model_mapping.keys()), 
                    value="Инструментал/Вокал",
                    interactive=True
                )
            with gr.Row():    
                model = gr.Dropdown(
                    label="Модель", 
                    choices=list(model_mapping["Инструментал/Вокал"].keys()),
                    interactive=True
                )
            with gr.Row():
                output_format = gr.Dropdown(
                    label="Формат вывода", 
                    choices=["wav", "mp3", "flac"], 
                    value="wav",
                    interactive=True
                )

            btn = gr.Button("Разделить", variant="primary")

            with gr.Row():
                with gr.Accordion("Результаты разделения", open=False):
                    stem_1 = gr.Audio(type="filepath", interactive=False, visible=False)
                    stem_2 = gr.Audio(type="filepath", interactive=False, visible=False)
                    stem_3 = gr.Audio(type="filepath", interactive=False, visible=False)
                    stem_4 = gr.Audio(type="filepath", interactive=False, visible=False)
                    stem_5 = gr.Audio(type="filepath", interactive=False, visible=False)
                    stem_6 = gr.Audio(type="filepath", interactive=False, visible=False)
                    stem_7 = gr.Audio(type="filepath", interactive=False, visible=False)
                output_file = gr.File(label="Файлы после разделения", interactive=False)
    # Обновляем видимость строк и аудиофайлы при загрузке файлов
           
            # Обработчик изменения типа разделения
            separation_type.change(
                fn=update_models,
                inputs=separation_type,
                outputs=model
            )

            # Обработчик кнопки
            btn.click(
                fn=separate_audio,
                inputs=[input_file, separation_type, model, output_format],
                outputs=[stem_1, stem_2, stem_3, stem_4, stem_5, stem_6, stem_7, output_file]
            )

        with gr.TabItem("Замена вокала"):
            with gr.Row():
                file_input = gr.Audio(label="Загрузить аудио", type="filepath")
                    
            with gr.Row():
                voicemodel_name = gr.Dropdown(
                    choices=list(get_models_list()), 
                    label="Имя модели", 
                    value="senko",
                    interactive=True
                )
                refresh_btn = gr.Button("Обновить")

            with gr.Accordion("Настройки RVC:", open=True):
                pitch_vocal = gr.Slider(-48, 48, value=0, step=12, label="Высота тона", interactive=True)
                method_pitch = gr.Dropdown(
                    label="Метод извлечения тона", 
                    choices=["rmvpe+", "mangio-crepe", "fcpe"], 
                    value="rmvpe+",
                    interactive=True
                )
                hop_length = gr.Slider(0, 255, value=73, step=1, label="Длина шага для mangio-crepe", interactive=True)
                index_rate = gr.Slider(0, 1, value=1, step=0.05, label="ИИ-акцент", interactive=True)
                filter_radius = gr.Slider(0, 7, value=7, step=1, label="Радиус фильтра", interactive=True)
                rms = gr.Slider(0, 1, value=0, step=0.1, label="Нормализация", interactive=True)
                protect = gr.Slider(0, 0.5, value=0.35, step=0.05, label="Защита согласных", interactive=True)
            with gr.Row():
                output_format_rvc = gr.Dropdown(
                    label="Формат вывода", 
                    choices=["wav", "mp3", "flac"], 
                    interactive=True
                )
            with gr.Row():
                convert_btn = gr.Button("Преобразовать!", variant="primary")
                
            with gr.Column():
                converted_voice = gr.Audio(type="filepath", interactive=False, visible=True)

            # Обработчики
            refresh_btn.click(
                fn=lambda: gr.update(choices=get_models_list()),
                outputs=voicemodel_name
            )
            
            convert_btn.click(
                fn=conversion_vocals,
                inputs=[file_input, pitch_vocal, voicemodel_name, index_rate, filter_radius, rms, protect, output_format_rvc, hop_length, method_pitch],
                outputs=converted_voice
            )

        with gr.TabItem("История"):
            with gr.Accordion("История разделений", open=True):
                history_list = gr.Dropdown(
                    choices=list(get_history_list(output_dir_uvr)), 
                    label="Задание", 
                    interactive=True
                )
                with gr.Accordion("Результаты разделения", open=False):
                    stem_1 = gr.Audio(type="filepath", interactive=False, visible=False)
                    stem_2 = gr.Audio(type="filepath", interactive=False, visible=False)
                    stem_3 = gr.Audio(type="filepath", interactive=False, visible=False)
                    stem_4 = gr.Audio(type="filepath", interactive=False, visible=False)
                    stem_5 = gr.Audio(type="filepath", interactive=False, visible=False)
                    stem_6 = gr.Audio(type="filepath", interactive=False, visible=False)
                    stem_7 = gr.Audio(type="filepath", interactive=False, visible=False)
                btn = gr.Button("Обновить список", variant="primary")
                show_audio = gr.Button("Показать аудио", variant="primary")
                show_audio.click(
                    update_audio_players,
                    inputs=[history_list],
                    outputs=[stem_1, stem_2, stem_3, stem_4, stem_5, stem_6, stem_7]
                )
            btn.click(
                fn=lambda: gr.update(choices=get_history_list(output_dir_uvr)),
                outputs=history_list
            )

if __name__ == "__main__": 
    demo.launch(share=True, allowed_paths=["/content"])
