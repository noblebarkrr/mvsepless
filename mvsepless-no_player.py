import gradio as gr
import os
import subprocess
import shutil
from datetime import datetime

# Функция для выполнения команды разделения
def separate_audio(input_file, separation_type, model, output_format):
    # Создаем директорию для входных файлов
    
    input_dir = "/content/input"
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    os.makedirs(input_dir, exist_ok=True)
    
    # Получаем путь к временному файлу
    temp_path = input_file.name  # Gradio возвращает путь как строку
    input_filename = os.path.basename(temp_path)
    input_path = os.path.join(input_dir, input_filename)
    
    # Копируем файл из временного хранилища
    
    shutil.copy(temp_path, input_path)
    
    # Создаем директорию для выходных файлов
    output_dir = os.path.join("/content/output", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем код модели
    model_code = model_mapping[separation_type][model]
    
    # Формируем команду
    command = [
        "/content/MVSEPLESS/msst/venv/bin/python", "/content/MVSEPLESS/main.py",
        "--input", str(input_dir), "--output", str(output_dir),
        "--modelcode", str(model_code), "--output_format", str(output_format)
    ]
    subprocess.run(command)
    
    # Возвращаем результат
    return [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(output_format)]

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
        "BS Roformer / SYH99999 / 4 Stems FT": 205,
        "BS Roformer / ZFTurbo / 4 Stems MUSDB18": 204,
        "MDX23C / ZFTurbo / 4 Stems MUSDB18": 304,
        "MDX23C / Aufr33 & Jarredou / Drumsep 6 Stems": 304,
        "SCNET / ZFTurbo / 4 Stems MUSDB18": 800,
        "SCNET / Starrytong / 4 Stems MUSDB18 XL": 801,
        "SCNET / ZFTurbo / 4 Stems MUSDB18 Large": 802,
        "HT DEMUCS / FT 4 Stems": 701,
        "HT DEMUCS / 4 Stems": 702,
        "HT DEMUCS / 6 Stems": 704,
        "HT DEMUCS / Drumsep 4 stems": 700,
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
with gr.Blocks(title="Разделение музыки и голоса",theme=gr.themes.Soft()) as app:
    gr.Markdown("## MVSEPLESS")
    
    with gr.Row():
        input_file = gr.File(label="Перетащите, чтобы загрузить файл на сервер или выберите файл на устройстве", file_types=[".wav", ".mp3", ".flac"])

    with gr.Row():
        separation_type = gr.Dropdown(
            label="Тип разделения", 
            choices=list(model_mapping.keys()), 
            value="Инструментал/Вокал"
        )
    with gr.Row():    
        model = gr.Dropdown(
            label="Модель", 
            choices=list(model_mapping["Инструментал/Вокал"].keys())
        )
    with gr.Row():
        output_format = gr.Dropdown(
            label="Формат вывода", 
            choices=["wav", "mp3", "flac"], 
            value="wav"
        )

    btn = gr.Button("Разделить", variant="primary")

    with gr.Row():
        output_file = gr.File(label="Файлы после разделения", interactive=False)

    
    # Обработчик изменения типа разделения
    separation_type.change(
        fn=lambda x: gr.Dropdown(choices=list(model_mapping[x].keys())),
        inputs=separation_type,
        outputs=model
    )

    # Обработчик кнопки
    btn.click(
        fn=separate_audio,
        inputs=[input_file, separation_type, model, output_format],
        outputs=output_file
    )

app.launch(share=True)
