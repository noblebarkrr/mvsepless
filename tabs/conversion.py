import gradio as gr
import os
import subprocess
from datetime import datetime
from rvc.scripts.voice_conversion import voice_pipeline

rvc_models_dir = "voice_models"

def get_models_list():
    """Получение списка моделей."""
    models = []
    if os.path.exists(rvc_models_dir):
        models = [d for d in os.listdir(rvc_models_dir) if os.path.isdir(os.path.join(rvc_models_dir, d))]
    return models

def conversion():
    with gr.Column() as conversion_group:
        file_input = gr.Audio(label="Загрузить аудио", type="filepath")
        models_list = get_models_list()
        default_model = models_list[0] if models_list else None
        voicemodel_name = gr.Dropdown(
            choices=models_list, 
            label="Имя модели", 
            value=default_model,
            interactive=True,
            filterable=False
        )
        refresh_btn = gr.Button("Обновить")
        pitch_vocal = gr.Slider(-48, 48, value=0, step=12, label="Высота тона", interactive=True)
        method_pitch = gr.Dropdown(
            label="Метод извлечения тона", 
            choices=["rmvpe+", "mangio-crepe", "fcpe"], 
            value="rmvpe+",
            interactive=True,
            filterable=False
        )
        hop_length = gr.Slider(0, 255, value=73, step=1, label="Длина шага для mangio-crepe", interactive=True)
        index_rate = gr.Slider(0, 1, value=1, step=0.05, label="ИИ-акцент", interactive=True)
        filter_radius = gr.Slider(0, 7, value=7, step=1, label="Радиус фильтра", interactive=True)
        rms = gr.Slider(0, 1, value=0, step=0.1, label="Нормализация", interactive=True)
        protect = gr.Slider(0, 0.5, value=0.35, step=0.05, label="Защита согласных", interactive=True)
        f0_max = gr.Slider(1100, 2700, value=1100, step=50, label="Максимальный диапазон тона", interactive=True)
        output_format_rvc = gr.Dropdown(
            label="Формат вывода", 
            choices=["wav", "mp3", "flac"], 
            value="wav",
            interactive=True,
            filterable=False
        )
        convert_btn = gr.Button("Преобразовать", variant="primary")
        
    with gr.Column() as output_voice_group:
        converted_voice = gr.Audio(type="filepath", interactive=False, visible=True)

    # Обработчики
    refresh_btn.click(
        fn=lambda: gr.update(choices=get_models_list()),
        outputs=voicemodel_name
    )
    
    convert_btn.click(
        fn=voice_pipeline,
        inputs=[
            file_input, 
            voicemodel_name, 
            pitch_vocal, 
            index_rate, 
            filter_radius, 
            rms, 
            method_pitch, 
            hop_length, 
            protect, 
            output_format_rvc,
            gr.Number(50, visible=False),  # Добавляем скрытый компонент для числа 50
            f0_max, 
            gr.Text("/content/voice_output", visible=False),  # Добавляем скрытый компонент для пути
            gr.Text(f"converted_voice_{voicemodel_name}_{method_pitch}_{pitch_vocal}", visible=False)  # И для имени файла
        ],
        outputs=converted_voice
    )