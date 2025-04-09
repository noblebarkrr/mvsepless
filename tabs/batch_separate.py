import gradio as gr
import os
from datetime import datetime, timezone, timedelta
from .models_list_ui import model_mapping
from code_infer import audio_separation

moscow_tz = timezone(timedelta(hours=3))

def update_models(separation_type):
    return gr.Dropdown(
        choices=list(model_mapping[separation_type].keys()),
    )

def separate_audio(input_files, separation_type, model, output_format):
    results = []
    infos = []
    
    for input_file in input_files:
        temp_path = input_file
        input_filename = os.path.basename(temp_path)
        from models_list import get_model_config
        model_code = model_mapping[separation_type][model]
        config = get_model_config(model_code)
        msc_time = datetime.now(moscow_tz).strftime("%Y%m%d_%H%M%S")
        
        if config:
            archr = config["arch"]
            if archr == "vr_arch" or archr == "mdx-net" or archr == "demucs":
                model_name = config["model_name"]
                output_name_folder = f"{msc_time}_{model_name}_{os.path.splitext(input_filename)[0]}"
            else:
                model_name = config["model_name"]
                output_name_folder = f"{msc_time}_{archr}_{model_name}_{os.path.splitext(input_filename)[0]}"
        
        output_dir = os.path.join("output_batch", output_name_folder)
        os.makedirs(output_dir, exist_ok=True)
        audio_separation(input_dir=temp_path, output_dir=output_dir, 
                        instrum=True, modelcode=model_code, 
                        output_format=output_format, use_tta=False, batch=False)

        audio_folder = output_dir
        audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) 
                      if f.endswith((".wav", ".mp3", ".flac"))][:7]
        
        results.append({
            "filename": input_filename,
            "stems": audio_files,
            "info": [
                f"**Архитектура:**\n{archr}",
                f"**Название модели:**\n{model_name}",
                f"**Дата:**\n{msc_time}",
                f"**Исходный файл:**\n{input_filename}"
            ]
        })
    
    return prepare_output(results)

def prepare_output(results):
    output_components = []
    
    # Сбрасываем все компоненты
    base_output = [
        gr.update(value=None),  # input_file
        gr.update(visible=False),  # input_group
        gr.update(visible=True),  # output_group
        *[gr.Markdown() for _ in range(4)],  # info fields
        *[gr.update(visible=False) for _ in range(7)]  # stems
    ]
    
    # Для каждого результата создаем свою секцию
    for result in results:
        # Добавляем информацию о файле
        output_components.extend([
            gr.Markdown(f"### {result['filename']}"),
            *[gr.Markdown(info) for info in result['info']]
        ])
        
        # Добавляем аудио компоненты
        for i in range(7):
            visible = i < len(result['stems'])
            output_components.append(
                gr.update(
                    visible=visible, 
                    value=result['stems'][i] if visible else None,
                    label=f"{result['filename']} - Stem {i+1}"
                )
            )
    
    return base_output + output_components

def batch_separate_ui():
    with gr.Blocks() as demo:
        # Первая группа (входные параметры)
        with gr.Column() as input_group:
            with gr.Row():
                input_file = gr.File(
                    label="Загрузите файлы", 
                    file_count="multiple",
                    file_types=[".wav", ".mp3", ".flac"],
                    type="filepath"
                )
            
            with gr.Row():
                separation_type = gr.Radio(
                    label="Тип разделения",
                    choices=list(model_mapping.keys()),
                    value=list(model_mapping.keys())[0] if model_mapping else "",
                    elem_classes=["radio-group"]
                )
            
            with gr.Row():    
                model = gr.Dropdown(
                    label="Модель",
                    choices=list(model_mapping["Музыка и вокал"].keys()) if "Музыка и вокал" in model_mapping else [],
                    interactive=True,
                    filterable=False
                )
            
            with gr.Row():
                output_format = gr.Radio(
                    label="Формат вывода",
                    choices=["wav", "mp3", "flac"],
                    value="flac",
                    elem_classes=["radio-group"]
                )

            btn = gr.Button("Разделить", variant="primary")

        # Вторая группа (результаты)
        with gr.Column(visible=False) as output_group:
            upload_another_btn = gr.Button("Загрузить ещё", variant="secondary")
            
            # Динамические компоненты будут добавляться в prepare_output
            info_components = [gr.Markdown(visible=False) for _ in range(4)]
            stem_components = [gr.Audio(visible=False) for _ in range(7)]

        # Обработчики событий
        separation_type.change(
            fn=update_models,
            inputs=separation_type,
            outputs=model
        )

        btn.click(
            fn=separate_audio,
            inputs=[input_file, separation_type, model, output_format],
            outputs=[input_file, input_group, output_group, *info_components, *stem_components]
        )
        
        upload_another_btn.click(
            fn=lambda: [
                gr.update(visible=True),  # input_group
                gr.update(visible=False),  # output_group
                *[gr.update(visible=False) for _ in range(4)],  # info
                *[gr.update(visible=False) for _ in range(7)]   # stems
            ],
            outputs=[input_group, output_group, *info_components, *stem_components]
        )
    
    return demo