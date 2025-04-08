import gradio as gr
import os
from datetime import datetime, timezone, timedelta
from .models_list_ui import experimental_model_mapping
from code_infer import audio_separation

moscow_tz = timezone(timedelta(hours=3))

def update_models(separation_type):
    return gr.Dropdown(
        choices=list(experimental_model_mapping[separation_type].keys()),
    )


def separate_audio(input_file, separation_type, model, output_format):
    temp_path = input_file
    input_filename = os.path.basename(temp_path)
    from models_list import get_model_config
    model_code = experimental_model_mapping[separation_type][model]
    config = get_model_config(model_code)
    msc_time = datetime.now(moscow_tz).strftime("%Y%m%d_%H%M%S")
    
    if config:
        archr = config["arch"]
        if archr == "vr_arch" or archr == "mdx-net" or archr == "demucs":
            model_name = config["model_name"]
            output_name_folder = f"{msc_time}_{model_name}"
            
        else:
            model_name = config["model_name"]
            output_name_folder = f"{msc_time}_{archr}_{model_name}"
    output_dir = os.path.join("/content/output", output_name_folder)
    os.makedirs(output_dir, exist_ok=True)
    audio_separation(input_dir=temp_path, output_dir=output_dir, instrum=True, modelcode=model_code, output_format=output_format, use_tta=False, batch=False)

    audio_folder = output_dir
    audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith((".wav", ".mp3", ".flac"))][:7]
    
    results = []
    for i in range(7):
        visible = i < len(audio_files)
        results.append(gr.update(visible=visible, value=audio_files[i] if visible else None))
    info = [
        f"**Архитектура:**\n{archr}",
        f"**Название модели:**\n{model_name}",
        f"**Дата:**\n{msc_time}"
    ]
    # Возвращаем обновленные состояния всех элементов
    return (
        gr.update(value=None),  # input_file
        gr.update(visible=False),  # input_group
        gr.update(visible=True),  # output_group
        *info,
        *results,                  # stems
    )

def update_models(separation_type):
    return gr.Dropdown(choices=list(model_mapping[separation_type].keys()))

def reset_ui():
    # Сбрасываем состояние UI к начальному
    return (
        gr.update(visible=True),  # group
        gr.update(visible=False),  # group
        *[gr.update(visible=False) for _ in range(7)],  # stems
    )

def ex_separate_ui():
    with gr.Blocks() as demo:
        # Первая группа (будет скрываться)
        with gr.Column() as input_group:
            with gr.Row():
                input_file = gr.Audio(label="Выберите файл", type="filepath", visible=True)
            
            with gr.Row():
                separation_type = gr.Radio(
                    label="Тип разделения",
                    choices=list(experimental_model_mapping.keys()),
                    value=list(experimental_model_mapping.keys())[0] if experimental_model_mapping else "",
                    elem_classes=["radio-group"],
                    visible=True
                )
            
            with gr.Row():    
                model = gr.Dropdown(
                    label="Модель",
                    choices=list(experimental_model_mapping["MDX23C"].keys()) if "MDX23C" in experimental_model_mapping else [],
                    interactive=True,
                    visible=True,
                    filterable=False
                )

            
            with gr.Row():
                output_format = gr.Radio(
                    label="Формат вывода",
                    choices=["wav", "mp3", "flac"],
                    value="flac",
                    elem_classes=["radio-group"],
                    visible=True
                )

            btn = gr.Button("Разделить", variant="primary", visible=True)

        # Вторая группа (будет показываться после разделения)
        with gr.Column(visible=False) as output_group:
            info_test = gr.Column(visible=True)
            with info_test:
                gr.Markdown("# Детали разделения")
                infos = [gr.Markdown() for _ in range(3)]
            files_after_separation = gr.Markdown("# Файлы после разделения", visible=True)
            stems = [gr.Audio(visible=False) for _ in range(7)]
            upload_another_btn = gr.Button("Загрузить ещё", variant="secondary", visible=True)
        
        separation_type.change(
            fn=update_models,
            inputs=separation_type,
            outputs=model
        )

        btn.click(
            fn=separate_audio,
            inputs=[input_file, separation_type, model, output_format],
            outputs=[input_file, input_group, output_group, *infos, *stems]
        )
        
        upload_another_btn.click(
            fn=reset_ui,
            outputs=[input_group, output_group, *stems]
        )
    
    return demo