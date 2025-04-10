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

def batch_separate_audio(input_files, separation_type, model, output_format):
    from models_list import get_model_config
    model_code = model_mapping[separation_type][model]
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
    
    base_output_dir = os.path.join("output_batch", output_name_folder)
    os.makedirs(base_output_dir, exist_ok=True)
    
    all_results = {}
    
    for input_file in input_files:
        input_filename = os.path.basename(input_file)
        file_name_no_ext = os.path.splitext(input_filename)[0]
        output_dir = os.path.join(base_output_dir, file_name_no_ext)
        os.makedirs(output_dir, exist_ok=True)
        
        audio_separation(
            input_dir=input_file, 
            output_dir=output_dir, 
            instrum=True, 
            modelcode=model_code, 
            output_format=output_format, 
            use_tta=False, 
            batch=False
        )
        
        audio_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) 
                       if f.endswith((".wav", ".mp3", ".flac"))][:7]
        
        all_results[file_name_no_ext] = {
            "files": audio_files,
            "output_dir": output_dir
        }
    
    # Prepare dropdown choices
    choices = list(all_results.keys())
    current_result = choices[0] if choices else None
    
    # Prepare info
    info = [
        f"**Архитектура:**\n{archr}",
        f"**Название модели:**\n{model_name}",
        f"**Дата:**\n{msc_time}",
        f"**Обработано файлов:**\n{len(input_files)}"
    ]
    
    # Prepare audio files for first result
    results = []
    if current_result:
        audio_files = all_results[current_result]["files"]
        for i in range(7):
            visible = i < len(audio_files)
            results.append(gr.update(visible=visible, value=audio_files[i] if visible else None))
    else:
        results = [gr.update(visible=False) for _ in range(7)]
    
    return (
        gr.update(value=None),  # input_files
        gr.update(visible=False),  # input_group
        gr.update(visible=True),  # output_group
        gr.update(choices=choices, value=current_result),  # result_selector
        *info,
        *results,
        gr.update(value=base_output_dir),  # output_dir_path
    )

def show_selected_result(selected_result, base_output_dir):
    if not selected_result:
        return [gr.update(visible=False) for _ in range(7)]
    
    result_dir = os.path.join(base_output_dir, selected_result)
    audio_files = [os.path.join(result_dir, f) for f in os.listdir(result_dir) 
                   if f.endswith((".wav", ".mp3", ".flac"))][:7]
    
    results = []
    for i in range(7):
        visible = i < len(audio_files)
        results.append(gr.update(visible=visible, value=audio_files[i] if visible else None))
    
    return results

def reset_ui():
    return (
        gr.update(visible=True),  # input_group
        gr.update(visible=False),  # output_group
        *[gr.update(visible=False) for _ in range(7)],  # stems
    )

def batch_separate_ui():
    with gr.Blocks() as demo:
        # Input group
        with gr.Column() as input_group:
            with gr.Row():
                input_files = gr.Files(
                    label="Выберите файлы", 
                    file_types=["audio"],
                    file_count="multiple"
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

        # Output group
        with gr.Column(visible=False) as output_group:
            info_test = gr.Column()
            with info_test:
                gr.Markdown("# Детали разделения")
                infos = [gr.Markdown() for _ in range(4)]  # Added one more for file count
            
            gr.Markdown("# Результаты обработки")
            output_dir_path = gr.Textbox(label="Папка с результатами", interactive=False)
            result_selector = gr.Dropdown(
                label="Выберите результат", 
                interactive=True,
                visible=True
            )
            
            files_after_separation = gr.Markdown("## Файлы после разделения")
            stems = [gr.Audio(visible=False) for _ in range(7)]
            upload_another_btn = gr.Button("Загрузить ещё", variant="secondary")
        
        separation_type.change(
            fn=update_models,
            inputs=separation_type,
            outputs=model
        )

        btn.click(
            fn=batch_separate_audio,
            inputs=[input_files, separation_type, model, output_format],
            outputs=[input_files, input_group, output_group, result_selector, *infos, *stems, output_dir_path]
        )
        
        result_selector.change(
            fn=show_selected_result,
            inputs=[result_selector, output_dir_path],
            outputs=stems
        )
        
        upload_another_btn.click(
            fn=reset_ui,
            outputs=[input_group, output_group, *stems]
        )
    
    return demo