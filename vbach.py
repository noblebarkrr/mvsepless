import gradio as gr
import os
import argparse
import subprocess
from datetime import datetime
from rvc.scripts.voice_conversion import voice_pipeline
import tempfile

from rvc.modules.model_manager import (
    download_from_url,
    upload_zip_file,
    upload_separate_files,
)





## Функция для замены голоса




def voice_conversion(input_path, model, pitch, ir, fr, rms, f0, hop, prtct, of, f0_min, f0_max, output_path=tempfile.mkdtemp(prefix="converted_voice_"), template="NAME_MODEL_F0METHOD_PITCH", batch=False, gradio=False):
    if batch:
        converted_files = []
        if gradio:
            if input_path is None or input_path == []:  # Check if input_path is None or empty list
                print("Error: No files provided for batch processing")
                return None
                
            for file in input_path:
                file_name = os.path.basename(file)
                namefile = os.path.splitext(file_name)[0]
                time_create_file = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_name = (
                    template
                    .replace("DATETIME", time_create_file)
                    .replace("NAME", namefile)
                    .replace("MODEL", model)
                    .replace("F0METHOD", f0)
                    .replace("PITCH", f"{pitch}")
                )
                converted_voice = voice_pipeline(file, model, pitch, ir, fr, rms, f0, hop, prtct, of, f0_min, f0_max, output_path, output_name)
                converted_files.append(converted_voice)

        else:
            if input_path is None or not os.path.isdir(input_path):  # Check if input_path is None or not a directory
                print(f"Error: {input_path} is not a directory (batch mode requires directory)")
                return []
            
            for filename in os.listdir(input_path):
                file = os.path.join(input_path, filename)
                if os.path.isfile(file):
                    file_name = os.path.basename(file)
                    namefile = os.path.splitext(file_name)[0]
                    time_create_file = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_name = (
                        template
                        .replace("DATETIME", time_create_file)
                        .replace("NAME", namefile)
                        .replace("MODEL", model)
                        .replace("F0METHOD", f0)
                        .replace("PITCH", f"{pitch}")
                    )
                    converted_voice = voice_pipeline(file, model, pitch, ir, fr, rms, f0, hop, prtct, of, f0_min, f0_max, output_path, output_name)
                    converted_files.append(converted_voice)
        return converted_files
    else:
        if input_path is None or not os.path.isfile(input_path):  # Check if input_path is None or not a file
            print(f"Error: {input_path} is not a file")
            return None
            
        file_name = os.path.basename(input_path)
        namefile = os.path.splitext(file_name)[0]
        time_create_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = (
            template
            .replace("DATETIME", time_create_file)
            .replace("NAME", namefile)
            .replace("MODEL", model)
            .replace("F0METHOD", f0)
            .replace("PITCH", f"{pitch}")
        )
        converted_voice = voice_pipeline(input_path, model, pitch, ir, fr, rms, f0, hop, prtct, of, f0_min, f0_max, output_path, output_name)
        return converted_voice



rvc_models_dir = "voice_models"

def get_models_list():
    """Get list of models."""
    models = []
    if os.path.exists(rvc_models_dir):
        models = [d for d in os.listdir(rvc_models_dir) if os.path.isdir(os.path.join(rvc_models_dir, d))]
    return models

def conversion():
    with gr.Column() as conversion_group:
        batch_mode = gr.Checkbox(label="Пакетная обработка", info="Позволяет обрабатывать несколько файлов подряд", value=False)
        
        # Initialize components for both modes
        with gr.Group(visible=False) as batch_group:
            batch_file_input = gr.File(label="Загрузить аудио (пакетный режим)", type="filepath", file_count="multiple", file_types=['.mp3', '.wav', '.flac'])
            batch_file_output = gr.Files(label="Результаты (пакетный режим)", interactive=False)
            convert_batch_btn = gr.Button("Конвертировать все", variant="primary", visible=True)
        with gr.Group(visible=True) as single_group:
            single_file_input = gr.Audio(label="Загрузить аудио", type="filepath")
            single_file_output = gr.Audio(label="Результат", type="filepath", interactive=False)
            convert_single_btn = gr.Button("Конвертировать один", variant="primary", visible=True)
        voicemodel_name = gr.Dropdown(
            choices=list(get_models_list()), 
            label="Имя модели", 
            value="senko",
            interactive=True,
            filterable=False
        )
        refresh_btn = gr.Button("Обновить список моделей")
        pitch_vocal = gr.Slider(-48, 48, value=0, step=12, label="Pitch", interactive=True)
        method_pitch = gr.Dropdown(
            label="Метод извлечения тона", 
            choices=["rmvpe+", "mangio-crepe", "fcpe"], 
            value="rmvpe+",
            interactive=True,
            filterable=False
        )
        template_voice = gr.Text(label="Формат имени преобразованного вокала", info="Список доступных ключей для формата имени можете найти в вкладке 'Настройки'", value="NAME_MODEL_F0METHOD_PITCH", visible=True)
        with gr.Accordion("Настройки RVC:", open=False):    
            hop_length = gr.Slider(0, 255, value=128, step=1, label="Длина шага (для mangio-crepe)", interactive=True, info="Длина шага определяет точность передачи тона. Меньше = дольше, точнее; Больше = быстрее, неточнее")
            index_rate = gr.Slider(0, 1, value=1, step=0.05, label="Влияние индекса", interactive=True, info="Индекс влияет на акцент и тембр ИИ-вокала")
            filter_radius = gr.Slider(0, 7, value=7, step=1, label="Радиус фильтра", interactive=True)
            rms = gr.Slider(0, 1, value=0, step=0.1, info="Ближе к 0 = оригинал, ближе к 1 = результат", label="Огибающая громкости", interactive=True)
            protect = gr.Slider(0, 0.5, value=0.35, step=0.05, label="Защита согласных", interactive=True, info="Помогает избежать артефактов связанных с дыханием и шипящими. Полная защита - 0.5.")
            f0_max = gr.Slider(1100, 2700, value=1100, step=50, label="Верхний лимит", interactive=True)
        output_format_rvc = gr.Dropdown(
            label="Формат результата", 
            choices=["wav", "mp3", "flac"], 
            interactive=True,
            filterable=False
        )
        
        constant_value = gr.Number(value=50, visible=False)
        
        def toggle_ui(batch):
            return {
                batch_group: gr.update(visible=batch),
                single_group: gr.update(visible=not batch)
            }
        
        batch_mode.change(
            fn=toggle_ui,
            inputs=[batch_mode],
            outputs=[batch_group, single_group]
        )

    refresh_btn.click(
        fn=lambda: gr.update(choices=get_models_list()),
        outputs=voicemodel_name
    )

    convert_single_btn.click(
        fn=voice_conversion,
        inputs=[
            single_file_input,
            voicemodel_name, 
            pitch_vocal, 
            index_rate, 
            filter_radius, 
            rms, 
            method_pitch, 
            hop_length, 
            protect, 
            output_format_rvc, 
            constant_value, 
            f0_max, 
            gr.State(tempfile.mkdtemp(prefix="converted_voice_")), 
            template_voice, 
            batch_mode, gr.State(True)
        ],
        outputs=single_file_output
    )
    convert_batch_btn.click(
        fn=voice_conversion,
        inputs=[
            batch_file_input,
            voicemodel_name, 
            pitch_vocal, 
            index_rate, 
            filter_radius, 
            rms, 
            method_pitch, 
            hop_length, 
            protect, 
            output_format_rvc, 
            constant_value, 
            f0_max, 
            gr.State(tempfile.mkdtemp(prefix="converted_voice_")), 
            template_voice, 
            batch_mode, gr.State(True)
        ],
        outputs=batch_file_output
    )











def url_download():
    with gr.Tab("Загрузить по ссылке"):
        with gr.Row():
            with gr.Column(variant="panel"):
                gr.HTML(
                    "<center><h3>Введите в поле ниже ссылку на ZIP-архив.</h3></center>"
                )
                model_zip_link = gr.Text(label="Ссылка на загрузку модели")
            with gr.Column(variant="panel"):
                with gr.Group():
                    model_name = gr.Text(
                        label="Имя модели",
                        info="Дайте вашей загружаемой модели уникальное имя, "
                        "отличное от других голосовых моделей.",
                    )
                    download_btn = gr.Button("Загрузить модель", variant="primary")

        gr.HTML(
            "<h3>"
            "Поддерживаемые сайты: "
            "<a href='https://huggingface.co/' target='_blank'>HuggingFace</a>, "
            "<a href='https://pixeldrain.com/' target='_blank'>Pixeldrain</a>, "
            "<a href='https://drive.google.com/' target='_blank'>Google Drive</a>, "
            "<a href='https://mega.nz/' target='_blank'>Mega</a>, "
            "<a href='https://disk.yandex.ru/' target='_blank'>Яндекс Диск</a>"
            "</h3>"
        )

        dl_output_message = gr.Text(label="Сообщение вывода", interactive=False)
        download_btn.click(
            download_from_url,
            inputs=[model_zip_link, model_name],
            outputs=dl_output_message,
        )


def zip_upload():
    with gr.Tab("Загрузить ZIP архивом"):
        with gr.Row():
            with gr.Column():
                zip_file = gr.File(
                    label="Zip-файл", file_types=[".zip"], file_count="single"
                )
            with gr.Column(variant="panel"):
                gr.HTML(
                    "<h3>1. Найдите и скачайте файлы: .pth и "
                    "необязательный файл .index</h3>"
                )
                gr.HTML(
                    "<h3>2. Закиньте файл(-ы) в ZIP-архив и "
                    "поместите его в область загрузки</h3>"
                )
                gr.HTML("<h3>3. Дождитесь полной загрузки ZIP-архива в интерфейс</h3>")
                with gr.Group():
                    local_model_name = gr.Text(
                        label="Имя модели",
                        info="Дайте вашей загружаемой модели уникальное имя, "
                        "отличное от других голосовых моделей.",
                    )
                    model_upload_button = gr.Button("Загрузить модель", variant="primary")

        local_upload_output_message = gr.Text(label="Сообщение вывода", interactive=False)
        model_upload_button.click(
            upload_zip_file,
            inputs=[zip_file, local_model_name],
            outputs=local_upload_output_message,
        )


def files_upload():
    with gr.Tab("Загрузить файлами"):
        with gr.Group():
            with gr.Row():
                pth_file = gr.File(
                    label="pth-файл", file_types=[".pth"], file_count="single"
                )
                index_file = gr.File(
                    label="index-файл", file_types=[".index"], file_count="single"
                )
        with gr.Column(variant="panel"):
            with gr.Group():
                separate_model_name = gr.Text(
                    label="Имя модели",
                    info="Дайте вашей загружаемой модели уникальное имя, "
                    "отличное от других голосовых моделей.",
                )
                separate_upload_button = gr.Button("Загрузить модель", variant="primary")

        separate_upload_output_message = gr.Text(
            label="Сообщение вывода", interactive=False
        )
        separate_upload_button.click(
            upload_separate_files,
            inputs=[pth_file, index_file, separate_model_name],
            outputs=separate_upload_output_message,
        )









def main():
    parser = argparse.ArgumentParser(description='Voice Conversion Tool')
    
    # Required arguments
    parser.add_argument('input', help='Input file or directory (for batch mode)')
    parser.add_argument('model', help='Model name to use for conversion')
    
    # Batch mode
    parser.add_argument('--batch', action='store_true', help='Enable batch processing of a directory')
    
    # Main settings
    parser.add_argument('--pitch', type=int, default=0, help='Pitch adjustment (-48 to 48)')
    parser.add_argument('--index_rate', type=float, default=0, help='Index rate (0 to 1)')
    parser.add_argument('--filter_radius', type=int, default=3, help='Filter radius (0 to 7)')
    parser.add_argument('--volume_envelope', type=float, default=0.25, help='Volume envelope (0 to 1)')
    
    # F0 settings
    parser.add_argument('--method', default='rmvpe+', choices=['rmvpe+', 'mangio-crepe', 'fcpe'], help='F0 extraction method')
    parser.add_argument('--hop_length', type=int, default=128, help='Hop length (32 to 512)')
    parser.add_argument('--protect', type=float, default=0.33, help='Protect (0 to 0.5)')
    parser.add_argument('--f0_min', type=int, default=50, help='Minimum F0 (0 to 500)')
    parser.add_argument('--f0_max', type=int, default=1100, help='Maximum F0 (100 to 2000)')
    
    # Output settings
    parser.add_argument('--output_format', default='mp3', choices=['mp3', 'wav', 'flac'], help='Output audio format')
    parser.add_argument('--output_path', default='', help='Output directory path')
    parser.add_argument('--template', default='DATETIME_NAME_PITCH', 
                       help='Output filename template (can include DATETIME, NAME, MODEL, F0METHOD, PITCH)')
    
    args = parser.parse_args()
    
    voice_conversion(
        input_path=args.input,
        model=args.model,
        pitch=args.pitch,
        ir=args.index_rate,
        fr=args.filter_radius,
        rms=args.volume_envelope,
        f0=args.method,
        hop=args.hop_length,
        prtct=args.protect,
        of=args.output_format,
        f0_min=args.f0_min,
        f0_max=args.f0_max,
        output_path=args.output_path,
        template=args.template,
        batch=args.batch
    )

if __name__ == '__main__':
    main()





