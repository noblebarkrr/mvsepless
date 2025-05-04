import gradio as gr
import os
import argparse
import subprocess
from datetime import datetime
from rvc.scripts.voice_conversion import voice_pipeline

from rvc.modules.model_manager import (
    download_from_url,
    upload_zip_file,
    upload_separate_files,
)


rvc_models_dir = "voice_models"

def get_models_list():
    """Get list of models."""
    models = []
    if os.path.exists(rvc_models_dir):
        models = [d for d in os.listdir(rvc_models_dir) if os.path.isdir(os.path.join(rvc_models_dir, d))]
    return models

def conversion():
    with gr.Column() as conversion_group:
        file_input = gr.Audio(label="Upload audio", type="filepath")
        voicemodel_name = gr.Dropdown(
            choices=list(get_models_list()), 
            label="Model name", 
            value="senko",
            interactive=True,
            filterable=False
        )
        refresh_btn = gr.Button("Refresh")
        pitch_vocal = gr.Slider(-48, 48, value=0, step=12, label="Pitch", interactive=True)
        method_pitch = gr.Dropdown(
            label="F0 method", 
            choices=["rmvpe+", "mangio-crepe", "fcpe"], 
            value="rmvpe+",
            interactive=True,
            filterable=False
        )
        hop_length = gr.Slider(0, 255, value=73, step=1, label="Hop length (mangio-crepe only)", interactive=True)
        index_rate = gr.Slider(0, 1, value=1, step=0.05, label="Index rate", interactive=True)
        filter_radius = gr.Slider(0, 7, value=7, step=1, label="Filter radius", interactive=True)
        rms = gr.Slider(0, 1, value=0, step=0.1, label="RMS", interactive=True)
        protect = gr.Slider(0, 0.5, value=0.35, step=0.05, label="Protect", interactive=True)
        f0_max = gr.Slider(1100, 2700, value=1100, step=50, label="F0 Max", interactive=True)
        output_format_rvc = gr.Dropdown(
            label="Export format", 
            choices=["wav", "mp3", "flac"], 
            interactive=True,
            filterable=False
        )
        # Add hidden components for constant values
        constant_value = gr.Number(value=50, visible=False)
        output_dir = gr.Text(value="/content/voice_output", visible=False)
        convert_btn = gr.Button("Convert", variant="primary")
        
    with gr.Column() as output_voice_group:
        converted_voice = gr.Audio(type="filepath", interactive=False, visible=True)
        output_filename = gr.Text(visible=False)

    refresh_btn.click(
        fn=lambda: gr.update(choices=get_models_list()),
        outputs=voicemodel_name
    )
    
    def create_output_filename(model, method, pitch):
        return f"converted_voice_{model}_{method}_{pitch}"
    
    convert_btn.click(
        fn=create_output_filename,
        inputs=[voicemodel_name, method_pitch, pitch_vocal],
        outputs=output_filename
    ).then(
        fn=voice_pipeline,
        inputs=[file_input, voicemodel_name, pitch_vocal, index_rate, 
                filter_radius, rms, method_pitch, hop_length, 
                protect, output_format_rvc, constant_value, f0_max, 
                output_dir, output_filename],
        outputs=converted_voice
    )




def url_download():
    with gr.Tab("From ZIP in URL"):
        with gr.Row():
            with gr.Column(variant="panel"):
                gr.HTML(
                    "<center><h3>Enter URL to ZIP archive with voice model files</h3></center>"
                )
                model_zip_link = gr.Text(label="Model URL")
            with gr.Column(variant="panel"):
                with gr.Group():
                    model_name = gr.Text(
                        label="Model name",
                        info="Give your downloaded model a unique name different from other voice models.",
                    )
                    download_btn = gr.Button("Download model", variant="primary")

        gr.HTML(
            "<h3>"
            "Supported sites: "
            "<a href='https://huggingface.co/' target='_blank'>HuggingFace</a>, "
            "<a href='https://pixeldrain.com/' target='_blank'>Pixeldrain</a>, "
            "<a href='https://drive.google.com/' target='_blank'>Google Drive</a>, "
            "<a href='https://mega.nz/' target='_blank'>Mega</a>, "
            "<a href='https://disk.yandex.ru/' target='_blank'>Yandex Disk</a>"
            "</h3>"
        )

        dl_output_message = gr.Text(label="Output message", interactive=False)
        download_btn.click(
            download_from_url,
            inputs=[model_zip_link, model_name],
            outputs=dl_output_message,
        )


def zip_upload():
    with gr.Tab("From uploaded ZIP"):
        with gr.Row():
            with gr.Column():
                zip_file = gr.File(
                    label="ZIP file", file_types=[".zip"], file_count="single"
                )
            with gr.Column(variant="panel"):
                gr.HTML(
                    "<h3>1. Find and download the files: .pth and "
                    "optional .index file</h3>"
                )
                gr.HTML(
                    "<h3>2. Put the file(s) in a ZIP archive and "
                    "place it in the upload area</h3>"
                )
                gr.HTML("<h3>3. Wait for the ZIP archive to fully upload to the interface</h3>")
                with gr.Group():
                    local_model_name = gr.Text(
                        label="Model name",
                        info="Give your downloaded model a unique name different from other voice models.",
                    )
                    model_upload_button = gr.Button("Download model", variant="primary")

        local_upload_output_message = gr.Text(label="Output message", interactive=False)
        model_upload_button.click(
            upload_zip_file,
            inputs=[zip_file, local_model_name],
            outputs=local_upload_output_message,
        )


def files_upload():
    with gr.Tab("From uploaded files"):
        with gr.Group():
            with gr.Row():
                pth_file = gr.File(
                    label="pth file", file_types=[".pth"], file_count="single"
                )
                index_file = gr.File(
                    label="index file", file_types=[".index"], file_count="single"
                )
        with gr.Column(variant="panel"):
            with gr.Group():
                separate_model_name = gr.Text(
                    label="Model name",
                    info="Give your downloaded model a unique name different from other voice models.",
                )
                separate_upload_button = gr.Button("Download model", variant="primary")

        separate_upload_output_message = gr.Text(
            label="Output message", interactive=False
        )
        separate_upload_button.click(
            upload_separate_files,
            inputs=[pth_file, index_file, separate_model_name],
            outputs=separate_upload_output_message,
        )








## Cli version




def voice_conversion(input_path, model, pitch, ir, fr, rms, f0, hop, prtct, of, f0_min, f0_max, output_path, template, batch):
    if batch:
        if not os.path.isdir(input_path):
            print(f"Error: {input_path} is not a directory (batch mode requires directory)")
            return
            
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
                voice_pipeline(file, model, pitch, ir, fr, rms, f0, hop, prtct, of, f0_min, f0_max, output_path, output_name)
    else:
        if not os.path.isfile(input_path):
            print(f"Error: {input_path} is not a file")
            return
            
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
        voice_pipeline(input_path, model, pitch, ir, fr, rms, f0, hop, prtct, of, f0_min, f0_max, output_path, output_name)

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




