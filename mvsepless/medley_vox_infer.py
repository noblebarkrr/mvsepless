import os
import sys
import ast
import gc
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import gradio as gr
import json
import argparse
import glob
import subprocess
import numpy as np
import soundfile as sf
import librosa
from device import all_ids, set_device, cuda_available
from audio import read, multi_channel_array_from_arrays, split_channels, multiwrite, output_formats, check, get_duration_from_array, convert_to_dtype
from gradio_helper import GradioHelper, tz
from datetime import datetime
from downloader import dw_file
from svs.utils import loudnorm, str2bool, db2linear
from namer import Namer
import torch
import pyloudnorm as pyln

stereo_modes = ("mono", "left/right")
spectral_features = ("mfcc", "spectral_centroid")
vad_methods = ("spec", "webrtc")
overlap_add_methods = (None, "ola", "ola_norm", "sf_chunk")
stems = ["vox_1", "vox_2", "residual"]
namer = Namer()

class MedleyVoxModelManager:
    def __init__(
        self,
        cache_dir=os.path.join(script_dir, "medley_vox_models_cache"),
    ):
        self.models_cache_dir = cache_dir
        self.models_info = {
            "multi_singing_librispeech": {
                "full_name": "Multi singing LibriSpeech model for Medley-Vox",
                "id": 900,
                "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/multi_singing_librispeech/vocals.pth?download=true",
                "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/multi_singing_librispeech/vocals.json?download=true"
            },
            "multi_singing_librispeech_138": {
                "full_name": "Multi singing LibriSpeech 138 epoch model for Medley-Vox",
                "id": 901,
                "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/multi_singing_librispeech_138/vocals.pth?download=true",
                "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/multi_singing_librispeech_138/vocals.json?download=true"
            },
            "singing_librispeech_ft_isrnet": {
                "full_name": "Singing LibriSpeech Finetuned model IsrNET for Medley-Vox",
                "id": 902,
                "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/singing_librispeech_ft_iSRNet/vocals.pth?download=true",
                "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/singing_librispeech_ft_iSRNet/vocals.json?download=true"
            },
            "singing_librispeech_isrnet": {
                "full_name": "Singing LibriSpeech IsrNET model for Medley-Vox",
                "id": 903,
                "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/singing_librispeech_iSRNet/vocals.pth?download=true",
                "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/singing_librispeech_iSRNet/vocals.json?download=true"
            },
            "vocal_231": {
                "full_name": "Vocal 231 model for Medley-Vox",
                "id": 904,
                "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocal%20231/vocals.pth?download=true",
                "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocal%20231/vocals.json?download=true"
            },
            "vocals_135": {
                "full_name": "Vocals 135 model for Medley-Vox",
                "id": 905,
                "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20135/vocals.pth?download=true",
                "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20135/vocals.json?download=true"
            },
            "vocals_163": {
                "full_name": "Vocals 163 model for Medley-Vox",
                "id": 906,
                "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20163/vocals.pth?download=true",
                "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20163/vocals.json?download=true"
            },
            "vocals_188": {
                "full_name": "Vocals 188 model for Medley-Vox",
                "id": 907,
                "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20188/vocals.pth?download=true",
                "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20188/vocals.json?download=true"
            },
            "vocals_200": {
                "full_name": "Vocals 200 model for Medley-Vox",
                "id": 908,
                "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20200/vocals.pth?download=true",
                "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20200/vocals.json?download=true"
            },
            "vocals_238": {
                "full_name": "Vocals 238 model for Medley-Vox",
                "id": 909,
                "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20238/vocals.pth?download=true",
                "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20238/vocals.json?download=true"
            }
        }

    def get_mn(self):
        return [mn for mn in self.models_info]
    
    def get_id(self, model_name):
        if model_name is not None and model_name != "":
            return self.models_info.get(model_name).get("id", 0)
        else:
            return 0

    def download_model(self, model_paths, model_name, ckpt_url, conf_url, only_check_exists=False):
        model_dir = os.path.join(model_paths)
        os.makedirs(model_dir, exist_ok=True)

        config_path = os.path.join(model_dir, f"{model_name}_config.json")
        checkpoint_path = os.path.join(
            model_dir,
            f"{model_name}.pth",
        )

        if config_path is None or checkpoint_path is None:
            raise RuntimeError()

        if os.path.exists(checkpoint_path) and os.path.exists(config_path):
            if (
                os.path.getsize(checkpoint_path) == 0
                or os.path.getsize(checkpoint_path) == 0
            ):
                if only_check_exists:
                    return False
                else:
                    for local_path, url_model in [
                        (checkpoint_path, ckpt_url),
                        (config_path, conf_url),
                    ]:
                        if not os.path.exists(local_path):
                            dw_file(url_model, local_path)
            else:
                if only_check_exists:
                    return True
        else:
            if only_check_exists:
                return False
            else:
                for local_path, url_model in [
                    (checkpoint_path, ckpt_url),
                    (config_path, conf_url),
                ]:
                    if not os.path.exists(local_path):

                        dw_file(url_model, local_path)

        return config_path, checkpoint_path

    def install_model(
        self,
        model_name: str,
        progress: any = None,
    ) -> tuple[int, str, str]:

        info = self.models_info.get(model_name, None)
        if not info:
            raise ValueError(
                f"Модель {model_name} не найдена"
            )
        id = self.get_id(model_name)
        conf, ckpt = self.download_model(
            self.models_cache_dir,
            model_name,
            info["checkpoint_url"],
            info["config_url"],
        )

        return id, conf, ckpt

    def check_model(
        self,
        model_name: str,
        progress: any = None,
    ) -> tuple[int, str, str]:

        info = self.models_info.get(model_name, None)
        if not info:
            raise ValueError(
                f"Модель {model_name} не найдена"
            )
        id = self.get_id(model_name)
        return self.download_model(
            self.models_cache_dir,
            model_name,
            info["checkpoint_url"],
            info["config_url"],
            only_check_exists=True
        )
    
    def get_mn_dwloaded(self):
        return [model for model in self.get_mn() if self.check_model(self.get_mt(model), model)]

class MedleyVoxSeparator(MedleyVoxModelManager, GradioHelper):

    def __init__(self, input_files, upload_files, user_directory, device):
        super().__init__()
        self.input_files = input_files
        self.upload_files = upload_files
        self.user_directory = user_directory
        self.device = device

    class OutputReader:
        def __init__(self, debug=False):
            self.debug = debug

        def parse_json_line(self, line):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None

        def reaction_line(self, line, progress, add_text):
            _add_text = ""
            if add_text != "" or add_text is not None:
                _add_text = f"| {add_text}"

            data = self.parse_json_line(line)
            if data is None:
                return None
            elif "done" in data:
                progress(1.0, desc=f"Завершено {_add_text}")
                print("\rЗавершено", end="\n")
                return data["done"]
            elif "error" in data:
                raise Exception(data["error"])

    output_reader = OutputReader()

    def separator_base(
        self,
        input_file: str,
        output_dir: str,
        model_name: str,
        output_format: str = "wav",
        template: str = "NAME_STEM",
        ckpt: str = None,
        conf: str = None,
        id: int = None,
        use_overlapadd: str = "ola", # По умолчанию для вокала лучше использовать OLA
        w2v_ckpt_path: str = None,
        stereo_mode: str = "mono",
        progress: any = None,
        add_text_progress: str = "",
    ) -> list[str]:

        # Формируем базовый список аргументов, соответствующий parser в main()
        cmd = [
            os.sys.executable,
            "-m", "medley_vox_infer",
            "--input", input_file,
            "--results_save_dir", output_dir,
            "--model_name", model_name,
            "--model_id", str(id),
            "--json_path", conf,
            "--checkpoint_path", ckpt,
            "--stereo", stereo_mode,
            "--output_format", output_format,
            "--template", template,
            "--device", self.device,
        ]

        # Добавляем специфичные параметры обработки
        if use_overlapadd and use_overlapadd != "None":
            cmd.extend(["--use_overlapadd", use_overlapadd])

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
            )

            result = None
            error_lines = []

            # Чтение stdout построчно
            for line in process.stdout:
                line = line.strip()
                if line:
                    if self.output_reader.debug:
                        print(f"[stdout] {line}")
                    
                    # Обработка строки для получения прогресса и результата
                    line_result = self.output_reader.reaction_line(
                        line, progress, add_text_progress
                    )
                    if line_result is not None:
                        result = line_result

            # Чтение stderr построчно
            for line in process.stderr:
                line = line.strip()
                if line:
                    if self.output_reader.debug:
                        print(f"[stderr] {line}")
                    error_lines.append(line)
                    
                    # Также проверяем stderr на наличие JSON-сообщений
                    line_result = self.output_reader.reaction_line(
                        line, progress, add_text_progress
                    )
                    if line_result is not None:
                        result = line_result

            # Ожидание завершения процесса
            process.wait()

            if process.returncode != 0:
                error_text = "\n".join(error_lines[-5:]) if error_lines else "Неизвестная ошибка"
                raise Exception(
                    f"Процесс завершился с ошибкой. Код возврата: {process.returncode}. Сообщения об ошибках:\n{error_text}"
                )

            if result is not None:
                return result
            else:
                raise Exception("Процесс завершился без возврата результата")

        except Exception as e:
            raise e
        finally:
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
                
    def separate(
        self,
        input: str | list = None,
        output_dir: str = None,
        model_name: str = "Mel-Band-Roformer_Vocals_kimberley_jensen",
        output_format: str = "mp3",
        template: str = "NAME_(STEM)_MODEL",
        use_overlapadd: str = "ola",
        stereo_mode: str = "mono",
        add_single_sep_text_progress: str | None = "",
        progress: any = gr.Progress(track_tqdm=True),
    ) -> list[tuple[str, str]] | list[str, list[tuple[str, str]]]:

        progress(0, desc="Начало обработки")

        if output_format not in output_formats:
            output_format = "flac"

        if output_dir is None:
            output_dir = os.getcwd()

        if output_dir:
            output_dir = os.path.abspath(output_dir)

        if not input:
            raise ValueError("Входной файл не указан")

        if "STEM" not in template and template is not None:
            template = template + "_STEM_"
        if not template:
            template = "mvsepless_NAME_(STEM)"

        os.makedirs(output_dir, exist_ok=True)

        add_progress_text_custom = add_single_sep_text_progress

        id, conf, ckpt = self.install_model(
            model_name, progress
        )

        if isinstance(input, str):
            if not os.path.exists(input):
                raise ValueError(f"Входной файл не найден: {input}")

            if not check(input):
                raise ValueError("Входной файл не содержит аудио")
            progress(0.5, desc=f"Обработка")
            basename = os.path.splitext(os.path.basename(input))[0]
            seped = self.separator_base(
                input_file=input,
                output_dir=output_dir,
                model_name=model_name,
                output_format=output_format,
                template=template,
                ckpt=ckpt,
                conf=conf,
                id=id,
                use_overlapadd=use_overlapadd,
                stereo_mode=stereo_mode,
                progress=progress,
                add_text_progress=add_progress_text_custom,
            )
            return seped

        elif isinstance(input, list):
            results = []
            for i, f in enumerate(input, 1):
                print(f"Файл {i} из {len(input)}: {f}")
                gr.Warning(title=f"Файл {i} из {len(input)}: {f}", message="")
                progress(0.5, desc=f"Обработка | {i} из {len(input)}")
                if os.path.exists(f):
                    if check(f):
                        basename = os.path.splitext(os.path.basename(f))[0]
                        seped = self.separator_base(
                            input_file=f,
                            output_dir=output_dir,
                            model_name=model_name,
                            output_format=output_format,
                            template=template,
                            ckpt=ckpt,
                            conf=conf,
                            id=id,
                            use_overlapadd=use_overlapadd,
                            stereo_mode=stereo_mode,
                            progress=progress,
                            add_text_progress=f"{i} из {len(input)}",
                        )
                        results.extend(seped)
            return results

    def UI(self, theme=None):
            
        default_models = self.get_mn()
        
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    # Компонент загрузки файлов из SeparatorGradio
                    upload = gr.Files(show_label=False, type="filepath", interactive=True)
                    refresh_input_btn = gr.Button("Обновить", variant="primary", interactive=True)
                    # Список файлов (используем input_files из родительского класса)
                    list_input_files = gr.Dropdown(
                        label="Загрузить файлы",
                        choices=reversed(self.input_files),
                        value=[],
                        multiselect=True,
                        interactive=True,
                        scale=10
                    )
                    gr.on(fn=lambda: gr.update(choices=reversed(self.input_files), value=[]), outputs=list_input_files, trigger_mode="once")
                    refresh_input_btn.click(lambda: gr.update(choices=reversed(self.input_files), value=[]), outputs=list_input_files)
                    @upload.upload(inputs=[upload], outputs=[list_input_files, upload])
                    def upload_files_medley(input_files):
                        files = self.upload_files(input_files)
                        return gr.update(choices=reversed(self.input_files), value=files), gr.update(value=[])

            with gr.Column():
                with gr.Group():
                    model_name = gr.Dropdown(
                        label="Имя модели", 
                        choices=default_models, 
                        value=default_models[0] if default_models else None,
                        interactive=True
                    )
                    
                    stereo_mode = gr.Radio(
                        label="Режим стерео",
                        choices=stereo_modes,
                        value=stereo_modes[1],
                        info="mono - монофоническая обработка аудио, \nleft/right - обработка левого и правого каналов отдельно",
                        interactive=True
                    )
                    
                    overlap_mode = gr.Radio(
                        label="Метод перекрытия (Overlap-Add)",
                        choices=overlap_add_methods,
                        value="ola", interactive=True
                    )
                    
                    output_format = gr.Dropdown(
                        label="Формат выходного файла",
                        choices=output_formats,
                        value=output_formats[0],
                        filterable=False, interactive=True
                    )
                    
                    template = gr.Textbox(
                        label="Шаблон именования выходных файлов",
                        value="NAME (STEM) MODEL",
                        info="Используйте ключи: \nNAME - имя входного файла без расширения, \nSTEM - имя стема, \nMODEL - имя модели разделения",
                        interactive=True
                    )

                    sep_state = gr.Textbox(visible=False) # Для хранения состояния результатов
                    status = gr.Textbox(
                        container=False, lines=3, interactive=False, max_lines=3, visible=False
                    )
                    separate_btn = gr.Button("Разделить", variant="primary", interactive=True).click(lambda: gr.update(visible=True), outputs=status)

        with gr.Column(variant="panel"):
            
            @gr.render(inputs=[sep_state], triggers=[sep_state.change])
            def render_medley_players(state):
                if not state:
                    return
                
                files = ast.literal_eval(state)
                with gr.Group():
                    for file_path in files:
                        with gr.Row(equal_height=True):
                            file_name = os.path.splitext(os.path.basename(file_path))[0]
                            output_stem = self.define_audio_with_size(value=file_path, label=file_name, type="filepath",
                                                        interactive=False,
                                                        show_download_button=True,
                                                        scale=15)
                            reuse_btn = gr.Button(
                                "Использовать снова", variant="secondary"
                            )

                            @reuse_btn.click(
                                inputs=[output_stem],
                                outputs=list_input_files,
                            )
                            def reuse_fn(stem_audio):
                                files = self.upload_files([stem_audio], copy=True)
                                return gr.update(choices=reversed(self.input_files), value=files)
        # Логика кнопки
        separate_btn.then(
            fn=self.wrap_separate,
            inputs=[
                list_input_files,
                model_name,
                output_format,
                template,
                overlap_mode,
                stereo_mode
            ],
            outputs=[sep_state, status]
        )

    def wrap_separate(self, files, model, fmt, tmpl, overlap, stereo, progress=gr.Progress()):
        """Обертка для вызова метода separate с обновлением UI"""
        if not files:
            raise gr.Error("Файлы не выбраны")
        
        # Путь сохранения (по аналогии с app.py)
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join(self.user_directory.path, "output_voxes", f"{timestamp}")
        
        results = self.separate(
            input=files,
            output_dir=out_dir,
            model_name=model,
            output_format=fmt,
            template=tmpl,
            use_overlapadd=overlap,
            stereo_mode=stereo,
            progress=progress
        )
        
        # Возвращаем строковое представление для gr.render
        return gr.update(value=str(results)), gr.update(visible=False)

def create_output_path(input_path, stem_name, model_name, model_id, output_format, store_dir, template):
    file_name = os.path.splitext(os.path.basename(input_path))[0]
    file_name_shorted = namer.short_input_name_template(
        template, STEM=stem_name, MODEL=model_name, ID=model_id, NAME=file_name
    )
    custom_name = namer.template(
        template,
        STEM=stem_name,
        MODEL=model_name,
        ID=model_id,
        NAME=file_name_shorted,
    )
    return os.path.join(store_dir, f"{custom_name}.{output_format}")

def main():
    parser = argparse.ArgumentParser(description="Адаптированный инференс Medley-Vox для MVSepLess Epsilon")
    parser.add_argument("--input", type=str, required=True, help="Путь к входному файлу или папке")
    parser.add_argument(
        "--template",
        type=str,
        default="NAME_STEM",
        help="Шаблон для имен выходных файлов",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="model",
        help="Имя модели для шаблона имен файлов",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="wav",
        choices=output_formats,
        help="Формат выходных файлов",
    )
    parser.add_argument("-m_id", "--model_id", type=int, required=True, help="Model ID")
    parser.add_argument(
        "--device", type=str, help="Какой девайс используется для разделения", default="cuda:0"
    )
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Путь к чекпоинту модели")
    parser.add_argument("--json_path", type=str, required=True, help="Путь к конфигурационному файлу модели")
    parser.add_argument(
        "--use_overlapadd",
        type=str,
        default=None,
        choices=overlap_add_methods,
        help="use overlapadd functions, ola, ola_norm, w2v will work with ola_window_len, ola_hop_len argugments. w2v_chunk and sf_chunk is chunk-wise processing based on VAD, so you have to specify the vad_method args. If you use sf_chunk (spectral_featrues_chunk), you also need to specify spectral_features.",
    )
    parser.add_argument(
        "--vad_method",
        type=str,
        default=vad_methods[1],
        choices=vad_methods,
        help="what method do you want to use for 'voice activity detection (vad) -- split chunks -- processing. Only valid when 'w2v_chunk' or 'sf_chunk' for args.use_overlapadd.",
    )
    parser.add_argument(
        "--spectral_features",
        type=str,
        default=spectral_features[0],
        choices=spectral_features,
        help="what spectral feature do you want to use in correlation calc in speaker assignment (only valid when using sf_chunk)",
    )
    parser.add_argument(
        "--w2v_ckpt_path",
        type=str,
        required=False,
        help="only valid when use_overlapadd is 'w2v' or 'w2v_chunk'.",
    )
    parser.add_argument(
        "--w2v_nth_layer_output",
        nargs="+",
        type=int,
        default=[0],
        help="wav2vec nth layer output",
    )
    parser.add_argument(
        "--ola_window_len",
        type=float,
        default=None,
        help="ola window size in [sec]",
    )
    parser.add_argument(
        "--ola_hop_len",
        type=float,
        default=None,
        help="ola hop size in [sec]",
    )
    parser.add_argument(
        "--use_ema_model",
        type=str2bool,
        default=True,
        help="use ema model or online model? only vaind when args.ema it True (model trained with ema)",
    )
    parser.add_argument(
        "--stereo",
        type=str,
        choices=stereo_modes,
        default="mono",
        help='',
    )
    parser.add_argument(
        "--mix_consistent_out",
        type=str2bool,
        default=True,
        help="only valid when the model is trained with mixture_consistency loss. Default is True.",
    )
    parser.add_argument(
        "--reorder_chunks",
        type=str2bool,
        default=True,
        help="ola reorder chunks",
    )
    parser.add_argument("--results_save_dir", type=str, default="./my_sep_results", help="Путь для сохранения результатов")
    args, _ = parser.parse_known_args()
    device = args.device
    from svs.models import load_model_with_args
    from svs.functions import load_ola_func_with_args
    processed_mixtures_1 = []
    processed_mixtures_2 = []
    processed_mixtures_3 = []
    if "cuda" in device.lower():
        # Извлекаем ID устройств для CUDA
        if ":" in device:
            device_spec = device.split(":")[1]
            device_ids = [int(id) for id in device_spec.split(",") if id.isdigit()]
        else:
            # Если указано просто "cuda", используем все доступные GPU
            device_ids = list(range(torch.cuda.device_count()))
        torch_device = torch.device("cuda" if not device_ids else f"cuda:{device_ids[0]}")
    elif "mps" in device.lower():
        device_ids = None
        torch_device = torch.device("mps")
    else:
        # CPU
        device_ids = None
        torch_device = torch.device("cpu")
    with open(args.json_path, "r") as f:
        args_dict = json.load(f)
    for key, value in args_dict["args"].items():
        setattr(args, key, value)

    model = load_model_with_args(args)

    device = torch_device
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if args.ema and args.use_ema_model:
        print("use ema model")
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        checkpoint = {
            k.replace("ema_model.module.", ""): v
            for k, v in checkpoint.items()
            if k.replace("ema_model.module.", "") in model_dict
        }
        # 2. overwrite entries in the existing state dict
        model_dict.update(checkpoint)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    elif args.ema and not args.use_ema_model:
        print("use ema online model")
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        checkpoint = {
            k.replace("online_model.module.", ""): v
            for k, v in checkpoint.items()
            if k.replace("online_model.module.", "") in model_dict
        }
        # 2. overwrite entries in the existing state dict
        model_dict.update(checkpoint)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    meter = pyln.Meter(args.sample_rate)
    if args.use_overlapadd:
        gc.collect()
        torch.cuda.empty_cache()
        continuous_nnet = load_ola_func_with_args(args, model, device, meter)

    os.makedirs(args.results_save_dir, exist_ok=True)

    if args.stereo == "left/right":
        mixture, sr = read(
            args.input,
            sr=args.sample_rate,
            mono=False
        )
        mixtures = split_channels(mixture)
    elif args.stereo == "mono":
        mixture, sr = read(
            args.input,
            sr=args.sample_rate,
            mono=True, flatten=True
        )
        mixtures = (mixture,)

    for mixture_ in mixtures:

        mixture_d, adjusted_gain = loudnorm(mixture_, -24.0, meter)
        max_samples = get_duration_from_array(mixture_d)
        mixture_d = mixture_d.reshape(1, 1, max_samples)
        mixture_d = torch.as_tensor(mixture_d, dtype=torch.float32).to(device)

        if args.use_overlapadd:
            out_wavs = continuous_nnet.forward(mixture_d)
        else:
            out_wavs = model.separate(mixture_d)

        if device.type == "cuda":
            out_wav_1 = out_wavs[0, 0, :].cpu().detach().numpy()
            out_wav_2 = out_wavs[0, 1, :].cpu().detach().numpy()
        else:
            out_wav_1 = out_wavs[0, 0, :]
            out_wav_2 = out_wavs[0, 1, :]

        out_wav_1 = out_wav_1 * db2linear(-adjusted_gain)
        out_wav_2 = out_wav_2 * db2linear(-adjusted_gain)

        processed_mixtures_1.append(out_wav_1)
        if args.use_overlapadd == "sf_chunk":
            processed_mixtures_2.append(mixture_ + -out_wav_1)
        else:
            processed_mixtures_2.append(out_wav_2)
            processed_mixtures_3.append(mixture_ + -(out_wav_1 + out_wav_2))

    vox_1 = multi_channel_array_from_arrays(*processed_mixtures_1, index=1, dtype=np.float32)
    vox_2 = multi_channel_array_from_arrays(*processed_mixtures_2, index=1, dtype=np.float32)
    if processed_mixtures_3:
        vox_3 = multi_channel_array_from_arrays(*processed_mixtures_3, index=1, dtype=np.float32)
        output_paths = [create_output_path(args.input, stem, args.model_name, args.model_id, args.output_format, args.results_save_dir, args.template) for stem in stems]
        output_arrays = [vox_1, vox_2, vox_3]
    else:
        output_paths = [create_output_path(args.input, stem, args.model_name, args.model_id, args.output_format, args.results_save_dir, args.template) for stem in [stems[0], stems[1]]]
        output_arrays = [vox_1, vox_2]       
    writed_files = multiwrite(output_arrays, [sr for __a in range(len(output_arrays))], [namer.iter(output_path_) for output_path_ in output_paths], 180 if args.output_format == "ogg" else 320, strict=True)
    sys.stdout.write(json.dumps({"done": writed_files}, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return writed_files

if __name__ == "__main__":
    main()
