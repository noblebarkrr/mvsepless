import os
import sys
import ast
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import gradio as gr
import json
import argparse
import subprocess
from device import all_ids, set_device, cuda_available
from audio import check, output_formats
from gradio_helper import GradioHelper, tz
from datetime import datetime
from downloader import dw_file
from functools import wraps
from svs.infer_m import overlap_add_methods, stereo_modes

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

class History:
    def __init__(self, user_directory):
        self.info = {}
        self.user_directory = user_directory
        self.path = os.path.join(self.user_directory.path, "history_medley_vox.json")
        self.load_from_file()
    
    def _save_to_file(func):
        """Декоратор для автоматического сохранения после вызова метода"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self._write_file()
            return result
        return wrapper
    
    def _write_file(self):
        """Записывает текущее состояние в файл"""
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.info, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Ошибка при записи в файл: {e}")
    
    @_save_to_file
    def add(self, state, model_name, timestamp, stereo_mode):
        self.info[f"{timestamp} / {model_name} / {stereo_mode}"] = state
    
    @_save_to_file
    def clear(self):
        self.info = {}
    
    def get_list(self):
        return sorted([key for key in self.info], reverse=True)
    
    def get(self, key):
        return self.info.get(key, [])
    
    def load_from_file(self):
        """Загрузить историю из файла"""
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                self.info = json.load(f)

class MedleyVoxSeparator(MedleyVoxModelManager, GradioHelper):

    def __init__(self, input_files=None, upload_files=None, user_directory=None, device=set_device()):
        super().__init__()
        self.input_files = input_files
        self.upload_files = upload_files
        self.user_directory = user_directory
        self.device = device
        self.history = History(self.user_directory)

    class OutputReader:
        def __init__(self, debug=False):
            self.debug = debug

        def parse_json_line(self, line):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None

        def reaction_line(self, line, progress, stereo_mode, add_text):
            _add_text = ""
            if add_text != "" or add_text is not None:
                _add_text = f"| {add_text}"

            data = self.parse_json_line(line)
            if data is None:
                return None
            elif "processing" in data:
                progress_a = data["processing"]
                processed = progress_a.get("processed", 0)
                total = progress_a.get("total", 1)
                if total > 0:
                    percent = int((processed / total) * 100)
                    if stereo_mode == "left/right":
                        mixture_info = progress_a.get("mixture")
                        match mixture_info:
                            case 1:
                                _add_text += " (L)"
                            case 2:
                                _add_text += " (R)"
                        progress((processed, total), desc=f"Обработано: {percent}% {_add_text}", unit=progress_a.get("unit", "сэмплов"))
                        print(f"\rОбработано: {percent}%", end="")
                        match mixture_info:
                            case 1:
                                print(f"\rОбработано: {percent}% (L)", end="")
                            case 2:
                                print(f"\rОбработано: {percent}% (R)", end="")
                    else:
                        progress((processed, total), desc=f"Обработано: {percent}% {_add_text}", unit=progress_a.get("unit", "сэмплов"))
                        print(f"\rОбработано: {percent}% ", end="")
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
            "-m", "svs.infer_m",
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
                        line, progress, stereo_mode, add_text_progress
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
                        line, progress, stereo_mode, add_text_progress
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
                        container=False, lines=4, interactive=False, max_lines=4, visible=False
                    )
                    separate_btn = gr.Button("Разделить", variant="primary", interactive=True).click(lambda: gr.update(visible=True), outputs=status)

        with gr.Column(variant="panel"):
            gr.Markdown("<center><h3>Результаты</h3></center>")
            with gr.Group():
                with gr.Row(equal_height=True):
                    list_seps = gr.Dropdown(
                        label="Выберите результаты разделения",
                        choices=[],
                        value=None,
                        interactive=True, scale=14
                    )
                    list_seps.change(lambda x: gr.update(value=str(self.history.get(x))), inputs=[list_seps], outputs=[sep_state])
                    refresh_conversions_btn = gr.Button("Обновить", scale=2, interactive=True)
                    refresh_conversions_btn.click(lambda: gr.update(choices=self.history.get_list(), value=None), outputs=[list_seps])
                    gr.on(fn=lambda: gr.update(choices=self.history.get_list(), value=None), outputs=[list_seps])

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
        self.history.add(results, model, timestamp, stereo)
        # Возвращаем строковое представление для gr.render
        return gr.update(value=str(results)), gr.update(visible=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVSepless (Medley-Vox) CLI")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Показать список моделей",
    )
    parser.add_argument(
        "--input", type=str, default="./", help="Входной аудиофайл или каталог."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Каталог для выходных файлов."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="multi_singing_librispeech", # Изменил на существующую в вашем словаре
        help="Имя модели разделения.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="wav",
        choices=output_formats,
        help="Формат выходного файла.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="NAME_(STEM)_MODEL",
        help="Шаблон именования выходных файлов.",
    )
    parser.add_argument(
        "--overlap",
        type=str,
        default="ola",
        choices=overlap_add_methods,
        help="Метод Overlap-Add.",
    )
    parser.add_argument(
        "--stereo",
        type=str,
        default="mono",
        choices=stereo_modes,
        help="Режим стерео (mono|left/right).",
    )

    args = parser.parse_args()

    if args.list:
        print("Доступные модели:")
        for mn in MedleyVoxModelManager().get_mn():
            print("  - ", mn, sep="")
    else:
        # Сбор списка файлов
        input_files = []
        if os.path.isdir(args.input):
            for file in os.listdir(args.input):
                full_path = os.path.join(args.input, file)
                if os.path.isfile(full_path) and check(full_path):
                    input_files.append(full_path)
        else:
            if os.path.exists(args.input) and check(args.input):
                input_files = args.input
            else:
                print(f"Ошибка: Файл {args.input} не найден или не является аудио.")
                sys.exit(1)

        if not input_files:
            print("Ошибка: Не найдено подходящих аудиофайлов для обработки.")
            sys.exit(1)

        separator = MedleyVoxSeparator()
        
        try:
            results = separator.separate(
                input=input_files,
                output_dir=args.output_dir,
                model_name=args.model_name,
                output_format=args.output_format,
                template=args.template,
                use_overlapadd=args.overlap,
                stereo_mode=args.stereo,
                progress=gr.Progress()
            )
            print("\nРазделение завершено успешно.")
            print(f"Результаты сохранены в: {args.output_dir if args.output_dir else 'текущую директорию'}")
            for r in results:
                print(f" - {r}")
                
        except Exception as e:
            print(f"\nПроизошла ошибка при выполнении: {e}")
