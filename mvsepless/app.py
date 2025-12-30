import os, sys, gradio as gr, tempfile, zipfile, ast, json, pickle, argparse, shutil, re, subprocess
from separator import Separator, script_dir
from downloader import dw_yt_dlp
from check_colab import easy_check_is_colab
from datetime import datetime, timezone, timedelta
from functools import wraps
from audio import output_formats, input_extensions, check

tz = timezone(timedelta(hours=3))

class UserDirectory:
    path = script_dir
    def change_dir(self, dir: str):
        self.path = dir
        os.makedirs(dir, exist_ok=True)
    
user_directory = UserDirectory()
IS_COLAB = easy_check_is_colab()

if IS_COLAB:

    print("Обнаружена среда выполнения Colab")
    result = subprocess.run(['/bin/mount'], capture_output=True, text=True)

    for line in result.stdout.strip().split('\n'):
        if 'type fuse.drive' in line:
            parts = line.split(' type ')
            if len(parts) >= 2:
                source_mount = parts[0]
                source, mount_point = source_mount.split(' on ')
                user_directory.change_dir(os.path.join(mount_point, "MyDrive", "mvsepless-data-gdrive"))
                os.makedirs(user_directory.path, exist_ok=True)
                print(f"Обнаружен привязанный Google Диск\nПуть к привязанному диску: {mount_point}")
                break

class GradioHelper:

    def return_list(self, list, none=False, **kwargs):
        if list:
            return gr.update(choices=list, value=list[0] if not none else None, **kwargs)
        else:
            return gr.update(choices=[], value=None, **kwargs)

    def return_audio(self, label, path):
        return gr.update(label=label, value=path)

    def create_archive_advanced(self, file_list, archive_name="archive.zip"):
        try:
            print("Генерация ZIP-архива с результатами разделения...")
            with zipfile.ZipFile(
                archive_name, "w", zipfile.ZIP_DEFLATED
            ) as zipf:
                successful_files = 0

                for basename, stems in file_list:
                    for stem_name, stem_path in stems:
                        try:
                            if os.path.exists(stem_path) and os.path.isfile(
                                stem_path
                            ):
                                basename_ = os.path.basename(stem_path)
                                zipf.write(stem_path, basename_)
                                successful_files += 1
                                print(
                                    f"✓ Добавлен: {stem_path} -> {basename}"
                                )
                            else:
                                print(
                                    f"✗ Файл не найден или не является файлом: {stem_path}"
                                )

                        except Exception as e:
                            print(
                                f"✗ Ошибка при добавлении {stem_path}: {e}"
                            )

                print(f"\nАрхив создан: {archive_name}")
                print(f"Успешно добавлено файлов: {successful_files}")
                return os.path.abspath(archive_name)

        except Exception as e:
            print(f"Ошибка при создании архива: {e}")

    def extract_zip(self, zip_file_path, output_dir=None):
        """
        Распаковывает ZIP-архив с обработкой ошибок
        
        Args:
            zip_file_path: путь к ZIP-архиву
            output_dir: папка для распаковки (по умолчанию - текущая директория)
        """
        
        if output_dir is None:
            output_dir = os.path.splitext(zip_file_path)[0]
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                if zip_ref.testzip() is not None:
                    print("Предупреждение: архив может быть поврежден")
                
                zip_ref.extractall(output_dir)
                
                file_count = len(zip_ref.namelist())
                print(f"Успешно распаковано {file_count} файлов в {output_dir}")                
        except zipfile.BadZipFile:
            print("Ошибка: файл не является ZIP-архивом или поврежден")
            return []
        except PermissionError:
            print("Ошибка: нет прав на запись в указанную директорию")
            return []
        except Exception as e:
            print(f"Неизвестная ошибка: {e}")
            return []
        finally:
            input_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    input_files.append(os.path.join(root, file))
            return input_files

class History:
    def __init__(self):
        self.info = {}
        self.path = os.path.join(user_directory.path, "history.json")
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
    def add(self, state, model_name, timestamp):
        self.info[f"{timestamp} / {model_name}"] = state
    
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

class SeparatorGradio(Separator, GradioHelper):
    def __init__(self):
        super().__init__()
        self.input_files = []
        self.input_base_dir = os.path.join(user_directory.path, "input")
        self.output_base_dir = os.path.join(user_directory.path, "output")
        self.inputs_json_path = os.path.join(user_directory.path, "inputs.json")
        self.history = History()
        self.output_reader.debug = True
        self.load_from_file()

    def _write_file(self):
        """Записывает текущее состояние в файл"""
        try:
            with open(self.inputs_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.input_files, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Ошибка при записи в файл: {e}")

    def _save_to_file(func):
        """Декоратор для автоматического сохранения после вызова метода"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self._write_file()
            return result
        return wrapper

    def load_from_file(self):
        """Загрузить историю из файла"""
        if os.path.exists(self.inputs_json_path):
            with open(self.inputs_json_path, 'r', encoding='utf-8') as f:
                self.input_files = json.load(f)

    @_save_to_file
    def clean(self):
        self.input_files = []

    @_save_to_file
    def upload_files(self, input_files: list, copy: bool = False):
        if input_files: 
            input_dir = os.path.join(self.input_base_dir, datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S"))
            os.makedirs(input_dir, exist_ok=True)
            valid_files = [file for file in input_files if check(file)]
            valid_files_moved = []
            if valid_files:
                for file in valid_files:
                    basename = os.path.basename(file)
                    output_path = os.path.join(input_dir, basename)
                    if copy:
                        shutil.copy(file, output_path)
                    else:
                       shutil.move(file, output_path)
                    valid_files_moved.append(output_path)
                    self.input_files.append(output_path)
            return valid_files_moved
        else:
            return []

    def _separate_batch(
        self,
        input=None,
        model_name="Mel-Band-Roformer_Vocals_kimberley_jensen",
        ext_inst=True,
        output_format="mp3",
        output_bitrate="320k",
        template="NAME_(STEM)_MODEL",
        selected_stems=None,
        vr_aggr=5,
        mdx_denoise=False,
        progress=gr.Progress(track_tqdm=True),
    ):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        results = self.separate(
            input=input,
            output_dir=os.path.join(self.output_base_dir, timestamp),
            model_name=model_name,
            ext_inst=ext_inst,
            output_format=output_format,
            output_bitrate=output_bitrate,
            template=template,
            selected_stems=selected_stems,
            add_settings={
                "mdx_denoise": mdx_denoise,
                "vr_aggr": vr_aggr,
                "add_single_sep_text_progress": None,
            },
            progress=progress,
        )
        self.history.add(results, model_name, timestamp)
        return results

    def UI(self, theme, add_app=True, plugins=True, add_vbach=False):
        with gr.Blocks(theme=theme, title="Разделение музыки и вокала") as MVSEPLESS_LITE_UI:
            with gr.Tab("Разделение"):
                default_model = self.get_mn()
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            upload = gr.Files(show_label=False, type="filepath", interactive=True)
                            refresh_input_btn = gr.Button("Обновить", variant="primary", interactive=True)
                            list_input_files = gr.Dropdown(
                                label="Загрузить файлы",
                                choices=self.input_files,
                                value=[],
                                multiselect=True,
                                interactive=True,
                                filterable=False, scale=15
                            )
                            refresh_input_btn.click(lambda: gr.update(choices=reversed(self.input_files), value=[]), outputs=list_input_files)
                                
                            @upload.upload(inputs=[upload], outputs=[list_input_files, upload])
                            def upload_files(input_files):
                                files = self.upload_files(input_files)
                                return gr.update(
                                    choices=reversed(self.input_files), value=files
                                ), gr.update(value=[])
    
                    with gr.Column():
                        with gr.Group():
                            model_name = gr.Dropdown(
                                label="Имя модели", choices=default_model, value=default_model[0]
                            )
                            extract_instrumental = gr.Checkbox(
                                label="Извлечь инструментал", value=True, interactive=True, scale=3
                            )
                            stems = gr.CheckboxGroup(
                                label="Выберите стемы",
                                choices=self.get_stems(default_model[0]),
                                value=[],
                                interactive=False, scale=8
                            )
                            with gr.Accordion(label="Дополнительные настройки", open=False):
                                vr_aggr, mdx_denoise = gr.Slider(
                                    label="Сила подавления для VR моделей",
                                    minimum=0,
                                    maximum=100,
                                    value=5,
                                    step=1,
                                    interactive=True,
                                ), gr.Checkbox(
                                    label="Включить шумоподавление для MDX-NET моделей",
                                    value=False,
                                    interactive=True,
                                )

                            @model_name.change(
                                inputs=[model_name], outputs=[extract_instrumental, stems]
                            )
                            def update_model_name(model_name):
                                stems = self.get_stems(model_name)
                                target_instrument = self.get_tgt_inst(model_name)
                                return gr.update(
                                    value=target_instrument is not None, 
                                    visible=(target_instrument is None and len(stems) > 2) or target_instrument is not None
                                ), gr.update(
                                    choices=stems, value=[], interactive=target_instrument is None
                                )
                            
                            with gr.Row():
                                output_format = gr.Dropdown(
                                    label="Формат выходного файла",
                                    interactive=True,
                                    choices=output_formats,
                                    value="mp3",
                                    filterable=False,
                                )
                                output_bitrate = gr.Slider(
                                    label="Битрейт выходного файла",
                                    minimum=64,
                                    maximum=512,
                                    step=32,
                                    value=320,
                                    interactive=True,
                                )
                                output_format.change(
                                    lambda x: gr.update(
                                        visible=False if x in ["wav", "flac", "aiff"] else True
                                    ),
                                    inputs=output_format,
                                    outputs=output_bitrate,
                                )
                            template = gr.Textbox(
                                label="Шаблон именования выходных файлов",
                                interactive=True,
                                value="NAME (STEM) MODEL",
                                info="Используйте ключи: \nNAME - имя входного файла без расширения, \nSTEM - имя стема, \nMODEL - имя модели разделения",
                            )
                            sep_state = gr.Textbox(
                                label="Состояние разделения",
                                interactive=False,
                                value="",
                                visible=False,
                            )
                            status = gr.Textbox(
                                container=False, lines=3, interactive=False, max_lines=3, visible=False
                            )
                            separate_btn = gr.Button("Разделить", variant="primary", interactive=True).click(lambda: gr.update(visible=True), outputs=status)
                            @separate_btn.then(
                                inputs=[
                                    list_input_files,
                                    model_name,
                                    extract_instrumental,
                                    output_format,
                                    output_bitrate,
                                    template,
                                    stems,
                                    mdx_denoise,
                                    vr_aggr,
                                ],
                                outputs=[sep_state, status],
                                show_progress="full",
                            )
                            def wrap(
                                i,
                                mn,
                                ei,
                                of,
                                ob,
                                t,
                                stems,
                                mdx_denoise,
                                vr_aggr,
                                progress=gr.Progress(track_tqdm=True),
                            ):
                                results = self._separate_batch(
                                    i,
                                    mn,
                                    ei,
                                    of,
                                    ob,
                                    t,
                                    stems,
                                    vr_aggr,
                                    mdx_denoise,
                                    progress=progress,
                                )
                                return gr.update(value=str(results)), gr.update(visible=False)

                with gr.Column(variant="panel"):
                    gr.Markdown("<center><h3>Результаты</h3></center>")

                    with gr.Group():
                        with gr.Row(equal_height=True):
                            list_separations = gr.Dropdown(
                                label="Выберите результаты разделения",
                                choices=[],
                                value=None,
                                interactive=True, scale=14
                            )
                            list_separations.change(lambda x: gr.update(value=str(self.history.get(x))), inputs=[list_separations], outputs=[sep_state])
                            refresh_separations_btn = gr.Button("Обновить", scale=2, interactive=True)
                            refresh_separations_btn.click(lambda: self.return_list(self.history.get_list(), none=True), outputs=[list_separations])
                            gr.on(fn=lambda: self.return_list(self.history.get_list(), none=True), outputs=[list_separations])

                    @gr.render(inputs=[sep_state], triggers=[sep_state.change])
                    def players(state):
                        if state != "":
                            state_loaded = ast.literal_eval(state)
                            if state_loaded:
                                archive_stems = self.create_archive_advanced(
                                    state_loaded,
                                    os.path.join(
                                        tempfile.tempdir,
                                        f"mvsepless_output_{datetime.now(tz).strftime('%Y%m%d_%H%M%S')}.zip",
                                    ),
                                )
                                for basename, stems in state_loaded:
                                    with gr.Group():
                                        gr.Markdown(f"<h4><center>{basename}</center></h4>")
                                        for stem_name, stem_path in stems:
                                            with gr.Row(equal_height=True):
                                                output_stem = gr.Audio(
                                                    value=stem_path,
                                                    label=stem_name,
                                                    type="filepath",
                                                    interactive=False,
                                                    show_download_button=True,
                                                    scale=15,
                                                )
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

                                gr.DownloadButton(
                                    label="Скачать как ZIP", value=archive_stems, interactive=True
                                )


            with gr.Tab("Загрузка аудио"):
                with gr.Tab("С интернета"):
                    input_url = gr.Textbox(
                        label="URL входного файла", interactive=True
                    )
                    with gr.Row(equal_height=True):
                        inputs_url_format = gr.Dropdown(
                            label="Формат входного файла",
                            interactive=True,
                            choices=output_formats,
                            value="mp3",
                            filterable=False,
                        )
                        inputs_url_bitrate = gr.Slider(
                            label="Битрейт входного файла",
                            minimum=32,
                            maximum=512,
                            step=32,
                            value=320,
                            interactive=True,
                        )
                        inputs_url_format.change(
                            lambda x: gr.update(
                                visible=False if x in ["wav", "flac", "aiff"] else True
                            ),
                            inputs=inputs_url_format,
                            outputs=inputs_url_bitrate,
                        )
                    with gr.Row(equal_height=True):
                        inputs_url_cookie = gr.UploadButton(
                            label="Файл cookie (необязательно)",
                            interactive=True,
                            type="filepath",
                            file_count="single",
                            file_types=[".txt", ".cookies"],
                            variant="secondary",
                        )
                        add_inputs_url_btn = gr.Button(
                            "Добавить файл", variant="primary"
                        )
                        @add_inputs_url_btn.click(
                            inputs=[
                                input_url,
                                inputs_url_format,
                                inputs_url_bitrate,
                                inputs_url_cookie,
                            ]
                        )
                        def add_inputs_from_url_fn(input_u, fmt, br, cookie):
                            if input_u:
                                downloaded_file = dw_yt_dlp(
                                    url=input_u,
                                    #output_dir=tempfile.mkdtemp(),
                                    output_format=fmt,
                                    output_bitrate=str(int(br)),
                                    cookie=cookie,
                                )
                                if downloaded_file:
                                    if os.path.exists(downloaded_file):
                                        if check(downloaded_file):
                                            self.upload_files([downloaded_file])
                                            gr.Warning(title="Файл успешно загружен", message="")

                with gr.Tab("С устройства"):
                    with gr.Row():
                        add_inputs_from_device_directory = gr.File(label="Загрузить аудио (директория)", interactive=True, file_count="directory")
                        add_inputs_from_device_files = gr.File(label="Загрузить аудио (файлы)", interactive=True, file_count="multiple")
                        add_inputs_from_device_zip = gr.File(label="Загрузить аудио (ZIP-архив)", interactive=True, file_count="single", file_types=[".zip"])
                        @add_inputs_from_device_directory.upload(
                            inputs=[add_inputs_from_device_directory], outputs=[add_inputs_from_device_directory]
                        )
                        def upload_from_directory(file_list):
                            files = self.upload_files(file_list)
                            gr.Warning(title="Файлы успешно загружены", message="")
                            return gr.update(value=[])
                        @add_inputs_from_device_files.upload(
                            inputs=[add_inputs_from_device_files], outputs=[add_inputs_from_device_files]
                        )
                        def upload_from_files(file_list):
                            files = self.upload_files(file_list)
                            gr.Warning(title="Файлы успешно загружены", message="")
                            return gr.update(value=[])
                        @add_inputs_from_device_zip.upload(
                            inputs=[add_inputs_from_device_zip], outputs=[add_inputs_from_device_zip]
                        )
                        def upload_from_zip(zip_file):
                            with tempfile.TemporaryDirectory() as tmp_zip:
                                _files = self.extract_zip(zip_file, tmp_zip)
                                files = self.upload_files(_files)
                                gr.Warning(title="Файлы успешно загружены", message="")
                            return gr.update(value=None)

            from additional_app import AutoEnsembless, ManualEnsembless, PluginManager, Inverter_UI
            if add_app:
                with gr.Tab("Ансамбль"):
                    
                    with gr.Tab("Авто-ансамбль"):
                        AutoEnsembless(self.input_files, self.upload_files, user_directory).UI()

                    with gr.Tab("Ручной ансамбль"):
                        ManualEnsembless(user_directory).UI()

                with gr.Tab("Вычитание"):
                    Inverter_UI().UI()

            if add_vbach:
                from vbach import Vbach, vbach_inference, model_manager as voice_model_manager
                with gr.Tab("Преобразование"):
                    Vbach(user_directory).UI()
                if add_app:
                    with gr.Tab("Генерация каверов"):
                        from vbachgen import VbachGen
                        VbachGen(voice_model_manager, self.input_files, self.upload_files, user_directory, vbach_inference).UI()

            if plugins:
                with gr.Tab("Плагины"):
                    PluginManager().UI()


        return MVSEPLESS_LITE_UI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVSepless")
    parser.add_argument(
        "--port", type=int, default=None, help="Порт для запуска сервера Gradio."
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Создать публичную ссылку для приложения Gradio.",
    )
    parser.add_argument(
        "--add_app",
        action="store_true",
        help="Включить дополнительные приложения",
    )
    parser.add_argument(
        "--use_plugins",
        action="store_true",
        help="Включить плагины",
    )
    parser.add_argument(
        "--vbach",
        action="store_true",
        help="Включить Vbach",
    )
    args = parser.parse_args()

    SeparatorGradio().UI(gr.themes.Citrus(
            primary_hue="teal",
            secondary_hue="blue",
            neutral_hue="blue",
            spacing_size="sm",
            font=[
                gr.themes.GoogleFont("Montserrat"),
                "ui-sans-serif",
                "system-ui",
                "sans-serif",
            ],
        ), args.add_app, args.use_plugins, args.vbach).launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
            allowed_paths=["/"],
            debug=True,
        )