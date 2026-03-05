import os, sys, gradio as gr, tempfile, zipfile, ast, json, pickle, argparse, shutil, re, subprocess
from separator import Separator, script_dir
from downloader import dw_yt_dlp
from check_colab import easy_check_is_colab
from datetime import datetime, timezone, timedelta
from functools import wraps
from device import all_ids, set_device, cuda_available, mps_available
from audio import output_formats, input_extensions, check
from gradio_helper import GradioHelper, tz

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

class DownloadModelManager(Separator):
    def __init__(self):
        super().__init__()
        self.dwm_preset_path = os.path.join(script_dir, "dwm_preset.json")
        self.load_dwm_preset(self.dwm_preset_path)

    def load_dwm_preset(self, path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.dwm_presets: dict = json.load(f)
        else:
            self.dwm_presets: dict = {"Все модели": self.get_mn()}

    def parse_models_from_dwm_preset(self, key):
        return self.dwm_presets.get(key, [])

    def batch_download(self, keys, progress=gr.Progress()):
        if keys:
            total = len(keys)
            for i, key in enumerate(keys, start=1):
                progress(i / total, desc=f"Модель {i}/{total}")
                print(f"Модель {i}/{total}")
                if key in self.models_info:
                    self.install_model(self.get_mt(key), key)
                else:
                    print(f"Указанной модели {key} не существует... Пропускаем")
                    gr.Warning(message="", title=f"Указанной модели {key} не существует... Пропускаем")
        print("Загрузка завершена")
        gr.Warning(message="", title="Загрузка завершена")
        return None
    
    def delete_models_cache(self):
        shutil.rmtree(self.models_cache_dir, ignore_errors=True)
        os.makedirs(self.models_cache_dir, exist_ok=True)
        print("Все скачанные модели удалены из памяти!")
        gr.Warning(message="", title="Все скачанные модели удалены из памяти!")

class SeparatorGradio(GradioHelper, DownloadModelManager):
    def __init__(self):
        super().__init__()
        self.input_files = []
        self.input_base_dir = os.path.join(user_directory.path, "input")
        self.output_base_dir = os.path.join(user_directory.path, "output")
        self.inputs_json_path = os.path.join(user_directory.path, "inputs.json")
        self.history = History()
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
        vr_post_process=False,
        vr_high_end_process=False,
        mdx_denoise=False,
        use_spec_invert=False,
        econom_mode=None,
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
                "vr_post_process": vr_post_process,
                "vr_high_end_process": vr_high_end_process,
                "econom_mode": econom_mode,
                "add_single_sep_text_progress": None,
            } if econom_mode is not None else {
                "mdx_denoise": mdx_denoise,
                "vr_aggr": vr_aggr,
                "vr_post_process": vr_post_process,
                "vr_high_end_process": vr_high_end_process,
                "add_single_sep_text_progress": None,
            },
            use_spec_invert=use_spec_invert,
            progress=progress,
        )
        self.history.add(results, model_name, timestamp)
        return results

    def UI(self, theme, add_app=True, plugins=True, add_vbach=False, medley_vox=False):
        with gr.Blocks(theme=theme, title="Разделение музыки и вокала") as MVSEPLESS_LITE_UI:
            if not cuda_available:
                gr.Markdown("<h2><center>ВНИМАНИЕ! Используется CPU, инференс слишком медленно работает<center><h2>")
            with gr.Tab("Разделение"):
                default_model = self.get_mn()
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            upload = gr.Files(show_label=False, type="filepath", interactive=True)
                            refresh_input_btn = gr.Button("Обновить", variant="primary", interactive=True)
                            list_input_files = gr.Dropdown(
                                label="Загрузить файлы",
                                choices=reversed(self.input_files),
                                value=[],
                                multiselect=True,
                                interactive=True,
                                filterable=False, scale=15
                            )
                            gr.on(fn=lambda: gr.update(choices=reversed(self.input_files), value=[]), outputs=list_input_files, trigger_mode="once")
                            refresh_input_btn.click(lambda: gr.update(choices=reversed(self.input_files), value=[]), outputs=list_input_files)
                                
                            @upload.upload(inputs=[upload], outputs=[list_input_files, upload])
                            def upload_files(input_files):
                                files = self.upload_files(input_files)
                                return gr.update(
                                    choices=reversed(self.input_files), value=files
                                ), gr.update(value=[])
    
                    with gr.Column():
                        with gr.Group():
                            with gr.Row(equal_height=True):
                                model_name = gr.Dropdown(
                                    label="Имя модели", choices=default_model, value=default_model[0], interactive=True, scale=9
                                )
                                model_name_refresh_btn = gr.Button("🔄", size="lg", scale=2, interactive=True, min_width=50)

                            show_only_downloaded_models = gr.Checkbox(
                                label="Показать только загруженные модели", value=False, interactive=True
                            )
                            @model_name_refresh_btn.click(inputs=[model_name, show_only_downloaded_models], outputs=model_name)
                            def refresh_model_fn(name, only_downloaded):
                                models = []
                                if only_downloaded:
                                    models = self.get_mn_dwloaded()
                                else:
                                    models = self.get_mn()

                                if models:
                                    first_value = models[0]
                                else:
                                    first_value = None

                                if name in models:
                                    value = name
                                else:
                                    value = first_value
                                return gr.update(choices=models, value=value)

                            @show_only_downloaded_models.change(inputs=[model_name, show_only_downloaded_models], outputs=model_name, trigger_mode="once")
                            def refresh_model_fn2(name, only_downloaded):
                                models = []
                                if only_downloaded:
                                    models = self.get_mn_dwloaded()
                                else:
                                    models = self.get_mn()

                                if models:
                                    first_value = models[0]
                                else:
                                    first_value = None

                                if name in models:
                                    value = name
                                else:
                                    value = first_value
                                return gr.update(choices=models, value=value)

                            extract_instrumental = gr.Checkbox(
                                label="Извлечь инструментал", value=False, interactive=True, visible=False
                            )
                            stems = gr.CheckboxGroup(
                                label="Выберите стемы",
                                choices=self.get_stems(default_model[0]),
                                value=[],
                                interactive=True, scale=8
                            )
                            with gr.Accordion(label="Дополнительные настройки", open=False):
                                with gr.Group():
                                    gr.Markdown("<h4>VR</h4>", container=True)
                                    vr_aggr = gr.Slider(
                                        label="Агрессивность",
                                        minimum=0,
                                        maximum=100,
                                        value=5,
                                        step=1,
                                        interactive=True,
                                    )
                                    vr_enable_post_process = gr.Checkbox(
                                        label="Дополнительная обработка для улучшения качества разделения", value=False, interactive=True
                                    )
                                    vr_enable_high_end_process = gr.Checkbox(
                                        label="Восстановление недостающих высоких частот", value=False, interactive=True
                                    )
                                    gr.Markdown("<h4>MDX-NET</h4>", container=True)
                                    mdx_denoise = gr.Checkbox(
                                        label="Шумоподавление",
                                        value=False,
                                        interactive=True,
                                    )
                                    gr.Markdown("<h4>Инвертирование результата</h4>", container=True)
                                    use_spec_for_extract_instrumental = gr.Checkbox(
                                        label="При извлечении инструментала/второго стема/остатка использовать спектрограмму", value=False, interactive=True
                                    )
                                    gr.Markdown("<h4>Экономия</h4>", container=True)
                                    econom_mode = gr.Checkbox(
                                        label="Включить эконом-режим", value=False, interactive=True
                                    )

                            @model_name.change(
                                inputs=[model_name], outputs=[extract_instrumental, stems]
                            )
                            def update_model_name(model_name):
                                stems = self.get_stems(model_name)
                                return gr.update(
                                    visible=len(stems) > 2
                                ), gr.update(
                                    choices=stems, value=[], interactive=True
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
                                    container=False, lines=4, interactive=False, max_lines=4, visible=False
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
                                    vr_enable_post_process,
                                    vr_enable_high_end_process,
                                    use_spec_for_extract_instrumental,
                                    econom_mode
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
                                vr_pp,
                                vr_hip,
                                u_spec,
                                ec_mode,
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
                                    vr_pp,
                                    vr_hip,
                                    mdx_denoise,
                                    u_spec,
                                    ec_mode,
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
                            list_separations.change(lambda x: gr.update(value=str(self.history.get(x))), inputs=[list_separations], outputs=[sep_state], trigger_mode="once")
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
                                                output_stem = self.define_audio_with_size(
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
                    with gr.Group():
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
                        with gr.Group():
                            gr.Markdown("<h3><center>Загрузить аудио-файлы из директории</h3></center>")
                            add_inputs_from_device_directory = gr.File(show_label=False, label="Загрузить аудио (директория)", interactive=True, file_count="directory")
                        with gr.Group():
                            gr.Markdown("<h3><center>Загрузить аудио-файлы</h3></center>")
                            add_inputs_from_device_files = gr.File(show_label=False, label="Загрузить аудио (файлы)", interactive=True, file_count="multiple")
                        with gr.Group():
                            gr.Markdown("<h3><center>Загрузить ZIP-архив с аудио</h3></center>")
                            add_inputs_from_device_zip = gr.File(show_label=False, label="Загрузить аудио (ZIP-архив)", interactive=True, file_count="single", file_types=[".zip"])
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

                    with gr.Group():
                        with gr.Row(equal_height=True):
                            add_inputs_from_path = gr.Textbox(label="Загрузить аудио (путь к файлу)", interactive=True, value="", scale=15)
                            add_inputs_from_path_btn = gr.Button("Загрузить", variant="primary", interactive=True, scale=3)
                            @add_inputs_from_path_btn.click(inputs=add_inputs_from_path, outputs=add_inputs_from_path)
                            def upload_from_path(path):
                                files = self.upload_files([path])
                                gr.Warning(title="Файл успешно загружен", message="")
                                return gr.update(value="")

            with gr.Tab("Менеджер моделей"):
                with gr.Tab("Скачать модель"):
                    with gr.Group():
                        select_dwm_preset = gr.Dropdown(
                            label="Выберите пресет",
                            interactive=True,
                            choices=self.dwm_presets,
                            value=None,
                        )
                        select_dwm_names = gr.Dropdown(
                            label="Выберите модели",
                            interactive=True,
                            choices=default_model, value=[],
                            multiselect=True
                        )
                        dwm_status = gr.Textbox(
                            container=False, lines=3, interactive=False, max_lines=3, visible=False
                        )
                        download_dwm_button = gr.Button("Скачать")
                        select_dwm_preset.change(lambda x: gr.update(value=self.parse_models_from_dwm_preset(x)), inputs=select_dwm_preset, outputs=select_dwm_names, trigger_mode="once")
                        download_dwm_button.click(lambda: gr.update(visible=True), outputs=dwm_status).then(lambda x: (self.batch_download(x), gr.update(visible=False)), inputs=select_dwm_names, outputs=[gr.State(None), dwm_status])
                with gr.Tab("Удалить все модели"):
                    gr.Markdown("<h3><center>Это действие необратимо</center></h3>")
                    delete_models_cache_btn = gr.Button("Удалить ВСЁ!")
                    delete_models_cache_btn.click(self.delete_models_cache, inputs=None, outputs=None)

            if medley_vox:
                from medley_vox_infer import MedleyVoxSeparator
                with gr.Tab("Разделение вокалов (Medley-Vox)"):
                    _medley_vox = MedleyVoxSeparator(self.input_files, self.upload_files, user_directory, device=self.device)
                    _medley_vox.UI()

            from additional_app import AutoEnsembless, ManualEnsembless, PluginManager, Inverter_UI, AudioApp
            if add_app:
                with gr.Tab("Обработка аудио"):
                    _audio_app = AudioApp(user_directory)
                    _audio_app.UI()
                with gr.Tab("Ансамбль"):
                    
                    with gr.Tab("Авто-ансамбль"):
                        _auto_ensembless = AutoEnsembless(self.input_files, self.upload_files, user_directory, device=self.device)
                        _auto_ensembless.UI()
                    with gr.Tab("Ручной ансамбль"):
                        ManualEnsembless(user_directory).UI()

                with gr.Tab("Вычитание"):
                    Inverter_UI().UI()

            if add_vbach:
                from vbach import Vbach, vbach_inference, model_manager as voice_model_manager
                with gr.Tab("Преобразование"):
                    _vbach = Vbach(user_directory, device=self.device)
                    _vbach.UI()
                if add_app:
                    with gr.Tab("Генерация каверов"):
                        from vbachgen import VbachGen
                        _vbach_gen = VbachGen(voice_model_manager, self.input_files, self.upload_files, user_directory, vbach_inference, device=self.device)
                        _vbach_gen.UI()

            if plugins:
                with gr.Tab("Плагины"):
                    PluginManager().UI()
            
            with gr.Tab("Устройство"):
                with gr.Group():
                    device_radio = gr.CheckboxGroup(label="ID устройств CUDA", choices=all_ids, interactive=True)
                    pref_cuda = gr.Checkbox(label="Отдать предпочтение устройствам CUDA (Если они есть)", value=True)
                    current_device = gr.Textbox(label="Текущее устройство", value=self.device)
                gr.on(fn=lambda: (self.device), outputs=[current_device])
                def show_device(a1, a2):
                    _device = set_device(a1, prefer_gpu=a2)
                    self.device = _device
                    if add_app:
                        _auto_ensembless.device = _device
                        if add_vbach:
                            _vbach_gen.device = _device
                    if add_vbach:
                        _vbach.device = _device
                    print(f"Выбранное устройство: {self.device}")
                    return self.device
                device_radio.change(show_device, inputs=[device_radio, pref_cuda], outputs=current_device, trigger_mode="once")
                pref_cuda.change(show_device, inputs=[device_radio, pref_cuda], outputs=current_device, trigger_mode="once")


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
    parser.add_argument(
        "--medley_vox",
        action="store_true",
        help="Включить Medley-Vox",
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
        ), args.add_app, args.use_plugins, args.vbach, args.medley_vox).launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
            allowed_paths=["/"],
            debug=True,
            inbrowser=True
        )
