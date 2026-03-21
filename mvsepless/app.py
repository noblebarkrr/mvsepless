import os
import gradio as gr
import tempfile
import ast
import json
import argparse
import shutil
import subprocess
from typing import List, Optional, Tuple, Dict, Any, Union
from datetime import datetime
from functools import wraps

from separator import Separator, script_dir
from datetime import datetime
from functools import wraps
from audio import output_formats, check
from gradio_helper import GradioHelper, tz, all_ids, set_device, cuda_available, easy_check_is_colab, dw_yt_dlp, hf_spaces_gpu, zerogpu_available
from i18n import _i18n, CURRENT_LANGUAGE, set_language


class UserDirectory:
    """Класс для управления пользовательской директорией"""
    
    def __init__(self) -> None:
        self.path: str = ""
    
    def change_dir(self, directory: str) -> None:
        """
        Изменить пользовательскую директорию
        
        Args:
            directory: Путь к директории
        """
        self.path = directory
        os.makedirs(directory, exist_ok=True)
    

user_directory: UserDirectory = UserDirectory()
IS_COLAB: bool = easy_check_is_colab()

if IS_COLAB:
    print(_i18n("msg_colab_detected"))
    result = subprocess.run(['/bin/mount'], capture_output=True, text=True)

    for line in result.stdout.strip().split('\n'):
        if 'type fuse.drive' in line:
            parts = line.split(' type ')
            if len(parts) >= 2:
                source_mount = parts[0]
                source, mount_point = source_mount.split(' on ')
                user_directory.change_dir(os.path.join(mount_point, "MyDrive", "mvsepless-data-gdrive"))
                os.makedirs(user_directory.path, exist_ok=True)
                print(_i18n("msg_gdrive_mounted", path=mount_point))
                break


class History:
    """Класс для управления историей разделений"""
    
    def __init__(self) -> None:
        self.info: Dict[str, List] = {}
        self.path: str = os.path.join(user_directory.path, "history", "mvsepless.json")
        os.makedirs(os.path.join(user_directory.path, "history"), exist_ok=True)
        self.load_from_file()
    
    def _save_to_file(func):
        """Декоратор для автоматического сохранения после вызова метода"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self._write_file()
            return result
        return wrapper
    
    def _write_file(self) -> None:
        """Записывает текущее состояние в файл"""
        try:
            dir_path: str = os.path.dirname(self.path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.info, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"{_i18n('error_writing_file')}: {e}")
    
    @_save_to_file
    def add(self, state: List, model_name: str, timestamp: str) -> None:
        """
        Добавить запись в историю
        
        Args:
            state: Состояние разделения
            model_name: Имя модели
            timestamp: Временная метка
        """
        self.info[f"{timestamp} / {model_name}"] = state
    
    @_save_to_file
    def clear(self) -> None:
        """Очистить историю"""
        self.info = {}
    
    def get_list(self) -> List[str]:
        """
        Получить список записей истории
        
        Returns:
            Список ключей истории
        """
        return sorted([key for key in self.info], reverse=True)
    
    def get(self, key: str) -> List:
        """
        Получить запись истории по ключу
        
        Args:
            key: Ключ записи
        
        Returns:
            Запись истории
        """
        return self.info.get(key, [])
    
    def load_from_file(self) -> None:
        """Загрузить историю из файла"""
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                self.info = json.load(f)


class DownloadModelManager(Separator):
    """Менеджер загрузки моделей"""
    
    def __init__(self) -> None:
        super().__init__()
        self.dwm_preset_path: str = os.path.join(script_dir, "dwm_preset.json")
        self.load_dwm_preset(self.dwm_preset_path)

    def load_dwm_preset(self, path: str) -> None:
        """
        Загрузить пресет для загрузки моделей
        
        Args:
            path: Путь к файлу пресета
        """
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.dwm_presets: Dict[str, List[str]] = json.load(f)
        else:
            self.dwm_presets: Dict[str, List[str]] = {_i18n("all_models"): self.get_mn()}

    def parse_models_from_dwm_preset(self, key: str) -> List[str]:
        """
        Получить список моделей из пресета
        
        Args:
            key: Ключ пресета
        
        Returns:
            Список имен моделей
        """
        return self.dwm_presets.get(key, [])

    def batch_download(self, keys: List[str], progress: gr.Progress = gr.Progress()) -> None:
        """
        Пакетная загрузка моделей
        
        Args:
            keys: Список ключей моделей
            progress: Прогресс
        """
        if keys:
            total: int = len(keys)
            for i, key in enumerate(keys, start=1):
                progress(i / total, desc=f"{_i18n('model')} {i}/{total}")
                print(f"{_i18n('model')} {i}/{total}")
                if key in self.get_mn():
                    self.install_model(key)
                else:
                    print(_i18n("msg_model_not_exists", model=key))
                    gr.Warning(message="", title=_i18n("msg_model_not_exists", model=key))
        print(_i18n("msg_download_complete"))
        gr.Warning(message="", title=_i18n("msg_download_complete"))
    
    def delete_models_cache(self) -> None:
        """Удалить кэш всех моделей"""
        shutil.rmtree(self.models_cache_dir, ignore_errors=True)
        os.makedirs(self.models_cache_dir, exist_ok=True)
        print(_i18n("msg_cache_cleared"))
        gr.Warning(message="", title=_i18n("msg_cache_cleared"))


class SeparatorGradio(GradioHelper, DownloadModelManager):
    """Класс для Gradio интерфейса разделителя"""
    
    def __init__(self) -> None:
        super().__init__()
        self.input_files: List[str] = []
        self.input_base_dir: str = os.path.join(user_directory.path, "input")
        self.output_base_dir: str = os.path.join(user_directory.path, "output", "mvsepless")
        self.inputs_json_path: str = os.path.join(self.input_base_dir, "inputs.json")
        self.history: History = History()
        self.load_from_file()

    def _write_file(self) -> None:
        """Записывает текущее состояние в файл"""
        try:
            with open(self.inputs_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.input_files, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"{_i18n('error_writing_file')}: {e}")

    def _save_to_file(func):
        """Декоратор для автоматического сохранения после вызова метода"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self._write_file()
            return result
        return wrapper

    def load_from_file(self) -> None:
        """Загрузить историю из файла"""
        if os.path.exists(self.inputs_json_path):
            with open(self.inputs_json_path, 'r', encoding='utf-8') as f:
                self.input_files = json.load(f)

    @_save_to_file
    def clean(self) -> None:
        """Очистить список входных файлов"""
        self.input_files = []

    @_save_to_file
    def upload_files(self, input_files: List[str], copy: bool = False) -> List[str]:
        """
        Загрузить файлы в пользовательскую директорию
        
        Args:
            input_files: Список путей к файлам
            copy: Копировать вместо перемещения
        
        Returns:
            Список путей к загруженным файлам
        """
        if input_files:
            input_dir: str = os.path.join(
                self.input_base_dir, 
                datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
            )
            os.makedirs(input_dir, exist_ok=True)
            
            valid_files: List[str] = [file for file in input_files if check(file)]
            valid_files_moved: List[str] = []
            
            if valid_files:
                for file in valid_files:
                    basename: str = os.path.basename(file)
                    output_path: str = os.path.join(input_dir, basename)
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
        input_files: Optional[List[str]] = None,
        model_name: str = "Mel-Band-Roformer_Vocals_kimberley_jensen",
        ext_inst: bool = True,
        output_format: str = "mp3",
        output_bitrate: str = "320k",
        template: str = "NAME_(STEM)_MODEL",
        selected_stems: Optional[List[str]] = None,
        vr_aggr: int = 5,
        vr_post_process: bool = False,
        vr_high_end_process: bool = False,
        mdx_denoise: bool = False,
        use_spec_invert: bool = False,
        econom_mode: Optional[bool] = None,
        chunk_duration: float = 300,
        progress: gr.Progress = gr.Progress(track_tqdm=True),
    ) -> List:
        """
        Пакетное разделение аудио
        
        Args:
            input_files: Список входных файлов
            model_name: Имя модели
            ext_inst: Извлечь инструментал
            output_format: Формат вывода
            output_bitrate: Битрейт
            template: Шаблон имени
            selected_stems: Выбранные стемы
            vr_aggr: Агрессивность для VR
            vr_post_process: Постобработка для VR
            vr_high_end_process: Обработка высоких частот для VR
            mdx_denoise: Шумоподавление для MDX
            use_spec_invert: Использовать инверсию спектрограммы
            econom_mode: Эконом-режим
            chunk_duration: Длительность чанка
            progress: Прогресс
        
        Returns:
            Результаты разделения
        """
        timestamp: str = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        self.chunk_duration = chunk_duration
        
        add_settings: Dict[str, Any] = {
            "mdx_denoise": mdx_denoise,
            "vr_aggr": vr_aggr,
            "vr_post_process": vr_post_process,
            "vr_high_end_process": vr_high_end_process,
            "add_single_sep_text_progress": None,
            "single_mode": False
        }
        
        if econom_mode is not None:
            add_settings["econom_mode"] = econom_mode
        
        results = self.separate(
                input=input_files,
                output_dir=os.path.join(self.output_base_dir, timestamp),
                model_name=model_name,
                ext_inst=ext_inst,
                output_format=output_format,
                output_bitrate=output_bitrate,
                template=template,
                selected_stems=selected_stems,
                add_settings=add_settings,
                use_spec_invert=use_spec_invert,
                progress=progress,
            )
        
        self.history.add(results, model_name, timestamp)
        return results

    def UI(
        self, 
        theme: gr.Theme, 
        add_app: bool = True, 
        plugins: bool = True, 
        add_vbach: bool = False,
        model_manager_add: bool = True
    ) -> gr.Blocks:
        """
        Создать пользовательский интерфейс
        
        Args:
            theme: Тема Gradio
            add_app: Добавить дополнительные приложения
            plugins: Включить плагины
            add_vbach: Включить Vbach
        
        Returns:
            Блоки интерфейса Gradio
        """
        with gr.Blocks(theme=theme, title=_i18n("app_title")) as MVSEPLESS_LITE_UI:
            if zerogpu_available:
                gr.Markdown(f"<h2><center>{_i18n('msg_zero_warning')}<center><h2>")
            else:
                if not cuda_available:
                    gr.Markdown(f"<h2><center>{_i18n('msg_cpu_warning')}<center><h2>")
            
            # Вкладка разделения
            with gr.Tab(_i18n("tab_separation")):
                default_model: List[str] = self.get_mn()
                
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            upload = gr.Files(show_label=False, type="filepath", interactive=True)
                            refresh_input_btn = gr.Button(_i18n("refresh_files"), variant="primary", interactive=True)
                            list_input_files = gr.Dropdown(
                                label=_i18n("select_input_files"),
                                choices=reversed(self.input_files),
                                value=[],
                                multiselect=True,
                                interactive=True,
                                filterable=False,
                                scale=15
                            )
                            
                            gr.on(
                                fn=lambda: gr.update(choices=reversed(self.input_files), value=[]), 
                                outputs=list_input_files, 
                                trigger_mode="once"
                            )
                            
                            refresh_input_btn.click(
                                lambda: gr.update(choices=reversed(self.input_files), value=[]), 
                                outputs=list_input_files
                            )
                            
                            @upload.upload(inputs=[upload], outputs=[list_input_files, upload])
                            def upload_files(input_files: List[str]) -> Tuple[gr.update, gr.update]:
                                files: List[str] = self.upload_files(input_files)
                                return (
                                    gr.update(choices=reversed(self.input_files), value=files),
                                    gr.update(value=[])
                                )
    
                    with gr.Column():
                        with gr.Group():
                            with gr.Row(equal_height=True):
                                model_name = gr.Dropdown(
                                    label=_i18n("model_name"), 
                                    choices=default_model, 
                                    value=default_model[0] if default_model else None, 
                                    interactive=True, 
                                    scale=9
                                )
                                model_name_refresh_btn = gr.Button("🔄", size="lg", scale=2, interactive=True, min_width=50)

                            show_only_downloaded_models = gr.Checkbox(
                                label=_i18n("show_only_downloaded"), 
                                value=False, 
                                interactive=True
                            )
                            
                            @model_name_refresh_btn.click(inputs=[model_name, show_only_downloaded_models], outputs=model_name)
                            def refresh_model_fn(name: str, only_downloaded: bool) -> gr.update:
                                models: List[str] = []
                                if only_downloaded:
                                    models = self.get_mn_dwloaded()
                                else:
                                    models = self.get_mn()

                                first_value: Optional[str] = models[0] if models else None
                                value: Optional[str] = name if name in models else first_value
                                return gr.update(choices=models, value=value)

                            @show_only_downloaded_models.change(
                                inputs=[model_name, show_only_downloaded_models], 
                                outputs=model_name, 
                                trigger_mode="once"
                            )
                            def refresh_model_fn2(name: str, only_downloaded: bool) -> gr.update:
                                models: List[str] = []
                                if only_downloaded:
                                    models = self.get_mn_dwloaded()
                                else:
                                    models = self.get_mn()

                                first_value: Optional[str] = models[0] if models else None
                                value: Optional[str] = name if name in models else first_value
                                return gr.update(choices=models, value=value)

                            extract_instrumental = gr.Checkbox(
                                label=_i18n("extract_instrumental"), 
                                value=False, 
                                interactive=True, 
                                visible=False
                            )
                            
                            stems = gr.CheckboxGroup(
                                label=_i18n("select_stems"),
                                choices=self.get_stems(default_model[0]) if default_model else [],
                                value=[],
                                interactive=True, 
                                scale=8
                            )
                            
                            with gr.Accordion(label=_i18n("additional_settings"), open=False):
                                with gr.Group():
                                    gr.Markdown(f"<h4>{_i18n('vr_settings')}</h4>", container=True)
                                    vr_aggr = gr.Slider(
                                        label=_i18n("vr_aggression"),
                                        minimum=0,
                                        maximum=100,
                                        value=5,
                                        step=1,
                                        interactive=True,
                                    )
                                    vr_enable_post_process = gr.Checkbox(
                                        label=_i18n("vr_post_process"), 
                                        value=False, 
                                        interactive=True
                                    )
                                    vr_enable_high_end_process = gr.Checkbox(
                                        label=_i18n("vr_high_end_process"), 
                                        value=False, 
                                        interactive=True
                                    )
                                    
                                    gr.Markdown(f"<h4>{_i18n('mdx_settings')}</h4>", container=True)
                                    mdx_denoise = gr.Checkbox(
                                        label=_i18n("mdx_denoise"),
                                        value=False,
                                        interactive=True,
                                    )
                                    
                                    gr.Markdown(f"<h4>{_i18n('invert_settings')}</h4>", container=True)
                                    use_spec_for_extract_instrumental = gr.Checkbox(
                                        label=_i18n("use_spectrogram_invert"), 
                                        value=False, 
                                        interactive=True
                                    )
                                    
                                    gr.Markdown(f"<h4>{_i18n('economy_settings')}</h4>", container=True)
                                    econom_mode = gr.Checkbox(
                                        label=_i18n("economy_mode"), 
                                        value=False, 
                                        interactive=True
                                    )
                                    chunk_dur_slider = gr.Slider(
                                        label=_i18n("chunk_duration"),
                                        minimum=1,
                                        maximum=10,
                                        value=5,
                                        step=0.1,
                                        interactive=True,
                                    )

                            @model_name.change(
                                inputs=[model_name], 
                                outputs=[extract_instrumental, stems]
                            )
                            def update_model_name(model_name: str) -> Tuple[gr.update, gr.update]:
                                stems_list: List[str] = self.get_stems(model_name)
                                return (
                                    gr.update(visible=len(stems_list) > 2),
                                    gr.update(choices=stems_list, value=[], interactive=True)
                                )
                            
                            with gr.Row():
                                output_format = gr.Dropdown(
                                    label=_i18n("output_format"),
                                    interactive=True,
                                    choices=output_formats,
                                    value="mp3",
                                    filterable=False,
                                )
                                output_bitrate = gr.Slider(
                                    label=_i18n("output_bitrate"),
                                    minimum=64,
                                    maximum=512,
                                    step=32,
                                    value=320,
                                    interactive=True,
                                )
                                
                                output_format.change(
                                    lambda x: gr.update(visible=(x not in ["wav", "flac", "aiff"])),
                                    inputs=output_format,
                                    outputs=output_bitrate,
                                )
                                
                            template = gr.Textbox(
                                label=_i18n("filename_template"),
                                interactive=True,
                                value="NAME_(STEM)_MODEL",
                                info=_i18n("template_info"),
                            )
                            
                            sep_state = gr.Textbox(
                                label=_i18n("separation_status"),
                                interactive=False,
                                value="",
                                visible=False,
                            )
                            
                            status = gr.Textbox(
                                container=False, 
                                lines=4, 
                                interactive=False, 
                                max_lines=4, 
                                visible=False
                            )
                            
                            separate_btn = gr.Button(_i18n("separate_btn"), variant="primary", interactive=True).click(lambda: gr.update(visible=True), outputs=status)
                            
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
                                    econom_mode,
                                    chunk_dur_slider
                                ],
                                outputs=[sep_state, status],
                                show_progress="full",
                                queue=True
                            )
                            def wrap(
                                input_files: List[str],
                                model_name: str,
                                ext_inst: bool,
                                output_format: str,
                                output_bitrate: int,
                                template: str,
                                stems: List[str],
                                mdx_denoise: bool,
                                vr_aggr: int,
                                vr_pp: bool,
                                vr_hip: bool,
                                u_spec: bool,
                                ec_mode: bool,
                                ch_dur: float,
                                progress: gr.Progress = gr.Progress(track_tqdm=True),
                            ) -> Tuple[gr.update, gr.update]:
                                results = self._separate_batch(
                                    input_files,
                                    model_name,
                                    ext_inst,
                                    output_format,
                                    f"{int(output_bitrate)}k",
                                    template,
                                    stems,
                                    vr_aggr,
                                    vr_pp,
                                    vr_hip,
                                    mdx_denoise,
                                    u_spec,
                                    ec_mode,
                                    ch_dur * 60,
                                    progress=progress,
                                )
                                return gr.update(value=str(results)), gr.update(visible=False)

                with gr.Column(variant="panel"):
                    gr.Markdown(f"<center><h3>{_i18n('results')}</h3></center>")

                    with gr.Group():
                        with gr.Row(equal_height=True):
                            list_separations = gr.Dropdown(
                                label=_i18n("select_separation_results"),
                                choices=[],
                                value=None,
                                interactive=True, 
                                scale=14
                            )
                            
                            list_separations.change(
                                lambda x: gr.update(value=str(self.history.get(x))), 
                                inputs=[list_separations], 
                                outputs=[sep_state], 
                                trigger_mode="once"
                            )
                            
                            refresh_separations_btn = gr.Button(_i18n("refresh"), scale=2, interactive=True)
                            refresh_separations_btn.click(
                                lambda: self.return_list(self.history.get_list(), none=True), 
                                outputs=[list_separations]
                            )
                            
                            gr.on(
                                fn=lambda: self.return_list(self.history.get_list(), none=True), 
                                outputs=[list_separations]
                            )

                    @gr.render(inputs=[sep_state], triggers=[sep_state.change])
                    def players(state: str) -> None:
                        if state:
                            try:
                                state_loaded = ast.literal_eval(state)
                                if state_loaded:
                                    archive_stems = self.create_archive_advanced(
                                        state_loaded,
                                        os.path.join(
                                            tempfile.tempdir,
                                            f"mvsepless_output_{datetime.now(tz).strftime('%Y%m%d_%H%M%S')}.zip",
                                        ),
                                    )
                                    for basename, stems_list in state_loaded:
                                        with gr.Group():
                                            gr.Markdown(f"<h4><center>{basename}</center></h4>")
                                            for stem_name, stem_path in stems_list:
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
                                                        _i18n("reuse_btn"), 
                                                        variant="secondary"
                                                    )

                                                    @reuse_btn.click(
                                                        inputs=[output_stem],
                                                        outputs=list_input_files,
                                                    )
                                                    def reuse_fn(stem_audio: str) -> gr.update:
                                                        files = self.upload_files([stem_audio], copy=True)
                                                        return gr.update(
                                                            choices=reversed(self.input_files), 
                                                            value=files
                                                        )

                                    gr.DownloadButton(
                                        label=_i18n("download_as_zip"), 
                                        value=archive_stems, 
                                        interactive=True
                                    )
                            except:
                                pass

            # Вкладка загрузки аудио
            with gr.Tab(_i18n("tab_download_audio")):
                with gr.Tab(_i18n("tab_from_internet")):
                    with gr.Group():
                        input_url = gr.Textbox(
                            label=_i18n("url_input"), 
                            interactive=True
                        )
                        with gr.Row(equal_height=True):
                            inputs_url_format = gr.Dropdown(
                                label=_i18n("output_format"),
                                interactive=True,
                                choices=output_formats,
                                value="mp3",
                                filterable=False,
                            )
                            inputs_url_bitrate = gr.Slider(
                                label=_i18n("output_bitrate"),
                                minimum=32,
                                maximum=512,
                                step=32,
                                value=320,
                                interactive=True,
                            )
                            
                            inputs_url_format.change(
                                lambda x: gr.update(visible=(x not in ["wav", "flac", "aiff"])),
                                inputs=inputs_url_format,
                                outputs=inputs_url_bitrate,
                            )

                        with gr.Row(equal_height=True):
                            inputs_url_cookie = gr.UploadButton(
                                label=_i18n("cookie_file"),
                                interactive=True,
                                type="filepath",
                                file_count="single",
                                file_types=[".txt", ".cookies"],
                                variant="secondary",
                            )
                            add_inputs_url_btn = gr.Button(
                                _i18n("add_file_btn"), 
                                variant="primary"
                            )
                            
                            @add_inputs_url_btn.click(
                                inputs=[
                                    input_url,
                                    inputs_url_format,
                                    inputs_url_bitrate,
                                    inputs_url_cookie,
                                ]
                            )
                            def add_inputs_from_url_fn(
                                input_u: str, 
                                fmt: str, 
                                br: int, 
                                cookie: Optional[str]
                            ) -> None:
                                if input_u:
                                    downloaded_file: Optional[str] = dw_yt_dlp(
                                        url=input_u,
                                        output_format=fmt,
                                        output_bitrate=str(int(br)),
                                        cookie=cookie,
                                    )
                                    if downloaded_file and os.path.exists(downloaded_file):
                                        if check(downloaded_file):
                                            self.upload_files([downloaded_file])
                                            gr.Warning(title=_i18n("msg_file_uploaded"), message="")

                with gr.Tab(_i18n("tab_from_device")):
                    with gr.Row():
                        with gr.Group():
                            gr.Markdown(f"<h3><center>{_i18n('upload_from_directory')}</h3></center>")
                            add_inputs_from_device_directory = gr.File(
                                show_label=False, 
                                label=_i18n("upload_from_directory"), 
                                interactive=True, 
                                file_count="directory"
                            )
                            
                        with gr.Group():
                            gr.Markdown(f"<h3><center>{_i18n('upload_from_files')}</h3></center>")
                            add_inputs_from_device_files = gr.File(
                                show_label=False, 
                                label=_i18n("upload_from_files"), 
                                interactive=True, 
                                file_count="multiple"
                            )
                            
                        with gr.Group():
                            gr.Markdown(f"<h3><center>{_i18n('upload_from_zip')}</h3></center>")
                            add_inputs_from_device_zip = gr.File(
                                show_label=False, 
                                label=_i18n("upload_from_zip"), 
                                interactive=True, 
                                file_count="single", 
                                file_types=[".zip"]
                            )
                            
                        @add_inputs_from_device_directory.upload(
                            inputs=[add_inputs_from_device_directory], 
                            outputs=[add_inputs_from_device_directory]
                        )
                        def upload_from_directory(file_list: List[str]) -> gr.update:
                            self.upload_files(file_list)
                            gr.Warning(title=_i18n("msg_files_uploaded"), message="")
                            return gr.update(value=[])
                            
                        @add_inputs_from_device_files.upload(
                            inputs=[add_inputs_from_device_files], 
                            outputs=[add_inputs_from_device_files]
                        )
                        def upload_from_files(file_list: List[str]) -> gr.update:
                            self.upload_files(file_list)
                            gr.Warning(title=_i18n("msg_files_uploaded"), message="")
                            return gr.update(value=[])
                            
                        @add_inputs_from_device_zip.upload(
                            inputs=[add_inputs_from_device_zip], 
                            outputs=[add_inputs_from_device_zip]
                        )
                        def upload_from_zip(zip_file: str) -> gr.update:
                            with tempfile.TemporaryDirectory() as tmp_zip:
                                _files = self.extract_zip(zip_file, tmp_zip)
                                self.upload_files(_files)
                                gr.Warning(title=_i18n("msg_files_uploaded"), message="")
                            return gr.update(value=None)

                    with gr.Group():
                        with gr.Row(equal_height=True):
                            add_inputs_from_path = gr.Textbox(
                                label=_i18n("upload_from_path"), 
                                interactive=True, 
                                value="", 
                                scale=15
                            )
                            add_inputs_from_path_btn = gr.Button(
                                _i18n("upload_btn"), 
                                variant="primary", 
                                interactive=True, 
                                scale=3
                            )
                            
                            @add_inputs_from_path_btn.click(
                                inputs=add_inputs_from_path, 
                                outputs=add_inputs_from_path
                            )
                            def upload_from_path(path: str) -> gr.update:
                                self.upload_files([path])
                                gr.Warning(title=_i18n("msg_file_uploaded"), message="")
                                return gr.update(value="")

            # Вкладка менеджера моделей
            if model_manager_add:
                with gr.Tab(_i18n("tab_model_manager")):
                    with gr.Tab(_i18n("tab_download_model")):
                        with gr.Group():
                            select_dwm_preset = gr.Dropdown(
                                label=_i18n("select_preset"),
                                interactive=True,
                                choices=list(self.dwm_presets.keys()),
                                value=None,
                            )
                            select_dwm_names = gr.Dropdown(
                                label=_i18n("select_models"),
                                interactive=True,
                                choices=default_model,
                                value=[],
                                multiselect=True
                            )
                            dwm_status = gr.Textbox(
                                container=False, 
                                lines=3, 
                                interactive=False, 
                                max_lines=3, 
                                visible=False
                            )
                            download_dwm_button = gr.Button(_i18n("download_btn"))
                            
                            select_dwm_preset.change(
                                lambda x: gr.update(value=self.parse_models_from_dwm_preset(x)),
                                inputs=select_dwm_preset, 
                                outputs=select_dwm_names, 
                                trigger_mode="once"
                            )
                            
                            download_dwm_button.click(
                                lambda: gr.update(visible=True), 
                                outputs=dwm_status
                            ).then(
                                lambda x: (self.batch_download(x), gr.update(visible=False)),
                                inputs=select_dwm_names, 
                                outputs=[gr.State(None), dwm_status]
                            )
                            
                    with gr.Tab(_i18n("tab_delete_models")):
                        gr.Markdown(f"<h3><center>{_i18n('delete_all_warning')}</center></h3>")
                        delete_models_cache_btn = gr.Button(_i18n("delete_all_btn"), variant="stop")
                        delete_models_cache_btn.click(self.delete_models_cache, inputs=None, outputs=None)

            # Импорт дополнительных модулей
            from additional_app import AutoEnsembless, ManualEnsembless, PluginManager, Inverter_UI, AudioApp, CustomSeparator
            
            if add_app:
                with gr.Tab(_i18n("tab_custom_separation")):
                    _custom_sep = CustomSeparator(
                        self.input_files, 
                        self.upload_files, 
                        user_directory, 
                        device=self.device,
                        history=self.history
                    )
                    _custom_sep.UI()
                with gr.Tab(_i18n("tab_audio_processing")):
                    _audio_app = AudioApp(user_directory)
                    _audio_app.UI()
                    
                with gr.Tab(_i18n("tab_ensemble")):
                    with gr.Tab(_i18n("tab_auto_ensemble")):
                        _auto_ensembless = AutoEnsembless(
                            self.input_files, 
                            self.upload_files, 
                            user_directory, 
                            device=self.device
                        )
                        _auto_ensembless.UI()
                        
                    with gr.Tab(_i18n("tab_manual_ensemble")):
                        ManualEnsembless(user_directory).UI()

                with gr.Tab(_i18n("tab_subtraction")):
                    Inverter_UI().UI()

            if add_vbach:
                from vbach import Vbach, vbach_inference, model_manager as voice_model_manager
                
                with gr.Tab(_i18n("tab_conversion")):
                    _vbach = Vbach(user_directory, device=self.device)
                    _vbach.upload_files = self.upload_files  # type: ignore
                    _vbach.input_files = self.input_files
                    _vbach.UI()
                    
                if add_app:
                    with gr.Tab(_i18n("tab_cover_generation")):
                        from vbachgen import VbachGen
                        _vbach_gen = VbachGen(
                            voice_model_manager, 
                            self.input_files, 
                            self.upload_files, 
                            user_directory, 
                            vbach_inference, 
                            device=self.device
                        )
                        _vbach_gen.UI()

            if plugins:
                with gr.Tab(_i18n("tab_plugins")):
                    PluginManager().UI()
            
            # Вкладка устройства
            with gr.Tab(_i18n("tab_device")):
                with gr.Group():
                    device_radio = gr.CheckboxGroup(
                        label=_i18n("cuda_device_ids"), 
                        choices=all_ids, 
                        interactive=True
                    )
                    pref_cuda = gr.Checkbox(
                        label=_i18n("prefer_cuda"), 
                        value=True
                    )
                    current_device = gr.Textbox(
                        label=_i18n("current_device"), 
                        value=self.device
                    )
                    
                gr.on(fn=lambda: self.device, outputs=[current_device])
                
                def show_device(device_ids: List[str], prefer_gpu: bool) -> str:
                    _device: str = set_device(device_ids, prefer_gpu=prefer_gpu)
                    self.device = _device
                    if add_app:
                        _auto_ensembless.device = _device
                        if add_vbach:
                            _vbach_gen.device = _device
                    if add_vbach:
                        _vbach.device = _device
                    print(f"{_i18n('selected_device')}: {self.device}")
                    return self.device
                    
                device_radio.change(
                    show_device, 
                    inputs=[device_radio, pref_cuda], 
                    outputs=current_device, 
                    trigger_mode="once"
                )
                pref_cuda.change(
                    show_device, 
                    inputs=[device_radio, pref_cuda], 
                    outputs=current_device, 
                    trigger_mode="once"
                )

        return MVSEPLESS_LITE_UI