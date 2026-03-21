import os
import sys
import ast
import shutil
import importlib.util
import gradio as gr
import pandas as pd
import subprocess
import json
import yaml
def tuple_constructor(loader: yaml.Loader, node: yaml.Node) -> tuple:
    values = loader.construct_sequence(node)
    return tuple(values)
yaml.SafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", tuple_constructor
)
import time
from datetime import datetime
import tempfile
from functools import wraps
from typing import List, Tuple, Optional, Dict, Any, Callable, Union
from pathlib import Path
from separator import Separator, script_dir
from gradio_helper import GradioHelper, tz, dw_file
from audio import output_formats, check, read, get_sr, get_duration_from_array, multiread, write, trim, concatenate
from i18n import _i18n


custom_model_types = [
    "mel_band_roformer",
    "bs_roformer",
    "mdx23c",
    "scnet",
    "scnet_masked",
    "scnet_tran",
    "htdemucs",
    "bandit",
    "bandit_v2"
]

def trim_audio(input_path: str, start: float = 0, end: float = -1, output_path: str = "./trimmed.mp3") -> Optional[str]:
    """
    Обрезать аудиофайл
    
    Args:
        input_path: Путь к входному файлу
        start: Начало обрезки в секундах
        end: Конец обрезки в секундах (-1 до конца)
        output_path: Путь для сохранения результата
    
    Returns:
        Путь к выходному файлу или None
    """
    y, sr = read(input_path)
    end_sample: int = int(end * sr) if end != -1 else -1
    y = trim(y, int(start * sr), end_sample)
    return write(output_path, y, sr)


def concat_audio(files: List[str], output_path: str) -> Optional[str]:
    """
    Склеить несколько аудиофайлов в один
    
    Args:
        files: Список путей к аудиофайлам
        output_path: Путь для сохранения результата
    
    Returns:
        Путь к выходному файлу или None
    """
    # Фильтруем файлы с помощью функции check
    valid_files: List[str] = [f for f in files if check(f)]
    
    if not valid_files:
        print(_i18n("msg_no_valid_audio"))
        return None

    print(_i18n("msg_processing_files", count=len(valid_files)))
    arrays, srs = multiread(valid_files)
    full_audio, max_sr = concatenate(arrays, srs, dtype="float32")
    return write(output_path, full_audio, max_sr)

class TempConfig:
    def __init__(self):
        self.data = {}
        self.stems = []
        self.target_instrument = None

    def load(self, url):
        if url != "" and url:
            _temp, path = tempfile.mkstemp(suffix=".yaml")
            dw_file(url, path)
            with open(path, "r", encoding="utf-8") as f:
                self.data = yaml.load(f, yaml.SafeLoader)

    def get_stems(self):
        if "training" in self.data:
            self.stems, self.target_instrument = self.data["training"]["instruments"], self.data["training"]["target_instrument"]
        else:
            self.stems, self.target_instrument = [], None
        
    def clear(self):
        self.data = {}
        self.stems = []
        self.target_instrument = None

class CustomSeparator(Separator, GradioHelper):
    def __init__(
        self, 
        input_files: List[str], 
        upload_files_func: Callable, 
        user_directory: Any, 
        device: str,
        history: Any
    ) -> None:
        super().__init__()
        self.input_files: List[str] = input_files
        self.upload_files: Callable = upload_files_func
        self.user_directory: Any = user_directory
        self.device: str = device
        self.input_base_dir: str = os.path.join(user_directory.path, "input")
        self.output_base_dir: str = os.path.join(user_directory.path, "output", "mvsepless")
        self.inputs_json_path: str = os.path.join(self.input_base_dir, "inputs.json")
        self.history: Any = history
        self.models_info: Dict = {}
        self.models_info_path: str = os.path.join(script_dir, "custom_models.json")
        self.models_cache_dir = os.path.join(script_dir, "mvsepless_custom_models_cache")
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
            dir_path: str = os.path.dirname(self.models_info_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(self.models_info_path, 'w', encoding='utf-8') as f:
                json.dump(self.models_info, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"{_i18n('error_writing_file')}: {e}")
    
    @_save_to_file
    def add(
        self, 
        model_name: str, 
        model_type: str,
        category: str | None,
        m_id: str | int,
        fullname: str,
        stems: str | List,
        target_instrument: str | None,
        ckpt_url: str,
        conf_url: str
    ) -> None:
        """
        Добавить модель
        
        Args:
            model_name: Имя модели
            model_type: Тип модели
            category: Категория
            m_id: ID модели
            fullname: Полное имя модели
            stems: Список стемов
            target_instrument: Целевой инструмент
            ckpt_url: Ссылка на чекпоинт (*.ckpt, *.pt, *.pth, *.chpt)
            conf_url: Ссылка на конфиг (*.yaml)
        """
        if not ckpt_url or not conf_url:
            gr.Warning(title=_i18n("ckpt_conf_required"), message="")
            raise ValueError(_i18n("ckpt_conf_required"))
        
        if isinstance(stems, str):
            list_stems = stems.split(',')
        else:
            list_stems = stems
        
        if target_instrument != "" and target_instrument:
            target_instrument_value = target_instrument
        else:
            target_instrument_value = None

        if category != "" and category:
            category_value = category
        else:
            category_value = ""

        if fullname != "" and fullname:
            fullname_value = fullname
        else:
            fullname_value = ""

        self.models_info[model_name] = {
            "model_type": model_type,
            "category": category_value,
            "id": int(m_id),
            "full_name": fullname_value,
            "stems": list_stems,
            "target_instrument": target_instrument_value,
            "checkpoint_url": ckpt_url,
            "config_url": conf_url
        }
        gr.Warning(title=_i18n("add_model_successed", model=model_name), message="")

    @_save_to_file
    def remove(self, model_name: str) -> None:
        if model_name in self.get_mn():
            del self.models_info[model_name]
            gr.Warning(title=_i18n("model_deleted", model=model_name), message="")
        else:
            gr.Warning(title=_i18n("model_not_found", model=model_name), message="")


    @_save_to_file
    def clear(self) -> None:
        """Очистить словарь моделей"""
        self.models_info = {}
    
    def load_from_file(self) -> None:
        """Загрузить словарь моделей из файла"""
        if os.path.exists(self.models_info_path):
            with open(self.models_info_path, 'r', encoding='utf-8') as f:
                self.models_info = json.load(f)

    def get_mn(self) -> List[str]:
        """
        Получить список всех доступных моделей
        
        Returns:
            Список имен моделей
        """
        return [mn for mn in self.models_info]

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

    def _separate_batch(
        self,
        input_files: Optional[List[str]] = None,
        model_name: str = "Mel-Band-Roformer_Vocals_kimberley_jensen",
        ext_inst: bool = True,
        output_format: str = "mp3",
        output_bitrate: str = "320k",
        template: str = "NAME_(STEM)_MODEL",
        selected_stems: Optional[List[str]] = None,
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

    def UI(self):
        default_model: List[str] = self.get_mn()
        with gr.Tab(_i18n("inference")):
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
                        
                        @model_name_refresh_btn.click(inputs=[model_name], outputs=model_name)
                        def refresh_model_fn(name: str) -> gr.update:
                            models: List[str] = []
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
        with gr.Tab(_i18n("tab_model_manager")):
            temp_config = TempConfig()
            with gr.Row():
                with gr.Group():
                    with gr.Column(variant="panel"):
                        gr.Markdown(f'<h3><center>{_i18n("add_model")}</center></h3>')
                    add_model_name = gr.Textbox(
                        label=_i18n("model_name"), 
                        value="custom_model",
                        interactive=True 
                    )
                    add_model_type = gr.Dropdown(
                        label=_i18n("model_type"),
                        choices=custom_model_types,
                        value=custom_model_types[0],
                        interactive=True 
                    )
                    add_model_id = gr.Number(
                        label=_i18n("model_id"),
                        minimum=0,
                        value=0,
                        interactive=True 
                    )
                    @gr.render(inputs=[add_model_id])
                    def show_info(m_id):
                        if m_id in [self.get_id(model) for model in self.get_mn()]:
                            gr.Markdown(container=True, value=f'<h3><center>{_i18n("model_id_is_already_exist")}</center></h3>')
                    add_category = gr.Textbox(
                        label=_i18n("category_optional"), 
                        value=_i18n("custom"),
                        interactive=True 
                    )
                    add_fname = gr.Textbox(
                        label=_i18n("full_name_optional"), 
                        value="",
                        interactive=True 
                    )
                    add_ckpt = gr.Textbox(
                        label=_i18n("ckpt_url"), 
                        value="",
                        interactive=True 
                    )
                    add_conf = gr.Textbox(
                        label=_i18n("conf_url"), 
                        value="",
                        interactive=True
                    )
                    @gr.render(inputs=[add_conf])
                    def show_info(url):
                        temp_config.load(url)
                        temp_config.get_stems()
                        if temp_config.stems:
                            gr.Radio(label=_i18n("stems"), value=None, choices=temp_config.stems, interactive=False)
                        if temp_config.target_instrument:
                            gr.Textbox(label=_i18n("target_instrument"), value=str(temp_config.target_instrument), interactive=False)
                    add_model_btn = gr.Button(_i18n("add"), interactive=True)
                with gr.Column():
                    with gr.Group():
                        with gr.Column(variant="panel"):
                            gr.Markdown(f'<h3><center>{_i18n("delete_model")}</center></h3>')
                        with gr.Row(equal_height=True):
                            del_model_name = gr.Dropdown(
                                label=_i18n("model_name"), 
                                choices=[], 
                                value=None, 
                                interactive=True, 
                                scale=9
                            )
                            del_model_name_refresh_btn = gr.Button("🔄", size="lg", scale=2, interactive=True, min_width=50)
                            
                            @del_model_name_refresh_btn.click(inputs=[del_model_name], outputs=del_model_name)
                            def refresh_model_fn(name: str) -> gr.update:
                                models: List[str] = []
                                models = self.get_mn()
                                first_value: Optional[str] = models[0] if models else None
                                value: Optional[str] = name if name in models else first_value
                                return gr.update(choices=models, value=value)
                        del_model_btn = gr.Button(
                            _i18n("delete"), 
                            variant="stop", 
                            interactive=True
                        )
                    with gr.Group():
                        with gr.Column(variant="panel"):
                            gr.Markdown(f'<h3><center>{_i18n("download_custom_model")}</center></h3>')
                        with gr.Row(equal_height=True):
                            select_dwm_names = gr.Dropdown(
                                label=_i18n("select_models"),
                                interactive=True,
                                choices=[],
                                value=[],
                                multiselect=True,
                                scale=9
                            )
                            dwm_model_name_refresh_btn = gr.Button("🔄", size="lg", scale=2, interactive=True, min_width=50)
                            
                            @dwm_model_name_refresh_btn.click(inputs=[select_dwm_names], outputs=select_dwm_names)
                            def refresh_model_fn(names: list) -> gr.update:
                                models: List[str] = []
                                models = self.get_mn()
                                first_value: Optional[str] = models[0] if models else None
                                value: Optional[List] = names if names in models else [first_value]
                                return gr.update(choices=models, value=value)
                        dwm_status = gr.Textbox(
                            container=False, 
                            lines=3, 
                            interactive=False, 
                            max_lines=3, 
                            visible=False
                        )
                        download_dwm_button = gr.Button(_i18n("download_btn"))
                        
                        download_dwm_button.click(
                            lambda: gr.update(visible=True), 
                            outputs=dwm_status
                        ).then(
                            lambda x: (self.batch_download(x), gr.update(visible=False)),
                            inputs=select_dwm_names, 
                            outputs=[gr.State(None), dwm_status]
                        )
                    with gr.Group():
                        with gr.Column(variant="panel"):
                            gr.Markdown(f'<h3><center>{_i18n("delete_all_custom_models")}</center></h3>')
                        with gr.Row(equal_height=True):
                            delete_models_cache_btn = gr.Button(_i18n("delete_all_custom_cache_btn"), variant="stop")
                            delete_models_cache_btn.click(self.delete_models_cache, inputs=None, outputs=None)
                            delete_models_info_btn = gr.Button(_i18n("delete_all_custom_models_info_btn"), variant="huggingface")
                    with gr.Group():
                        with gr.Column(variant="panel"):
                            gr.Markdown(f'<h3><center>{_i18n("save_models_info")}</center></h3>')
                        with gr.Row(equal_height=True):
                            export_btn = gr.DownloadButton(
                                _i18n("export"), 
                                variant="secondary",
                                value=self.models_info_path,
                                scale=3,
                                interactive=True
                            )
                            gr.on(fn=lambda: gr.update(value=self.models_info_path), outputs=[export_btn])
                            @delete_models_info_btn.click(outputs=[export_btn, del_model_name, model_name, select_dwm_names])
                            def del_model_info_fn():
                                self.clear()
                                models = self.get_mn()
                                first_model = models[0] if models else None
                                first_model2 = [models[0]] if models else []
                                return gr.update(value=self.models_info_path), gr.update(choices=models, value=first_model), gr.update(choices=models, value=first_model), gr.update(choices=models, value=first_model2)
                            @del_model_btn.click(inputs=del_model_name, outputs=[export_btn, del_model_name, model_name, select_dwm_names])
                            def del_model_btn_fn(name: str):
                                self.remove(name)
                                models = self.get_mn()
                                first_model = models[0] if models else None
                                first_model2 = [models[0]] if models else []
                                return gr.update(value=self.models_info_path), gr.update(choices=models, value=first_model), gr.update(choices=models, value=first_model), gr.update(choices=models, value=first_model2)
                            @add_model_btn.click(inputs=[add_model_name, add_model_type, add_model_id, add_category, add_fname, add_ckpt, add_conf], outputs=[export_btn, del_model_name, model_name, select_dwm_names])
                            def add_model_bth_fn(mn, mt, add_model_id, c, fn, ckpt, conf):
                                self.add(mn, mt, c, add_model_id, fn, temp_config.stems, temp_config.target_instrument, ckpt, conf)
                                models = self.get_mn()
                                first_model = models[0] if models else None
                                first_model2 = [models[0]] if models else []
                                return gr.update(value=self.models_info_path), gr.update(choices=models, value=first_model), gr.update(choices=models, value=first_model), gr.update(choices=models, value=first_model2)
                            import_btn = gr.UploadButton(
                                _i18n("import"),
                                file_types=[".json"],
                                file_count="single",
                                scale=3,
                                interactive=True,
                            )
                            @import_btn.click(
                                inputs=import_btn,
                                outputs=[del_model_name, model_name, select_dwm_names]
                            )
                            def refresh_all_models(path) -> Tuple[gr.update, gr.update, gr.update]:
                                shutil.copy(path, self.models_info_path)
                                self.load_from_file()
                                models = self.get_mn()
                                first_model = models[0] if models else None
                                first_model2 = [models[0]] if models else []
                                return (
                                    gr.update(choices=models, value=first_model),
                                    gr.update(choices=models, value=first_model),
                                    gr.update(choices=models, value=first_model2),
                                )

        @gr.on(
            inputs=None, 
            outputs=[del_model_name, model_name, select_dwm_names]
        )
        def refresh_all_models() -> Tuple[gr.update, gr.update, gr.update]:
            models = self.get_mn()
            first_model = models[0] if models else None
            return (
                gr.update(choices=models, value=first_model),
                gr.update(choices=models, value=first_model),
                gr.update(choices=models, value=[first_model]),
            )

class AutoEnsembless(Separator, GradioHelper):
    """Класс для автоматического ансамбля"""
    
    def __init__(
        self, 
        input_files: List[str], 
        upload_files_func: Callable, 
        user_directory: Any, 
        device: str
    ) -> None:
        """
        Инициализация авто-ансамбля
        
        Args:
            input_files: Список входных файлов
            upload_files_func: Функция загрузки файлов
            user_directory: Пользовательская директория
            device: Устройство для вычислений
        """
        super().__init__()
        self.input_files: List[str] = input_files
        self.upload_files_func: Callable = upload_files_func
        self.user_directory: Any = user_directory
        self.device: str = device

    class ModelManager(Separator):
        """Менеджер моделей для ансамбля"""
        
        def __init__(self) -> None:
            super().__init__()
            self.data: List[List[Union[str, int, float]]] = []
            self.dir_presets: str = os.path.join(tempfile.gettempdir(), "presets")
            os.makedirs(self.dir_presets, exist_ok=True)

        def save(self, name: str) -> str:
            """
            Сохранить пресет
            
            Args:
                name: Имя пресета
            
            Returns:
                Путь к сохраненному файлу
            """
            if not name:
                name = "ensembless_preset"
            filepath: str = os.path.join(
                self.dir_presets,
                f"{self.namer.short(self.namer.sanitize(name), length=50)}.json",
            )
            with open(filepath, "w", encoding='utf-8') as f:
                json.dump(self.data, f, indent=4, ensure_ascii=False)
            return filepath

        def load(self, filepath: str) -> None:
            """
            Загрузить пресет
            
            Args:
                filepath: Путь к файлу пресета
            """
            with open(filepath, "r", encoding='utf-8') as f:
                ensemble_data_temp: List = json.load(f)
            self.data = []
            for mn, s_stem, i_stem, weight in ensemble_data_temp:
                model_names: List[str] = [str(model[0]) for model in self.data]
                if mn not in model_names:
                    self.data.append([mn, s_stem, i_stem, weight])

        def add(self, model_name: str, primary_stem: str, invert_stem: str, weight: float) -> None:
            """
            Добавить модель в ансамбль
            
            Args:
                model_name: Имя модели
                primary_stem: Основной стем
                invert_stem: Инверсный стем
                weight: Вес
            """
            model_names: List[str] = [str(model[0]) for model in self.data]
            if model_name not in model_names:
                if primary_stem and invert_stem:
                    self.data.append([model_name, primary_stem, invert_stem, weight])

        def replace(
            self, 
            model_name: str, 
            primary_stem: str, 
            invert_stem: str, 
            weight: float, 
            index: int = 1
        ) -> None:
            """
            Заменить модель в ансамбле
            
            Args:
                model_name: Имя модели
                primary_stem: Основной стем
                invert_stem: Инверсный стем
                weight: Вес
                index: Индекс для замены
            """
            if self.data:
                len_data: int = len(self.data)
                if index >= 1:
                    if index <= len_data:
                        self.data[index - 1] = [model_name, primary_stem, invert_stem, weight]
                elif index == 0:
                    self.data[0] = [model_name, primary_stem, invert_stem, weight]

        def remove(self, index: int = 1) -> None:
            """
            Удалить модель из ансамбля
            
            Args:
                index: Индекс для удаления
            """
            if self.data:
                len_data: int = len(self.data)
                if index >= 1:
                    if index <= len_data:
                        del self.data[index - 1]
                elif index == 0:
                    del self.data[0]

        def clear(self) -> None:
            """Очистить ансамбль"""
            self.data = []

        def get_df(self) -> pd.DataFrame:
            """
            Получить DataFrame с текущим состоянием ансамбля
            
            Returns:
                DataFrame с данными
            """
            if not self.data:
                columns: List[str] = ["#", _i18n("model_name"), _i18n("primary_stem"), _i18n("inversion_stem"), _i18n("weight")]
                return pd.DataFrame(columns=columns)

            data_rows: List[List] = []
            for i, model in enumerate(self.data):
                data_rows.append(
                    [
                        f"{i+1}",
                        str(model[0]),
                        str(model[1]),
                        str(model[2]),
                        float(model[3]),
                    ]
                )
            columns = ["#", _i18n("model_name"), _i18n("primary_stem"), _i18n("inversion_stem"), _i18n("weight")]
            return pd.DataFrame(data_rows, columns=columns)

    class History:
        """Класс для управления историей ансамблей"""
        
        def __init__(self, user_directory: Any) -> None:
            self.info: Dict[str, Dict] = {}
            self.path: str = os.path.join(user_directory.path, "history", "ensembless.json")
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
        def add(
            self, 
            input_file: str, 
            last_result: str, 
            last_result_wav: str, 
            last_inverted_result: Optional[str], 
            ensemble_sources: List[str], 
            method: str, 
            timestamp: str, 
            models: List
        ) -> None:
            """
            Добавить запись в историю
            
            Args:
                input_file: Входной файл
                last_result: Последний результат
                last_result_wav: WAV версия результата
                last_inverted_result: Инверсный результат
                ensemble_sources: Исходники ансамбля
                method: Метод
                timestamp: Временная метка
                models: Список моделей
            """
            self.info[f"{timestamp} / {method} / {len(models)}"] = {
                "input_file": input_file,
                "last_result": last_result,
                "last_result_wav": last_result_wav,
                "last_inverted_result": last_inverted_result,
                "source": ensemble_sources
            }
        
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
        
        def get(self, key: str) -> Dict:
            """
            Получить запись истории по ключу
            
            Args:
                key: Ключ записи
            
            Returns:
                Запись истории
            """
            return self.info.get(key, {})
        
        def load_from_file(self) -> None:
            """Загрузить историю из файла"""
            if os.path.exists(self.path):
                with open(self.path, 'r', encoding='utf-8') as f:
                    self.info = json.load(f)

    def UI(self) -> None:
        """Создать пользовательский интерфейс"""
        ensemble_model_manager = self.ModelManager()
        history = self.History(self.user_directory)

        def get_stems(model_name: str) -> List[str]:
            """
            Получить список стемов для модели
            
            Args:
                model_name: Имя модели
            
            Returns:
                Список стемов
            """
            stems: List[str] = []
            for stem in self.get_stems(model_name):
                stems.append(stem)

            if not self.get_tgt_inst(model_name):
                if set(stems) == {"bass", "drums", "other", "vocals"} or set(stems) == {
                    "bass",
                    "drums",
                    "other",
                    "vocals",
                    "piano",
                    "guitar",
                }:
                    stems.append("instrumental +")
                    stems.append("instrumental -")

            return stems

        def get_invert_stems(model_name: str, primary_stem: str) -> List[str]:
            """
            Получить список инверсных стемов
            
            Args:
                model_name: Имя модели
                primary_stem: Основной стем
            
            Returns:
                Список инверсных стемов
            """
            orig_stems: List[str] = []
            stems: List[str] = []
            for stem in self.get_stems(model_name):
                orig_stems.append(stem)

            for stem in orig_stems:
                if stem != primary_stem:
                    stems.append(stem)

            if not self.get_tgt_inst(model_name):
                if len(orig_stems) > 2:
                    if primary_stem not in ["instrumental +", "instrumental -"]:
                        stems.append("inverted +")
                        stems.append("inverted -")

            return stems

        available_models: List[str] = self.get_mn()
        default_model_name: str = available_models[0] if available_models else ""
        default_stems: List[str] = get_stems(default_model_name) if default_model_name else []
        default_invert_stems: List[str] = get_invert_stems(
            default_model_name, 
            default_stems[0] if default_stems else ""
        ) if default_model_name else []

        default_model: Dict[str, Any] = {
            "mn": available_models,
            "stem": default_stems,
            "invert_stem": default_invert_stems,
            "weight": 1,
        }

        gr.Markdown(f"<h3>{_i18n('ensemble_preset')}</h3>")
        with gr.Group():
            with gr.Row(equal_height=True):
                export_preset_name = gr.Textbox(
                    label=_i18n("preset_name"),
                    interactive=True,
                    value="ensembless_preset",
                    scale=9,
                )
                export_btn = gr.DownloadButton(
                    _i18n("export"), 
                    variant="secondary", 
                    scale=3, 
                    interactive=True
                )
                import_btn = gr.UploadButton(
                    _i18n("import"),
                    file_types=[".json"],
                    file_count="single",
                    scale=3,
                    interactive=True,
                )
                
        gr.Markdown(f"<h3>{_i18n('ensemble')}</h3>")
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Group():
                    model_name = gr.Dropdown(
                        label=_i18n("model_name"),
                        choices=default_model["mn"],
                        value=default_model["mn"][0] if default_model["mn"] else None,
                        interactive=True,
                        filterable=True,
                    )
                    primary_stem = gr.Dropdown(
                        label=_i18n("primary_stem"),
                        choices=default_model["stem"],
                        value=default_model["stem"][0] if default_model["stem"] else None,
                        interactive=True,
                        filterable=False,
                    )
                    secondary_stem = gr.Dropdown(
                        label=_i18n("inversion_stem"),
                        choices=default_model["invert_stem"],
                        value=default_model["invert_stem"][0] if default_model["invert_stem"] else None,
                        interactive=True,
                        filterable=False,
                    )
                    weight = gr.Slider(
                        label=_i18n("weight"),
                        minimum=0,
                        maximum=10,
                        step=0.01,
                        value=1,
                        interactive=True,
                    )

                    @model_name.change(
                        inputs=[model_name],
                        outputs=[primary_stem, secondary_stem],
                    )
                    def update_stems_after_model_change(model_name: str) -> Tuple[gr.update, gr.update]:
                        stems: List[str] = get_stems(model_name)
                        invert_stems: List[str] = get_invert_stems(model_name, stems[0]) if stems else []

                        new_s_stem: str = stems[0] if stems else ""
                        new_i_stem: str = invert_stems[0] if invert_stems else ""

                        return (
                            gr.update(choices=stems, value=new_s_stem),
                            gr.update(choices=invert_stems, value=new_i_stem),
                        )

                    @primary_stem.change(
                        inputs=[model_name, primary_stem],
                        outputs=[secondary_stem],
                    )
                    def update_invert_stems(model_name: str, primary_stem: str) -> gr.update:
                        stems: List[str] = get_invert_stems(model_name, primary_stem)
                        new_i_stem: str = stems[0] if stems else ""
                        return gr.update(choices=stems, value=new_i_stem)

                    model_add_button = gr.Button(_i18n("add_model_btn"), interactive=True)
                    
            with gr.Column(scale=10):
                df = gr.DataFrame(
                    value=ensemble_model_manager.get_df(),
                    headers=["#", _i18n("model_name"), _i18n("primary_stem"), _i18n("inversion_stem"), _i18n("weight")],
                    datatype=["number", "str", "str", "str", "number"],
                    interactive=False,
                )

                with gr.Group():
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            model_index = gr.Number(
                                label=_i18n("model_index"), 
                                value=1, 
                                interactive=True
                            )
                            model_clear_btn = gr.Button(
                                _i18n("clear_all_btn"), 
                                variant="stop", 
                                interactive=True
                            )
                        with gr.Column():
                            model_replace_btn = gr.Button(
                                _i18n("replace_model_btn"), 
                                variant="primary", 
                                interactive=True
                            )
                            model_delete_btn = gr.Button(
                                _i18n("delete_model_btn"), 
                                variant="stop", 
                                interactive=True
                            )

                @model_add_button.click(
                    inputs=[
                        model_name,
                        primary_stem,
                        secondary_stem,
                        weight,
                    ],
                    outputs=df,
                )
                def add_model_to_auto_ensemble(
                    model_name: str, 
                    primary_stem: str, 
                    secondary_stem: str, 
                    weight: float
                ) -> pd.DataFrame:
                    ensemble_model_manager.add(model_name, primary_stem, secondary_stem, weight)
                    return ensemble_model_manager.get_df()

                @model_replace_btn.click(
                    inputs=[
                        model_name,
                        primary_stem,
                        secondary_stem,
                        weight,
                        model_index,
                    ],
                    outputs=df,
                )
                def replace_model_to_auto_ensemble(
                    model_name: str, 
                    primary_stem: str, 
                    secondary_stem: str, 
                    weight: float, 
                    index: int
                ) -> pd.DataFrame:
                    ensemble_model_manager.replace(
                        model_name, primary_stem, secondary_stem, weight, index
                    )
                    return ensemble_model_manager.get_df()

                @model_delete_btn.click(inputs=[model_index], outputs=df)
                def delete_model_to_auto_ensemble(index: int) -> pd.DataFrame:
                    ensemble_model_manager.remove(index)
                    return ensemble_model_manager.get_df()

                @model_clear_btn.click(outputs=df)
                def clear_model_to_auto_ensemble() -> pd.DataFrame:
                    ensemble_model_manager.clear()
                    return ensemble_model_manager.get_df()

                gr.on(fn=ensemble_model_manager.get_df, outputs=df)

                df.change(
                    fn=ensemble_model_manager.save,
                    inputs=export_preset_name,
                    outputs=export_btn,
                )

                export_preset_name.change(
                    fn=ensemble_model_manager.save,
                    inputs=export_preset_name,
                    outputs=export_btn,
                )

                @import_btn.upload(inputs=import_btn, outputs=df)
                def load_ensemble_preset(filepath: str) -> pd.DataFrame:
                    ensemble_model_manager.load(filepath)
                    return ensemble_model_manager.get_df()

        with gr.Row():
            with gr.Column():
                gr.Markdown(f"<h3>{_i18n('input_audio')}</h3>")
                with gr.Group():
                    with gr.Group():
                        upload = gr.File(show_label=False, type="filepath", interactive=True)
                        refresh_input_btn = gr.Button(_i18n("refresh"), variant="primary", interactive=True)
                        list_input_files = gr.Dropdown(
                            label=_i18n("select_input_files"),
                            choices=self.input_files,
                            value=None,
                            multiselect=False,
                            interactive=True,
                            filterable=False, 
                            scale=15
                        )
                        
                        gr.on(
                            fn=lambda: gr.update(choices=reversed(self.input_files), value=None), 
                            outputs=list_input_files, 
                            trigger_mode="once"
                        )
                        
                        refresh_input_btn.click(
                            lambda: gr.update(choices=reversed(self.input_files), value=None), 
                            outputs=list_input_files
                        )
                            
                        @upload.upload(inputs=[upload], outputs=[list_input_files, upload])
                        def upload_files(input_file: str) -> Tuple[gr.update, gr.update]:
                            files: List[str] = self.upload_files_func([input_file])
                            return (
                                gr.update(choices=reversed(self.input_files), value=files[0] if files else None),
                                gr.update(value=None)
                            )
                            
            with gr.Column():
                gr.Markdown(f"<h3>{_i18n('ensemble_settings')}</h3>")
                with gr.Group():
                    method = gr.Dropdown(
                        label=_i18n("ensemble_algorithm"),
                        choices=["min_fft", "max_fft", "avg_fft", "median_fft"],
                        value="avg_fft",
                        filterable=False,
                    )
                    invert_ensemble = gr.Checkbox(
                        label=_i18n("ensemble_invert"), 
                        interactive=True, 
                        value=False
                    )
                    output_format = gr.Dropdown(
                        label=_i18n("output_format"),
                        interactive=True,
                        choices=output_formats,
                        value="mp3",
                        filterable=False,
                    )
                    run_btn = gr.Button(
                        _i18n("create_ensemble_btn"), 
                        variant="primary", 
                        interactive=True
                    )

        with gr.Group():
            with gr.Row(equal_height=True):
                list_ensembless_out = gr.Dropdown(
                    label=_i18n("select_ensemble_results"),
                    choices=[],
                    value=None,
                    interactive=True, 
                    scale=14
                )
                refresh_ensembless_out_btn = gr.Button(_i18n("refresh"), scale=2, interactive=True)
                refresh_ensembless_out_btn.click(
                    lambda: gr.update(choices=history.get_list(), value=None), 
                    outputs=[list_ensembless_out]
                )
                gr.on(
                    fn=lambda: gr.update(choices=history.get_list(), value=None), 
                    outputs=[list_ensembless_out]
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown(f"<h3>{_i18n('ensemble_results')}</h3>")
                output_audio = gr.Audio(
                    label=_i18n("ensemble_result"),
                    type="filepath",
                    interactive=False,
                    show_download_button=True,
                )
                output_audio_wav = gr.Textbox(
                    label=_i18n("ensemble_wav_result"), 
                    interactive=False, 
                    visible=False
                )
                with gr.Group():
                    invert_method = gr.Radio(
                        choices=self.methods_subtract,
                        label=_i18n("invert_method"),
                        value=self.methods_subtract[0] if self.methods_subtract else "waveform",
                    )
                    invert_btn = gr.Button(_i18n("invert_btn"))
                output_inverted_audio = gr.Audio(
                    label=_i18n("inversion_result"),
                    type="filepath",
                    interactive=False,
                    show_download_button=True,
                )

                @invert_btn.click(
                    inputs=[
                        list_input_files,
                        output_audio_wav,
                        invert_method,
                        output_format,
                    ],
                    outputs=[output_inverted_audio],
                )
                def invert_result_ensemble(
                    input_file: Optional[str], 
                    output_file: Optional[str], 
                    method: str, 
                    out_format: str
                ) -> Optional[Dict]:
                    if input_file and output_file:
                        o_dir: str = os.path.dirname(output_file)
                        basename: str = os.path.splitext(os.path.basename(input_file))[0]
                        output_path: str = os.path.join(
                            o_dir,
                            f"ensembless_{self.namer.short(basename, length=50)}_{method}_invert.{out_format}",
                        )
                        inverted: Optional[str] = self.subtract(
                            audio1_path=input_file,
                            audio2_path=output_file,
                            method=method,
                            output_path=output_path,
                        )
                        return self.return_audio_with_size(value=inverted, label=_i18n("inversion_result"))
                    else:
                        return None

            with gr.Column():
                gr.Markdown(f"<h3>{_i18n('ensemble_sources')}</h3>")
                output_source_files = gr.Files(
                    type="filepath", 
                    interactive=False, 
                    show_label=False
                )

        @list_ensembless_out.change(
            inputs=[list_ensembless_out], 
            outputs=[list_input_files, output_audio, output_audio_wav, output_inverted_audio, output_source_files]
        )
        def get_state(key: str) -> Tuple[gr.update, Optional[Dict], gr.update, Optional[Dict], gr.update]:
            state: Dict = history.get(key)
            input_file: str = state.get("input_file", "")
            last_result: Optional[str] = state.get("last_result")
            last_result_wav: Optional[str] = state.get("last_result_wav")
            last_inverted_result: Optional[str] = state.get("last_inverted_result")
            source: List[str] = state.get("source", [])
            
            return (
                gr.update(
                    value=input_file if input_file in self.input_files else None, 
                    choices=reversed(self.input_files)
                ),
                self.return_audio_with_size(value=last_result, label=_i18n("ensemble_result")),
                gr.update(value=last_result_wav),
                self.return_audio_with_size(value=last_inverted_result, label=_i18n("inversion_result")),
                gr.update(value=source)
            )

        @run_btn.click(
            inputs=[
                list_input_files,
                method,
                output_format,
                invert_ensemble,
            ],
            outputs=[
                output_audio,
                output_audio_wav,
                output_inverted_audio,
                output_source_files,
            ],
        )
        def auto_ensemble_run_(
            input_file: Optional[str],
            method: str,
            out_format: str,
            invert_ensemble: bool,
            progress: gr.Progress = gr.Progress(track_tqdm=True),
        ) -> Tuple[Optional[Dict], gr.update, Optional[Dict], gr.update]:
            timestamp: str = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
            output_dir: str = os.path.join(self.user_directory.path, "output", "ensembless", f"auto_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            ensemble_state: List = ensemble_model_manager.data
            result: Tuple[Optional[str], Optional[str], Optional[str], List[str]] = self.auto_ensemble(
                input_file, 
                ensemble_state, 
                output_dir, 
                method, 
                out_format, 
                invert_ensemble, 
                progress=progress
            )
            
            auto_ensemble_out_file, auto_ensemble_out_file_wav, auto_ensemble_invout_file, ensemble_sources_list = result
            
            history.add(
                input_file or "", 
                auto_ensemble_out_file or "", 
                auto_ensemble_out_file_wav or "", 
                auto_ensemble_invout_file, 
                ensemble_sources_list, 
                method, 
                timestamp, 
                ensemble_state
            )
            
            return (
                self.return_audio_with_size(value=auto_ensemble_out_file, label=_i18n("ensemble_result")),
                auto_ensemble_out_file_wav or "",
                self.return_audio_with_size(value=auto_ensemble_invout_file, label=_i18n("inversion_result")),
                gr.update(value=ensemble_sources_list),
            )


class ManualEnsembless(Separator, GradioHelper):
    """Класс для ручного ансамбля"""
    
    def __init__(self, user_directory: Any) -> None:
        """
        Инициализация ручного ансамбля
        
        Args:
            user_directory: Пользовательская директория
        """
        super().__init__()
        self.user_directory: Any = user_directory

    def UI(self) -> None:
        """Создать пользовательский интерфейс"""
        with gr.Row():
            with gr.Column():
                input_ensemble_files = gr.File(
                    label=_i18n("input_audio_files"),
                    interactive=True,
                    type="filepath",
                    file_count="multiple",
                )

            with gr.Column():
                @gr.render(inputs=[input_ensemble_files])
                def input_ensemble_files_fn(input_files: Optional[List[str]]) -> None:
                    check_ensemble_files_status: str = f"{_i18n('audio_analysis')}\n---"
                    hz_: List[int] = []
                    err_list: List[str] = []
                    
                    if input_files:
                        for file in input_files:
                            basename: str = os.path.splitext(os.path.basename(file))[0]
                            if os.path.exists(file):
                                if check(file):
                                    hz: int = get_sr(file)
                                    check_ensemble_files_status += (
                                        f"\n{basename} - {_i18n('msg_file_check_ok')} ({hz} {_i18n('unit_hz')})"
                                    )
                                    hz_.append(hz)
                                else:
                                    check_ensemble_files_status += (
                                        f"\n{basename} - {_i18n('msg_file_check_no_audio')}"
                                    )
                                    err_list.append(file)
                            else:
                                check_ensemble_files_status += (
                                    f"\n{basename} - {_i18n('msg_file_not_found')}"
                                )
                                err_list.append(file)

                    check_ensemble_files_result: str = _i18n("msg_valid_files_count", count=len(hz_))

                    all_same: bool = True
                    common_rate: Optional[int] = None

                    for hz_val in hz_:
                        if common_rate is None:
                            common_rate = hz_val
                        elif common_rate != hz_val:
                            all_same = False

                    if hz_ and len(hz_) > 1:
                        check_ensemble_files_result += (
                            f"\n{_i18n('msg_same_sample_rate') if all_same else _i18n('msg_diff_sample_rate')}"
                        )
                    else:
                        check_ensemble_files_result += f"\n{_i18n('msg_min_files_needed')}"

                    check_ensemble_files_status += f"\n \n{check_ensemble_files_result}"

                    gr.Textbox(
                        container=False,
                        lines=len(check_ensemble_files_status.split("\n")),
                        interactive=False,
                        value=check_ensemble_files_status,
                    )

        weights = gr.Textbox(
            label=_i18n("weights_input"), 
            value="1.0,1.0",
            info=_i18n("weights_format")
        )
        
        @input_ensemble_files.change(
            inputs=[input_ensemble_files], 
            outputs=[weights]
        )
        def parse_weights(files: Optional[List[str]]) -> str:
            if files:
                total: int = len(files)
                weights_list: List[str] = [str(1.0) for _c in range(total)]
                return ",".join(weights_list)
            else:
                return ""
        
        method = gr.Dropdown(
            label=_i18n("ensemble_algorithm"),
            choices=["min_fft", "max_fft", "avg_fft", "median_fft"],
            value="avg_fft",
            filterable=False,
        )

        output_format = gr.Dropdown(
            label=_i18n("output_format"),
            interactive=True,
            choices=output_formats,
            value="mp3",
            filterable=False,
        )

        output_manual_ensemble_filename = gr.Textbox(
            label=_i18n("output_filename"), 
            value="ensemble", 
            interactive=True
        )

        make_manual_ensemble_btn = gr.Button(
            value=_i18n("create_manual_ensemble_btn"), 
            variant="primary"
        )

        manual_ensemble_output_audio = gr.Audio(
            label=_i18n("ensemble_result"),
            type="filepath",
            interactive=False,
            show_download_button=True,
        )

        @make_manual_ensemble_btn.click(
            inputs=[
                input_ensemble_files,
                method,
                output_format,
                output_manual_ensemble_filename,
                weights,
            ],
            outputs=manual_ensemble_output_audio,
        )
        def make_manual_ensemble_fn(
            input_files_list: Optional[List[str]],
            method: str,
            out_format: str,
            output_filename: str,
            weights_str: str,
        ) -> Optional[Dict]:
            if not input_files_list:
                return None
                
            timestamp: str = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
            output_dir: str = os.path.join(
                self.user_directory.path, 
                "output", 
                "ensembless", 
                f"manual_{timestamp}"
            )
            os.makedirs(output_dir, exist_ok=True)

            safe_filename: str = self.namer.sanitize(output_filename)
            safe_filename = self.namer.short(safe_filename)

            try:
                weights_list: List[float] = [float(x.strip()) for x in weights_str.split(",") if x.strip()]
            except ValueError:
                weights_list = [1.0] * len(input_files_list)

            output_file: Optional[str] = self.manual_ensemble(
                files=input_files_list,
                output_name=os.path.join(output_dir, safe_filename),
                weights=weights_list,
                ensemble_type=method,
                out_format=out_format,
            )
            return self.return_audio_with_size(value=output_file, label=_i18n("ensemble_result"))


class Inverter_UI(Separator, GradioHelper):
    """Класс для интерфейса инвертора"""
    
    def __init__(self) -> None:
        super().__init__()
        
    def UI(self) -> None:
        """Создать пользовательский интерфейс"""
        with gr.Group():
            with gr.Row():
                original_audio = gr.File(
                    label=_i18n("original_audio"),
                    interactive=True,
                    type="filepath",
                    file_count="single",
                )
                stem_audio = gr.File(
                    label=_i18n("stem_to_subtract"),
                    interactive=True,
                    type="filepath",
                    file_count="single",
                )
            with gr.Group():
                output_format = gr.Dropdown(
                    label=_i18n("output_format"),
                    interactive=True,
                    choices=output_formats,
                    value="mp3",
                    filterable=False,
                )
                method = gr.Radio(
                    choices=self.methods_subtract,
                    label=_i18n("subtract_method"),
                    value=self.methods_subtract[0] if self.methods_subtract else "waveform",
                )
                btn = gr.Button(_i18n("subtract_btn"))
                
        output_audio = gr.Audio(
            label=_i18n("inversion_result"),
            type="filepath",
            interactive=False,
            show_download_button=True,
        )

        @btn.click(
            inputs=[original_audio, stem_audio, method, output_format],
            outputs=[output_audio],
        )
        def invert_result_ensemble(
            input_file: Optional[str], 
            stem_file: Optional[str], 
            method: str, 
            out_format: str
        ) -> Optional[Dict]:
            if input_file and stem_file:
                o_dir: str = tempfile.mkdtemp(suffix="_inverter")
                basename: str = os.path.splitext(os.path.basename(input_file))[0]
                output_path: str = os.path.join(
                    o_dir,
                    f"inverter_{self.namer.short(basename, length=50)}_{method}.{out_format}",
                )
                inverted: Optional[str] = self.subtract(
                    audio1_path=input_file,
                    audio2_path=stem_file,
                    method=method,
                    output_path=output_path,
                )
                return self.return_audio_with_size(value=inverted, label=_i18n("inversion_result"))
            else:
                return None


class AudioApp(Separator, GradioHelper):
    """Класс для приложения обработки аудио"""
    
    def __init__(self, user_directory: Any) -> None:
        """
        Инициализация аудио приложения
        
        Args:
            user_directory: Пользовательская директория
        """
        super().__init__()
        self.user_directory: Any = user_directory
    
    def UI(self) -> None:
        """Создать пользовательский интерфейс"""
        with gr.Tab(_i18n("concat_audio")):
            with gr.Group():
                input_concat_files = gr.File(
                    label=_i18n("input_audio_files"), 
                    file_count="multiple", 
                    type="filepath", 
                    interactive=True
                )
                output_format = gr.Dropdown(
                    label=_i18n("output_format"),
                    interactive=True,
                    choices=output_formats,
                    value="mp3",
                    filterable=False,
                )
                concat_btn = gr.Button(_i18n("concat_btn"), variant="primary", interactive=True)
                
            concated_audio = gr.Audio(
                label=_i18n("ensemble_result"),
                type="filepath",
                interactive=False,
                show_download_button=True,
            )
            
            @concat_btn.click(inputs=[input_concat_files, output_format], outputs=concated_audio)
            def concat_fn(files: Optional[List[str]], out_format: str) -> Optional[Dict]:
                if not files:
                    return None
                    
                timestamp: str = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
                output_dir: str = os.path.join(
                    self.user_directory.path, 
                    "output", 
                    "audio-editor", 
                    timestamp
                )
                os.makedirs(output_dir, exist_ok=True)
                output_path: str = os.path.join(output_dir, f"concated_{timestamp}.{out_format}")
                result: Optional[str] = concat_audio(files, output_path)
                return self.return_audio_with_size(value=result, label=_i18n("ensemble_result"))
                
        with gr.Tab(_i18n("trim_audio")):
            with gr.Group():
                input_trim_file = gr.File(
                    label=_i18n("input_audio"), 
                    file_count="single", 
                    type="filepath", 
                    interactive=True
                )
                
                @gr.render(inputs=[input_trim_file])
                def preview_input_file(file: Optional[str]) -> None:
                    if file:
                        self.define_audio_with_size(
                            value=file, 
                            label=_i18n("trim_preview")
                        )
                        
                with gr.Row():
                    start_num = gr.Number(
                        label=_i18n("trim_start"), 
                        minimum=0, 
                        maximum=1, 
                        value=0, 
                        interactive=True, 
                        min_width=80
                    )
                    end_num = gr.Number(
                        label=_i18n("trim_end"), 
                        minimum=0, 
                        maximum=1, 
                        value=1, 
                        interactive=True, 
                        min_width=80
                    )
                    
                @input_trim_file.change(inputs=[input_trim_file], outputs=[start_num, end_num])
                def input_trim_fn(file: Optional[str]) -> Tuple[gr.update, gr.update]:
                    if file:
                        y, sr = read(file)
                        duration: float = get_duration_from_array(y, sr)
                        return (
                            gr.update(minimum=0, maximum=duration, value=0),
                            gr.update(minimum=0, maximum=duration, value=duration, placeholder=str(duration))
                        )
                    else:
                        return (
                            gr.update(minimum=0, maximum=1, value=0),
                            gr.update(minimum=0, maximum=1, value=1, placeholder="1")
                        )
                        
                out_format2 = gr.Dropdown(
                    label=_i18n("output_format"),
                    interactive=True,
                    choices=output_formats,
                    value="mp3",
                    filterable=False,
                )
                trim_btn = gr.Button(_i18n("trim_btn"), variant="primary")
                
            trimmed_audio = gr.Audio(
                label=_i18n("ensemble_result"),
                type="filepath",
                interactive=False,
                show_download_button=True,
            )
            
            @trim_btn.click(inputs=[input_trim_file, start_num, end_num, out_format2], outputs=trimmed_audio)
            def trim_fn(
                input_file: Optional[str], 
                start: float, 
                end: float, 
                out_format: str
            ) -> Optional[Dict]:
                if not input_file:
                    return None
                    
                timestamp: str = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
                output_dir: str = os.path.join(
                    self.user_directory.path, 
                    "output", 
                    "audio-editor", 
                    timestamp
                )
                os.makedirs(output_dir, exist_ok=True)
                
                basename, ext = os.path.splitext(os.path.basename(input_file))
                basename = self.namer.short(basename, length=50)
                filename: str = f"{basename}_trimmed_{timestamp}.{out_format}"
                output_path: str = os.path.join(output_dir, filename)
                
                result: Optional[str] = trim_audio(input_file, start, end, output_path)
                return self.return_audio_with_size(value=result, label=_i18n("ensemble_result"))
                
        with gr.Tab(_i18n("extract_phantom_center")):
            with gr.Group():
                input_stereo_file = gr.File(
                    label=_i18n("input_audio"), 
                    file_count="single", 
                    type="filepath", 
                    interactive=True
                )
                
                @gr.render(inputs=[input_stereo_file])
                def preview_input_file(file: Optional[str]) -> None:
                    if file:
                        self.define_audio_with_size(
                            value=file, 
                            label=_i18n("trim_preview")
                        )
                        
                out_format3 = gr.Dropdown(
                    label=_i18n("output_format"),
                    interactive=True,
                    choices=output_formats,
                    value="mp3",
                    filterable=False,
                )
                separate_mid_side_btn = gr.Button(_i18n("separate_ms_btn"), variant="primary")
                
            with gr.Row():
                mid_audio = gr.Audio(
                    label=_i18n("phantom_center"),
                    type="filepath",
                    interactive=False,
                    show_download_button=True,
                )
                side_audio = gr.Audio(
                    label=_i18n("stereo_base"),
                    type="filepath",
                    interactive=False,
                    show_download_button=True,
                )
                
            @separate_mid_side_btn.click(
                inputs=[input_stereo_file, out_format3], 
                outputs=[mid_audio, side_audio]
            )
            def sep_ms_fn(input_file: Optional[str], out_format: str) -> Tuple[Optional[Dict], Optional[Dict]]:
                if not input_file:
                    return None, None
                    
                timestamp: str = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
                output_dir: str = os.path.join(
                    self.user_directory.path, 
                    "output", 
                    "audio-editor", 
                    timestamp
                )
                os.makedirs(output_dir, exist_ok=True)
                
                basename, ext = os.path.splitext(os.path.basename(input_file))
                basename = self.namer.short(basename, length=50)
                filename_mid: str = f"{basename}_center_{timestamp}.{out_format}"
                filename_side: str = f"{basename}_stereo_base_{timestamp}.{out_format}"
                output_path_mid: str = os.path.join(output_dir, filename_mid)
                output_path_side: str = os.path.join(output_dir, filename_side)
                
                mid_result, side_result = self.extract_phantom_center(
                    input_file, 
                    output_path_mid, 
                    output_path_side
                )
                
                return (
                    self.return_audio_with_size(value=mid_result, label=_i18n("phantom_center")),
                    self.return_audio_with_size(value=side_result, label=_i18n("stereo_base"))
                )


class PluginManager(Separator):
    """Класс для управления плагинами"""
    
    plugins_dir: str = os.path.join(script_dir, "plugins")
    os.makedirs(plugins_dir, exist_ok=True)

    def restart_after_install_plugin(self) -> None:
        """Перезапустить приложение после установки плагина"""
        subprocess.Popen(
            [os.sys.executable] + [os.path.join(script_dir, "app.py")] + sys.argv[1:]
        )
        os._exit(0)

    def parse_plugins(self) -> None:
        """Загрузить и отобразить плагины"""
        for plugin_file in os.listdir(self.plugins_dir):
            if not plugin_file.endswith(".py") or plugin_file == "__init__.py":
                continue

            plugin_module_name: str = os.path.splitext(plugin_file)[0]

            try:
                if __package__:
                    plugin_module = importlib.import_module(
                        f".plugins.{plugin_module_name}", package=__package__
                    )
                else:
                    try:
                        plugin_module = importlib.import_module(
                            f"plugins.{plugin_module_name}"
                        )
                    except ImportError:
                        plugin_path: str = os.path.join(self.plugins_dir, plugin_file)
                        spec = importlib.util.spec_from_file_location(
                            plugin_module_name, plugin_path
                        )
                        plugin_module = importlib.util.module_from_spec(spec)
                        if spec and spec.loader:
                            spec.loader.exec_module(plugin_module)

                plugin_class = getattr(plugin_module, "Plugin")
                plugin_instance = plugin_class()

                with gr.Tab(plugin_instance.name):
                    plugin_instance.UI()

            except Exception as e:
                print(f"{_i18n('plugin_load_error')} {plugin_module_name}: {e}")
                continue

    def UI(self) -> None:
        """Создать пользовательский интерфейс"""
        with gr.Tab(_i18n("tab_install_plugin")):
            with gr.Group():
                upload_plugins_files = gr.File(
                    label=_i18n("upload_plugins"),
                    file_types=[".py"],
                    file_count="multiple",
                    interactive=True,
                )
                install_plugins_btn = gr.Button(_i18n("install_plugins_btn"), interactive=True)

            @install_plugins_btn.click(inputs=[upload_plugins_files])
            def upload_plugin_list(files: Optional[List[str]]) -> None:
                if not files:
                    return
                    
                for file in files:
                    try:
                        if file.endswith(".py"):
                            shutil.copy(
                                file,
                                os.path.join(
                                    self.plugins_dir,
                                    os.path.basename(file).replace(" ", "_"),
                                ),
                            )
                    except Exception as e:
                        print(f"{_i18n('file_copy_error')} {file}: {e}")
                        
                time.sleep(2)
                self.restart_after_install_plugin()

        self.parse_plugins()