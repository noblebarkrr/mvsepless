import os
import sys
import shutil
import importlib.util
import gradio as gr
import pandas as pd
import subprocess
import json
import time
from datetime import datetime
import tempfile
from functools import wraps
from typing import List, Tuple, Optional, Dict, Any, Callable, Union
from pathlib import Path

from separator import Separator, script_dir
from gradio_helper import GradioHelper, tz
from audio import output_formats, check, read, get_sr, get_duration_from_array, multiread, write, trim, concatenate
from i18n import _i18n


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