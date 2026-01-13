import os
import sys
import shutil
import logging
import zipfile
import importlib.util
from pathlib import Path
from typing import Literal
import gradio as gr
import pandas as pd
import subprocess
import json
import threading
import queue
import time
import argparse
from datetime import datetime
import tempfile
import ast
import numpy as np
import librosa
from functools import wraps
from separator import Separator, script_dir
from gradio_helper import GradioHelper, tz
from namer import Namer
from audio import input_extensions, output_formats, check, read, write, get_sr
from ensemble import ensemble_audio_files

namer = Namer()

class Inverter:
    def __init__(self):
        self.test = "test"
        self.w_types = [
            "boxcar",
            "triang",
            "blackman",
            "hamming",
            "hann",
            "bartlett",
            "flattop",
            "parzen",
            "bohman",
            "blackmanharris",
            "nuttall",
            "barthann",
            "cosine",
            "exponential",
            "tukey",
            "taylor",
            "lanczos",
        ]

    def load_audio(self, filepath):
        try:
            y, sr = read(path=filepath, sr=None, mono=False)
            return y, sr
        except Exception as e:
            print(f"Ошибка загрузки аудио: {e}")
            return None, None

    def process_channel(
        self, y1_ch, y2_ch, sr, method, w_size=2048, overlap=2, w_type="hann"
    ):
        HOP_LENGTH = w_size // overlap
        if method == "waveform":
            return y1_ch - y2_ch

        elif method == "spectrogram":
            S1 = librosa.stft(
                y1_ch, n_fft=w_size, hop_length=HOP_LENGTH, win_length=w_size
            )
            S2 = librosa.stft(
                y2_ch, n_fft=w_size, hop_length=HOP_LENGTH, win_length=w_size
            )

            mag1 = np.abs(S1)
            mag2 = np.abs(S2)

            mag_result = np.maximum(mag1 - mag2, 0)

            phase = np.angle(S1)

            S_result = mag_result * np.exp(1j * phase)

            return librosa.istft(
                S_result,
                n_fft=w_size,
                hop_length=HOP_LENGTH,
                win_length=w_size,
                length=len(y1_ch),
            )

    def process_audio(
        self,
        audio1_path,
        audio2_path,
        out_format,
        method,
        output_path="./inverted.mp3",
        w_size=2048,
        overlap=2,
        w_type="hann",
    ):
        y1, sr1 = self.load_audio(audio1_path)
        y2, sr2 = self.load_audio(audio2_path)

        if sr1 is None or sr2 is None:
            raise Exception("Произошла ошибка при чтении файлов")

        channels1 = 1 if y1.ndim == 1 else y1.shape[0]
        channels2 = 1 if y2.ndim == 1 else y2.shape[0]

        if channels1 > 1:
            y1 = y1.T
        else:
            y1 = y1.reshape(-1, 1)

        if channels2 > 1:
            y2 = y2.T
        else:
            y2 = y2.reshape(-1, 1)

        if sr1 != sr2:
            if channels2 > 1:
                y2_resampled_list = []
                for c in range(channels2):
                    channel_resampled = librosa.resample(
                        y2[:, c], orig_sr=sr2, target_sr=sr1
                    )
                    y2_resampled_list.append(channel_resampled)

                min_channel_length = min(len(ch) for ch in y2_resampled_list)

                y2_resampled = np.zeros(
                    (min_channel_length, channels2), dtype=np.float32
                )
                for c, channel in enumerate(y2_resampled_list):
                    y2_resampled[:, c] = channel[:min_channel_length]

                y2 = y2_resampled
            else:
                y2 = librosa.resample(y2[:, 0], orig_sr=sr2, target_sr=sr1)
                y2 = y2.reshape(-1, 1)
            sr2 = sr1

        min_len = min(len(y1), len(y2))
        y1 = y1[:min_len]
        y2 = y2[:min_len]

        result_channels = []

        if channels1 == 1 and channels2 > 1:
            y2 = y2.mean(axis=1, keepdims=True)
            channels2 = 1

        for c in range(channels1):
            y1_ch = y1[:, c]

            if channels2 == 1:
                y2_ch = y2[:, 0]
            else:
                y2_ch = y2[:, min(c, channels2 - 1)]

            result_ch = self.process_channel(
                y1_ch, y2_ch, sr1, method, w_size=w_size, overlap=overlap, w_type=w_type
            )
            result_channels.append(result_ch)

        if len(result_channels) > 1:
            result = np.column_stack(result_channels)
        else:
            result = np.array(result_channels[0])

        if result.ndim > 1:
            for c in range(result.shape[1]):
                channel = result[:, c]
                max_val = np.max(np.abs(channel))
                if max_val > 0:
                    result[:, c] = channel * 0.9 / max_val
        else:
            max_val = np.max(np.abs(result))
            if max_val > 0:
                result = result * 0.9 / max_val

        inverted = write(
            output_path, result.T, sr1, "320k"
        )
        return inverted


class AutoEnsembless(Separator, GradioHelper):
    def __init__(self, input_files, upload_files, user_directory, device):
        super().__init__()
        self.inverter = Inverter()
        self.input_files = input_files
        self.upload_files = upload_files
        self.user_directory = user_directory
        self.device = device
        
    class ModelManager(Separator):
        def __init__(self):
            self.data: list[list[str, str, str, int]] = []
            self.ensemble_methods = ("min_fft", "max_fft", "avg_fft", "median_fft")
            self.ensemble_invert_methods_map = {
                "min_fft": "max_fft",
                "max_fft": "min_fft",
                "avg_fft": "avg_fft",
                "median_fft": "median_fft",
            }
            self.dir_presets = os.path.join(tempfile.tempdir, "presets")
            os.makedirs(self.dir_presets, exist_ok=True)

        def save(self, name):
            if not name:
                name = "ensembless_preset"
            filepath = os.path.join(
                self.dir_presets,
                f"{namer.short(namer.sanitize(name), length=50)}.json",
            )
            with open(filepath, "w") as f:
                json.dump(self.data, f, indent=4, ensure_ascii=False)
            return filepath

        def load(self, filepath):
            with open(filepath, "r") as f:
                ensemble_data_temp = json.load(f)
            self.data = []
            for mn, s_stem, i_stem, weight in ensemble_data_temp:
                if {mn} not in [{model[0]} for model in self.data]:
                    self.data.append((mn, s_stem, i_stem, weight))

        def add(self, mn, s_stem, i_stem, weight):
            if {mn} not in [{model[0]} for model in self.data]:
                if s_stem and i_stem:
                    self.data.append((mn, s_stem, i_stem, weight))

        def replace(self, mn, s_stem, i_stem, weight, index=1):
            if self.data:
                len_data = len(self.data)
                if index >= 1:
                    if index <= len_data:
                        self.data[index - 1] = (mn, s_stem, i_stem, weight)
                elif index == 0:
                    self.data[0] = (mn, s_stem, i_stem, weight)

        def remove(self, index=1):
            if self.data:
                len_data = len(self.data)
                if index >= 1:
                    if index <= len_data:
                        del self.data[index - 1]
                elif index == 0:
                    del self.data[0]

        def clear(self):
            self.data = []

        def get_df(self):
            if not self.data:
                columns = ["#", "Имя модели", "Основной стем", "Инверсия", "Вес"]
                return pd.DataFrame(columns=columns)

            data = []
            for i, model in enumerate(self.data):
                data.append(
                    [
                        f"{i+1}",
                        model[0],
                        model[1],
                        model[2],
                        model[3],
                    ]
                )
            columns = ["#", "Имя модели", "Основной стем", "Инверсия", "Вес"]
            return pd.DataFrame(data, columns=columns)

    class History:
        def __init__(self, user_directory):
            self.info = {}
            self.path = os.path.join(user_directory.path, "history_auto_ensemble.json")
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
        def add(self, input_file: str, last_result: str, last_result_wav: str, last_inverted_result: str | None, ensemble_sources: list, method: str, timestamp: str, models: list):
            self.info[f"{timestamp} / {method} / {len(models)}"] = {
                "input_file": input_file,
                "last_result": last_result,
                "last_result_wav": last_result_wav,
                "last_inverted_result": last_inverted_result,
                "source": ensemble_sources
            }
        
        @_save_to_file
        def clear(self):
            self.info = {}
        
        def get_list(self):
            return sorted([key for key in self.info], reverse=True)
        
        def get(self, key):
            return self.info.get(key, {})
        
        def load_from_file(self):
            """Загрузить историю из файла"""
            if os.path.exists(self.path):
                with open(self.path, 'r', encoding='utf-8') as f:
                    self.info = json.load(f)

    def UI(self):
        ensemble_model_manager = self.ModelManager()
        history = self.History(self.user_directory)

        def get_stems(mn):
            stems = []
            for stem in self.get_stems(mn):
                stems.append(stem)

            if not self.get_tgt_inst(mn):
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

        def get_invert_stems(mn, s_stem):
            orig_stems = []
            stems = []
            for stem in self.get_stems(mn):
                orig_stems.append(stem)

            for stem in orig_stems:
                if stem != s_stem:
                    stems.append(stem)

            if not self.get_tgt_inst(mn):
                if len(orig_stems) > 2:
                    if s_stem not in ["instrumental +", "instrumental -"]:
                        stems.append("inverted +")
                        stems.append("inverted -")

            return stems

        default_model = {
            "mn": self.get_mn(),
            "stem": get_stems(
                self.get_mn()[0]),
            "invert_stem": get_invert_stems(
                self.get_mn()[0],
                "vocals",
            ),
            "weight": 1,
        }

        gr.Markdown("<h3>Пресет</h3>")
        with gr.Group():
            with gr.Row(equal_height=True):
                export_preset_name = gr.Textbox(
                    label="Имя пресета",
                    interactive=True,
                    value="ensembless_preset",
                    scale=9,
                )
                export_btn = gr.DownloadButton(
                    "Экспорт", variant="secondary", scale=3, interactive=True
                )
                import_btn = gr.UploadButton(
                    "Импорт",
                    file_types=[".json"],
                    file_count="single",
                    scale=3,
                    interactive=True,
                )
        gr.Markdown("<h3>Ансамбль</h3>")
        with gr.Row():
            with gr.Column(scale=3):
                model_name = gr.Dropdown(
                    label="Имя модели",
                    choices=default_model["mn"],
                    value=default_model["mn"][0],
                    interactive=True,
                    filterable=False,
                )
                primary_stem = gr.Dropdown(
                    label="Основной стем",
                    choices=default_model["stem"],
                    value=default_model["stem"][0],
                    interactive=True,
                    filterable=False,
                )
                secondary_stem = gr.Dropdown(
                    label="Инверсия",
                    choices=default_model["invert_stem"],
                    value=default_model["invert_stem"][0],
                    interactive=True,
                    filterable=False,
                )
                weight = gr.Slider(
                    label="Вес",
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
                def update_stems_after_model_change(mn):
                    stems = get_stems(mn)
                    invert_stems = get_invert_stems(mn, stems[0]) if stems else []

                    new_s_stem = stems[0] if stems else ""
                    new_i_stem = invert_stems[0] if invert_stems else ""

                    return (
                        gr.update(choices=stems, value=new_s_stem),
                        gr.update(choices=invert_stems, value=new_i_stem),
                    )

                @primary_stem.change(
                    inputs=[model_name, primary_stem],
                    outputs=[secondary_stem],
                )
                def update_invert_stems(mn, s_stem):
                    stems = get_invert_stems(mn, s_stem)
                    new_i_stem = stems[0] if stems else ""
                    return gr.update(choices=stems, value=new_i_stem)

                model_add_button = gr.Button("Добавить", interactive=True)
            with gr.Column(scale=10):
                df = gr.DataFrame(
                    value=ensemble_model_manager.get_df(),
                    headers=["#", "Имя модели", "Основной стем", "Инверсия", "Вес"],
                    datatype=["number", "str", "str", "str", "number"],
                    interactive=False,
                )

                with gr.Group():
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            model_index = gr.Number(
                                label="Индекс модели", value=1, interactive=True
                            )
                            model_clear_btn = gr.Button(
                                "Очистить", variant="stop", interactive=True
                            )
                        with gr.Column():
                            model_replace_btn = gr.Button(
                                "Заменить", variant="primary", interactive=True
                            )
                            model_delete_btn = gr.Button(
                                "Удалить", variant="stop", interactive=True
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
                def add_model_to_auto_ensemble(mn, s_stem, i_stem, weight):
                    ensemble_model_manager.add(mn, s_stem, i_stem, weight)
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
                    mn, s_stem, i_stem, weight, index
                ):
                    ensemble_model_manager.replace(
                        mn, s_stem, i_stem, weight, index
                    )
                    return ensemble_model_manager.get_df()

                @model_delete_btn.click(inputs=[model_index], outputs=df)
                def delete_model_to_auto_ensemble(index):
                    ensemble_model_manager.remove(index)
                    return ensemble_model_manager.get_df()

                @model_clear_btn.click(outputs=df)
                def clear_model_to_auto_ensemble():
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
                def load_ensemble_preset(filepath):
                    ensemble_model_manager.load(filepath)
                    return ensemble_model_manager.get_df()

        with gr.Row():
            with gr.Column():
                gr.Markdown("<h3>Входное аудио</h3>")
                with gr.Group():
                    with gr.Group():
                        upload = gr.File(show_label=False, type="filepath", interactive=True)
                        refresh_input_btn = gr.Button("Обновить", variant="primary", interactive=True)
                        list_input_files = gr.Dropdown(
                            label="Загрузить файлы",
                            choices=self.input_files,
                            value=None,
                            multiselect=False,
                            interactive=True,
                            filterable=False, scale=15
                        )
                        refresh_input_btn.click(lambda: gr.update(choices=reversed(self.input_files), value=None), outputs=list_input_files)
                            
                        @upload.upload(inputs=[upload], outputs=[list_input_files, upload])
                        def upload_files(input_file):
                            files = self.upload_files([input_file])
                            return gr.update(
                                choices=reversed(self.input_files), value=files[0]
                            ), gr.update(value=None)
            with gr.Column():
                gr.Markdown("<h3>Настройки</h3>")
                with gr.Group():
                    method = gr.Dropdown(
                        label="Алгоритм склеивания",
                        choices=["min_fft", "max_fft", "avg_fft", "median_fft"],
                        value="avg_fft",
                        filterable=False,
                    )
                    invert_ensemble = gr.Checkbox(
                        label="Инверсия ансамбля", interactive=True, value=False
                    )
                    output_format = gr.Dropdown(
                        label="Формат выходного файла",
                        interactive=True,
                        choices=output_formats,
                        value="mp3",
                        filterable=False,
                    )
                    run_btn = gr.Button(
                        "Создать ансамбль", variant="primary", interactive=True
                    )

        with gr.Group():
            with gr.Row(equal_height=True):
                list_ensembless_out = gr.Dropdown(
                    label="Выберите результаты авто-ансамбля",
                    choices=[],
                    value=None,
                    interactive=True, scale=14
                )
                refresh_ensembless_out_btn = gr.Button("Обновить", scale=2, interactive=True)
                refresh_ensembless_out_btn.click(lambda: gr.update(choices=history.get_list(), value=None), outputs=[list_ensembless_out])
                gr.on(fn=lambda: gr.update(choices=history.get_list(), value=None), outputs=[list_ensembless_out])

        with gr.Row():
            with gr.Column():
                gr.Markdown("<h3>Результаты</h3>")
                output_audio = gr.Audio(
                    label="Результат",
                    type="filepath",
                    interactive=False,
                    show_download_button=True,
                )
                output_audio_wav = gr.Textbox(
                    label="Результат в WAV", interactive=False, visible=False
                )
                with gr.Group():
                    invert_method = gr.Radio(
                        choices=["waveform", "spectrogram"],
                        label="Метод создания инверсии",
                        value="waveform",
                    )
                    invert_btn = gr.Button("Инвертировать")
                output_inverted_audio = gr.Audio(
                    label="Инверсия",
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
                def invert_result_ensemble(input_file, output_file, method, out_format):
                    if input_file and output_file:
                        o_dir = os.path.dirname(output_file)
                        basename = os.path.splitext(os.path.basename(input_file))[0]
                        output_path = os.path.join(
                            o_dir,
                            f"ensembless_{namer.short(basename, length=50)}_{method}_invert.{out_format}",
                        )
                        inverted = self.inverter.process_audio(
                            audio1_path=input_file,
                            audio2_path=output_file,
                            out_format=out_format,
                            method=method,
                            output_path=output_path,
                        )
                        return self.return_audio_with_size(value=inverted, label="Инверсия")
                    else:
                        return None

            with gr.Column():
                gr.Markdown("<h3>Исходники ансамбля (WAV)</h3>")
                output_source_files = gr.Files(
                    type="filepath", interactive=False, show_label=False
                )

        @list_ensembless_out.change(inputs=[list_ensembless_out], outputs=[list_input_files, output_audio, output_audio_wav, output_inverted_audio, output_source_files])
        def get_state(key):
            state = history.get(key)
            input_file = state.get("input_file")
            last_result = state.get("last_result")
            last_result_wav = state.get("last_result_wav")
            last_inverted_result = state.get("last_inverted_result")
            source = state.get("source", [])
            return gr.update(value=input_file if input_file in self.input_files else None, choices=reversed(self.input_files)), self.return_audio_with_size(value=last_result, label="Результат"), gr.update(value=last_result_wav), self.return_audio_with_size(value=last_inverted_result, label="Инверсия"), gr.update(value=source)

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
        def auto_ensemble_run(
            input_file,
            method,
            out_format,
            invert_ensemble,
            progress=gr.Progress(track_tqdm=True),
        ):
            ensemble_state = ensemble_model_manager.data
            invert_methods_map = ensemble_model_manager.ensemble_invert_methods_map
            if not input_file:
                return None, None, None, None, []
            if not os.path.exists(input_file):
                return None, None, None, None, []
            if not check(input_file):
                return None, None, None, None, []
            
            timestamp = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
            o = os.path.join(self.user_directory.path, "ensembless_output", f"ensembless_outputs_{timestamp}")
            os.makedirs(o, exist_ok=True)

            basename = os.path.splitext(os.path.basename(input_file))[0]

            def invert_weights(weights):
                total_weight = sum(weights)
                return [total_weight - w for w in weights]

            success_separations = []
            ensemble_sources_list = []
            if ensemble_state:
                total_ensemble_models = len(ensemble_state)
                for i, model in enumerate(ensemble_state, start=1):

                    ens_mn = model[0]
                    ens_s_stem = model[1]
                    ens_i_stem = model[2]
                    weight = model[3]

                    s_stem = None
                    i_stem = None

                    try:
                        result_seped_auto_ensemble = self.separate(
                            input=input_file,
                            output_dir=os.path.join(o, ens_mn),
                            model_name=ens_mn,
                            ext_inst=True,
                            template="NAME - MODEL - STEM",
                            output_format="wav",
                            add_settings={
                                "add_single_sep_text_progress": f"{i} из {total_ensemble_models}"
                            },
                            progress=progress,
                        )
                        if result_seped_auto_ensemble:
                            for stem, path in result_seped_auto_ensemble:
                                ensemble_sources_list.append(path)
                                if stem == ens_s_stem:
                                    s_stem = path
                                elif stem == ens_i_stem:
                                    i_stem = path

                        if invert_ensemble:
                            if not i_stem:
                                result_seped_auto_ensemble_invert = self.separate(
                                    input=input_file,
                                    output_dir=os.path.join(o, f"{ens_mn}_invert"),
                                    model_name=ens_mn,
                                    ext_inst=True,
                                    template="NAME - MODEL - STEM",
                                    output_format="wav",
                                    selected_stems=[ens_s_stem],
                                    add_settings={
                                        "add_single_sep_text_progress": f"{i} из {total_ensemble_models} (инверт.)"
                                    },
                                    progress=progress,
                                )
                                if result_seped_auto_ensemble_invert:
                                    for stem, path in result_seped_auto_ensemble_invert:
                                        if stem == ens_i_stem:
                                            i_stem = path
                                            ensemble_sources_list.append(path)

                    except Exception as e:
                        print(f"\nПроизошла ошибка при разделении: {e}")
                        progress(
                            0,
                            desc="Произошла ошибка при разделении, модель пропускается...",
                        )
                        continue
                    finally:
                        if s_stem:
                            success_separations.append((ens_mn, s_stem, i_stem, weight))

            ensemble_sources_stems = []
            ensemble_sources_invert_stems = []
            weights = []

            for out_mn, out_s_stem, out_i_stem, out_weight in success_separations:
                ensemble_sources_stems.append(out_s_stem)
                ensemble_sources_invert_stems.append(out_i_stem)
                weights.append(out_weight)

            auto_ensemble_invout_file = None
            auto_ensemble_invout_file_wav = None

            if not ensemble_sources_stems:
                return None, None, None, None, []
            auto_ensemble_output_name = f"ensembless_{namer.short(basename, length=50)}_{len(ensemble_sources_stems)}_{method}"
            auto_ensemble_inverted_output_name = f"ensembless_{namer.short(basename, length=50)}_{len(ensemble_sources_stems)}_{invert_methods_map[method]}_invert"
            auto_ensemble_out_file, auto_ensemble_out_file_wav = ensemble_audio_files(
                files=ensemble_sources_stems,
                weights=weights,
                output=os.path.join(o, auto_ensemble_output_name),
                ensemble_type=method,
                out_format=out_format,
                add_wav=True,
            )

            if invert_ensemble:
                if ensemble_sources_invert_stems:
                    auto_ensemble_invout_file, auto_ensemble_invout_file_wav = (
                        ensemble_audio_files(
                            files=ensemble_sources_invert_stems,
                            weights=invert_weights(weights),
                            output=os.path.join(o, auto_ensemble_inverted_output_name),
                            ensemble_type=invert_methods_map[method],
                            out_format=out_format,
                            add_wav=True,
                        )
                    )
            history.add(input_file, auto_ensemble_out_file, auto_ensemble_out_file_wav, auto_ensemble_invout_file, ensemble_sources_list, method, timestamp, ensemble_state)
            return (
                self.return_audio_with_size(value=auto_ensemble_out_file, label="Результат"),
                auto_ensemble_out_file_wav,
                self.return_audio_with_size(value=auto_ensemble_invout_file, label="Инверсия"),
                ensemble_sources_list,
            )

class ManualEnsembless(GradioHelper):
    def __init__(self, user_directory):
        super().__init__()
        self.user_directory = user_directory
    def UI(self):
        with gr.Row():
            with gr.Column():
                with gr.Group(visible=False) as add_ensemble_inputs:
                    input_ensemble_path = gr.Textbox(
                        label="Путь к входному файлу", interactive=True
                    )
                    add_ensemble_inputs_btn = gr.Button(
                        "Добавить файл", variant="primary"
                    )
                add_ensemble_path_btn = gr.Button(
                    "Добавить файл по пути", variant="secondary"
                )
                input_ensemble_files = gr.File(
                    label="Входное аудио",
                    interactive=True,
                    type="filepath",
                    file_count="multiple",
                )

            with gr.Column():

                @gr.render(inputs=[input_ensemble_files])
                def input_ensemble_files_fn(input_files):
                    check_ensemble_files_status = f"""Анализ входных файлов
---"""
                    hz_ = []
                    err_list = []
                    if input_files:
                        for file in input_files:
                            basename = os.path.splitext(os.path.basename(file))[0]
                            if os.path.exists(file):
                                if check(file):
                                    hz = get_sr(file)
                                    check_ensemble_files_status += (
                                        f"\n{basename} - \u2713 ({hz} hz)"
                                    )
                                    hz_.append(hz)
                                else:
                                    check_ensemble_files_status += (
                                        f"\n{basename} - Нет аудио"
                                    )
                                    err_list.append(file)
                            else:
                                check_ensemble_files_status += (
                                    f"\n{basename} - Файл не найден"
                                )
                                err_list.append(file)

                    check_ensemble_files_result = f"Действительных файлов: {len(hz_)}"

                    all_same = True

                    common_rate = None

                    for hz_hz in hz_:
                        if common_rate is None:
                            common_rate = hz_hz
                        elif common_rate != hz_hz:
                            all_same = False

                    if hz_ and len(hz_) > 1:
                        check_ensemble_files_result += (
                            "\nВсе действительные файлы имеют одинаковую частоту дискретизации"
                            if all_same
                            else "\nОшибка! Все действительные файлы имеют РАЗНУЮ частоту дискретизации"
                        )
                    else:
                        check_ensemble_files_result += "\nДля создания ансамбля нужно загрузить, как минимум - 2 файла, содержащие аудио"

                    check_ensemble_files_status += f"\n \n{check_ensemble_files_result}"

                    gr.Textbox(
                        container=False,
                        lines=len(check_ensemble_files_status.split("\n")),
                        interactive=False,
                        value=check_ensemble_files_status,
                    )

        weights = gr.Textbox(label="Веса", value="1.0,1.0")

        method = gr.Dropdown(
            label="Алгоритм склеивания",
            choices=["min_fft", "max_fft", "avg_fft", "median_fft"],
            value="avg_fft",
            filterable=False,
        )

        output_format = gr.Dropdown(
            label="Формат выходного файла",
            interactive=True,
            choices=output_formats,
            value="mp3",
            filterable=False,
        )

        output_manual_ensemble_filename = gr.Textbox(
            label="Имя выходного файла", value="ensemble", interactive=True
        )

        make_manual_ensemble_btn = gr.Button(
            value="Создать ансамбль", variant="primary"
        )

        manual_ensemble_output_audio = gr.Audio(
            label="Результат",
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
            input_files_list,
            method,
            out_format,
            o_filename,
            weights: str,
        ):
            timestamp = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
            o = os.path.join(self.user_directory.path, "manual_ensembless", f"ensembless_outputs_{timestamp}")
            os.makedirs(o, exist_ok=True)

            o_filename = namer.sanitize(o_filename)
            o_filename = namer.short(o_filename)

            output_file = ensemble_audio_files(
                files=input_files_list,
                output=os.path.join(o, o_filename),
                weights=[float(x) for x in weights.split(",")],
                ensemble_type=method,
                out_format=out_format,
            )
            return self.return_audio_with_size(value=output_file, label="Результат")

class Inverter_UI(GradioHelper):
    def __init__(self):
        super().__init__()
        self.inverter = Inverter()
    def UI(self):
        with gr.Group():
            with gr.Row():
                original_audio = gr.File(
                    label="Оригинал",
                    interactive=True,
                    type="filepath",
                    file_count="single",
                )
                stem_audio = gr.File(
                    label="Cтем, который будет вычтен из оригинала",
                    interactive=True,
                    type="filepath",
                    file_count="single",
                )
            with gr.Group():
                output_format = gr.Dropdown(
                    label="Формат выходного файла",
                    interactive=True,
                    choices=output_formats,
                    value="mp3",
                    filterable=False,
                )
                method = gr.Radio(
                    choices=["waveform", "spectrogram"],
                    label="Метод вычитания",
                    value="waveform",
                )
                btn = gr.Button("Вычесть")
        output_audio = gr.Audio(
            label="Инверсия",
            type="filepath",
            interactive=False,
            show_download_button=True,
        )

        @btn.click(
            inputs=[original_audio, stem_audio, method, output_format],
            outputs=[output_audio],
        )
        def invert_result_ensemble(input_file, output_file, method, out_format):
            if input_file and output_file:
                o_dir = tempfile.mkdtemp(suffix="_inverter")
                basename = os.path.splitext(os.path.basename(input_file))[0]
                output_path = os.path.join(
                    o_dir,
                    f"inverter_{namer.short(basename, length=50)}_{method}.{out_format}",
                )
                inverted = self.inverter.process_audio(
                    audio1_path=input_file,
                    audio2_path=output_file,
                    out_format=out_format,
                    method=method,
                    output_path=output_path,
                )
                return self.return_audio_with_size(value=inverted, label="Инверсия")
            else:
                return None

class PluginManager(Separator):
    plugins_dir = os.path.join(script_dir, "plugins")
    os.makedirs(plugins_dir, exist_ok=True)

    def restart_after_install_plugin(self):
        subprocess.Popen([os.sys.executable] + [os.path.join(script_dir, "app.py")] + sys.argv[1:])
        os._exit(0)

    def parse_plugins(self):
        for plugin_file in os.listdir(self.plugins_dir):
            if not plugin_file.endswith(".py") or plugin_file == "__init__.py":
                continue

            plugin_module_name = os.path.splitext(plugin_file)[0]

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
                        plugin_path = os.path.join(self.plugins_dir, plugin_file)
                        spec = importlib.util.spec_from_file_location(
                            plugin_module_name, plugin_path
                        )
                        plugin_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(plugin_module)

                plugin_class = getattr(plugin_module, "Plugin")

                plugin_instance = plugin_class()

                with gr.Tab(plugin_instance.name):
                    plugin_instance.UI()

            except Exception as e:
                print(f"Ошибка загрузки плагина {plugin_module_name}: {e}")
                continue

    def UI(self):
        with gr.Tab("Установка"):
            upload_plugins_files = gr.File(
                label="Загрузить плагины",
                file_types=[".py"],
                file_count="multiple",
                interactive=True,
            )
            install_plugins_btn = gr.Button("Установить", interactive=True)

            @install_plugins_btn.click(inputs=[upload_plugins_files])
            def upload_plugin_list(files):
                if not files:
                    return
                for file in files:
                    try:
                        if file.name.endswith(".py"):
                            shutil.copy(
                                file,
                                os.path.join(
                                    self.plugins_dir,
                                    os.path.basename(file).replace(" ", "_"),
                                ),
                            )
                    except Exception as e:
                        print(f"Ошибка копирования файла {file}: {e}")
                time.sleep(2)
                self.restart_after_install_plugin()


        self.parse_plugins()
