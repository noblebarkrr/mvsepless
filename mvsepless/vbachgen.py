import os
import subprocess
import gradio as gr
import soundfile as sf
import librosa
import numpy as np
from pedalboard import Pedalboard, Compressor, Reverb, Delay, NoiseGate, Chorus
import tempfile
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from functools import wraps

from audio import check, read, write, output_formats
from namer import Namer
from separator import Separator
from vbach import f0_methods, VbachModelManager
from gradio_helper import GradioHelper, tz
from i18n import _i18n, CURRENT_LANGUAGE


class VbachGen(Separator, GradioHelper):
    """Класс для генерации каверов с использованием Vbach"""
    
    def __init__(
        self, 
        model_manager: VbachModelManager, 
        input_files: List[str], 
        upload_files_func: callable, 
        user_directory: Any, 
        vbach_inference: callable, 
        device: str
    ) -> None:
        """
        Инициализация генератора каверов
        
        Args:
            model_manager: Менеджер моделей Vbach
            input_files: Список входных файлов
            upload_files_func: Функция загрузки файлов
            user_directory: Пользовательская директория
            vbach_inference: Функция инференса Vbach
            device: Устройство
        """
        super().__init__()
        self.device: str = device
        self.namer: Namer = Namer()
        self.processing_data: Dict[str, Any] = {}
        self.separation_stages: Dict[str, Dict[str, Any]] = {}
        self.conversion_cache: Dict[str, Dict[str, Any]] = {}
        self.vbach_model_manager: VbachModelManager = model_manager
        self.vbach_inference: callable = vbach_inference
        self.input_files: List[str] = input_files
        self.upload_files_func: callable = upload_files_func
        self.user_directory: Any = user_directory
        
        # Создаем базовую директорию для хранения результатов
        self.output_base_directory: str = os.path.join(self.user_directory.path, "output", "vbachgen")
        os.makedirs(self.output_base_directory, exist_ok=True)
        
        self.fairseq_embedders: List[str] = list(
            self.vbach_model_manager.huberts_fairseq_dict.keys()
        )
        self.transformers_embedders: List[str] = list(
            self.vbach_model_manager.huberts_transformers_dict.keys()
        )

    def get_output_directory(self, input_audio_path: str, stage: str, model_name: str = "") -> str:
        """
        Создает путь для сохранения файлов в постоянной директории
        
        Args:
            input_audio_path: Путь к входному аудио
            stage: Этап обработки
            model_name: Имя модели
        
        Returns:
            Путь к директории для сохранения
        """
        basename: str = os.path.splitext(os.path.basename(input_audio_path))[0]
        # Заменяем недопустимые символы в имени файла
        basename = "".join(c if c.isalnum() or c in " _-" else "_" for c in basename)
        
        timestamp: str = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
        
        if model_name:
            model_suffix: str = f"_{model_name}"
        else:
            model_suffix = ""
            
        output_dir: str = os.path.join(
            self.output_base_directory, 
            basename, 
            f"{timestamp}_{stage}{model_suffix}"
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def get_cache_key(self, params_dict: Dict[str, Any]) -> str:
        """
        Получить ключ кэша на основе параметров
        
        Args:
            params_dict: Словарь параметров
        
        Returns:
            Хеш-ключ
        """
        params_str: str = json.dumps(params_dict, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()

    def parse_voice_models_actual(self) -> List[str]:
        """
        Получить список голосовых моделей
        
        Returns:
            Список имен моделей
        """
        return self.vbach_model_manager.parse_voice_models()

    def find_file_from_stem(self, results: List[Tuple[str, str]], stem_names: List[str]) -> Optional[str]:
        """
        Найти файл по имени стема
        
        Args:
            results: Список результатов разделения
            stem_names: Список имен стемов для поиска
        
        Returns:
            Путь к файлу или None
        """
        for stem_name, stem_file in results:
            if stem_name in stem_names:
                return stem_file
        return None

    def extract_inst_voc(
        self, 
        input_audio: str, 
        model_name: str, 
        progress: Optional[gr.Progress] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Извлечь инструментал и вокал
        
        Args:
            input_audio: Путь к входному аудио
            model_name: Имя модели
            progress: Прогресс
        
        Returns:
            Кортеж (путь к инструменталу, путь к вокалу)
        """
        key: str = f"inst_voc_{hashlib.md5(input_audio.encode()).hexdigest()}_{model_name}"

        if key in self.separation_stages:
            print(_i18n("skip_separation", model=model_name))
            inst_file: Optional[str] = self.separation_stages[key].get("inst_file")
            voc_file: Optional[str] = self.separation_stages[key].get("voc_file")
            return inst_file, voc_file

        if progress:
            progress(0.2, desc=_i18n("extract_inst_voc_progress"))

        # Используем постоянную директорию вместо временной
        output_dir: str = self.get_output_directory(input_audio, "inst_voc", model_name)
        
        inst_output: List[Tuple[str, str]] = self.separate(
            input=input_audio,
            output_dir=output_dir,
            model_name=model_name,
            ext_inst=True,
            output_format="flac",
            template="VbachGen_NAME_STEM",
            add_settings={
                "add_single_sep_text_progress": _i18n("extract_inst_voc_progress")
            },
            progress=progress,
        )

        inst_file = self.find_file_from_stem(
            inst_output,
            [
                "Instrument",
                "instrument",
                "Instrumental",
                "instrumental",
                "other",
                "Other",
            ],
        )
        voc_file = self.find_file_from_stem(inst_output, ["Vocals", "vocals"])

        self.separation_stages[key] = {
            "inst_file": inst_file,
            "voc_file": voc_file,
            "model": model_name,
            "input_file": input_audio,
        }

        return inst_file, voc_file

    def extract_lead_back(
        self, 
        vocals_file: Optional[str], 
        model_name: str, 
        progress: Optional[gr.Progress] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Извлечь лид и бэк-вокал
        
        Args:
            vocals_file: Путь к файлу вокала
            model_name: Имя модели
            progress: Прогресс
        
        Returns:
            Кортеж (путь к лид-вокалу, путь к бэк-вокалу)
        """
        if not vocals_file:
            return None, None

        key: str = f"lead_back_{hashlib.md5(vocals_file.encode()).hexdigest()}_{model_name}"

        if key in self.separation_stages:
            print(_i18n("skip_lead_back_separation", model=model_name))
            lead_file: Optional[str] = self.separation_stages[key].get("lead_file")
            back_file: Optional[str] = self.separation_stages[key].get("back_file")
            return lead_file, back_file

        if progress:
            progress(0.4, desc=_i18n("extract_lead_back_progress"))

        # Используем постоянную директорию
        output_dir: str = self.get_output_directory(vocals_file, "lead_back", model_name)
        
        karaoke_output: List[Tuple[str, str]] = self.separate(
            input=vocals_file,
            output_dir=output_dir,
            model_name=model_name,
            ext_inst=True,
            output_format="flac",
            template="karaoke_NAME_STEM",
            add_settings={
                "add_single_sep_text_progress": _i18n("extract_lead_back_progress")
            },
            progress=progress,
        )

        back_file = self.find_file_from_stem(
            karaoke_output,
            ["Instrumental", "instrumental", "other", "Other", "back", "Back"],
        )
        lead_file = self.find_file_from_stem(
            karaoke_output, ["Vocals", "vocals", "karaoke", "lead", "Lead"]
        )

        self.separation_stages[key] = {
            "lead_file": lead_file,
            "back_file": back_file,
            "model": model_name,
            "input_file": vocals_file,
        }

        return lead_file, back_file

    def clear_vocals(
        self, 
        vocals_file: Optional[str], 
        model_name: str, 
        progress: Optional[gr.Progress] = None, 
        vocal_type: str = "vocals"
    ) -> Optional[str]:
        """
        Очистить вокал от реверберации/эха
        
        Args:
            vocals_file: Путь к файлу вокала
            model_name: Имя модели
            progress: Прогресс
            vocal_type: Тип вокала
        
        Returns:
            Путь к очищенному файлу или None
        """
        if not vocals_file:
            return None

        key: str = f"clear_{vocal_type}_{hashlib.md5(vocals_file.encode()).hexdigest()}_{model_name}"

        if key in self.separation_stages:
            print(_i18n("skip_clear", vocal_type=vocal_type, model=model_name))
            return self.separation_stages[key].get("cleared_file")

        if progress:
            progress(0.6, desc=_i18n("clear_vocals_progress", vocal_type=vocal_type))

        # Используем постоянную директорию
        output_dir: str = self.get_output_directory(vocals_file, f"clear_{vocal_type}", model_name)
        
        clear_output: List[Tuple[str, str]] = self.separate(
            input=vocals_file,
            output_dir=output_dir,
            model_name=model_name,
            ext_inst=True,
            output_format="flac",
            template="precleared_NAME_STEM",
            add_settings={"add_single_sep_text_progress": _i18n("clear_vocals_progress", vocal_type=vocal_type)},
            progress=progress,
        )

        cleared_file: Optional[str] = self.find_file_from_stem(
            clear_output, ["No Echo", "No Reverb", "Dry"]
        )

        self.separation_stages[key] = {
            "cleared_file": cleared_file,
            "model": model_name,
            "input_file": vocals_file,
            "vocal_type": vocal_type,
        }

        return cleared_file

    def separation_only(
        self,
        input_audio: str,
        anti_instrum_model: str,
        karaoke_model: str,
        dereverb_model: str,
        karaoke_check: bool,
        preclear_vocals_check: bool,
        progress: Optional[gr.Progress] = None,
    ) -> Dict[str, Any]:
        """
        Только разделение аудио
        
        Args:
            input_audio: Путь к входному аудио
            anti_instrum_model: Модель для извлечения инструментала
            karaoke_model: Модель для караоке
            dereverb_model: Модель для удаления реверберации
            karaoke_check: Разделить на лид/бэк
            preclear_vocals_check: Очистить вокал
            progress: Прогресс
        
        Returns:
            Словарь с результатами разделения
        """
        if progress is None:
            progress = gr.Progress(track_tqdm=True)

        progress(0, desc=_i18n("start_separation"))

        progress(0.1, desc=_i18n("check_previous_results"))

        inst_file: Optional[str] = None
        full_vocals_file: Optional[str] = None
        back_vocals_file: Optional[str] = None
        lead_vocals_file: Optional[str] = None

        inst_voc_key: str = f"inst_voc_{hashlib.md5(input_audio.encode()).hexdigest()}_{anti_instrum_model}"

        if inst_voc_key in self.separation_stages:
            print(_i18n("using_cached_inst_voc"))
            inst_file = self.separation_stages[inst_voc_key].get("inst_file")
            full_vocals_file = self.separation_stages[inst_voc_key].get("voc_file")
            progress(0.2, desc=_i18n("skip_inst_voc_extracted"))
        else:
            progress(0.2, desc=_i18n("extract_inst_voc_progress"))
            inst_file, full_vocals_file = self.extract_inst_voc(
                input_audio, anti_instrum_model, progress
            )

        if karaoke_check and full_vocals_file:
            lead_back_key: str = f"lead_back_{hashlib.md5(full_vocals_file.encode()).hexdigest()}_{karaoke_model}"

            if lead_back_key in self.separation_stages:
                print(_i18n("using_cached_lead_back"))
                lead_vocals_file = self.separation_stages[lead_back_key].get("lead_file")
                back_vocals_file = self.separation_stages[lead_back_key].get("back_file")
                progress(0.4, desc=_i18n("skip_lead_back_extracted"))
            else:
                progress(0.4, desc=_i18n("extract_lead_back_progress"))
                lead_vocals_file, back_vocals_file = self.extract_lead_back(
                    full_vocals_file, karaoke_model, progress
                )

        cleared_vocals: List[Tuple[str, str]] = []
        if preclear_vocals_check:
            if full_vocals_file:
                clear_key: str = f"clear_vocals_{hashlib.md5(full_vocals_file.encode()).hexdigest()}_{dereverb_model}"

                if clear_key in self.separation_stages:
                    print(_i18n("using_cached_clear_full"))
                    cleared_full_vocals: Optional[str] = self.separation_stages[clear_key].get("cleared_file")
                    if cleared_full_vocals:
                        cleared_vocals.append(("full_vocals", cleared_full_vocals))
                    progress(0.6, desc=_i18n("skip_clear_full"))
                else:
                    progress(0.6, desc=_i18n("clear_full_vocals_progress"))
                    cleared_full_vocals = self.clear_vocals(
                        full_vocals_file, dereverb_model, progress, vocal_type="vocals"
                    )
                    if cleared_full_vocals:
                        cleared_vocals.append(("full_vocals", cleared_full_vocals))

            if lead_vocals_file:
                clear_key = f"clear_lead_{hashlib.md5(lead_vocals_file.encode()).hexdigest()}_{dereverb_model}"

                if clear_key in self.separation_stages:
                    print(_i18n("using_cached_clear_lead"))
                    cleared_lead_vocals = self.separation_stages[clear_key].get("cleared_file")
                    if cleared_lead_vocals:
                        cleared_vocals.append(("lead_vocals", cleared_lead_vocals))
                else:
                    cleared_lead_vocals = self.clear_vocals(
                        lead_vocals_file, dereverb_model, progress, vocal_type="lead"
                    )
                    if cleared_lead_vocals:
                        cleared_vocals.append(("lead_vocals", cleared_lead_vocals))

            if back_vocals_file:
                clear_key = f"clear_back_{hashlib.md5(back_vocals_file.encode()).hexdigest()}_{dereverb_model}"

                if clear_key in self.separation_stages:
                    print(_i18n("using_cached_clear_back"))
                    cleared_back_vocals = self.separation_stages[clear_key].get("cleared_file")
                    if cleared_back_vocals:
                        cleared_vocals.append(("back_vocals", cleared_back_vocals))
                else:
                    cleared_back_vocals = self.clear_vocals(
                        back_vocals_file, dereverb_model, progress, vocal_type="back"
                    )
                    if cleared_back_vocals:
                        cleared_vocals.append(("back_vocals", cleared_back_vocals))

        list_vocals: List[Tuple[str, Optional[str]]] = [
            ("full_vocals", full_vocals_file),
            ("back_vocals", back_vocals_file),
            ("lead_vocals", lead_vocals_file),
        ]

        for cleared_name, cleared_file in cleared_vocals:
            for i, (name, file) in enumerate(list_vocals):
                if name == cleared_name:
                    list_vocals[i] = (name, cleared_file)
                    break

        generated_files: List[str] = []
        if inst_file:
            generated_files.append(inst_file)
        for name, file in list_vocals:
            if file:
                generated_files.append(file)

        progress(1.0, desc=_i18n("separation_complete"))

        # Создаем директорию для результатов разделения
        separation_dir: str = self.get_output_directory(input_audio, "separation_results")
        
        return {
            "inst_file": inst_file,
            "list_vocals": list_vocals,
            "temp_dir": separation_dir,
            "generated_files": generated_files,
        }

    def clear_separation_cache(self) -> None:
        """Очистить кэш разделения"""
        self.separation_stages.clear()
        print(_i18n("separation_cache_cleared"))

    def conversion_only(
        self,
        separation_result: Dict[str, Any],
        voice_name: str,
        conversion_mode: str,
        pitch1_val: float,
        pitch2_val: float,
        method_pitch: str,
        index_rate: float,
        fr: int,
        rms: float,
        protect: float,
        hop_mangio_crepe: int,
        f0_max: int,
        output_format: str,
        unconv_vocals_check: bool,
        use_effects: bool,
        instrumental_gain: float,
        vocal1_gain: float,
        vocal2_gain: float,
        echo_delay: float,
        echo_feedback: float,
        echo_mix: float,
        reverb_rm_size: float,
        reverb_width: float,
        reverb_wet: float,
        reverb_dry: float,
        reverb_damping: float,
        chorus_rate_hz: float,
        chorus_depth: float,
        chorus_centre_delay_ms: float,
        chorus_feedback: float,
        chorus_mix: float,
        compressor_ratio: float,
        compressor_threshold: float,
        compressor_attack: float,
        compressor_release: float,
        noise_gate_threshold: float,
        noise_gate_ratio: float,
        noise_gate_attack: float,
        noise_gate_release: float,
        embedder_name: Optional[str] = None,
        transformers_mode: bool = False,
        input_audio: Optional[str] = None,
        progress: Optional[gr.Progress] = None,
        always_new_conversion: bool = True,
    ) -> Dict[str, Any]:
        """
        Только преобразование голоса
        
        Args:
            separation_result: Результаты разделения
            voice_name: Имя голосовой модели
            conversion_mode: Режим преобразования
            pitch1_val: Высота тона для первого вокала
            pitch2_val: Высота тона для второго вокала
            method_pitch: Метод извлечения тона
            index_rate: Влияние индекса
            fr: Радиус фильтра
            rms: Огибающая громкости
            protect: Защита согласных
            hop_mangio_crepe: Длина шага
            f0_max: Максимальная частота F0
            output_format: Формат вывода
            unconv_vocals_check: Добавить непреобразованный вокал
            use_effects: Использовать эффекты
            instrumental_gain: Громкость инструментала
            vocal1_gain: Громкость первого вокала
            vocal2_gain: Громкость второго вокала
            echo_delay: Задержка эха
            echo_feedback: Обратная связь эха
            echo_mix: Смешение эха
            reverb_rm_size: Размер комнаты реверберации
            reverb_width: Ширина реверберации
            reverb_wet: Влажность реверберации
            reverb_dry: Сухость реверберации
            reverb_damping: Демпфирование реверберации
            chorus_rate_hz: Скорость хоруса
            chorus_depth: Глубина хоруса
            chorus_centre_delay_ms: Задержка центра хоруса
            chorus_feedback: Обратная связь хоруса
            chorus_mix: Смешение хоруса
            compressor_ratio: Соотношение компрессора
            compressor_threshold: Порог компрессора
            compressor_attack: Атака компрессора
            compressor_release: Спад компрессора
            noise_gate_threshold: Порог шумоподавления
            noise_gate_ratio: Соотношение шумоподавления
            noise_gate_attack: Атака шумоподавления
            noise_gate_release: Спад шумоподавления
            embedder_name: Имя эмбеддера
            transformers_mode: Режим transformers
            input_audio: Входное аудио
            progress: Прогресс
            always_new_conversion: Всегда создавать новое преобразование
        
        Returns:
            Словарь с результатами преобразования
        """
        conversion_params: Dict[str, Any] = {
            "voice_name": voice_name,
            "conversion_mode": conversion_mode,
            "pitch1_val": pitch1_val,
            "pitch2_val": pitch2_val,
            "method_pitch": method_pitch,
            "index_rate": index_rate,
            "fr": fr,
            "rms": rms,
            "protect": protect,
            "hop_mangio_crepe": hop_mangio_crepe,
            "f0_max": f0_max,
            "output_format": output_format,
            "unconv_vocals_check": unconv_vocals_check,
            "use_effects": use_effects,
            "instrumental_gain": instrumental_gain,
            "vocal1_gain": vocal1_gain,
            "vocal2_gain": vocal2_gain,
            "echo_delay": echo_delay,
            "echo_feedback": echo_feedback,
            "echo_mix": echo_mix,
            "reverb_rm_size": reverb_rm_size,
            "reverb_width": reverb_width,
            "reverb_wet": reverb_wet,
            "reverb_dry": reverb_dry,
            "reverb_damping": reverb_damping,
            "chorus_rate_hz": chorus_rate_hz,
            "chorus_depth": chorus_depth,
            "chorus_centre_delay_ms": chorus_centre_delay_ms,
            "chorus_feedback": chorus_feedback,
            "chorus_mix": chorus_mix,
            "compressor_ratio": compressor_ratio,
            "compressor_threshold": compressor_threshold,
            "compressor_attack": compressor_attack,
            "compressor_release": compressor_release,
            "noise_gate_threshold": noise_gate_threshold,
            "noise_gate_ratio": noise_gate_ratio,
            "noise_gate_attack": noise_gate_attack,
            "noise_gate_release": noise_gate_release,
            "embedder_name": embedder_name,
            "transformers_mode": transformers_mode,
        }
        cache_key: str = self.get_cache_key(conversion_params)

        if not always_new_conversion:
            if cache_key in self.conversion_cache:
                print(_i18n("using_cached_conversion"))
                return self.conversion_cache[cache_key]

        if progress is None:
            progress = gr.Progress(track_tqdm=True)

        progress(0, desc=_i18n("start_conversion"))

        inst_path: Optional[str] = separation_result.get("inst_file")
        list_vocals: List[Tuple[str, Optional[str]]] = separation_result.get("list_vocals", [])
        temp_dir: str = separation_result.get("temp_dir", "")

        rvc_params: Dict[str, Any] = {
            "model_name": voice_name,
            "pitch1": pitch1_val,
            "pitch2": pitch2_val,
            "f0_method": method_pitch,
            "index_rate": index_rate,
            "filter_radius": fr,
            "rms": rms,
            "protect": protect,
            "hop_length": hop_mangio_crepe,
            "f0_max": f0_max,
            "embedder_name": embedder_name,
            "transformers_mode": transformers_mode,
        }

        params: Dict[str, Any] = {"output_format": output_format, "conversion_mode": conversion_mode}

        mix_params: Dict[str, Any] = {
            "add_unconverted_vocals_to_instrumental": unconv_vocals_check,
            "use_effects": use_effects,
            "gain": {
                "instrum": instrumental_gain,
                "vocals1": vocal1_gain,
                "vocals2": vocal2_gain,
            },
            "pedalboard_settings": {
                "echo": {
                    "delay": echo_delay,
                    "feedback": echo_feedback,
                    "mix": echo_mix,
                },
                "reverb": {
                    "room_size": reverb_rm_size,
                    "wet": reverb_wet,
                    "dry": reverb_dry,
                    "damping": reverb_damping,
                    "width": reverb_width,
                },
                "compressor": {
                    "ratio": compressor_ratio,
                    "threshold": compressor_threshold,
                    "attack": compressor_attack,
                    "release": compressor_release,
                },
                "noise_gate": {
                    "threshold": noise_gate_threshold,
                    "ratio": noise_gate_ratio,
                    "attack": noise_gate_attack,
                    "release": noise_gate_release,
                },
                "chorus": {
                    "rate": chorus_rate_hz,
                    "depth": chorus_depth,
                    "center_delay": chorus_centre_delay_ms,
                    "feedback": chorus_feedback,
                    "mix": chorus_mix,
                },
            },
        }

        progress(0.3, desc=_i18n("converting_vocals"))

        converted_vocals_list: List[str] = []
        
        # Создаем директорию для преобразованных файлов
        if input_audio:
            conversion_dir: str = self.get_output_directory(input_audio, "converted", voice_name)
        else:
            conversion_dir = tempfile.mkdtemp()
        
        full_vocals_file: Optional[str] = next(
            (f[1] for f in list_vocals if f[0] == "full_vocals"), None
        )
        back_vocals_file = next(
            (f[1] for f in list_vocals if f[0] == "back_vocals"), None
        )
        lead_vocals_file = next(
            (f[1] for f in list_vocals if f[0] == "lead_vocals"), None
        )

        stack: str = "transformers" if transformers_mode else "fairseq"

        if conversion_mode == "full" and full_vocals_file:
            full_vocals_converted_path: Optional[str] = self.vbach_inference(
                input_file=full_vocals_file,
                output_dir=conversion_dir,
                model_name=rvc_params["model_name"],
                format_name=False,
                output_name=f"full_vocals_converted-{self.namer.short(os.path.splitext(os.path.basename(input_audio))[0], length=60)}" if input_audio else "full_vocals_converted",
                pitch=rvc_params["pitch1"],
                method_pitch=rvc_params["f0_method"],
                output_bitrate=320,
                output_format="flac",
                pipeline_mode="alt",
                embedder_name=embedder_name,
                stack=stack,
                add_params={
                    "index_rate": rvc_params["index_rate"],
                    "filter_radius": rvc_params["filter_radius"],
                    "protect": rvc_params["protect"],
                    "rms": rvc_params["rms"],
                    "mangio_crepe_hop_length": rvc_params["hop_length"],
                    "f0_min": 50,
                    "f0_max": rvc_params["f0_max"],
                    "stereo_mode": "mono",
                },
                device=self.device
            )
            if full_vocals_converted_path:
                converted_vocals_list.append(full_vocals_converted_path)

        elif conversion_mode == "lead/back" and lead_vocals_file and back_vocals_file:
            lead_vocals_converted_path = self.vbach_inference(
                input_file=lead_vocals_file,
                output_dir=conversion_dir,
                model_name=rvc_params["model_name"],
                format_name=False,
                output_name=f"lead_vocals_converted-{self.namer.short(os.path.splitext(os.path.basename(input_audio))[0], length=60)}" if input_audio else "lead_vocals_converted",
                pitch=rvc_params["pitch1"],
                method_pitch=rvc_params["f0_method"],
                output_bitrate=320,
                output_format="flac",
                pipeline_mode="alt",
                embedder_name=embedder_name,
                stack=stack,
                add_params={
                    "index_rate": rvc_params["index_rate"],
                    "filter_radius": rvc_params["filter_radius"],
                    "protect": rvc_params["protect"],
                    "rms": rvc_params["rms"],
                    "mangio_crepe_hop_length": rvc_params["hop_length"],
                    "f0_min": 50,
                    "f0_max": rvc_params["f0_max"],
                    "stereo_mode": "mono",
                },
                device=self.device
            )

            back_vocals_converted_path = self.vbach_inference(
                input_file=back_vocals_file,
                output_dir=conversion_dir,
                model_name=rvc_params["model_name"],
                format_name=False,
                output_name=f"back_vocals_converted-{self.namer.short(os.path.splitext(os.path.basename(input_audio))[0], length=60)}" if input_audio else "back_vocals_converted",
                pitch=rvc_params["pitch2"],
                method_pitch=rvc_params["f0_method"],
                output_bitrate=320,
                output_format="flac",
                pipeline_mode="alt",
                embedder_name=embedder_name,
                stack=stack,
                add_params={
                    "index_rate": rvc_params["index_rate"],
                    "filter_radius": rvc_params["filter_radius"],
                    "protect": rvc_params["protect"],
                    "rms": rvc_params["rms"],
                    "mangio_crepe_hop_length": rvc_params["hop_length"],
                    "f0_min": 50,
                    "f0_max": rvc_params["f0_max"],
                    "stereo_mode": "mono",
                },
                device=self.device
            )

            if back_vocals_converted_path:
                converted_vocals_list.append(back_vocals_converted_path)
            if lead_vocals_converted_path:
                converted_vocals_list.append(lead_vocals_converted_path)

        elif conversion_mode == "back" and back_vocals_file:
            back_vocals_converted_path = self.vbach_inference(
                input_file=back_vocals_file,
                output_dir=conversion_dir,
                model_name=rvc_params["model_name"],
                format_name=False,
                output_name=f"back_vocals_converted-{self.namer.short(os.path.splitext(os.path.basename(input_audio))[0], length=60)}" if input_audio else "back_vocals_converted",
                pitch=rvc_params["pitch2"],
                method_pitch=rvc_params["f0_method"],
                output_bitrate=320,
                output_format="flac",
                pipeline_mode="alt",
                embedder_name=embedder_name,
                stack=stack,
                add_params={
                    "index_rate": rvc_params["index_rate"],
                    "filter_radius": rvc_params["filter_radius"],
                    "protect": rvc_params["protect"],
                    "rms": rvc_params["rms"],
                    "mangio_crepe_hop_length": rvc_params["hop_length"],
                    "f0_min": 50,
                    "f0_max": rvc_params["f0_max"],
                    "stereo_mode": "mono",
                },
                device=self.device
            )
            if back_vocals_converted_path:
                converted_vocals_list.append(back_vocals_converted_path)

        elif conversion_mode == "lead" and lead_vocals_file:
            lead_vocals_converted_path = self.vbach_inference(
                input_file=lead_vocals_file,
                output_dir=conversion_dir,
                model_name=rvc_params["model_name"],
                format_name=False,
                output_name=f"lead_vocals_converted-{self.namer.short(os.path.splitext(os.path.basename(input_audio))[0], length=60)}" if input_audio else "lead_vocals_converted",
                pitch=rvc_params["pitch1"],
                method_pitch=rvc_params["f0_method"],
                output_bitrate=320,
                output_format="flac",
                pipeline_mode="alt",
                embedder_name=embedder_name,
                stack=stack,
                add_params={
                    "index_rate": rvc_params["index_rate"],
                    "filter_radius": rvc_params["filter_radius"],
                    "protect": rvc_params["protect"],
                    "rms": rvc_params["rms"],
                    "mangio_crepe_hop_length": rvc_params["hop_length"],
                    "f0_min": 50,
                    "f0_max": rvc_params["f0_max"],
                    "stereo_mode": "mono",
                },
                device=self.device
            )
            if lead_vocals_converted_path:
                converted_vocals_list.append(lead_vocals_converted_path)

        generated_files: List[str] = []
        if inst_path:
            generated_files.append(inst_path)
        for name, file in list_vocals:
            if file:
                generated_files.append(file)
        generated_files.extend(converted_vocals_list)

        self.processing_data = {
            "inst_path": inst_path,
            "list_vocals": list_vocals,
            "converted_vocals_list": converted_vocals_list,
            "params": params,
            "rvc_params": rvc_params,
            "input_audio": input_audio,
        }

        progress(0.7, desc=_i18n("mixing_final_cover"))

        if input_audio:
            final_path: Optional[str] = self.mix_and_save(
                inst_path,
                list_vocals,
                converted_vocals_list,
                mix_params,
                params,
                rvc_params,
                temp_dir,
                input_audio,
            )
        else:
            final_path = None

        if final_path:
            generated_files.append(final_path)

        result: Dict[str, Any] = {
            "generated_files": generated_files,
            "final_path": final_path,
            "converted_vocals_list": converted_vocals_list,
        }
        self.conversion_cache[cache_key] = result

        progress(1.0, desc=_i18n("conversion_complete"))

        return result

    def mix_and_save(
        self,
        inst_path: Optional[str],
        list_vocals: List[Tuple[str, Optional[str]]],
        converted_vocals_list: List[str],
        mix_params: Dict[str, Any],
        params: Dict[str, Any],
        rvc_params: Dict[str, Any],
        temp_dir: str,
        input_audio: Optional[str],
    ) -> Optional[str]:
        """
        Смешать и сохранить финальный кавер
        
        Args:
            inst_path: Путь к инструменталу
            list_vocals: Список вокалов
            converted_vocals_list: Список преобразованных вокалов
            mix_params: Параметры сведения
            params: Параметры
            rvc_params: Параметры RVC
            temp_dir: Временная директория
            input_audio: Входное аудио
        
        Returns:
            Путь к финальному файлу или None
        """
        final_audio: Optional[np.ndarray] = None
        samplerate: int = 44100
        
        # Функция для выравнивания длин аудио
        def align_audio_length(audio1: Optional[np.ndarray], audio2: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            """
            Выровнять длины двух аудио массивов
            
            Args:
                audio1: Первый аудио массив
                audio2: Второй аудио массив
            
            Returns:
                Кортеж выровненных массивов
            """
            if audio1 is None:
                return audio2, None
            if audio2 is None:
                return audio1, None
                
            # Получаем минимальную длину
            min_len: int = min(audio1.shape[1], audio2.shape[1])
            
            # Обрезаем оба массива до минимальной длины
            if audio1.shape[1] > min_len:
                audio1 = audio1[:, :min_len]
            if audio2.shape[1] > min_len:
                audio2 = audio2[:, :min_len]
                
            return audio1, audio2

        if inst_path and os.path.exists(inst_path):
            inst_data, samplerate = read(
                path=inst_path, mono=False, sr=None, dtype="float32"
            )
            inst_gain: float = 10 ** (mix_params["gain"]["instrum"] / 20.0)
            inst_data *= inst_gain
            final_audio = inst_data.copy()

        if mix_params["add_unconverted_vocals_to_instrumental"]:
            full_vocals_file = next(
                (f[1] for f in list_vocals if f[0] == "full_vocals"), None
            )
            back_vocals_file = next(
                (f[1] for f in list_vocals if f[0] == "back_vocals"), None
            )
            lead_vocals_file = next(
                (f[1] for f in list_vocals if f[0] == "lead_vocals"), None
            )

            conversion_mode: str = params.get("conversion_mode", "full")
            
            if conversion_mode == "lead" and back_vocals_file and os.path.exists(back_vocals_file):
                back_vocals, sr = read(
                    path=back_vocals_file, mono=False, sr=samplerate, dtype="float32"
                )
                back_vocals *= (10 ** (mix_params["gain"]["vocals2"] / 20.0))
                
                if final_audio is None:
                    final_audio = back_vocals
                else:
                    # Выравниваем длины перед сложением
                    final_audio, back_vocals = align_audio_length(final_audio, back_vocals)
                    if final_audio is not None and back_vocals is not None:
                        final_audio = final_audio + back_vocals
                    
            elif conversion_mode == "back" and lead_vocals_file and os.path.exists(lead_vocals_file):
                lead_vocals, sr = read(
                    path=lead_vocals_file, mono=False, sr=samplerate, dtype="float32"
                )
                lead_vocals *= (10 ** (mix_params["gain"]["vocals1"] / 20.0))
                
                if final_audio is None:
                    final_audio = lead_vocals
                else:
                    # Выравниваем длины перед сложением
                    final_audio, lead_vocals = align_audio_length(final_audio, lead_vocals)
                    if final_audio is not None and lead_vocals is not None:
                        final_audio = final_audio + lead_vocals
                    
            elif conversion_mode == "lead/back":
                # В режиме lead/back НЕ добавляем непреобразованные вокалы
                # так как мы преобразовываем и лид, и бэк отдельно
                pass

        # Определяем порядок вокалов в зависимости от режима преобразования
        conversion_mode = params.get("conversion_mode", "full")
        
        for i, vocal_path in enumerate(converted_vocals_list):
            if not vocal_path or not os.path.exists(vocal_path):
                continue

            vocal_data, sr = read(
                path=vocal_path, mono=False, sr=samplerate, dtype="float32"
            )

            if mix_params.get("use_effects", False):
                board = Pedalboard()
                effects = mix_params.get("pedalboard_settings", {})

                if "compressor" in effects:
                    comp = effects["compressor"]
                    board.append(
                        Compressor(
                            ratio=comp["ratio"],
                            threshold_db=comp["threshold"],
                            attack_ms=comp["attack"],
                            release_ms=comp["release"],
                        )
                    )

                if "noise_gate" in effects:
                    ng = effects["noise_gate"]
                    board.append(
                        NoiseGate(
                            threshold_db=ng["threshold"],
                            ratio=ng["ratio"],
                            attack_ms=ng["attack"],
                            release_ms=ng["release"],
                        )
                    )

                if "echo" in effects:
                    echo = effects["echo"]
                    board.append(
                        Delay(
                            delay_seconds=echo["delay"],
                            feedback=echo["feedback"],
                            mix=echo["mix"],
                        )
                    )

                if "reverb" in effects:
                    rev = effects["reverb"]
                    board.append(
                        Reverb(
                            room_size=rev["room_size"],
                            dry_level=rev["dry"],
                            wet_level=rev["wet"],
                            damping=rev["damping"],
                            width=rev["width"],
                        )
                    )

                if "chorus" in effects:
                    chorus = effects["chorus"]
                    board.append(
                        Chorus(
                            rate_hz=chorus["rate"],
                            depth=chorus["depth"],
                            centre_delay_ms=chorus["center_delay"],
                            feedback=chorus["feedback"],
                            mix=chorus["mix"],
                        )
                    )

                vocal_data = board(vocal_data, samplerate)

            # Определяем какой gain использовать для каждого вокала
            if conversion_mode == "lead/back":
                # В режиме lead/back первый вокал в списке - бэк-вокал, второй - лид-вокал
                # Это соответствует порядку в методе conversion_only
                if "back_vocals_converted" in os.path.basename(vocal_path) or i == 0:
                    gain_db = mix_params["gain"]["vocals2"]  # Бэк-вокал
                else:
                    gain_db = mix_params["gain"]["vocals1"]  # Лид-вокал
            else:
                # Для других режимов используем стандартную логику
                gain_db = (
                    mix_params["gain"]["vocals1"]
                    if i == 0
                    else mix_params["gain"]["vocals2"]
                )
            
            vocal_data *= 10 ** (gain_db / 20.0)

            if final_audio is None:
                final_audio = vocal_data.copy()
            else:
                # Выравниваем длины перед сложением
                final_audio, vocal_data = align_audio_length(final_audio, vocal_data)
                if final_audio is not None and vocal_data is not None:
                    final_audio = final_audio + vocal_data

        # Убеждаемся, что final_audio не None
        if final_audio is None:
            # Создаем пустой аудиофайл с минимальной длиной
            final_audio = np.zeros((2, samplerate), dtype=np.float32)

        # Нормализация
        max_amplitude = np.max(np.abs(final_audio))
        if max_amplitude > 0:
            final_audio = final_audio / max_amplitude
            
        filename: str = (
            f"{rvc_params['model_name']} - {self.namer.short(os.path.splitext(os.path.basename(input_audio))[0], length=60)}.{params['output_format']}"
            if input_audio
            else f"remixed.{params['output_format']}"
        )
        
        # Сохраняем в постоянной директории
        if input_audio:
            final_dir: str = self.get_output_directory(input_audio, "final_cover")
            final_path: str = os.path.join(final_dir, filename)
        else:
            final_path = os.path.join(temp_dir, filename)
        
        final_path = write(
            final_path,
            final_audio,
            samplerate,
            "320k",
        )

        return final_path

    def gen_cover(
        self,
        input_audio: Optional[str],
        anti_instrum_model: str,
        karaoke_model: str,
        dereverb_model: str,
        output_format: str,
        karaoke_check: bool,
        conversion_mode: str,
        preclear_vocals_check: bool,
        voice_name: str,
        pitch1_val: float,
        pitch2_val: float,
        method_pitch: str,
        index_rate: float,
        fr: int,
        rms: float,
        protect: float,
        hop_mangio_crepe: int,
        f0_max: int,
        unconv_vocals_check: bool,
        use_effects: bool,
        instrumental_gain: float,
        vocal1_gain: float,
        vocal2_gain: float,
        echo_delay: float,
        echo_feedback: float,
        echo_mix: float,
        reverb_rm_size: float,
        reverb_width: float,
        reverb_wet: float,
        reverb_dry: float,
        reverb_damping: float,
        chorus_rate_hz: float,
        chorus_depth: float,
        chorus_centre_delay_ms: float,
        chorus_feedback: float,
        chorus_mix: float,
        compressor_ratio: float,
        compressor_threshold: float,
        compressor_attack: float,
        compressor_release: float,
        noise_gate_threshold: float,
        noise_gate_ratio: float,
        noise_gate_attack: float,
        noise_gate_release: float,
        embedder_name: str,
        transformers_mode: bool,
    ) -> Tuple[List[str], Optional[Dict]]:
        """
        Сгенерировать кавер
        
        Args:
            input_audio: Входное аудио
            anti_instrum_model: Модель для извлечения инструментала
            karaoke_model: Модель для караоке
            dereverb_model: Модель для удаления реверберации
            output_format: Формат вывода
            karaoke_check: Разделить на лид/бэк
            conversion_mode: Режим преобразования
            preclear_vocals_check: Очистить вокал
            voice_name: Имя голосовой модели
            pitch1_val: Высота тона для первого вокала
            pitch2_val: Высота тона для второго вокала
            method_pitch: Метод извлечения тона
            index_rate: Влияние индекса
            fr: Радиус фильтра
            rms: Огибающая громкости
            protect: Защита согласных
            hop_mangio_crepe: Длина шага
            f0_max: Максимальная частота F0
            unconv_vocals_check: Добавить непреобразованный вокал
            use_effects: Использовать эффекты
            instrumental_gain: Громкость инструментала
            vocal1_gain: Громкость первого вокала
            vocal2_gain: Громкость второго вокала
            echo_delay: Задержка эха
            echo_feedback: Обратная связь эха
            echo_mix: Смешение эха
            reverb_rm_size: Размер комнаты реверберации
            reverb_width: Ширина реверберации
            reverb_wet: Влажность реверберации
            reverb_dry: Сухость реверберации
            reverb_damping: Демпфирование реверберации
            chorus_rate_hz: Скорость хоруса
            chorus_depth: Глубина хоруса
            chorus_centre_delay_ms: Задержка центра хоруса
            chorus_feedback: Обратная связь хоруса
            chorus_mix: Смешение хоруса
            compressor_ratio: Соотношение компрессора
            compressor_threshold: Порог компрессора
            compressor_attack: Атака компрессора
            compressor_release: Спад компрессора
            noise_gate_threshold: Порог шумоподавления
            noise_gate_ratio: Соотношение шумоподавления
            noise_gate_attack: Атака шумоподавления
            noise_gate_release: Спад шумоподавления
            embedder_name: Имя эмбеддера
            transformers_mode: Режим transformers
        
        Returns:
            Кортеж (список сгенерированных файлов, аудио для отображения)
        """
        if not input_audio:
            raise gr.Error(_i18n("upload_audio_first"))

        if not voice_name:
            raise gr.Error(_i18n("select_model_first"))

        progress = gr.Progress(track_tqdm=True)
        progress(0, desc=_i18n("start_processing"))

        progress(0.1, desc=_i18n("separation_stage"))
        separation_result = self.separation_only(
            input_audio=input_audio,
            anti_instrum_model=anti_instrum_model,
            karaoke_model=karaoke_model,
            dereverb_model=dereverb_model,
            karaoke_check=karaoke_check,
            preclear_vocals_check=preclear_vocals_check,
            progress=progress,
        )

        progress(0.5, desc=_i18n("conversion_mixing_stage"))
        conversion_result = self.conversion_only(
            separation_result=separation_result,
            voice_name=voice_name,
            conversion_mode=conversion_mode,
            pitch1_val=pitch1_val,
            pitch2_val=pitch2_val,
            method_pitch=method_pitch,
            index_rate=index_rate,
            fr=fr,
            rms=rms,
            protect=protect,
            hop_mangio_crepe=hop_mangio_crepe,
            f0_max=f0_max,
            output_format=output_format,
            unconv_vocals_check=unconv_vocals_check,
            use_effects=use_effects,
            instrumental_gain=instrumental_gain,
            vocal1_gain=vocal1_gain,
            vocal2_gain=vocal2_gain,
            echo_delay=echo_delay,
            echo_feedback=echo_feedback,
            echo_mix=echo_mix,
            reverb_rm_size=reverb_rm_size,
            reverb_width=reverb_width,
            reverb_wet=reverb_wet,
            reverb_dry=reverb_dry,
            reverb_damping=reverb_damping,
            chorus_rate_hz=chorus_rate_hz,
            chorus_depth=chorus_depth,
            chorus_centre_delay_ms=chorus_centre_delay_ms,
            chorus_feedback=chorus_feedback,
            chorus_mix=chorus_mix,
            compressor_ratio=compressor_ratio,
            compressor_threshold=compressor_threshold,
            compressor_attack=compressor_attack,
            compressor_release=compressor_release,
            noise_gate_threshold=noise_gate_threshold,
            noise_gate_ratio=noise_gate_ratio,
            noise_gate_attack=noise_gate_attack,
            noise_gate_release=noise_gate_release,
            embedder_name=embedder_name,
            transformers_mode=transformers_mode,
            input_audio=input_audio,
            progress=progress,
        )

        return conversion_result["generated_files"], self.return_audio_with_size(value=conversion_result.get("final_path"), label=_i18n("final_result"))

    def regenerate_conversion(
        self,
        voice_name: str,
        conversion_mode: str,
        pitch1_val: float,
        pitch2_val: float,
        method_pitch: str,
        index_rate: float,
        fr: int,
        rms: float,
        protect: float,
        hop_mangio_crepe: int,
        f0_max: int,
        output_format: str,
        unconv_vocals_check: bool,
        use_effects: bool,
        instrumental_gain: float,
        vocal1_gain: float,
        vocal2_gain: float,
        echo_delay: float,
        echo_feedback: float,
        echo_mix: float,
        reverb_rm_size: float,
        reverb_width: float,
        reverb_wet: float,
        reverb_dry: float,
        reverb_damping: float,
        chorus_rate_hz: float,
        chorus_depth: float,
        chorus_centre_delay_ms: float,
        chorus_feedback: float,
        chorus_mix: float,
        compressor_ratio: float,
        compressor_threshold: float,
        compressor_attack: float,
        compressor_release: float,
        noise_gate_threshold: float,
        noise_gate_ratio: float,
        noise_gate_attack: float,
        noise_gate_release: float,
        embedder_name: str,
        transformers_mode: bool,
    ) -> Tuple[List[str], Optional[Dict]]:
        """
        Перегенерировать преобразование
        
        Args:
            voice_name: Имя голосовой модели
            conversion_mode: Режим преобразования
            pitch1_val: Высота тона для первого вокала
            pitch2_val: Высота тона для второго вокала
            method_pitch: Метод извлечения тона
            index_rate: Влияние индекса
            fr: Радиус фильтра
            rms: Огибающая громкости
            protect: Защита согласных
            hop_mangio_crepe: Длина шага
            f0_max: Максимальная частота F0
            output_format: Формат вывода
            unconv_vocals_check: Добавить непреобразованный вокал
            use_effects: Использовать эффекты
            instrumental_gain: Громкость инструментала
            vocal1_gain: Громкость первого вокала
            vocal2_gain: Громкость второго вокала
            echo_delay: Задержка эха
            echo_feedback: Обратная связь эха
            echo_mix: Смешение эха
            reverb_rm_size: Размер комнаты реверберации
            reverb_width: Ширина реверберации
            reverb_wet: Влажность реверберации
            reverb_dry: Сухость реверберации
            reverb_damping: Демпфирование реверберации
            chorus_rate_hz: Скорость хоруса
            chorus_depth: Глубина хоруса
            chorus_centre_delay_ms: Задержка центра хоруса
            chorus_feedback: Обратная связь хоруса
            chorus_mix: Смешение хоруса
            compressor_ratio: Соотношение компрессора
            compressor_threshold: Порог компрессора
            compressor_attack: Атака компрессора
            compressor_release: Спад компрессора
            noise_gate_threshold: Порог шумоподавления
            noise_gate_ratio: Соотношение шумоподавления
            noise_gate_attack: Атака шумоподавления
            noise_gate_release: Спад шумоподавления
            embedder_name: Имя эмбеддера
            transformers_mode: Режим transformers
        
        Returns:
            Кортеж (список сгенерированных файлов, аудио для отображения)
        """
        if not self.processing_data:
            raise gr.Error(_i18n("generate_first"))

        progress = gr.Progress(track_tqdm=True)
        progress(0, desc=_i18n("regenerating_conversion"))

        separation_result: Dict[str, Any] = {
            "inst_file": self.processing_data.get("inst_path"),
            "list_vocals": self.processing_data.get("list_vocals", []),
            "temp_dir": tempfile.mkdtemp(),
        }

        conversion_result = self.conversion_only(
            separation_result=separation_result,
            voice_name=voice_name,
            conversion_mode=conversion_mode,
            pitch1_val=pitch1_val,
            pitch2_val=pitch2_val,
            method_pitch=method_pitch,
            index_rate=index_rate,
            fr=fr,
            rms=rms,
            protect=protect,
            hop_mangio_crepe=hop_mangio_crepe,
            f0_max=f0_max,
            output_format=output_format,
            unconv_vocals_check=unconv_vocals_check,
            use_effects=use_effects,
            instrumental_gain=instrumental_gain,
            vocal1_gain=vocal1_gain,
            vocal2_gain=vocal2_gain,
            echo_delay=echo_delay,
            echo_feedback=echo_feedback,
            echo_mix=echo_mix,
            reverb_rm_size=reverb_rm_size,
            reverb_width=reverb_width,
            reverb_wet=reverb_wet,
            reverb_dry=reverb_dry,
            reverb_damping=reverb_damping,
            chorus_rate_hz=chorus_rate_hz,
            chorus_depth=chorus_depth,
            chorus_centre_delay_ms=chorus_centre_delay_ms,
            chorus_feedback=chorus_feedback,
            chorus_mix=chorus_mix,
            compressor_ratio=compressor_ratio,
            compressor_threshold=compressor_threshold,
            compressor_attack=compressor_attack,
            compressor_release=compressor_release,
            noise_gate_threshold=noise_gate_threshold,
            noise_gate_ratio=noise_gate_ratio,
            noise_gate_attack=noise_gate_attack,
            noise_gate_release=noise_gate_release,
            embedder_name=embedder_name,
            transformers_mode=transformers_mode,
            input_audio=self.processing_data.get("input_audio"),
            progress=progress,
        )

        return conversion_result["generated_files"], self.return_audio_with_size(value=conversion_result.get("final_path"), label=_i18n("final_result"))

    def remix_cover(
        self,
        use_effects: bool,
        instrumental_gain: float,
        vocal1_gain: float,
        vocal2_gain: float,
        echo_delay: float,
        echo_feedback: float,
        echo_mix: float,
        reverb_rm_size: float,
        reverb_width: float,
        reverb_wet: float,
        reverb_dry: float,
        reverb_damping: float,
        chorus_rate_hz: float,
        chorus_depth: float,
        chorus_centre_delay_ms: float,
        chorus_feedback: float,
        chorus_mix: float,
        compressor_ratio: float,
        compressor_threshold: float,
        compressor_attack: float,
        compressor_release: float,
        noise_gate_threshold: float,
        noise_gate_ratio: float,
        noise_gate_attack: float,
        noise_gate_release: float,
    ) -> Optional[Dict]:
        """
        Пересвести кавер
        
        Args:
            use_effects: Использовать эффекты
            instrumental_gain: Громкость инструментала
            vocal1_gain: Громкость первого вокала
            vocal2_gain: Громкость второго вокала
            echo_delay: Задержка эха
            echo_feedback: Обратная связь эха
            echo_mix: Смешение эха
            reverb_rm_size: Размер комнаты реверберации
            reverb_width: Ширина реверберации
            reverb_wet: Влажность реверберации
            reverb_dry: Сухость реверберации
            reverb_damping: Демпфирование реверберации
            chorus_rate_hz: Скорость хоруса
            chorus_depth: Глубина хоруса
            chorus_centre_delay_ms: Задержка центра хоруса
            chorus_feedback: Обратная связь хоруса
            chorus_mix: Смешение хоруса
            compressor_ratio: Соотношение компрессора
            compressor_threshold: Порог компрессора
            compressor_attack: Атака компрессора
            compressor_release: Спад компрессора
            noise_gate_threshold: Порог шумоподавления
            noise_gate_ratio: Соотношение шумоподавления
            noise_gate_attack: Атака шумоподавления
            noise_gate_release: Спад шумоподавления
        
        Returns:
            Аудио для отображения
        """
        if not self.processing_data:
            raise gr.Error(_i18n("generate_cover_first"))

        data: Dict[str, Any] = self.processing_data
        temp_dir: str = tempfile.mkdtemp()

        mix_params: Dict[str, Any] = {
            "add_unconverted_vocals_to_instrumental": True,
            "use_effects": use_effects,
            "gain": {
                "instrum": instrumental_gain,
                "vocals1": vocal1_gain,
                "vocals2": vocal2_gain,
            },
            "pedalboard_settings": {
                "echo": {
                    "delay": echo_delay,
                    "feedback": echo_feedback,
                    "mix": echo_mix,
                },
                "reverb": {
                    "room_size": reverb_rm_size,
                    "wet": reverb_wet,
                    "dry": reverb_dry,
                    "damping": reverb_damping,
                    "width": reverb_width,
                },
                "compressor": {
                    "ratio": compressor_ratio,
                    "threshold": compressor_threshold,
                    "attack": compressor_attack,
                    "release": compressor_release,
                },
                "noise_gate": {
                    "threshold": noise_gate_threshold,
                    "ratio": noise_gate_ratio,
                    "attack": noise_gate_attack,
                    "release": noise_gate_release,
                },
                "chorus": {
                    "rate": chorus_rate_hz,
                    "depth": chorus_depth,
                    "center_delay": chorus_centre_delay_ms,
                    "feedback": chorus_feedback,
                    "mix": chorus_mix,
                },
            },
        }

        final_path: Optional[str] = self.mix_and_save(
            data.get("inst_path"),
            data.get("list_vocals", []),
            data.get("converted_vocals_list", []),
            mix_params,
            data.get("params", {}),
            data.get("rvc_params", {}),
            temp_dir,
            data.get("input_audio"),
        )

        return self.return_audio_with_size(value=final_path, label=_i18n("final_result"))

    def UI(self) -> None:
        """Создать пользовательский интерфейс"""
        with gr.Row(equal_height=False, variant="panel"):
            with gr.Column():
                with gr.Group():
                    upload = gr.File(show_label=False, type="filepath", interactive=True)
                    refresh_input_btn = gr.Button(_i18n("refresh"), variant="primary", interactive=True)
                    list_input_files = gr.Dropdown(
                        label=_i18n("select_input_files"),
                        choices=reversed(self.input_files) if self.input_files else [],
                        value=None,
                        multiselect=False,
                        interactive=True,
                        filterable=False,
                        scale=15
                    )
                    
                    gr.on(
                        fn=lambda: gr.update(choices=reversed(self.input_files) if self.input_files else [], value=None), 
                        outputs=list_input_files, 
                        trigger_mode="once"
                    )
                    
                    refresh_input_btn.click(
                        lambda: gr.update(choices=reversed(self.input_files) if self.input_files else [], value=None), 
                        outputs=list_input_files
                    )
                        
                    @upload.upload(inputs=[upload], outputs=[list_input_files, upload])
                    def upload_files(input_file: str) -> Tuple[gr.update, gr.update]:
                        files = self.upload_files_func([input_file])
                        return (
                            gr.update(choices=reversed(self.input_files) if self.input_files else [], value=files[0] if files else None),
                            gr.update(value=None)
                        )

            with gr.Column():
                with gr.Group():
                    model_name = gr.Dropdown(
                        label=_i18n("model_name"), 
                        interactive=True, 
                        filterable=False, 
                        scale=6
                    )
                    model_update_btn = gr.Button(
                        _i18n("refresh"), 
                        variant="primary", 
                        scale=3, 
                        size="lg"
                    )
                    
                    with gr.Column(variant="panel"):
                        with gr.Tab(_i18n("tab_separation")):
                            with gr.Group():
                                preclear_vocals_check = gr.Checkbox(
                                    label=_i18n("clear_vocals_reverb"), 
                                    value=False
                                )
                                karaoke_check = gr.Checkbox(
                                    label=_i18n("split_lead_back"), 
                                    value=False
                                )

                                with gr.Group() as extract_vocals_group:
                                    anti_instrum_model = gr.Dropdown(
                                        label=_i18n("vocal_model"),
                                        choices=self.get_list_mn_from_category(
                                            [_i18n("instrumental"), _i18n("vocals"), _i18n("instrumental_vocals")],
                                            [
                                                "mel_band_roformer",
                                                "bs_roformer",
                                                "mdx23c",
                                                "mdxnet",
                                                "htdemucs",
                                            ],
                                        ),
                                        interactive=True,
                                        filterable=False,
                                    )

                                with gr.Group(visible=False) as deecho_group:
                                    dereverb_model = gr.Dropdown(
                                        label=_i18n("dereverb_model"),
                                        choices=self.get_list_mn_from_category(
                                            [_i18n("reverb_echo"), _i18n("reverb"), _i18n("echo")], ["vr"]
                                        ),
                                        interactive=True,
                                        filterable=False,
                                    )

                                with gr.Group(visible=False) as karaoke_group:
                                    karaoke_model = gr.Dropdown(
                                        label=_i18n("karaoke_model"),
                                        choices=self.get_list_mn_from_category([_i18n("karaoke")]),
                                        interactive=True,
                                        filterable=False,
                                    )

                                with gr.Group(visible=False):
                                    separate_only_btn = gr.Button(
                                        _i18n("separate_only"), variant="primary"
                                    )
                                    clear_cache_btn = gr.Button(
                                        _i18n("clear_separation_cache"), variant="stop", size="sm"
                                    )
                                separation_status = gr.Textbox(
                                    label=_i18n("separation_status"), interactive=False, visible=False
                                )

                        with gr.Tab(_i18n("voice_conversion_settings")):
                            with gr.Group():
                                conversion_mode = gr.Dropdown(
                                    label=_i18n("conversion_mode"),
                                    choices=["lead", "back", "lead/back", "full"],
                                    value="full",
                                    filterable=False,
                                    visible=False,
                                    info=_i18n("conversion_mode_info"),
                                )
                                with gr.Row():
                                    pitch1 = gr.Slider(
                                        -48,
                                        48,
                                        value=0,
                                        step=12,
                                        label=_i18n("vocal_pitch"),
                                        interactive=True,
                                    )
                                    pitch2 = gr.Slider(
                                        -48,
                                        48,
                                        value=0,
                                        step=12,
                                        label=_i18n("back_vocal_pitch"),
                                        visible=False,
                                        interactive=True,
                                    )
                                with gr.Row():
                                    method_pitch = gr.Dropdown(
                                        label=_i18n("f0_method"),
                                        choices=f0_methods,
                                        value=f0_methods[0] if f0_methods else "rmvpe+",
                                        interactive=True,
                                        filterable=False,
                                    )
                                    f0_max = gr.Slider(
                                        50,
                                        2000,
                                        value=1100,
                                        step=50,
                                        label=_i18n("f0_max_limit"),
                                        interactive=True,
                                    )
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        index_rate = gr.Slider(
                                            0,
                                            1,
                                            value=0,
                                            step=0.05,
                                            label=_i18n("index_rate"),
                                            interactive=True,
                                        )
                                        fr = gr.Slider(
                                            0,
                                            7,
                                            value=3,
                                            step=1,
                                            label=_i18n("filter_radius"),
                                            interactive=True,
                                        )
                                    with gr.Column(scale=1):
                                        rms = gr.Slider(
                                            0,
                                            1,
                                            value=0.25,
                                            step=0.05,
                                            label=_i18n("rms_envelope"),
                                            interactive=True,
                                        )
                                        protect = gr.Slider(
                                            minimum=0,
                                            maximum=0.5,
                                            step=0.01,
                                            value=0.33,
                                            label=_i18n("protect"),
                                            interactive=True,
                                        )
                                hop_mangio_crepe = gr.Slider(
                                    8,
                                    512,
                                    value=128,
                                    step=8,
                                    label=_i18n("hop_length"),
                                    interactive=True,
                                    visible=False,
                                )
                                with gr.Accordion(label=_i18n("embedder"), open=False):
                                    with gr.Group():
                                        embedder_name = gr.Radio(
                                            label=_i18n("hubert_model"),
                                            choices=self.fairseq_embedders,
                                            value=self.fairseq_embedders[0] if self.fairseq_embedders else None,
                                        )
                                        transformers_mode = gr.Checkbox(
                                            label=_i18n("use_transformers"),
                                            value=False,
                                            interactive=True,
                                        )

                                        @transformers_mode.change(
                                            inputs=[transformers_mode], outputs=[embedder_name]
                                        )
                                        def change_embedders(tr_m: bool) -> gr.update:
                                            if tr_m:
                                                return gr.update(
                                                    value=self.transformers_embedders[0] if self.transformers_embedders else None,
                                                    choices=self.transformers_embedders,
                                                )
                                            else:
                                                return gr.update(
                                                    choices=self.fairseq_embedders,
                                                    value=self.fairseq_embedders[0] if self.fairseq_embedders else None,
                                                )

                        with gr.Tab(_i18n("mixing_settings")):
                            gr.Markdown(f"<center>{_i18n('volume_adjustment')}</center>", container=True)
                            with gr.Group():
                                vocal1_gain = gr.Slider(
                                    -60,
                                    60,
                                    value=-3,
                                    step=1,
                                    label=_i18n("vocals"),
                                    scale=3,
                                    interactive=True,
                                )
                                vocal2_gain = gr.Slider(
                                    -60,
                                    60,
                                    value=-3,
                                    step=1,
                                    label=_i18n("back_vocals"),
                                    scale=3,
                                    visible=False,
                                    interactive=True,
                                )
                                instrumental_gain = gr.Slider(
                                    -60,
                                    60,
                                    value=0,
                                    step=1,
                                    label=_i18n("instrumental"),
                                    scale=3,
                                    interactive=True,
                                )

                                output_format = gr.Dropdown(
                                    label=_i18n("output_format"),
                                    choices=output_formats,
                                    value=output_formats[0] if output_formats else "wav",
                                    interactive=True,
                                    filterable=False,
                                )
                                unconv_vocals_check = gr.Checkbox(
                                    label=_i18n("add_unconverted_vocals"),
                                    visible=False,
                                )
                                use_effects = gr.Checkbox(
                                    label=_i18n("add_effects"), value=True
                                )
                                with gr.Column(variant="panel", visible=True) as effects_accordion:
                                    with gr.Tab(_i18n("effects")):
                                        with gr.Tab(_i18n("echo")):
                                            with gr.Group():
                                                with gr.Row():
                                                    echo_delay = gr.Slider(
                                                        0,
                                                        3,
                                                        value=0,
                                                        label=_i18n("delay_time"),
                                                        interactive=True,
                                                    )
                                                    echo_feedback = gr.Slider(
                                                        0,
                                                        1,
                                                        value=0,
                                                        label=_i18n("feedback"),
                                                        interactive=True,
                                                    )
                                                    echo_mix = gr.Slider(
                                                        0,
                                                        1,
                                                        value=0,
                                                        label=_i18n("mix"),
                                                        interactive=True,
                                                    )

                                        with gr.Tab(_i18n("reverb")):
                                            with gr.Group():
                                                with gr.Row():
                                                    reverb_rm_size = gr.Slider(
                                                        0,
                                                        1,
                                                        value=0.1,
                                                        label=_i18n("room_size"),
                                                        interactive=True,
                                                    )
                                                    reverb_width = gr.Slider(
                                                        0,
                                                        1,
                                                        value=1.0,
                                                        label=_i18n("reverb_width"),
                                                        interactive=True,
                                                    )
                                                with gr.Row():
                                                    reverb_wet = gr.Slider(
                                                        0,
                                                        1,
                                                        value=0.3,
                                                        label=_i18n("wet_level"),
                                                        interactive=True,
                                                    )
                                                    reverb_dry = gr.Slider(
                                                        0,
                                                        1,
                                                        value=0.8,
                                                        label=_i18n("dry_level"),
                                                        interactive=True,
                                                    )
                                                with gr.Row():
                                                    reverb_damping = gr.Slider(
                                                        0,
                                                        1,
                                                        value=0.9,
                                                        label=_i18n("damping"),
                                                        interactive=True,
                                                    )

                                        with gr.Tab(_i18n("chorus")):
                                            with gr.Group():
                                                with gr.Row():
                                                    chorus_rate_hz = gr.Slider(
                                                        0,
                                                        10,
                                                        value=0,
                                                        label=_i18n("chorus_rate"),
                                                        interactive=True,
                                                    )
                                                    chorus_depth = gr.Slider(
                                                        0,
                                                        1,
                                                        value=0,
                                                        label=_i18n("chorus_depth"),
                                                        interactive=True,
                                                    )
                                                with gr.Row():
                                                    chorus_centre_delay_ms = gr.Slider(
                                                        0,
                                                        50,
                                                        value=0,
                                                        label=_i18n("center_delay"),
                                                        interactive=True,
                                                    )
                                                    chorus_feedback = gr.Slider(
                                                        0,
                                                        1,
                                                        value=0,
                                                        label=_i18n("feedback"),
                                                        interactive=True,
                                                    )
                                                with gr.Row():
                                                    chorus_mix = gr.Slider(
                                                        0,
                                                        1,
                                                        value=0,
                                                        label=_i18n("mix"),
                                                        interactive=True,
                                                    )

                                    with gr.Tab(_i18n("processing")):
                                        with gr.Tab(_i18n("compressor")):
                                            with gr.Group():
                                                with gr.Row():
                                                    compressor_ratio = gr.Slider(
                                                        1,
                                                        50,
                                                        value=16,
                                                        label=_i18n("ratio"),
                                                        interactive=True,
                                                    )
                                                    compressor_threshold = gr.Slider(
                                                        -60,
                                                        0,
                                                        value=-16,
                                                        label=_i18n("threshold"),
                                                        interactive=True,
                                                    )
                                                    compressor_attack = gr.Slider(
                                                        0,
                                                        2000,
                                                        value=40,
                                                        label=_i18n("attack_ms"),
                                                        interactive=True,
                                                    )
                                                    compressor_release = gr.Slider(
                                                        0,
                                                        2000,
                                                        value=100,
                                                        label=_i18n("release_ms"),
                                                        interactive=True,
                                                    )

                                        with gr.Tab(_i18n("noise_gate")):
                                            with gr.Group():
                                                with gr.Row():
                                                    noise_gate_threshold = gr.Slider(
                                                        -60,
                                                        0,
                                                        value=-40,
                                                        label=_i18n("threshold"),
                                                        interactive=True,
                                                    )
                                                    noise_gate_ratio = gr.Slider(
                                                        1,
                                                        20,
                                                        value=8,
                                                        label=_i18n("ratio"),
                                                        interactive=True,
                                                    )
                                                with gr.Row():
                                                    noise_gate_attack = gr.Slider(
                                                        0,
                                                        100,
                                                        value=10,
                                                        label=_i18n("attack_ms"),
                                                        interactive=True,
                                                    )
                                                    noise_gate_release = gr.Slider(
                                                        0,
                                                        1000,
                                                        value=100,
                                                        label=_i18n("release_ms"),
                                                        interactive=True,
                                                    )
                        with gr.Tab(_i18n("intermediate_files")):
                            with gr.Group():
                                generated_files_list = gr.Files(
                                    label=_i18n("intermediate_files"), 
                                    interactive=False, 
                                    type="filepath", 
                                    show_label=False
                                )
                final_ai_cover = gr.Audio(
                    label=_i18n("final_result"),
                    interactive=False,
                    show_download_button=True,
                )
                with gr.Group():
                    with gr.Row(equal_height=True):
                        generate_btn = gr.Button(_i18n("generate_cover"), variant="primary")
                        regenerate_btn = gr.Button(
                            _i18n("regenerate_vocals"), variant="secondary"
                        )
                        remix_btn = gr.Button(_i18n("remix_cover"), variant="huggingface")

        status_text = gr.Textbox(label=_i18n("status"), interactive=False, visible=False)

        method_pitch.change(
            fn=lambda x: gr.update(
                visible=(
                    True
                    if x in ["mangio-crepe", "mangio-crepe-tiny", "pyin"]
                    else False
                )
            ),
            inputs=method_pitch,
            outputs=hop_mangio_crepe,
        )

        @model_update_btn.click(inputs=None, outputs=model_name)
        def update_voice_models() -> gr.update:
            models = self.parse_voice_models_actual()
            first_model = models[0] if models else None
            return gr.update(choices=models, value=first_model)

        @gr.on(
            inputs=None, 
            outputs=model_name
        )
        def update_voice_models() -> gr.update:
            models = self.parse_voice_models_actual()
            first_model = models[0] if models else None
            return gr.update(choices=models, value=first_model)

        use_effects.change(
            fn=lambda x: gr.update(visible=x),
            inputs=use_effects,
            outputs=effects_accordion,
        )

        karaoke_check.change(
            fn=lambda x: gr.update(visible=x),
            inputs=karaoke_check,
            outputs=karaoke_group,
        ).then(
            fn=lambda x: gr.update(value="full", visible=x),
            inputs=karaoke_check,
            outputs=conversion_mode,
        ).then(
            fn=lambda x: gr.update(
                visible=True if x in ["back", "lead"] else False, value=False
            ),
            inputs=conversion_mode,
            outputs=unconv_vocals_check,
        )

        preclear_vocals_check.change(
            fn=lambda x: gr.update(visible=x),
            inputs=preclear_vocals_check,
            outputs=deecho_group,
        )

        conversion_mode.change(
            fn=lambda mode: (
                gr.update(visible=mode in ["lead", "lead/back"]),
                gr.update(visible=mode in ["back", "lead/back"]),
                gr.update(visible=mode in ["lead/back"]),
            ),
            inputs=conversion_mode,
            outputs=[vocal1_gain, vocal2_gain, pitch2],
        ).then(
            fn=lambda x: gr.update(
                visible=True if x in ["back", "lead"] else False, value=False
            ),
            inputs=conversion_mode,
            outputs=unconv_vocals_check,
        )

        separate_only_btn.click(
            fn=lambda audio, a_model, k_model, d_model, k_check, p_check: (
                (
                    self.separation_only(
                        input_audio=audio,
                        anti_instrum_model=a_model,
                        karaoke_model=k_model,
                        dereverb_model=d_model,
                        karaoke_check=k_check,
                        preclear_vocals_check=p_check,
                    )["generated_files"],
                    _i18n("separation_complete"),
                )
                if audio and a_model
                else (gr.update(), _i18n("upload_and_select_model"))
            ),
            inputs=[
                list_input_files,
                anti_instrum_model,
                karaoke_model,
                dereverb_model,
                karaoke_check,
                preclear_vocals_check,
            ],
            outputs=[generated_files_list, separation_status],
        )

        clear_cache_btn.click(
            fn=lambda: (self.clear_separation_cache(), _i18n("separation_cache_cleared")),
            inputs=None,
            outputs=[separation_status],
        )

        generate_btn.click(
            fn=self.gen_cover,
            inputs=[
                list_input_files,
                anti_instrum_model,
                karaoke_model,
                dereverb_model,
                output_format,
                karaoke_check,
                conversion_mode,
                preclear_vocals_check,
                model_name,
                pitch1,
                pitch2,
                method_pitch,
                index_rate,
                fr,
                rms,
                protect,
                hop_mangio_crepe,
                f0_max,
                unconv_vocals_check,
                use_effects,
                instrumental_gain,
                vocal1_gain,
                vocal2_gain,
                echo_delay,
                echo_feedback,
                echo_mix,
                reverb_rm_size,
                reverb_width,
                reverb_wet,
                reverb_dry,
                reverb_damping,
                chorus_rate_hz,
                chorus_depth,
                chorus_centre_delay_ms,
                chorus_feedback,
                chorus_mix,
                compressor_ratio,
                compressor_threshold,
                compressor_attack,
                compressor_release,
                noise_gate_threshold,
                noise_gate_ratio,
                noise_gate_attack,
                noise_gate_release,
                embedder_name,
                transformers_mode,
            ],
            outputs=[generated_files_list, final_ai_cover],
        )

        regenerate_btn.click(
            fn=self.regenerate_conversion,
            inputs=[
                model_name,
                conversion_mode,
                pitch1,
                pitch2,
                method_pitch,
                index_rate,
                fr,
                rms,
                protect,
                hop_mangio_crepe,
                f0_max,
                output_format,
                unconv_vocals_check,
                use_effects,
                instrumental_gain,
                vocal1_gain,
                vocal2_gain,
                echo_delay,
                echo_feedback,
                echo_mix,
                reverb_rm_size,
                reverb_width,
                reverb_wet,
                reverb_dry,
                reverb_damping,
                chorus_rate_hz,
                chorus_depth,
                chorus_centre_delay_ms,
                chorus_feedback,
                chorus_mix,
                compressor_ratio,
                compressor_threshold,
                compressor_attack,
                compressor_release,
                noise_gate_threshold,
                noise_gate_ratio,
                noise_gate_attack,
                noise_gate_release,
                embedder_name,
                transformers_mode,
            ],
            outputs=[generated_files_list, final_ai_cover],
        )

        remix_btn.click(
            fn=self.remix_cover,
            inputs=[
                use_effects,
                instrumental_gain,
                vocal1_gain,
                vocal2_gain,
                echo_delay,
                echo_feedback,
                echo_mix,
                reverb_rm_size,
                reverb_width,
                reverb_wet,
                reverb_dry,
                reverb_damping,
                chorus_rate_hz,
                chorus_depth,
                chorus_centre_delay_ms,
                chorus_feedback,
                chorus_mix,
                compressor_ratio,
                compressor_threshold,
                compressor_attack,
                compressor_release,
                noise_gate_threshold,
                noise_gate_ratio,
                noise_gate_attack,
                noise_gate_release,
            ],
            outputs=[final_ai_cover],
        )