import os
import sys
import json
import subprocess
import argparse
import gradio as gr
import yaml
import numpy as np
import tempfile
import shutil
import traceback
import torch
from packaging import version
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
from pathlib import Path

script_dir: str = os.path.dirname(os.path.abspath(__file__))
env: Dict[str, str] = os.environ.copy()
env['PYTHONPATH'] = script_dir

from gradio_helper import dw_file, all_ids, set_device, cuda_available
from audio import check, output_formats, multiread, write, ensemble, read, subtractor, split_mid_side, get_duration_from_array
from namer import Namer
from i18n import _i18n, CURRENT_LANGUAGE, set_language

is_pytorch2: bool = version.parse(torch.__version__) >= version.parse("2.0.0")
is_pytorch2_4: bool = version.parse(torch.__version__) >= version.parse("2.4.0")
unsupported_models: List[str] = ["bs_inst_fno_unwa", "mbr_wsa"] if not is_pytorch2 else ["bs_inst_fno_unwa"] if not is_pytorch2_4 else []
MVSEPLESS_ECONOM: bool = not cuda_available
MVSEPLESS_ECONOM_SEGMENT: int = int(os.environ.get("MVSEPLESS_ECO_SEG", "7"))


def get_files_from_list(input_paths: Union[str, List[str]], only_files: bool = False) -> List[str]:
    """
    Получить список аудиофайлов из переданных путей
    
    Args:
        input_paths: Путь к файлу или директории или список путей
        only_files: Только файлы (не рекурсивно)
    
    Returns:
        Список путей к аудиофайлам
    """
    input_list: List[str] = []
    
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    
    for path in input_paths:
        if os.path.isdir(path):
            if not only_files:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        full_path: str = os.path.join(root, file)
                        if check(full_path):
                            input_list.append(full_path)
        elif os.path.isfile(path):
            if check(path):
                input_list.append(path)
        else:
            pass
    return input_list


def format_end_count_models(count: int) -> str:
    """
    Форматирование окончания для слова "модель" в зависимости от числа
    
    Args:
        count: Количество моделей
    
    Returns:
        Окончание слова
    """
    if CURRENT_LANGUAGE == "ru":
        if count % 10 == 1 and count % 100 != 11:
            return "ь"  # 1 модель, 21 модель, 101 модель
        elif (count % 10 in [2, 3, 4]) and (count % 100 not in [12, 13, 14]):
            return "и"  # 2-4 модели, 22-24 модели
        else:
            return "ей"  # 5-20 моделей, 25-30 моделей и т.д.
    else:
        return "s" if count != 1 else ""


class MvseplessModelManager:
    """Менеджер моделей для MVSepless"""
    
    def __init__(
        self,
        models_info_path: str = os.path.join(script_dir, "models.json"),
        cache_dir: str = os.path.join(script_dir, "mvsepless_models_cache"),
    ) -> None:
        """
        Инициализация менеджера моделей
        
        Args:
            models_info_path: Путь к файлу с информацией о моделях
            cache_dir: Директория для кэша моделей
        """
        self.models_cache_dir: str = cache_dir
        self.models_info_path: str = models_info_path
        
        with open(self.models_info_path, "r", encoding="utf-8") as f:
            models_info: Dict = json.load(f)
        self.models_info: Dict = models_info

    def get_mt(self, model_name: str) -> str:
        """
        Получить тип модели
        
        Args:
            model_name: Имя модели
        
        Returns:
            Тип модели
        """
        return self.models_info.get(model_name, {}).get("model_type", "")

    def get_mn(self) -> List[str]:
        """
        Получить список всех доступных моделей
        
        Returns:
            Список имен моделей
        """
        return [mn for mn in self.models_info if mn not in unsupported_models]

    def get_stems(self, model_name: Optional[str]) -> List[str]:
        """
        Получить список стемов для модели
        
        Args:
            model_name: Имя модели
        
        Returns:
            Список стемов
        """
        if model_name is not None and model_name != "":
            return [
                stem
                for stem in self.models_info
                .get(model_name, {})
                .get("stems", [])
            ]
        else:
            return []
        
    def get_id(self, model_name: str) -> int:
        """
        Получить ID модели
        
        Args:
            model_name: Имя модели
        
        Returns:
            ID модели
        """
        if model_name is not None and model_name != "":
            return self.models_info.get(model_name, {}).get("id", 0)
        else:
            return 0

    def get_tgt_inst(self, model_name: str) -> Optional[str]:
        """
        Получить целевой инструмент модели
        
        Args:
            model_name: Имя модели
        
        Returns:
            Название целевого инструмента или None
        """
        if model_name is not None and model_name != "":
            return (
                self.models_info
                .get(model_name, {})
                .get("target_instrument", None)
            )
        else:
            return None

    def get_category(self, model_name: str) -> str:
        """
        Получить категорию модели
        
        Args:
            model_name: Имя модели
        
        Returns:
            Категория модели
        """
        if model_name is not None and model_name != "":
            return self.models_info.get(model_name, {}).get("category", "")
        else:
            return ""

    def calculate_dimensions(self, chunk_size: int, hop_length: int = 441) -> Tuple[int, int]:
        """
        Рассчитать размерности для чанка
        
        Args:
            chunk_size: Размер чанка
            hop_length: Длина шага
        
        Returns:
            Кортеж (dim_t, actual_chunk_size)
        """
        dim_t: int = (chunk_size // hop_length) + 1
        actual_chunk_size: int = (dim_t - 1) * hop_length
        return dim_t, actual_chunk_size

    def generate_econom_params(self, sr: int = 44100, seconds: int = MVSEPLESS_ECONOM_SEGMENT, hop_length: int = 441) -> Tuple[int, int]:
        """
        Сгенерировать параметры для эконом-режима
        
        Args:
            sr: Частота дискретизации
            seconds: Длительность в секундах
            hop_length: Длина шага
        
        Returns:
            Кортеж (dim_t, chunk_size)
        """
        chunk_size: int = sr * seconds
        dim_t, chunk_size = self.calculate_dimensions(chunk_size, hop_length)
        return dim_t, chunk_size

    def get_list_supported_models(
        self, 
        limit: Optional[int] = None, 
        stem: Optional[str] = None, 
        model_types: Optional[List[str]] = None, 
        category: Optional[List[str]] = None, 
        only_installed: bool = False
    ) -> None:
        """
        Вывести список поддерживаемых моделей с фильтрацией
        
        Args:
            limit: Ограничение количества
            stem: Фильтр по стему
            model_types: Фильтр по типам моделей
            category: Фильтр по категориям
            only_installed: Только установленные модели
        """
        models: List[str] = self.get_list_mn_from_category(category, model_types)
        if not models:
            return
            
        installed_models: List[str] = [model for model in models if self.install_model(model, only_check=True)]
        
        if stem and stem != "":
            models = [
                model for model in models 
                if (stem in self.get_stems(model) or 
                    stem.lower() in self.get_stems(model) or 
                    stem.upper() in self.get_stems(model) or 
                    stem.capitalize() in self.get_stems(model) or 
                    stem.title() in self.get_stems(model))
            ]
        
        if limit:
            models = models[:limit]
            
        f_key: str = _i18n("model_name")
        s_key: str = _i18n("output_stems")

        filename_width: int = max(len(f_key), max(len(model) for model in models)) + 2
        stems_width: int = max(len(s_key), max(len(", ".join(self.get_stems(model))) for model in models)) + 1
        
        print("|-", "-" * filename_width, "---", "-" * stems_width, "-|", sep="")

        if only_installed:
            print(f"| {_i18n('installed_models'):<{filename_width + stems_width}}    |")
        else:
            print(f"| {_i18n('installed_marker'):<{filename_width + stems_width}}    |")
        print(f"| {_i18n('target_stem_marker'):<{filename_width + stems_width}}    |")
        print("|-", "-" * filename_width, "---", "-" * stems_width, "-|", sep="")
        
        if category:
            print(f"| {_i18n('categories')}:{'':<{filename_width + stems_width - len(_i18n('categories')) - 2}}     |")
            for c in category:
                print(f"|  - {c:<{filename_width + stems_width - 4}}     |")
        else:
            print(f"| {_i18n('categories_all')}:{'':<{filename_width + stems_width - len(_i18n('categories_all')) - 2}}     |")

        if model_types:
            print(f"| {_i18n('model_types')}:{'':<{filename_width + stems_width - len(_i18n('model_types')) - 2}}     |")
            for mt_ in model_types:
                print(f"|  - {mt_:<{filename_width + stems_width - 4}}     |")
        else:
            print(f"| {_i18n('model_types_all')}:{'':<{filename_width + stems_width - len(_i18n('model_types_all')) - 2}}     |")

        if stem and stem != "":
             print(f"| {_i18n('selected_stem')}: {stem:<{filename_width + stems_width - len(_i18n('selected_stem')) - 3}}     |")
             
        print("|-", "-" * filename_width, "-+-", "-" * stems_width, "-|", sep="")
        print(f"| {f_key:<{filename_width}} | {s_key:<{stems_width}} |")
        print("|-", "-" * filename_width, "-+-", "-" * stems_width, "-|", sep="")
        
        if only_installed:
            if installed_models:
                for model in installed_models:
                    stems_list: List[str] = self.get_stems(model)
                    stems_str: str = ", ".join([
                        _stem + '*' if _stem == self.get_tgt_inst(model) else _stem 
                        for _stem in stems_list
                    ])
                    print(f"| {model:<{filename_width}} | {stems_str:<{stems_width}} |")
                    print("|-", "-" * filename_width, "-+-", "-" * stems_width, "-|", sep="")
            else:
                print(f"| {'n/a':<{filename_width}} | {'n/a':<{stems_width}} |")
                print("|-", "-" * filename_width, "-+-", "-" * stems_width, "-|", sep="")
        else:
            if models:
                for model in models:
                    stems_list: List[str] = self.get_stems(model)
                    stems_str: str = ", ".join([
                        _stem + '*' if _stem == self.get_tgt_inst(model) else _stem 
                        for _stem in stems_list
                    ])
                    if model in installed_models:
                        print(f"| {model+' ✔':<{filename_width}} | {stems_str:<{stems_width}} |")
                    else:
                        print(f"| {model:<{filename_width}} | {stems_str:<{stems_width}} |")
                    print("|-", "-" * filename_width, "-+-", "-" * stems_width, "-|", sep="")
            else:
                print(f"| {'n/a':<{filename_width}} | {'n/a':<{stems_width}} |")
                print("|-", "-" * filename_width, "-+-", "-" * stems_width, "-|", sep="")

    def get_list_mn_from_category(
        self, 
        category: Optional[Union[str, List[str]]] = None, 
        model_type: Optional[List[str]] = None
    ) -> List[str]:
        """
        Получить список моделей по категориям и типам
        
        Args:
            category: Категория или список категорий
            model_type: Список типов моделей
        
        Returns:
            Список имен моделей
        """
        list_models: List[str] = []
        categories: List[str] = []
        
        if category:
            if isinstance(category, str) and category != "":
                categories.append(category)
            elif isinstance(category, list):
                categories.extend(category)
                
            if categories:
                if not model_type:
                    list_models = [
                        model for model in self.get_mn() 
                        if self.get_category(model) in category
                    ]
                else:
                    list_models = [
                        model for model in self.get_mn() 
                        if self.get_category(model) in category and self.get_mt(model) in model_type
                    ]
            else:
                if not model_type:
                    list_models = [model for model in self.get_mn()]
                else:
                    list_models = [
                        model for model in self.get_mn() 
                        if self.get_mt(model) in model_type
                    ]
        else:
            if not model_type:
                list_models = [model for model in self.get_mn()]
            else:
                list_models = [
                    model for model in self.get_mn() 
                    if self.get_mt(model) in model_type
                ]
                
        return list_models

    def get_list_categories(self) -> None:
        """Вывести список категорий моделей"""
        categories: List[str] = self.get_categories()
        categories_with_count: List[List[Union[str, int]]] = [
            [cat__, len([m__ for m__ in self.get_mn() if self.get_category(m__) == cat__])] 
            for cat__ in categories
        ]
        
        f_key: str = _i18n("categories")
        s_key: str = _i18n("models_count")
        
        category_width: int = max(len(f_key), max(len(c_) for c_ in categories))
        models_count_width: int = max([
            len(_c1 + " " + s_key.format(count=_n, end=format_end_count_models(_n))) 
            for (_c1, _n) in categories_with_count
        ])
        
        print("|-", "-" * models_count_width, "-|", sep="")
        print(f"| {f_key:<{models_count_width}} |")
        print("|-", "-" * models_count_width, "-|", sep="")
        
        if categories_with_count:
            for (cat_, num_) in categories_with_count:
                print(f"| {cat_+' '+_i18n('models_count').format(count=num_, end=format_end_count_models(num_)):<{models_count_width}} |")
                print("|-", "-" * models_count_width, "-|", sep="")
        else:
            print(f"| {'n/a':<{models_count_width}} |")
            print("|-", "-" * models_count_width, "-|", sep="")

    def get_list_model_types(self) -> None:
        """Вывести список типов моделей"""
        mtypes: List[str] = self.get_model_types()
        mtypes_with_count: List[List[Union[str, int]]] = [
            [mt__, len([m__ for m__ in self.get_mn() if self.get_mt(m__) == mt__])] 
            for mt__ in mtypes
        ]
        
        f_key: str = _i18n("model_types")
        s_key: str = _i18n("models_count")
        
        mtype_width: int = max(len(f_key), max([len(c_) for c_ in mtypes]))
        models_count_width: int = max([
            len(_c1 + " " + s_key.format(count=_n, end=format_end_count_models(_n))) 
            for (_c1, _n) in mtypes_with_count
        ])
        
        print("|-", "-" * models_count_width, "-|", sep="")
        print(f"| {f_key:<{models_count_width}} |")
        print("|-", "-" * models_count_width, "-|", sep="")
        
        if mtypes_with_count:
            for (mt_, num_) in mtypes_with_count:
                print(f"| {mt_+' '+_i18n('models_count').format(count=num_, end=format_end_count_models(num_)):<{models_count_width}} |")
                print("|-", "-" * models_count_width, "-|", sep="")
        else:
            print(f"| {'n/a':<{models_count_width}} |")
            print("|-", "-" * models_count_width, "-|", sep="")

    def get_categories(self) -> List[str]:
        """
        Получить список всех категорий
        
        Returns:
            Список категорий
        """
        categories: List[str] = []
        for model in self.get_mn():
            c_: str = self.get_category(model)
            if c_ and c_ not in categories:
                categories.append(c_)
        return categories
    
    def get_model_types(self) -> List[str]:
        """
        Получить список всех типов моделей
        
        Returns:
            Список типов моделей
        """
        model_types: List[str] = []
        for model in self.get_mn():
            mt_: str = self.get_mt(model)
            if mt_ and mt_ not in model_types:
                model_types.append(mt_)
        return model_types

    def download_model(
        self, 
        model_paths: str, 
        model_name: str, 
        model_type: str, 
        ckpt_url: str, 
        conf_url: str, 
        only_check_exists: bool = False
    ) -> Union[bool, Tuple[str, str]]:
        """
        Скачать модель
        
        Args:
            model_paths: Путь для сохранения модели
            model_name: Имя модели
            model_type: Тип модели
            ckpt_url: URL чекпоинта
            conf_url: URL конфига
            only_check_exists: Только проверить существование
        
        Returns:
            True если только проверка, иначе кортеж (config_path, checkpoint_path)
        """
        model_dir: str = os.path.join(model_paths, model_type)
        os.makedirs(model_dir, exist_ok=True)

        config_path: str = os.path.join(model_dir, f"{model_name}_config.yaml")
        checkpoint_path: str = os.path.join(
            model_dir,
            f"{model_name}.onnx" if model_type == "mdxnet" else f"{model_name}.ckpt",
        )

        if config_path is None or checkpoint_path is None:
            raise RuntimeError(_i18n("model_paths_error"))

        if os.path.exists(checkpoint_path) and os.path.exists(config_path):
            if (
                os.path.getsize(checkpoint_path) == 0
                or os.path.getsize(config_path) == 0
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

    def conf_editor(
        self, 
        config_path: str, 
        mdx_denoise: bool, 
        vr_aggr: int, 
        vr_enable_post_process: bool, 
        vr_high_end_process: bool, 
        model_type: str, 
        econom_mode: bool
    ) -> None:
        """
        Редактирование конфигурации модели
        
        Args:
            config_path: Путь к конфигу
            mdx_denoise: Шумоподавление для MDX
            vr_aggr: Агрессивность для VR
            vr_enable_post_process: Постобработка для VR
            vr_high_end_process: Обработка высоких частот для VR
            model_type: Тип модели
            econom_mode: Эконом-режим
        """

        class IndentDumper(yaml.Dumper):
            def increase_indent(self, flow: bool = False, indentless: bool = False) -> str:
                return super(IndentDumper, self).increase_indent(flow, False)

        def tuple_constructor(loader: yaml.Loader, node: yaml.Node) -> tuple:
            values = loader.construct_sequence(node)
            return tuple(values)

        yaml.SafeLoader.add_constructor(
            "tag:yaml.org,2002:python/tuple", tuple_constructor
        )

        def conf_edit(
            config_path: str, 
            mdx_denoise: bool, 
            vr_aggr: int, 
            vr_enable_post_process: bool, 
            vr_high_end_process: bool, 
            model_type: str, 
            econom_mode: bool
        ) -> None:
            with open(config_path, "r") as f:
                data: Dict = yaml.load(f, Loader=yaml.SafeLoader)

            if "use_amp" not in data.keys():
                if "training" not in data:
                    data["training"] = {}
                data["training"]["use_amp"] = True

            if model_type not in ["vr", "htdemucs"]:
                if "inference" not in data:
                    data["inference"] = {}
                if data["inference"].get("num_overlap") != 2:
                    data["inference"]["num_overlap"] = 2

            if "inference" in data and data["inference"].get("batch_size") != 1:
                data["inference"]["batch_size"] = 1

            if model_type == "mdxnet":
                if "inference" not in data:
                    data["inference"] = {}
                data["inference"]["denoise"] = mdx_denoise

            elif model_type == "vr":
                if "inference" not in data:
                    data["inference"] = {}
                data["inference"]["aggression"] = vr_aggr
                data["inference"]["enable_post_process"] = vr_enable_post_process
                data["inference"]["high_end_process"] = vr_high_end_process

            if econom_mode:
                if model_type in ["mel_band_roformer", "bs_roformer"]:
                    if "audio" not in data:
                        data["audio"] = {}
                    old_chunk_size: int = data["audio"].get("chunk_size", 0)
                    hop_length: int = data["audio"].get("hop_length", 441)
                    dim_t, new_chunk_size = self.generate_econom_params(hop_length=hop_length)
                    if old_chunk_size >= new_chunk_size:
                        print(_i18n("economy_chunk_resize", new_size=new_chunk_size))
                        data["audio"]["new_chunk_size"] = new_chunk_size
                        data["audio"]["new_dim_t"] = dim_t
                elif model_type in ["htdemucs"]:
                    if "training" not in data:
                        data["training"] = {}
                    old_segment: int = data["training"].get("segment", 0)
                    if old_segment >= MVSEPLESS_ECONOM_SEGMENT:
                        print(_i18n("economy_segment_resize", new_segment=MVSEPLESS_ECONOM_SEGMENT))
                        data["training"]["new_segment"] = MVSEPLESS_ECONOM_SEGMENT
            else:
                if model_type in ["mel_band_roformer", "bs_roformer"]:
                    if "audio" in data:
                        if "new_chunk_size" in data["audio"]:
                            del data["audio"]["new_chunk_size"]
                        if "new_dim_t" in data["audio"]:
                            del data["audio"]["new_dim_t"]
                elif model_type in ["htdemucs"]:
                    if "training" in data:
                        if "new_segment" in data["training"]:
                            del data["training"]["new_segment"]

            with open(config_path, "w") as f:
                yaml.dump(
                    data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    Dumper=IndentDumper,
                    allow_unicode=True,
                )

        conf_edit(
            config_path, 
            mdx_denoise, 
            vr_aggr, 
            vr_enable_post_process, 
            vr_high_end_process, 
            model_type, 
            econom_mode
        )

    def install_model(
        self,
        model_name: str,
        mdx_denoise: bool = False,
        vr_aggr: int = 5,
        vr_post_process: bool = False,
        vr_high_end_process: bool = False,
        econom_mode: bool = False,
        only_check: bool = False,
        progress: Optional[gr.Progress] = None,
    ) -> Union[bool, Tuple[int, str, str, str]]:
        """
        Установить модель
        
        Args:
            model_name: Имя модели
            mdx_denoise: Шумоподавление для MDX
            vr_aggr: Агрессивность для VR
            vr_post_process: Постобработка для VR
            vr_high_end_process: Обработка высоких частот для VR
            econom_mode: Эконом-режим
            only_check: Только проверить наличие
            progress: Прогресс
        
        Returns:
            True если только проверка, иначе кортеж (id, conf, ckpt, model_type)
        """
        info: Optional[Dict] = self.models_info.get(model_name, None)
        if not info:
            raise ValueError(
                _i18n("model_not_found", model=model_name)
            )
            
        id: int = self.get_id(model_name)
        model_type: str = self.get_mt(model_name)
        
        result = self.download_model(
            model_paths=self.models_cache_dir,
            model_name=model_name,
            model_type=model_type,
            ckpt_url=info["checkpoint_url"],
            conf_url=info["config_url"],
            only_check_exists=only_check
        )
        
        if isinstance(result, tuple):
            conf, ckpt = result[0], result[1]
            self.conf_editor(
                conf, 
                mdx_denoise, 
                vr_aggr, 
                vr_post_process, 
                vr_high_end_process, 
                model_type, 
                econom_mode
            )
            return id, conf, ckpt, model_type
        elif isinstance(result, bool):
            return result
        else:
            return False
    
    def get_mn_dwloaded(self) -> List[str]:
        """
        Получить список установленных моделей
        
        Returns:
            Список имен установленных моделей
        """
        return [model for model in self.get_mn() if self.install_model(model, only_check=True)]


class Separator(MvseplessModelManager):
    """Основной класс для разделения аудио"""
    
    def __init__(self, chunk_duration: float = 300) -> None:
        """
        Инициализация разделителя
        
        Args:
            chunk_duration: Длительность чанка в секундах
        """
        super().__init__()
        self.device: str = set_device(0)
        self.chunk_duration: float = chunk_duration
        self.ensemble_methods: Tuple[str, ...] = ("min_fft", "max_fft", "avg_fft", "median_fft")
        self.methods_subtract: Tuple[str, ...] = ("waveform", "spectrogram")
        self.ensemble_invert_methods_map: Dict[str, str] = {
            "min_fft": "max_fft",
            "max_fft": "min_fft",
            "avg_fft": "avg_fft",
            "median_fft": "median_fft",
        }
        self.namer: Namer = Namer()

    def check_duration_audio(self, path: str) -> bool:
        """
        Проверить длительность аудио
        
        Args:
            path: Путь к аудиофайлу
        
        Returns:
            True если длительность превышает chunk_duration
        """
        mixture, sr = read(path, sr=16000, mono=True, dtype="int8", flatten=True)
        duration: float = get_duration_from_array(mixture, sr)
        del mixture, sr
        if self.chunk_duration:
            return duration > self.chunk_duration
        return False

    def chunk_wise_processing(
        self,
        path: str,
        output_dir: str,
        model_type: str,
        model_name: str,
        ext_inst: bool,
        output_format: str,
        output_bitrate: str,
        template: str,
        selected_stems: List[str],
        ckpt: str,
        conf: str,
        id: int,
        progress: gr.Progress,
        use_spec_invert: bool,
        add_text_progress: str,
        device: str,
    ) -> List[Tuple[str, str]]:
        """
        Обработка длинного аудио по частям
        
        Args:
            path: Путь к аудиофайлу
            output_dir: Директория для вывода
            model_type: Тип модели
            model_name: Имя модели
            ext_inst: Извлечь инструментал
            output_format: Формат вывода
            output_bitrate: Битрейт
            template: Шаблон имени
            selected_stems: Выбранные стемы
            ckpt: Путь к чекпоинту
            conf: Путь к конфигу
            id: ID модели
            progress: Прогресс
            use_spec_invert: Использовать инверсию спектрограммы
            add_text_progress: Дополнительный текст прогресса
        
        Returns:
            Список кортежей (имя стема, путь к файлу)
        """
        print(_i18n("msg_trimming"))
        temp_dir: str = tempfile.mkdtemp()
        
        # Читаем исходное аудио
        mixture, sr = read(path, sr=44100)
        duration: float = get_duration_from_array(mixture)
        
        # Параметры для нарезки с минимальным перекрытием
        chunk_size: int = int(self.chunk_duration * sr)
        overlap_duration: float = 2.0  # перекрытие в секундах
        overlap_samples: int = int(overlap_duration * sr)
        fade_size: int = overlap_samples // 2  # плавный переход на половине перекрытия
        
        # Шаг между чанками
        step: int = chunk_size - overlap_samples
        
        print(_i18n("msg_chunk_size", duration=self.chunk_duration, samples=chunk_size))
        print(_i18n("msg_overlap", duration=overlap_duration, samples=overlap_samples))
        print(_i18n("msg_step_percent", percent=(step/chunk_size*100)))
        
        # Создаем окно для плавного склеивания
        window: np.ndarray = np.ones(chunk_size)
        
        # Плавное затухание в конце
        fadeout: np.ndarray = np.linspace(1, 0, fade_size)
        window[-fade_size:] = fadeout
        
        # Для первого чанка добавим плавное нарастание в начале
        fadein: np.ndarray = np.linspace(0, 1, fade_size)
        window[:fade_size] = fadein
        
        # Нарезаем аудио на чанки
        input_chunks: List[Tuple[str, np.ndarray]] = []
        chunk_positions: List[Tuple[int, int]] = []  # храним позиции начала и длину каждого чанка
        
        i: int = 0
        chunk_index: int = 0
        
        while i < mixture.shape[1]:
            # Вырезаем чанк
            end_pos: int = min(i + chunk_size, mixture.shape[1])
            part: np.ndarray = mixture[:, i:end_pos]
            chunk_len: int = part.shape[1]
            
            # Сохраняем позицию и длину
            chunk_positions.append((i, chunk_len))
            
            # Дополняем до нужного размера если нужно
            if chunk_len < chunk_size:
                pad_len: int = chunk_size - chunk_len
                pad_mode: str = "reflect" if chunk_len > chunk_size // 2 else "constant"
                part = np.pad(part, ((0, 0), (0, pad_len)), mode=pad_mode)
                
                # Корректируем окно для последнего чанка
                last_window: np.ndarray = np.ones(chunk_size)
                last_window[-fade_size:] = fadeout
                last_window[chunk_len:] = 0  # обнуляем заполненную часть
            else:
                last_window = window
            
            # Сохраняем чанк во временный файл
            chunk_path: str = os.path.join(temp_dir, f"chunk_{chunk_index:04d}.wav")
            write(chunk_path, part, sr)
            input_chunks.append((chunk_path, last_window if chunk_len < chunk_size else window))
            
            i += step
            chunk_index += 1
        
        total_chunks: int = len(input_chunks)
        
        # Определяем стемы, которые будут получены от модели
        stems_list: List[str] = []
        model_stems: List[str] = self.get_stems(model_name)
        
        if len(model_stems) == 2:
            stems_list = [stem for stem in selected_stems] if selected_stems else model_stems
        elif len(model_stems) >= 3:
            stems_list = [stem for stem in selected_stems] if selected_stems else model_stems
            if ext_inst:
                if selected_stems:
                    stems_list.extend(["inverted +", "inverted -"])
                else:
                    if (all(instr in model_stems for instr in ["bass", "drums", "other", "vocals"]) or
                        all(instr in model_stems for instr in ["bass", "drums", "other", "vocals", "piano", "guitar"])):
                        stems_list.extend(["instrumental +", "instrumental -"])
        
        # Словарь для накопления результатов по каждому стему
        result_accumulators: Dict[str, np.ndarray] = {stem: np.zeros_like(mixture) for stem in stems_list}
        counter_accumulators: Dict[str, np.ndarray] = {stem: np.zeros(mixture.shape[1]) for stem in stems_list}
        
        # Обрабатываем каждый чанк
        for chunk_idx, (chunk_path, chunk_window) in enumerate(input_chunks):
            print(_i18n("msg_processing_chunk", current=chunk_idx + 1, total=total_chunks))
            
            # Обрабатываем чанк
            chunk_results: List[Tuple[str, str]] = self.separator_base(
                input_file=chunk_path,
                output_dir=os.path.join(temp_dir, f"output_chunk_{chunk_idx:04d}"),
                model_type=model_type,
                model_name=model_name,
                ext_inst=ext_inst,
                output_format="wav",
                output_bitrate="320k",
                template=template,
                selected_stems=selected_stems,
                ckpt=ckpt,
                conf=conf,
                id=id,
                progress=progress,
                use_spec_invert=use_spec_invert,
                add_text_progress=f"{_i18n('msg_processing_chunk', current=chunk_idx + 1, total=total_chunks)}",
                device=device,
            )
            
            start_pos, chunk_len = chunk_positions[chunk_idx]
            
            # Загружаем обработанные стемы для этого чанка
            for stem_name, stem_path in chunk_results:
                if stem_name in result_accumulators:
                    # Читаем обработанный стем
                    stem_audio, _c = read(stem_path)
                    
                    # Применяем окно для плавного склеивания
                    window_segment: np.ndarray = chunk_window[:chunk_len]
                    
                    # Добавляем в аккумулятор с применением окна
                    result_accumulators[stem_name][:, start_pos:start_pos + chunk_len] += \
                        stem_audio[:, :chunk_len] * window_segment
                    counter_accumulators[stem_name][start_pos:start_pos + chunk_len] += window_segment
            
            # Очищаем временные файлы чанка
            try:
                os.remove(chunk_path)
                shutil.rmtree(os.path.join(temp_dir, f"output_chunk_{chunk_idx:04d}"))
            except:
                pass
        
        # Финальное усреднение и сохранение результатов
        print(_i18n("msg_assembling_chunks"))
        progress(1, desc=_i18n("msg_assembling_chunks"))
        final_results: List[Tuple[str, str]] = []
        os.makedirs(output_dir, exist_ok=True)
        
        for stem_name in stems_list:
            counter: np.ndarray = counter_accumulators[stem_name]
            valid_mask: np.ndarray = counter > 1e-6
            
            final_audio: np.ndarray = np.zeros_like(result_accumulators[stem_name])
            final_audio[:, valid_mask] = result_accumulators[stem_name][:, valid_mask] / counter[valid_mask]
            final_audio = np.nan_to_num(final_audio, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Генерируем имя файла
            file_name: str = os.path.splitext(os.path.basename(path))[0]
            file_name_shorted: str = self.namer.short_input_name_template(
                template, STEM=stem_name, MODEL=model_name, ID=id, NAME=file_name
            )
            custom_name: str = self.namer.template(
                template,
                STEM=stem_name,
                MODEL=model_name,
                ID=id,
                NAME=file_name_shorted,
            )
            output_path: str = os.path.join(output_dir, f"{custom_name}.{output_format}")
            
            write(output_path, final_audio, sr, output_bitrate)
            final_results.append((stem_name, output_path))
            
            print(_i18n("msg_saved_stem", stem=stem_name, path=output_path))
        
        # Очищаем временную директорию
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        
        return final_results

    def _get_windowing_array(self, window_size: int, fade_size: int) -> np.ndarray:
        """
        Создает окно для плавного склеивания чанков
        
        Args:
            window_size: Размер окна
            fade_size: Размер зоны затухания
        
        Returns:
            Массив окна
        """
        fadein: np.ndarray = np.linspace(0, 1, fade_size)
        fadeout: np.ndarray = np.linspace(1, 0, fade_size)
        
        window: np.ndarray = np.ones(window_size)
        window[:fade_size] = fadein
        window[-fade_size:] = fadeout
        return window

    def print_error_list(self, errors: List[str]) -> None:
        """
        Вывести список ошибок
        
        Args:
            errors: Список ошибок
        """
        if errors:
            print(_i18n("failed_separations"))
            for _e in errors:
                print(f"  - {_e}")

    class OutputReader:
        """Читатель вывода процесса"""
        
        def __init__(self, debug: bool = False) -> None:
            self.debug: bool = debug

        def parse_json_line(self, line: str) -> Optional[Dict]:
            """
            Парсинг JSON строки
            
            Args:
                line: Строка для парсинга
            
            Returns:
            """
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None

        def reaction_line(
            self, 
            line: str, 
            progress: gr.Progress, 
            add_text: str
        ) -> Optional[List]:
            """
            Обработка строки вывода
            
            Args:
                line: Строка вывода
                progress: Прогресс
                add_text: Дополнительный текст
            
            Returns:
                Результат обработки или None
            """
            _add_text: str = ""
            if add_text:
                _add_text = f"| {add_text}"

            data: Optional[Dict] = self.parse_json_line(line)
            if data is None:
                return None
            elif "reading" in data:
                progress(0.05, desc=_i18n("progress_reading", text=_add_text))
                print(_i18n("msg_reading_file"))
                return None
            elif "stems" in data:
                stems_str: str = ','.join(data['stems'])
                progress(0.05, desc=_i18n("progress_stems", stems=stems_str, text=_add_text))
                print(_i18n("msg_stems", stems=stems_str))
                return None
            elif "processing" in data:
                progress_a: Dict = data["processing"]
                processed: int = progress_a.get("processed", 0)
                total: int = progress_a.get("total", 1)
                if total > 0:
                    percent: int = int((processed / total) * 100)
                    progress(
                        (processed, total), 
                        desc=_i18n("progress_processing", percent=percent, text=_add_text),
                        unit=progress_a.get("unit", _i18n("unit_samples"))
                    )
                    print(f"\r{_i18n('msg_processed_percent', percent=percent)}", end="")
                return None
            elif "writing" in data:
                progress(0.9, desc=_i18n("progress_writing", text=_add_text))
                print(f"\r{_i18n('msg_writing_file', file=data['writing'])}", end="")
                return None
            elif "done" in data:
                progress(1.0, desc=_i18n("progress_completed", text=_add_text))
                print(f"\r{_i18n('msg_completed')}", end="\n")
                return data["done"]
            elif "error" in data:
                raise Exception(data["error"])

    output_reader: OutputReader = OutputReader()

    def separator_base(
        self,
        input_file: str,
        output_dir: str,
        model_type: str,
        model_name: str,
        ext_inst: bool,
        output_format: str,
        output_bitrate: str,
        template: str,
        selected_stems: List[str],
        ckpt: str,
        conf: str,
        id: int,
        progress: gr.Progress,
        use_spec_invert: bool = False,
        add_text_progress: str = "",
        device: str = "cpu",
    ) -> List[Tuple[str, str]]:
        """
        Базовый метод разделения
        
        Args:
            input_file: Входной файл
            output_dir: Выходная директория
            model_type: Тип модели
            model_name: Имя модели
            ext_inst: Извлечь инструментал
            output_format: Формат вывода
            output_bitrate: Битрейт
            template: Шаблон имени
            selected_stems: Выбранные стемы
            ckpt: Путь к чекпоинту
            conf: Путь к конфигу
            id: ID модели
            progress: Прогресс
            use_spec_invert: Использовать инверсию спектрограммы
            add_text_progress: Дополнительный текст прогресса
        
        Returns:
            Список кортежей (имя стема, путь к файлу)
        """
        cmd: List[str] = [
            os.sys.executable,
            "-m",
            "infer",
            "--input",
            input_file,
            "--store_dir",
            output_dir,
            "--model_type",
            model_type,
            "--model_name",
            model_name,
            "--model_id",
            str(id),
            "--config_path",
            conf,
            "--start_check_point",
            ckpt,
            "--output_format",
            output_format,
            "--output_bitrate",
            str(output_bitrate),
            "--template",
            template,
            "--device",
            device
        ]
        
        if ext_inst:
            cmd.append("--extract_instrumental")
        if use_spec_invert:
            cmd.append("--use_spec_invert")
        if selected_stems:
            cmd.append("--selected_instruments")
            cmd.extend(selected_stems)

        process: Optional[subprocess.Popen] = None
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
            )

            result: Optional[List] = None
            error_lines: List[str] = []

            # Чтение stdout построчно
            if process.stdout:
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
            if process.stderr:
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
                error_text: str = "\n".join(error_lines[-5:]) if error_lines else _i18n("unknown_error")
                raise Exception(
                    _i18n("process_error", code=process.returncode, error=error_text)
                )

            if result is not None:
                return result
            else:
                raise Exception(_i18n("no_result_error"))

        except Exception as e:
            raise e
        finally:
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except:
                    try:
                        process.kill()
                    except:
                        pass

    def separate(
        self,
        input: Union[str, List[str], None] = None,
        output_dir: Optional[str] = None,
        model_name: str = "bs_6stem",
        ext_inst: bool = True,
        output_format: str = "mp3",
        output_bitrate: str = "320k",
        template: str = "NAME_(STEM)_MODEL",
        selected_stems: Optional[List[str]] = None,
        add_settings: Dict[str, Any] = {
            "mdx_denoise": False,
            "vr_aggr": 5,
            "vr_post_process": False,
            "vr_high_end_process": False,
            "add_single_sep_text_progress": None,
            "device": "cpu"
        },
        use_spec_invert: bool = False,
        progress: gr.Progress = gr.Progress(track_tqdm=True),
    ) -> Union[List[Tuple[str, str]], List[Tuple[str, List[Tuple[str, str]]]]]:
        """
        Разделение аудио
        
        Args:
            input: Входной файл или список файлов
            output_dir: Выходная директория
            model_name: Имя модели
            ext_inst: Извлечь инструментал
            output_format: Формат вывода
            output_bitrate: Битрейт
            template: Шаблон имени
            selected_stems: Выбранные стемы
            add_settings: Дополнительные настройки
            use_spec_invert: Использовать инверсию спектрограммы
            progress: Прогресс
        
        Returns:
            Результаты разделения
        """
        progress(0, desc=_i18n("start_processing"))

        if output_format not in output_formats:
            output_format = "flac"

        if output_dir is None:
            output_dir = os.getcwd()

        if selected_stems is None:
            selected_stems = []

        if not input:
            raise ValueError(_i18n("no_input_error"))

        if "STEM" not in template and template is not None:
            template = template + "_STEM_"
        if not template:
            template = "mvsepless_NAME_(STEM)"

        os.makedirs(output_dir, exist_ok=True)

        mdx_denoise: bool = add_settings.get("mdx_denoise", False)
        vr_aggr: int = add_settings.get("vr_aggr", 5)
        vr_post_process: bool = add_settings.get("vr_post_process", False)
        vr_high_end_process: bool = add_settings.get("vr_high_end_process", False)
        econom_mode: bool = add_settings.get("econom_mode", MVSEPLESS_ECONOM)
        single_mode: bool = add_settings.get("single_mode", True)
        add_progress_text_custom: str = add_settings.get("add_single_sep_text_progress", "")
        device = add_settings.get("device", self.device)

        id, conf, ckpt, model_type = self.install_model(
            model_name, mdx_denoise, vr_aggr, vr_post_process, vr_high_end_process, econom_mode, progress=progress
        )

        input_list: List[str] = []
        errors: List[str] = []
        output_state: List = []

        if isinstance(input, str):
            input = [input]

        input_list = get_files_from_list(input)

        if len(input_list) == 0:
            print(_i18n("no_input_files"))

        print(_i18n("input_files_count", count=len(input_list)))

        if single_mode:
            if len(input_list) == 1:
                _input_file: str = input_list[0]
                try:
                    if self.check_duration_audio(_input_file):
                        output_state = self.chunk_wise_processing(
                            path=_input_file,
                            output_dir=output_dir,
                            model_type=model_type,
                            model_name=model_name,
                            ext_inst=ext_inst,
                            output_format=output_format,
                            output_bitrate=output_bitrate,
                            template=template,
                            selected_stems=selected_stems,
                            ckpt=ckpt,
                            conf=conf,
                            id=id,
                            progress=progress,
                            use_spec_invert=use_spec_invert,
                            add_text_progress=add_progress_text_custom,
                            device=device
                        )
                    else:
                        output_state = self.separator_base(
                            input_file=_input_file,
                            output_dir=output_dir,
                            model_type=model_type,
                            model_name=model_name,
                            ext_inst=ext_inst,
                            output_format=output_format,
                            output_bitrate=output_bitrate,
                            template=template,
                            selected_stems=selected_stems,
                            ckpt=ckpt,
                            conf=conf,
                            id=id,
                            progress=progress,
                            use_spec_invert=use_spec_invert,
                            add_text_progress=add_progress_text_custom,
                            device=device
                        )

                except Exception as e:
                    errors.append(_input_file)
                    traceback.print_exc()
            elif len(input_list) > 1:
                single_mode = False

        if not single_mode:
            if len(input_list) >= 1:
                for i, f in enumerate(input_list, 1):
                    print(_i18n("processing_file", current=i, total=len(input_list), file=f))
                    gr.Warning(
                        title=_i18n("processing_file_title", current=i, total=len(input_list)), 
                        message=f
                    )
                    try:
                        if self.check_duration_audio(f):
                            seped: List[Tuple[str, str]] = self.chunk_wise_processing(
                                path=f,
                                output_dir=output_dir,
                                model_type=model_type,
                                model_name=model_name,
                                ext_inst=ext_inst,
                                output_format=output_format,
                                output_bitrate=output_bitrate,
                                template=template,
                                selected_stems=selected_stems,
                                ckpt=ckpt,
                                conf=conf,
                                id=id,
                                progress=progress,
                                use_spec_invert=use_spec_invert,
                                add_text_progress=_i18n("file_progress", current=i, total=len(input_list)),
                            )
                        else:
                            seped = self.separator_base(
                                input_file=f,
                                output_dir=output_dir,
                                model_type=model_type,
                                model_name=model_name,
                                ext_inst=ext_inst,
                                output_format=output_format,
                                output_bitrate=output_bitrate,
                                template=template,
                                selected_stems=selected_stems,
                                ckpt=ckpt,
                                conf=conf,
                                id=id,
                                progress=progress,
                                use_spec_invert=use_spec_invert,
                                add_text_progress=_i18n("file_progress", current=i, total=len(input_list)),
                            )
                        basename: str = os.path.splitext(os.path.basename(f))[0]
                        output_state.append([basename, seped])
                    except Exception as e:
                        errors.append(f)
                        traceback.print_exc()
            else:
                pass

        self.print_error_list(errors)
        return output_state

    def manual_ensemble(
        self,
        files: List[str],
        weights: List[float],
        output_name: str,
        ensemble_type: str,
        out_format: str = "mp3",
        add_wav: bool = False
    ) -> Union[str, Tuple[str, str], None]:
        """
        Ручной ансамбль из готовых файлов
        
        Args:
            files: Список файлов
            weights: Веса файлов
            output_name: Имя выходного файла
            ensemble_type: Тип ансамбля
            out_format: Формат вывода
            add_wav: Добавить WAV версию
        
        Returns:
            Путь к выходному файлу или кортеж (mp3, wav)
        """
        if not files:
            print(_i18n("no_input_files"))
            return (None, None) if add_wav else None
            
        valid_files: List[str] = get_files_from_list(files, only_files=True)
        if not valid_files:
            print(_i18n("no_audio_files"))
            return (None, None) if add_wav else None
        
        arrays, srs = multiread(valid_files)
        results, max_sr = ensemble(arrays, srs, weights, ensemble_type)

        if add_wav:
            print(_i18n("writing_files", file1=f"{output_name}.{out_format}", file2=f"{output_name}_orig.wav"))
            return (
                write(self.namer.iter(f"{output_name}.{out_format}"), results, max_sr),
                write(self.namer.iter(f"{output_name}_orig.wav"), results, max_sr)
            )
        else:
            print(_i18n("writing_file", file=f"{output_name}.{out_format}"))
            return write(self.namer.iter(f"{output_name}.{out_format}"), results, max_sr)

    def auto_ensemble(
        self,
        input_file: str,
        ensemble_state: List[List[Union[str, int, float]]],
        output_dir: str,
        method: str,
        out_format: str,
        invert_ensemble: bool,
        progress: gr.Progress = gr.Progress(track_tqdm=True),
    ) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        """
        Автоматический ансамбль
        
        Args:
            input_file: Входной файл
            ensemble_state: Состояние ансамбля
            output_dir: Выходная директория
            method: Метод объединения
            out_format: Формат вывода
            invert_ensemble: Инвертировать ансамбль
            progress: Прогресс
        
        Returns:
            Кортеж (выходной файл, WAV файл, инвертированный файл, список исходников)
        """
        invert_methods_map: Dict[str, str] = self.ensemble_invert_methods_map
        
        if not input_file:
            print(_i18n("no_input_error"))
            return None, None, None, []
        if not os.path.exists(input_file):
            print(_i18n("file_not_exists"))
            return None, None, None, []
        if not check(input_file):
            print(_i18n("file_no_audio"))
            return None, None, None, []
        
        o: str = output_dir
        os.makedirs(o, exist_ok=True)

        basename: str = os.path.splitext(os.path.basename(input_file))[0]

        def invert_weights(weights: List[float]) -> List[float]:
            """Инвертировать веса"""
            total_weight: float = sum(weights)
            return [total_weight - w for w in weights]

        success_separations: List[Tuple[str, Optional[str], Optional[str], float]] = []
        ensemble_sources_list: List[str] = []
        
        if ensemble_state:
            total_ensemble_models: int = len(ensemble_state)
            for i, model in enumerate(ensemble_state, start=1):
                ens_mn: str = str(model[0])
                ens_s_stem: str = str(model[1])
                ens_i_stem: str = str(model[2])
                weight: float = float(model[3])

                s_stem: Optional[str] = None
                i_stem: Optional[str] = None

                try:
                    result_seped_auto_ensemble = self.separate(
                        input=input_file,
                        output_dir=os.path.join(o, ens_mn),
                        model_name=ens_mn,
                        ext_inst=True,
                        template="NAME - MODEL - STEM",
                        output_format="wav",
                        add_settings={
                            "add_single_sep_text_progress": _i18n("ensemble_progress", current=i, total=total_ensemble_models)
                        },
                        progress=progress,
                    )
                    
                    if result_seped_auto_ensemble:
                        if isinstance(result_seped_auto_ensemble, list):
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
                                    "add_single_sep_text_progress": _i18n("ensemble_invert_progress", current=i, total=total_ensemble_models)
                                },
                                progress=progress,
                            )
                            if result_seped_auto_ensemble_invert:
                                if isinstance(result_seped_auto_ensemble_invert, list):
                                    for stem, path in result_seped_auto_ensemble_invert:
                                        if stem == ens_i_stem:
                                            i_stem = path
                                            ensemble_sources_list.append(path)

                except Exception as e:
                    print(f"\n{_i18n('msg_error_occurred', error=str(e))}")
                    progress(0, desc=_i18n("msg_skipping_model"))
                    continue
                finally:
                    if s_stem:
                        success_separations.append((ens_mn, s_stem, i_stem, weight))

        ensemble_sources_stems: List[str] = []
        ensemble_sources_invert_stems: List[str] = []
        weights: List[float] = []

        for out_mn, out_s_stem, out_i_stem, out_weight in success_separations:
            if out_s_stem:
                ensemble_sources_stems.append(out_s_stem)
            if out_i_stem:
                ensemble_sources_invert_stems.append(out_i_stem)
            weights.append(out_weight)

        auto_ensemble_invout_file: Optional[str] = None

        if not ensemble_sources_stems:
            return None, None, None, []
            
        auto_ensemble_output_name: str = f"ensembless_{self.namer.short(basename, length=50)}_{len(ensemble_sources_stems)}_{method}"
        auto_ensemble_inverted_output_name: str = f"ensembless_{self.namer.short(basename, length=50)}_{len(ensemble_sources_stems)}_{invert_methods_map[method]}_invert"
        
        ensemble_result = self.manual_ensemble(
            files=ensemble_sources_stems,
            weights=weights,
            output_name=os.path.join(o, auto_ensemble_output_name),
            ensemble_type=method,
            out_format=out_format,
            add_wav=True,
        )
        
        if isinstance(ensemble_result, tuple):
            auto_ensemble_out_file, auto_ensemble_out_file_wav = ensemble_result
        else:
            auto_ensemble_out_file, auto_ensemble_out_file_wav = ensemble_result, None

        if invert_ensemble and ensemble_sources_invert_stems:
            invert_result = self.manual_ensemble(
                files=ensemble_sources_invert_stems,
                weights=invert_weights(weights),
                output_name=os.path.join(o, auto_ensemble_inverted_output_name),
                ensemble_type=invert_methods_map[method],
                out_format=out_format,
                add_wav=True,
            )
            if isinstance(invert_result, tuple):
                auto_ensemble_invout_file, _c = invert_result
            else:
                auto_ensemble_invout_file = invert_result
                
        return (
            auto_ensemble_out_file,
            auto_ensemble_out_file_wav,
            auto_ensemble_invout_file,
            ensemble_sources_list,
        )

    def subtract(
        self,
        audio1_path: str,
        audio2_path: str,
        method: str,
        output_path: str = "./inverted.mp3",
    ) -> Optional[str]:
        """
        Вычитание одного аудио из другого
        
        Args:
            audio1_path: Путь к первому аудио (оригинал)
            audio2_path: Путь ко второму аудио (стем для вычитания)
            method: Метод вычитания
            output_path: Путь для сохранения результата
        
        Returns:
            Путь к выходному файлу или None
        """
        if not audio1_path or not audio2_path:
            if not audio1_path:
                print(_i18n("original_not_specified"))
            if not audio2_path:
                print(_i18n("stem_not_specified"))
            return None
            
        if not os.path.exists(audio1_path) or not os.path.exists(audio2_path):
            if not os.path.exists(audio1_path):
                print(_i18n("original_not_exists"))
            if not os.path.exists(audio2_path):
                print(_i18n("stem_not_exists"))
            return None
            
        if not check(audio1_path) or not check(audio2_path):
            if not check(audio1_path):
                print(_i18n("original_no_audio"))
            if not check(audio2_path):
                print(_i18n("stem_no_audio"))
            return None
            
        y1, sr1 = read(audio1_path)
        y2, sr2 = read(audio2_path)
        inverted, min_sr = subtractor(y1, y2, sr1, sr2, spectrogram=(method == "spectrogram"))
        
        print(_i18n("writing_file", file=output_path))
        return write(self.namer.iter(output_path), inverted, min_sr)

    def extract_phantom_center(
        self, 
        input_path: str, 
        output_path_mid: Optional[str] = None, 
        output_path_side: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Извлечение фантомного центра (Mid/Side)
        
        Args:
            input_path: Входной файл
            output_path_mid: Путь для Mid канала
            output_path_side: Путь для Side канала
        
        Returns:
            Кортеж (путь к Mid, путь к Side)
        """
        if not input_path:
            print(_i18n("no_input_error"))
            return None, None
            
        if not os.path.exists(input_path):
            print(_i18n("file_not_exists"))
            return None, None
            
        if not check(input_path):
            print(_i18n("file_no_audio"))
            return None, None
            
        dirname: str = os.path.dirname(input_path)
        basename, ext = os.path.splitext(os.path.basename(input_path))
        
        if not output_path_mid:
            output_path_mid = os.path.join(dirname, f"{self.namer.short(basename, length=80)}_mid{ext}")
        if not output_path_side:
            output_path_side = os.path.join(dirname, f"{self.namer.short(basename, length=80)}_side{ext}")
            
        y, sr = read(input_path)
        mid, side = split_mid_side(y, var=3, sr=sr)
        
        print(_i18n("writing_files", file1=output_path_mid, file2=output_path_side))
        return (
            write(self.namer.iter(output_path_mid), mid, sr),
            write(self.namer.iter(output_path_side), side, sr)
        )


if __name__ == "__main__":
    mvsepless: Separator = Separator()

    parser = argparse.ArgumentParser(
        description=_i18n("cli_description"),
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60),
        # Добавляем allow_abbrev=False чтобы избежать конфликтов
        allow_abbrev=False
    )
    parser._positionals.title = _i18n("additional_modes")
    parser._optionals.title = _i18n("main_parameters")

    # Создаем субпарсеры с required=True
    subparsers = parser.add_subparsers(dest="command", help=None, required=False)
    
    # --- Общие параметры переносим в каждый субпарсер ---
    # Создаем функцию для добавления общих аргументов
    def add_common_args(subparser):
        subparser.add_argument("-i", "--input", nargs='+', default=[], help=_i18n("input_path_help"))
        subparser.add_argument("-o", "--output_dir", type=str, default="", help=_i18n("output_dir_help"))
        subparser.add_argument("-of", "--output_format", type=str, default="mp3", choices=output_formats, help=_i18n("output_format_help"))
        subparser.add_argument("-ob", "--output_bitrate", type=str, default="320k", help=_i18n("output_bitrate_help"), metavar="BITRATE")
        subparser.add_argument("-on", "--output_name", type=str, default="ensemble", help=_i18n("output_name_help"))
        subparser.add_argument("-op", "--output_path", type=str, default="inverted.mp3", help=_i18n("output_path_help"))
        return subparser
    
    # Команда separator
    sep_p = subparsers.add_parser("separator", help=_i18n("separator_help"))
    sep_p = add_common_args(sep_p)
    sep_p.add_argument("-mn", "--model_name", type=str, default="bs_6stem", help=_i18n("model_name_help"))
    sep_p.add_argument("-tmpl", "--template", type=str, default="NAME_(STEM)_MODEL", help=_i18n("template_help"))
    sep_p.add_argument("-stem", "--selected_stems", type=str, nargs="*", default=None, help=_i18n("selected_stems_help"), metavar="STEM")
    sep_p.add_argument("-inst", "--ext_inst", action="store_true", help=_i18n("ext_inst_help"))
    sep_p.add_argument("-invspec", "--use_spec_invert", action="store_true", help=_i18n("use_spec_invert_help"))
    sep_p.add_argument("-dw", "--install_only", action="store_true", help=_i18n("install_only_help"))
    sep_p.add_argument("--mdx_enable_denoise", action="store_true", help=_i18n("mdx_denoise_help"))
    sep_p.add_argument("--vr_aggression", type=int, default=5, help=_i18n("vr_aggression_help"), metavar="AGGR")
    sep_p.add_argument("--vr_high_end_process", action="store_true", help=_i18n("vr_high_end_help"))
    sep_p.add_argument("--vr_enable_post_process", action="store_true", help=_i18n("vr_post_process_help"))
    sep_p.add_argument("--econom_mode", action="store_true", help=_i18n("econom_mode_help"))
    sep_p.add_argument("--chunk_duration", type=float, default=None, help=_i18n("chunk_duration_help"))

    # Команда info
    info_p = subparsers.add_parser("info", help=_i18n("info_help"))
    info_p = add_common_args(info_p)
    info_p.add_argument("-limit", "--limit", type=int, default=0, help=_i18n("limit_help"))
    info_p.add_argument("-stem", "--stem", type=str, default=None, help=_i18n("stem_filter_help"))
    info_p.add_argument("-t","--model_types", nargs='*', help=_i18n("model_types_help"))
    info_p.add_argument("-c", "--categories", nargs='*', help=_i18n("categories_help"))
    info_p.add_argument("-oi", "--only_installed", action="store_true", help=_i18n("only_installed_help"))
    
    info_other_group = info_p.add_mutually_exclusive_group(required=False)
    info_other_group.add_argument("-lc", "--list_categories", action="store_true", help=_i18n("list_categories_help"))
    info_other_group.add_argument("-lt", "--list_model_types", action="store_true", help=_i18n("list_model_types_help"))
    info_other_group.add_argument("-u", "--update", action="store_true", help=_i18n("update_help"))
   
    # Команда auto_ensemble
    auto_p = subparsers.add_parser("auto_ensemble", help=_i18n("auto_ensemble_help"))
    auto_p = add_common_args(auto_p)
    auto_p.add_argument("-m", "--method", type=str, default="avg_fft", 
                       choices=("min_fft", "max_fft", "avg_fft", "median_fft"), 
                       help=_i18n("method_help"))
    auto_p.add_argument("-inv", "--invert", action="store_true", help=_i18n("invert_ensemble_help"))
    
    auto_group = auto_p.add_mutually_exclusive_group(required=True)
    auto_group.add_argument("-ml", '--model_list', nargs='+', 
                           help=_i18n("model_list_help"), 
                           metavar="MODEL,STEM1,STEM2,WEIGHT")
    auto_group.add_argument("-json", "--json", type=str, help=_i18n("json_help"))

    # Команда manual_ensemble
    manual_p = subparsers.add_parser("manual_ensemble", help=_i18n("manual_ensemble_help"))
    manual_p = add_common_args(manual_p)
    manual_p.add_argument("-w", "--weights", nargs='+', type=float, help=_i18n("weights_help"))
    manual_p.add_argument("-m", "--method", type=str, default="avg_fft", 
                         choices=("min_fft", "max_fft", "avg_fft", "median_fft"), 
                         help=_i18n("method_help"))

    # Команда subtract
    sub_p = subparsers.add_parser("subtract", help=_i18n("subtract_help"))
    sub_p = add_common_args(sub_p)
    sub_p.add_argument("--stem", type=str, required=True, help=_i18n("stem_path_help"))
    sub_p.add_argument("--method", choices=["waveform", "spectrogram"], 
                      default="waveform", help=_i18n("subtract_method_help"))

    # Команда ext_phantom_center
    center_p = subparsers.add_parser("ext_phantom_center", help=_i18n("phantom_center_help"))
    center_p = add_common_args(center_p)
    center_p.add_argument("--mid", type=str, help=_i18n("mid_path_help"))
    center_p.add_argument("--side", type=str, help=_i18n("side_path_help"))

    # Команда app
    app_p = subparsers.add_parser("app", help=_i18n("app_help"))
    app_p = add_common_args(app_p)
    app_p.add_argument("-p", "--port", type=int, default=None, help=_i18n("port_help"))
    app_p.add_argument("-s", "--share", action="store_true", help=_i18n("share_help"))
    app_p.add_argument("-a", "--add_app", action="store_true", help=_i18n("add_app_help"))
    app_p.add_argument("-pl", "--use_plugins", action="store_true", help=_i18n("plugins_help"))
    app_p.add_argument("-vb", "--vbach", action="store_true", help=_i18n("vbach_help"))
    app_p.add_argument("-udir", "--user_dir", type=str, default=None, help=_i18n("user_dir_help"))
    
    args = parser.parse_args()

    # 1. Список моделей
    if args.command == "info":
        if args.list_categories:
            mvsepless.get_list_categories()
        elif args.list_model_types:
            mvsepless.get_list_model_types()
        elif args.update:
            file_path: str = MvseplessModelManager().models_info_path
            url_link: str = "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/models.json?download=true"
            dw_file(url_link, file_path, retries=999999)
        else:
            mvsepless.get_list_supported_models(
                limit=args.limit, 
                stem=args.stem, 
                model_types=args.model_types, 
                category=args.categories, 
                only_installed=args.only_installed
            )
        sys.exit(0)

    # 2. Логика подкоманд
    if args.command == "auto_ensemble":
        ensemble_state: List[List] = []
        if args.json:
            with open(args.json, 'r', encoding='utf-8') as f:
                ensemble_state = json.load(f)
        else:
            for i, item in enumerate(args.model_list):
                parts = item.split(',')
                if len(parts) == 4:
                    parts[3] = float(parts[3])
                    ensemble_state.append(parts)
                else:
                    print(_i18n("model_format_error", item=item))
                    sys.exit(1)
                    
        if not args.input:
            sys.exit(1)
        else:
            first_file: str = args.input[0]
            mvsepless.auto_ensemble(
                input_file=first_file,
                ensemble_state=ensemble_state,
                output_dir=args.output_dir,
                method=args.method,
                out_format=args.output_format,
                invert_ensemble=args.invert
            )

    elif args.command == "manual_ensemble":
        weights: List[float] = args.weights if args.weights else [1.0] * len(args.input)
        if len(weights) < len(args.input):
            weights += [1.0] * (len(args.input) - len(weights))
        
        mvsepless.manual_ensemble(
            files=args.input,
            weights=weights[:len(args.input)],
            output_name=args.output_name,
            ensemble_type=args.method,
            out_format=args.output_format
        )

    elif args.command == "subtract":
        if not args.input:
            sys.exit(1)
        else:
            first_file = args.input[0]
            mvsepless.subtract(first_file, args.stem, args.method, args.output_path)

    elif args.command == "ext_phantom_center":
        if not args.input:
            sys.exit(1)
        else:
            first_file = args.input[0]
            mvsepless.extract_phantom_center(first_file, args.mid, args.side)

    elif args.command == "app":
        from app import SeparatorGradio, user_directory
        
        if args.user_dir:
            user_directory.change_dir(args.user_dir)
            
        SeparatorGradio().UI(
            gr.themes.Citrus(
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
            ),
            args.add_app, 
            args.use_plugins, 
            args.vbach
        ).launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
            allowed_paths=["/"],
            debug=True,
            inbrowser=True
        )

    elif args.command == "separator":
        if args.install_only:
            mvsepless.install_model(args.model_name)
        else:
            if args.chunk_duration is not None:
                mvsepless.chunk_duration = args.chunk_duration
                
            mvsepless.separate(
                input=args.input,
                output_dir=args.output_dir,
                model_name=args.model_name,
                ext_inst=args.ext_inst,
                output_format=args.output_format,
                output_bitrate=args.output_bitrate,
                template=args.template,
                add_settings={
                    "mdx_denoise": args.mdx_enable_denoise,
                    "vr_aggr": args.vr_aggression,
                    "vr_post_process": args.vr_enable_post_process,
                    "vr_high_end_process": args.vr_high_end_process,
                    "econom_mode": args.econom_mode
                },
                selected_stems=args.selected_stems,
                use_spec_invert=args.use_spec_invert
            )
    else:
        # Получение актуального списка моделей
        file_path: str = MvseplessModelManager().models_info_path
        url_link: str = "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/models.json?download=true"
        dw_file(url_link, file_path, retries=999999)
        from app import SeparatorGradio

        SeparatorGradio().UI(
            gr.themes.Citrus(
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
            ),
            False, 
            False, 
            True
        ).launch(
            server_name="0.0.0.0",
            server_port=None,
            share=False,
            allowed_paths=["/"],
            debug=True,
            inbrowser=True
        )
