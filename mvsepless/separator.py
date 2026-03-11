import os, sys, json, subprocess, argparse, gradio as gr, yaml
import numpy as np
import tempfile
import shutil
import traceback
import torch
from packaging import version
script_dir = os.path.dirname(os.path.abspath(__file__))
env = os.environ.copy()
env['PYTHONPATH'] = script_dir
from gradio_helper import dw_file, all_ids, set_device, cuda_available
from audio import check, output_formats, multiread, write, ensemble, read, subtractor, split_mid_side, get_duration_from_array
from namer import Namer
is_pytorch2 = version.parse(torch.__version__) >= version.parse("2.0.0")
is_pytorch2_4 = version.parse(torch.__version__) >= version.parse("2.4.0")
unsupported_models = ["bs_inst_fno_unwa", "mbr_wsa"] if not is_pytorch2 else ["bs_inst_fno_unwa"] if not is_pytorch2_4 else []
MVSEPLESS_ECONOM = not cuda_available
MVSEPLESS_ECONOM_SEGMENT = int(os.environ.get("MVSEPLESS_ECO_SEG", "7"))

def get_files_from_list(input: list, only_files: bool = False):
    input_list = []
    for path in input:
        if os.path.isdir(path):
            if not only_files:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        if check(full_path):
                            input_list.append(full_path)
        elif os.path.isfile(path):
            if check(path):
                input_list.append(path)
        else:
            pass
    return input_list

def format_end_count_models(count: int):
    if count % 10 == 1 and count % 100 != 11:
        return "ь"  # 1 модель, 21 модель, 101 модель
    elif (count % 10 in [2, 3, 4]) and (count % 100 not in [12, 13, 14]):
        return "и"  # 2-4 модели, 22-24 модели
    else:
        return "ей"  # 5-20 моделей, 25-30 моделей и т.д.

class MvseplessModelManager:
    def __init__(
        self,
        models_info_path=os.path.join(script_dir, "models.json"),
        cache_dir=os.path.join(script_dir, "mvsepless_models_cache"),
    ):
        self.models_cache_dir = cache_dir
        self.models_info_path = models_info_path
        with open(self.models_info_path, "r", encoding="utf-8") as f:
            models_info = json.load(f)
        self.models_info = models_info

    def get_mt(self, model_name):
        return self.models_info.get(model_name).get("model_type")

    def get_mn(self):
        return [mn for mn in self.models_info if mn not in unsupported_models]

    def get_stems(self, model_name):
        if model_name is not None and model_name != "":
            return [
                stem
                for stem in self.models_info
                .get(model_name)
                .get("stems", [])
            ]
        else:
            return []
        
    def get_id(self, model_name):
        if model_name is not None and model_name != "":
            return self.models_info.get(model_name).get("id", 0)
        else:
            return 0

    def get_tgt_inst(self, model_name):
        if model_name is not None and model_name != "":
            return (
                self.models_info
                .get(model_name)
                .get("target_instrument", None)
            )
        else:
            return None

    def get_category(self, model_name):
        if model_name is not None and model_name != "":
            return self.models_info.get(model_name).get("category", "")
        else:
            return ""

    def calculate_dimensions(self, chunk_size, hop_length=441):
        # Находим dim_t
        dim_t = (chunk_size // hop_length) + 1
        
        # Проверяем, чтобы chunk_size был кратен hop_length для идеального совмещения
        actual_chunk_size = (dim_t - 1) * hop_length
        
        return dim_t, actual_chunk_size

    def generate_econom_params(self, sr=44100, seconds=MVSEPLESS_ECONOM_SEGMENT, hop_length=441):
        chunk_size = sr * seconds
        dim_t, chunk_size = self.calculate_dimensions(chunk_size, hop_length)
        return dim_t, chunk_size

    def get_list_supported_models(self, limit: None | int = None, stem: None | str = None, model_types: None | list = None, category: None | list = None, only_installed: bool = False):
        models = self.get_list_mn_from_category(category, model_types)
        if not models:
            return
        installed_models = [model for model in models if self.install_model(model, only_check=True)]
        if stem and stem != "":
            models = [model for model in models if (stem in self.get_stems(model) or stem.lower() in self.get_stems(model) or stem.upper() in self.get_stems(model) or stem.capitalize() in self.get_stems(model) or stem.title() in self.get_stems(model))]
        if limit:
            models = models[:limit]
        f_key, s_key = "Имя модели", "Выходные стемы"

        filename_width = max(len(f_key), max(len(model) for model in models)) + 2
        stems_width = max(len(s_key), max(len(", ".join(self.get_stems(model))) for model in models)) + 1
        print("|-", "-" * filename_width, "---", "-" * stems_width, "-|", sep="")

        if only_installed:
            print(f"| {f'[Установленные модели]':<{filename_width + stems_width}}    |")
        else:
            print(f"| {f'✔ - установлено':<{filename_width + stems_width}}    |")
        print(f"| {f'* - целевой стем':<{filename_width + stems_width}}    |")
        print("|-", "-" * filename_width, "---", "-" * stems_width, "-|", sep="")
        if category:
            print(f"| {f'Категории:':<{filename_width + stems_width}}    |")
            for c in category:
                print(f"| {f'  - {c}':<{filename_width + stems_width}}    |")
        else:
            print(f"| {f'Категории: Все':<{filename_width + stems_width}}    |")

        if model_types:
            print(f"| {f'Типы моделей:':<{filename_width + stems_width}}    |")
            for mt_ in model_types:
                print(f"| {f'  - {mt_}':<{filename_width + stems_width}}    |")
        else:
            print(f"| {f'Типы моделей: Все':<{filename_width + stems_width}}    |")

        if stem and stem != "":
             print(f"| {f'Выбранный стем: {stem}':<{filename_width + stems_width}}    |")
        print("|-", "-" * filename_width, "-+-", "-" * stems_width, "-|", sep="")
        print(f"| {f_key:<{filename_width}} | {s_key:<{stems_width}} |")
        print("|-", "-" * filename_width, "-+-", "-" * stems_width, "-|", sep="")
        if only_installed:
            if installed_models:
                for model in installed_models:
                    stems = ", ".join([_st+'*' if _st == self.get_tgt_inst(model) else _st for _st in self.get_stems(model)])
                    print(f"| {model:<{filename_width}} | {stems:<{stems_width}} |")
                    print("|-", "-" * filename_width, "-+-", "-" * stems_width, "-|", sep="")
            else:
                print(f"| {'н/д':<{filename_width}} | {'н/д':<{stems_width}} |")
                print("|-", "-" * filename_width, "-+-", "-" * stems_width, "-|", sep="")
        else:
            if models:
                for model in models:
                    stems = ", ".join([_st+'*' if _st == self.get_tgt_inst(model) else _st for _st in self.get_stems(model)])
                    if model in installed_models:
                        print(f"| {f'{model} ✔':<{filename_width}} | {stems:<{stems_width}} |")
                    else:
                        print(f"| {model:<{filename_width}} | {stems:<{stems_width}} |")
                    print("|-", "-" * filename_width, "-+-", "-" * stems_width, "-|", sep="")
            else:
                print(f"| {'н/д':<{filename_width}} | {'н/д':<{stems_width}} |")
                print("|-", "-" * filename_width, "-+-", "-" * stems_width, "-|", sep="")

    def get_list_mn_from_category(self, category: list | str, model_type: list | None = None):
        list_models = []
        categories = []
        if category:
            if isinstance(category, str) and category != "":
                categories.append(category)
            elif isinstance(category, list):
                categories.extend(category)
            if categories:
                if not model_type:
                    list_models = [model for model in self.get_mn() if self.get_category(model) in category]
                else:
                    list_models = [model for model in self.get_mn() if self.get_category(model) in category and self.get_mt(model) in model_type]
            else:
                if not model_type:
                    list_models = [model for model in self.get_mn()]
                else:
                    list_models = [model for model in self.get_mn() if self.get_mt(model) in model_type]
        else:
            if not model_type:
                list_models = [model for model in self.get_mn()]
            else:
                list_models = [model for model in self.get_mn() if self.get_mt(model) in model_type]
        return list_models

    def get_list_categories(self):
        categories = self.get_categories()
        categories_with_count_models = [[cat__, len([m__ for m__ in self.get_mn() if self.get_category(m__) == cat__])] for cat__ in categories]
        f_key, s_key = "Категории:", " ({count} модел{end})"
        category_width = max(len(f_key), max(len(c_) for c_ in categories))
        models_count_width = max([len(_c1+s_key.format(count=_n, end=format_end_count_models(_n))) for (_c1, _n) in categories_with_count_models])
        print("|-", "-" * models_count_width, "-|", sep="")
        print(f"| {f_key:<{models_count_width}} |")
        print("|-", "-" * models_count_width, "-|", sep="")
        if categories_with_count_models:
            for (cat_, num_) in categories_with_count_models:
                print(f"| {cat_+s_key.format(count=num_, end=format_end_count_models(num_)):<{models_count_width}} |")
                print("|-", "-" * models_count_width, "-|", sep="")
        else:
            print(f"| {'н/д':<{models_count_width}} |")
            print("|-", "-" * models_count_width, "-|", sep="")

    def get_list_model_types(self):
        mtypes = self.get_model_types()
        mtypes_with_count_models = [[mt__, len([m__ for m__ in self.get_mn() if self.get_mt(m__) == mt__])] for mt__ in mtypes]
        f_key, s_key = "Типы моделей:", " ({count} модел{end})"
        mtype_width = max(len(f_key), max([len(c_) for c_ in mtypes]))
        models_count_width = max([len(_c1+s_key.format(count=_n, end=format_end_count_models(_n))) for (_c1, _n) in mtypes_with_count_models])
        print("|-", "-" * models_count_width, "-|", sep="")
        print(f"| {f_key:<{models_count_width}} |")
        print("|-", "-" * models_count_width, "-|", sep="")
        if mtypes_with_count_models:
            for (mt_, num_) in mtypes_with_count_models:
                print(f"| {mt_+s_key.format(count=num_, end=format_end_count_models(num_)):<{models_count_width}} |")
                print("|-", "-" * models_count_width, "-|", sep="")
        else:
            print(f"| {'н/д':<{models_count_width}} |")
            print("|-", "-" * models_count_width, "-|", sep="")

    def get_categories(self):
        categories = []
        for model in self.get_mn():
            c_ = self.get_category(model)
            if c_ not in categories:
                categories.append(c_)
        return categories
    
    def get_model_types(self):
        model_types = []
        for model in self.get_mn():
            mt_ = self.get_mt(model)
            if mt_ not in model_types:
                model_types.append(mt_)
        return model_types

    def download_model(self, model_paths, model_name, model_type, ckpt_url, conf_url, only_check_exists=False):
        model_dir = os.path.join(model_paths, model_type)
        os.makedirs(model_dir, exist_ok=True)

        config_path = os.path.join(model_dir, f"{model_name}_config.yaml")
        checkpoint_path = os.path.join(
            model_dir,
            f"{model_name}.onnx" if model_type == "mdxnet" else f"{model_name}.ckpt",
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

    def conf_editor(self, config_path, mdx_denoise, vr_aggr, vr_enable_post_process, vr_high_end_process, model_type, econom_mode):

        class IndentDumper(yaml.Dumper):
            def increase_indent(self, flow=False, indentless=False):
                return super(IndentDumper, self).increase_indent(flow, False)

        def tuple_constructor(loader, node):
            values = loader.construct_sequence(node)
            return tuple(values)

        yaml.SafeLoader.add_constructor(
            "tag:yaml.org,2002:python/tuple", tuple_constructor
        )

        def conf_edit(config_path: str, mdx_denoise: bool, vr_aggr: int, vr_enable_post_process: bool, vr_high_end_process: bool, model_type: str, econom_mode: bool):
            with open(config_path, "r") as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)

            if "use_amp" not in data.keys():
                data["training"]["use_amp"] = True

            if model_type not in ["vr", "htdemucs"]:
                if data["inference"]["num_overlap"] != 2:
                    data["inference"]["num_overlap"] = 2

            if data["inference"]["batch_size"] != 1:
                data["inference"]["batch_size"] = 1

            if model_type == "mdxnet":
                data["inference"]["denoise"] = mdx_denoise

            elif model_type == "vr":
                data["inference"]["aggression"] = vr_aggr
                data["inference"]["enable_post_process"] = vr_enable_post_process
                data["inference"]["high_end_process"] = vr_high_end_process

            if econom_mode:
                if model_type in ["mel_band_roformer", "bs_roformer"]:
                    old_chunk_size = data["audio"]["chunk_size"]
                    hop_length = data["audio"]["hop_length"]
                    dim_t, new_chunk_size = self.generate_econom_params(hop_length=hop_length)
                    if old_chunk_size >= new_chunk_size:
                        print(f"Для экономии ресурсов размер чанка был изменен на {new_chunk_size}")
                        data["audio"]["new_chunk_size"] = new_chunk_size
                        data["audio"]["new_dim_t"] = dim_t
                elif model_type in ["htdemucs"]:
                    old_segment = data["training"]["segment"]
                    if old_segment >= MVSEPLESS_ECONOM_SEGMENT:
                        print(f"Для экономии ресурсов размер сегмента был изменен на {MVSEPLESS_ECONOM_SEGMENT}")
                        data["training"]["new_segment"] = MVSEPLESS_ECONOM_SEGMENT
            else:
                if model_type in ["mel_band_roformer", "bs_roformer"]:
                    if "new_chunk_size" in data["audio"]:
                        del data["audio"]["new_chunk_size"]
                    if "new_dim_t" in data["audio"]:
                        del data["audio"]["new_dim_t"]
                elif model_type in ["htdemucs"]:
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

        conf_edit(config_path, mdx_denoise, vr_aggr, vr_enable_post_process, vr_high_end_process, model_type, econom_mode)

    def install_model(
        self,
        model_name: str,
        mdx_denoise: bool = False,
        vr_aggr: bool = 5,
        vr_post_process: bool = False,
        vr_high_end_process: bool = False,
        econom_mode: bool = False,
        only_check: bool = False,
        progress: any = None,
    ) -> tuple[int, str, str, str] | bool:

        info = self.models_info.get(model_name, None)
        if not info:
            raise ValueError(
                f"Модель {model_name} не найдена"
            )
        id = self.get_id(model_name)
        model_type = self.get_mt(model_name)
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
            self.conf_editor(conf, mdx_denoise, vr_aggr, vr_post_process, vr_high_end_process, model_type, econom_mode)
            return id, conf, ckpt, model_type
        elif isinstance(result, bool):
            return result
    
    def get_mn_dwloaded(self):
        return [model for model in self.get_mn() if self.install_model(model, only_check=True)]

class Separator(MvseplessModelManager):

    def __init__(self, chunk_duration: float = 300):
        super().__init__()
        self.device = set_device(0)
        self.chunk_duration = chunk_duration
        self.ensemble_methods = ("min_fft", "max_fft", "avg_fft", "median_fft")
        self.methods_subtract = ("waveform", "spectrogram")
        self.ensemble_invert_methods_map = {
            "min_fft": "max_fft",
            "max_fft": "min_fft",
            "avg_fft": "avg_fft",
            "median_fft": "median_fft",
        }
        self.namer = Namer()

    def check_duration_audio(self, path):
        mixture, sr = read(path, sr=16000, mono=True, dtype="int8", flatten=True)
        duration = get_duration_from_array(mixture, sr)
        del mixture, sr
        if self.chunk_duration:
            return duration > self.chunk_duration
        return False

    def chunk_wise_processing(
        self, 
        path, 
        output_dir, 
        model_type, 
        model_name, 
        ext_inst, 
        output_format, 
        output_bitrate, 
        template,
        selected_stems,
        ckpt,
        conf,
        id,
        progress,
        use_spec_invert,
        add_text_progress,
    ):
        print("Обрезка аудио на чанки с минимальным перекрытием...")
        temp_dir = tempfile.mkdtemp()
        
        # Читаем исходное аудио
        mixture, sr = read(path, sr=44100)
        duration = get_duration_from_array(mixture)
        
        # Параметры для нарезки с минимальным перекрытием
        chunk_size = int(self.chunk_duration * sr)
        overlap_duration = 2  # перекрытие в секундах (можно регулировать)
        overlap_samples = int(overlap_duration * sr)
        fade_size = overlap_samples // 2  # плавный переход на половине перекрытия
        
        # Шаг между чанками (почти полный chunk_size, минус перекрытие)
        step = chunk_size - overlap_samples
        
        print(f"Размер чанка: {self.chunk_duration}с ({chunk_size} сэмплов)")
        print(f"Перекрытие: {overlap_duration}с ({overlap_samples} сэмплов)")
        print(f"Шаг: {step/chunk_size*100:.1f}% от размера чанка")
        
        # Создаем окно для плавного склеивания
        # Окно теперь имеет плавный спад только в конце
        window = np.ones(chunk_size)
        
        # Плавное затухание в конце
        fadeout = np.linspace(1, 0, fade_size)
        window[-fade_size:] = fadeout
        
        # Для первого чанка добавим плавное нарастание в начале
        # (опционально, чтобы избежать щелчка в самом начале)
        fadein = np.linspace(0, 1, fade_size)
        window[:fade_size] = fadein
        
        # Нарезаем аудио на чанки
        input_chunks = []
        chunk_positions = []  # храним позиции начала и длину каждого чанка
        
        i = 0
        chunk_index = 0
        
        while i < mixture.shape[1]:
            # Вырезаем чанк
            end_pos = min(i + chunk_size, mixture.shape[1])
            part = mixture[:, i:end_pos]
            chunk_len = part.shape[1]
            
            # Сохраняем позицию и длину
            chunk_positions.append((i, chunk_len))
            
            # Дополняем до нужного размера если нужно (только для последнего чанка)
            if chunk_len < chunk_size:
                pad_len = chunk_size - chunk_len
                pad_mode = "reflect" if chunk_len > chunk_size // 2 else "constant"
                part = np.pad(part, ((0, 0), (0, pad_len)), mode=pad_mode)
                
                # Корректируем окно для последнего чанка
                last_window = np.ones(chunk_size)
                last_window[-fade_size:] = fadeout
                last_window[chunk_len:] = 0  # обнуляем заполненную часть
            else:
                last_window = window
            
            # Сохраняем чанк во временный файл
            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_index:04d}.wav")
            write(chunk_path, part, sr)
            input_chunks.append((chunk_path, last_window if chunk_len < chunk_size else window))
            
            i += step
            chunk_index += 1
        
        total_chunks = len(input_chunks)
        
        # Определяем стемы, которые будут получены от модели
        if len(self.get_stems(model_name)) == 2:
            stems_list = [stem for stem in selected_stems] if selected_stems else self.get_stems(model_name)
        elif len(self.get_stems(model_name)) >= 3:
            stems_list = [stem for stem in selected_stems] if selected_stems else self.get_stems(model_name)
            if ext_inst:
                if selected_stems:
                    stems_list.extend(["inverted +", "inverted -"])
                else:
                    if (
                        all(
                            instr in self.get_stems(model_name)
                            for instr in ["bass", "drums", "other", "vocals"]
                        )
                        or all(
                            instr in self.get_stems(model_name)
                            for instr in ["bass", "drums", "other", "vocals", "piano", "guitar"]
                        )
                    ):
                        stems_list.extend(["instrumental +", "instrumental -"])
        # Словарь для накопления результатов по каждому стему
        result_accumulators = {stem: np.zeros_like(mixture) for stem in stems_list}
        counter_accumulators = {stem: np.zeros(mixture.shape[1]) for stem in stems_list}
        
        # Обрабатываем каждый чанк
        for chunk_idx, (chunk_path, chunk_window) in enumerate(input_chunks):
            print(f"Чанк {chunk_idx + 1}/{total_chunks}")
            
            # Обрабатываем чанк
            chunk_results = self.separator_base(
                input_file=chunk_path,
                output_dir=os.path.join(temp_dir, f"output_chunk_{chunk_idx:04d}"),
                model_type=model_type,
                model_name=model_name,
                ext_inst=ext_inst,
                output_format="wav",
                output_bitrate=320,
                template=template,
                selected_stems=selected_stems,
                ckpt=ckpt,
                conf=conf,
                id=id,
                progress=progress,
                use_spec_invert=use_spec_invert,
                add_text_progress=f"{add_text_progress} [Чанк {chunk_idx + 1}/{total_chunks}]",
            )
            
            start_pos, chunk_len = chunk_positions[chunk_idx]
            
            # Загружаем обработанные стемы для этого чанка
            for stem_name, stem_path in chunk_results:
                if stem_name in result_accumulators:
                    # Читаем обработанный стем
                    stem_audio, _ = read(stem_path)
                    
                    # Применяем окно для плавного склеивания
                    window_segment = chunk_window[:chunk_len]
                    
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
        print("Сборка обработанных чанков в единое аудио")
        progress(1, desc="Сборка обработанных чанков")
        final_results = []
        os.makedirs(output_dir, exist_ok=True)
        
        for stem_name in stems_list:
            counter = counter_accumulators[stem_name]
            valid_mask = counter > 1e-6
            
            final_audio = np.zeros_like(result_accumulators[stem_name])
            final_audio[:, valid_mask] = result_accumulators[stem_name][:, valid_mask] / counter[valid_mask]
            final_audio = np.nan_to_num(final_audio, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Генерируем имя файла
            file_name = os.path.splitext(os.path.basename(path))[0]
            file_name_shorted = self.namer.short_input_name_template(
                template, STEM=stem_name, MODEL=model_name, ID=id, NAME=file_name
            )
            custom_name = self.namer.template(
                template,
                STEM=stem_name,
                MODEL=model_name,
                ID=id,
                NAME=file_name_shorted,
            )
            output_path = os.path.join(output_dir, f"{custom_name}.{output_format}")
            
            write(output_path, final_audio, sr, output_bitrate)
            final_results.append((stem_name, output_path))
            
            print(f"Сохранен стем {stem_name}: {output_path}")
        
        # Очищаем временную директорию
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        
        return final_results

    def _get_windowing_array(self, window_size: int, fade_size: int) -> np.ndarray:
        """Создает окно для плавного склеивания чанков"""
        fadein = np.linspace(0, 1, fade_size)
        fadeout = np.linspace(1, 0, fade_size)
        
        window = np.ones(window_size)
        window[:fade_size] = fadein
        window[-fade_size:] = fadeout
        return window

    def print_error_list(self, errors: list):
        if errors:
            print("Неудачные разделения:")
            for _e in errors:
                print(f"  - {_e}")

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
            elif "reading" in data:
                progress(0.05, desc=f"Чтение файла {_add_text}")
                print("Чтение файла")
                return None
            elif "stems" in data:
                progress(0.05, desc=f"Стемы: {','.join(data['stems'])} {_add_text}")
                print(f"Стемы: {data['stems']}")
                return None
            elif "processing" in data:
                progress_a = data["processing"]
                processed = progress_a.get("processed", 0)
                total = progress_a.get("total", 1)
                if total > 0:
                    percent = int((processed / total) * 100)
                    progress((processed, total), desc=f"Обработано: {percent}% {_add_text}", unit=progress_a.get("unit", "сэмплов"))
                    print(f"\rОбработано: {percent}%", end="")
                return None
            elif "writing" in data:
                progress(0.9, desc=f"Запись результатов {_add_text}")
                print(f"\rЗапись в файл {data['writing']}", end="")
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
        model_type: str = "mel_band_roformer",
        model_name: str = "bs_6stem",
        ext_inst: bool = True,
        output_format: str = "mp3",
        output_bitrate: str = "320k",
        template: str = "NAME_(STEM)_MODEL",
        selected_stems: list = None,
        ckpt: str = None,
        conf: str = None,
        id: int = None,
        progress: any = None,
        use_spec_invert: bool = False,
        add_text_progress: str = "",
    ) -> list[tuple[str, str]]:

        cmd = [
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
            self.device
        ]
        if ext_inst:
            cmd.append("--extract_instrumental")
        if use_spec_invert:
            cmd.append("--use_spec_invert")
        if selected_stems:
            cmd.append("--selected_instruments")
            cmd.extend(selected_stems)

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
        model_name: str = "bs_6stem",
        ext_inst: bool = True,
        output_format: str = "mp3",
        output_bitrate: str = "320k",
        template: str = "NAME_(STEM)_MODEL",
        selected_stems: list = None,
        add_settings: dict = {
            "mdx_denoise": False,
            "vr_aggr": 5,
            "vr_post_process": False,
            "vr_high_end_process": False,
            #"econom_mode": False,
            "add_single_sep_text_progress": None,
        },
        use_spec_invert: bool = False,
        progress: any = gr.Progress(track_tqdm=True),
    ) -> list[tuple[str, str]] | list[str, list[tuple[str, str]]]:

        progress(0, desc="Начало обработки")

        if output_format not in output_formats:
            output_format = "flac"

        if output_dir is None:
            output_dir = os.getcwd()

        if selected_stems is None:
            selected_stems = []

        if not input:
            raise ValueError("Входной файл не указан")

        if "STEM" not in template and template is not None:
            template = template + "_STEM_"
        if not template:
            template = "mvsepless_NAME_(STEM)"

        os.makedirs(output_dir, exist_ok=True)

        mdx_denoise = add_settings.get("mdx_denoise", False)
        vr_aggr = add_settings.get("vr_aggr", 5)
        vr_post_process = add_settings.get("vr_post_process", False)
        vr_high_end_process = add_settings.get("vr_high_end_process", False)
        econom_mode = add_settings.get("econom_mode", MVSEPLESS_ECONOM)
        single_mode = add_settings.get("single_mode", True)
        add_progress_text_custom = add_settings.get("add_single_sep_text_progress", "")

        id, conf, ckpt, model_type = self.install_model(
            model_name, mdx_denoise, vr_aggr, vr_post_process, vr_high_end_process, econom_mode, progress
        )

        input_list = []
        errors = []
        output_state = []

        if isinstance(input, str):
            input = [input]

        input_list = get_files_from_list(input)

        if len(input_list) == 0:
            print("Входные файлы не указаны")

        print(f"Входных файлов: {len(input_list)}")

        if single_mode:
            if len(input_list) == 1:
                _input_file = input_list[0]
                basename = os.path.splitext(os.path.basename(_input_file))[0]
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
                        )

                except Exception as e:
                    errors.append(_input_file)
                    traceback.print_exc()
            elif len(input_list) > 1:
                single_mode = False

        if not single_mode:
            if len(input_list) >= 1:
                for i, f in enumerate(input_list, 1):
                    print(f"Файл {i} из {len(input_list)}: {f}")
                    gr.Warning(title=f"Файл {i} из {len(input_list)}: {f}", message="")
                    basename = os.path.splitext(os.path.basename(f))[0]
                    try:
                        if self.check_duration_audio(f):
                            seped = self.chunk_wise_processing(
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
                                add_text_progress=f"{i} из {len(input_list)}",
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
                                add_text_progress=f"{i} из {len(input_list)}",
                            )
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
        files: list,
        weights: list,
        output_name: str,
        ensemble_type: str,
        out_format="mp3",
        add_wav=False
    ):
        if not files:
            print("Входные файлы не указаны")
            return None, None if add_wav else None
        valid_files = get_files_from_list(files, only_files=True)
        if not valid_files:
            print("Входные файлы не содержат аудио")
            return None, None if add_wav else None
        
        arrays, srs = multiread(valid_files)
        results, max_sr = ensemble(arrays, srs, weights, ensemble_type)

        if add_wav:
            print(f"Запись в файлы: {output_name}.{out_format} и {output_name}_orig.wav")
            return write(self.namer.iter(f"{output_name}.{out_format}"), results, max_sr), write(self.namer.iter(f"{output_name}_orig.wav"), results, max_sr)
        else:
            print(f"Запись в файл: {output_name}.{out_format}")
            return write(self.namer.iter(f"{output_name}.{out_format}"), results, max_sr)

    def auto_ensemble(
        self,
        input_file: str,
        ensemble_state: list[list[str, str, str, int]],
        output_dir: str,
        method: str,
        out_format: str,
        invert_ensemble: bool,
        progress=gr.Progress(track_tqdm=True),
    ):
        ensemble_state = ensemble_state
        invert_methods_map = self.ensemble_invert_methods_map
        if not input_file:
            print("Входной файл не указан")
            return None, None, None, []
        if not os.path.exists(input_file):
            print("Входного файла не существует")
            return None, None, None, []
        if not check(input_file):
            print("Входной файл не содержит аудио")
            return None, None, None, []
        
        o = output_dir
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
            return None, None, None, []
        auto_ensemble_output_name = f"ensembless_{self.namer.short(basename, length=50)}_{len(ensemble_sources_stems)}_{method}"
        auto_ensemble_inverted_output_name = f"ensembless_{self.namer.short(basename, length=50)}_{len(ensemble_sources_stems)}_{invert_methods_map[method]}_invert"
        auto_ensemble_out_file, auto_ensemble_out_file_wav = self.manual_ensemble(
            files=ensemble_sources_stems,
            weights=weights,
            output_name=os.path.join(o, auto_ensemble_output_name),
            ensemble_type=method,
            out_format=out_format,
            add_wav=True,
        )

        if invert_ensemble:
            if ensemble_sources_invert_stems:
                auto_ensemble_invout_file, auto_ensemble_invout_file_wav = (
                    self.manual_ensemble(
                        files=ensemble_sources_invert_stems,
                        weights=invert_weights(weights),
                        output_name=os.path.join(o, auto_ensemble_inverted_output_name),
                        ensemble_type=invert_methods_map[method],
                        out_format=out_format,
                        add_wav=True,
                    )
                )
        return (
            auto_ensemble_out_file,
            auto_ensemble_out_file_wav,
            auto_ensemble_invout_file,
            ensemble_sources_list,
        )

    def subtract(
        self,
        audio1_path,
        audio2_path,
        method,
        output_path="./inverted.mp3",
    ):
        if not audio1_path or not audio2_path:
            if not audio1_path:
                print(f"Оригинал - не указано")
            if not audio2_path:
                print(f"Стем - не указано")
            return None
        if not os.path.exists(audio1_path) or not os.path.exists(audio2_path):
            if not os.path.exists(audio1_path):
                print(f"Оригинал - не существует")
            if not os.path.exists(audio2_path):
                print(f"Стем - не существует")
            return None
        if not check(audio1_path) or not check(audio2_path):
            if not check(audio1_path):
                print(f"Оригинал - не содержит аудио")
            if not check(audio2_path):
                print(f"Стем - не содержит аудио")
            return None
        y1, sr1 = read(audio1_path)
        y2, sr2 = read(audio2_path)
        inverted, min_sr = subtractor(y1, y2, sr1, sr2, spectrogram=method == "spectrogram")
        print(f"Запись в файл: {output_path}")
        return write(self.namer.iter(output_path), inverted, min_sr)

    def extract_phantom_center(self, i, output_path_mid=None, output_path_side=None):
        if not i:
            print("Входной файл не указан")
            return None, None
        if not os.path.exists(i):
            print("Входного файла не существует")
            return None, None
        if not check(i):
            print("Входной файл не содержит аудио")
            return None, None
        dirname = os.path.dirname(i)
        basename, ext = os.path.splitext(os.path.basename(i))
        if not output_path_mid:
            output_path_mid = os.path.join(dirname, f"{self.namer.short(basename, length=80)}_mid{ext}")
        if not output_path_side:
            output_path_side = os.path.join(dirname, f"{self.namer.short(basename, length=80)}_side{ext}")
        y, sr = read(i)
        mid, side = split_mid_side(y, var=3, sr=sr)
        print(f"Запись в файлы: {output_path_mid} и {output_path_side}")
        return write(self.namer.iter(output_path_mid), mid, sr), write(self.namer.iter(output_path_side), side, sr)

if __name__ == "__main__":

    mvsepless = Separator()

    parser = argparse.ArgumentParser(
        description="MVSepless: Обертка для MSST и UVR (audio-separator)",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60)
    )
    parser._positionals.title = 'Дополнительные режимы'
    parser._optionals.title = 'Основные параметры'

    # --- Общие параметры ---
    parser.add_argument("-i", "--input", nargs='+', default=[], help="Путь к входному аудиофайлу или папке.")
    parser.add_argument("-o", "--output_dir", type=str, default="", help="Директория для сохранения (по умолчанию: текущая).")
    parser.add_argument("-of", "--output_format", type=str, default="mp3", choices=output_formats, help="Формат вывода (по умолчанию: %(default)s).")
    parser.add_argument("-ob", "--output_bitrate", type=str, default="320k", help="Битрейт аудио (например, 320k).", metavar="BITRATE")
    parser.add_argument("-on", "--output_name", type=str, default="ensemble", help="Имя выходного файла. (путь/к/файлу_без_расширения)")
    parser.add_argument("-op", "--output_path", type=str, default="inverted.mp3", help="Путь к выходному файлу. (путь/к/файлу)")
    subparsers = parser.add_subparsers(dest="command", help=None)
    sep_p = subparsers.add_parser("separator", help="Разделение. Общие параметры: --input, --output_dir, --output_format, --output_bitrate")
    sep_p.add_argument("-mn", "--model_name", type=str, default="bs_6stem", help="Имя модели (по умолчанию: %(default)s).")
    sep_p.add_argument("-tmpl", "--template", type=str, default="NAME_(STEM)_MODEL", help="Шаблон имени: NAME, STEM, MODEL.")
    sep_p.add_argument("-stem", "--selected_stems", type=str, nargs="*", default=None, help="Список стемов (напр. Vocals Drums).", metavar="STEM")
    sep_p.add_argument("-inst", "--ext_inst", action="store_true", help="Извлечь инструментал вычитанием.")
    sep_p.add_argument("-invspec", "--use_spec_invert", action="store_true", help="Инверсия спектрограммы для вторичного стема.")
    sep_p.add_argument("-dw", "--install_only", action="store_true", help="Только установка модели")
    sep_p.add_argument("--mdx_enable_denoise", action="store_true", help="Шумоподавление для MDX-NET моделей")
    sep_p.add_argument("--vr_aggression", type=int, default=5, help="Агрессивность для VR моделей (по умолчанию: %(default)s).", metavar="AGGR")
    sep_p.add_argument("--vr_high_end_process", action="store_true", help="Восстановление недостающих высоких частот на VR моделях")
    sep_p.add_argument("--vr_enable_post_process", action="store_true", help="Дополнительная обработка для улучшения качества разделения VR модели")
    sep_p.add_argument("--econom_mode", action="store_true", help="Эконом-режим для Demucs и BS/Mel-Band Roformer (Уменьшение размера чанка)")
    sep_p.add_argument("--chunk_duration", type=float, default=None, help="Разделить аудио на фрагменты указанной длительности в секундах (по умолчанию: %(default)s = без разделения). Полезно для обработки очень длинных аудиофайлов в системах с ограниченной памятью. Рекомендуемое значение: 600 (10 минут) для файлов длительностью более 1 часа.")

    info_p = subparsers.add_parser("info", help="Показать список всех поддерживаемых моделей с фильтрацией.")
    info_p.add_argument("-limit", "--limit", type=int, default=0, help="Ограничить количество выводимых моделей (0 — без лимита).")
    info_p.add_argument("-stem", "--stem", type=str, default=None, help="Фильтровать по конкретному стему (напр. vocals, drums, bass).")
    info_p.add_argument("-t","--model_types", nargs='*', help="Фильтровать по типу архитектуры (напр. mdxnet, mel_band_roformer, scnet).")
    info_p.add_argument("-c", "--categories", nargs='*', help="Фильтровать по категории (напр. Вокал, Инструментал).")
    info_p.add_argument("-oi", "--only_installed", action="store_true", help="Фильтровать по факту установки")
    info_other_group = info_p.add_mutually_exclusive_group(required=False)
    info_other_group.add_argument("-lc", "--list_categories", action="store_true", help="Показать доступные категории")
    info_other_group.add_argument("-lt", "--list_model_types", action="store_true", help="Показать доступные типы моделей")
    info_other_group.add_argument("-u", "--update", action="store_true", help="Обновить информацию о моделях")
   
    auto_p = subparsers.add_parser("auto_ensemble", help="Автоматический ансамбль из нескольких моделей. Общие параметры: --input, --output_dir, --output_format")
    auto_p.add_argument("-m", "--method", type=str, default="avg_fft", choices=("min_fft", "max_fft", "avg_fft", "median_fft"), help="Метод объединения.")
    auto_p.add_argument("-inv", "--invert", action="store_true", help="Включить инверсию для ансамбля.")
    auto_group = auto_p.add_mutually_exclusive_group(required=True)
    auto_group.add_argument("-ml", '--model_list', nargs='+', help="Список моделей формата: MODEL,STEM1,STEM2,WEIGHT", metavar="MODEL,STEM1,STEM2,WEIGHT")
    auto_group.add_argument("-json", "--json", type=str, help="Путь к JSON конфигурации ансамбля.")

    manual_p = subparsers.add_parser("manual_ensemble", help="Сборка ансамбля из готовых файлов. Общие параметры: --input, --output_name, --output_format")
    manual_p.add_argument("-w", "--weights", nargs='+', type=float, help="Веса файлов.")
    manual_p.add_argument("-m", "--method", type=str, default="avg_fft", choices=("min_fft", "max_fft", "avg_fft", "median_fft"), help="Метод объединения.")

    sub_p = subparsers.add_parser("subtract", help="Вычитание стемов. Общие параметры: --input, --output_path")
    sub_p.add_argument("--stem", type=str, required=True, help="Файл стема для вычитания.")
    sub_p.add_argument("--method", choices=["waveform", "spectrogram"], default="waveform", help="Метод вычитания.")

    center_p = subparsers.add_parser("ext_phantom_center", help="Извлечение фантомного центра (Mid/Side). Общие параметры: --input")
    center_p.add_argument("--mid", type=str, help="Путь для Mid канала.")
    center_p.add_argument("--side", type=str, help="Путь для Side канала.")

    app_p = subparsers.add_parser("app", help="Веб-приложение")
    app_p.add_argument(
        "-p", "--port", type=int, default=None, help="Порт для запуска сервера Gradio."
    )
    app_p.add_argument(
        "-s", "--share",
        action="store_true",
        help="Создать публичную ссылку для приложения Gradio.",
    )
    app_p.add_argument(
        "-a", "--add_app",
        action="store_true",
        help="Включить вкладку с дополнительными приложениями",
    )
    app_p.add_argument(
        "-pl", "--use_plugins",
        action="store_true",
        help="Включить плагины",
    )
    app_p.add_argument(
        "-vb", "--vbach",
        action="store_true",
        help="Включить вкладку Vbach",
    )
    app_p.add_argument("-udir", "--user_dir", type=str, default=None, help="Путь к пользовательской папке")
    args = parser.parse_args()

    # 1. Список моделей
    if args.command == "info":
        if args.list_categories:
            mvsepless.get_list_categories()
        elif args.list_model_types:
            mvsepless.get_list_model_types()
        elif args.update:
            file_path = MvseplessModelManager().models_info_path
            url_link = "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/models.json?download=true"
            dw_file(url_link, file_path, retries=999999)
        else:
            mvsepless.get_list_supported_models(limit=args.limit, stem=args.stem, model_types=args.model_types, category=args.categories, only_installed=args.only_installed)
        sys.exit(0)

    # 2. Логика подкоманд
    if args.command == "auto_ensemble":
        ensemble_state = []
        if args.json:
            with open(args.json, 'r') as f:
                ensemble_state = json.load(f)
        else:
            for i, item in enumerate(args.model_list):
                parts = item.split(',')
                if len(parts) == 4:
                    parts[3] = float(parts[3])
                    ensemble_state.append(parts)
                else:
                    print(f"Ошибка в формате модели: {item}")
                    sys.exit(1)
        if not args.input:
            sys.exit(1)
        else:
            first_file = args.input[0]
            mvsepless.auto_ensemble(
                input_file=first_file,
                ensemble_state=ensemble_state,
                output_dir=args.output_dir,
                method=args.method,
                out_format=args.output_format,
                invert_ensemble=args.invert
            )

    elif args.command == "manual_ensemble":
        weights = args.weights if args.weights else [1.0] * len(args.input)
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
        if args.user_dir != "" and args.user_dir:
            user_directory.change_dir(args.user_dir)
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
            inbrowser=True
        )

    elif args.command == "separator":
        if args.install_only:
            mvsepless.install_model(args.model_name)
        else:
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
        parser.print_help()