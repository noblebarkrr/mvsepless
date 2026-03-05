import os, json, sys, subprocess, threading, time, argparse, gradio as gr, yaml, queue
from downloader import dw_file
from audio import check, output_formats
from device import all_ids, set_device, cuda_available
import torch
from check_colab import easy_check_is_colab
from packaging import version
is_pytorch2 = version.parse(torch.__version__) >= version.parse("2.0.0")
is_pytorch2_4 = version.parse(torch.__version__) >= version.parse("2.4.0")
unsupported_models = ["bs_inst_fno_unwa", "mbr_wsa"] if not is_pytorch2 else ["bs_inst_fno_unwa"] if not is_pytorch2_4 else []
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
MVSEPLESS_ECONOM = not cuda_available
MVSEPLESS_ECONOM_SEGMENT = int(os.environ.get("MVSEPLESS_ECONOM_SEGMENT", "7"))
def calculate_dimensions(chunk_size, hop_length=441):
    # Находим dim_t
    dim_t = (chunk_size // hop_length) + 1
    
    # Проверяем, чтобы chunk_size был кратен hop_length для идеального совмещения
    actual_chunk_size = (dim_t - 1) * hop_length
    
    return dim_t, actual_chunk_size

def generate_econom_params(sr=44100, seconds=MVSEPLESS_ECONOM_SEGMENT, hop_length=441):
    chunk_size = sr * seconds
    dim_t, chunk_size = calculate_dimensions(chunk_size, hop_length)
    return dim_t, chunk_size

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

    def get_list_mn_from_category(self, category: list, model_type: list | None = None):
        list_models = []
        if not model_type:
            list_models = [model for model in self.get_mn() if self.get_category(model) in category]
        else:
            list_models = [model for model in self.get_mn() if self.get_category(model) in category and self.get_mt(model) in model_type]
        return list_models


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
                    dim_t, new_chunk_size = generate_econom_params(hop_length=hop_length)
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
        model_type: str,
        model_name: str,
        mdx_denoise: bool = False,
        vr_aggr: bool = 5,
        vr_post_process: bool = False,
        vr_high_end_process: bool = False,
        econom_mode: bool = False,
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
            model_type,
            info["checkpoint_url"],
            info["config_url"],
        )
        self.conf_editor(conf, mdx_denoise, vr_aggr, vr_post_process, vr_high_end_process, model_type, econom_mode)

        return id, conf, ckpt
    
    def check_model(
        self,
        model_type: str,
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
            model_type,
            info["checkpoint_url"],
            info["config_url"],
            only_check_exists=True
        )
    
    def get_mn_dwloaded(self):
        return [model for model in self.get_mn() if self.check_model(self.get_mt(model), model)]

class Separator(MvseplessModelManager):

    def __init__(self):
        super().__init__()
        self.device = set_device()

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
        model_name: str = "Mel-Band-Roformer_Vocals_kimberley_jensen",
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
        model_name: str = "Mel-Band-Roformer_Vocals_kimberley_jensen",
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

        if output_dir:
            output_dir = os.path.abspath(output_dir)

        if selected_stems is None:
            selected_stems = []

        if not input:
            raise ValueError("Входной файл не указан")

        if "STEM" not in template and template is not None:
            template = template + "_STEM_"
        if not template:
            template = "mvsepless_NAME_(STEM)"

        model_type = self.get_mt(model_name)

        os.makedirs(output_dir, exist_ok=True)

        mdx_denoise = add_settings.get("mdx_denoise", False)
        vr_aggr = add_settings.get("vr_aggr", 5)
        vr_post_process = add_settings.get("vr_post_process", False)
        vr_high_end_process = add_settings.get("vr_high_end_process", False)
        econom_mode = add_settings.get("econom_mode", MVSEPLESS_ECONOM)
        add_progress_text_custom = add_settings.get("add_single_sep_text_progress", "")

        id, conf, ckpt = self.install_model(
            model_type, model_name, mdx_denoise, vr_aggr, vr_post_process, vr_high_end_process, econom_mode, progress
        )

        if isinstance(input, str):
            if not os.path.exists(input):
                raise ValueError(f"Входной файл не найден: {input}")

            if not check(input):
                raise ValueError("Входной файл не содержит аудио")

            basename = os.path.splitext(os.path.basename(input))[0]
            seped = self.separator_base(
                input_file=input,
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
            return seped

        elif isinstance(input, list):
            results = []
            for i, f in enumerate(input, 1):
                print(f"Файл {i} из {len(input)}: {f}")
                gr.Warning(title=f"Файл {i} из {len(input)}: {f}", message="")
                if os.path.exists(f):
                    if check(f):
                        basename = os.path.splitext(os.path.basename(f))[0]
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
                            add_text_progress=f"{i} из {len(input)}",
                        )
                        results.append([basename, seped])
            return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVSepless")
    parser.add_argument(
        "--input", type=str, required=True, help="Входной аудиофайл или каталог."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Каталог для выходных файлов."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Mel-Band-Roformer_Vocals_kimberley_jensen",
        help="Имя модели разделения.",
    )
    parser.add_argument(
        "--ext_inst", action="store_true", help="Извлечь инструментал."
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="mp3",
        choices=output_formats,
        help="Формат выходного файла.",
    )
    parser.add_argument(
        "--output_bitrate", type=str, default="320k", help="Битрейт выходного файла."
    )
    parser.add_argument(
        "--template",
        type=str,
        default="NAME (STEM) MODEL",
        help="Шаблон именования выходных файлов.",
    )
    parser.add_argument(
        "--selected_stems",
        type=str,
        nargs="*",
        default=None,
        help="Выбранные стемы для разделения.",
    )
    args = parser.parse_args()
    input_data = args.input
    if os.path.isdir(input_data):
        list_valid_files = []
        for file in os.listdir(args.input):
            if os.path.isfile(os.path.join(args.input, file)):
                if check(os.path.join(args.input, file)):
                    list_valid_files.append(os.path.join(args.input, file))

        input_files = list_valid_files
    else:
        input_files = input_data

    results = Separator().separate(
        input=input_files,
        output_dir=args.output_dir,
        model_name=args.model_name,
        ext_inst=args.ext_inst,
        output_format=args.output_format,
        output_bitrate=args.output_bitrate,
        template=args.template,
        selected_stems=args.selected_stems,
    )
    print("Разделение завершено.")
