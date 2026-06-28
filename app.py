import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*show_api.*") # Предупреждения скрыты
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*theme.*")
import gradio as gr
import sys
import json
import zipfile
from urllib.parse import urlparse
from pathlib import Path, PurePosixPath
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
from extra_utils import tz, define_audio_with_size, define_download_button_with_size, update_audio_with_size, base_c_params, easy_check_is_colab, get_gdrive_dir, one_element_list_to_value, dw_file, dw_yt_dlp, get_disk_usage
from inference import Separator, add_params, add_params_list, ensemble_types, BASE_DIR, get_stems_from_config_simple, custom_model_types, default_add_params
from vbach_lib.infer import VbachConverter, stereo_modes
from vbach_lib.f0_extractor import f0_methods, crepe_like_f0_methods, f0_extract_and_write
from vbach_lib.hubert_manager import download_hubert, huberts_fairseq, huberts_transformers
from audio import output_formats, get_audio_files_from_list, check, check_taglib_not_installed
from datetime import datetime
from namer import Namer
from i18n import _i18n
from args_parser import parse_app_args
import tempfile
import shutil
from tqdm import tqdm
from copy import deepcopy

def generate_add_params_component():
    add_params_components = []
    for tab, components in add_params.items():
          with gr.Tab(_i18n(tab)):
              for component_name, params in components.items():
                  component_type = params["component"]
                  if component_type == "slider":
                      add_params_components.append(gr.Slider(label=_i18n(component_name), minimum=params["minimum"], maximum=params["maximum"], step=params["step"], value=params["default"], info=_i18n(params.get("info", "")), **base_c_params["base"]))
                  elif component_type == "number":
                      add_params_components.append(gr.Number(label=_i18n(component_name), minimum=params["minimum"], maximum=params["maximum"], value=params["default"], info=_i18n(params.get("info", "")), **base_c_params["base"]))
                  elif component_type == "checkbox":
                      add_params_components.append(gr.Checkbox(label=_i18n(component_name), value=params["default"], info=_i18n(params.get("info", "")), **base_c_params["base"]))
    return add_params_components

USER_DIR = ""
GDRIVE_DIR = get_gdrive_dir()
def generate_user_dir_from_gdrive():
    global GDRIVE_DIR
    if GDRIVE_DIR:
        user_dir = Path(GDRIVE_DIR, "MyDrive", "mvsepless-data")
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir.as_posix()
    else:
        return None
GDRIVE_USER_DIR = generate_user_dir_from_gdrive()

def get_default_user_dir():
    if easy_check_is_colab():
        if GDRIVE_DIR:
            print(_i18n("gdrive_mount_found"))
            return GDRIVE_USER_DIR
        else:
            return USER_DIR
    else:
        return USER_DIR

DEFAULT_USER_DIR = get_default_user_dir()

def rename_user_dir_path(path: str, mode=0):
    global GDRIVE_USER_DIR, USER_DIR
    if path:
        if mode == 0:
            return (PurePosixPath(GDRIVE_USER_DIR) / PurePosixPath(path).relative_to(USER_DIR)).as_posix()
        elif mode == 1:
            return (PurePosixPath(USER_DIR) / PurePosixPath(path).relative_to(GDRIVE_USER_DIR)).as_posix()
    else:
        return None

base_names_app_dirs = (
    "input",
    "output_mvsepless",
    "history",
    "ensemble_flows",
    "vbach_models",
    "f0_curves",
    "custom_separation_models",
    "vbach_output",
    "iterative_ensemble_flows"
)

def copy_to_gdrive():
    global GDRIVE_DIR, GDRIVE_USER_DIR, USER_DIR
    if GDRIVE_DIR:
        copied_dirs = []
        dirs = [[dir, Path(USER_DIR, dir)] for dir in base_names_app_dirs]
        for (dir_name, dir_path) in tqdm(dirs, desc=_i18n("copy_to_gdrive"), unit=_i18n("dirs")):
            if dir_path.exists():
                shutil.copytree(dir_path, Path(GDRIVE_USER_DIR, dir_name), dirs_exist_ok=True)
                copied_dirs.append("")
        print(_i18n("copied_dirs")+": "+str(len(copied_dirs)))
        print(_i18n("copy_to_gdrive_done"))
        gr.Info(title=_i18n("copy_to_gdrive_done"), message="")

def copy_to_runtime():
    global GDRIVE_DIR, GDRIVE_USER_DIR, USER_DIR
    if GDRIVE_DIR:
        copied_dirs = []
        dirs = [[dir, Path(GDRIVE_USER_DIR, dir)] for dir in base_names_app_dirs]
        for (dir_name, dir_path) in tqdm(dirs, desc=_i18n("copy_to_current_user_dir"), unit=_i18n("dirs")):
            if dir_path.exists():
                shutil.copytree(dir_path, Path(USER_DIR, dir_name), dirs_exist_ok=True)
                copied_dirs.append("")
        print(_i18n("copied_dirs")+": "+str(len(copied_dirs)))
        print(_i18n("copy_to_gdrive_done"))
        gr.Info(title=_i18n("copy_to_gdrive_done"), message="")

def generate_zip_archive(files: str | Path | list[str | Path] | tuple[str | Path, ...], output_path: str | Path):

    if isinstance(files, (str, Path)):
        input_files = [files]
    else:
        input_files = files

    added_files = []

    output_path_ = Path(output_path)
    output_path_.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path_, mode="w") as zip_file:
        for path in input_files:
            p = Path(path)
            if p.exists():
                name = p.name
                name_list = zip_file.namelist()
                if name not in name_list:
                    zip_file.write(p, name)
                else:
                    zip_file.write(p, Namer.iter_in_list(Namer.short(name), name_list))
                added_files.append(name)

    print(_i18n("added_files")+": "+str(len(added_files)))
    return output_path_.as_posix()

def get_zip_output_path(name):
    temp_dir = Path(tempfile.gettempdir())
    timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
    return Namer.iter(temp_dir / f"{name}_{timestamp}.zip")


class UserDirectory:
    def __init__(self, custom_dir=USER_DIR):
        self.user_directory = Path(custom_dir if custom_dir else DEFAULT_USER_DIR)

    def change_dir(self, dir: str):
        self.user_directory = Path(dir)

    def generate(self, name: str):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        generated_directory = self.user_directory / name / timestamp
        generated_directory.mkdir(parents=True, exist_ok=True)
        return generated_directory
    
    def generate_from_dir(self, dir: str):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        generated_directory = Path(dir) / timestamp
        generated_directory.mkdir(parents=True, exist_ok=True)
        return generated_directory

class InputFilesDatabase(UserDirectory):
    def __init__(self):
        super().__init__()
        self.input_dir_base = self.user_directory / base_names_app_dirs[0]
        self.input_dir_base.mkdir(parents=True, exist_ok=True)
        self.input_base_json = self.input_dir_base / "inputs.json"
        self.input_base = []
        self.load()

    def _write_decorator(func):
        def wrapper(self, *args, **kwargs):
            results_ = func(self, *args, **kwargs)
            self.write()
            return results_
        return wrapper

    def _load_decorator(func):
        def wrapper(self, *args, **kwargs):
            self.load()
            results_ = func(self, *args, **kwargs)
            return results_
        return wrapper

    @_write_decorator
    def update_data(self, mode: int):
        current_data = deepcopy(self.input_base)
        new_data = []
        if self.input_base_json.exists():
            new_data: list = json.loads(self.input_base_json.read_text("utf-8"))

        new_data2 = []
        new_data_to_merge = []

        for file_path in new_data:
            new_data2.append(rename_user_dir_path(file_path, mode=mode))

        for path2 in new_data2:
            if path2 not in current_data:
                new_data_to_merge.append(path2)

        self.input_base = list(dict.fromkeys([*current_data, *new_data_to_merge]))

    def write(self):
        self.input_base_json.write_text(json.dumps(self.input_base, ensure_ascii=False, indent=4), encoding="utf-8")

    def load(self):
        if self.input_base_json.exists():
            self.input_base = json.loads(self.input_base_json.read_text("utf-8"))
            print(_i18n("input_base_loaded"))

    @_write_decorator
    def upload(self, files, copy=False):
        input_dir = self.generate_from_dir(self.input_dir_base)
        uploaded_input_files = []
        valid_files = get_audio_files_from_list(files, only_files=True)
        for file in valid_files:
            new_file = Namer.iter(input_dir / Path(file).name)
            if copy:
                shutil.copy2(file, new_file)
            else:
                shutil.move(file, new_file)
            uploaded_input_files.append(new_file)
        self.input_base.extend(uploaded_input_files)
        return uploaded_input_files

    @_write_decorator
    def clear(self):
        for path in self.input_base:
            Path(path).unlink(missing_ok=True)
        self.input_base.clear()
        print(_i18n("input_base_cleared"))

    def get_input_list(self):
        return list(reversed(self.input_base))
    
class OutputDir(UserDirectory):
    def __init__(self, dir: str = base_names_app_dirs[1]):
        super().__init__()
        self.output_dir_name = dir

    def gen_output_dir(self):
        return self.generate(self.output_dir_name)

class History(UserDirectory):
    def __init__(self, name: str = "mvsepless"):
        super().__init__()
        self.history_dir_base = self.user_directory / base_names_app_dirs[2]
        self.history_dir_base.mkdir(parents=True, exist_ok=True)
        self.history_dict_json = self.history_dir_base / f"{name}.json"
        self.history_dict = {}
        self.load()

    def _write_decorator(func):
        def wrapper(self, *args, **kwargs):
            results_ = func(self, *args, **kwargs)
            self.write()
            return results_
        return wrapper

    def _load_decorator(func):
        def wrapper(self, *args, **kwargs):
            self.load()
            results_ = func(self, *args, **kwargs)
            return results_
        return wrapper

    def write(self):
        self.history_dict_json.write_text(json.dumps(self.history_dict, ensure_ascii=False, indent=4), encoding="utf-8")

    def load(self):
        if self.history_dict_json.exists():
            self.history_dict = json.loads(self.history_dict_json.read_text("utf-8"))
            print(_i18n("history_loaded"))

    @_write_decorator
    def update_data(self, mode: int):
        current_data = deepcopy(self.history_dict)
        new_data = {}
        if self.history_dict_json.exists():
            new_data: dict = json.loads(self.history_dict_json.read_text("utf-8"))
            
        new_data_to_merge = {}

        for key, state in new_data.items():
            new_state = []
            for basename, stems_list in state:
                new_stems_list = [basename]
                new_stems_list.append([[stem_name, rename_user_dir_path(stem_path, mode=mode)] for stem_name, stem_path in stems_list])
                new_state.append(deepcopy(new_stems_list))
            new_data[key] = deepcopy(new_state)

        for key2, state2 in new_data.items():
            if key2 not in list(current_data.keys()) and state2 != current_data.get(key2):
                new_data_to_merge[key2] = state2

        self.history_dict: dict = {
            **current_data,
            **new_data_to_merge
        }

    def get_list(self, update_from_file=False):
        if update_from_file:
            self.load()
        return deepcopy(list(reversed([key_h for key_h in self.history_dict])))

    @_write_decorator
    def add_to_history(self, model_name: str, state: list):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        self.history_dict.update([(f"{timestamp} | {model_name}", deepcopy(state))])

    def get_from_history(self, key: str):
        return deepcopy(self.history_dict.get(key, None))

class HistoryAutoEnsemble(History):
    def __init__(self):
        super().__init__("ensembless")

    def _write_decorator(func):
        def wrapper(self, *args, **kwargs):
            results_ = func(self, *args, **kwargs)
            self.write()
            return results_
        return wrapper

    def _load_decorator(func):
        def wrapper(self, *args, **kwargs):
            self.load()
            results_ = func(self, *args, **kwargs)
            return results_
        return wrapper
    
    @_write_decorator
    def update_data(self, mode: int):
        current_data = deepcopy(self.history_dict)
        new_data = {}
        if self.history_dict_json.exists():
            new_data: dict = json.loads(self.history_dict_json.read_text("utf-8"))
        new_data_to_merge = {}

        for key, state in new_data.items():
            new_state = [
                rename_user_dir_path(state[0], mode=mode),  # result
                rename_user_dir_path(state[1], mode=mode),  # invert
                [rename_user_dir_path(stem_path, mode=mode) for stem_path in state[2]]  # primary_stems_list
            ]
            new_data[key] = deepcopy(new_state)
        for key2, state2 in new_data.items():
            if key2 not in list(current_data.keys()) and state2 != current_data.get(key2):
                new_data_to_merge[key2] = state2

        self.history_dict: dict = {
            **current_data,
            **new_data_to_merge
        }

    @_write_decorator
    def add_to_history(self, etype: str, output: str, inverted_output: str, primary_stems_list: list = []):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        self.history_dict.update([(f"{timestamp} | {etype}", (output, inverted_output, primary_stems_list))])

    def get_from_history(self, key: str):
        return deepcopy(self.history_dict.get(key, (None, None, [])))

class HistoryManualEnsemble(History):
    def __init__(self):
        super().__init__("manual_ensembless")

    def _write_decorator(func):
        def wrapper(self, *args, **kwargs):
            results_ = func(self, *args, **kwargs)
            self.write()
            return results_
        return wrapper

    def _load_decorator(func):
        def wrapper(self, *args, **kwargs):
            self.load()
            results_ = func(self, *args, **kwargs)
            return results_
        return wrapper

    @_write_decorator
    def update_data(self, mode: int):
        current_data = deepcopy(self.history_dict)
        new_data = {}
        if self.history_dict_json.exists():
            new_data: dict = json.loads(self.history_dict_json.read_text("utf-8"))
        new_data_to_merge = {}

        for key, state in new_data.items():
            new_state = None
            if state:
                new_state = rename_user_dir_path(state, mode=mode)
            new_data[key] = deepcopy(new_state)

        for key2, state2 in new_data.items():
            if key2 not in list(current_data.keys()) and state2 != current_data.get(key2):
                new_data_to_merge[key2] = state2

        self.history_dict: dict = {
            **current_data,
            **new_data_to_merge
        }

    @_write_decorator
    def add_to_history(self, etype: str, state: str):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        self.history_dict.update([(f"{timestamp} | {etype}", deepcopy(state))])

    def get_from_history(self, key: str):
        return deepcopy(self.history_dict.get(key, None))

class HistorySubtractor(History):
    def __init__(self):
        super().__init__("subtract")

    def _write_decorator(func):
        def wrapper(self, *args, **kwargs):
            results_ = func(self, *args, **kwargs)
            self.write()
            return results_
        return wrapper

    def _load_decorator(func):
        def wrapper(self, *args, **kwargs):
            self.load()
            results_ = func(self, *args, **kwargs)
            return results_
        return wrapper

    @_write_decorator
    def update_data(self, mode: int):
        current_data = deepcopy(self.history_dict)
        new_data = {}
        if self.history_dict_json.exists():
            new_data: dict = json.loads(self.history_dict_json.read_text("utf-8"))
        new_data_to_merge = {}

        for key, state in new_data.items():
            new_state = None
            if state:
                new_state = rename_user_dir_path(state, mode=mode)
            new_data[key] = deepcopy(new_state)

        for key2, state2 in new_data.items():
            if key2 not in list(current_data.keys()) and state2 != current_data.get(key2):
                new_data_to_merge[key2] = state2

        self.history_dict: dict = {
            **current_data,
            **new_data_to_merge
        }

    @_write_decorator
    def add_to_history(self, itype: str, state: str):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        self.history_dict.update([(f"{timestamp} | {itype}", deepcopy(state))])

class HistoryVbach(History):
    def __init__(self):
        super().__init__("vbach")

    def _write_decorator(func):
        def wrapper(self, *args, **kwargs):
            results_ = func(self, *args, **kwargs)
            self.write()
            return results_
        return wrapper

    def _load_decorator(func):
        def wrapper(self, *args, **kwargs):
            self.load()
            results_ = func(self, *args, **kwargs)
            return results_
        return wrapper

    @_write_decorator
    def update_data(self, mode: int):
        current_data = deepcopy(self.history_dict)
        new_data = {}
        if self.history_dict_json.exists():
            new_data: dict = json.loads(self.history_dict_json.read_text("utf-8"))
        new_data_to_merge = {}

        for key, state in new_data.items():
            new_state = []
            if state:
                new_state = [rename_user_dir_path(file_path, mode=mode) for file_path in state]
            new_data[key] = deepcopy(new_state)

        for key2, state2 in new_data.items():
            if key2 not in list(current_data.keys()) and state2 != current_data.get(key2):
                new_data_to_merge[key2] = state2

        self.history_dict: dict = {
            **current_data,
            **new_data_to_merge
        }

    @_write_decorator
    def add_to_history(self, model_name: str, f0_method: str, pitch: int, output_files: list):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        self.history_dict.update([(f"{timestamp} | {model_name} | {f0_method} | {pitch}", deepcopy(output_files))])

    def get_from_history(self, key: str):
        return deepcopy(self.history_dict.get(key, []))

class HistoryIterativeEnsemble(History):
    def __init__(self):
        super().__init__("iterative_ensembless")
    
    def _write_decorator(func):
        def wrapper(self, *args, **kwargs):
            results_ = func(self, *args, **kwargs)
            self.write()
            return results_
        return wrapper

    def _load_decorator(func):
        def wrapper(self, *args, **kwargs):
            self.load()
            results_ = func(self, *args, **kwargs)
            return results_
        return wrapper
    
    @_write_decorator
    def update_data(self, mode: int):
        current_data = deepcopy(self.history_dict)
        new_data = {}
        if self.history_dict_json.exists():
            new_data: dict = json.loads(self.history_dict_json.read_text("utf-8"))
        new_data_to_merge = {}

        for key, state in new_data.items():
            new_state = None
            if state:
                new_state = [
                    rename_user_dir_path(state[0], mode=mode),  # result
                    [rename_user_dir_path(path, mode=mode) for path in state[1]]  # intermediate files
                ]
            new_data[key] = deepcopy(new_state)

        for key2, state2 in new_data.items():
            if key2 not in list(current_data.keys()) and state2 != current_data.get(key2):
                new_data_to_merge[key2] = state2

        self.history_dict: dict = {
            **current_data,
            **new_data_to_merge
        }

    @_write_decorator
    def add_to_history(self, result_path: str, intermediate_files: list, count_iters: int):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        self.history_dict.update([(f"{timestamp} | {count_iters} iterations", (result_path, intermediate_files))])

    def get_from_history(self, key: str):
        state = self.history_dict.get(key, (None, []))
        if state:
            return state[0], state[1]
        return None, []

class AutoEnsembleApp(UserDirectory, Separator):
    def __init__(self):
        UserDirectory.__init__(self)
        Separator.__init__(self)
        self.ensemble_base = self.user_directory / base_names_app_dirs[3]
        self.ensemble_base.mkdir(parents=True, exist_ok=True)
        
    def write_flow(self, name: str, state: list[list]):
        if name:
            path = self.ensemble_base / f"{Namer.sanitize(name)}.json"
            path.write_text(json.dumps(state, ensure_ascii=False, indent=4), encoding="utf-8")
            print(_i18n("ensemble_flow_saved")+": "+name)
            gr.Info(title=_i18n("ensemble_flow_saved")+": "+name, message="")
        else:
            print(_i18n("name_not_specified"))
            gr.Warning(title=_i18n("name_not_specified"), message="")

    def load_flow(self, name: str):
        if name:
            path = self.ensemble_base / f"{Namer.sanitize(name)}.json"
            if path.exists():
                try:
                    state = json.loads(path.read_text("utf-8"))
                    state, warns_str = self.validate_flow(state, non_exists_warn=True)
                    print(warns_str)
                    print(_i18n("ensemble_flow_loaded")+": "+name)
                    gr.Info(title=_i18n("ensemble_flow_loaded")+": "+name, message="")
                    if warns_str:
                        gr.Warning(message="<b>"+warns_str.replace("\n", "<br> <br>")+"</b>", title="")
                        gr.Warning(title=_i18n("ensemble_run_error_with_incorrect_flow"), message="")
                    return state
                except Exception as e:
                    print(_i18n("ensemble_flow_invalid")+": "+name)
                    print(e)
                    gr.Warning(title=_i18n("ensemble_flow_invalid")+": "+name, message="")
                    return []
                
            else:
                print(_i18n("ensemble_flow_not_exist")+": "+name)
                gr.Warning(title=_i18n("ensemble_flow_not_exist")+": "+name, message="")
                return []
        else:
            print(_i18n("name_not_specified"))
            gr.Warning(title=_i18n("name_not_specified"), message="")
            return []

    def import_flows(self, flows: list):
        added_flows = []
        if flows:
            for flow in flows:
                path = Path(flow)
                if path.exists() and path.suffix == ".json":
                    new_path = Namer.iter(self.ensemble_base / path.name)
                    shutil.move(path, new_path)
                    added_flows.append(new_path)
        status = _i18n("flows_imported")+": "+str(len(added_flows))
        print(status)
        gr.Info(title=status, message="")

    def export_flow(self, flow: str):
        if flow:
            path = self.ensemble_base / f"{flow}.json"
            if path.exists():
                return path.as_posix()
            else:
                return None
        else:
            return None

    def clear_flows(self):
        for flow in self.ensemble_base.glob("*.json"):
            flow.unlink(missing_ok=True)
        status = _i18n("all_ensemble_flow_cleared")
        print(status)
        gr.Info(title=status, message="")

    def get_flows(self):
        flows = [flow.stem for flow in self.ensemble_base.glob("*.json")]
        return flows

    def add_model(self, model_name: str, primary_stem: str, invert: bool, weight: float, state: list[list]):
        state.append([model_name, primary_stem, invert, weight])
        print(_i18n("ae_added_model")+": "+model_name)
        gr.Info(title=_i18n("ae_added_model")+": "+model_name, message="")
        return state

    def replace_model(self, number: int, model_name: str, primary_stem: str, invert: bool, weight: float, state: list[list]):
        index = number - 1
        if index < len(state):
            state[index] = [model_name, primary_stem, invert, weight]
            print(_i18n("ae_replaced_model")+": "+model_name+"/"+str(index))
            gr.Info(title=_i18n("ae_replaced_model")+": "+model_name+"/"+str(number), message="")
        return state

    def insert_model(self, number: int, model_name: str, primary_stem: str, invert: bool, weight: float, state: list[list]):
        index = number - 1
        if index < len(state):
            state.insert(index, [model_name, primary_stem, invert, weight])
            print(_i18n("ae_inserted_model")+": "+model_name+"/"+str(index))
            gr.Info(title=_i18n("ae_inserted_model")+": "+model_name+"/"+str(number), message="")
        return state

    def delete_model(self, number: int, state: list[list]):
        index = number - 1
        if index < len(state):
            deleted_value = state.pop(index)
            print(_i18n("ae_deleted_model")+": "+deleted_value[0])
            gr.Info(title=_i18n("ae_deleted_model")+": "+deleted_value[0], message="")
        return state

    def clear_all_model(self, state: list[list]):
        state.clear()
        print(_i18n("ae_all_cleared"))
        gr.Info(title=_i18n("ae_all_cleared"), message="")
        return state

    def get_models(self, state: list[list]):
        return state
    
    def get_models_df(self, state: list[list]):
        models = state
        models_df = [[num+1] for num in range(len(models))]

        for model, model_df in list(zip(models, models_df)):
            model_df.extend(model)
        return models_df

class IterativeEnsembleApp(UserDirectory, Separator):
    def __init__(self):
        UserDirectory.__init__(self)
        Separator.__init__(self)
        self.ensemble_base = self.user_directory / base_names_app_dirs[8]  # iterative_ensemble_flows
        self.ensemble_base.mkdir(parents=True, exist_ok=True)
        
    def write_flow(self, name: str, state: list[list]):
        if name:
            path = self.ensemble_base / f"{Namer.sanitize(name)}.json"
            path.write_text(json.dumps(state, ensure_ascii=False, indent=4), encoding="utf-8")
            print(_i18n("ensemble_flow_saved")+": "+name)
            gr.Info(title=_i18n("ensemble_flow_saved")+": "+name, message="")
        else:
            print(_i18n("name_not_specified"))
            gr.Warning(title=_i18n("name_not_specified"), message="")

    def load_flow(self, name: str):
        if name:
            path = self.ensemble_base / f"{Namer.sanitize(name)}.json"
            if path.exists():
                try:
                    state = json.loads(path.read_text("utf-8"))
                    state, warns_str = self.validate_flow(state, non_exists_warn=True, iterative=True)
                    print(warns_str)
                    print(_i18n("ensemble_flow_loaded")+": "+name)
                    gr.Info(title=_i18n("ensemble_flow_loaded")+": "+name, message="")
                    if warns_str:
                        gr.Warning(message="<b>"+warns_str.replace("\n", "<br> <br>")+"</b>", title="")
                        gr.Warning(title=_i18n("ensemble_run_error_with_incorrect_flow"), message="")
                    return state
                except Exception as e:
                    print(_i18n("ensemble_flow_invalid")+": "+name)
                    print(e)
                    gr.Warning(title=_i18n("ensemble_flow_invalid")+": "+name, message="")
                    return []
            else:
                print(_i18n("ensemble_flow_not_exist")+": "+name)
                gr.Warning(title=_i18n("ensemble_flow_not_exist")+": "+name, message="")
                return []
        else:
            print(_i18n("name_not_specified"))
            gr.Warning(title=_i18n("name_not_specified"), message="")
            return []

    def import_flows(self, flows: list):
        added_flows = []
        if flows:
            for flow in flows:
                path = Path(flow)
                if path.exists() and path.suffix == ".json":
                    new_path = Namer.iter(self.ensemble_base / path.name)
                    shutil.move(path, new_path)
                    added_flows.append(new_path)
        status = _i18n("flows_imported")+": "+str(len(added_flows))
        print(status)
        gr.Info(title=status, message="")

    def export_flow(self, flow: str):
        if flow:
            path = self.ensemble_base / f"{flow}.json"
            if path.exists():
                return path.as_posix()
            else:
                return None
        else:
            return None

    def clear_flows(self):
        for flow in self.ensemble_base.glob("*.json"):
            flow.unlink(missing_ok=True)
        status = _i18n("all_ensemble_flow_cleared")
        print(status)
        gr.Info(title=status, message="")

    def get_flows(self):
        flows = [flow.stem for flow in self.ensemble_base.glob("*.json")]
        return flows

    def add_model(self, model_name: str, primary_stem: str, invert: bool, state: list[list]):
        state.append([model_name, primary_stem, invert])
        print(_i18n("ae_added_model")+": "+model_name)
        gr.Info(title=_i18n("ae_added_model")+": "+model_name, message="")
        return state

    def replace_model(self, number: int, model_name: str, primary_stem: str, invert: bool, state: list[list]):
        index = number - 1
        if index < len(state):
            state[index] = [model_name, primary_stem, invert]
            print(_i18n("ae_replaced_model")+": "+model_name+"/"+str(index))
            gr.Info(title=_i18n("ae_replaced_model")+": "+model_name+"/"+str(number), message="")
        return state

    def insert_model(self, number: int, model_name: str, primary_stem: str, invert: bool, state: list[list]):
        index = number - 1
        if index < len(state):
            state.insert(index, [model_name, primary_stem, invert])
            print(_i18n("ae_inserted_model")+": "+model_name+"/"+str(index))
            gr.Info(title=_i18n("ae_inserted_model")+": "+model_name+"/"+str(number), message="")
        return state

    def delete_model(self, number: int, state: list[list]):
        index = number - 1
        if index < len(state):
            deleted_value = state.pop(index)
            print(_i18n("ae_deleted_model")+": "+deleted_value[0])
            gr.Info(title=_i18n("ae_deleted_model")+": "+deleted_value[0], message="")
        return state

    def clear_all_model(self, state: list[list]):
        state.clear()
        print(_i18n("ae_all_cleared"))
        gr.Info(title=_i18n("ae_all_cleared"), message="")
        return state

    def get_models(self, state: list[list]):
        return state
    
    def get_models_df(self, state: list[list]):
        models = state
        models_df = [[num+1] for num in range(len(models))]
        for model, model_df in list(zip(models, models_df)):
            model_df.extend(model)
        return models_df

class VbachModelsDir(UserDirectory):
    """Manage Vbach models directory and model discovery"""
    
    def __init__(self):
        super().__init__()
        self.vbach_models_base = self.user_directory / base_names_app_dirs[4]
        self.pth_models_dir = self.vbach_models_base / "pth"
        self.index_models_dir = self.vbach_models_base / "index"
        self.pth_models_dir.mkdir(parents=True, exist_ok=True)
        self.index_models_dir.mkdir(parents=True, exist_ok=True)
        
        self.supported_extensions = (".pth", ".ckpt", ".pt", ".th", ".chpt")
        
    def get_pth_models(self):
        """Get list of available checkpoint models"""
        models = []
        for ext in self.supported_extensions:
            models.extend([f.as_posix() for f in self.pth_models_dir.glob(f"*{ext}")])
        return models
    
    def get_index_files(self):
        """Get list of available index files"""
        return [f.as_posix() for f in self.index_models_dir.glob("*.index")]

    def extract_zip(self, zip_path: str | Path):
        status = ""
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_extracted_path = Path(tmpdirname)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_extracted_path)
                print(_i18n("vbach_model_zip_unpacked"))
                gr.Info(title=_i18n("vbach_model_zip_unpacked"), message="")
            indexes = []
            pths = []
            for ext in self.supported_extensions:
                pths.extend([f.as_posix() for f in tmp_extracted_path.rglob(f"*{ext}")])
            indexes.extend([f.as_posix() for f in tmp_extracted_path.rglob("*.index")])
            if not pths and not indexes:
                print(_i18n("vbach_model_zip_not_model_files"))
                gr.Info(title=_i18n("vbach_model_zip_not_model_files"), message="")
            status += self.upload_index_model(indexes)+"\n"
            status += self.upload_pth_model(pths)+"\n"
        return status

    def get_pth_name_from_link(self, url):
        clean_url = urlparse(url)._replace(query="", fragment="").geturl()
        file_name = PurePosixPath(PurePosixPath(clean_url).name)
        if file_name.suffix not in self.supported_extensions:
            file_name = file_name.with_suffix(".pth")
        return str(file_name)
    
    def get_index_name_from_link(self, url):
        clean_url = urlparse(url)._replace(query="", fragment="").geturl()
        file_name = PurePosixPath(PurePosixPath(clean_url).name)
        if file_name.suffix != ".index":
            file_name = file_name.with_suffix(".index")
        return str(file_name)

    def download_model(self, zip_url=None, pth_url=None, index_url=None):
        status = ""
        if zip_url:
            temp_zip = Path(tempfile.mkstemp(suffix=".zip")[1])
            dw_file(zip_url, temp_zip)
            status = self.extract_zip(temp_zip)
        else:
            index_status = None
            pth_status = None
            if index_url:
                dw_file(index_url, Namer.iter(self.index_models_dir / self.get_index_name_from_link(index_url)))
                index_status = _i18n("vbach_model_index_downloaded")+"\n"
            if pth_url:
                dw_file(pth_url, Namer.iter(self.pth_models_dir / self.get_pth_name_from_link(pth_url)))
                pth_status = _i18n("vbach_model_pth_downloaded")+"\n"
            status_list = [status_unit for status_unit in [index_status, pth_status] if status_unit]
            status = "".join(status_list).rsplit("\n", maxsplit=1)[0]
            print(status)
            gr.Info(message="<b>"+status.replace("\n", "<br>")+"</b>", title="")
        return status

    def upload_pth_model(self, pth_paths: str | Path | list[str | Path]):
        added_files = []
        if isinstance(pth_paths, (str, Path)):
            pth_paths = [pth_paths]

        for p_str in pth_paths:
            p = Path(p_str)
            dst = Namer.iter(self.pth_models_dir / p.name)
            if p.suffix in self.supported_extensions:
                shutil.move(p, dst)
                added_files.append(dst)
        status = _i18n("vbach_added_pths")+": "+str(len(added_files))
        print(status)
        gr.Info(message="<b>"+status.replace("\n", "<br>")+"</b>", title="")
        return status

    def upload_index_model(self, index_paths: str | Path):
        added_files = []
        if isinstance(index_paths, (str, Path)):
            index_paths = [index_paths]

        for p_str in index_paths:
            p = Path(p_str)
            if p.exists():
                dst = Namer.iter(self.index_models_dir / p.name)
                if p.suffix in [".index"]:
                    shutil.move(p, dst)
                    added_files.append(dst)
        status = _i18n("vbach_added_indexes")+": "+str(len(added_files))
        print(status)
        gr.Info(message="<b>"+status.replace("\n", "<br>")+"</b>", title="")
        return status

    def get_model_name(self, model_path: str | Path) -> str:
        """Extract model name from path"""
        return Path(model_path).stem

class F0GenerateOutPath(UserDirectory):
    def __init__(self):
        super().__init__()
        self.f0_curves_dir = self.user_directory / base_names_app_dirs[5]
        self.f0_curves_dir.mkdir(parents=True, exist_ok=True)

    def generate_output_path(self, name: str, f0_method: str):
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        generated_path = self.f0_curves_dir / f"{timestamp}_{f0_method}_{Namer.short(name, length=90)}.json"
        return generated_path.as_posix()

class CustomSeparationModelsDir(UserDirectory):
    """Manage custom separation models directory and model discovery"""
    
    def __init__(self):
        super().__init__()
        self.custom_models_base = self.user_directory / base_names_app_dirs[6]
        self.checkpoints_dir = self.custom_models_base / "checkpoints"
        self.configs_dir = self.custom_models_base / "configs"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        
        self.supported_extensions = (".pth", ".ckpt", ".pt", ".th", ".chpt")
        self.config_extensions = (".yaml", ".yml")
        
    def get_checkpoints(self):
        """Get list of available checkpoint files"""
        checkpoints = []
        for ext in self.supported_extensions:
            checkpoints.extend([f.as_posix() for f in self.checkpoints_dir.glob(f"*{ext}")])
        return checkpoints
    
    def get_configs(self):
        """Get list of available config files"""
        configs = []
        for ext in self.config_extensions:
            configs.extend([f.as_posix() for f in self.configs_dir.glob(f"*{ext}")])
        return configs

    def get_checkpoint_name_from_link(self, url):
        clean_url = urlparse(url)._replace(query="", fragment="").geturl()
        file_name = PurePosixPath(PurePosixPath(clean_url).name)
        if file_name.suffix not in self.supported_extensions:
            file_name = file_name.with_suffix(".pth")
        return str(file_name)
    
    def get_config_name_from_link(self, url):
        clean_url = urlparse(url)._replace(query="", fragment="").geturl()
        file_name = PurePosixPath(PurePosixPath(clean_url).name)
        if file_name.suffix not in self.config_extensions:
            file_name = file_name.with_suffix(".yaml")
        return str(file_name)

    def download_model(self, zip_url=None, checkpoint_url=None, config_url=None):
        """Download model from URL"""
        status = ""
        config_status = None
        checkpoint_status = None
        if config_url:
            dw_file(config_url, Namer.iter(self.configs_dir / self.get_config_name_from_link(config_url)))
            config_status = _i18n("custom_model_config_downloaded")+"\n"
        if checkpoint_url:
            dw_file(checkpoint_url, Namer.iter(self.checkpoints_dir / self.get_checkpoint_name_from_link(checkpoint_url)))
            checkpoint_status = _i18n("custom_model_checkpoint_downloaded")+"\n"
        status_list = [status_unit for status_unit in [config_status, checkpoint_status] if status_unit]
        status = "".join(status_list).rsplit("\n", maxsplit=1)[0]
        print(status)
        gr.Info(message="<b>"+status.replace("\n", "<br>")+"</b>", title="")
        return status

    def upload_checkpoint(self, checkpoint_paths: str | Path | list[str | Path]):
        """Upload checkpoint files"""
        added_files = []
        if isinstance(checkpoint_paths, (str, Path)):
            checkpoint_paths = [checkpoint_paths]

        for p_str in checkpoint_paths:
            p = Path(p_str)
            dst = Namer.iter(self.checkpoints_dir / p.name)
            if p.suffix in self.supported_extensions:
                shutil.move(p, dst)
                added_files.append(dst)
        status = _i18n("custom_added_checkpoints")+": "+str(len(added_files))
        print(status)
        gr.Info(message="<b>"+status.replace("\n", "<br>")+"</b>", title="")
        return status

    def upload_config(self, config_paths: str | Path | list[str | Path]):
        """Upload config files"""
        added_files = []
        if isinstance(config_paths, (str, Path)):
            config_paths = [config_paths]

        for p_str in config_paths:
            p = Path(p_str)
            if p.exists():
                dst = Namer.iter(self.configs_dir / p.name)
                if p.suffix in self.config_extensions:
                    shutil.move(p, dst)
                    added_files.append(dst)
        status = _i18n("custom_added_configs")+": "+str(len(added_files))
        print(status)
        gr.Info(message="<b>"+status.replace("\n", "<br>")+"</b>", title="")
        return status

    def get_model_pair(self, checkpoint_path: str, config_path: str) -> tuple[str, str]:
        """Get checkpoint and config paths"""
        return checkpoint_path, config_path

class App(Separator):
    def __init__(self):
        super().__init__()
        self.input_files = InputFilesDatabase()
        self.output_dir = OutputDir()
        self.history = History()
        self.vbach_converter = VbachConverter()
        self.vbach_model_manager = VbachModelsDir()
        self.auto_ensemble_app = AutoEnsembleApp()
        self.auto_ensemble_history_app = HistoryAutoEnsemble()
        self.manual_ensemble_history_app = HistoryManualEnsemble()
        self.subtract_history_app = HistorySubtractor()
        self.vbach_history_app = HistoryVbach()
        self.f0_gen_output_path = F0GenerateOutPath()
        self.custom_sep_model_manager = CustomSeparationModelsDir()
        self.iterative_ensemble_app = IterativeEnsembleApp()
        self.iterative_ensemble_history_app = HistoryIterativeEnsemble()
        self.add_params_dict = {}

    def update_model_name(self, model_name):
        stems = self.get_stems(model_name)
        return gr.update(value=False, visible=len(stems) > 2), gr.update(value=[], choices=stems)

    def update_model_name_ensemble(self, model_name):
        stems = self.get_stems(model_name)
        if stems:
            first_value = stems[0]
        else:
            first_value = None
        return gr.update(value=first_value, choices=stems)

    def update_add_params(self, *add_params_values):
        return dict(zip(add_params_list, add_params_values))

    def get_actual_history_list(self, value, state):
        current_history = self.history.get_list()
        if current_history == state:
            return gr.skip()
        return gr.update(choices=current_history, value=value), current_history

    def get_actual_auto_ensemble_history_list(self, value, state):
        current_history = self.auto_ensemble_history_app.get_list()
        if current_history == state:
            return gr.skip()
        return gr.update(choices=current_history, value=value), current_history
    
    def get_actual_manual_ensemble_history_list(self, value, state):
        current_history = self.manual_ensemble_history_app.get_list()
        if current_history == state:
            return gr.skip()
        return gr.update(choices=current_history, value=value), current_history

    def get_actual_auto_ensemble_flows_list(self, value, state):
        current_flows = self.auto_ensemble_app.get_flows()
        if current_flows == state:
            return gr.skip()
        return gr.update(choices=current_flows, value=value), current_flows

    def get_actual_subtract_history_list(self, value, state):
        current_history = self.subtract_history_app.get_list()
        if current_history == state:
            return gr.skip()
        return gr.update(choices=current_history, value=value), current_history

    def get_actual_input_list(self, value, state):
        current_files = self.input_files.get_input_list()
        if current_files == state:
            return gr.skip()
        return gr.update(choices=current_files, value=value), current_files

    def get_actual_vbach_history_list(self, value, state):
        """Get updated history list"""
        current_history = self.vbach_history_app.get_list()
        if current_history == state:
            return gr.skip()
        return gr.update(choices=current_history, value=value), current_history
    
    def get_actual_vbach_models_list(self, value, state):
        """Get updated models list"""
        current_models = self.vbach_model_manager.get_pth_models()
        if current_models == state:
            return gr.skip()
        return gr.update(choices=current_models, value=value), current_models
    
    def get_actual_vbach_index_list(self, value, state):
        """Get updated index files list"""
        current_indexes = self.vbach_model_manager.get_index_files()
        if current_indexes == state:
            return gr.skip()
        return gr.update(choices=current_indexes, value=value), current_indexes

    def get_actual_custom_sep_checkpoints_list(self, value, state):
        """Get updated checkpoints list"""
        current_checkpoints = self.custom_sep_model_manager.get_checkpoints()
        if current_checkpoints == state:
            return gr.skip()
        return gr.update(choices=current_checkpoints, value=value), current_checkpoints

    def get_actual_custom_sep_configs_list(self, value, state):
        """Get updated configs list"""
        current_configs = self.custom_sep_model_manager.get_configs()
        if current_configs == state:
            return gr.skip()
        return gr.update(choices=current_configs, value=value), current_configs

    def get_actual_iterative_ensemble_history_list(self, value, state):
        current_history = self.iterative_ensemble_history_app.get_list()
        if current_history == state:
            return gr.skip()
        return gr.update(choices=current_history, value=value), current_history

    def get_actual_iterative_ensemble_flows_list(self, value, state):
        current_flows = self.iterative_ensemble_app.get_flows()
        if current_flows == state:
            return gr.skip()
        return gr.update(choices=current_flows, value=value), current_flows

    def UI(self, theme=None, hf_space_mode=False) -> gr.Blocks:
        global GDRIVE_DIR, IS_CUSTOM_DIR
        all_models = self.get_all_models()
        default_model = all_models[0]
        stems_default = self.get_stems(default_model)
        ext_inst_visible_default = len(stems_default) > 2
        with gr.Blocks(theme=theme) as mvsepless_app:
            sep_input_state = gr.State([])
            auto_ensemble_input_state = gr.State([])
            manual_ensemble_input_state = gr.State([])
            sep_history_list_state = gr.State([])
            auto_ensemble_history_state = gr.State([])
            manual_ensemble_history_state = gr.State([])
            subtract_1_input_state = gr.State([])
            subtract_2_input_state = gr.State([])
            subtract_history_state = gr.State([])
            vbach_input_state = gr.State([])
            vbach_history_state = gr.State([])
            vbach_models_state = gr.State([])
            vbach_index_state = gr.State([])
            vbach_custom_models_state = gr.State([])
            vbach_custom_index_state = gr.State([])
            vbach_custom_input_state = gr.State([])
            f0_input_state = gr.State([])
            custom_sep_input_state = gr.State([])
            custom_sep_history_state = gr.State([])
            custom_sep_checkpoints_state = gr.State([])
            custom_sep_configs_state = gr.State([])
            iterative_ensemble_input_state = gr.State([])
            iterative_ensemble_history_state = gr.State([])
            iterative_ensemble_flows_state = gr.State([])
            with gr.Tab(_i18n("separation_tab")):
                sep_state = gr.State()
                with gr.Row():
                    with gr.Column():
                        sep_upload_files = gr.File(show_label=False, **base_c_params["input_files_multi"])
                        with gr.Group():
                            sep_input_files = gr.Dropdown(container=False, allow_custom_value=True, **base_c_params["dropdown_multi"])
                            sep_input_files.focus(self.get_actual_input_list, inputs=[sep_input_files, sep_input_state], outputs=[sep_input_files, sep_input_state], show_progress="hidden")
                            sep_add_uploaded_files = gr.Checkbox(label=_i18n("add_uploaded_files_to_current_list"), value=False, **base_c_params["base"])
                            sep_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                            @sep_upload_files.upload(inputs=[sep_upload_files, sep_input_files, sep_add_uploaded_files], outputs=[sep_upload_files, sep_input_files])
                            def upload_files_fn(files: list, input_files: list, sep_add_uploaded_files: bool):
                                uploaded_files = self.input_files.upload(files)
                                if sep_add_uploaded_files:
                                    return gr.update(value=None), gr.update(choices=self.input_files.get_input_list(), value=[*uploaded_files, *input_files])
                                return gr.update(value=None), gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)
                            @gr.render(inputs=[sep_input_files, sep_input_preview_check])
                            def preview_inputs(input: list, preview: bool):
                                if preview:
                                    if input:
                                        for f_ in input:
                                            define_audio_with_size(basename=True, label="", value=f_, **base_c_params["output_audio"])
                    with gr.Column():
                        with gr.Group():
                            sep_model_name = gr.Dropdown(label=_i18n("model_name"), choices=all_models, value=default_model, **base_c_params["base"])
                            sep_selected_stems = gr.CheckboxGroup(label=_i18n("select_stems"), info=_i18n("select_stems_info"), choices=stems_default, value=[], **base_c_params["base"])
                            sep_extract_instrumental = gr.Checkbox(label=_i18n("extract_instrumental"), visible=ext_inst_visible_default, value=False, **base_c_params["base"])
                            sep_model_name.change(self.update_model_name, inputs=sep_model_name, outputs=[sep_extract_instrumental, sep_selected_stems])
                            sep_use_spec_invert = gr.Checkbox(label=_i18n("use_spec_invert"), value=False, **base_c_params["base"])
                            sep_sum_stems = gr.Checkbox(label=_i18n("invert_plus"), info=_i18n("invert_plus_info"), value=False, **base_c_params["base"])
                            with gr.Accordion(label=_i18n("separation_params"), open=False):
                                add_params_comp_seq = generate_add_params_component()
                                add_params_user_state = gr.State(default_add_params)

                                for comp in add_params_comp_seq:
                                    comp.change(
                                        fn=self.update_add_params,
                                        inputs=[*add_params_comp_seq], outputs=add_params_user_state,
                                        show_progress="hidden"
                                    )
                            sep_template = gr.Textbox(label=_i18n("output_template"), info=_i18n("output_template_info"), value="NAME_(STEM)_MODEL", **base_c_params["base"])
                            sep_output_format = gr.Dropdown(label=_i18n("output_format"), choices=output_formats, value=output_formats[0], filterable=False, **base_c_params["base"])
                            sep_prefer_float = gr.Checkbox(label=_i18n("prefer_float"), value=False, **base_c_params["base"])
                            separate_btn = gr.Button(_i18n("separate"), variant="primary", **base_c_params["base"])
                with gr.Group():
                    with gr.Row(equal_height=True):
                        with gr.Column(min_width=110):
                            gr.Markdown("<h4><center>"+_i18n("history")+"</center></h4>")
                        sep_history = gr.Dropdown(container=False, scale=13, multiselect=True, max_choices=1, **base_c_params["base"])
                        sep_history.focus(self.get_actual_history_list, inputs=[sep_history, sep_history_list_state], outputs=[sep_history, sep_history_list_state], show_progress="hidden")
                        @sep_history.input(inputs=sep_history, outputs=sep_state)
                        def separation_show_history_fn(key: list):
                            state = self.history.get_from_history(one_element_list_to_value(key))
                            return state
                    sep_off_players_output = gr.Checkbox(label=_i18n("off_audio_players_output"), info=_i18n("off_audio_players_output_info"), value=False, **base_c_params["base"])
                    @separate_btn.click(inputs=[sep_input_files, sep_model_name, sep_selected_stems, sep_extract_instrumental, sep_use_spec_invert, sep_template, sep_output_format, sep_sum_stems, add_params_user_state, sep_prefer_float], outputs=[sep_state, sep_upload_files], trigger_mode="once", concurrency_id="mvsepless_app_inference")
                    def separator_wrap(input_files: list, model_name: str, sel_stems: list, ext_inst: bool, spec_invert: bool, tmpl: str, output_format: str, sum_stems: bool, add_params: dict, pref_f: bool, progress=gr.Progress(track_tqdm=True)):
                        results = []
                        results = self.separate(
                            input_files=input_files, output_dir=self.output_dir.gen_output_dir(), 
                            output_format=output_format, template=tmpl, model_name=model_name, extract_instrumental=ext_inst, 
                            use_spec_invert=spec_invert, invert_plus=sum_stems, prefer_float=pref_f, selected_stems=sel_stems, add_params=add_params
                        )
                        self.history.add_to_history(model_name, results)
                        return results, gr.skip()
                    @gr.render(inputs=[sep_state, sep_off_players_output])
                    def show_players(state, off_players_output: bool):
                        if state:
                            zip_is_generated = False
                            all_stems_dict = {}
                            all_stems = set(stem_name_ for stem_list in (stems_list_ for basename_, stems_list_ in state) for stem_name_, stem_path_ in stem_list)
                            all_files = []
                            for stem in all_stems:
                                all_stems_dict[stem] = []
                            for basename, stems_list in state:
                                with gr.Group():
                                    gr.Markdown(f"<h4><center>{basename}</center></h4>")
                                    for stem_name, stem_path in stems_list:
                                        all_files.append(stem_path)
                                        all_stems_dict[stem_name].append(stem_path) 
                                        with gr.Row(equal_height=True):
                                            if off_players_output:
                                                output_audio = define_download_button_with_size(
                                                    value=stem_path,
                                                    label=stem_name,
                                                    **base_c_params["base"], variant="huggingface",
                                                    scale=15,
                                                )
                                            else:
                                                output_audio = define_audio_with_size(
                                                    value=stem_path,
                                                    label=stem_name,
                                                    **base_c_params["output_audio"],
                                                    scale=15,
                                                )
                                            reuse_btn = gr.Button(
                                                _i18n("reuse_btn"), 
                                                variant="secondary", **base_c_params["base"]
                                            )
                                            @reuse_btn.click(
                                                outputs=sep_input_files,
                                            )
                                            def reuse_fn(stem=deepcopy(stem_path)) -> gr.update:
                                                uploaded_files = self.input_files.upload([stem], copy=True)
                                                return gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)
                                            
                            with gr.Column(variant="panel"):
                                with gr.Column(variant="panel"):
                                    for stem_a in all_stems:
                                        if len(all_stems_dict[stem_a]) > 1:
                                            reuse_all_stem_btn = gr.Button(_i18n("reuse_all_stem", stem=stem_a), variant="huggingface", **base_c_params["base"])
                                            @reuse_all_stem_btn.click(outputs=sep_input_files)
                                            def reuse_all_stem_fn(stem=deepcopy(stem_a)):
                                                uploaded_files = self.input_files.upload(all_stems_dict[stem], copy=True)
                                                return gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)
                                    
                                reuse_all_stems_btn = gr.Button(_i18n("reuse_all_stems"), variant="primary", **base_c_params["base"])
                                @reuse_all_stems_btn.click(outputs=sep_input_files)
                                def reuse_all_stems_fn():
                                    uploaded_files = self.input_files.upload(all_files, copy=True)
                                    return gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)
                                
                                generate_zip_btn = gr.DownloadButton(label=_i18n("generate_zip_archive"), variant="huggingface", **base_c_params["base"])
                                @generate_zip_btn.click(outputs=generate_zip_btn, trigger_mode="once")
                                def generate_zip_fn():
                                    nonlocal zip_is_generated
                                    if zip_is_generated:
                                        return gr.skip()
                                    else:
                                        zip_file = generate_zip_archive(all_files, get_zip_output_path("mvsepless"))
                                        zip_is_generated = True
                                        return gr.DownloadButton(label=_i18n("download_zip_archive"), variant="huggingface", value=zip_file, **base_c_params["base"])

                        else:
                            gr.Markdown("<h3><center>"+_i18n("not_separated")+"</center></h3>", container=True)
            with gr.Tab(_i18n("custom_separation_tab")):
                custom_sep_state = gr.State()
                with gr.Row():
                    with gr.Column():
                        custom_sep_upload_files = gr.File(show_label=False, **base_c_params["input_files_multi"])
                        with gr.Group():
                            custom_sep_input_files = gr.Dropdown(container=False, allow_custom_value=True, **base_c_params["dropdown_multi"])
                            custom_sep_input_files.focus(
                                self.get_actual_input_list, 
                                inputs=[custom_sep_input_files, custom_sep_input_state], 
                                outputs=[custom_sep_input_files, custom_sep_input_state], 
                                show_progress="hidden"
                            )
                            custom_sep_add_uploaded_files = gr.Checkbox(label=_i18n("add_uploaded_files_to_current_list"), value=False, **base_c_params["base"])
                            custom_sep_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                            
                            @custom_sep_upload_files.upload(inputs=[custom_sep_upload_files, custom_sep_input_files, custom_sep_add_uploaded_files], outputs=[custom_sep_upload_files, custom_sep_input_files])
                            def upload_files_fn(files: list, input_files: list, add_uploaded_files: bool):
                                uploaded_files = self.input_files.upload(files)
                                if add_uploaded_files:
                                    return gr.update(value=None), gr.update(choices=self.input_files.get_input_list(), value=[*uploaded_files, *input_files])
                                return gr.update(value=None), gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)
                            
                            @gr.render(inputs=[custom_sep_input_files, custom_sep_input_preview_check])
                            def preview_inputs(input: list, preview: bool):
                                if preview:
                                    if input:
                                        for f_ in input:
                                            define_audio_with_size(basename=True, label="", value=f_, **base_c_params["output_audio"])
                    
                    with gr.Column():
                        with gr.Group():
                            custom_sep_model_type = gr.Dropdown(
                                label=_i18n("model_type"),
                                choices=custom_model_types,
                                value=custom_model_types[0],
                                **base_c_params["base"]
                            )
                            
                            # Новые компоненты для выбора модели как в vbach_tab
                            custom_sep_checkpoint = gr.Dropdown(
                                label=_i18n("checkpoint_path"), 
                                multiselect=True, allow_custom_value=True,
                                max_choices=1,
                                **base_c_params["base"]
                            )
                            custom_sep_checkpoint.focus(
                                self.get_actual_custom_sep_checkpoints_list,
                                inputs=[custom_sep_checkpoint, custom_sep_checkpoints_state],
                                outputs=[custom_sep_checkpoint, custom_sep_checkpoints_state],
                                show_progress="hidden"
                            )
                            
                            custom_sep_config = gr.Dropdown(
                                label=_i18n("config_path"), 
                                multiselect=True,  allow_custom_value=True,
                                max_choices=1,
                                **base_c_params["base"]
                            )
                            custom_sep_config.focus(
                                self.get_actual_custom_sep_configs_list,
                                inputs=[custom_sep_config, custom_sep_configs_state],
                                outputs=[custom_sep_config, custom_sep_configs_state],
                                show_progress="hidden"
                            )
                            
                            custom_sep_selected_stems = gr.CheckboxGroup(label=_i18n("select_stems"), info=_i18n("select_stems_info"), choices=[], value=[], **base_c_params["base"])
                            custom_sep_extract_instrumental = gr.Checkbox(label=_i18n("extract_instrumental"), visible=ext_inst_visible_default, value=False, **base_c_params["base"])
                            @custom_sep_config.input(inputs=[custom_sep_config, custom_sep_model_type], outputs=[custom_sep_extract_instrumental, custom_sep_selected_stems])
                            def get_stems_from_config_fn(path: str, model_type: str):
                                stems = get_stems_from_config_simple(one_element_list_to_value(path), model_type)
                                return gr.update(value=False, visible=len(stems) > 2), gr.update(value=[], choices=stems)

                            custom_sep_use_spec_invert = gr.Checkbox(
                                label=_i18n("use_spec_invert"), 
                                value=False, 
                                **base_c_params["base"]
                            )
                            custom_sep_sum_stems = gr.Checkbox(label=_i18n("invert_plus"), info=_i18n("invert_plus_info"), value=False, **base_c_params["base"])

                            with gr.Accordion(label=_i18n("separation_params"), open=False):
                                custom_add_params_comp_seq = generate_add_params_component()
                                custom_add_params_user_state = gr.State(default_add_params)
                                
                                for comp in custom_add_params_comp_seq:
                                    comp.input(
                                        fn=self.update_add_params, 
                                        inputs=[*custom_add_params_comp_seq], outputs=custom_add_params_user_state,
                                        show_progress="hidden"
                                    )
                            
                            custom_sep_template = gr.Textbox(
                                label=_i18n("output_template"), 
                                info=_i18n("output_template_info"), 
                                value="NAME_(STEM)_MODEL", 
                                **base_c_params["base"]
                            )
                            custom_sep_output_format = gr.Dropdown(
                                label=_i18n("output_format"), 
                                choices=output_formats, 
                                value=output_formats[0], 
                                filterable=False, 
                                **base_c_params["base"]
                            )
                            custom_sep_prefer_float = gr.Checkbox(label=_i18n("prefer_float"), value=False, **base_c_params["base"])
                            custom_separate_btn = gr.Button(_i18n("separate"), variant="primary", **base_c_params["base"])
                
                with gr.Group():
                    with gr.Row(equal_height=True):
                        with gr.Column(min_width=110):
                            gr.Markdown("<h4><center>"+_i18n("history")+"</center></h4>")
                        custom_sep_history = gr.Dropdown(container=False, scale=13, multiselect=True, max_choices=1, **base_c_params["base"])
                        custom_sep_history.focus(
                            self.get_actual_history_list,
                            inputs=[custom_sep_history, custom_sep_history_state], 
                            outputs=[custom_sep_history, custom_sep_history_state], 
                            show_progress="hidden"
                        )
                        
                        @custom_sep_history.input(inputs=custom_sep_history, outputs=custom_sep_state)
                        def custom_separation_show_history_fn(key: list):
                            state = self.history.get_from_history(one_element_list_to_value(key))
                            return state
                
                    custom_sep_off_players_output = gr.Checkbox(label=_i18n("off_audio_players_output"), info=_i18n("off_audio_players_output_info"), value=False, **base_c_params["base"])
                    @custom_separate_btn.click(
                        inputs=[custom_sep_input_files, custom_sep_model_type, custom_sep_checkpoint, custom_sep_config,
                                custom_sep_selected_stems, custom_sep_extract_instrumental, custom_sep_use_spec_invert,
                                custom_sep_template, custom_sep_output_format, custom_sep_sum_stems, custom_add_params_user_state, custom_sep_prefer_float],
                        outputs=[custom_sep_state, custom_sep_upload_files],
                        trigger_mode="once", concurrency_id="mvsepless_app_inference"
                    )
                    def custom_separator_wrap(
                        input_files: list, model_type: str, checkpoint: list, config: list,
                        sel_stems: list, ext_inst: bool, spec_invert: bool, 
                        tmpl: str, output_format: str, sum_stems: bool, add_params: dict, pref_f: bool, progress=gr.Progress(track_tqdm=True)
                    ):
                        checkpoint_path = one_element_list_to_value(checkpoint)
                        config_path = one_element_list_to_value(config)
                        
                        if not checkpoint_path or not config_path:
                            gr.Warning(_i18n("paths_not_specified"))
                            return [], gr.skip()
                        
                        results = self.custom_separate(
                            input_files=input_files, 
                            output_dir=self.output_dir.gen_output_dir(), 
                            output_format=output_format, 
                            template=tmpl,
                            model_type=model_type,
                            ckpt=checkpoint_path,
                            conf=config_path,
                            extract_instrumental=ext_inst,
                            use_spec_invert=spec_invert,
                            invert_plus=sum_stems,
                            prefer_float=pref_f,
                            selected_stems=sel_stems,
                            add_params=add_params
                        )
                        
                        model_name = Path(checkpoint_path).stem
                        self.history.add_to_history(model_name, results)
                        return results, gr.skip()
                    
                    @gr.render(inputs=[custom_sep_state, custom_sep_off_players_output])
                    def show_custom_players(state, off_players_output: bool):
                        if state:
                            zip_is_generated = False
                            all_stems_dict = {}
                            all_stems = set(stem_name_ for stem_list in (stems_list_ for basename_, stems_list_ in state) for stem_name_, stem_path_ in stem_list)
                            all_files = []
                            for stem in all_stems:
                                all_stems_dict[stem] = []
                            for basename, stems_list in state:
                                with gr.Group():
                                    gr.Markdown(f"<h4><center>{basename}</center></h4>")
                                    for stem_name, stem_path in stems_list:
                                        all_files.append(stem_path)
                                        all_stems_dict[stem_name].append(stem_path) 
                                        with gr.Row(equal_height=True):
                                            if off_players_output:
                                                output_audio = define_download_button_with_size(
                                                    value=stem_path,
                                                    label=stem_name,
                                                    **base_c_params["base"], variant="huggingface",
                                                    scale=15,
                                                )
                                            else:
                                                output_audio = define_audio_with_size(
                                                    value=stem_path,
                                                    label=stem_name,
                                                    **base_c_params["output_audio"],
                                                    scale=15,
                                                )
                                            reuse_btn = gr.Button(
                                                _i18n("reuse_btn"), 
                                                variant="secondary", **base_c_params["base"]
                                            )
                                            @reuse_btn.click(
                                                outputs=custom_sep_input_files,
                                            )
                                            def reuse_fn(stem=deepcopy(stem_path)) -> gr.update:
                                                uploaded_files = self.input_files.upload([stem], copy=True)
                                                return gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)
                                            
                            with gr.Column(variant="panel"):
                                with gr.Column(variant="panel"):
                                    for stem_a in all_stems:
                                        if len(all_stems_dict[stem_a]) > 1:
                                            reuse_all_stem_btn = gr.Button(_i18n("reuse_all_stem", stem=stem_a), variant="huggingface", **base_c_params["base"])
                                            @reuse_all_stem_btn.click(outputs=custom_sep_input_files)
                                            def reuse_all_stem_fn(stem=deepcopy(stem_a)):
                                                uploaded_files = self.input_files.upload(all_stems_dict[stem], copy=True)
                                                return gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)
                                    
                                reuse_all_stems_btn = gr.Button(_i18n("reuse_all_stems"), variant="primary", **base_c_params["base"])
                                @reuse_all_stems_btn.click(outputs=custom_sep_input_files)
                                def reuse_all_stems_fn():
                                    uploaded_files = self.input_files.upload(all_files, copy=True)
                                    return gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)
                                
                                generate_zip_btn = gr.DownloadButton(label=_i18n("generate_zip_archive"), variant="huggingface", **base_c_params["base"])
                                @generate_zip_btn.click(outputs=generate_zip_btn, trigger_mode="once")
                                def generate_zip_fn():
                                    nonlocal zip_is_generated
                                    if zip_is_generated:
                                        return gr.skip()
                                    else:
                                        zip_file = generate_zip_archive(all_files, get_zip_output_path("mvsepless"))
                                        zip_is_generated = True
                                        return gr.DownloadButton(label=_i18n("download_zip_archive"), variant="huggingface", value=zip_file, **base_c_params["base"])
                        else:
                            gr.Markdown("<h3><center>"+_i18n("not_separated")+"</center></h3>", container=True)

            with gr.Tab(_i18n("ensemble_tab")):
                with gr.Tab(_i18n("auto_ensemble_tab")):
                    auto_ensemble_user_flow_state = gr.BrowserState([])
                    with gr.Row():
                        with gr.Column():
                            auto_ensemble_upload_file = gr.File(show_label=False, **base_c_params["input_file"])
                            with gr.Group():
                                auto_ensemble_input_file = gr.Dropdown(container=False, allow_custom_value=True, multiselect=True, max_choices=1, **base_c_params["base"])
                                auto_ensemble_input_file.focus(self.get_actual_input_list, inputs=[auto_ensemble_input_file, auto_ensemble_input_state], outputs=[auto_ensemble_input_file, auto_ensemble_input_state], show_progress="hidden")
                                auto_ensemble_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                                @auto_ensemble_upload_file.upload(inputs=auto_ensemble_upload_file, outputs=[auto_ensemble_upload_file, auto_ensemble_input_file])
                                def upload_file_fn(file: str):
                                    uploaded_files = self.input_files.upload([file])
                                    all_uploaded_files = self.input_files.get_input_list()
                                    if uploaded_files:
                                        first_value = [uploaded_files[0]]
                                    else:
                                        first_value = []
                                    return gr.update(value=None), gr.update(choices=all_uploaded_files, value=first_value)
                                @gr.render(inputs=[auto_ensemble_input_file, auto_ensemble_input_preview_check])
                                def preview_input(input: list, preview: bool):
                                    if preview:
                                        if input:
                                            define_audio_with_size(basename=True, label="", value=one_element_list_to_value(input), **base_c_params["output_audio"])
                        with gr.Column():
                            with gr.Group():
                                auto_ensemble_save_primary_stems = gr.Checkbox(label=_i18n("enable_save_primary_stems"), value=False, **base_c_params["base"])
                                auto_ensemble_use_spec_invert = gr.Checkbox(label=_i18n("use_spec_invert"), value=False, **base_c_params["base"])
                                auto_ensemble_type = gr.Dropdown(label=_i18n("ensemble_type"), info=_i18n("ensemble_type_info"), choices=ensemble_types, value=ensemble_types[0], filterable=False, **base_c_params["base"])
                                auto_ensemble_template = gr.Textbox(label=_i18n("output_template"), info=_i18n("output_etemplate_info"), value="NAME_(COUNT)_TYPE", **base_c_params["base"])
                                auto_ensemble_format = gr.Dropdown(label=_i18n("output_format"), choices=output_formats, value=output_formats[0], filterable=False, **base_c_params["base"])
                                auto_ensemble_prefer_float = gr.Checkbox(label=_i18n("prefer_float"), value=False, **base_c_params["base"])
                                auto_ensemble_run_btn = gr.Button(_i18n("run_ensemble"), variant="primary", **base_c_params["base"])
                    with gr.Column():
                        auto_ensemble_show_timer = gr.Timer()
                        with gr.Accordion(label=_i18n("ensemble_settings"), open=False):
                            with gr.Row():
                                with gr.Column(scale=3):
                                    with gr.Group():
                                        auto_ensemble_model_name = gr.Dropdown(label=_i18n("model_name"), choices=all_models, value=default_model, **base_c_params["base"])
                                        auto_ensemble_primary_stem = gr.Dropdown(label=_i18n("primary_stem"), choices=stems_default, filterable=False, value=stems_default[0], **base_c_params["base"])
                                        auto_ensemble_model_name.change(self.update_model_name_ensemble, inputs=auto_ensemble_model_name, outputs=[auto_ensemble_primary_stem])
                                        auto_ensemble_invert = gr.Dropdown(label=_i18n("invert"), choices=[True, False], value=False, filterable=False, **base_c_params["base"])
                                        auto_ensemble_weight = gr.Number(label=_i18n("weights"), minimum=0, value=1, **base_c_params["base"])
                                        auto_ensemble_add_btn = gr.Button(_i18n("add_model"), variant="primary", **base_c_params["base"])
                                        auto_ensemble_add_btn.click(self.auto_ensemble_app.add_model, inputs=[auto_ensemble_model_name, auto_ensemble_primary_stem, auto_ensemble_invert, auto_ensemble_weight, auto_ensemble_user_flow_state], outputs=auto_ensemble_user_flow_state)
                                with gr.Column(scale=11):
                                    with gr.Group():
                                        auto_ensemble_model_index = gr.Number(label=_i18n("model_index"), value=1, **base_c_params["base"])
                                        with gr.Row(equal_height=True):
                                            auto_ensemble_replace_btn = gr.Button(_i18n("replace"), variant="huggingface", **base_c_params["base"])
                                            auto_ensemble_insert_btn = gr.Button(_i18n("insert"), variant="secondary", **base_c_params["base"])
                                        with gr.Row(equal_height=True):
                                            auto_ensemble_delete_btn = gr.Button(_i18n("delete"), variant="stop", **base_c_params["base"])
                                            auto_ensemble_clear_all_btn = gr.Button(_i18n("clear"), variant="stop", **base_c_params["base"])
                                        auto_ensemble_replace_btn.click(self.auto_ensemble_app.replace_model, inputs=[auto_ensemble_model_index, auto_ensemble_model_name, auto_ensemble_primary_stem, auto_ensemble_invert, auto_ensemble_weight, auto_ensemble_user_flow_state], outputs=auto_ensemble_user_flow_state)
                                        auto_ensemble_insert_btn.click(self.auto_ensemble_app.insert_model, inputs=[auto_ensemble_model_index, auto_ensemble_model_name, auto_ensemble_primary_stem, auto_ensemble_invert, auto_ensemble_weight, auto_ensemble_user_flow_state], outputs=auto_ensemble_user_flow_state)
                                        auto_ensemble_delete_btn.click(self.auto_ensemble_app.delete_model, inputs=[auto_ensemble_model_index, auto_ensemble_user_flow_state], outputs=auto_ensemble_user_flow_state)
                                        auto_ensemble_clear_all_btn.click(self.auto_ensemble_app.clear_all_model, inputs=auto_ensemble_user_flow_state, outputs=auto_ensemble_user_flow_state)
                                        auto_ensemble_show_flow = gr.DataFrame(value=self.auto_ensemble_app.get_models_df([]), type="array", headers=["#", _i18n("model_name"), _i18n("primary_stem"), _i18n("invert"), _i18n("weights")], interactive=False, datatype=["number", "str", "str", "bool", "number"])
                                        auto_ensemble_show_timer.tick(self.auto_ensemble_app.get_models_df, inputs=auto_ensemble_user_flow_state, outputs=[auto_ensemble_show_flow])

                        with gr.Accordion(label=_i18n("ensemble_preset_settings"), open=False):
                            auto_ensemble_list_flows_state = gr.State([])
                            auto_ensemble_list_flows = gr.Dropdown(label=_i18n("auto_ensemble_name_preset"), allow_custom_value=True, multiselect=True, max_choices=1, **base_c_params["base"])
                            auto_ensemble_list_flows.focus(self.get_actual_auto_ensemble_flows_list, inputs=[auto_ensemble_list_flows, auto_ensemble_list_flows_state], outputs=[auto_ensemble_list_flows, auto_ensemble_list_flows_state], show_progress="hidden")
                            with gr.Group():
                                with gr.Row(equal_height=True):
                                    auto_ensemble_flow_load_btn = gr.Button(_i18n("load"), variant="primary", min_width=30, **base_c_params["base"])
                                    auto_ensemble_flow_save_btn = gr.Button(_i18n("save"), variant="secondary", min_width=30, **base_c_params["base"])
                                    @auto_ensemble_flow_save_btn.click(inputs=[auto_ensemble_list_flows, auto_ensemble_user_flow_state], outputs=[auto_ensemble_list_flows])
                                    def auto_ensemble_save_flow_fn(name: list, state: list):
                                        self.auto_ensemble_app.write_flow(one_element_list_to_value(name), state)
                                        return gr.update(choices=self.auto_ensemble_app.get_flows(), value=name)
                                    @auto_ensemble_flow_load_btn.click(inputs=[auto_ensemble_list_flows], outputs=auto_ensemble_user_flow_state)
                                    def auto_ensemble_load_flow_fn(name: list):
                                        return self.auto_ensemble_app.load_flow(one_element_list_to_value(name))
                                with gr.Row(equal_height=True):
                                    auto_ensemble_flow_import_btn = gr.UploadButton(label=_i18n("import"), variant="secondary", min_width=30, file_count="multiple", type="filepath", file_types=[".json"], **base_c_params["base"])
                                    auto_ensemble_flow_export_btn = gr.DownloadButton(label=_i18n("export"), variant="huggingface", min_width=30, **base_c_params["base"])
                                    auto_ensemble_flow_import_btn.upload(self.auto_ensemble_app.import_flows, inputs=auto_ensemble_flow_import_btn, outputs=auto_ensemble_flow_import_btn)
                                    @gr.on(inputs=auto_ensemble_list_flows, outputs=auto_ensemble_flow_export_btn, triggers=[auto_ensemble_list_flows.change, auto_ensemble_list_flows.focus, auto_ensemble_list_flows.blur, auto_ensemble_flow_save_btn.click, auto_ensemble_flow_load_btn.click])
                                    def export_flow_fn(input_key: list):
                                        return self.auto_ensemble_app.export_flow(one_element_list_to_value(input_key))
                                with gr.Row(equal_height=True):
                                    auto_ensemble_flow_clear_btn = gr.Button(_i18n("clear"), variant="stop", visible=not hf_space_mode, **base_c_params["base"])
                                    @auto_ensemble_flow_clear_btn.click()
                                    def clear_all_flows_fn():
                                        self.auto_ensemble_app.clear_flows()
                        with gr.Group():
                            with gr.Row(equal_height=True):
                                with gr.Column(min_width=110):
                                    gr.Markdown("<h4><center>"+_i18n("history")+"</center></h4>")
                                auto_ensemble_history = gr.Dropdown(container=False, scale=13, multiselect=True, max_choices=1, **base_c_params["base"])
                        with gr.Row():
                            with gr.Column():
                                auto_ensemble_output_audio = gr.Audio(label=_i18n("ensemble_result"), value=None, **base_c_params["output_audio"])
                                auto_ensemble_ioutput_audio = gr.Audio(label=_i18n("inverted_result"), value=None, **base_c_params["output_audio"])
                                with gr.Row(equal_height=True):
                                    auto_ensemble_output_audio_reuse_btn = gr.Button(
                                        _i18n("reuse_output_btn"), 
                                        variant="secondary", visible=False, **base_c_params["base"]
                                    )
                                    auto_ensemble_ioutput_reuse_btn = gr.Button(
                                        _i18n("reuse_invert_btn"), 
                                        variant="huggingface", visible=False, **base_c_params["base"]
                                    )
                                @auto_ensemble_output_audio_reuse_btn.click(
                                    inputs=[auto_ensemble_output_audio],
                                    outputs=auto_ensemble_input_file,
                                )
                                def reuse_fn(stem_audio: str) -> gr.update:
                                    uploaded_files = self.input_files.upload([stem_audio], copy=True)
                                    all_uploaded_files = self.input_files.get_input_list()
                                    if all_uploaded_files:
                                        first_value = [all_uploaded_files[0]]
                                    else:
                                        first_value = []
                                    return gr.update(choices=all_uploaded_files, value=first_value)
                                @auto_ensemble_ioutput_reuse_btn.click(
                                    inputs=[auto_ensemble_ioutput_audio],
                                    outputs=auto_ensemble_input_file,
                                )
                                def reuse_fn(stem_audio: str) -> gr.update:
                                    uploaded_files = self.input_files.upload([stem_audio], copy=True)
                                    all_uploaded_files = self.input_files.get_input_list()
                                    if uploaded_files:
                                        first_value = [uploaded_files[0]]
                                    else:
                                        first_value = []
                                    return gr.update(choices=all_uploaded_files, value=first_value)
                                
                                auto_ensemble_zip_is_generated = gr.State(False)
                                auto_ensemble_generate_zip_btn = gr.DownloadButton(label=_i18n("generate_zip_archive"), variant="huggingface", visible=False, **base_c_params["base"])
    
                            with gr.Column():
                                with gr.Group():
                                    gr.Markdown("<h3><center>"+_i18n("saved_primary_stems")+"</center></h3>", container=True)
                                    auto_ensebmle_primary_stems_off_players_output = gr.Checkbox(label=_i18n("off_audio_players_output"), info=_i18n("off_audio_players_output_info"), value=False, **base_c_params["base"])
                                    auto_ensemble_primary_stems_state = gr.State([])
                                    @gr.render(inputs=[auto_ensemble_primary_stems_state, auto_ensebmle_primary_stems_off_players_output])
                                    def preview_pr_stems(input: list, off_players_output: bool):
                                        if input:
                                            for f_ in input:
                                                if off_players_output:
                                                    eoutput_audio = define_download_button_with_size(basename=True, label="", value=f_, scale=15, **base_c_params["base"], variant="huggingface")
                                                else:
                                                    eoutput_audio = define_audio_with_size(basename=True, label="", value=f_, scale=15, **base_c_params["output_audio"])
                                                ereuse_btn = gr.Button(
                                                    _i18n("reuse_btn"), 
                                                    variant="secondary", **base_c_params["base"]
                                                )
                                                @ereuse_btn.click(
                                                    inputs=[eoutput_audio],
                                                    outputs=auto_ensemble_input_file,
                                                )
                                                def reuse_fn(stem_audio: str) -> gr.update:
                                                    uploaded_files = self.input_files.upload([stem_audio], copy=True)
                                                    all_uploaded_files = self.input_files.get_input_list()
                                                    if all_uploaded_files:
                                                        first_value = [all_uploaded_files[0]]
                                                    else:
                                                        first_value = []
                                                    return gr.update(choices=all_uploaded_files, value=first_value)
                                        else:
                                            gr.Markdown("<h3><center>"+_i18n("not_ensembled_with_primary_stems")+"</center></h3>", container=True)

                    @auto_ensemble_run_btn.click(inputs=[auto_ensemble_input_file, auto_ensemble_template, auto_ensemble_type, auto_ensemble_use_spec_invert, auto_ensemble_format, auto_ensemble_save_primary_stems, auto_ensemble_user_flow_state, auto_ensemble_prefer_float], outputs=[auto_ensemble_output_audio, auto_ensemble_ioutput_audio, auto_ensemble_primary_stems_state, auto_ensemble_upload_file, auto_ensemble_output_audio_reuse_btn, auto_ensemble_ioutput_reuse_btn, auto_ensemble_generate_zip_btn, auto_ensemble_zip_is_generated], concurrency_id="mvsepless_app_inference_ensemble")
                    def auto_ensemble_wrapper_fn(input_file: list, template: str, etype: str, spec_invert: bool, out_format: str, save_pr_stems: bool, flow: list[list], pref_f: bool, progress=gr.Progress(track_tqdm=True)):
                        out, iout, pr_stems = self.auto_ensemble(input_file=one_element_list_to_value(input_file), output_dir=self.output_dir.gen_output_dir(), flow=flow, template=template, etype=etype, output_format=out_format, use_spec_invert=spec_invert, save_primary_stems=save_pr_stems, prefer_float=pref_f)
                        self.auto_ensemble_history_app.add_to_history(etype, out, iout, pr_stems)
                        return update_audio_with_size(label=_i18n("ensemble_result"), value=out), update_audio_with_size(label=_i18n("inverted_result"), value=iout), pr_stems, gr.skip(), gr.update(visible=True), gr.update(visible=True), gr.DownloadButton(label=_i18n("generate_zip_archive"), variant="huggingface", visible=True, **base_c_params["base"]), gr.update(value=False)

                    auto_ensemble_history.focus(self.get_actual_auto_ensemble_history_list, inputs=[auto_ensemble_history, auto_ensemble_history_state], outputs=[auto_ensemble_history, auto_ensemble_history_state], show_progress="hidden")
                    @auto_ensemble_history.input(inputs=auto_ensemble_history, outputs=[auto_ensemble_output_audio, auto_ensemble_ioutput_audio, auto_ensemble_primary_stems_state, auto_ensemble_output_audio_reuse_btn, auto_ensemble_ioutput_reuse_btn, auto_ensemble_generate_zip_btn, auto_ensemble_zip_is_generated])
                    def auto_ensemble_show_history_fn(key: list):
                        out, iout, pr_stems = self.auto_ensemble_history_app.get_from_history(one_element_list_to_value(key))
                        visible = all([out, iout])
                        return update_audio_with_size(label=_i18n("ensemble_result"), value=out), update_audio_with_size(label=_i18n("inverted_result"), value=iout), pr_stems, gr.update(visible=visible), gr.update(visible=visible), gr.DownloadButton(label=_i18n("generate_zip_archive"), variant="huggingface", visible=visible, **base_c_params["base"]), gr.update(value=False)

                    @auto_ensemble_generate_zip_btn.click(inputs=[auto_ensemble_output_audio, auto_ensemble_ioutput_audio, auto_ensemble_primary_stems_state, auto_ensemble_zip_is_generated], outputs=[auto_ensemble_generate_zip_btn, auto_ensemble_zip_is_generated], trigger_mode="once")
                    def generate_zip_fn(out, iout, e_state, zip_is_generated):
                        all_files = []

                        if out:
                            all_files.append(out)

                        if iout:
                            all_files.append(iout)

                        if e_state:
                            all_files.extend(e_state)

                        if zip_is_generated:
                            return gr.skip(), gr.skip()
                        else:
                            zip_file = generate_zip_archive(all_files, get_zip_output_path("ensembless"))
                            zip_is_generated = True
                            return gr.DownloadButton(label=_i18n("download_zip_archive"), variant="huggingface", value=zip_file, **base_c_params["base"]), zip_is_generated

                with gr.Tab(_i18n("iterative_ensemble_tab")):
                    iterative_ensemble_user_flow_state = gr.BrowserState([])
                    with gr.Row():
                        with gr.Column():
                            iterative_ensemble_upload_file = gr.File(show_label=False, **base_c_params["input_file"])
                            with gr.Group():
                                iterative_ensemble_input_file = gr.Dropdown(container=False, allow_custom_value=True, multiselect=True, max_choices=1, **base_c_params["base"])
                                iterative_ensemble_input_file.focus(
                                    self.get_actual_input_list, 
                                    inputs=[iterative_ensemble_input_file, iterative_ensemble_input_state], 
                                    outputs=[iterative_ensemble_input_file, iterative_ensemble_input_state], 
                                    show_progress="hidden"
                                )
                                iterative_ensemble_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                                
                                @iterative_ensemble_upload_file.upload(inputs=iterative_ensemble_upload_file, outputs=[iterative_ensemble_upload_file, iterative_ensemble_input_file])
                                def upload_file_fn(file: str):
                                    uploaded_files = self.input_files.upload([file])
                                    all_uploaded_files = self.input_files.get_input_list()
                                    if uploaded_files:
                                        first_value = [uploaded_files[0]]
                                    else:
                                        first_value = []
                                    return gr.update(value=None), gr.update(choices=all_uploaded_files, value=first_value)
                                
                                @gr.render(inputs=[iterative_ensemble_input_file, iterative_ensemble_input_preview_check])
                                def preview_input(input: list, preview: bool):
                                    if preview:
                                        if input:
                                            define_audio_with_size(basename=True, label="", value=one_element_list_to_value(input), **base_c_params["output_audio"])
                        
                        with gr.Column():
                            with gr.Group():
                                iterative_ensemble_num_iters = gr.Number(
                                    label=_i18n("num_iters"), 
                                    minimum=1, 
                                    maximum=20, 
                                    value=4, 
                                    step=1, 
                                    **base_c_params["base"]
                                )
                                iterative_ensemble_save_intermediate = gr.Checkbox(
                                    label=_i18n("save_intermediate"), 
                                    value=False, 
                                    **base_c_params["base"]
                                )
                                iterative_ensemble_template = gr.Textbox(
                                    label=_i18n("output_template"), 
                                    info=_i18n("output_iterative_template_info"),
                                    value="NAME_ITER", 
                                    **base_c_params["base"]
                                )
                                iterative_ensemble_format = gr.Dropdown(
                                    label=_i18n("output_format"), 
                                    choices=output_formats, 
                                    value=output_formats[0], 
                                    filterable=False, 
                                    **base_c_params["base"]
                                )
                                iterative_ensemble_prefer_float = gr.Checkbox(label=_i18n("prefer_float"), value=False, **base_c_params["base"])
                                iterative_ensemble_run_btn = gr.Button(
                                    _i18n("run_iterative_ensemble"), 
                                    variant="primary", 
                                    **base_c_params["base"]
                                )
                    
                    with gr.Column():
                        iterative_ensemble_show_timer = gr.Timer()
                        with gr.Accordion(label=_i18n("ensemble_settings"), open=False):
                            with gr.Row():
                                with gr.Column(scale=3):
                                    with gr.Group():
                                        iterative_ensemble_model_name = gr.Dropdown(
                                            label=_i18n("model_name"), 
                                            choices=all_models, 
                                            value=default_model, 
                                            **base_c_params["base"]
                                        )
                                        iterative_ensemble_primary_stem = gr.Dropdown(
                                            label=_i18n("primary_stem"), 
                                            choices=stems_default, 
                                            filterable=False, 
                                            value=stems_default[0], 
                                            **base_c_params["base"]
                                        )
                                        iterative_ensemble_model_name.change(
                                            self.update_model_name_ensemble, 
                                            inputs=iterative_ensemble_model_name, 
                                            outputs=[iterative_ensemble_primary_stem]
                                        )
                                        iterative_ensemble_invert = gr.Dropdown(
                                            label=_i18n("invert"), 
                                            choices=[True, False], 
                                            value=False, 
                                            filterable=False, 
                                            **base_c_params["base"]
                                        )
                                        iterative_ensemble_add_btn = gr.Button(
                                            _i18n("add_model"), 
                                            variant="primary", 
                                            **base_c_params["base"]
                                        )
                                        iterative_ensemble_add_btn.click(
                                            self.iterative_ensemble_app.add_model, 
                                            inputs=[iterative_ensemble_model_name, iterative_ensemble_primary_stem, iterative_ensemble_invert, iterative_ensemble_user_flow_state], 
                                            outputs=iterative_ensemble_user_flow_state
                                        )
                                
                                with gr.Column(scale=11):
                                    with gr.Group():
                                        iterative_ensemble_model_index = gr.Number(
                                            label=_i18n("model_index"), 
                                            value=1, 
                                            **base_c_params["base"]
                                        )
                                        with gr.Row(equal_height=True):
                                            iterative_ensemble_replace_btn = gr.Button(
                                                _i18n("replace"), 
                                                variant="huggingface", 
                                                **base_c_params["base"]
                                            )
                                            iterative_ensemble_insert_btn = gr.Button(
                                                _i18n("insert"), 
                                                variant="secondary", 
                                                **base_c_params["base"]
                                            )
                                        with gr.Row(equal_height=True):
                                            iterative_ensemble_delete_btn = gr.Button(
                                                _i18n("delete"), 
                                                variant="stop", 
                                                **base_c_params["base"]
                                            )
                                            iterative_ensemble_clear_all_btn = gr.Button(
                                                _i18n("clear"), 
                                                variant="stop", 
                                                **base_c_params["base"]
                                            )
                                        
                                        iterative_ensemble_replace_btn.click(
                                            self.iterative_ensemble_app.replace_model, 
                                            inputs=[iterative_ensemble_model_index, iterative_ensemble_model_name, iterative_ensemble_primary_stem, iterative_ensemble_invert, iterative_ensemble_user_flow_state], 
                                            outputs=iterative_ensemble_user_flow_state
                                        )
                                        iterative_ensemble_insert_btn.click(
                                            self.iterative_ensemble_app.insert_model, 
                                            inputs=[iterative_ensemble_model_index, iterative_ensemble_model_name, iterative_ensemble_primary_stem, iterative_ensemble_invert, iterative_ensemble_user_flow_state], 
                                            outputs=iterative_ensemble_user_flow_state
                                        )
                                        iterative_ensemble_delete_btn.click(
                                            self.iterative_ensemble_app.delete_model, 
                                            inputs=[iterative_ensemble_model_index, iterative_ensemble_user_flow_state], 
                                            outputs=iterative_ensemble_user_flow_state
                                        )
                                        iterative_ensemble_clear_all_btn.click(
                                            self.iterative_ensemble_app.clear_all_model, 
                                            inputs=iterative_ensemble_user_flow_state, 
                                            outputs=iterative_ensemble_user_flow_state
                                        )
                                        
                                        iterative_ensemble_show_flow = gr.DataFrame(
                                            value=self.iterative_ensemble_app.get_models_df([]), 
                                            type="array", 
                                            headers=["#", _i18n("model_name"), _i18n("primary_stem"), _i18n("invert")], 
                                            interactive=False, 
                                            datatype=["number", "str", "str", "bool"]
                                        )
                                        iterative_ensemble_show_timer.tick(
                                            self.iterative_ensemble_app.get_models_df, 
                                            inputs=iterative_ensemble_user_flow_state, 
                                            outputs=[iterative_ensemble_show_flow]
                                        )

                        with gr.Accordion(label=_i18n("ensemble_preset_settings"), open=False):
                            iterative_ensemble_list_flows = gr.Dropdown(
                                label=_i18n("iterative_ensemble_name_preset"), 
                                allow_custom_value=True, 
                                multiselect=True, 
                                max_choices=1, 
                                **base_c_params["base"]
                            )
                            iterative_ensemble_list_flows.focus(
                                self.get_actual_iterative_ensemble_flows_list, 
                                inputs=[iterative_ensemble_list_flows, iterative_ensemble_flows_state], 
                                outputs=[iterative_ensemble_list_flows, iterative_ensemble_flows_state], 
                                show_progress="hidden"
                            )
                            with gr.Group():
                                with gr.Row(equal_height=True):
                                    iterative_ensemble_flow_load_btn = gr.Button(
                                        _i18n("load"), 
                                        variant="primary", 
                                        min_width=30, 
                                        **base_c_params["base"]
                                    )
                                    iterative_ensemble_flow_save_btn = gr.Button(
                                        _i18n("save"), 
                                        variant="secondary", 
                                        min_width=30, 
                                        **base_c_params["base"]
                                    )
                                    @iterative_ensemble_flow_save_btn.click(
                                        inputs=[iterative_ensemble_list_flows, iterative_ensemble_user_flow_state], 
                                        outputs=[iterative_ensemble_list_flows]
                                    )
                                    def iterative_ensemble_save_flow_fn(name: list, state: list):
                                        self.iterative_ensemble_app.write_flow(one_element_list_to_value(name), state)
                                        return gr.update(choices=self.iterative_ensemble_app.get_flows(), value=name)
                                    
                                    @iterative_ensemble_flow_load_btn.click(
                                        inputs=[iterative_ensemble_list_flows], 
                                        outputs=iterative_ensemble_user_flow_state
                                    )
                                    def iterative_ensemble_load_flow_fn(name: list):
                                        return self.iterative_ensemble_app.load_flow(one_element_list_to_value(name))
                                
                                with gr.Row(equal_height=True):
                                    iterative_ensemble_flow_import_btn = gr.UploadButton(
                                        label=_i18n("import"), 
                                        variant="secondary", 
                                        min_width=30, 
                                        file_count="multiple", 
                                        type="filepath", 
                                        file_types=[".json"], 
                                        **base_c_params["base"]
                                    )
                                    iterative_ensemble_flow_export_btn = gr.DownloadButton(
                                        label=_i18n("export"), 
                                        variant="huggingface", 
                                        min_width=30, 
                                        **base_c_params["base"]
                                    )
                                    iterative_ensemble_flow_import_btn.upload(
                                        self.iterative_ensemble_app.import_flows, 
                                        inputs=iterative_ensemble_flow_import_btn, 
                                        outputs=iterative_ensemble_flow_import_btn
                                    )
                                    @gr.on(
                                        inputs=iterative_ensemble_list_flows, 
                                        outputs=iterative_ensemble_flow_export_btn,
                                        triggers=[iterative_ensemble_list_flows.change, iterative_ensemble_list_flows.focus, iterative_ensemble_list_flows.blur, iterative_ensemble_flow_load_btn.click, iterative_ensemble_flow_save_btn.click]
                                    )
                                    def export_flow_fn(input_key: list):
                                        return self.iterative_ensemble_app.export_flow(one_element_list_to_value(input_key))
                                
                                with gr.Row(equal_height=True):
                                    iterative_ensemble_flow_clear_btn = gr.Button(
                                        _i18n("clear"), 
                                        variant="stop", 
                                        visible=not hf_space_mode, 
                                        **base_c_params["base"]
                                    )
                                    @iterative_ensemble_flow_clear_btn.click()
                                    def clear_all_flows_fn():
                                        self.iterative_ensemble_app.clear_flows()
                    
                    with gr.Group():
                        with gr.Row(equal_height=True):
                            with gr.Column(min_width=110):
                                gr.Markdown("<h4><center>"+_i18n("history")+"</center></h4>")
                            iterative_ensemble_history = gr.Dropdown(
                                container=False, 
                                scale=13, 
                                multiselect=True, 
                                max_choices=1, 
                                **base_c_params["base"]
                            )
                            iterative_ensemble_history.focus(
                                self.get_actual_iterative_ensemble_history_list,
                                inputs=[iterative_ensemble_history, iterative_ensemble_history_state],
                                outputs=[iterative_ensemble_history, iterative_ensemble_history_state],
                                show_progress="hidden"
                            )
                    
                    with gr.Row():
                        with gr.Column():
                            iterative_ensemble_output_audio = gr.Audio(
                                label=_i18n("ensemble_result"), 
                                value=None, 
                                **base_c_params["output_audio"]
                            )
                            iterative_ensemble_intermediate_state = gr.State([])
                            with gr.Row(equal_height=True):
                                iterative_ensemble_output_reuse_btn = gr.Button(
                                    _i18n("reuse_output_btn"), 
                                    variant="secondary", 
                                    visible=False, 
                                    **base_c_params["base"]
                                )
                            @iterative_ensemble_output_reuse_btn.click(
                                inputs=[iterative_ensemble_output_audio],
                                outputs=iterative_ensemble_input_file,
                            )
                            def reuse_fn(stem_audio: str) -> gr.update:
                                uploaded_files = self.input_files.upload([stem_audio], copy=True)
                                all_uploaded_files = self.input_files.get_input_list()
                                if all_uploaded_files:
                                    first_value = [all_uploaded_files[0]]
                                else:
                                    first_value = []
                                return gr.update(choices=all_uploaded_files, value=first_value)
                            
                            iterative_ensemble_zip_is_generated = gr.State(False)
                            iterative_ensemble_generate_zip_btn = gr.DownloadButton(
                                label=_i18n("generate_zip_archive"), 
                                variant="huggingface", 
                                visible=False, 
                                **base_c_params["base"]
                            )

                        with gr.Column():
                            with gr.Group():
                                gr.Markdown("<h3><center>"+_i18n("intermediate_results")+"</center></h3>", container=True)
                                iterative_ensemble_intermediate_off_players_output = gr.Checkbox(label=_i18n("off_audio_players_output"), info=_i18n("off_audio_players_output_info"), value=False, **base_c_params["base"])
                                @gr.render(inputs=[iterative_ensemble_intermediate_state, iterative_ensemble_intermediate_off_players_output])
                                def preview_intermediate_results(input: list, off_players_output: bool):
                                    if input:
                                        for f_ in input:
                                            if off_players_output:
                                                define_download_button_with_size(basename=True, label="", value=f_, scale=15, **base_c_params["base"], variant="huggingface")
                                            else:
                                                define_audio_with_size(basename=True, label="", value=f_, scale=15, **base_c_params["output_audio"])
                                    else:
                                        gr.Markdown("<h3><center>"+_i18n("no_intermediate_results")+"</center></h3>", container=True)
                    
                    @iterative_ensemble_run_btn.click(
                        inputs=[iterative_ensemble_input_file, iterative_ensemble_num_iters, iterative_ensemble_save_intermediate, 
                                iterative_ensemble_template, iterative_ensemble_format, iterative_ensemble_user_flow_state, iterative_ensemble_prefer_float], 
                        outputs=[iterative_ensemble_output_audio, iterative_ensemble_intermediate_state, iterative_ensemble_upload_file, 
                                iterative_ensemble_output_reuse_btn, iterative_ensemble_generate_zip_btn, iterative_ensemble_zip_is_generated], 
                        trigger_mode="once", 
                        concurrency_id="mvsepless_app_inference_ensemble"
                    )
                    def iterative_ensemble_wrapper_fn(input_file: list, num_iters: int, save_intermediate: bool, template: str, out_format: str, flow: list[list], pref_f: bool, progress=gr.Progress(track_tqdm=True)):
                        if not flow:
                            gr.Warning(_i18n("flow_empty"))
                            return update_audio_with_size(label=_i18n("ensemble_result"), value=None), [], gr.skip(), gr.update(visible=False), gr.DownloadButton(label=_i18n("generate_zip_archive"), variant="huggingface", visible=False, **base_c_params["base"]), gr.update(value=False)
                        
                        result_path, intermediate_files = self.iterative_ensemble(
                            input_file=one_element_list_to_value(input_file),
                            output_dir=self.output_dir.gen_output_dir(),
                            flow=flow,
                            num_iters=num_iters,
                            output_format=out_format,
                            template=template,
                            save_intermediate=save_intermediate,
                            prefer_float=pref_f
                        )
                        
                        self.iterative_ensemble_history_app.add_to_history(result_path, intermediate_files, num_iters)
                        
                        return update_audio_with_size(label=_i18n("ensemble_result"), value=result_path), intermediate_files, gr.skip(), gr.update(visible=True), gr.DownloadButton(label=_i18n("generate_zip_archive"), variant="huggingface", visible=True, **base_c_params["base"]), gr.update(value=False)
                    
                    @iterative_ensemble_history.input(
                        inputs=iterative_ensemble_history, 
                        outputs=[iterative_ensemble_output_audio, iterative_ensemble_intermediate_state, iterative_ensemble_output_reuse_btn, 
                                iterative_ensemble_generate_zip_btn, iterative_ensemble_zip_is_generated]
                    )
                    def iterative_ensemble_show_history_fn(key: list):
                        result, intermediate = self.iterative_ensemble_history_app.get_from_history(one_element_list_to_value(key))
                        return update_audio_with_size(label=_i18n("ensemble_result"), value=result), intermediate, gr.update(visible=result is not None), gr.DownloadButton(label=_i18n("generate_zip_archive"), variant="huggingface", visible=result is not None, **base_c_params["base"]), gr.update(value=False)
                    
                    @iterative_ensemble_generate_zip_btn.click(
                        inputs=[iterative_ensemble_output_audio, iterative_ensemble_intermediate_state, iterative_ensemble_zip_is_generated], 
                        outputs=[iterative_ensemble_generate_zip_btn, iterative_ensemble_zip_is_generated], 
                        trigger_mode="once"
                    )
                    def generate_zip_fn(out, intermediate_state, zip_is_generated):
                        all_files = []
                        
                        if out:
                            all_files.append(out)
                        
                        if intermediate_state:
                            all_files.extend(intermediate_state)
                        
                        if zip_is_generated:
                            return gr.skip(), gr.skip()
                        else:
                            zip_file = generate_zip_archive(all_files, get_zip_output_path("iterative_ensembless"))
                            zip_is_generated = True
                            return gr.DownloadButton(label=_i18n("download_zip_archive"), variant="huggingface", value=zip_file, **base_c_params["base"]), zip_is_generated
            
                with gr.Tab(_i18n("man_ensemble_tab")):
                    with gr.Row():
                        with gr.Column():
                            manual_ensemble_upload_files = gr.File(show_label=False, **base_c_params["input_files_multi"])
                            with gr.Group():
                                manual_ensemble_input_files = gr.Dropdown(container=False, allow_custom_value=True, **base_c_params["dropdown_multi"])
                                manual_ensemble_input_files.focus(self.get_actual_input_list, inputs=[manual_ensemble_input_files, manual_ensemble_input_state], outputs=[manual_ensemble_input_files, manual_ensemble_input_state], show_progress="hidden")
                                manual_ensemble_add_uploaded_files = gr.Checkbox(label=_i18n("add_uploaded_files_to_current_list"), value=False, **base_c_params["base"])
                                manual_ensemble_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                                @manual_ensemble_upload_files.upload(inputs=[manual_ensemble_upload_files, manual_ensemble_input_files, manual_ensemble_add_uploaded_files], outputs=[manual_ensemble_upload_files, manual_ensemble_input_files])
                                def upload_files_fn(files: list, input_files: list, add_uploaded_files: bool):
                                    uploaded_files = self.input_files.upload(files)
                                    if add_uploaded_files:
                                        return gr.update(value=None), gr.update(choices=self.input_files.get_input_list(), value=[*uploaded_files, *input_files])
                                    return gr.update(value=None), gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)
                                @gr.render(inputs=[manual_ensemble_input_files, manual_ensemble_input_preview_check])
                                def preview_inputs(input: list, preview: bool):
                                    if preview:
                                        if input:
                                            for f_ in input:
                                                define_audio_with_size(basename=True, label="", value=f_, **base_c_params["output_audio"])

                        with gr.Column():
                            with gr.Group():
                                manual_ensemble_weights = gr.Textbox(label=_i18n("weights_only_for_avg_fft"), info=_i18n("weights_split"), value="", **base_c_params["base"])
                                manual_ensemble_type = gr.Dropdown(label=_i18n("ensemble_type"), info=_i18n("ensemble_type_info"), choices=ensemble_types, value=ensemble_types[0], filterable=False, **base_c_params["base"])
                                manual_ensemble_template = gr.Textbox(label=_i18n("output_template"), info=_i18n("output_metemplate_info"), value="ensemble_(COUNT)_TYPE", **base_c_params["base"])
                                manual_ensemble_format = gr.Dropdown(label=_i18n("output_format"), choices=output_formats, value=output_formats[0], filterable=False, **base_c_params["base"])
                                manual_ensemble_prefer_float = gr.Checkbox(label=_i18n("prefer_float"), value=False, **base_c_params["base"])
                                manual_ensemble_run_btn = gr.Button(_i18n("run_ensemble"), variant="primary", **base_c_params["base"])

                    with gr.Group():
                        with gr.Row(equal_height=True):
                            with gr.Column(min_width=110):
                                gr.Markdown("<h4><center>"+_i18n("history")+"</center></h4>")
                            manual_ensemble_history = gr.Dropdown(container=False, scale=13, multiselect=True, max_choices=1, **base_c_params["base"])

                    with gr.Row(equal_height=True):
                        manual_ensemble_output_audio = gr.Audio(label=_i18n("ensemble_result"), value=None, scale=15, **base_c_params["output_audio"])
                        manual_ensemble_reuse_btn = gr.Button(
                            _i18n("reuse_btn"), 
                            variant="secondary", visible=False, **base_c_params["base"]
                        )
                        @manual_ensemble_reuse_btn.click(
                            inputs=[manual_ensemble_output_audio],
                            outputs=manual_ensemble_input_files,
                        )
                        def reuse_fn(stem_audio: str) -> gr.update:
                            uploaded_files = self.input_files.upload([stem_audio], copy=True)
                            return gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)


                    @manual_ensemble_run_btn.click(inputs=[manual_ensemble_input_files, manual_ensemble_type, manual_ensemble_template, manual_ensemble_format, manual_ensemble_weights, manual_ensemble_prefer_float], outputs=[manual_ensemble_output_audio, manual_ensemble_reuse_btn])
                    def manual_ensemble_wrapper_fn(input_files: list, etype: str, template: str, out_format: str, weights_str: str, pref_f: bool, progress=gr.Progress(track_tqdm=True)):
                        weights_str = weights_str.strip(" ")
                        if weights_str != "":
                            weights = [float(weight) for weight in weights_str.split(",")]
                        else:
                            weights = []
                        result = self.manual_ensemble(input_files, self.output_dir.gen_output_dir(), weights, template, etype, out_format, pref_f)
                        self.manual_ensemble_history_app.add_to_history(etype, result)
                        return update_audio_with_size(label=_i18n("ensemble_result"), value=result), gr.update(visible=True)

                    manual_ensemble_history.focus(self.get_actual_manual_ensemble_history_list, inputs=[manual_ensemble_history, manual_ensemble_history_state], outputs=[manual_ensemble_history, manual_ensemble_history_state], show_progress="hidden")
                    @manual_ensemble_history.input(inputs=manual_ensemble_history, outputs=[manual_ensemble_output_audio, manual_ensemble_reuse_btn])
                    def manual_ensemble_show_fn(key: list):
                        output = self.manual_ensemble_history_app.get_from_history(one_element_list_to_value(key))
                        return update_audio_with_size(label=_i18n("ensemble_result"), value=output), gr.update(visible=output is not None)

            with gr.Tab(_i18n("subtract_tab")):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("<h3><center>"+_i18n("original")+"</center></h3>", container=True)
                            subtract_1_upload_file = gr.File(show_label=False, **base_c_params["input_file"])
                        with gr.Group():
                            subtract_1_input_file = gr.Dropdown(container=False, allow_custom_value=True, multiselect=True, max_choices=1, **base_c_params["base"])
                            subtract_1_input_file.focus(self.get_actual_input_list, inputs=[subtract_1_input_file, subtract_1_input_state], outputs=[subtract_1_input_file, subtract_1_input_state], show_progress="hidden")
                            subtract_1_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                            @subtract_1_upload_file.upload(inputs=subtract_1_upload_file, outputs=[subtract_1_upload_file, subtract_1_input_file])
                            def upload_file_fn(file: str):
                                uploaded_files = self.input_files.upload([file])
                                all_uploaded_files = self.input_files.get_input_list()
                                if uploaded_files:
                                    first_value = [uploaded_files[0]]
                                else:
                                    first_value = []
                                return gr.update(value=None), gr.update(choices=all_uploaded_files, value=first_value)
                            @gr.render(inputs=[subtract_1_input_file, subtract_1_input_preview_check])
                            def preview_input(input: list, preview: bool):
                                if preview:
                                    if input:
                                        define_audio_with_size(basename=True, label="", value=one_element_list_to_value(input), **base_c_params["output_audio"])
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("<h3><center>"+_i18n("stem")+"</center></h3>", container=True)
                            subtract_2_upload_file = gr.File(show_label=False, **base_c_params["input_file"])
                        with gr.Group():
                            subtract_2_input_file = gr.Dropdown(container=False, allow_custom_value=True, multiselect=True, max_choices=1, **base_c_params["base"])
                            subtract_2_input_file.focus(self.get_actual_input_list, inputs=[subtract_2_input_file, subtract_2_input_state], outputs=[subtract_2_input_file, subtract_2_input_state], show_progress="hidden")
                            subtract_2_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                            @subtract_2_upload_file.upload(inputs=subtract_2_upload_file, outputs=[subtract_2_upload_file, subtract_2_input_file])
                            def upload_file_fn(file: str):
                                uploaded_files = self.input_files.upload([file])
                                all_uploaded_files = self.input_files.get_input_list()
                                if uploaded_files:
                                    first_value = [uploaded_files[0]]
                                else:
                                    first_value = []
                                return gr.update(value=None), gr.update(choices=all_uploaded_files, value=first_value)
                            @gr.render(inputs=[subtract_2_input_file, subtract_2_input_preview_check])
                            def preview_input(input: list, preview: bool):
                                if preview:
                                    if input:
                                        define_audio_with_size(basename=True, label="", value=one_element_list_to_value(input), **base_c_params["output_audio"])

                with gr.Group():
                    subtract_use_spec_invert = gr.Checkbox(label=_i18n("use_spec_invert"), value=False, **base_c_params["base"])
                    subtract_template = gr.Textbox(label=_i18n("output_template"), info=_i18n("output_itemplate_info"), value="NAME_(TYPE)_inverted", **base_c_params["base"])
                    subtract_output_format = gr.Dropdown(label=_i18n("output_format"), choices=output_formats, value=output_formats[0], filterable=False, **base_c_params["base"])
                    subtract_prefer_float = gr.Checkbox(label=_i18n("prefer_float"), value=False, **base_c_params["base"])
                    subtract_run_btn = gr.Button(_i18n("subtract"), variant="primary", **base_c_params["base"])

                with gr.Group():
                    with gr.Row(equal_height=True):
                        with gr.Column(min_width=110):
                            gr.Markdown("<h4><center>"+_i18n("history")+"</center></h4>")
                        subtract_history = gr.Dropdown(container=False, scale=13, multiselect=True, max_choices=1, **base_c_params["base"])

                with gr.Row(equal_height=True):
                    subtract_output_audio = gr.Audio(label=_i18n("inverted_result"), value=None, scale=15, **base_c_params["output_audio"])
                    subtract_output_audio_reuse_btn = gr.Button(
                        _i18n("reuse_btn"), 
                        variant="secondary", visible=False, **base_c_params["base"]
                    )
                    @subtract_output_audio_reuse_btn.click(
                        inputs=[subtract_output_audio],
                        outputs=subtract_1_input_file,
                    )
                    def reuse_fn(stem_audio: str) -> gr.update:
                        uploaded_files = self.input_files.upload([stem_audio], copy=True)
                        return gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)

                    @subtract_run_btn.click(inputs=[subtract_1_input_file, subtract_2_input_file, subtract_output_format, subtract_use_spec_invert, subtract_template, subtract_prefer_float], outputs=[subtract_output_audio, subtract_output_audio_reuse_btn])
                    def subtract_run_fn(input_1: list, input_2: list, out_format: str, spec_invert: bool, template: str, pref_f: bool, progress=gr.Progress(track_tqdm=True)):
                        result = self.subtract(one_element_list_to_value(input_1), one_element_list_to_value(input_2), self.output_dir.gen_output_dir(), out_format, spec_invert, template, pref_f)
                        self.subtract_history_app.add_to_history(("spectrogram" if spec_invert else "waveform"), result)
                        return update_audio_with_size(label=_i18n("inverted_result"), value=result), gr.update(visible=True)

                    subtract_history.focus(self.get_actual_subtract_history_list, inputs=[subtract_history, subtract_history_state], outputs=[subtract_history, subtract_history_state], show_progress="hidden")
                    @subtract_history.input(inputs=subtract_history, outputs=[subtract_output_audio, subtract_output_audio_reuse_btn])
                    def subtract_show_fn(key: list):
                        output = self.subtract_history_app.get_from_history(one_element_list_to_value(key))
                        return update_audio_with_size(label=_i18n("inverted_result"), value=output), gr.update(visible=output is not None)

            with gr.Tab(_i18n("vbach_tab")):
                with gr.Tab(_i18n("inference")):
                    with gr.Row():
                        with gr.Column():
                            vbach_upload_files = gr.File(show_label=False, **base_c_params["input_files_multi"])
                            with gr.Group():
                                vbach_input_files = gr.Dropdown(container=False, allow_custom_value=True, **base_c_params["dropdown_multi"])
                                vbach_input_files.focus(
                                    self.get_actual_input_list, 
                                    inputs=[vbach_input_files, vbach_input_state], 
                                    outputs=[vbach_input_files, vbach_input_state], 
                                    show_progress="hidden"
                                )
                                vbach_input_add_uploaded_files = gr.Checkbox(label=_i18n("add_uploaded_files_to_current_list"), value=False, **base_c_params["base"])
                                vbach_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                                
                                @vbach_upload_files.upload(inputs=[vbach_upload_files, vbach_input_files, vbach_input_add_uploaded_files], outputs=[vbach_upload_files, vbach_input_files])
                                def upload_files_fn(files: list, input_files: list, add_uploaded_files: bool):
                                    uploaded_files = self.input_files.upload(files)
                                    if add_uploaded_files:
                                        return gr.update(value=None), gr.update(choices=self.input_files.get_input_list(), value=[*uploaded_files, *input_files])
                                    return gr.update(value=None), gr.update(choices=self.input_files.get_input_list(), value=uploaded_files)
                                
                                @gr.render(inputs=[vbach_input_files, vbach_input_preview_check])
                                def preview_inputs(input: list, preview: bool):
                                    if preview and input:
                                        for f_ in input:
                                            define_audio_with_size(basename=True, label="", value=f_, **base_c_params["output_audio"])
                        
                        with gr.Column():
                            with gr.Group():
                                vbach_model_path = gr.Dropdown(
                                    label=_i18n("model_path"), multiselect=True, allow_custom_value=True, max_choices=1, **base_c_params["base"])
                                vbach_model_path.focus(
                                    self.get_actual_vbach_models_list,
                                    inputs=[vbach_model_path, vbach_models_state],
                                    outputs=[vbach_model_path, vbach_models_state],
                                    show_progress="hidden"
                                )
                                
                                vbach_index_path = gr.Dropdown(label=_i18n("index_path"), allow_custom_value=True, multiselect=True, max_choices=1, **base_c_params["base"])
                                vbach_index_path.focus(
                                    self.get_actual_vbach_index_list,
                                    inputs=[vbach_index_path, vbach_index_state],
                                    outputs=[vbach_index_path, vbach_index_state],
                                    show_progress="hidden"
                                )
                                
                                vbach_pitch = gr.Slider(
                                    label=_i18n("pitch"),
                                    minimum=-36,
                                    maximum=36,
                                    step=0.1,
                                    value=0,
                                    **base_c_params["base"]
                                )
                                
                                vbach_f0method = gr.Dropdown(
                                    label=_i18n("f0_method"),
                                    choices=f0_methods,
                                    value=f0_methods[0],
                                    **base_c_params["base"]
                                )

                                vbach_crepe_hop_length = gr.Slider(
                                    label=_i18n("crepe_hop_length"),
                                    info=_i18n("crepe_hop_length_info"),
                                    minimum=24,
                                    maximum=512,
                                    step=8,
                                    value=128,
                                    visible=False,
                                    **base_c_params["base"]
                                )
                                vbach_f0method.change(lambda x: gr.update(visible=x in crepe_like_f0_methods), inputs=vbach_f0method, outputs=vbach_crepe_hop_length, show_progress="hidden")

                                vbach_index_rate = gr.Slider(
                                    label=_i18n("index_rate"),
                                    info=_i18n("index_rate_info"),
                                    minimum=0,
                                    maximum=1,
                                    step=0.05,
                                    value=0,
                                    **base_c_params["base"]
                                )
                                
                                vbach_volume_envelope = gr.Slider(
                                    label=_i18n("volume_envelope"),
                                    info=_i18n("volume_envelope_info"),
                                    minimum=0,
                                    maximum=1,
                                    step=0.05,
                                    value=0,
                                    **base_c_params["base"]
                                )
                                
                                vbach_protect = gr.Slider(
                                    label=_i18n("protect"),
                                    info=_i18n("protect_info"),
                                    minimum=0,
                                    maximum=0.5,
                                    step=0.05,
                                    value=0.35,
                                    **base_c_params["base"]
                                )
                                
                                with gr.Accordion(label=_i18n("advanced_params"), open=False):
                                    vbach_use_transformers = gr.Checkbox(
                                        label=_i18n("vbach_use_transformers"),
                                        value=False,
                                        **base_c_params["base"]
                                    )
                                    
                                    vbach_embedder = gr.Dropdown(
                                        label=_i18n("vbach_embedder"),
                                        info=_i18n("vbach_embedder_info"),
                                        choices=huberts_fairseq,
                                        value=huberts_fairseq[0],
                                        **base_c_params["dropdown"]
                                    )
                                    
                                    @vbach_use_transformers.change(inputs=vbach_use_transformers, outputs=vbach_embedder, show_progress="hidden")
                                    def show_embedders_vbach_fn(transformers: bool):
                                        if transformers:
                                            return gr.update(choices=huberts_transformers, value=huberts_transformers[0])
                                        else:
                                            return gr.update(choices=huberts_fairseq, value=huberts_fairseq[0])
                                    
                                    vbach_f0_min = gr.Slider(
                                        label=_i18n("f0_min"),
                                        minimum=50,
                                        maximum=1100,
                                        value=50,
                                        step=1,
                                        **base_c_params["base"]
                                    )
                                    vbach_f0_max = gr.Slider(
                                        label=_i18n("f0_max"),
                                        minimum=350,
                                        maximum=3500,
                                        value=1100,
                                        step=1,
                                        **base_c_params["base"]
                                    )

                                    vbach_chunk_duration = gr.Number(
                                        label=_i18n("chunk_duration"),
                                        minimum=1,
                                        maximum=30,
                                        value=7,
                                        step=1,
                                        **base_c_params["base"]
                                    )
                                    
                                    vbach_stereo_mode = gr.Dropdown(
                                        label=_i18n("stereo_mode"),
                                        info=_i18n("stereo_mode_info"),
                                        choices=stereo_modes,
                                        value=stereo_modes[0],
                                        **base_c_params["base"]
                                    )
                                
                                vbach_template = gr.Textbox(
                                    label=_i18n("output_template"), 
                                    info=_i18n("output_vbach_template_info"),
                                    value="MODEL_NAME_F0METHOD_PITCH", 
                                    **base_c_params["base"]
                                )
                                vbach_output_format = gr.Dropdown(
                                    label=_i18n("output_format"), 
                                    choices=output_formats, 
                                    value=output_formats[0], 
                                    filterable=False, 
                                    **base_c_params["base"]
                                )
                                vbach_convert_btn = gr.Button(_i18n("convert"), variant="primary", **base_c_params["base"])
                    
                    with gr.Group():
                        with gr.Row(equal_height=True):
                            with gr.Column(min_width=110):
                                gr.Markdown(f"<h4><center>{_i18n('history')}</center></h4>")
                            vbach_history = gr.Dropdown(container=False, scale=13, multiselect=True, max_choices=1, **base_c_params["base"])
                            vbach_history.focus(
                                self.get_actual_vbach_history_list,
                                inputs=[vbach_history, vbach_history_state],
                                outputs=[vbach_history, vbach_history_state],
                                show_progress="hidden"
                            )
                        vbach_off_players_output = gr.Checkbox(label=_i18n("off_audio_players_output"), info=_i18n("off_audio_players_output_info"), value=False, **base_c_params["base"])
                        vbach_state = gr.State()
                        @gr.render(inputs=[vbach_state, vbach_off_players_output])
                        def show_results(state, off_players_output: bool):
                            if state:
                                zip_is_generated = False
                                for result_path in state:
                                    with gr.Group():
                                        if off_players_output:
                                            define_download_button_with_size(
                                                value=result_path,
                                                label=Path(result_path).stem,
                                                **base_c_params["base"],
                                                variant="huggingface",
                                                scale=15,
                                            )
                                        else:
                                            define_audio_with_size(
                                                value=result_path,
                                                label=Path(result_path).stem,
                                                **base_c_params["output_audio"]
                                            )

                                generate_zip_btn = gr.DownloadButton(label=_i18n("generate_zip_archive"), variant="huggingface", **base_c_params["base"])
                                @generate_zip_btn.click(outputs=generate_zip_btn, trigger_mode="once")
                                def generate_zip_fn():
                                    nonlocal zip_is_generated
                                    if zip_is_generated:
                                        return gr.skip()
                                    else:
                                        zip_file = generate_zip_archive(state, get_zip_output_path("vbach"))
                                        zip_is_generated = True
                                        return gr.DownloadButton(label=_i18n("download_zip_archive"), variant="huggingface", value=zip_file, **base_c_params["base"])
                            else:
                                gr.Markdown(f"<h3><center>{_i18n('no_conversion_results')}</center></h3>", container=True)

                    
                    @vbach_convert_btn.click(
                        inputs=[vbach_input_files, vbach_model_path, vbach_index_path, vbach_pitch, vbach_f0method, 
                                vbach_index_rate, vbach_volume_envelope, vbach_protect, vbach_crepe_hop_length,
                                vbach_chunk_duration, vbach_stereo_mode, vbach_embedder, vbach_use_transformers,
                                vbach_template, vbach_output_format, vbach_f0_min, vbach_f0_max],
                        outputs=[vbach_state, vbach_upload_files],
                        trigger_mode="once", concurrency_id="mvsepless_app_inference_vbach"
                    )
                    def convert_wrapper(
                        input_files: list, model_path: str, index_path: str, pitch: float, f0_method: str,
                        index_rate: float, volume_envelope: float, protect: float, hop_length: int,
                        chunk_duration: int, stereo_mode: str, embedder_model: str, use_transformers: bool,
                        template: str, output_format: str, f0_min: int, f0_max: int, progress=gr.Progress(track_tqdm=True)
                    ):
                        
                        if not model_path:
                            gr.Warning(_i18n("model_not_selected"))
                            return [], gr.skip()
                        
                        output_dir = self.output_dir.generate(base_names_app_dirs[7])
                        results = self.vbach_converter.convert_audio(
                            audio_input=input_files,
                            output_dir=output_dir,
                            model_path=one_element_list_to_value(model_path),
                            index_path=one_element_list_to_value(index_path),
                            pitch=pitch,
                            f0_method=f0_method,
                            index_rate=index_rate,
                            volume_envelope=volume_envelope,
                            protect=protect,
                            hop_length=hop_length,
                            embedder_model=embedder_model,
                            use_transformers=use_transformers,
                            output_format=output_format,
                            stereo_mode=stereo_mode,
                            chunk_duration=chunk_duration,
                            f0_min=f0_min,
                            f0_max=f0_max,
                            template=template
                        )
                        
                        self.vbach_history_app.add_to_history(Path(one_element_list_to_value(model_path)).stem, f0_method, pitch, results)
                        return results, gr.skip()
                                
                    @vbach_history.input(inputs=vbach_history, outputs=vbach_state)
                    def show_history_fn(key: list):
                        state = self.vbach_history_app.get_from_history(one_element_list_to_value(key))
                        return state

                with gr.Tab(_i18n("f0_extraction_tab")):
                    with gr.Row():
                        with gr.Column():
                            f0_upload_file = gr.File(show_label=False, **base_c_params["input_file"])
                            with gr.Group():
                                f0_input_file = gr.Dropdown(container=False, allow_custom_value=True, multiselect=True, max_choices=1, **base_c_params["base"])
                                f0_input_file.focus(self.get_actual_input_list, inputs=[f0_input_file, f0_input_state], outputs=[f0_input_file, f0_input_state], show_progress="hidden")
                                f0_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                                
                                @f0_upload_file.upload(inputs=f0_upload_file, outputs=[f0_upload_file, f0_input_file])
                                def upload_f0_file_fn(file: str):
                                    uploaded_files = self.input_files.upload([file])
                                    all_uploaded_files = self.input_files.get_input_list()
                                    if uploaded_files:
                                        first_value = [uploaded_files[0]]
                                    else:
                                        first_value = []
                                    return gr.update(value=None), gr.update(choices=all_uploaded_files, value=first_value)
                                
                                @gr.render(inputs=[f0_input_file, f0_input_preview_check])
                                def preview_f0_input(input: list, preview: bool):
                                    if preview and input:
                                        define_audio_with_size(basename=True, label="", value=one_element_list_to_value(input), **base_c_params["output_audio"])
                        
                        with gr.Column():
                            with gr.Group():
                                f0_method_dropdown = gr.Dropdown(
                                    label=_i18n("f0_method"),
                                    choices=f0_methods,
                                    value=f0_methods[0],
                                    **base_c_params["base"]
                                )
                                
                                f0_min_slider = gr.Slider(
                                    label=_i18n("f0_min"),
                                    minimum=50,
                                    maximum=1100,
                                    value=50,
                                    step=1,
                                    **base_c_params["base"]
                                )
                                
                                f0_max_slider = gr.Slider(
                                    label=_i18n("f0_max"),
                                    minimum=350,
                                    maximum=3500,
                                    value=1100,
                                    step=1,
                                    **base_c_params["base"]
                                )
                                
                                f0_extract_btn = gr.Button(_i18n("extract_f0"), variant="primary", **base_c_params["base"])
                    
                    with gr.Group():
                        with gr.Row(equal_height=True):
                            with gr.Column(min_width=110):
                                gr.Markdown(_i18n("f0_file_info"), container=True)
                                gr.Markdown(f"<h4><center>{_i18n('f0_extraction_results')}</center></h4>")
                                f0_result_file = gr.File(value=None, label=_i18n("download_f0_json"), type="filepath", interactive=False)

                    @f0_extract_btn.click(
                        inputs=[f0_input_file, f0_method_dropdown, f0_min_slider, f0_max_slider],
                        outputs=[f0_result_file, f0_upload_file],
                        trigger_mode="once", concurrency_id="mvsepless_app_inference"
                    )
                    def extract_f0_wrapper(input_file: list, method: str, f0_min: int, f0_max: int, progress=gr.Progress(track_tqdm=True)):
                        if not input_file:
                            gr.Warning(_i18n("no_audio_selected"))
                            return gr.skip(), gr.skip()
                        
                        audio_path = one_element_list_to_value(input_file)
                        input_name = Path(audio_path).stem
                        result_path = f0_extract_and_write(
                            audio_path, 
                            f0_method=method, 
                            f0_min=f0_min, 
                            f0_max=f0_max, 
                            output_path=Namer.iter(self.f0_gen_output_path.generate_output_path(input_name, method))
                        )
                        gr.Info(_i18n("f0_extraction_complete"))
                        return result_path, gr.skip()

                with gr.Tab(_i18n("vbach_inference_custom_f0")):
                    with gr.Row():
                        with gr.Column():
                            vbach_custom_upload_file = gr.File(show_label=False, **base_c_params["input_file"])
                            with gr.Group():
                                vbach_custom_input_file = gr.Dropdown(container=False, allow_custom_value=True, multiselect=True, max_choices=1, **base_c_params["base"])
                                vbach_custom_input_file.focus(self.get_actual_input_list, inputs=[vbach_custom_input_file, vbach_custom_input_state], outputs=[vbach_custom_input_file, vbach_custom_input_state], show_progress="hidden")
                                vbach_custom_input_preview_check = gr.Checkbox(label=_i18n("show_preview"), value=False, **base_c_params["base"])
                                
                                @vbach_custom_upload_file.upload(inputs=vbach_custom_upload_file, outputs=[vbach_custom_upload_file, vbach_custom_input_file])
                                def upload_vbach_custom_file_fn(file: str):
                                    uploaded_files = self.input_files.upload([file])
                                    all_uploaded_files = self.input_files.get_input_list()
                                    if uploaded_files:
                                        first_value = [uploaded_files[0]]
                                    else:
                                        first_value = []
                                    return gr.update(value=None), gr.update(choices=all_uploaded_files, value=first_value)
                                
                                @gr.render(inputs=[vbach_custom_input_file, vbach_custom_input_preview_check])
                                def preview_vbach_custom_input(input: list, preview: bool):
                                    if preview and input:
                                        define_audio_with_size(basename=True, label="", value=one_element_list_to_value(input), **base_c_params["output_audio"])
                        
                        with gr.Column():
                            with gr.Group():
                                vbach_custom_model_path = gr.Dropdown(
                                    label=_i18n("model_path"), multiselect=True, allow_custom_value=True, max_choices=1, **base_c_params["base"])
                                vbach_custom_model_path.focus(
                                    self.get_actual_vbach_models_list,
                                    inputs=[vbach_custom_model_path, vbach_custom_models_state],
                                    outputs=[vbach_custom_model_path, vbach_custom_models_state],
                                    show_progress="hidden"
                                )
                                
                                vbach_custom_index_path = gr.Dropdown(
                                    label=_i18n("index_path"), multiselect=True, allow_custom_value=True, max_choices=1, **base_c_params["base"])
                                vbach_custom_index_path.focus(
                                    self.get_actual_vbach_index_list,
                                    inputs=[vbach_custom_index_path, vbach_custom_index_state],
                                    outputs=[vbach_custom_index_path, vbach_custom_index_state],
                                    show_progress="hidden"
                                )
                                gr.Markdown(_i18n("f0_file_info"), container=True)
                                vbach_custom_f0_file = gr.File(
                                    label=_i18n("f0_json_file"),
                                    file_types=[".json"],
                                    type="filepath",
                                    **base_c_params["base"]
                                )
                                
                                vbach_custom_pitch = gr.Slider(
                                    label=_i18n("pitch"),
                                    minimum=-36,
                                    maximum=36,
                                    step=0.1,
                                    value=0,
                                    **base_c_params["base"]
                                )
                                
                                vbach_custom_index_rate = gr.Slider(
                                    label=_i18n("index_rate"),
                                    info=_i18n("index_rate_info"),
                                    minimum=0,
                                    maximum=1,
                                    step=0.05,
                                    value=0,
                                    **base_c_params["base"]
                                )
                                
                                vbach_custom_volume_envelope = gr.Slider(
                                    label=_i18n("volume_envelope"),
                                    info=_i18n("volume_envelope_info"),
                                    minimum=0,
                                    maximum=1,
                                    step=0.05,
                                    value=0,
                                    **base_c_params["base"]
                                )
                                
                                vbach_custom_protect = gr.Slider(
                                    label=_i18n("protect"),
                                    info=_i18n("protect_info"),
                                    minimum=0,
                                    maximum=0.5,
                                    step=0.05,
                                    value=0.35,
                                    **base_c_params["base"]
                                )
                                
                                with gr.Accordion(label=_i18n("advanced_params"), open=False):
                                    vbach_custom_use_transformers = gr.Checkbox(
                                        label=_i18n("vbach_use_transformers"),
                                        value=False,
                                        **base_c_params["base"]
                                    )
                                    
                                    vbach_custom_embedder = gr.Dropdown(
                                        label=_i18n("vbach_embedder"),
                                        info=_i18n("vbach_embedder_info"),
                                        choices=huberts_fairseq,
                                        value=huberts_fairseq[0],
                                        **base_c_params["dropdown"]
                                    )
                                    
                                    @vbach_custom_use_transformers.change(inputs=vbach_custom_use_transformers, outputs=vbach_custom_embedder, show_progress="hidden")
                                    def show_embedders_vbach_custom_fn(transformers: bool):
                                        if transformers:
                                            return gr.update(choices=huberts_transformers, value=huberts_transformers[0])
                                        else:
                                            return gr.update(choices=huberts_fairseq, value=huberts_fairseq[0])
                                    
                                    vbach_custom_f0_min = gr.Slider(
                                        label=_i18n("f0_min"),
                                        minimum=50,
                                        maximum=1100,
                                        value=50,
                                        step=1,
                                        **base_c_params["base"]
                                    )
                                    vbach_custom_f0_max = gr.Slider(
                                        label=_i18n("f0_max"),
                                        minimum=350,
                                        maximum=3500,
                                        value=1100,
                                        step=1,
                                        **base_c_params["base"]
                                    )

                                    vbach_custom_chunk_duration = gr.Number(
                                        label=_i18n("chunk_duration"),
                                        minimum=1,
                                        maximum=30,
                                        value=7,
                                        step=1,
                                        **base_c_params["base"]
                                    )
                                    
                                
                                vbach_custom_template = gr.Textbox(
                                    label=_i18n("output_template"), 
                                    info=_i18n("output_vbach_custom_template_info"),
                                    value="MODEL_NAME_F0METHOD_PITCH", 
                                    **base_c_params["base"]
                                )
                                vbach_custom_output_format = gr.Dropdown(
                                    label=_i18n("output_format"), 
                                    choices=output_formats, 
                                    value=output_formats[0], 
                                    filterable=False, 
                                    **base_c_params["base"]
                                )
                                vbach_custom_convert_btn = gr.Button(_i18n("convert_custom_f0"), variant="primary", **base_c_params["base"])
                    
                    with gr.Row(equal_height=True):
                        vbach_custom_output_audio = gr.Audio(
                            value=None,
                            label=_i18n("vbach_result"),
                            **base_c_params["output_audio"]
                        )
                    
                    @vbach_custom_convert_btn.click(
                        inputs=[vbach_custom_input_file, vbach_custom_model_path, vbach_custom_index_path, 
                                vbach_custom_pitch, vbach_custom_f0_file, vbach_custom_index_rate, 
                                vbach_custom_volume_envelope, vbach_custom_protect, vbach_custom_embedder, 
                                vbach_custom_use_transformers, vbach_custom_template, vbach_custom_output_format, 
                                vbach_custom_f0_min, vbach_custom_f0_max, vbach_custom_chunk_duration],
                        outputs=[vbach_custom_output_audio, vbach_custom_upload_file],
                        trigger_mode="once", concurrency_id="mvsepless_app_inference_vbach"
                    )
                    def convert_custom_f0_wrapper(
                        input_files: list, model_path: str, index_path: str, pitch: float, f0_file_path: str, 
                        index_rate: float, volume_envelope: float, protect: float, embedder_model: str, 
                        use_transformers: bool, template: str, output_format: str, f0_min: int, f0_max: int, 
                        chunk_duration: int, progress=gr.Progress(track_tqdm=True)
                    ):
                        if not model_path:
                            gr.Warning(_i18n("model_not_selected"))
                            return update_audio_with_size(label=_i18n("vbach_result"), value=None), gr.skip()
                        
                        if not f0_file_path:
                            gr.Warning(_i18n("no_f0_file_selected"))
                            return update_audio_with_size(label=_i18n("vbach_result"), value=None), gr.skip()
                        
                        output_dir = self.output_dir.generate(base_names_app_dirs[7])
                        
                        result = self.vbach_converter.convert_audio_custom_f0(
                            audio_input=one_element_list_to_value(input_files),
                            output_dir=output_dir,
                            model_path=one_element_list_to_value(model_path),
                            index_path=one_element_list_to_value(index_path),
                            pitch=pitch,
                            f0_file=f0_file_path,
                            index_rate=index_rate,
                            volume_envelope=volume_envelope,
                            protect=protect,
                            embedder_model=embedder_model,
                            use_transformers=use_transformers,
                            output_format=output_format,
                            f0_min=f0_min,
                            f0_max=f0_max,
                            chunk_duration=chunk_duration,
                            template=template
                        )
                        self.vbach_history_app.add_to_history(Path(one_element_list_to_value(model_path)).stem, "custom", pitch, [result])
                        return update_audio_with_size(label="", basename=True, value=result), gr.skip()
            with gr.Tab(_i18n("upload_audio")):
                
                with gr.Accordion(label=_i18n("upload_from_zip"), open=True):
                    with gr.Group():
                        gr.Markdown("<h3><center>"+_i18n("upload_zip_placeholder")+"</center></h3>")
                        upload_zip_file = gr.File(
                            show_label=False, 
                            type="filepath", 
                            file_count="single", 
                            file_types=[".zip"],
                            **base_c_params["base"]
                        )
                        upload_zip_extract_btn = gr.Button(_i18n("extract_and_upload"), variant="primary", **base_c_params["base"])
                        upload_zip_status = gr.Textbox(label=_i18n("status"), value="", interactive=False, lines=3, max_lines=3)
                        
                        @upload_zip_extract_btn.click(inputs=upload_zip_file, outputs=[upload_zip_status, upload_zip_file])
                        def extract_and_upload_zip(zip_path: str, progress=gr.Progress(track_tqdm=True)):
                            if not zip_path:
                                return _i18n("path_not_specified"), gr.skip()
                            
                            extracted_files = []
                            with tempfile.TemporaryDirectory() as tmpdirname:
                                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                    zip_ref.extractall(tmpdirname)
                                
                                tmp_path = Path(tmpdirname)
                                audio_files = get_audio_files_from_list([f.as_posix() for f in tmp_path.rglob("*")], only_files=False)
                                
                                if audio_files:
                                    uploaded = self.input_files.upload(audio_files, copy=True)
                                    extracted_files.extend(uploaded)
                                    status = _i18n("uploaded_files_count", count=len(uploaded))
                                else:
                                    status = _i18n("no_audio_files_in_zip")
                            
                            gr.Info(title=status, message="")
                            return status, gr.update(value=None)
                
                with gr.Accordion(label=_i18n("upload_from_files"), open=True):
                    with gr.Group():
                        upload_files_component = gr.File(
                            show_label=False,
                            **base_c_params["input_files_multi"]
                        )
                        upload_files_btn = gr.Button(_i18n("upload"), variant="primary", **base_c_params["base"])
                        upload_files_status = gr.Textbox(label=_i18n("status"), value="", interactive=False, lines=3, max_lines=3)
                        
                        @upload_files_btn.click(inputs=upload_files_component, outputs=[upload_files_status, upload_files_component])
                        def upload_audio_files(files: list, progress=gr.Progress(track_tqdm=True)):
                            if not files:
                                return _i18n("paths_not_specified"), gr.skip()
                            
                            uploaded = self.input_files.upload(files, copy=True)
                            status = _i18n("uploaded_files_count", count=len(uploaded))
                            gr.Info(title=status, message="")
                            return status, gr.update(value=[])

                with gr.Accordion(label=_i18n("upload_from_url"), open=False):
                    with gr.Group():
                        gr.Markdown("<h3><center>"+_i18n("supported_yt_dlp_info")+"</center></h3>")
                        
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=3):
                                upload_url_input = gr.Textbox(
                                    label=_i18n("audio_url"), 
                                    placeholder="https://youtu.be/... или https://example.com/audio.mp3", 
                                    **base_c_params["base"]
                                )
                            with gr.Column(scale=1):
                                upload_url_btn = gr.Button(_i18n("download_and_upload"), variant="primary", **base_c_params["base"])
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                upload_url_format = gr.Dropdown(
                                    label=_i18n("output_format"),
                                    choices=output_formats,
                                    value=output_formats[0],
                                    **base_c_params["dropdown"]
                                )
                            with gr.Column(scale=1):
                                upload_url_bitrate = gr.Dropdown(
                                    label=_i18n("bitrate"),
                                    choices=["64", "128", "192", "256", "320"],
                                    value="320",
                                    **base_c_params["dropdown"]
                                )
                        
                        with gr.Accordion(label=_i18n("cookie_settings"), open=False):
                            gr.Markdown(_i18n("cookie_explanation"))
                            
                            with gr.Row():
                                with gr.Column(scale=2):
                                    upload_cookie_file = gr.File(
                                        label=_i18n("cookie_file"),
                                        file_types=[".txt", ".netscape"],
                                        **base_c_params["input_file"]
                                    )
                                with gr.Column(scale=1):
                                    upload_cookie_status = gr.Textbox(
                                        label=_i18n("cookie_status"),
                                        value=_i18n("cookie_not_loaded"),
                                        interactive=False,
                                        lines=5
                                    )
                            
                            @upload_cookie_file.change(inputs=upload_cookie_file, outputs=upload_cookie_status)
                            def cookie_file_selected(file: str):
                                if file and Path(file).exists():
                                    return _i18n("cookie_loaded", path=file)
                                return _i18n("cookie_not_loaded")
                        
                        upload_url_status = gr.Textbox(
                            label=_i18n("status"), 
                            value="", 
                            interactive=False, 
                            lines=5, 
                            max_lines=5
                        )
                        
                        @upload_url_btn.click(
                            inputs=[upload_url_input, upload_url_format, upload_url_bitrate, upload_cookie_file], 
                            outputs=[upload_url_status, upload_url_input]
                        )
                        def download_from_url(url: str, audio_format: str, bitrate: str, cookie_path: str, progress=gr.Progress(track_tqdm=True)):
                            if not url:
                                return _i18n("path_not_specified"), gr.skip()
                            
                            try:
                                # Используем dw_yt_dlp для скачивания
                                output_dir = self.input_files.input_dir_base / "yt_downloads"
                                output_dir.mkdir(parents=True, exist_ok=True)
                                
                                result_path = dw_yt_dlp(
                                    url=url,
                                    output_dir=output_dir,
                                    output_format=audio_format,
                                    output_bitrate=bitrate,
                                    cookie=cookie_path if cookie_path else None
                                )
                                
                                if result_path and Path(result_path).exists():
                                    # Загружаем скачанный файл в базу входных файлов
                                    uploaded = self.input_files.upload([result_path], copy=True)
                                    status = _i18n("downloaded_and_uploaded", count=len(uploaded))
                                    gr.Info(title=status, message="")
                                    return status, gr.update(value="")
                                else:
                                    return _i18n("download_failed_no_file"), gr.skip()
                                    
                            except Exception as e:
                                return _i18n("download_error", error=str(e)), gr.skip()
                
                with gr.Accordion(label=_i18n("upload_from_path"), open=False):
                    with gr.Group():
                        upload_path_input = gr.Textbox(label=_i18n("folder_path"), placeholder="/path/to/audio/folder", **base_c_params["base"])
                        upload_path_btn = gr.Button(_i18n("scan_and_upload"), variant="primary", **base_c_params["base"])
                        upload_path_status = gr.Textbox(label=_i18n("status"), value="", interactive=False, lines=5, max_lines=5)
                        
                        @upload_path_btn.click(inputs=upload_path_input, outputs=[upload_path_status, upload_path_input])
                        def upload_from_path(path: str, progress=gr.Progress(track_tqdm=True)):
                            if not path:
                                return _i18n("path_not_specified"), gr.skip()
                            
                            path_obj = Path(path)
                            if not path_obj.exists():
                                return _i18n("path_not_exist"), gr.skip()
                            
                            if path_obj.is_file():
                                if check(path_obj):
                                    uploaded = self.input_files.upload([path_obj.as_posix()], copy=True)
                                    status = _i18n("uploaded_files_count", count=len(uploaded))
                                else:
                                    status = _i18n("file_is_not_audio")
                            elif path_obj.is_dir():
                                audio_files = get_audio_files_from_list([f.as_posix() for f in path_obj.rglob("*")], only_files=False)
                                if audio_files:
                                    uploaded = self.input_files.upload(audio_files, copy=True)
                                    status = _i18n("uploaded_files_count", count=len(uploaded))
                                else:
                                    status = _i18n("no_audio_files_in_directory")
                            else:
                                status = _i18n("invalid_path")
                            
                            gr.Info(title=status, message="")
                            return status, gr.skip()
                
            with gr.Tab(_i18n("download_model")):
                with gr.Tab(_i18n("separation_tab")):
                    with gr.Group():
                        sep_download_model_name = gr.Dropdown(label=_i18n("model_name"), choices=all_models, value=default_model, **base_c_params["base"])
                        sep_download_btn = gr.Button(_i18n("download"), variant="primary", **base_c_params["base"])
                        sep_download_status = gr.Textbox(label=_i18n("status"), value="", interactive=False, lines=3, max_lines=3)
                        @sep_download_btn.click(inputs=sep_download_model_name, outputs=sep_download_status)
                        def download_sep_model_fn(name: str, progress=gr.Progress(track_tqdm=True)):
                            status = self.download(name)
                            gr.Info(title=status, message="")
                            return status
                with gr.Tab(_i18n("custom_separation_models_tab")):
                    with gr.Tab(_i18n("download_from_internet")):
                        gr.Markdown("<h3><center>"+_i18n("supported_only_direct_links")+"</center></h3>")
                        with gr.Accordion(label=_i18n("download_model_files"), open=False):
                            with gr.Group():
                                custom_sep_url_checkpoint = gr.Textbox(label=_i18n("custom_checkpoint_link"), **base_c_params["base"])
                                custom_sep_url_config = gr.Textbox(label=_i18n("custom_config_link"), **base_c_params["base"])
                                custom_sep_url_model_btn = gr.Button(_i18n("download_and_move_to_models_dir"), variant="primary", **base_c_params["base"])
                                custom_sep_url_model_status = gr.Textbox(label=_i18n("status"), value="", interactive=False, lines=3, max_lines=3)
                                @custom_sep_url_model_btn.click(inputs=[custom_sep_url_checkpoint, custom_sep_url_config], outputs=custom_sep_url_model_status)
                                def download_custom_sep_files_fn(url_checkpoint: str, url_config: str, progress=gr.Progress(track_tqdm=True)):
                                    status = self.custom_sep_model_manager.download_model(checkpoint_url=url_checkpoint, config_url=url_config)
                                    return status
                    with gr.Tab(_i18n("download_from_local_device")):
                        with gr.Accordion(label=_i18n("download_model_files"), open=False):
                            with gr.Row():
                                with gr.Column():
                                    with gr.Group():
                                        gr.Markdown("<h3><center>"+_i18n("custom_checkpoint_placeholder")+"</center></h3>")
                                        custom_sep_local_checkpoint = gr.File(show_label=False, type="filepath", file_count="multiple", file_types=[".pth", ".ckpt", ".pt", ".chpt"], **base_c_params["base"])
                                        @custom_sep_local_checkpoint.upload(inputs=custom_sep_local_checkpoint, outputs=custom_sep_local_checkpoint)
                                        def upload_custom_sep_checkpoint_fn(files: list, progress=gr.Progress(track_tqdm=True)):
                                            self.custom_sep_model_manager.upload_checkpoint(files)
                                            return gr.update(value=[])
                                
                                with gr.Column():
                                    with gr.Group():
                                        gr.Markdown("<h3><center>"+_i18n("custom_config_placeholder")+"</center></h3>")
                                        custom_sep_local_config = gr.File(show_label=False, type="filepath", file_count="multiple", file_types=[".yaml", ".yml"], **base_c_params["base"])
                                        @custom_sep_local_config.upload(inputs=custom_sep_local_config, outputs=custom_sep_local_config)
                                        def upload_custom_sep_config_fn(files: list, progress=gr.Progress(track_tqdm=True)):
                                            self.custom_sep_model_manager.upload_config(files)
                                            return gr.update(value=[])
                with gr.Tab(_i18n("vbach_models_tab")):
                    with gr.Tab(_i18n("download_from_internet")):
                        gr.Markdown("<h3><center>"+_i18n("supported_only_direct_links")+"</center></h3>")
                        with gr.Accordion(label=_i18n("download_model_files_from_zip"), open=False):
                            with gr.Group():
                                vbach_url_model_zip = gr.Textbox(label=_i18n("vbach_zip_link"), **base_c_params["base"])
                                vbach_url_model_zip_btn = gr.Button(_i18n("download_and_unzip"), variant="primary", **base_c_params["base"])
                                vbach_url_model_zip_status = gr.Textbox(label=_i18n("status"), value="", interactive=False, lines=3, max_lines=3)
                                @vbach_url_model_zip_btn.click(inputs=vbach_url_model_zip, outputs=vbach_url_model_zip_status)
                                def download_vbach_zip_fn(url: str, progress=gr.Progress(track_tqdm=True)):
                                    status = self.vbach_model_manager.download_model(zip_url=url)
                                    return status
                        with gr.Accordion(label=_i18n("download_model_files"), open=False):
                            with gr.Group():
                                vbach_url_model_pth = gr.Textbox(label=_i18n("vbach_pth_link"), **base_c_params["base"])
                                vbach_url_model_index = gr.Textbox(label=_i18n("vbach_index_link"), **base_c_params["base"])
                                vbach_url_model_pth_btn = gr.Button(_i18n("download_and_move_to_models_dir"), variant="primary", **base_c_params["base"])
                                vbach_url_model_pth_status = gr.Textbox(label=_i18n("status"), value="", interactive=False, lines=3, max_lines=3)
                                @vbach_url_model_pth_btn.click(inputs=[vbach_url_model_pth, vbach_url_model_index], outputs=vbach_url_model_pth_status)
                                def download_vbach_zip_fn(url_pth: str, url_index: str, progress=gr.Progress(track_tqdm=True)):
                                    status = self.vbach_model_manager.download_model(pth_url=url_pth, index_url=url_index)
                                    return status
                    with gr.Tab(_i18n("download_from_local_device")):
                        with gr.Accordion(label=_i18n("download_model_files_from_zip"), open=False):
                            with gr.Group():
                                gr.Markdown("<h3><center>"+_i18n("vbach_zip_placeholder")+"</center></h3>")
                                vbach_local_model_zip = gr.File(show_label=False, type="filepath", file_count="single", file_types=[".zip"], **base_c_params["base"])
                                @vbach_local_model_zip.upload(inputs=vbach_local_model_zip, outputs=vbach_local_model_zip)
                                def upload_vbach_zip_fn(file: str, progress=gr.Progress(track_tqdm=True)):
                                    self.vbach_model_manager.extract_zip(file)
                                    return gr.update(value=None)
                        with gr.Accordion(label=_i18n("download_model_files"), open=False):
                            with gr.Row():
                                with gr.Column():
                                    with gr.Group():
                                        gr.Markdown("<h3><center>"+_i18n("vbach_checkpoint_pth_placeholder")+"</center></h3>")
                                        vbach_local_model_pth = gr.File(show_label=False, type="filepath", file_count="multiple", file_types=[".pth"], **base_c_params["base"])
                                        @vbach_local_model_pth.upload(inputs=vbach_local_model_pth, outputs=vbach_local_model_pth)
                                        def upload_vbach_pth_fn(files: list, progress=gr.Progress(track_tqdm=True)):
                                            self.vbach_model_manager.upload_pth_model(files)
                                            return gr.update(value=[])
                                with gr.Column():
                                    with gr.Group():
                                        gr.Markdown("<h3><center>"+_i18n("vbach_index_file_placeholder")+"</center></h3>")
                                        vbach_local_model_index = gr.File(show_label=False, type="filepath", file_count="multiple", file_types=[".index"], **base_c_params["base"])
                                        @vbach_local_model_index.upload(inputs=vbach_local_model_index, outputs=vbach_local_model_index)
                                        def upload_vbach_index_fn(files: list, progress=gr.Progress(track_tqdm=True)):
                                            self.vbach_model_manager.upload_index_model(files)
                                            return gr.update(value=[])
                                        
            if GDRIVE_USER_DIR:
                with gr.Tab(_i18n("google_drive")):
                    gdrive_info = gr.Textbox(lines=3, label=_i18n("status"), interactive=False)
                    gr.Timer().tick(lambda: gr.update(value=get_disk_usage(GDRIVE_DIR)), outputs=gdrive_info)
                    copy_to_gdrive_btn = gr.Button(_i18n("copy_from_current_user_dir_to_gdrive"), **base_c_params["base"])
                    @copy_to_gdrive_btn.click()
                    def copy_to_gdrive_fn():
                        copy_to_gdrive()
                        self.input_files.update_data(0)
                        self.history.update_data(0)
                        self.auto_ensemble_history_app.update_data(0)
                        self.manual_ensemble_history_app.update_data(0)
                        self.subtract_history_app.update_data(0)
                        self.vbach_history_app.update_data(0)
                        self.iterative_ensemble_history_app.update_data(0)

        return mvsepless_app

theme = gr.themes.Base(
    primary_hue=gr.themes.Color(c100="#D5E8F2", c200="#ABD1E6", c300="#80BBD9", c400="#56A4CC", c50="#EAF4F9", c500="#2C8DC0", c600="#0276B3", c700="#025886", c800="#013B5A", c900="#001E2D", c950="#000F16"),
    secondary_hue=gr.themes.Color(c100="#CDCDD6", c200="#9A9BAD", c300="#686985", c400="#35375C", c50="#E6E6EB", c500="#020533", c600="#000637", c700="#00052E", c800="#000424", c900="#00031A", c950="#000210"),
    neutral_hue=gr.themes.Color(c100="#DBEAFE", c200="#BFDBFE", c300="#93C5FD", c400="#60A5FA", c50="#EFF6FF", c500="#3B82F6", c600="#2563EB", c700="#1D4ED8", c800="#1E40AF", c900="#1E3A8A", c950="#172554"),
    spacing_size="sm",
    radius_size="none",
    font=[gr.themes.GoogleFont('Montserrat'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
).set(
    background_fill_primary='#F7F8FA',
    background_fill_primary_dark='*secondary_950',
    background_fill_secondary='*background_fill_primary',
    background_fill_secondary_dark='*background_fill_primary',
    border_color_accent_dark='*secondary_50',
    color_accent='*primary_600',
    color_accent_soft_dark='*neutral_800',
    link_text_color='*secondary_700',
    block_background_fill_dark='*secondary_800',
    block_border_color_dark='*neutral_800',
    input_background_fill='*body_background_fill',
    input_background_fill_dark='*secondary_900',
    input_border_width='1px',
    input_border_width_dark='1px',
    button_primary_background_fill_hover='*body_background_fill',
    button_primary_background_fill_hover_dark='*neutral_400',
    button_primary_text_color_hover='*body_text_color',
    button_secondary_background_fill='*body_background_fill',
    button_secondary_background_fill_dark='*body_background_fill',
    button_secondary_background_fill_hover='*secondary_600',
    button_secondary_background_fill_hover_dark='*neutral_500',
    button_secondary_border_color_dark='*primary_500',
    button_secondary_border_color_hover='*secondary_600',
    button_secondary_text_color_dark='*neutral_600',
    button_secondary_text_color_hover='*button_primary_text_color',
    button_secondary_text_color_hover_dark='*button_primary_text_color',
    button_cancel_background_fill='red',
    button_cancel_background_fill_dark='red',
    button_cancel_background_fill_hover='white',
    button_cancel_background_fill_hover_dark='white',
    button_cancel_text_color='*button_primary_text_color',
    button_cancel_text_color_dark='*button_primary_text_color',
    button_cancel_text_color_hover='red',
    button_cancel_text_color_hover_dark='red',
    checkbox_label_background_fill_hover='*neutral_300',
    table_even_background_fill='*body_background_fill',
    table_even_background_fill_dark='*body_background_fill',
    table_odd_background_fill='*body_background_fill',
    table_odd_background_fill_dark='*body_background_fill'
)

if __name__ == "__main__":
    check_taglib_not_installed()
    args = parse_app_args()
    app = App()
    if not args.full:
        app.update_info()
        app.load_info()
    app_ui = app.UI(theme, not args.full)
    app_ui.queue(default_concurrency_limit=1)
    app_ui.launch(allowed_paths=["/"], debug=True, share=args.share, server_port=args.port, show_api=True, server_name="0.0.0.0")
