import os
import time
import tempfile
from datetime import datetime
import requests
from requests.exceptions import RequestException
from typing import Dict, List, Optional, Union
import json
import argparse
import gradio as gr

from multi_inference import MVSEPLESS
mvsepless = MVSEPLESS()

API_TOKEN = ""
algorithm_names = {}
al_by_name = {}
output_formats = ["mp3", "wav", "flac", "m4a"]

TRANSLATIONS = {
    "ru": {
        "upload_label": "Входное аудио",
        "url_label": "Введите ссылку",
        "path_label": "Введите путь к аудиофайлу",
        "url_placeholder": "Ссылка на аудиофайл",
        "path_placeholder": "/путь/к/аудио/",
        "url_btn": "Ввести URL",
        "path_btn": "Ввести путь к файлу",
        "upload_cookie": "Загрузить cookie",
        "download_audio_btn": "Скачать аудио",
        "upload_btn": "Загрузить аудио",
        "model_type": "Тип модели",
        "model_name": "Имя модели",
        "output_format": "Формат вывода",
        "separate": "Разделить",
        "error_no_input": "Ошибка: нет входного аудио.",
        "error_no_model": "Ошибка: не выбрана модель.",
        "error_invalid_format": "Ошибка: неверный формат вывода.",
        "output_zip": "Скачать ZIP",
        "inference_tab": "Инференс",
        "results": "Результаты",
        "api_token": "API ключ",
        "algo": "Тип разделения",
        "add_opt1": "Доп опция 1",
        "add_opt2": "Доп опция 2",
        "add_opt3": "Доп опция 3",
        "stem": "Стем",
        "processing": "Обработка...",
        "separation_success": "Разделение завершено",
        "separation_created": "Разделение создаётся...",
        "hash": "Хэш",
        "error": "Ошибка",
        "mvsep_api_off": "<h1><center>Плагин MVSEP API неактивен</center></h1>"

    },
    "en": {
        "upload_label": "Input audio",
        "url_label": "Enter URL",
        "path_label": "Enter path to audio",
        "url_placeholder": "Link to audio file",
        "path_placeholder": "/path/to/audio",
        "url_btn": "Input URL",
        "path_btn": "Input audio path",
        "upload_cookie": "Upload cookies",
        "download_audio_btn": "Download",
        "upload_btn": "Upload audio",
        "model_type": "Model type",
        "model_name": "Model name",
        "output_format": "Output format",
        "separate": "Separate",
        "error_no_input": "Error: No input audio.",
        "error_no_model": "Error: No model selected.",
        "error_invalid_format": "Error: Invalid output format.",
        "output_zip": "Download ZIP",
        "inference_tab": "Inference",
        "results": "Results",
        "api_token": "API Key",
        "algo": "Separation type",
        "add_opt1": "Add option 1",
        "add_opt2": "Add option 2",
        "add_opt3": "Add option 3",
        "stem": "Stem",
        "processing": "Processing...",
        "separation_success": "Separation success",
        "separation_created": "Separation creating...",
        "hash": "Hash",
        "error": "Error",
        "mvsep_api_off": "<h1><center>Plugin MVSEP API not active</center></h1>"
    }
}


CURRENT_LANG = "ru"

def set_lang(lang):
    """Функция для установки текущего языка"""
    global CURRENT_LANG
    if lang in TRANSLATIONS:
        CURRENT_LANG = lang
    else:
        raise ValueError(f"Unsupported language: {lang}")

def t(key, **kwargs):
    """Функция для получения перевода с подстановкой значений"""
    lang = CURRENT_LANG
    translation = TRANSLATIONS.get(lang, {}).get(key, key)
    return translation.format(**kwargs) if kwargs else translation

def download_wrapper(url, cookie):
    t = mvsepless.downloader_audio.dw_yt_dlp(url, cookie)
    return gr.update(value=t), gr.update(value=t), gr.update(visible=True), gr.update(visible=False)

def set_api_token(token: str):
    global API_TOKEN
    API_TOKEN = token
    gr.Warning(f"API-KEY - {token}", duration=2, title="API TEST")
    return token

class MVSEPClient:
    def __init__(self, api_key: str, retries: int = 9999, retry_interval: int = 1, debug: bool = True):
        self.api_key = api_key
        self.retries = retries
        self.retry_interval = retry_interval
        self.base_url = "https://mvsep.com/api"
        self.headers = {"User-Agent": "MVSEP Python Client/0.1"}
        self.debug = debug

    def _log_debug(self, message: str) -> None:
        """Helper method for debug logging"""
        if self.debug:
            print(f"[DEBUG] {message}")

    def _make_request(self, method: str, endpoint: str, 
                    params: Optional[Dict] = None, data: Optional[Dict] = None,
                    files: Optional[Dict] = None, stream: bool = False) -> requests.Response:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        self._log_debug(f"Making {method} request to {url}")
        self._log_debug(f"Params: {params}")
        self._log_debug(f"Data: {data}")
        if files:
            self._log_debug(f"Files: {list(files.keys())} (content not logged)")
        
        for attempt in range(self.retries + 1):
            try:
                response = requests.request(
                    method, url,
                    params=params,
                    data=data,
                    files=files,
                    headers=self.headers,
                    stream=stream,
                    timeout=(600, 1200)
                )
                
                self._log_debug(f"Response status: {response.status_code}")
                self._log_debug(f"Response headers: {dict(response.headers)}")
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", self.retry_interval))
                    self._log_debug(f"Rate limited, retrying after {retry_after}s")
                    time.sleep(retry_after)
                    continue
                if response.status_code == 400:
                    #print(response)
                    time.sleep(self.retry_interval)
                    continue
                if 500 <= response.status_code < 600 and attempt < self.retries:
                    self._log_debug(f"Server error {response.status_code}, retrying...")
                    time.sleep(self.retry_interval)
                    continue

                response.raise_for_status()
                return response

            except requests.exceptions.HTTPError as e:
                self._log_debug(f"HTTP error: {str(e)}")
                if e.response.status_code // 100 == 4 and e.response.status_code != 429:
                    raise
                if attempt == self.retries:
                    raise
                time.sleep(self.retry_interval)
            except RequestException as e:
                self._log_debug(f"Request exception: {str(e)}")
                if attempt == self.retries:
                    raise Exception(f"Request failed after {self.retries} retries: {str(e)}")
                time.sleep(self.retry_interval)
        raise Exception("Unexpected error in request handling")

    # Core Separation Functions (updated with debug logs)
    def create_separation(self, file_path: Optional[str] = None, url: Optional[str] = None,
                        sep_type: int = 11, add_opt1: Optional[Union[str, int]] = None,
                        add_opt2: Optional[Union[str, int]] = None, add_opt3: Optional[Union[str, int]] = None,
                        output_format: int = 0, is_demo: bool = False,
                        remote_type: Optional[str] = None) -> Dict:
        self._log_debug(f"Creating separation with params: sep_type={sep_type}, output_format={output_format}")
        
        data = {
            "api_token": self.api_key,
            "sep_type": str(sep_type),
            "output_format": str(output_format),
            "is_demo": "1" if is_demo else "0"
        }
        files = {}
        
        if file_path and url:
            raise ValueError("Cannot specify both file_path and url")
        if file_path:
            self._log_debug(f"Uploading local file: {file_path}")
            files["audiofile"] = open(file_path, "rb")
        elif url:
            self._log_debug(f"Processing remote URL: {url}")
            data["url"] = url
            if remote_type:
                data["remote_type"] = remote_type
        else:
            raise ValueError("Either file_path or url must be provided")
        
        for opt, val in [("add_opt1", add_opt1), ("add_opt2", add_opt2), ("add_opt3", add_opt3)]:
            if val is not None:
                data[opt] = str(val)
        
        response = self._make_request("POST", "separation/create", data=data, files=files)
        json_response = response.json()
        self._log_debug(f"Create separation response: {json_response}")
        return json_response

    def get_separation_status(self, task_hash: str, mirror: int = 0) -> Dict:
        self._log_debug(f"Getting status for hash: {task_hash}, mirror={mirror}")
        params = {"hash": task_hash, "mirror": str(mirror)}
        if mirror == 1:
            params["api_token"] = self.api_key
        response = self._make_request("GET", "separation/get", params=params)
        json_response = response.json()
        self._log_debug(f"Status response: {json_response}")
        return json_response

    def download_track(self, url: str, output_path: str) -> None:
        """Download a track directly using the full URL from the API response"""
        self._log_debug(f"Downloading track directly from {url}")
        
        # Bypass the base URL since we have full download URLs
        response = requests.get(url, stream=True, headers=self.headers)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        self._log_debug(f"Finished downloading to {output_path}")


    # Updated process_directory with debug logs
    def process_file(self, input_file: str, output_dir: str, progress: any = gr.Progress(), **kwargs) -> None:
        self._log_debug(f"Processing file: {input_file} -> {output_dir}")
        supported_ext = [".mp3", ".wav", ".flac", ".m4a", ".mp4", ".ogg", ".opus", ".aiff"]
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.basename(input_file)

        if os.path.splitext(filename)[1].lower() not in supported_ext:
            self._log_debug(f"Skipping unsupported file: {filename}")
            return
        
        file_path = input_file
        self._log_debug(f"Processing {filename}")
        
        try:
            create_resp = self.create_separation(file_path=file_path, **kwargs)
            if not create_resp.get("success"):
                self._log_debug(f"Creation failed response: {create_resp}")
                return
            
            task_hash = create_resp["data"]["hash"]
            self._log_debug(f"Created separation task: {task_hash}")
            gr.Warning(title=t("separation_created"), message=f"{t('hash')}: {task_hash}")
            
            while True:
                status_resp = self.get_separation_status(task_hash)
                self._log_debug(f"Status poll response: {status_resp}")
                
                status = status_resp.get("status")
                if status == "done":
                    self._log_debug("Processing completed successfully")
                    progress(0.9, desc=t("separation_success"))
                    break
                if status in ["failed", "error"]:
                    self._log_debug("Processing failed")
                    break
                if status in ["waiting", "processing", "distributing", "merging"]:
                    self._log_debug(f"Current status: {status}, waiting {self.retry_interval}s")
                    if status == "waiting":
                        progress(0.2, desc=f'{status_resp["data"]["current_order"]} | {status_resp["data"]["queue_count"]}')
                    if status == "processing":
                        progress(0.5, desc=t("processing"))
                    time.sleep(self.retry_interval)
                else:
                    self._log_debug(f"Unknown status: {status}")
                    break
            
            if status != "done":
                pass
            
            # FIXED: Use 'download' key instead of 'name'
            for file_info in status_resp["data"]["files"]:
                output_filename = file_info.get("download", f"unknown_{time.time()}.mp3")
                output_path = os.path.join(output_dir, output_filename)
                self._log_debug(f"Downloading {output_filename}")
                # FIXED: Use 'url' key instead of 'link'
                self.download_track(file_info["url"], output_path)
        
        except Exception as e:
            self._log_debug(f"Exception during processing: {str(e)}")
            gr.Error(title=t("error"), message=e)
            print(f"Error processing {filename}: {str(e)}")

    # Updated get_algorithms with debug logs
    def get_algorithms(self) -> Dict:
        self._log_debug("Fetching algorithm list")
        response = self._make_request("GET", "app/algorithms")
        sorted_algos = sorted(response.json(), key=lambda algo: algo['render_id'])
        algo_dict = {}

        for algo in sorted_algos:
            s1 = f"\nID:{algo['render_id']} - {algo['name']}"
            algo_dict[algo['render_id']] = s1 + '\n'
            # print(s1)
            for field in algo['algorithm_fields']:
                s1 = f"\t{field['name']}"
                algo_dict[algo['render_id']] += s1 + '\n'
                # print(s1)
                options = json.loads(field['options'])
                for key, value in sorted(options.items()):
                    s1 = f"\t\t{key}: {value}"
                    algo_dict[algo['render_id']] += s1 + '\n'
                    # print(s1)
        return algo_dict

    # Premium Management
    def enable_premium(self) -> Dict:
        data = {"api_token": self.api_key}
        response = self._make_request("POST", "app/enable_premium", data=data)
        return response.json()

    def disable_premium(self) -> Dict:
        data = {"api_token": self.api_key}
        response = self._make_request("POST", "app/disable_premium", data=data)
        return response.json()

    # Additional API Endpoints
    def get_queue_info(self) -> Dict:
        response = self._make_request("GET", "app/queue")
        return response.json()

    def get_separation_history(self, start: int = 0, limit: int = 10) -> Dict:
        params = {"api_token": self.api_key, "start": start, "limit": limit}
        response = self._make_request("GET", "app/separation_history", params=params)
        return response.json()

    def enable_long_filenames(self) -> Dict:
        data = {"api_token": self.api_key}
        response = self._make_request("POST", "app/enable_long_filenames", data=data)
        return response.json()

    def disable_long_filenames(self) -> Dict:
        data = {"api_token": self.api_key}
        response = self._make_request("POST", "app/disable_long_filenames", data=data)
        return response.json()

def mvsep_api(i: str, o: str, of: str, st: int, ao1: int, ao2: int, ao3: int, token: str, progress: any = gr.Progress()):

    # Example Usage
    API_KEY = token
    client = MVSEPClient(api_key=API_KEY, debug=True)  # USE DEBUG, ELSE NOTHING WILL BE PRINTED ON TERMINAL, normal prints are not done yet

    algos = client.get_algorithms()
    print('Separate with algorithm: {}'.format(st))
    print(algos[st])

    if of == "mp3":
        of_bool = 0
    elif of == "wav":
        of_bool = 1
    elif of == "flac":
        of_bool = 2
    elif of == "m4a":
        of_bool = 3
    else:
        of_bool == 1



    # Process directory example / need to check if retries are working correctly !!!
    client.process_file(
        input_file = i,
        output_dir = o,
        progress = progress,
        output_format = of_bool,  # MP3=0, WAV=1, FLAC=2, M4A=3
        sep_type = st, # use client.get_algorithms() or check documentation details https://mvsep.com/en/full_api for now
        add_opt1 = ao1, # use client.get_algorithms() or check documentation details https://mvsep.com/en/full_api for now
        add_opt2 = ao2, # use client.get_algorithms() or check documentation details https://mvsep.com/en/full_api for now
        add_opt3 = ao3, # use client.get_algorithms() or check documentation details https://mvsep.com/en/full_api for now
    )

    output_files = []

    if os.listdir(o):
        for file in os.listdir(o):
            if os.path.exists(os.path.join(o, file)):
                output_files.append(os.path.join(o, file))
        return output_files
    else:
        return []

def write_dict_algos(algos: dict):
    dicts_dir = os.path.join(os.getcwd(), "algos")
    os.makedirs(dicts_dir, exist_ok=True)
    dict_filename = f"algos_{datetime.now().strftime('%y%m%d_%H%M%S')}.json"
    full_path_dict = os.path.join(dicts_dir, dict_filename)
    with open(full_path_dict, "w") as f:
        json.dump(algos, f)

def get_algos(token: str, names: bool = False):

    def parse_add_opts(algorithm_id: int, algorithms_dict: dict) -> dict:
        """
        Parses the add_opt1, add_opt2, and add_opt3 and name from the algorithm details.

        Args:
            algorithm_id: The ID of the algorithm.
            algorithms_dict: The dictionary containing algorithm details.

        Returns:
            A dictionary with parsed add_opt options and the algorithm name.
        """
        if algorithm_id not in algorithms_dict:
            return {"error": f"Algorithm with ID {algorithm_id} not found."}

        algo_details = algorithms_dict[algorithm_id]

        add_opts = {}
        current_opt = None
        algorithm_name = None

        lines = algo_details.splitlines()

        if lines:
            first_line = lines[1].strip()
            if first_line.startswith("ID:"):
                parts = first_line.split(" - ", 1)
                if len(parts) > 1:
                    algorithm_name = parts[1]
                else:
                    pass
            else:
                pass


        for line in lines:
            line = line.strip()
            if line.startswith("add_opt"):
                current_opt = line
                add_opts[current_opt] = {}
            elif current_opt and line and not line.startswith("ID:"):
                try:
                    key, value = line.split(":", 1)
                    add_opts[current_opt][key.strip()] = value.strip()
                except ValueError:
                    pass

        result = {"name": algorithm_name}
        result.update(add_opts)
        return result

    client = MVSEPClient(api_key=token, debug=True)
    algos = client.get_algorithms()

    # write_dict_algos(algos=algos)

    full_algos_dict = {}

    for algo in algos:
       full_algos_dict[algo] = parse_add_opts(algo, algos)

    if names == True:

        al_by_name = {}

        for algo_id, algo_details in full_algos_dict.items():
            if "name" in algo_details and algo_details["name"] is not None:
                algo_name = algo_details["name"]
                al_by_name[algo_name] = algo_details
                al_by_name[algo_name]["id"] = algo_id

                for opt_key in ["add_opt1", "add_opt2", "add_opt3"]:
                    if opt_key in al_by_name[algo_name]:
                        reversed_add_opt = {v: k for k, v in al_by_name[algo_name][opt_key].items()}
                        al_by_name[algo_name][opt_key] = reversed_add_opt

        return al_by_name

    return full_algos_dict

def process_audio(audio_file, output_format, algorithm_name, add_opt1_value=None, add_opt2_value=None, add_opt3_value=None, progress=gr.Progress()):
    """
    Processes an audio file using the MVSEP API based on the selected algorithm and options.

    Args:
        audio_file: The uploaded audio file path.
        algorithm_name: The name of the selected algorithm.
        add_opt1_value: The selected value for add_opt1.
        add_opt2_value: The selected value for add_opt2.
        add_opt3_value: The selected value for add_opt3.

    Returns:
        A list of paths to the separated audio files.
    """
    global al_by_name
    global API_TOKEN

    if algorithm_name not in al_by_name:
        return f"Error: Algorithm '{algorithm_name}' not found."

    algorithm_info = al_by_name[algorithm_name]
    algorithm_id = algorithm_info.get("id")

    if algorithm_id is None:
         return f"Error: Algorithm '{algorithm_name}' does not have an ID."

    add_opt1_int = -1
    if add_opt1_value and "add_opt1" in algorithm_info:
        add_opt1_int = algorithm_info["add_opt1"].get(add_opt1_value, -1)

    add_opt2_int = -1
    if add_opt2_value and "add_opt2" in algorithm_info:
        add_opt2_int = algorithm_info["add_opt2"].get(add_opt2_value, -1)

    add_opt3_int = -1
    if add_opt3_value and "add_opt3" in algorithm_info:
        add_opt3_int = algorithm_info["add_opt3"].get(add_opt3_value, -1)

    temp_dir = tempfile.mkdtemp()

    output_files = mvsep_api(
        i=audio_file,
        o=temp_dir,
        of=output_format,
        st=algorithm_id,
        ao1=add_opt1_int,
        ao2=add_opt2_int,
        ao3=add_opt3_int,
        token=API_TOKEN,
        progress=progress
    )

    audio_updates = [
        gr.update(
            label=os.path.basename(output_files[i]) if i < len(output_files) else None,
            value=output_files[i] if i < len(output_files) else None,
            visible=i < len(output_files)
        ) 
        for i in range(64)
    ]

    return tuple(audio_updates)

def update_add_opts(algorithm_name):
    if algorithm_name in al_by_name:
        algorithm_info = al_by_name[algorithm_name]
        add_opt1_choices = list(algorithm_info.get("add_opt1", {}).keys())
        add_opt2_choices = list(algorithm_info.get("add_opt2", {}).keys())
        add_opt3_choices = list(algorithm_info.get("add_opt3", {}).keys())

        return gr.update(choices=add_opt1_choices, interactive=True, value=add_opt1_choices[0] if add_opt1_choices else None, visible=bool(add_opt1_choices)), \
               gr.update(choices=add_opt2_choices, interactive=True, value=add_opt2_choices[0] if add_opt2_choices else None, visible=bool(add_opt2_choices)), \
               gr.update(choices=add_opt3_choices, interactive=True, value=add_opt3_choices[0] if add_opt3_choices else None, visible=bool(add_opt3_choices))
    else:
        return gr.update(choices=[], interactive=False, value=None, visible=False), \
               gr.update(choices=[], interactive=False, value=None, visible=False), \
               gr.update(choices=[], interactive=False, value=None, visible=False)

token = set_api_token(token="")


MVSEP_API = os.environ.get("MVSEP_API_PLUGIN", False)

if MVSEP_API == "True":
    algos_test = get_algos(token=token, names=True)
    algorithm_names = list(algos_test.keys())
    al_by_name = algos_test

    def plugin_name():
        return "MVSEP API"

    def plugin(lang):
        set_lang(lang)
        with gr.Row():
            with gr.Column():
                with gr.Group() as local:
                    input_audio = gr.Audio(label=t("upload_label"), type="filepath", interactive=True)
                    with gr.Row(equal_height=True):
                        path_0_btn = gr.Button(t("path_btn"))
                        url_0_btn = gr.Button(t("url_btn"))
                with gr.Group(visible=False) as url:
                    with gr.Column(variant="compact"):
                        with gr.Row(equal_height=True):
                            upload_cookie = gr.UploadButton(label=t("upload_cookie"), file_types=[".txt"], file_count="single", scale=1, variant="primary")
                            input_link = gr.Textbox(label=t("url_label"), placeholder=t("url_placeholder"), interactive=True, scale=10)
                            download_audio_btn = gr.Button(t("download_audio_btn"), scale=1, variant="stop")
                    with gr.Row(equal_height=True):
                        path_1_btn = gr.Button(t("path_btn"))
                        upload_0_btn = gr.Button(t("upload_btn"), variant="primary")
                with gr.Group(visible=False) as path:
                    input_path = gr.Textbox(label=t("path_label"), placeholder=t("path_placeholder"), interactive=True)
                    with gr.Row(equal_height=True):
                        upload_1_btn = gr.Button(t("upload_btn"), variant="primary")
                        url_1_btn = gr.Button(t("url_btn"))

            with gr.Column():
                api_token = gr.Textbox(label=t("api_token"), value=API_TOKEN, type="password", interactive=True)
                
                algorithm_dropdown = gr.Dropdown(choices=algorithm_names, label=t("algo"))

                add_opt1_dropdown = gr.Dropdown(label=t("add_opt1"), interactive=True)
                add_opt2_dropdown = gr.Dropdown(label=t("add_opt2"), interactive=True)
                add_opt3_dropdown = gr.Dropdown(label=t("add_opt3"), interactive=True)

                o_format = gr.Radio(choices=output_formats, label=t("output_format"), value="mp3")

                process_button = gr.Button(t("separate"))
        with gr.Group():
            output_stems = []

            for i in range(64):
                audio = gr.Audio(label=t("stem"), type="filepath", interactive=False, show_download_button=True, visible=False, scale=4)
                output_stems.append(audio)

        input_audio.change(
            lambda x: gr.update(value=x),
            inputs=input_audio,
            outputs=input_path
        )

        path_0_btn.click(            
            lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[local, path]
        )

        path_1_btn.click(            
            lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[url, path]
        )

        url_0_btn.click(
            lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[local, url]
        )

        url_1_btn.click(
            lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[path, url]
        )

        upload_0_btn.click(
            lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[local, url]
        )

        upload_1_btn.click(
            lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[local, path]
        )

        download_audio_btn.click(
            download_wrapper,
            inputs=[input_link, upload_cookie],
            outputs=[input_audio, input_path, local, url],
            show_progress=True
        )

        algorithm_dropdown.change(
            fn=update_add_opts,
            inputs=algorithm_dropdown,
            outputs=[add_opt1_dropdown, add_opt2_dropdown, add_opt3_dropdown]
        )

        process_button.click(
            fn=process_audio,
            inputs=[
                input_path,
                o_format,
                algorithm_dropdown,
                add_opt1_dropdown,
                add_opt2_dropdown,
                add_opt3_dropdown
            ],
            outputs=[*output_stems],
            show_progress_on=[input_path, input_audio]
        )
        
        api_token.change(
            set_api_token,
            inputs=api_token,
            outputs=gr.State()
        )

        gr.on(
            fn=update_add_opts,
            inputs=algorithm_dropdown,
            outputs=[add_opt1_dropdown, add_opt2_dropdown, add_opt3_dropdown]
        )

else:
    def plugin_name():
        return "MVSEP API (OFF)"

    def plugin(lang):
        set_lang(lang)
        gr.Markdown(t("mvsep_api_off"))

