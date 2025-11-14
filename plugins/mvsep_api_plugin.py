import os
import time
import tempfile
import re
from datetime import datetime
import requests
from requests.exceptions import RequestException
from typing import Dict, List, Optional, Union
import json
import argparse
import gradio as gr
import yt_dlp
import urllib.request

API_TOKEN = ""
algorithm_names = {}
al_by_name = {}
output_formats = ["mp3", "wav", "flac", "m4a"]

MAX_LENGTH_NAME = 255

def clean_filename(filename, length=240):
    # Список символов, запрещенных в обеих системах
    universal_forbidden = r"\\/*?:<>|"

    # Дополнительные символы, запрещенные в Linux
    linux_forbidden = r"&;~\'`()[]$#^%!"

    # Создаем набор всех запрещенных символов
    forbidden_chars = set(universal_forbidden + linux_forbidden)

    # Удаляем запрещенные символы
    cleaned = "".join(c for c in filename if c not in forbidden_chars)

    # Удаляем пробелы в начале и конце
    cleaned = cleaned.strip()

    # Проверяем на зарезервированные имена Windows
    reserved_windows = {
        "CON",
        "AUX",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "LPT1",
        "LPT2",
        "LPT3",
        "PRN",
        "NUL",
    }

    # Если имя файла зарезервировано, добавляем префикс
    if cleaned.upper() in reserved_windows:
        cleaned = f"file_{cleaned}"
    if len(cleaned) > length:
        return f"{cleaned[:length // 2]}...{cleaned[-(length // 3):]}"
    return cleaned

def remove_duplicate_keys(input_str, keys=("NAME", "STEM", "MODEL")):
    # Создаем множество для отслеживания найденных ключей
    seen = set()
    # Шаблон для поиска любого из ключей
    pattern = r"({})".format("|".join(re.escape(key) for key in keys))

    def replace(match):
        key = match.group(1)
        if key in seen:
            return ""  # Удаляем дубликат
        seen.add(key)
        return key  # Оставляем первое вхождение

    # Заменяем дубликаты на пустую строку
    result = re.sub(pattern, replace, input_str)
    return result


def shorter_name(template, file_name, stem, model):
    # Удаляем дубликаты ключей в шаблоне перед расчетами
    clean_template = remove_duplicate_keys(template)

    template_no_keys_length = len(
        clean_template.replace("NAME", "")
        .replace("STEM", "")
        .replace("MODEL", "")
    )
    key_values_length = (len(stem)
            if "STEM" in clean_template
            else 0 + len(model) if "MODEL" in clean_template else 0
    )
    free_length = MAX_LENGTH_NAME - (template_no_keys_length + key_values_length)
    if len(file_name) > (free_length - 7):
        shorted_name = f"{file_name[:(free_length // 2)]}...{file_name[-((free_length // 2) - 7):]}"
        return shorted_name
    else:
        return file_name

def output_file_template(template, input_file_name, stem, model):
    # Удаляем дубликаты ключей перед заменой
    clean_template = remove_duplicate_keys(template)

    input_file_name = shorter_name(
        clean_template, input_file_name, stem, model
    )
    template_name = (
        clean_template.replace("STEM", f"{stem}")
        .replace("MODEL", f"{model}")
        .replace("NAME", f"{input_file_name}")
    )
    output_name = f"{template_name}"
    return output_name

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
        "mvsep_api_off": "<h1><center>Плагин MVSEP API неактивен</center></h1>",
        "template": "Формат имени",
        "current_order": "Ваше задание",
        "queue_count": "Файлы в очереди",
        "reuse": "Использовать снова"
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
        "mvsep_api_off": "<h1><center>Plugin MVSEP API not active</center></h1>",
        "template": "Name format",
        "current_order": "Your order",
        "queue_count": "Files in Queue",
        "reuse": "Reuse"
    },
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
    dw = Downloader()
    t = dw.dw_yt_dlp(url, cookie)
    return (
        gr.update(value=t),
        gr.update(value=t),
        gr.update(visible=True),
        gr.update(visible=False),
    )

def set_api_token(token: str):
    global API_TOKEN
    API_TOKEN = token
    gr.Warning(f"API-KEY - {token}", duration=2, title="API TEST")
    return token

class Downloader:
    def __init__(self, output_dir=os.environ.get(
            "MVSEPLESS_DOWNLOAD_DIR", os.path.join(os.getcwd(), "downloaded")
        )):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def dw_yt_dlp(
        self,
        url,
        cookie=None,
        output_format="mp3",
        output_bitrate="320",
        title=None,
    ):
        # Подготовка шаблона имени файла
        outtmpl = "%(title)s.%(ext)s" if title is None else f"{title}.%(ext)s"

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(self.output_dir, outtmpl),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": output_format,
                    "preferredquality": output_bitrate,
                }
            ],
            "noplaylist": True,  # Скачивать только одно видео, не плейлист
            "quiet": True,  # Отключить вывод в консоль
            "no_warnings": True,  # Скрыть предупреждения
        }

        # Добавляем cookies если указаны
        if cookie and os.path.exists(cookie):
            ydl_opts["cookiefile"] = cookie

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
                if "_type" in info and info["_type"] == "playlist":
                    # Для плейлистов берем первое видео
                    entry = info["entries"][0]
                    filename = ydl.prepare_filename(entry)
                else:
                    # Для одиночного видео
                    filename = ydl.prepare_filename(info)

                # Заменяем оригинальное расширение на выбранный формат
                base, _ = os.path.splitext(filename)
                audio_file = base + f".{output_format}"

                return os.path.join(self.output_dir, audio_file)
            except Exception as e:
                print(e)
                gr.Warning(e)
                return url

    def dw_from_url(self, url, title=None):
        try:
            response = urllib.request.urlopen(url)
            content_type = response.info().get_content_type()
            if "audio" in content_type:
                filename = os.path.join(
                    self.output_dir, title or "downloaded_audio"
                )
                with open(filename, "wb") as f:
                    f.write(response.read())
                return filename
            else:
                raise ValueError("URL does not point to an audio file.")
        except Exception as e:
            print(e)
            gr.Warning(e)
            return url

class MVSEPClient:
    def __init__(
        self,
        api_key: str,
        retries: int = 999999999,
        retry_interval: int = 5,
        debug: bool = True,
    ):
        self.api_key = api_key
        self.retries = retries
        self.retry_interval = retry_interval
        self.base_url = "https://mvsep.com/api"
        self.headers = {"User-Agent": "MVSEP Python Client for MVSEPLESS"}
        self.debug = debug

    def parse_model_from_output_filename(self, task_hash, stem, filename):

        escaped_task_hash = re.escape(task_hash)
        escaped_stem = re.escape(stem)

        pattern = rf"^{escaped_task_hash}_([^_]+_.+?)_{escaped_stem}$"

        match = re.match(pattern, filename)

        if match:
            return match.group(1)
        else:
            return ""

    def _log_debug(self, message: str) -> None:
        """Helper method for debug logging"""
        if self.debug:
            print(f"[DEBUG] {message}")

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        files: Optional[Dict] = None,
        stream: bool = False,
    ) -> requests.Response:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        self._log_debug(f"Making {method} request to {url}")
        self._log_debug(f"Params: {params}")
        self._log_debug(f"Data: {data}")
        if files:
            self._log_debug(f"Files: {list(files.keys())} (content not logged)")

        for attempt in range(self.retries + 1):
            try:
                response = requests.request(
                    method,
                    url,
                    params=params,
                    data=data,
                    files=files,
                    headers=self.headers,
                    stream=stream,
                    timeout=(600, 1200),
                )

                self._log_debug(f"Response status: {response.status_code}")
                self._log_debug(f"Response headers: {dict(response.headers)}")

                if response.status_code == 429:
                    retry_after = int(
                        response.headers.get("Retry-After", self.retry_interval)
                    )
                    self._log_debug(f"Rate limited, retrying after {retry_after}s")
                    time.sleep(retry_after)
                    continue
                if response.status_code == 400:
                    # print(response)
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
                    raise Exception(
                        f"Request failed after {self.retries} retries: {str(e)}"
                    )
                time.sleep(self.retry_interval)
        raise Exception("Unexpected error in request handling")

    # Core Separation Functions (updated with debug logs)
    def create_separation(
        self,
        file_path: Optional[str] = None,
        url: Optional[str] = None,
        sep_type: int = 11,
        add_opt1: Optional[Union[str, int]] = None,
        add_opt2: Optional[Union[str, int]] = None,
        add_opt3: Optional[Union[str, int]] = None,
        output_format: int = 0,
        is_demo: bool = False,
        remote_type: Optional[str] = None,
    ) -> Dict:
        self._log_debug(
            f"Creating separation with params: sep_type={sep_type}, output_format={output_format}"
        )

        data = {
            "api_token": self.api_key,
            "sep_type": str(sep_type),
            "output_format": str(output_format),
            "is_demo": "1" if is_demo else "0",
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

        for opt, val in [
            ("add_opt1", add_opt1),
            ("add_opt2", add_opt2),
            ("add_opt3", add_opt3),
        ]:
            if val is not None:
                data[opt] = str(val)

        response = self._make_request(
            "POST", "separation/create", data=data, files=files
        )
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
    def process_file(
        self, input_file: str, output_dir: str, template: str = "MODEL - NAME - STEM", progress: any = gr.Progress(), **kwargs
    ) -> None:
        self._log_debug(f"Processing file: {input_file} -> {output_dir}")
        supported_ext = [
            ".mp3",
            ".wav",
            ".flac",
            ".m4a",
            ".mp4",
            ".ogg",
            ".opus",
            ".aiff",
        ]
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.basename(input_file)

        basename, _ = os.path.splitext(filename)

        cleaned_basename = basename[-(160 / 3):(160 / 3)] if len(basename) > 160 else basename

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
            gr.Warning(
                title=t("separation_created"), message=f"{t('hash')}: {task_hash}"
            )

            while True:
                status_resp = self.get_separation_status(task_hash)
                self._log_debug(f"Status poll response: {status_resp}")

                status = status_resp.get("status")
                if status == "done":
                    self._log_debug("Processing completed successfully")
                    progress(0.9, desc=t("separation_success"))
                    gr.Warning(message="", title=t("separation_success"))
                    break
                if status in ["failed", "error"]:
                    self._log_debug("Processing failed")
                    break
                if status in ["waiting", "processing", "distributing", "merging"]:
                    self._log_debug(
                        f"Current status: {status}, waiting {self.retry_interval}s"
                    )
                    if status == "waiting":
                        progress(
                            0.2,
                            desc=f'{status_resp["data"]["current_order"]} | {status_resp["data"]["queue_count"]}',
                        )
                        gr.Warning(message=f'{t("current_order")}:{status_resp["data"]["current_order"]} \n {t("queue_count")}:{status_resp["data"]["queue_count"]}', title="", duration=self.retry_interval)
                    if status == "processing":
                        progress(0.5, desc=t("processing"))
                        gr.Warning(title=t("processing"), message="")
                    time.sleep(self.retry_interval)
                else:
                    self._log_debug(f"Unknown status: {status}")
                    break

            if status != "done":
                pass

            output_audios = {"algorithm": None, "stems": []}

            output_audios["algorithm"] = status_resp["data"]["algorithm"]

            for file_info in status_resp["data"]["files"]:
                stem = file_info["type"]
                download = file_info.get(
                    "download", f"unknown_{time.time()}.mp3"
                )
                basename_from_task_hash = os.path.splitext(task_hash)[0].split('-', 2)[-1]
                model_name = self.parse_model_from_output_filename(basename_from_task_hash, stem.lower(), os.path.splitext(download)[0])
                output_filename = output_file_template(template, cleaned_basename, stem, model_name)
                output_path = os.path.join(output_dir, f"{output_filename}{os.path.splitext(download)[1]}")
                self._log_debug(f"Downloading {output_filename}")
                self.download_track(file_info["url"], output_path)
                output_audios["stems"].append((stem, output_path))

            return output_audios

        except Exception as e:
            self._log_debug(f"Exception during processing: {str(e)}")
            gr.Error(title=t("error"), message=e)
            print(f"Error processing {filename}: {str(e)}")

    # Updated get_algorithms with debug logs
    def get_algorithms(self) -> Dict:
        self._log_debug("Fetching algorithm list")
        response = self._make_request("GET", "app/algorithms")
        sorted_algos = sorted(response.json(), key=lambda algo: algo["render_id"])
        algo_dict = {}

        for algo in sorted_algos:
            s1 = f"\nID:{algo['render_id']} - {algo['name']}"
            algo_dict[algo["render_id"]] = s1 + "\n"
            # print(s1)
            for field in algo["algorithm_fields"]:
                s1 = f"\t{field['name']}"
                algo_dict[algo["render_id"]] += s1 + "\n"
                # print(s1)
                options = json.loads(field["options"])
                for key, value in sorted(options.items()):
                    s1 = f"\t\t{key}: {value}"
                    algo_dict[algo["render_id"]] += s1 + "\n"
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


def mvsep_api(
    i: str,
    o: str,
    of: str,
    st: int,
    ao1: int,
    ao2: int,
    ao3: int,
    token: str,
    template: str = "MODEL - NAME - STEM",
    progress: any = gr.Progress(),
):

    # Example Usage
    API_KEY = token
    client = MVSEPClient(
        api_key=API_KEY, debug=True
    )  # USE DEBUG, ELSE NOTHING WILL BE PRINTED ON TERMINAL, normal prints are not done yet

    algos = client.get_algorithms()
    print("Разделение с алгоритмом: {}".format(st))
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
    output = client.process_file(
        input_file=i,
        output_dir=o,
        template=template,
        progress=progress,
        output_format=of_bool,  # MP3=0, WAV=1, FLAC=2, M4A=3
        sep_type=st,  # use client.get_algorithms() or check documentation details https://mvsep.com/en/full_api for now
        add_opt1=ao1,  # use client.get_algorithms() or check documentation details https://mvsep.com/en/full_api for now
        add_opt2=ao2,  # use client.get_algorithms() or check documentation details https://mvsep.com/en/full_api for now
        add_opt3=ao3,  # use client.get_algorithms() or check documentation details https://mvsep.com/en/full_api for now
    )

    output_files = output["stems"]
    return output_files


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
                        reversed_add_opt = {
                            v: k for k, v in al_by_name[algo_name][opt_key].items()
                        }
                        al_by_name[algo_name][opt_key] = reversed_add_opt

        return al_by_name

    return full_algos_dict


def update_add_opts(algorithm_name):
    if algorithm_name in al_by_name:
        algorithm_info = al_by_name[algorithm_name]
        add_opt1_choices = list(algorithm_info.get("add_opt1", {}).keys())
        add_opt2_choices = list(algorithm_info.get("add_opt2", {}).keys())
        add_opt3_choices = list(algorithm_info.get("add_opt3", {}).keys())

        return (
            gr.update(
                choices=add_opt1_choices,
                interactive=True,
                value=add_opt1_choices[0] if add_opt1_choices else None,
                visible=bool(add_opt1_choices),
            ),
            gr.update(
                choices=add_opt2_choices,
                interactive=True,
                value=add_opt2_choices[0] if add_opt2_choices else None,
                visible=bool(add_opt2_choices),
            ),
            gr.update(
                choices=add_opt3_choices,
                interactive=True,
                value=add_opt3_choices[0] if add_opt3_choices else None,
                visible=bool(add_opt3_choices),
            ),
        )
    else:
        return (
            gr.update(choices=[], interactive=False, value=None, visible=False),
            gr.update(choices=[], interactive=False, value=None, visible=False),
            gr.update(choices=[], interactive=False, value=None, visible=False),
        )

online = os.environ.get("MVSEP_API_OFF", True)

if online == True:
    env_token = os.environ.get("MVSEP_API_TOKEN", None)

    token = set_api_token(token=env_token if env_token else "")

    algos_test = get_algos(token=token, names=True)
    algorithm_names = list(algos_test.keys())
    al_by_name = algos_test

    def plugin_name():
        return "MVSEP API"

    def plugin(lang):
        set_lang(lang)
        with gr.Row():
            with gr.Column():
                with gr.Group() as mvsep_api_ui_local:
                    mvsep_api_ui_input_audio = gr.Audio(
                        label=t("upload_label"), type="filepath", interactive=True
                    )
                    with gr.Row(equal_height=True):
                        mvsep_api_ui_path_0_btn = gr.Button(t("path_btn"))
                        mvsep_api_ui_url_0_btn = gr.Button(t("url_btn"))
                with gr.Group(visible=False) as mvsep_api_ui_url:
                    with gr.Column(variant="compact"):
                        with gr.Row(equal_height=True):
                            mvsep_api_ui_upload_cookie = gr.UploadButton(
                                label=t("upload_cookie"),
                                file_types=[".txt"],
                                file_count="single",
                                scale=1,
                                variant="primary",
                            )
                            mvsep_api_ui_input_link = gr.Textbox(
                                label=t("url_label"),
                                placeholder=t("url_placeholder"),
                                interactive=True,
                                scale=10,
                            )
                            mvsep_api_ui_download_audio_btn = gr.Button(
                                t("download_audio_btn"), scale=1, variant="stop"
                            )
                    with gr.Row(equal_height=True):
                        mvsep_api_ui_path_1_btn = gr.Button(t("path_btn"))
                        mvsep_api_ui_upload_0_btn = gr.Button(t("upload_btn"), variant="primary")
                with gr.Group(visible=False) as mvsep_api_ui_path:
                    mvsep_api_ui_input_path = gr.Textbox(
                        label=t("path_label"),
                        placeholder=t("path_placeholder"),
                        interactive=True,
                    )
                    with gr.Row(equal_height=True):
                        mvsep_api_ui_upload_1_btn = gr.Button(t("upload_btn"), variant="primary")
                        mvsep_api_ui_url_1_btn = gr.Button(t("url_btn"))

            with gr.Column():
                mvsep_api_ui_api_token = gr.Textbox(
                    label=t("api_token"),
                    value=API_TOKEN,
                    type="password",
                    interactive=True,
                )

                mvsep_api_ui_algorithm_dropdown = gr.Dropdown(
                    choices=algorithm_names, label=t("algo")
                )

                mvsep_api_ui_add_opt1_dropdown = gr.Dropdown(label=t("add_opt1"), interactive=True)
                mvsep_api_ui_add_opt2_dropdown = gr.Dropdown(label=t("add_opt2"), interactive=True)
                mvsep_api_ui_add_opt3_dropdown = gr.Dropdown(label=t("add_opt3"), interactive=True)

                mvsep_api_ui_template = gr.Textbox(label=t("template"), value="NAME_MODEL_STEM", interactive=True)

                mvsep_api_ui_o_format = gr.Radio(
                    choices=output_formats, label=t("output_format"), value="mp3"
                )

                mvsep_api_ui_process_button = gr.Button(t("separate"))
        with gr.Group():
            @gr.render(inputs=[
                mvsep_api_ui_input_path,
                mvsep_api_ui_o_format,
                mvsep_api_ui_algorithm_dropdown,
                mvsep_api_ui_add_opt1_dropdown,
                mvsep_api_ui_add_opt2_dropdown,
                mvsep_api_ui_add_opt3_dropdown,
                mvsep_api_ui_template
            ], triggers=[mvsep_api_ui_process_button.click])
            def process_audio(
                audio_file,
                output_format,
                algorithm_name,
                add_opt1_value=None,
                add_opt2_value=None,
                add_opt3_value=None,
                template="MODEL - NAME - STEM",
                progress=gr.Progress(),
            ):
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
                template = clean_filename(template, length=40)
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
                    template=template,
                    progress=progress,
                )

                if output_files:
                    for stem, path in output_files:
                        with gr.Row(equal_height=True):
                            audio = gr.Audio(label=stem, value=path, type="filepath", interactive=False, show_download_button=True, scale=15)
                            reuse_btn = gr.Button(t("reuse"), scale=1)
                            reuse_btn.click(
                                lambda x: (gr.update(value=x), gr.update(value=x)),
                                inputs=audio,
                                outputs=[mvsep_api_ui_input_path, mvsep_api_ui_input_audio]
                            )

        mvsep_api_ui_input_audio.change(
            lambda x: gr.update(value=x), inputs=mvsep_api_ui_input_audio, outputs=mvsep_api_ui_input_path
        )

        mvsep_api_ui_path_0_btn.click(
            lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[mvsep_api_ui_local, mvsep_api_ui_path],
        )

        mvsep_api_ui_path_1_btn.click(
            lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[mvsep_api_ui_url, mvsep_api_ui_path],
        )

        mvsep_api_ui_url_0_btn.click(
            lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[mvsep_api_ui_local, mvsep_api_ui_url],
        )

        mvsep_api_ui_url_1_btn.click(
            lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[mvsep_api_ui_path, mvsep_api_ui_url],
        )

        mvsep_api_ui_upload_0_btn.click(
            lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[mvsep_api_ui_local, mvsep_api_ui_url],
        )

        mvsep_api_ui_upload_1_btn.click(
            lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[mvsep_api_ui_local, mvsep_api_ui_path],
        )

        mvsep_api_ui_download_audio_btn.click(
            download_wrapper,
            inputs=[mvsep_api_ui_input_link, mvsep_api_ui_upload_cookie],
            outputs=[mvsep_api_ui_input_audio, mvsep_api_ui_input_path, mvsep_api_ui_local, mvsep_api_ui_url],
            show_progress=True,
        )

        mvsep_api_ui_algorithm_dropdown.change(
            fn=update_add_opts,
            inputs=mvsep_api_ui_algorithm_dropdown,
            outputs=[mvsep_api_ui_add_opt1_dropdown, mvsep_api_ui_add_opt2_dropdown, mvsep_api_ui_add_opt3_dropdown],
        )

        mvsep_api_ui_api_token.change(set_api_token, inputs=mvsep_api_ui_api_token, outputs=gr.State())

        gr.on(
            fn=update_add_opts,
            inputs=mvsep_api_ui_algorithm_dropdown,
            outputs=[mvsep_api_ui_add_opt1_dropdown, mvsep_api_ui_add_opt2_dropdown, mvsep_api_ui_add_opt3_dropdown],
        )

else:
    def plugin_name():
        return "MVSEP API (OFF)"
    def plugin(lang):
        pass

if __name__ == "__main__":
    theme = gr.themes.Base(  # Тема соответствующая цветовой стилистике MVSep.com
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="slate",
        font=[
            gr.themes.GoogleFont("Poppins"),
            gr.themes.GoogleFont("Montserrat"),
            "Arial",
            "sans-serif",
        ],
        font_mono=[
            gr.themes.GoogleFont("Roboto Mono"),
            "Courier New",
            "monospace",
        ],
    ).set(
        button_primary_background_fill="#3a7bd5",
        button_primary_background_fill_hover="#2c65c0",
        button_primary_text_color="#ffffff",
        input_background_fill="#ffffff",
        input_border_color="#d0d0d6",
        block_background_fill="#ffffff",
        border_color_primary="#d0d0d6",
    )

    app = argparse.ArgumentParser(description='Vbach APP')
    app.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio app on (default: 7860)",
    )
    app.add_argument(
        "--share", action="store_true", help="Share the Gradio app publicly"
    )
    app.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = app.parse_args()

    with gr.Blocks(theme=theme) as demo:
        plugin("ru")
    demo.launch(server_port=args.port, share=args.share, debug=args.debug, allowed_paths=[
                        os.path.join(os.path.abspath(os.sep), "none"),
                        os.getcwd(),
                        os.path.expanduser('~'),
                        os.path.join(os.path.abspath(os.sep), "sdcard"),
                        os.path.join(os.path.abspath(os.sep), "content"),
                    ])
