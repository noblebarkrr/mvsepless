from typing import Optional, Union, Callable, Dict
from tqdm import tqdm
import json
import requests
from pathlib import Path
import os
import sys
import time
from datetime import timezone, timedelta
from copy import deepcopy
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
from i18n import _i18n
from extra_utils import dw_file_legacy as dw_file, DownloadError
from args_parser import parse_mvsep_api_args 
from audio import get_audio_files_from_list
from namer import Namer

from typing import Optional, Union, Callable, Dict, List, Any

# Количество попыток получить алгоритмы с сервера
ALGOS_FETCH_RETRIES = 7
# Задержка между попытками (сек)
ALGOS_FETCH_RETRY_DELAY = 2

class APITokenIsNotCorrect(Exception):
    pass


class SeparationNotExist(Exception):
    pass


class SeparationFailed(Exception):
    pass


class SeparationCreateError(Exception):
    pass


class MVSepAPIError(Exception):
    """Общая ошибка API MVSEP."""
    pass


# ======================================================================
# Регионы
# ======================================================================
REGIONS = {
    "main": "https://mvsep.com/api",
    "de": "https://de.mvsep.com/api",
    "de2": "https://de2.mvsep.com/api",
    "sg": "https://sg.mvsep.com/api",
    "hk": "https://hk.mvsep.com/api",  # устаревший
}


class MVSEP_Client:
    def __init__(self, api_token: str = "", region: str = "main"):
        """
        :param api_token: API-ключ пользователя (если пуст — будет загружен из файла)
        :param region: регион ('main', 'de', 'de2', 'sg', 'hk')
        """
        if region not in REGIONS:
            raise ValueError(_i18n(
                "mvsep_api_unknown_region",
                region=region,
                available=list(REGIONS.keys()),
            ))
        self.region = region
        self.base_url = REGIONS[region]
        self.algos_cache_path = BASE_DIR / "mvsep_algorithms.json"
        self.api_token_path = BASE_DIR / "mvsep_api_token.txt"

        # Внутренние структуры
        self.algo_dict: Dict[Any, dict] = {}
        self.algo_dict_map: Dict[str, dict] = {}
        self.algo_raw: list = []          # сырой ответ сервера

        # API-ключ: приоритет — переданный аргумент, иначе — из файла
        self.api_token = ""
        if api_token:
            self.set_api_token(api_token)
        else:
            self._load_api_token_from_file()

        self.fail_retries = 20
        self.retries = 999999
        self.retry_interval = 1
        self.output_formats_map = {
            ("mp3", True): 0, ("mp3", False): 0,
            ("wav", False): 1,
            ("flac", False): 2,
            ("m4a", False): 3, ("m4a", True): 3,
            ("wav", True): 4,
            ("flac", True): 5
        }
        self.output_formats = ("mp3", "wav", "flac", "m4a")

        # Пытаемся получить алгоритмы с сервера (несколько попыток),
        # при неудаче — из кэша. Если и кэша нет — просто пустые словари.
        try:
            self._load_or_fetch_algorithms()
        except Exception:
            # не падаем, если совсем ничего не удалось
            pass

    # ------------------------------------------------------------------
    # API-ключ: хранение в txt
    # ------------------------------------------------------------------
    def _load_api_token_from_file(self):
        """Загрузить API-ключ из txt-файла, если он существует."""
        try:
            if self.api_token_path.exists():
                token = self.api_token_path.read_text(encoding="utf-8").strip()
                if token:
                    self.api_token = token
        except Exception:
            # не критично — работаем без токена
            self.api_token = ""

    def save_api_token(self):
        """Сохранить текущий API-ключ в txt-файл."""
        try:
            self.api_token_path.write_text(self.api_token or "", encoding="utf-8")
        except Exception:
            pass

    def set_api_token(self, token: str):
        """Установить API-ключ и сохранить его в файл."""
        self.api_token = token or ""
        self.save_api_token()

    def check_api_token_is_installed(self):
        return bool(self.api_token)

    # ------------------------------------------------------------------
    # Алгоритмы: кэш + несколько попыток
    # ------------------------------------------------------------------
    def _load_or_fetch_algorithms(self):
        """
        Загружает алгоритмы:
        1) Пытается получить с сервера несколько раз.
        2) Если все попытки исчерпаны — берёт из кэша.
        3) Если и кэша нет — оставляет пустые структуры.
        """
        raw = self._fetch_algorithms_with_retries()
        if raw is not None:
            self._set_algorithms_from_raw(raw, save_cache=True)
            return

        # Все попытки исчерпаны — пробуем кэш
        if self._load_algorithms_from_cache():
            return

        # Ничего не осталось — пустые структуры
        self.algo_raw = []
        self.algo_dict = {}
        self.algo_dict_map = {}

    def _fetch_algorithms_with_retries(self):
        """
        Пытается получить алгоритмы с сервера.
        Возвращает сырой список либо None, если все попытки провалились.
        """
        scopes = ["single_upload"]
        last_error = None
        for attempt in range(1, ALGOS_FETCH_RETRIES + 1):
            try:
                response = requests.get(
                    f"{self.base_url}/app/algorithms",
                    params={"scopes": ",".join(scopes)},
                    timeout=30,
                )
                response_json = response.json()
                if isinstance(response_json, list):
                    return response_json
                last_error = MVSepAPIError(_i18n("mvsep_api_algorithms_failed"))
            except requests.RequestException as e:
                last_error = e
            except Exception as e:
                last_error = e
            # пауза перед следующей попыткой (кроме последней)
            if attempt < ALGOS_FETCH_RETRIES:
                time.sleep(ALGOS_FETCH_RETRY_DELAY)

        # Все попытки провалились
        return None

    def _parse_algorithms_raw(self, response_json: list):
        """
        Разбирает сырой список алгоритмов в (algo_dict, algo_dict_map).
        Возвращает кортеж словарей.
        """
        algo_dict = {}
        algo_dict_map = {}
        for algorithm in response_json:
            render_id = algorithm["render_id"]
            algorithm_name = algorithm["name"]
            algo_fields = algorithm.get("algorithm_fields", [])
            add_opts = {}
            add_opts_map = {}
            for add_opt in algo_fields:
                add_opt_name = add_opt["name"]       # например "add_opt1"
                add_opt_text = add_opt["text"]       # человекочитаемый текст
                try:
                    add_opt_options = json.loads(add_opt["options"])
                except (json.JSONDecodeError, TypeError):
                    add_opt_options = {}
                # храним текст отдельно
                add_opts[add_opt_name] = {
                    "text": add_opt_text,
                    "options": add_opt_options,
                }
                add_opts_map[add_opt_name] = {
                    "text": add_opt_text,
                    "options": {
                        option_name: option_number
                        for option_number, option_name in add_opt_options.items()
                    },
                }
            algo_dict[render_id] = {
                "name": algorithm_name,
                "add_opts": add_opts,
                "algorithm_fields": algo_fields,
            }
            algo_dict_map[algorithm_name] = {
                "id": render_id,
                "add_opts": add_opts_map,
            }
        return algo_dict, algo_dict_map

    def _set_algorithms_from_raw(self, raw: list, save_cache: bool = False):
        """Устанавливает структуры из сырого списка и (опционально) сохраняет кэш."""
        algo_dict, algo_dict_map = self._parse_algorithms_raw(raw)
        self.algo_raw = raw
        self.algo_dict = algo_dict
        self.algo_dict_map = algo_dict_map
        if save_cache:
            self._save_algorithms_cache()

    def _save_algorithms_cache(self):
        """Сохраняет сырой оригинал + отдельные маппинги в JSON-кэш."""
        try:
            cache = {
                "raw": self.algo_raw,
                "algo_dict": self.algo_dict,
                "algo_dict_map": self.algo_dict_map,
            }
            self.algos_cache_path.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            # не критично, если не удалось сохранить кэш
            pass

    def _load_algorithms_from_cache(self) -> bool:
        """
        Загружает алгоритмы из кэша.
        Возвращает True, если удалось загрузить.
        """
        if not self.algos_cache_path.exists():
            return False
        try:
            cache = json.loads(
                self.algos_cache_path.read_text(encoding="utf-8")
            )
        except Exception:
            return False

        raw = cache.get("raw", [])
        algo_dict = cache.get("algo_dict", {})
        algo_dict_map = cache.get("algo_dict_map", {})

        # На случай, если в кэше сохранён только raw (старый формат)
        if raw and (not algo_dict or not algo_dict_map):
            algo_dict, algo_dict_map = self._parse_algorithms_raw(raw)

        if not raw and not algo_dict and not algo_dict_map:
            return False

        self.algo_raw = raw
        self.algo_dict = algo_dict
        self.algo_dict_map = algo_dict_map
        return True

    # ------------------------------------------------------------------
    # Публичный метод получения алгоритмов (с кэшем)
    # ------------------------------------------------------------------
    def get_all_sep_types(self, scopes: list = None):
        """
        Получить все типы разделения.
        Сначала пытается обновить с сервера (несколько попыток),
        при неудаче использует кэш. Всегда заполняет self.algo_dict
        и self.algo_dict_map.
        """
        if scopes is None:
            scopes = ["single_upload"]

        # Пытаемся получить с сервера (несколько попыток)
        raw = None
        last_error = None
        for attempt in range(1, ALGOS_FETCH_RETRIES + 1):
            try:
                response = requests.get(
                    f"{self.base_url}/app/algorithms",
                    params={"scopes": ",".join(scopes)},
                    timeout=30,
                )
                response_json = response.json()
                if isinstance(response_json, list):
                    raw = response_json
                    break
                last_error = MVSepAPIError(
                    _i18n("mvsep_api_algorithms_failed")
                )
            except requests.RequestException as e:
                last_error = MVSepAPIError(
                    _i18n("mvsep_api_request_failed", error=str(e))
                )
            except Exception as e:
                last_error = e
            if attempt < ALGOS_FETCH_RETRIES:
                time.sleep(ALGOS_FETCH_RETRY_DELAY)

        if raw is not None:
            self._set_algorithms_from_raw(raw, save_cache=True)
            return

        # Все попытки провалились — пробуем кэш
        if self._load_algorithms_from_cache():
            return

        # Если кэша нет — бросаем последнюю ошибку
        if last_error is not None:
            raise last_error
        raise MVSepAPIError(_i18n("mvsep_api_algorithms_failed"))

    def get_algorithms_raw(self, scopes: list = None):
        """
        Сырые данные по алгоритмам.
        Сначала пытается получить с сервера, при неудаче — из кэша.
        """
        if scopes is None:
            scopes = ["single_upload"]

        for attempt in range(1, ALGOS_FETCH_RETRIES + 1):
            try:
                response = requests.get(
                    f"{self.base_url}/app/algorithms",
                    params={"scopes": ",".join(scopes)},
                    timeout=30,
                )
                response_json = response.json()
                if isinstance(response_json, list):
                    return response_json
            except requests.RequestException:
                pass
            except Exception:
                pass
            if attempt < ALGOS_FETCH_RETRIES:
                time.sleep(ALGOS_FETCH_RETRY_DELAY)

        # Из кэша
        if self.algo_raw:
            return self.algo_raw
        if self._load_algorithms_from_cache():
            return self.algo_raw
        raise MVSepAPIError(_i18n("mvsep_api_algorithms_failed"))

    # ------------------------------------------------------------------
    # Внутренние хелперы
    # ------------------------------------------------------------------
    def _get(self, path: str, params: dict = None, timeout: int = 60):
        """Универсальный GET-запрос с обработкой ошибок."""
        params = params or {}
        if self.api_token:
            params.setdefault("api_token", self.api_token)
        try:
            response = requests.get(
                f"{self.base_url}{path}", params=params, timeout=timeout
            )
        except requests.RequestException as e:
            raise MVSepAPIError(
                _i18n("mvsep_api_request_failed", error=str(e))
            )
        if response.status_code == 401:
            raise APITokenIsNotCorrect(_i18n("mvsep_api_invalid_token"))
        return response

    def _post(self, path: str, data: dict = None, files: dict = None,
              timeout: int = 300):
        """Универсальный POST-запрос с обработкой ошибок."""
        data = data or {}
        if self.api_token:
            data.setdefault("api_token", self.api_token)
        try:
            response = requests.post(
                f"{self.base_url}{path}",
                data=data,
                files=files,
                timeout=timeout,
            )
        except requests.RequestException as e:
            raise MVSepAPIError(
                _i18n("mvsep_api_request_failed", error=str(e))
            )
        if response.status_code == 401:
            raise APITokenIsNotCorrect(_i18n("mvsep_api_invalid_token"))
        if response.status_code == 400:
            raise MVSepAPIError(_i18n("mvsep_api_invalid_params"))
        return response

    def _check_success(self, response_json: dict, error_cls=MVSepAPIError,
                       error_key: str = "mvsep_api_unknown_error"):
        """Проверяет success и бросает исключение при ошибке."""
        if not response_json.get("success", False):
            data = response_json.get("data", {})
            message = ""
            if isinstance(data, dict):
                message = data.get("message", "")
            if not message:
                message = response_json.get("message", "")
            if not message:
                message = _i18n(error_key)
            raise error_cls(message)
        return response_json

    # ------------------------------------------------------------------
    # Аутентификация
    # ------------------------------------------------------------------
    def register(self, name: str, email: str, password: str):
        """Регистрация нового пользователя."""
        try:
            response = requests.post(
                f"{self.base_url}/app/register",
                data={
                    "name": name,
                    "email": email,
                    "password": password,
                    "password_confirmation": password,
                },
            )
        except requests.RequestException as e:
            raise MVSepAPIError(
                _i18n("mvsep_api_request_failed", error=str(e))
            )
        if response.status_code == 400:
            raise MVSepAPIError(_i18n("mvsep_api_register_invalid_form"))
        response_json = response.json()
        success = response_json.get("success", False)
        message = response_json.get("message", "")
        if not success:
            raise ValueError(_i18n(
                "mvsep_api_register_failed", message=message
            ))
        return response_json

    def login(self, email: str, password: str):
        """Аутентификация и получение API-ключа."""
        try:
            response = requests.post(
                f"{self.base_url}/app/login",
                data={"email": email, "password": password},
            )
        except requests.RequestException as e:
            raise MVSepAPIError(
                _i18n("mvsep_api_request_failed", error=str(e))
            )
        if response.status_code == 400:
            raise MVSepAPIError(
                _i18n("mvsep_api_login_invalid_credentials")
            )
        response_json = response.json()
        if not response_json.get("success", False):
            raise MVSepAPIError(_i18n(
                "mvsep_api_login_failed",
                message=response_json.get("message", ""),
            ))
        data = response_json.get("data", {})
        if data.get("api_token"):
            # сохраняем в файл
            self.set_api_token(data["api_token"])
        return data

    # ------------------------------------------------------------------
    # Пользователь / профиль
    # ------------------------------------------------------------------
    def get_user_info(self, return_data: bool = False):
        """Получить информацию о пользователе."""
        response = self._get("/app/user")
        if response.status_code == 400:
            raise APITokenIsNotCorrect(_i18n("mvsep_api_invalid_token"))
        response_json = response.json()
        if not response_json.get("success", False):
            raise APITokenIsNotCorrect(_i18n("mvsep_api_invalid_token"))
        data_user = response_json.get("data", {})
        if return_data:
            return data_user
        copy_metadata = data_user.get("copy_metadata", 0)
        long_filenames_enabled = data_user.get("long_filenames_enabled", 0)
        premium_minutes = data_user.get("premium_minutes", 0)
        premium_enabled = data_user.get("premium_enabled", 0)
        return (
            bool(copy_metadata),
            bool(long_filenames_enabled),
            int(premium_minutes),
            bool(premium_enabled),
        )

    def check_valid_api_token(self):
        response = self._get("/app/user")
        if response.status_code == 400:
            return False
        response_json = response.json()
        if not response_json.get("success", False):
            return False
        return True

    def get_current_queue(self):
        """Текущие разделения пользователя."""
        return self.get_user_info(return_data=True).get("current_queue", [])

    def get_separation_history(self, start: int = 0, limit: int = 10,
                               show_only_job_exists: bool = True, no_raise: bool = False):
        """История разделений (отфильтрованная)."""
        list_separations = []
        if limit > 20:
            limit = 20
        response = self._get(
            "/app/separation_history",
            params={"start": start, "limit": limit},
        )
        if response.status_code == 400:
            if not no_raise:
                raise APITokenIsNotCorrect(_i18n("mvsep_api_invalid_token"))
        response_json = response.json()
        if not response_json.get("success", False):
            if not no_raise:
                raise MVSepAPIError(_i18n("mvsep_api_history_failed"))
        
        for separation in response_json.get("data", []):
            separation_hash = separation.get("hash", "")
            job_exists = separation.get("job_exists", False)
            if not job_exists and show_only_job_exists:
                continue
            list_separations.append(separation_hash)
        return list_separations

    def get_separation_history_raw(self, start: int = 0, limit: int = 10):
        """История разделений (сырые данные)."""
        if limit > 20:
            limit = 20
        response = self._get(
            "/app/separation_history",
            params={"start": start, "limit": limit},
        )
        response_json = response.json()
        self._check_success(response_json, error_key="mvsep_api_history_failed")
        return response_json.get("data", [])

    def get_purchases(self, limit: int = 20, offset: int = 0,
                      status: str = "all"):
        """История покупок кредитов."""
        params = {"limit": limit, "offset": offset, "status": status}
        response = self._get("/purchases", params=params)
        response_json = response.json()
        self._check_success(response_json, error_key="mvsep_api_purchases_failed")
        return response_json

    def get_credit_alert(self):
        """Текущие настройки уведомления о низком балансе."""
        response = self._get("/credit_alert")
        response_json = response.json()
        self._check_success(
            response_json, error_key="mvsep_api_credit_alert_failed"
        )
        return response_json

    def set_credit_alert(self, credit_threshold: int):
        """Установить порог уведомления о низком балансе."""
        response = self._post(
            "/credit_alert",
            data={"credit_threshold": int(credit_threshold)},
        )
        response_json = response.json()
        self._check_success(
            response_json, error_key="mvsep_api_credit_alert_set_failed"
        )
        return response_json

    def enable_premium(self):
        """Включить использование премиума."""
        response = self._post("/app/enable_premium")
        response_json = response.json()
        self._check_success(
            response_json, error_key="mvsep_api_enable_premium_failed"
        )
        return response_json

    def disable_premium(self):
        """Отключить использование премиума."""
        response = self._post("/app/disable_premium")
        response_json = response.json()
        self._check_success(
            response_json, error_key="mvsep_api_disable_premium_failed"
        )
        return response_json

    def enable_long_filenames(self):
        """Включить длинные имена файлов."""
        response = self._post("/app/enable_long_filenames")
        response_json = response.json()
        self._check_success(
            response_json,
            error_key="mvsep_api_enable_long_filenames_failed",
        )
        return response_json

    def disable_long_filenames(self):
        """Отключить длинные имена файлов."""
        response = self._post("/app/disable_long_filenames")
        response_json = response.json()
        self._check_success(
            response_json,
            error_key="mvsep_api_disable_long_filenames_failed",
        )
        return response_json

    # ------------------------------------------------------------------
    # Информация о сайте
    # ------------------------------------------------------------------
    def get_news(self, lang: str = "en", start: int = 0, limit: int = 10):
        """Получить новости MVSEP."""
        if limit > 20:
            limit = 20
        try:
            response = requests.get(
                f"{self.base_url}/app/news",
                params={"lang": lang, "start": start, "limit": limit},
            )
        except requests.RequestException as e:
            raise MVSepAPIError(
                _i18n("mvsep_api_request_failed", error=str(e))
            )
        return response.json()

    def get_queue(self):
        """Информация об очереди сайта."""
        params = {}
        if self.api_token:
            params["api_token"] = self.api_token
        try:
            response = requests.get(
                f"{self.base_url}/app/queue", params=params
            )
        except requests.RequestException as e:
            raise MVSepAPIError(
                _i18n("mvsep_api_request_failed", error=str(e))
            )
        return response.json()

    def get_queue_summary(self):
        """Статус вашей очереди."""
        response = self._get("/app/queue/summary")
        response_json = response.json()
        self._check_success(
            response_json, error_key="mvsep_api_queue_summary_failed"
        )
        return response_json

    def get_demo_separations(self, start: int = 0, limit: int = 10,
                             algorithm_id: int = None,
                             options: dict = None):
        """Получить демо-разделения."""
        if limit > 20:
            limit = 20
        params = {"start": start, "limit": limit}
        if algorithm_id is not None:
            params["algorithm_id"] = algorithm_id
        if options:
            for field, value in options.items():
                params[f"options[{field}]"] = value
        try:
            response = requests.get(
                f"{self.base_url}/app/demo", params=params
            )
        except requests.RequestException as e:
            raise MVSepAPIError(
                _i18n("mvsep_api_request_failed", error=str(e))
            )
        return response.json()

    # ------------------------------------------------------------------
    # Разделение
    # ------------------------------------------------------------------
    def create_separation(
        self,
        input_file: Union[str, Path, None] = None,
        sep_type: int = 20,
        add_opt1: str = None,
        add_opt2: str = None,
        add_opt3: str = None,
        output_format: int = 0,
        url: str = None,
        remote_type: str = "direct",
        preset_id: int = None,
        is_demo: bool = False,
        webhook_url: str = None,
    ):
        """Создать задание на разделение."""
        data = {
            "sep_type": str(sep_type),
            "output_format": str(output_format),
        }
        if preset_id is not None:
            data["preset_id"] = str(preset_id)
        if add_opt1 is not None:
            data["add_opt1"] = str(add_opt1)
        if add_opt2 is not None:
            data["add_opt2"] = str(add_opt2)
        if add_opt3 is not None:
            data["add_opt3"] = str(add_opt3)
        if is_demo:
            data["is_demo"] = "1"
        if webhook_url:
            data["webhook_url"] = webhook_url
        if url:
            data["url"] = url
            data["remote_type"] = remote_type

        files = None
        if input_file is not None:
            files = {"audiofile": open(input_file, "rb")}

        try:
            response = requests.post(
                f"{self.base_url}/separation/create",
                data={"api_token": self.api_token, **data},
                files=files,
            )
        except requests.RequestException as e:
            raise SeparationCreateError(
                _i18n("mvsep_api_request_failed", error=str(e))
            )
        finally:
            if files:
                files["audiofile"].close()

        if response.status_code == 401:
            raise APITokenIsNotCorrect(_i18n("mvsep_api_invalid_token"))
        if response.status_code == 400:
            raise MVSepAPIError(_i18n("mvsep_api_create_invalid_params"))

        response_json = response.json()
        if not response_json.get("success", False):
            data_result = response_json.get("data", {})
            message = data_result.get("message", "")
            raise SeparationCreateError(_i18n(
                "mvsep_api_create_failed", message=message
            ))
        return response_json.get("data", {}).get("hash", "unknown_hash")

    def get_separation_info(self, separation_hash: str, progress: tqdm):
        """Получить статус разделения."""
        try:
            response = requests.get(
                f"{self.base_url}/separation/get",
                params={
                    "api_token": self.api_token,
                    "hash": separation_hash,
                },
            )
        except requests.RequestException as e:
            raise MVSepAPIError(
                _i18n("mvsep_api_request_failed", error=str(e))
            )
        if response.status_code == 400:
            raise APITokenIsNotCorrect(_i18n("mvsep_api_invalid_token"))
        response_json = response.json()
        status = response_json.get("status", "not_found")
        data_result = response_json.get("data", {})
        if isinstance(progress, tqdm):
            progress.close()
        if status == "not_found":
            progress.close()
            raise SeparationNotExist(
                _i18n("mvsep_api_separation_not_exists")
            )
        elif status == "waiting":
            queue_count = data_result.get("queue_count", 0)
            current_order = data_result.get("current_order", 0)
            progress = tqdm(
                desc=_i18n("mvsep_api_queue_progress_bar"), total=queue_count, unit=""
            )
            progress.update(current_order)
            return False, progress
        elif status in ("processing", "distributing", "merging"):
            progress = tqdm(
                desc=_i18n(
                    "mvsep_api_processing_status", status=status
                ), total=3, unit=""
            )
            progress.update(2)
            return False, progress
        elif status == "done":
            progress.close()
            return True, progress
        elif status == "failed":
            message = data_result.get("message", "")
            progress.close()
            raise SeparationFailed(_i18n(
                "mvsep_api_separation_failed", message=message
            ))
        return False, progress

    def get_separation_result(self, separation_hash: str, mirror: int = 0):
        """Получить полный ответ по разделению."""
        params = {"hash": separation_hash}
        if mirror:
            params["mirror"] = 1
            params["api_token"] = self.api_token
        try:
            response = requests.get(
                f"{self.base_url}/separation/get", params=params
            )
        except requests.RequestException as e:
            raise MVSepAPIError(
                _i18n("mvsep_api_request_failed", error=str(e))
            )
        return response.json()

    def get_separation_result_remote(self, separation_hash: str):
        """Получить результат удалённой задачи."""
        try:
            response = requests.get(
                f"{self.base_url}/separation/get-remote",
                params={"hash": separation_hash},
            )
        except requests.RequestException as e:
            raise MVSepAPIError(
                _i18n("mvsep_api_request_failed", error=str(e))
            )
        return response.json()

    def cancel_separation(self, separation_hash: str):
        """Отменить разделение."""
        try:
            response = self._post(
                "/separation/cancel", data={"hash": separation_hash}
            )
        except MVSepAPIError:
            raise
        response_json = response.json()
        if not response_json.get("success", False):
            raise MVSepAPIError(_i18n("mvsep_api_cancel_failed"))
        return True

    def delete_separation(self, separation_hash: str):
        """Удалить разделение и его выходные файлы."""
        response = self._post(
            "/separation/delete", data={"hash": separation_hash}
        )
        response_json = response.json()
        if not response_json.get("success", False):
            message = ""
            data = response_json.get("data", {})
            if isinstance(data, dict):
                message = data.get("message", "")
            raise MVSepAPIError(_i18n(
                "mvsep_api_delete_failed"
            ) + (f": {message}" if message else ""))
        return response_json

    # ------------------------------------------------------------------
    # Опции / пресеты
    # ------------------------------------------------------------------
    def get_list_algos_map(self):
        return list(self.algo_dict_map.keys())

    def get_add_opts_from_algo_map(self, algo_name: str):
        if algo_name in self.algo_dict_map:
            return self.algo_dict_map[algo_name]["add_opts"]
        return {}

    def name_to_id_sep_kwargs(self, sep_type: str, add_opt1: str = None,
                              add_opt2: str = None, add_opt3: str = None):
        """Преобразовать имена опций в ID для конкретного алгоритма."""
        sep_kwargs = {}
        if sep_type not in self.algo_dict_map:
            return sep_kwargs

        algo_info = self.algo_dict_map[sep_type]
        sep_kwargs["sep_type"] = algo_info["id"]
        
        # Структура add_opts в algo_dict_map:
        # { "add_opt1": { "text": "...", "options": { "option_name": option_id, ... } }, ... }
        add_opts = algo_info.get("add_opts", {})

        mapping = {
            "add_opt1": add_opt1,
            "add_opt2": add_opt2,
            "add_opt3": add_opt3,
        }

        for opt_key, opt_value in mapping.items():
            if opt_value is None:
                continue
            
            if opt_key in add_opts:
                # Получаем словарь вариантов для конкретной опции
                options_map = add_opts[opt_key].get("options", {})
                
                # Пытаемся найти ID по имени варианта
                if opt_value in options_map:
                    sep_kwargs[opt_key] = options_map[opt_value]
                else:
                    # Если имя не найдено, возможно, уже передан ID или некорректное значение.
                    # Можно добавить логирование или оставить как есть, если API принимает строки.
                    # Для безопасности передаем исходное значение, но лучше предупредить.
                    print(f"Warning: Option value '{opt_value}' not found for {opt_key} in algorithm '{sep_type}'. Available: {list(options_map.keys())}")
                    sep_kwargs[opt_key] = opt_value 

        return sep_kwargs

    def get_add_opts_info(self, algo_name: str):
        """
        Возвращает для алгоритма словарь:
        {
            "add_opt1": {"text": "...", "choices": [...]},
            ...
        }
        """
        result = {}
        if algo_name not in self.algo_dict_map:
            return result
        add_opts = self.algo_dict_map[algo_name]["add_opts"]
        for key, value in add_opts.items():
            result[key] = {
                "text": value.get("text", key),
                "choices": list(value.get("options", {}).keys()),
            }
        return result

    # ------------------------------------------------------------------
    # Полный цикл обработки
    # ------------------------------------------------------------------
    def process_single_file(self, input_file: Union[str, Path],
                            output_dir: Union[str, Path],
                            **separation_kwargs):
        with tqdm(desc=_i18n("mvsep_api_uploading_file"), total=3, initial=0, unit="") as progress:
            separation_hash = self.create_separation(
                input_file, **separation_kwargs
            )
        progress = tqdm(
            desc=_i18n("mvsep_api_created_separation"), total=3, initial=1, unit=""
        )
        fail_retries = deepcopy(self.fail_retries)
        for _ in range(self.retries):
            try:
                time.sleep(self.retry_interval)
                status_done, progress = self.get_separation_info(separation_hash, progress)
                fail_retries = deepcopy(self.fail_retries)
                if status_done:
                    break
            except Exception as e:
                fail_retries -= 1
                print(_i18n("mvsep_api_request_failed", error=str(e)))
                progress.close()
                if fail_retries <= 0:
                    raise
        return self.download_results(separation_hash, output_dir)

    def batch_inference(
        self,
        input_files: list,
        output_dir: Union[str, Path],
        sep_type: str,
        output_format: int = 0,
        add_opt1: str = None,
        add_opt2: str = None,
        add_opt3: str = None,
        progress_bar: tqdm = None
    ):
        """
        Пакетная обработка файлов через MVSEP API.
        
        Returns:
            tuple: (results_list, errors_list)
            results_list формат: [[basename, [[stem_name, path], ...]], ...]
        """
        results = []
        errors = []
        total = len(input_files)
        
        # Если внешний прогресс-бар не передан, создаем свой
        if progress_bar is None:
            progress_bar = tqdm(total=total, desc=_i18n("processing"), unit=_i18n("files"))
            own_progress = True
        else:
            own_progress = False

        try:
            for i, file_path in enumerate(input_files):
                basename = Path(file_path).stem
                try:
                    progress_bar.close()
                    progress_bar = tqdm(total=total, desc=f"{_i18n('processing')} {i+1}/{total}", unit=_i18n("files"))
                    
                    # process_single_file уже имеет внутренний tqdm для загрузки/ожидания/скачивания
                    # Но для пакетной обработки мы хотим видеть общий прогресс по файлам
                    stems_list, status = self.process_single_file(
                        input_file=file_path,
                        output_dir=output_dir,
                        sep_type=sep_type,
                        output_format=output_format,
                        add_opt1=add_opt1,
                        add_opt2=add_opt2,
                        add_opt3=add_opt3
                    )
                    # stems_list от process_single_file: [[stem_name, path], ...]
                    results.append([basename, stems_list])
                    
                except Exception as e:
                    err_msg = f"{Path(file_path).name}: {str(e)}"
                    errors.append(err_msg)
                    print(err_msg)
                
                progress_bar.update(1)
        finally:
            progress_bar.close()

        return results, errors

    def download_results(self, separation_hash: str,
                         output_dir: Union[str, Path] = "", no_raise: bool = False, no_overwrite: bool = True):
        """Скачать результаты разделения."""
        output_dir = Path(output_dir)
        output_stems = []
        error_message = ""
        try:
            response = requests.get(
                f"{self.base_url}/separation/get",
                params={
                    "api_token": self.api_token,
                    "hash": separation_hash,
                },
            )
        except requests.RequestException as e:
            if not no_raise:
                raise MVSepAPIError(
                    _i18n("mvsep_api_request_failed", error=str(e))
                )
        if response.status_code == 400:
            if not no_raise:
                raise APITokenIsNotCorrect(_i18n("mvsep_api_invalid_token"))
        response_json = response.json()
        status = response_json.get("status", "not_found")
        data_result = response_json.get("data", {})

        if status == "not_found":
            print(_i18n("mvsep_api_separation_not_exists"))
        elif status == "waiting":
            queue_count = data_result.get("queue_count", 0)
            current_order = data_result.get("current_order", 0)
            print(_i18n(
                "mvsep_api_queue",
                current_order=current_order,
                queue_count=queue_count,
            ))
        elif status in ("processing", "distributing", "merging"):
            print(_i18n(
                "mvsep_api_processing_status", status=status
            ))
        elif status == "done":
            output_format = data_result.get(
                "output_format", "mp3 (320 kbps)"
            ).split(" (")[0]
            print(_i18n("mvsep_api_output_format", fmt=output_format))
            all_files = data_result.get("files", [])
            total_files = len(all_files)
            for stem_num, output_file in tqdm(list(enumerate(
                    all_files, start=1
            )), desc=_i18n("writing"), unit=_i18n('files')):
                stem_name = output_file.get("type", f"Stem {stem_num}")
                stem_url = output_file.get("url", "")
                file_name = output_file.get(
                    "download", f"stem_{stem_num}.{output_format}"
                )
                if not no_overwrite:
                    output_path = output_dir / file_name
                else:
                    output_path = Namer.iter(output_dir / file_name)
                try:
                    if not Path(output_path).exists():
                        dw_file(stem_url, output_path)
                except Exception as e:
                    raise DownloadError(_i18n(
                        "mvsep_api_download_results_failed"
                    ) + f": {e}")
                print(_i18n(
                    "mvsep_api_download_stem",
                    stem_name=stem_name,
                    path=output_path,
                ))
                output_stems.append([stem_name, output_path])
        elif status == "failed":
            error_message = data_result.get("message", "")
            print(_i18n(
                "mvsep_api_separation_failed", message=error_message
            ))

        status_str = _i18n(f"mvsep_api_status_{status}")
        if error_message:
            status_str += f"\n{error_message}"
        return output_stems, status_str

    # ------------------------------------------------------------------
    # Quality Checker
    # ------------------------------------------------------------------
    def qc_get_queue(self, start: int = 0, limit: int = 10):
        """Записи в очереди Quality Checker."""
        if limit > 20:
            limit = 20
        try:
            response = requests.get(
                f"{self.base_url}/quality_checker/queue",
                params={"start": start, "limit": limit},
            )
        except requests.RequestException as e:
            raise MVSepAPIError(
                _i18n("mvsep_api_request_failed", error=str(e))
            )
        return response.json()

    def qc_get_leaderboard(self, dataset_type: str = "0", start: int = 0,
                           limit: int = 10, algo_name_filter: str = None,
                           sort: str = None):
        """Таблица лидеров Quality Checker."""
        if limit > 20:
            limit = 20
        params = {
            "dataset_type": dataset_type,
            "start": start,
            "limit": limit,
        }
        if algo_name_filter:
            params["algo_name_filter"] = algo_name_filter
        if sort:
            params["sort"] = sort
        try:
            response = requests.get(
                f"{self.base_url}/quality_checker/leaderboard",
                params=params,
            )
        except requests.RequestException as e:
            raise MVSepAPIError(
                _i18n("mvsep_api_request_failed", error=str(e))
            )
        return response.json()

    def qc_add_entry(self, zipfile: Union[str, Path], algo_name: str,
                     main_text: str, password: str,
                     dataset_type: str = "0", ensemble: int = 0):
        """Создать запись Quality Checker."""
        data = {
            "api_token": self.api_token,
            "algo_name": algo_name,
            "main_text": main_text,
            "dataset_type": dataset_type,
            "password": password,
            "ensemble": str(ensemble),
        }
        try:
            with open(zipfile, "rb") as f:
                files = {"zipfile": f}
                response = requests.post(
                    f"{self.base_url}/quality_checker/add",
                    data=data,
                    files=files,
                )
        except requests.RequestException as e:
            raise MVSepAPIError(
                _i18n("mvsep_api_request_failed", error=str(e))
            )
        if response.status_code == 401:
            raise APITokenIsNotCorrect(_i18n("mvsep_api_invalid_token"))
        if response.status_code == 400:
            raise MVSepAPIError(_i18n("mvsep_api_invalid_params"))
        response_json = response.json()
        if not response_json.get("success", False):
            message = ""
            data_resp = response_json.get("data", {})
            if isinstance(data_resp, dict):
                message = data_resp.get("message", "")
            raise MVSepAPIError(_i18n(
                "mvsep_api_qc_add_failed", message=message
            ))
        return response_json.get("data", {})

    def qc_get_entry(self, entry_id: int):
        """Получить запись Quality Checker по ID."""
        try:
            response = requests.get(
                f"{self.base_url}/quality_checker/entry",
                params={"id": entry_id},
            )
        except requests.RequestException as e:
            raise MVSepAPIError(
                _i18n("mvsep_api_request_failed", error=str(e))
            )
        return response.json()

    def qc_delete_entry(self, entry_id: int, password: str):
        """Удалить запись Quality Checker."""
        try:
            response = requests.post(
                f"{self.base_url}/quality_checker/delete",
                data={"id": str(entry_id), "password": password},
            )
        except requests.RequestException as e:
            raise MVSepAPIError(
                _i18n("mvsep_api_request_failed", error=str(e))
            )
        return response.json()


def _print_json(data):
    """Красивый вывод JSON-ответов API."""
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _handle_command(args, client: MVSEP_Client):
    """Диспетчеризация команд MVSEP API."""
    cmd = args.command

    # ==================================================================
    # Аутентификация
    # ==================================================================
    if cmd == "register":
        result = client.register(args.name, args.email, args.password)
        print(_i18n("mvsep_api_register_success"))
        _print_json(result)

    elif cmd == "login":
        data = client.login(args.email, args.password)
        token_preview = data.get('api_token', 'N/A')[:8] + "..."
        print(_i18n("mvsep_api_token_saved", token=token_preview))
        _print_json(data)

    elif cmd == "set_token":
        # set_api_token возвращает None, поэтому формируем ответ вручную
        client.set_api_token(args.token)
        token_preview = args.token[:8] + "..." if len(args.token) > 8 else args.token
        print(_i18n("mvsep_api_token_saved", token=token_preview))

    elif cmd == "check_token":
        valid = client.check_valid_api_token()
        status_msg = _i18n("mvsep_api_token_valid") if valid else _i18n("mvsep_api_token_invalid")
        print(f"{_i18n('status')}: {status_msg}")

    # ==================================================================
    # Профиль / Настройки
    # ==================================================================
    elif cmd == "user_info":
        data = client.get_user_info(return_data=True)
        _print_json(data)

    elif cmd == "separation_history":
        history = client.get_separation_history(
            start=args.start,
            limit=args.limit,
            show_only_job_exists=not args.show_all
        )
        for label, hash_val in history.items():
            print(f"  {label}")
        print(f"\n{_i18n('history')}: {len(history)} entries")

    elif cmd == "purchases":
        purchases = client.get_purchases(
            limit=args.limit, offset=args.offset, status=args.status
        )
        _print_json(purchases)

    elif cmd == "credit_alert":
        if args.set_threshold is not None:
            result = client.set_credit_alert(args.set_threshold)
            print(_i18n("mvsep_api_credit_alert_set_success", threshold=args.set_threshold))
            _print_json(result)
        else:
            result = client.get_credit_alert()
            _print_json(result)

    elif cmd == "premium":
        if args.enable:
            result = client.enable_premium()
            print(_i18n("mvsep_api_premium_enabled"))
        else:
            result = client.disable_premium()
            print(_i18n("mvsep_api_premium_disabled"))
        _print_json(result)

    elif cmd == "long_filenames":
        if args.enable:
            result = client.enable_long_filenames()
            print(_i18n("mvsep_api_long_fn_enabled"))
        else:
            result = client.disable_long_filenames()
            print(_i18n("mvsep_api_long_fn_disabled"))
        _print_json(result)

    # ==================================================================
    # Информация / Новости / Очередь
    # ==================================================================
    elif cmd == "news":
        news = client.get_news(lang=args.lang, start=args.start, limit=args.limit)
        _print_json(news)

    elif cmd == "queue":
        if args.summary:
            result = client.get_queue_summary()
        else:
            result = client.get_queue()
        _print_json(result)

    elif cmd == "demo":
        options = {}
        demos = client.get_demo_separations(
            start=args.start, limit=args.limit,
            algorithm_id=args.algorithm_id, options=options or None
        )
        _print_json(demos)

    elif cmd == "algorithms":
        if args.raw:
            result = client.get_algorithms_raw(scopes=args.scopes.split(","))
        else:
            algos = client.get_list_algos_map()
            result = {"algorithms": algos, "count": len(algos)}
        _print_json(result)

    # ==================================================================
    # Разделение (основная команда)
    # ==================================================================
    elif cmd == "separate":
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ {_i18n('path_not_exist')}: {input_path}", file=sys.stderr)
            sys.exit(1)

        stems = client.process_single_file(
            input_file=str(input_path),
            output_dir=args.output_dir,
            sep_type=args.sep_type,
            add_opt1=args.add_opt1,
            add_opt2=args.add_opt2,
            add_opt3=args.add_opt3,
            output_format=args.output_format,
            url=args.url,
            remote_type=args.remote_type,
            preset_id=args.preset_id,
            is_demo=args.is_demo,
            webhook_url=args.webhook_url,
        )
        print(f"\n✅ {_i18n('download_complete')}")
        for stem_name, stem_path in stems:
            print(f"  📁 {stem_name}: {stem_path}")

    # ==================================================================
    # Статус / Скачивание / Отмена / Удаление
    # ==================================================================
    elif cmd == "status":
        info = client.get_separation_result(args.hash)
        _print_json(info)

    elif cmd == "download":
        stems, status_str = client.download_results(
            separation_hash=args.hash,
            output_dir=args.output_dir,
        )
        print(f"\n{_i18n('status')}: {status_str}")
        for stem_name, stem_path in stems:
            print(f"  📁 {stem_name}: {stem_path}")

    elif cmd == "cancel":
        try:
            client.cancel_separation(args.hash)
            print(_i18n("mvsep_api_cancel_success"))
        except MVSepAPIError as e:
            print(f"❌ {e}", file=sys.stderr)

    elif cmd == "delete":
        try:
            client.delete_separation(args.hash)
            print(_i18n("mvsep_api_delete_success"))
        except MVSepAPIError as e:
            print(f"❌ {e}", file=sys.stderr)

    # ==================================================================
    # Quality Checker
    # ==================================================================
    elif cmd == "qc_add":
        zip_path = Path(args.zipfile)
        if not zip_path.exists():
            print(f"❌ {_i18n('path_not_exist')}: {zip_path}", file=sys.stderr)
            sys.exit(1)

        result = client.qc_add_entry(
            zipfile=str(zip_path),
            algo_name=args.algo_name,
            main_text=args.main_text,
            password=args.password,
            dataset_type=args.dataset_type,
            ensemble=args.ensemble,
        )
        print(_i18n("mvsep_api_qc_add_success"))
        _print_json(result)

    elif cmd == "qc_get":
        result = client.qc_get_entry(args.entry_id)
        _print_json(result)

    elif cmd == "qc_delete":
        result = client.qc_delete_entry(args.entry_id, args.password)
        print(_i18n("mvsep_api_qc_delete_success"))
        _print_json(result)

    elif cmd == "qc_queue":
        result = client.qc_get_queue(start=args.start, limit=args.limit)
        _print_json(result)

    elif cmd == "qc_leaderboard":
        result = client.qc_get_leaderboard(
            dataset_type=args.dataset_type,
            start=args.start,
            limit=args.limit,
            algo_name_filter=args.algo_filter,
            sort=args.sort,
        )
        _print_json(result)

    # ==================================================================
    # Пакетная обработка
    # ==================================================================
    elif cmd == "batch":
        from audio import get_audio_files_from_list
        input_files = get_audio_files_from_list(args.input)
        if not input_files:
            print(f"❌ {_i18n('files_is_not_audio')}", file=sys.stderr)
            sys.exit(1)

        print(f"📂 {_i18n('processing')} {len(input_files)} {_i18n('files')}...")

        # Преобразуем строковое имя типа в ID через клиент
        sep_kwargs = client.name_to_id_sep_kwargs(
            sep_type=args.sep_type,
            add_opt1=args.add_opt1,
            add_opt2=args.add_opt2,
            add_opt3=args.add_opt3,
        )
        sep_type_id = sep_kwargs.get("sep_type", args.sep_type)

        results, errors = client.batch_inference(
            input_files=input_files,
            output_dir=args.output_dir,
            sep_type=sep_type_id,
            output_format=args.output_format,
            add_opt1=sep_kwargs.get("add_opt1", args.add_opt1),
            add_opt2=sep_kwargs.get("add_opt2", args.add_opt2),
            add_opt3=sep_kwargs.get("add_opt3", args.add_opt3),
        )

        print(f"\n✅ {_i18n('download_complete')}: {len(results)}/{len(input_files)}")
        if errors:
            print(f"\n❌ {_i18n('errors')} ({len(errors)}):")
            for err in errors:
                print(f"  ⚠️  {err}")

    else:
        print(f"❌ {_i18n('unknown_command')}: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    args = parse_mvsep_api_args()

    if not args.command:
        # Если команда не указана — показать справку
        parse_mvsep_api_args.__wrapped__() if hasattr(parse_mvsep_api_args, '__wrapped__') else None
        import argparse
        parser = argparse.ArgumentParser(description=_i18n("mvsep_api_cli_description"))
        parser.print_help()
        sys.exit(0)

    client = MVSEP_Client(
        api_token=args.token or "",
        region=args.region,
    )
    _handle_command(args, client)