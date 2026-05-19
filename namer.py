import re
from typing import Dict, Any, Optional, List
from pathlib import Path
from i18n import _i18n

MAX_LENGTH = 255
OFFSET = 10
if OFFSET < MAX_LENGTH:
    SAFE_MAX_LENGTH = MAX_LENGTH - OFFSET
else:
    SAFE_MAX_LENGTH = MAX_LENGTH

class Namer:

    @staticmethod
    def sanitize(name: str) -> str:
        """
        Очистить имя файла от недопустимых символов
        
        Args:
            name: Исходное имя
        
        Returns:
            Очищенное имя
        """
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
        sanitized = re.sub(r"_+", "_", sanitized)
        sanitized = sanitized.strip("_. ")
        return sanitized

    @staticmethod
    def short(name: str, length: Optional[int] = None) -> str:
        """
        Сократить длинное имя
        
        Args:
            name: Исходное имя
            length: Желаемая длина
        
        Returns:
            Сокращенное имя
        """
        if length:
            if len(name) > length:
                return f"{name[:int(length // 2)]}...{name[-int(length // 2.5):]}"
            else:
                return name
        else:
            if len(name) > SAFE_MAX_LENGTH:
                return f"{name[:int(SAFE_MAX_LENGTH // 4)]}...{name[-int(SAFE_MAX_LENGTH // 4):]}"
            else:
                return name

    @staticmethod
    def iter(filepath: str | Path) -> str:
        """
        Создать уникальное имя файла, добавляя (n) если файл существует
        
        Args:
            filepath: Исходный путь
        
        Returns:
            Уникальный путь
        """
        filepath_ = Path(filepath)
        if not filepath_.exists():
            return filepath_.as_posix()

        counter = 1
        while True:
            new_filename = filepath_.stem + f" ({counter})"
            new_filepath = filepath_.with_stem(new_filename)
            if not new_filepath.exists():
                return new_filepath.as_posix()
            counter += 1

    @staticmethod
    def iter_in_list(filepath: str | Path, list_paths: list[str | Path] | tuple[str | Path, ...]) -> str:
        """
        Создать уникальное имя файла, добавляя (n) если файл существует
        
        Args:
            filepath: Исходный путь
            list_paths: Список путей
        
        Returns:
            Уникальный путь
        """
        filepath_ = Path(filepath)
        list_paths_str = [path.as_posix() for path in [Path(p) for p in list_paths]]
        if filepath_.as_posix() not in list_paths_str:
            return filepath_.as_posix()

        counter = 1
        while True:
            new_filename = filepath_.stem + f" ({counter})"
            new_filepath = filepath_.with_stem(new_filename)
            if new_filepath.as_posix() not in list_paths_str:
                return new_filepath.as_posix()
            counter += 1

    @staticmethod
    def template(template: str, **kwargs: Any) -> str:
        """
        Применить шаблон с подстановкой ключей
        
        Args:
            template: Шаблон
            **kwargs: Ключи для подстановки
        
        Returns:
            Результат подстановки
        """
        if kwargs:
            for key in kwargs:
                template = template.replace(str(key), str(kwargs[key]))
        return template

    @staticmethod
    def dedup_template(template: str, keys: List[str] = []) -> str:
        """
        Удалить дублирующиеся ключи из шаблона
        
        Args:
            template: Шаблон
            keys: Список ключей
        
        Returns:
            Шаблон без дубликатов
        """
        seen = set()
        pattern = r"({})".format("|".join(re.escape(key) for key in keys))

        def replace(match: re.Match) -> str:
            key = match.group(1)
            if key in seen:
                return ""
            seen.add(key)
            return key

        result = re.sub(pattern, replace, template)
        return result

    @staticmethod
    def short_input_name_template(template: str, **kwargs: Any) -> str:
        """
        Сократить имя входного файла с учетом шаблона
        
        Args:
            template: Шаблон
            **kwargs: Ключи для подстановки
        
        Returns:
            Сокращенное имя
        """
        if kwargs:
            input_file_name = kwargs.get("NAME", None)
            if input_file_name:
                merged_keys_value = ""
                no_keys_template = template
                for key in kwargs:
                    if key != "NAME":
                        merged_keys_value += str(kwargs[key])
                for key in kwargs:
                    no_keys_template = no_keys_template.replace(str(key), "")
                len_merged_keys = len(merged_keys_value)
                len_no_keys = len(no_keys_template)
                free_length = SAFE_MAX_LENGTH - (len_merged_keys + len_no_keys)
                len_file_name = len(input_file_name)
                start_index = free_length // 2
                end_index = free_length // 2.5
                if len_file_name > free_length:
                    return f"{input_file_name[:int(start_index)]}...{input_file_name[-int(end_index):]}"
                else:
                    return input_file_name
            else:
                print(_i18n("name_key_missing"))
                return ""
        else:
            print(_i18n("keys_required"))
            return ""