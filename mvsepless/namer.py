import os
import re

class Namer:
    def __init__(self, max_length: int = 255, offset: int = 10):
        if max_length < 40:
            self.max_length = 40
        else:
            self.max_length = max_length
        if offset < max_length:
            self.safe_max_length = max_length - offset
        else:
            self.safe_max_length = max_length

    def sanitize(self, name: str):
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
        sanitized = re.sub(r"_+", "_", sanitized)
        sanitized = sanitized.strip("_. ")
        return sanitized

    def short(self, name: str, length: int = None):
        if length:
            if len(name) > length:
                return f"{name[:int(length // 2)]}...{name[-int(length // 2.5):]}"
            else:
                return name
        else:
            if len(name) > self.safe_max_length:
                return f"{name[:int(self.safe_max_length // 4)]}...{name[-int(self.safe_max_length // 4):]}"
            else:
                return name

    def iter(self, filepath: str):
        if not os.path.exists(filepath):
            return filepath

        directory, filename = os.path.split(filepath)
        name, ext = os.path.splitext(filename)

        counter = 1
        while True:
            new_filename = f"{name} ({counter}){ext}"
            new_filepath = os.path.join(directory, new_filename)
            if not os.path.exists(new_filepath):
                return new_filepath
            counter += 1

    def template(self, template: str, **kwargs):
        if kwargs:
            for key in kwargs:
                template = template.replace(str(key), str(kwargs[key]))
        return template

    def dedup_template(self, template: str, keys: list = []):
        seen = set()
        pattern = r"({})".format("|".join(re.escape(key) for key in keys))

        def replace(match):
            key = match.group(1)
            if key in seen:
                return ""
            seen.add(key)
            return key

        result = re.sub(pattern, replace, template)
        return result

    def short_input_name_template(self, template: str, **kwargs):
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
                free_length = self.safe_max_length - (len_merged_keys + len_no_keys)
                len_file_name = len(input_file_name)
                start_index = free_length // 2
                end_index = free_length // 2.5
                if len_file_name > free_length:
                    return f"{input_file_name[:int(start_index)]}...{input_file_name[-int(end_index):]}"
                else:
                    return input_file_name
            else:
                print("Не был введен ключ NAME")
                return ""
        else:
            print("Сначала введите ключи")
            return ""
