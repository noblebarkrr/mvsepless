import os
import re

MAX_LENGTH_NAME = 255


def remove_duplicate_keys(input_str, keys=("NAME", "MODEL", "STEM", "ID")):
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


def shorter_name(template, file_name, stem, model_name, id):
    # Удаляем дубликаты ключей в шаблоне перед расчетами
    clean_template = remove_duplicate_keys(template)

    str_id = str(id)
    template_no_keys_length = len(
        clean_template.replace("NAME", "")
        .replace("MODEL", "")
        .replace("STEM", "")
        .replace("ID", "")
    )
    key_values_length = (
        len(model_name)
        if "MODEL" in clean_template
        else (
            0 + len(stem)
            if "STEM" in clean_template
            else 0 + len(str_id) if "ID" in clean_template else 0
        )
    )
    free_length = MAX_LENGTH_NAME - (template_no_keys_length + key_values_length)
    if len(file_name) > (free_length - 7):
        shorted_name = f"{file_name[:(free_length // 2)]}...{file_name[-((free_length // 2) - 7):]}"
        return shorted_name
    else:
        return file_name


def output_file_template(template, input_file_name, stem, model_name, id):
    # Удаляем дубликаты ключей перед заменой
    clean_template = remove_duplicate_keys(template)

    input_file_name = shorter_name(
        clean_template, input_file_name, stem, model_name, id
    )
    template_name = (
        clean_template.replace("MODEL", f"{model_name}")
        .replace("STEM", f"{stem}")
        .replace("ID", f"{id}")
        .replace("NAME", f"{input_file_name}")
    )
    output_name = f"{template_name}"
    return output_name


def audio_separator_rename_stems(audio, template, name_model, id):
    # Удаляем дубликаты ключей в основном шаблоне
    clean_template = remove_duplicate_keys(template)

    base_name = os.path.splitext(os.path.basename(audio))[0]
    stems = {
        "Bass": output_file_template(clean_template, base_name, "Bass", name_model, id),
        "Crowd": output_file_template(
            clean_template, base_name, "Crowd", name_model, id
        ),
        "Drums": output_file_template(
            clean_template, base_name, "Drums", name_model, id
        ),
        "Dry": output_file_template(clean_template, base_name, "Dry", name_model, id),
        "Breath": output_file_template(
            clean_template, base_name, "Breath", name_model, id
        ),
        "Echo": output_file_template(clean_template, base_name, "Echo", name_model, id),
        "Instrumental": output_file_template(
            clean_template, base_name, "Instrumental", name_model, id
        ),
        "No Bass": output_file_template(
            clean_template, base_name, "No Bass", name_model, id
        ),
        "No Crowd": output_file_template(
            clean_template, base_name, "No Crowd", name_model, id
        ),
        "No Drums": output_file_template(
            clean_template, base_name, "No Drums", name_model, id
        ),
        "No Dry": output_file_template(
            clean_template, base_name, "No Dry", name_model, id
        ),
        "No Echo": output_file_template(
            clean_template, base_name, "No Echo", name_model, id
        ),
        "No Noise": output_file_template(
            clean_template, base_name, "No Noise", name_model, id
        ),
        "No Other": output_file_template(
            clean_template, base_name, "No Other", name_model, id
        ),
        "No Breath": output_file_template(
            clean_template, base_name, "No Breath", name_model, id
        ),
        "No Reverb": output_file_template(
            clean_template, base_name, "No Reverb", name_model, id
        ),
        "No Woodwinds": output_file_template(
            clean_template, base_name, "No Woodwinds", name_model, id
        ),
        "Noise": output_file_template(
            clean_template, base_name, "Noise", name_model, id
        ),
        "Other": output_file_template(
            clean_template, base_name, "Other", name_model, id
        ),
        "Reverb": output_file_template(
            clean_template, base_name, "Reverb", name_model, id
        ),
        "Vocals": output_file_template(
            clean_template, base_name, "Vocals", name_model, id
        ),
        "Woodwinds": output_file_template(
            clean_template, base_name, "Woodwinds", name_model, id
        ),
        "Guitar": output_file_template(
            clean_template, base_name, "Guitar", name_model, id
        ),
        "Piano": output_file_template(
            clean_template, base_name, "Piano", name_model, id
        ),
    }
    return stems
