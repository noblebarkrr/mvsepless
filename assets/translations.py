TRANSLATIONS = {
    "ru": {
        "title": "Разделение музыки и вокала",
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
        "extract_instrumental": "Извлечь инструментал",
        "stems_list": "Список стемов",
        "target_instrument": "Целевой инструмент",
        "stems_info": "Выбор стемов недоступен\nДля извлечения второго стема включите \"Извлечь инструментал\"",
        "stems_info2": "Для получения остатка (при выбранных стемах), включите \"Извлечь инструментал\"",
        "add_settings": "Дополнительные настройки",
        "template": "Формат имени",
        "vr_aggr_slider": "Агрессивность",
        "template_help": """
> Формат имени результатов в мульти-инференсе.

> Доступные ключи для формата имени стемов:
> (изменить формат имени стемов можно здесь)
> * **NAME** - Имя входного файла
> * **STEM** - Название стема (например, vocals, drums, bass)
> * **MODEL** - Имя модели (например, Mel-Band-Roformer_Instrumental_FvX_gabox, UVR-MDX-NET-Inst_HQ_3)
> * **ID** - ID модели (например, BS-Roformer_SW: 217)

> Пример:
> * **Шаблон:** NAME_STEM_MODEL
> * **Результат:** test_vocals_Mel-Band-Roformer_Instrumental_FvX_gabox

<div style="color: red; font-weight: bold; background-color: #ffecec; padding: 10px; border-left: 3px solid red; margin: 10px 0;">

Используйте ТОЛЬКО указанные ключи (NAME, STEM, MODEL) во избежание повреждения файлов. 

НЕ добавляйте дополнительный текст или символы вне этих ключей, либо делайте это с осторожностью.

</div>
        """,
    },
    "en": {
        "title": "Separate music and vocals",
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
        "extract_instrumental": "Extract Instrumental",
        "stems_list": "Stems List",
        "target_instrument": "Target instrument",
        "stems_info": "Stem selection unavailable\nEnable \"Extract Instrumental\" to extract the second stem",
        "stems_info2": "To extract the residual (with selected_stems), enable \"Extract Instrumental\"",
        "add_settings": "Additional settings",
        "template": "Name format",
        "vr_aggr_slider": "Aggressive",
        "template_help": """
> The format for naming results in multi-inference.

> Available keys for stem name formatting:
> (you can change the stem name format here)
> * **NAME** - Input file name  
> * **STEM** - Stem name (e.g., vocals, drums, bass)  
> * **MODEL** - Model name (e.g., Mel-Band-Roformer_Instrumental_FvX_gabox, UVR-MDX-NET-Inst_HQ_3)  
> * **ID** - Model ID (e.g., BS-Roformer_SW: 217)

> Example:  
> * **Template:** NAME_STEM_MODEL  
> * **Result:** test_vocals_Mel-Band-Roformer_Instrumental_FvX_gabox  

<div style="color: red; font-weight: bold; background-color: #ffecec; padding: 10px; border-left: 3px solid red; margin: 10px 0;">

Use ONLY the specified keys (NAME, STEM, MODEL) to avoid file corruption.  

DO NOT add extra text or symbols outside these keys, or do so with caution.  

</div>
        """,
    }
}

TRANSLATIONS_STEMS = {
    "ru": {
        "vocals": "Вокал",
        "Vocals": "Вокал",
        "Lead": "Ведущий вокал",
        "lead": "Ведущий вокал",
        "Back": "Бэк-вокал",
        "back": "Бэк-вокал",
        "other": "Другое",
        "Other": "Другое",
        "Instrumental": "Инструментал",
        "instrumnetal": "Инструментал",
        "instrumental +": "Инструментал +",
        "instrumental -": "Инструментал -",
        "Bleed": "Фон",
        "Guitar": "Гитара",
        "drums": "Барабаны",
        "bass": "Бас",
        "karaoke": "Караоке",
        "reverb": "Реверберация",
        "noreverb": "Без реверберации",
        "aspiration": "Придыхание",
        "dry": "Сухой звук",
        "crowd": "Звуки толпы",
        "percussions": "Перкуссия",
        "piano": "Пианино",
        "guitar": "Гитара",
        "male": "Мужской",
        "female": "Женский",
        "kick": "Кик",
        "snare": "Малый барабан",
        "toms": "Том-томы",
        "hh": "Хай-хэт",
        "ride": "Райд",
        "crash": "Крэш",
        "similarity": "Фантомный центр",
        "difference": "Стерео база",
        "inst": "Инструмент",
        "orch": "Оркестр",
        "No Woodwinds": "Без деревянных духовых",
        "Woodwinds": "Деревянные духовые",
        "No Echo": "Без эха",
        "Echo": "Эхо",
        "No Reverb": "Без реверберации",
        "Reverb": "Реверберация",
        "Noise": "Шум",
        "No Noise": "Без шума",
        "Dry": "Сухой звук",
        "No Dry": "Не сухой звук",
        "Breath": "Дыхание",
        "No Breath": "Без дыхания",
        "No Crowd": "Без звуков толпы",
        "Crowd": "Звуки толпы",
        "No Other": "Без другого",
        "Bass": "Бас",
        "No Bass": "Без баса",
        "Drums": "Барабаны",
        "No Drums": "Без барабанов",
        "speech": "Речь",
        "music": "Музыка",
        "effects": "Эффекты",
        "sfx": "Звуковые эффекты",
        "inverted +": "Инверсия +",
        "inverted -": "Инверсия -",
        "Unsupported model type": "Данный тип модели не поддерживается",
        "Error": "Ошибка",
        "Model not exist": "Данной модели не существует",
        "Input file not exist": "Указанного файла не существует",
        "Input path is none": "Не указан путь к файлу"
    },
    "en": {
        "vocals": "Vocals",
        "Vocals": "Vocals",
        "Lead": "Lead vocals",
        "lead": "Lead vocals",
        "Back": "Back vocals",
        "back": "Back vocals",
        "other": "Other",
        "Other": "Other",
        "Instrumental": "Instrumental",
        "instrumnetal": "Instrumental",
        "instrumental +": "Instrumental +",
        "instrumental -": "Instrumental -",
        "Bleed": "Bleed",
        "Guitar": "Guitar",
        "drums": "Drums",
        "bass": "Bass",
        "karaoke": "Karaoke",
        "reverb": "Reverb",
        "noreverb": "No reverb",
        "aspiration": "Aspiration",
        "dry": "Dry",
        "crowd": "Crowd",
        "percussions": "Percussions",
        "piano": "Piano",
        "guitar": "Guitar",
        "male": "Male",
        "female": "Female",
        "kick": "Kick",
        "snare": "Snare",
        "toms": "Toms",
        "hh": "Hi-hat",
        "ride": "Ride",
        "crash": "Crash",
        "similarity": "Pnahtom center",
        "difference": "Stereo base",
        "inst": "Instrument",
        "orch": "Orchestra",
        "No Woodwinds": "No woodwinds",
        "Woodwinds": "Woodwinds",
        "No Echo": "No echo",
        "Echo": "Echo",
        "No Reverb": "No reverb",
        "Reverb": "Reverb",
        "Noise": "Noise",
        "No Noise": "No noise",
        "Dry": "Dry",
        "No Dry": "No dry",
        "Breath": "Breath",
        "No Breath": "No breath",
        "No Crowd": "No crowd",
        "Crowd": "Crowd",
        "No Other": "No other",
        "Bass": "Bass",
        "No Bass": "No bass",
        "Drums": "Drums",
        "No Drums": "No drums",
        "speech": "Speech",
        "music": "Music",
        "effects": "Effects",
        "sfx": "SFX",
        "inverted +": "Inverted +",
        "inverted -": "Inverted -",
        "Unsupported model type": "Current model type unsupported",
        "Error": "Error",
        "Model not exist": "Current model is not exist",
        "Input file not exist": "Current file is not exist",
        "Input path is none": "File path not specified"
    }
}
