"""Модуль интернационализации для MVSepless"""
import os
from typing import Dict, Literal, Union, List

# Тип для поддерживаемых языков
Language = Literal["ru", "en"]

# Получение языка из переменной окружения
def get_language() -> Language:
    """Определение языка из переменной окружения"""
    lang = os.environ.get("MVSEPLESS_LANGUAGE", "ru").lower()
    return "ru" if lang == "ru" else "en"

# Текущий язык
CURRENT_LANGUAGE: Language = get_language()

# Словарь с переводами
TRANSLATIONS: Dict[Language, Dict[str, str]] = {
    "ru": {
        "bytes" : "Б",
        "kbytes": "КБ",
        "mbytes": "МБ",
        "gbytes": "ГБ",
        "tbytes": "ТБ",
        "separate": "Разделить",
        "refresh": "Обновить",
        "ensemble_type": "Тип ансамбля",
        "ensemble_type_info": """<details>
<summary><b>Типы ансамбля</b></summary>

| Тип | Описание |
|-----|----------|
| `avg_fft` | Среднее, стабильный результат (требуются веса) |
| `median_fft` | Медиана (эффективно от 3+ аудио) |
| `min_fft` | Минимум, более чистый результат |
| `max_fft` | Максимум, более полный, но "грязный" результат |

</details>""",
"output_template": "Шаблон имени",
"output_template_info": """
<details>
<summary><b>Доступные ключи</b></summary>

- `NAME` — имя входного файла (без расширения)
- `STEM` — название стема (например: vocals, instrumental)
- `MODEL` — имя модели (например: bs_6stem)

Пример: `NAME_STEM_MODEL` → `Song_vocals_bs_6stem`

</details>""",
        "output_etemplate_info": """<details>
<summary><b>Доступные ключи</b></summary>

- `NAME` — имя входного файла (без расшириения)
- `TYPE` — тип ансамбля (например: min_fft, avg_fft и др.)
- `COUNT` — количество используемых моделей/файлов

Пример: `NAME_COUNT_TYPE` → `Song_7_min_fft`

</details>""",
        "output_metemplate_info": """<details>
<summary><b>Доступные ключи</b></summary>

- `NAME` — имя входного файла (без расшириения)
- `TYPE` — тип ансамбля (напрмиер: min_fft, avg_fft и др.)

Пример: `NAME_TYPE` → `Song_min_fft`

</details>""",
        "output_itemplate_info": """<details>
<summary><b>Доступные ключи</b></summary>

- `NAME` — имя входного файла (без расшириения)
- `TYPE` — тип инверсии (например: waveform, spectrogram)

Пример: `NAME_TYPE` → `Song_waveform`

</details>""",
        "output_name": "Имя выходного файла",
        "output_file": "Выходной файл",
        "output_directory": "Выходная директория",
        "output_format": "Формат вывода",
        "unload_model": "Выгрузить модель из памяти",
        "extract_instrumental": "Извлечь инструментал",
        "use_spec_invert": "Использовать спектрограмму при создании инверсии",
        "select_stems": "Выберите выходные стемы",
        "select_stems_info": "(Если не выбрано ни одного стема, по умолчанию будут сохранены все стемы)",
        "show_preview": "Предпросмотр",
        "model_name": "Имя модели",
        "vr": "VR",
        "mdx": "MDX",
        "mdxc": "MDXC",
        "mvox": "Medley-Vox",
        "demucs": "Demucs",
        "separation_params": "Параметры разделения",
        "separation_window_size_info": "Влияет на качество разделения\nЧем ниже значение, тем выше качество",
        "separation_hop_info": "Точность анализа аудио",
        "separation_aggresion_info": "Влияет на то, насколько глубоко будет извлечен стем",
        "separation_hi-end_process_info": "Зеркальное восстановление недостающих высоких частот",
        "separation_overlap_info": "Определяет насколько плавными будут переходы между фрагментами\nЧем выше это значение, там выше качество разделения, но обработка будет медленнее",
        "separation_segment_size_info": "Влияет на качество результатов разделения\n(Слишком высокие значения могут привести к переполнению ОЗУ и видеопамяти)",
        "separation_batch_size_info": "Количество фрагментов обрабатываемых одновременно\nЧем выше это значение, тем больше ресурсов будет использовано при разделении\n(Слишком высокие значения могут привести к переполнению ОЗУ и видеопамяти)",
        "mdxc_segment_size": "Размер сегмента (dim_t)", 
        "mdxc_batch_size": "Размер батча", 
        "mdxc_overlap": "Перекрытие", 
        "mdxc_denoise": "Включить дейнозинг", 
        "mdxc_override_segment": "Переопределить размер сегмента", 
        "demucs_segment": "Размер сегмента (сек)", 
        "demucs_batch_size": "Размер батча", 
        "demucs_overlap": "Перекрытие", 
        "demucs_denoise": "Включить дейнозинг",
        "demucs_override_segment": "Переопределить размер сегмента",
        "mdx_hop_length": "Длина шага", 
        "mdx_segment_size": "Размер сегмента (dim_t)", 
        "mdx_batch_size": "Размер батча", 
        "mdx_overlap": "Перекрытие", 
        "mdx_denoise": "Включить дейнозинг", 
        "mdx_override_segment": "Переопределить размер сегмента", 
        "vr_window_size": "Размер окна", 
        "vr_batch_size": "Размер батча", 
        "vr_aggression": "Агрессивность", 
        "vr_post_process": "Включить постобработку", 
        "vr_post_process_threshold": "Порог постобработки", 
        "vr_high_end_process": "Восстановление высоких частот", 
        "mvox_segment": "Размер сегмента (сек)", 
        "mvox_overlap": "Перекрытие", 
        "mvox_override_segment": "Переопределить размер сегмента",
        "unknown_model_type": "Неизвестный тип модели: {model_type}",
        "config_loaded": "Загружен конфиг",
        "checkpoint_loaded": "Загружен чекпоинт",
        "hubert_checkpoint_loaded": "Загружен эмбеддер",
        "load_state_dict_error": "Ошибка при загрузке state_dict: {error}",
        "load_checkpoint_error": "Ошибка при загрузке чекпоинта: {error}",
        "config_not_found": "Конфиг не найден: {path}",
        "config_load_error": "Ошибка при загрузке конфига: {error}",
        "config_is_not_loaded": "Конфиг не был загружен\nСначала загрузите конфиг",
        "unknown_model_type": "Неизвестный тип модели: {model_type}",
        "freed_ram": "Освобождено RAM",
        "emeergency_ram": "Экстренно освобождено",
        "deleted_stems": "Удалены невыбранные стемы",
        "added_second_stem": "Добавлен второй стем",
        "writing": "Запись в файл",
        "format_return": "Формат возвращения",
        "denoise": "[денойз]",
        "mix_not_found": "Микс не найден\nСначала загрузите микс",
        "mix_is_empty": "Микс является пустым",
        "model_not_loaded": "Модель не была загружена\nСначала загрузите модель",
        "demix_error": "Ошибка при демиксе: {error}",
        "name_stems_list": "[имя файла, [[стем1, путь_к_стему1], [стем2, путь_к_стему2]]]",
        "stems_list": "[[стем1, путь_к_стему1], [стем2, путь_к_стему2]]",
        "stems_list_append_self": "[[стем1, путь_к_стему1], [стем2, путь_к_стему2]]",
        "name_stems_list_append_self": "[имя файла, [[стем1, путь_к_стему1], [стем2, путь_к_стему2]]]",
        "processing": "Обработка",
        "loaded_mix": "Загружен микс",
        "array_shape" : "Форма массива",
        "reuse_btn": "Использовать снова",
        "bands": "полос",
        "patches": "патчей",
        "chunks": "чанков",
        "files": "файлов",
        "samples": "сэмплов",
        "path_not_specified": "Сначала укажите путь к файлу",
        "path_not_exist": "Данного файла не существует",
        "file_is_not_audio": "Это не аудио-файл",
        "sr_required": "Не указана частота дискретизации\nУкажите частоту дискретизации в параметре sr",
        "ffmpeg_error": "Ошибка FFMPEG: {error}",
        "ffmpeg_exit_code": "Код выхода: {code}",
        "write_critical_error": "Критическая ошибка при записи: {error}",
        "write_error": "Ошибка при записи: {error}",
        "no_files_written": "Ни один из аудио-массивов не был записан\nОшибки:\n{errors}",
        "concatenate_complete": "Склейка завершена",
        "ensemble_complete": "Ансамбль успешно создан",
        "unknown_etype": "Неизвестный тип ансамбля: {alg}",
        "subtract_spectrogram": "Вычитание из спектрограммы",
        "subtract_phase": "Вычитание противофазой",
        "arrays": "аудио-массивы",
        "extending_progress": "Удлинение аудио",
        "arrays_srs_mismatch": "Должно быть одинаковое количество массивов и частот дискретизации",
        "fitting_progress": "Подгонка аудио",
        "unknown_shape": "Неизвестная форма массива",
        "unknown_var": "Неизвестный вариант: {var}",
        "mid_side_var0": "Вычитание сайд-канала из оригинала",
        "mid_side_var1": "Вычитание моно-канала из оригинала",
        "mid_side_var2": "Вычитание фантомного центра из оригинала",
        "mid_side_var3": "Вычитание фантомного центра из оригинала (Audacity-like)",
        "mid_side_var4": "Вычитание правого канала из левого",
        "unexpected_min_val": "Неожиданное минимальное значение: {value}",
        "array_dim_error": "Неподдерживаемое колчиество осей: {axis}\nПоддерживаются только 1D и 2D массивы",
        "invalid_bitrate": "Недействительный битрейт: {bitrate}",
        "multi_reading": "Мульти-чтение",
        "channels_read_error": "Ошибка при получении количества каналов: {path}",
        "sr_read_error": "Ошибка при получении частоты дисректизации: {path}",
        "ffmpeg_found": "FFMPEG найден",
        "ffmpeg_not_found": "FFMPEG не найден",
        "ffprobe_found": "FFPROBE найден",
        "ffprobe_not_found": "FFPROBE не найден",
        "history": "История",
        "history_select_info": "При выборе значения в данном списке будут показаны плееры со стемами из выбранного прошлого разделения",
        "last_separations": "Последние разделения",
        "not_last_separation": "Сначала впервые разделите хотя бы одно аудио",
        "separation_tab": "Разделение",
        "ensemble_tab": "Ансамбль",
        "auto_ensemble_tab": "Авто",
        "man_ensemble_tab": "Вручную",
        "history_loaded": "История загружена",
        "input_base_loaded": "Список входных файлов загружен",
        "input_base_cleared": "Список входных файлов был очищен",
        "primary_stem": "Основной стем",
        "invert": "Инверсия",
        "weights": "Веса",
        "delete": "Удалить",
        "replace": "Заменить",
        "add_model": "Добавить модель",
        "insert": "Вставить",
        "ae_added_model": "Добавлена модель",
        "ae_deleted_model": "Удалена модель",
        "ae_replaced_model": "Заменена модель",
        "ae_inserted_model": "Вставлена модель",
        "clear": "Очистить",
        "ae_all_cleared": "Все модели очищены",
        "model_index": "Индекс модели",
        "ensemble_result": "Результат ансамбля",
        "saved_primary_stems": "Сохраненные основные стемы",
        "enable_save_primary_stems": "Включить сохранение основных стемов",
        "inverted_result": "Инверсия",
        "error_occured_separation": "Произошла ошибка при разделении\nМодель будет пропущена",
        "reuse_output_btn": "Использовать результат снова",
        "reuse_invert_btn": "Использовать инверсию снова",
        "run_ensemble": "Создать ансамбль",
        "ensemble_models_count": "Моделей",
        "ensemble_settings": "Настройки ансамбля",
        "from_array": "из аудио-массива",
        "not_separated": "Сначала разделите аудио",
        "ensemble_flow_saved": "Пресет ансамбля сохранен",
        "ensemble_flow_loaded": "Пресет ансамбля загружен",
        "preset_name": "Имя пресета",
        "load": "Загрузить",
        "save": "Сохранить",
        "import": "Импорт",
        "export": "Экспорт",
        "weights_only_for_avg_fft": "Веса (обязательно только для типа avg_fft)",
        "presets": "Пресеты",
        "not_ensembled_with_primary_stems": "При создании ансамбля включите сохранение основных стемов, чтобы их увидеть после создания ансамбля",
        "paths_not_specified": "Сначала укажите пути к файлам",
        "weights_split": "Разделение через запятую (например: 1.0,0.3,0.2)",
        "paths_not_exist": "Данных файлов не сущестует",
        "files_is_not_audio": "Это не аудио-файлы",
        "original": "Оригинал",
        "stem": "Стем",
        "subtract_tab": "Вычитание",
        "multi_writing": "Мульти-запись",
        "ensemble_preset_settings": "Пресет",
        "auto_ensemble_name_preset": "Имя пресета",
        "ensemble_flow_not_exist": "Данного пресета ансамбля не существует",
        "name_not_specified": "Сначала укажите имя",
        "subtract": "Вычесть",
        "index_rate": "Влияние индекса",
        "index_rate_info": "Чем ниже значение, тем больше голос похож на исходный; чем выше, тем ближе к модели",
        "volume_envelope": "Соотношение огибающих громкости",
        "volume_envelope_info": "Заменить или смешать с огибающей громкости вывода\nЧем ближе значения к 1, тем больше используется огибающая вывода",
        "protect": "Защита согласных",
        "protect_info": "Предовращает роботизацию дыхания и согласных (Может влиять на четкость речи)\nЗначение 0.5 обеспечивает полную защиту\n",
        "crepe_hop_length": "Длина шага",
        "crepe_hop_length_info": "Точность извлечения F0\nЧем меньше значение, чем точнее будет извлечен F0",
        "vbach_embedder": "Эмбеддер",
        "vbach_embedder_info": "Модель, используемая для анализа содержания звуков",
        "vbach_embedder_not": "Данного эмбеддера не существует",
        "stereo_mode": "Стерео-режим",
        "stereo_mode_info": """<details>
<summary><b>Стерео-режимы</b></summary>

| Режим | Описание |
|-------|----------|
| `mono` | Преобразование моно-сигнала (по умолчанию) |
| `left/right` | Обработка левого и правого канала независимо |
| `sim/dif` | Обработка фантомного центра и стерео-базы независимо |

</details>""",
        "chunk_duration": "Длина чанка (сек)",
        "vbach_use_transformers": "Использовать стек Transformers",
        "pitch": "Высота тона (полутона)",
        "speaker_id": "ID спикера",
        "f0_method": "Метод извлечения F0",
        "using_vocoder": "Используется вокодер",
        "vbach_tab": "Преобразование вокала",
        "convert": "Конвертировать",
        "advanced_params": "Расширенные параметры",
        "model_unloaded": "Модель выгружена из памяти",
        "model_not_selected": "Сначала выберите путь к модели",
        "no_conversion_results": "Сначала выполните преобразование",
        "output_vbach_template_info": """<details>
<summary><b>Доступные ключи</b></summary>

- `NAME` - имя входного файла (без расширения)
- `F0METHOD` - метод извлечения F0 (например: rmvpe+, fcpe)
- `MODEL` — имя модели (например: rvc_test)
- `PITCH` - изменение высоты тона (например: 0, 12)

Пример: `NAME_F0METHOD_PITCH` → `Song_rmvpe+_0`
        
</details>""",
        "model_path": "Путь к файлу модели",
        "index_path": "Путь к файлу индекса",
        "extracting_f0": "Извлечение F0...",
        "extracting_f0_success": "Извлечение F0 завершено",
        "reading_faiss_index_error": "Ошибка при чтении FAISS индекса: {error}",
        "mix_complete": "Микс создан",
        "download_attempt_failed": "Попытка {attempt}/{retries} не удалась. Ошибка: {error}",
        "all_download_attempts_failed": "Все попытки загрузки завершились неудачно",
        "retrying": "Повторная попытка...",
        "unknown_f0_method": "Неизвестный метод извлечения F0: {method}",
        "models": "моделей",
        "f0_min": "Минимальная частота F0",
        "f0_max": "Максимальная частота F0",
        "download_from_internet": "Загрузить c интернета",
        "download_from_local_device": "Загрузить с устройства",
        "supported_only_direct_links": "Поддерживаются только прямые ссылки",
        "download_model_files_from_zip": "Загрузить файлы модели из ZIP-архива",
        "download_model_files": "Загрузить файлы модели отдельно",
        "vbach_checkpoint_pth_placeholder": "Чекпоинт (*.pth)",
        "vbach_index_file_placeholder": "Индекс-файл (*.index, необязательно)",
        "vbach_zip_placeholder": "Архив (*.zip)",
        "vbach_zip_link": "Ссылка на ZIP-архив",
        "vbach_pth_link": "Ссылка на чекпоинт",
        "vbach_index_link": "Ссылка на индекс-файл",
        "download_and_unzip": "Скачать и распаковать",
        "download_and_move_to_models_dir": "Скачать и перенести в папку с моделями",
        "vbach_models_tab": "Модели для преобразования вокала",
        "download_model": "Загрузить модель",
        "vbach_model_zip_unpacked": "ZIP-архив распакован",
        "vbach_model_zip_not_model_files": "В ZIP-архиве не обнаружено файлов модели",
        "vbach_model_pths_uploaded": "Чекпоинты загружены",
        "vbach_model_indexes_uploaded": "Индекс-файлы загружены",
        "vbach_model_pth_uploaded": "Чекпоинт загружен",
        "vbach_model_index_uploaded": "Индекс-файл загружен",
        "vbach_model_pth_downloaded": "Чекпоинт скачан",
        "vbach_model_index_downloaded": "Индекс-файл скачан",
        "vbach_added_pths": "Добавлено чекпоинтов",
        "vbach_added_indexes": "Добавлено индекс-файлов",
        "model_downloaded": "Модель скачана",
        "model_already_downloaded": "Модель уже скачана",
        "download": "Скачать",
        "f0_curve_not_found": "В указанном файле нет кривой F0",
        "importing_f0": "Импортирование F0",
        "importing_f0_success": "Импортирование F0 завершено",
        "custom_f0": "Кастомный F0",
        "vbach_result": "Результат преобразования",
        "f0_extraction_tab": "Извлечение F0",
        "extract_f0": "Извлечь F0",
        "output_path": "Путь для сохранения",
        "f0_output_path_info": "Оставьте пустым для автоматического сохранения рядом с аудиофайлом",
        "f0_output_path_placeholder": "Например: /path/to/output.json",
        "f0_extraction_results": "Результаты извлечения F0",
        "f0_extracted_success": "F0 успешно извлечен",
        "no_f0_extracted": "F0 еще не извлечен",
        "download_f0_json": "Скачать JSON с F0",
        "no_audio_selected": "Сначала выберите аудиофайл",
        "f0_extraction_complete": "Извлечение F0 завершено",
        "f0_json_file": "JSON файл с F0 кривой",
        "convert_custom_f0": "Конвертировать с кастомным F0",
        "no_f0_file_selected": "Сначала выберите JSON файл с F0 кривой",
        "conversion_results": "Результаты конвертации",
        "download_audio": "Скачать аудио",
        "conversion_complete": "Конвертация завершена",
        "output_vbach_custom_template_info": """<details>
<summary><b>Доступные ключи</b></summary>

- `NAME` - имя входного файла (без расширения)
- `F0METHOD` - метод извлечения F0 (всегда custom)
- `MODEL` — имя модели (например: rvc_test)
- `PITCH` - изменение высоты тона (например: 0, 12)

Пример: `NAME_F0METHOD_PITCH` → `Song_rmvpe+_0`
        
</details>""",
        "status": "Статус",
        "inference": "Инференс",
        "vbach_inference_custom_f0": "Инференс с кастомным F0",
        "model_info_updated": "Информация о моделях обновлена\nЧтобы изменения вступили в силу, перезапустите запущенное приложение",
        "model_info": "Информация о моделях",
        "output_stems": "Выходные стемы",
        "table_model_info_installed_legend": "[green]установлено[/]",
        "table_model_info_target_instrument_legend": "[green]целевой инструмент[/]",
        "na": "н/д",
        "supported_yt_dlp_info": "Поддерживаются YouTube, SoundCloud, TikTok и многие другие сайты (через yt-dlp)",
        "downloaded_and_uploaded": "Скачано и загружено файлов: {count}",
        "download_failed_no_file": "Не удалось скачать аудио. Проверьте ссылку.",
        "bitrate": "Битрейт",
        "cookie_settings": "Настройки Cookie (для обхода ограничений)",
        "cookie_explanation": """<details>
        <summary><b>📖 Зачем нужны cookie-файлы?</b></summary>
        <br>
        <b>Cookie-файлы необходимы в следующих случаях:</b>
        <ul>
        <li><b>Возрастные ограничения:</b> YouTube требует авторизации для доступа к видео 18+</li>
        <li><b>Региональные блокировки:</b> Некоторые видео недоступны в вашей стране без авторизации</li>
        <li><b>Премиум-контент:</b> Некоторые платформы требуют подписку</li>
        <li><b>Антибот-защита:</b> Некоторые сайты блокируют массовые запросы</li>
        <li><b>Приватные видео:</b> Доступ к видео, доступным только по ссылке с авторизацией</li>
        </ul>

        <b>Как получить cookie-файл:</b>
        <ol>
        <li>Установите расширение для браузера (например, Get cookies.txt LOCALLY для Chrome/Firefox)</li>
        <li>Авторизуйтесь на нужном сайте (YouTube, SoundCloud и т.д.)</li>
        <li>Нажмите на иконку расширения и экспортируйте cookies в формате Netscape</li>
        <li>Загрузите полученный .txt файл в поле выше</li>
        </ol>

        ⚠️ <b>Важно:</b> Cookie-файлы содержат данные вашей сессии. Не передавайте их третьим лицам и удалите после использования.
        </details>""",
        "cookie_file": "Cookie-файл (Netscape format)",
        "cookie_status": "Статус cookie",
        "cookie_loaded": "✅ Cookie загружен: {path}",
        "cookie_not_loaded": "❌ Cookie не загружен (доступ к ограниченному контенту может быть невозможен)",
        "custom_separation_tab": "Разделение с кастомной моделью",
        "model_type": "Тип модели",
        "checkpoint_path": "Путь к чекпоинту",
        "config_path": "Путь к конфигу",
        "upload_audio": "Загрузить аудио",
        "upload_from_zip": "Загрузить из ZIP-архива",
        "upload_zip_placeholder": "",
        "uploaded_files_count": "Загруженных файлов: {count}",
        "extract_and_upload": "Загрузить и распаковать",
        "audio_url": "Ссылка на аудио",
        "download_and_upload": "Загрузить",
        "upload_from_path": "Загрузить по пути",
        "folder_path": "Путь к файлу или папке",
        "scan_and_upload": "Сканировать",
        "custom_separation_models_tab": "Кастомные модели для разделения",
        "custom_checkpoint_link": "Ссылка на чекпоинт",
        "custom_config_link": "Ссылка на конфиг",
        "custom_checkpoint_placeholder": "Чекпоинт (*.pth)",
        "custom_config_placeholder": "Конфиг (*.yaml)",
        "custom_added_configs": "Добавлено конфигов",
        "custom_added_checkpoints": "Добавлено чекпоинтов",
        "custom_model_config_downloaded": "Конфиг загружен",
        "custom_model_checkpoint_downloaded": "Чекпоинт загружен",
        "upload_from_files": "Загрузить файлы",
        "upload": "Загрузить",
        "upload_from_url": "Загрузить по ссылке",
        "download_error": "Ошибка при скачивании файла: {error}",
        "download_start": "Начало загрузки, размер:",
        "download_complete": "Загрузка завершена",
        "flows_imported": "Пресетов импортировано",
        "import": "Импорт",
        "export": "Экспорт",
        "all_ensemble_flow_cleared": "Все пресеты ансамбля удалены",
        "f0_file_info": """<details><summary><b>Формат данных в JSON-файле кривой F0:</b></summary>

```python
{
    "method": "rmvpe+", # Метод извлечения F0
    "sample_rate": 16000, # Частота дискретизации
    "window": 160, # Размер окна в сэмплах
    "p_len": 500, # ожидаемое количество кадров
    "freqs": [0, 0, 120, 121.3, ...] # Список с частотами
}
```

</details>""",
        "arg_main_description": "MVSepless - инструмент для разделения музыкальных источников (Source Separation) и создания ансамблей из нескольких моделей.",
        "arg_main_epilog": "Поддерживаются модели VR, MDX, MDXC, Demucs, Medley-Vox. Документация: https://github.com/noblebarkrr/mvsepless/tree/dzeta",
        "arg_subcommands_title": "subcommands",
        "arg_subcommands_description": "Доступные режимы работы",
        "arg_subcommands_help": "Выберите действие",
        "arg_separate_help": "Разделение аудио на стемы с использованием встроенной модели",
        "arg_separate_description": "Разделение аудиофайлов на стемы (вокал, инструментал, барабаны и т.д.) с помощью предустановленных моделей.",
        "arg_separate_epilog": "Пример: python inference.py separate -i audio.mp3 -o output -of mp3 -mn bs_6stem -tm NAME_STEM",
        "arg_custom_separate_help": "Разделение аудио с кастомной моделью (свои чекпоинт и конфиг)",
        "arg_custom_separate_description": "Разделение аудиофайлов с использованием собственной модели, загруженной из чекпоинта и конфигурационного файла.",
        "arg_custom_separate_epilog": "Пример: python inference.py custom_separate -i audio.mp3 -o output -ckpt model.ckpt -conf config.yaml -mt bs_roformer",
        "arg_info_help": "Информация о доступных моделях",
        "arg_info_description": "Показать список доступных моделей для разделения с возможностью фильтрации по стему, лимитирования вывода и обновления кэша.",
        "arg_info_epilog": "Пример: python inference.py info -limit 10 -stem vocals -oi",
        "arg_auto_ensemble_help": "Автоматический ансамбль из нескольких моделей",
        "arg_auto_ensemble_description": "Создание ансамбля результатов нескольких моделей разделения с автоматической загрузкой моделей по пресету.",
        "arg_auto_ensemble_epilog": "Пример: python inference.py auto_ensemble -i audio.mp3 -flow bs_6stem:vocals:True:1 mbr_inst1e_unwa:other:False:1 -type avg_fft",
        "arg_manual_ensemble_help": "Ручной ансамбль из готовых аудиофайлов",
        "arg_manual_ensemble_description": "Создание ансамбля из уже готовых аудиофайлов (результатов разделения) с возможностью указания весов.",
        "arg_manual_ensemble_epilog": "Пример: python inference.py manual_ensemble -i audio1.wav audio2.wav -o output -w 1.0 0.5 -type median_fft",
        "arg_subtract_help": "Вычитание стема из оригинала (создание инструментала)",
        "arg_subtract_description": "Вычитание одного аудиофайла (стема) из другого (оригинала) для получения инструментальной версии или других комбинаций.",
        "arg_subtract_epilog": "Пример: python inference.py subtract -i1 original.mp3 -i2 vocals.mp3 -o output -ispec",
        "arg_input_help": "Путь к входному аудиофайлу, папке с файлами или список путей",
        "arg_input_single_help": "Путь к входному аудиофайлу",
        "arg_output_dir_help": "Директория для сохранения результатов (по умолчанию: текущая папка)",
        "arg_output_format_help": "Формат выходного аудио. Доступны: {formats} (по умолчанию: {default})",
        "arg_template_help": "Шаблон имени выходного файла. Доступные ключи: {keys}. Пример: {example}",
        "arg_extract_instrumental_help": "Создать инверсию выбранных стемов (инструментал) - стем с именем 'invert'",
        "arg_use_spec_invert_help": "Использовать вычитание из спектрограммы вместо противофазы при создании инверсии",
        "arg_selected_stems_help": "Список стемов для сохранения (например: vocals drums). Если не указаны - сохраняются все стемы",
        "arg_model_name_help": "Имя модели для разделения (по умолчанию: bs_6stem). Полный список: python inference.py info",
        "arg_model_type_help": "Тип модели (например: bs_roformer, demucs, mdx). По умолчанию: bs_roformer",
        "arg_checkpoint_path_help": "Путь к файлу чекпоинта модели (*.ckpt или *.pth)",
        "arg_config_path_help": "Путь к конфигурационному файлу модели (*.yaml)",
        "arg_ensemble_type_help": "Тип ансамбля: avg_fft (среднее, требуются веса), median_fft (медиана, от 3+ аудио), min_fft (минимум), max_fft (максимум). По умолчанию: avg_fft",
        "arg_save_primary_stems_help": "Сохранять основные стемы в выходную директорию",
        "arg_flow_help": "Пресет в виде строк: МОДЕЛЬ:ОСНОВНОЙ_СТЕМ:ИНВЕРСИЯ:ВЕС. Пример: bs_6stem:vocals:True:1 mbr_inst1e_unwa:other:False:1",
        "arg_preset_json_help": "Путь к JSON-файлу с пресетом ансамбля",
        "arg_weights_help": "Веса для каждого аудиофайла (обязательно для типа avg_fft). Пример: -w 1.0 0.5 0.2",
        "arg_input1_help": "Путь к оригинальному аудиофайлу (из которого вычитаем)",
        "arg_input2_help": "Путь к аудиофайлу стема (который вычитаем)",
        "arg_update_help": "Обновить кэш информации о моделях из репозитория",
        "arg_clear_cache_help": "Очистить кэш информации о моделях",
        "arg_download_help": "Скачать указанную модель (требуется -mn)",
        "arg_limit_help": "Лимит количества отображаемых моделей (0 или None - без лимита)",
        "arg_stem_filter_help": "Показать только модели, которые содержат указанный стем (например: vocals, drums)",
        "arg_only_installed_help": "Показывать только установленные (скачанные) модели",
        "arg_add_param_help": "Дополнительный параметр разделения",
        "vbach_main_description": "VBach - форк PolGen для преобразования вокала с изменением высоты тона и тембра",
        "vbach_main_epilog": "Поддерживаются методы извлечения F0: rmvpe+, hpa-rmvpe, fcpe, mangio-crepe, mangio-crepe-tiny, harvest, pm, pyin. Документация: vbach_lib/README.md",
        "vbach_infer_help": "Преобразование вокала (изменение голоса) с использованием модели",
        "vbach_infer_description": "Преобразование вокала с помощью модели RVC/PollGen: изменение высоты тона, тембра, применение индекс-файла для улучшения качества.",
        "vbach_infer_epilog": "Пример: python infer.py infer -i audio.mp3 -o output -m model.pth -p 2 -f0m rmvpe+",
        "vbach_infer_custom_f0_help": "Преобразование вокала с использованием кастомной F0 кривой из JSON-файла",
        "vbach_infer_custom_f0_description": "Преобразование вокала с предварительно извлеченной и сохраненной кривой F0 (позволяет точно контролировать мелодию).",
        "vbach_infer_custom_f0_epilog": "Пример: python infer.py infer_custom_f0 -i audio.mp3 -o output -m model.pth -f0f f0.json -p 0",
        "vbach_download_hubert_help": "Скачать модель эмбеддера (HuBERT) для преобразования вокала",
        "vbach_download_hubert_description": "Загрузка модели эмбеддера HuBERT (Fairseq или Transformers версии), необходимой для работы преобразования вокала.",
        "vbach_download_hubert_epilog": "Пример: python infer.py download_hubert -emb hubert_base -tf",
        "vbach_model_path_help": "Путь к файлу чекпоинта модели (*.pth)",
        "vbach_index_path_help": "Путь к файлу индекса FAISS (*.index) для улучшения качества (опционально)",
        "vbach_pitch_help": "Изменение высоты тона в полутонах (положительные значения - выше, отрицательные - ниже). По умолчанию: 0",
        "vbach_f0_method_help": "Метод извлечения F0. Доступны: rmvpe+, hpa-rmvpe, fcpe, mangio-crepe, mangio-crepe-tiny, harvest, pm, pyin. По умолчанию: rmvpe+",
        "vbach_index_rate_help": "Влияние индекса (0-1). Чем ниже значение, тем больше голос похож на исходный; чем выше, тем ближе к модели. По умолчанию: 0.75",
        "vbach_volume_envelope_help": "Соотношение огибающих громкости (0-1). Заменяет или смешивает с огибающей громкости вывода. По умолчанию: 0.25",
        "vbach_protect_help": "Защита согласных (0-0.5). Предотвращает роботизацию дыхания и согласных. По умолчанию: 0.33",
        "vbach_hop_length_help": "Длина шага для извлечения F0 методами CREPE. Чем меньше значение, тем точнее. По умолчанию: 128",
        "vbach_embedder_help": "Модель эмбеддера: hubert_base или spin. По умолчанию: hubert_base",
        "vbach_use_transformers_help": "Использовать стек Transformers для эмбеддера (вместо Fairseq)",
        "vbach_stereo_mode_help": "Стерео-режим: mono (преобразование моно-сигнала), left/right (обработка каналов независимо), sim/dif (обработка центра и стереобазы). По умолчанию: mono",
        "vbach_f0_min_help": "Минимальная частота F0 в Гц. По умолчанию: 50",
        "vbach_f0_max_help": "Максимальная частота F0 в Гц. По умолчанию: 1100",
        "vbach_chunk_duration_help": "Длина чанка для обработки в секундах. По умолчанию: 7",
        "vbach_f0_file_help": "Путь к JSON-файлу с кастомной кривой F0 (полученному через f0_extractor.py)",
        "f0_extract_description": "Извлечение кривой F0 (основной частоты тона) из аудиофайла для последующего использования в преобразовании вокала",
        "f0_extract_epilog": "Пример: python f0_extractor.py -i audio.mp3 -f0m rmvpe+ -f0min 50 -f0max 1100 -o f0.json",
        "f0_extract_output_help": "Путь для сохранения JSON-файла с F0 кривой. Если не указан, сохраняется рядом с аудиофайлом",
        "app_description": "MVSepless Web-UI - графический интерфейс для разделения музыки и преобразования вокала",
        "app_epilog": "Пример: python app.py --share --full --port 7860",
        "app_share_help": "Создать публичную ссылку через Gradio Share (для доступа из интернета)",
        "app_port_help": "Порт для запуска сервера (по умолчанию: 7860)",
        "app_full_help": "Запустить полную версию интерфейса (не режим Hugging Face Spaces)",
        "template_keys_separate": "NAME (имя файла без расширения), STEM (название стема), MODEL (имя модели)",
        "template_keys_auto_ensemble": "NAME (имя файла), TYPE (тип ансамбля), COUNT (количество моделей)",
        "template_keys_manual_ensemble": "NAME (имя первого файла), TYPE (тип ансамбля)",
        "template_keys_subtract": "NAME (имя оригинального файла), TYPE (тип инверсии: waveform или spectrogram)",
        "template_keys_vbach": "NAME (имя файла), F0METHOD (метод F0), PITCH (изменение тона)",
        "stems": "Стемы",
        "target_instrument": "Целевой инструмент",
        "yes": "Да",
        "no": "Нет",
        "zerogpu=true": "Среда выполнения - ZeroGPU",
        "ensemble_processing": "Создание ансамбля",
        "tracks": "треков",
        "app_user_dir_help": "Путь к директории для хранения пользовательских файлов",
        "gdrive_mount_found": "Обнаружен привязанный Google Диск",
        "copy_to_gdrive": "Копирование данных на Google Диск",
        "dirs": "директорий",
        "copy_to_gdrive_done": "Копирование завершено",
        "copied_dirs": "Скопировано директорий",
        "copy_from_current_user_dir_to_gdrive": "Копировать все пользовательские данные на Google Диск",
        "google_drive": "Google Диск",
        "copy_from_gdrive_to_current_user_dir": "Копировать все пользовательские данные с Google Диска в среду выполнения",
        "copy_to_current_user_dir": "Копирование данных в среду выполнения",
        "free_space": "Свободно",
        "used_space": "Использовано",
        "all_space": "Всего",
        "used_space_data_local": "Обьем пользовательских данных в среде выполнения",
        "used_space_data_gdrive": "Обьем пользовательских данных на Google Диске",
        "added_files": "Добавлено файлов",
        "reuse_all_stem": "Использовать снова все {stem}",
        "reuse_all_stems": "Использовать снова все стемы",
        "generate_zip_archive": "Сгенерировать ZIP-архив",
        "download_zip_archive": "Скачать ZIP-архив",
        "invert_plus": "При создании инструментала складывать все невыбранные стемы",
        "invert_plus_info": "Уменьшает остаток от выбранных стемов при извлечении инструментала (актуально для Roformer моделей)",
        "invert_plus_applied": "Применена инверсия \"плюс\"",
        "flow_empty": "Добавьте хотя бы одну модель в пресет",
        "iteration": "Итерация",
        "num_iters": "Количество итераций",
        "no_models_succeeded": "Ни одна модель не была использована",
        "saved_file": "Сохранен файл",
        "arg_iterative_ensemble_help": "Итеративный ансамбль для последовательного улучшения разделения",
        "arg_iterative_ensemble_description": "Последовательное применение набора моделей для постепенного улучшения качества разделения (извлечение остатков с каждой итерацией).",
        "arg_iterative_ensemble_epilog": "Пример: python inference.py iterative_ensemble -i audio.mp3 -o output -flow bs_6stem:vocals:True mbr_inst1e_unwa:other:False -n 4 -save_intermediate",
        "arg_iterative_flow_help": "Пресет в виде строк: МОДЕЛЬ:ОСНОВНОЙ_СТЕМ:ИНВЕРСИЯ. Пример: bs_6stem:vocals:True mbr_inst1e_unwa:other:False",
        "arg_num_iters_help": "Количество итераций (по умолчанию: 4). На каждой итерации применяется весь набор моделей к остатку от предыдущей итерации",
        "arg_save_intermediate_help": "Сохранять промежуточные результаты каждой итерации",
        "template_keys_iterative_ensemble": "NAME (имя файла), ITER (номер итерации)",
        "saved_intermediate_files": "Сохранены промежуточные файлы",
        "iterative_ensemble_name_preset": "Имя пресета итеративного ансамбля",
        "run_iterative_ensemble": "Запустить итеративный ансамбль",
        "save_intermediate": "Сохранять промежуточные результаты",
        "intermediate_results": "Промежуточные результаты",
        "no_intermediate_results": "Нет промежуточных результатов",
        "output_iterative_template_info": """<details><summary><b>Доступные ключи</b></summary>

- `NAME` — имя входного файла (без расширения)
- `ITER` — номер итерации (iter_N)

Пример: `NAME_ITER` → `Song_iter_3`

</details>""",
        "iterative_ensemble_tab": "Итеративный",
        "model": "Модель",
        "selected_stems": "Выбранные стемы",
        "corrected_selected_stems": "Существующие выбранные стемы",
        "uncorrected_selected_stems": "Внимание! Не существующие стемы",
        "prefer_float": "Предочитать высокую точность вывода",
        "current_device": "Текущее устройство: {device}",
        "no_has_taglib": "Зависимость \'pytaglib\' не установлена\nВыходные аудио-файлы будут записаны без метаданных",
        "write_metadata": "Запись метаданных",
        "write_metadata_info": "Добавляет информацию о обработке в метаданные файла\nЕсли в оригинальном аудио есть метаданные, то эта информация записывается вместе с оригинальной в метаданных",
        "write_metadata_error": "Ошибка при записи метаданных: {error}",
        "off_audio_players_output": "Отключить аудиоплееры на выводе",
        "off_audio_players_output_info": "Отключает аудиоплееры на выводе, заменяя их на простые кнопки скачивания, что может ускорить работу интерфейса при большом количестве выходных файлов",
        "add_uploaded_files_to_current_list": "Добавить загруженные файлы в текущий список файлов",
        "vr_aggr_and_post_process_not_applied_vr_6": "Агрессивность и постобработка не были применены, как так модель на VR 6 предсказывает два стема одновременно"
    },
    "en": {
        "bytes": "B",
        "kbytes": "KB",
        "mbytes": "MB",
        "gbytes": "GB",
        "tbytes": "TB",
        "separate": "Separate",
        "refresh": "Refresh",
        "ensemble_type": "Ensemble type",
        "ensemble_type_info": """<details>
<summary><b>Ensemble types</b></summary>

| Type | Description |
|------|-------------|
| `avg_fft` | Average, stable result (weights required) |
| `median_fft` | Median (effective for 3+ audio files) |
| `min_fft` | Minimum, cleaner result |
| `max_fft` | Maximum, more complete but "dirtier" result |

</details>""",
        "output_template": "Name template",
        "output_template_info": """
<details>
<summary><b>Available keys</b></summary>

- `NAME` — input file name (without extension)
- `STEM` — stem name (e.g., vocals, instrumental)
- `MODEL` — model name (e.g., bs_6stem)

Example: `NAME_STEM_MODEL` → `Song_vocals_bs_6stem`

</details>""",
        "output_etemplate_info": """<details>
<summary><b>Available keys</b></summary>

- `NAME` — input file name (without extension)
- `TYPE` — ensemble type (e.g., min_fft, avg_fft, etc.)
- `COUNT` — number of used models/files

Example: `NAME_COUNT_TYPE` → `Song_7_min_fft`

</details>""",
        "output_metemplate_info": """<details>
<summary><b>Available keys</b></summary>

- `NAME` — input file name (without extension)
- `TYPE` — ensemble type (e.g., min_fft, avg_fft, etc.)

Example: `NAME_TYPE` → `Song_min_fft`

</details>""",
        "output_itemplate_info": """<details>
<summary><b>Available keys</b></summary>

- `NAME` — input file name (without extension)
- `TYPE` — inversion type (e.g., waveform, spectrogram)

Example: `NAME_TYPE` → `Song_waveform`

</details>""",
        "output_name": "Output file name",
        "output_file": "Output file",
        "output_directory": "Output directory",
        "output_format": "Output format",
        "unload_model": "Unload model from memory",
        "extract_instrumental": "Extract instrumental",
        "use_spec_invert": "Use spectrogram for inversion creation",
        "select_stems": "Select output stems",
        "select_stems_info": "(If no stem is selected, all stems will be saved by default)",
        "show_preview": "Preview",
        "model_name": "Model name",
        "vr": "VR",
        "mdx": "MDX",
        "mdxc": "MDXC",
        "mvox": "Medley-Vox",
        "demucs": "Demucs",
        "separation_params": "Separation parameters",
        "separation_window_size_info": "Affects separation quality\nThe lower the value, the higher the quality",
        "separation_hop_info": "Audio analysis accuracy",
        "separation_aggresion_info": "Affects how deeply the stem will be extracted",
        "separation_hi-end_process_info": "Mirror restoration of missing high frequencies",
        "separation_overlap_info": "Determines how smooth the transitions between segments will be\nThe higher the value, the better the separation quality, but processing will be slower",
        "separation_segment_size_info": "Affects the quality of separation results\n(Too high values may cause RAM and VRAM overflow)",
        "separation_batch_size_info": "Number of segments processed simultaneously\nThe higher the value, the more resources will be used during separation\n(Too high values may cause RAM and VRAM overflow)",
        "mdxc_segment_size": "Segment size (dim_t)",
        "mdxc_batch_size": "Batch size",
        "mdxc_overlap": "Overlap",
        "mdxc_denoise": "Enable denoising",
        "mdxc_override_segment": "Override segment size",
        "demucs_segment": "Segment size (sec)",
        "demucs_batch_size": "Batch size",
        "demucs_overlap": "Overlap",
        "demucs_denoise": "Enable denoising",
        "demucs_override_segment": "Override segment size",
        "mdx_hop_length": "Hop length",
        "mdx_segment_size": "Segment size (dim_t)",
        "mdx_batch_size": "Batch size",
        "mdx_overlap": "Overlap",
        "mdx_denoise": "Enable denoising",
        "mdx_override_segment": "Override segment size",
        "vr_window_size": "Window size",
        "vr_batch_size": "Batch size",
        "vr_aggression": "Aggression",
        "vr_post_process": "Enable post-processing",
        "vr_post_process_threshold": "Post-processing threshold",
        "vr_high_end_process": "High frequency restoration",
        "mvox_segment": "Segment size (sec)",
        "mvox_overlap": "Overlap",
        "mvox_override_segment": "Override segment size",
        "unknown_model_type": "Unknown model type: {model_type}",
        "config_loaded": "Config loaded",
        "checkpoint_loaded": "Checkpoint loaded",
        "hubert_checkpoint_loaded": "Embedder loaded",
        "load_state_dict_error": "Error loading state_dict: {error}",
        "load_checkpoint_error": "Error loading checkpoint: {error}",
        "config_not_found": "Config not found: {path}",
        "config_load_error": "Error loading config: {error}",
        "config_is_not_loaded": "Config has not been loaded\nLoad config first",
        "freed_ram": "RAM freed",
        "emeergency_ram": "Emergency freed",
        "deleted_stems": "Unselected stems deleted",
        "added_second_stem": "Second stem added",
        "writing": "Writing to file",
        "format_return": "Return format",
        "denoise": "[denoise]",
        "mix_not_found": "Mix not found\nLoad mix first",
        "mix_is_empty": "Mix is empty",
        "model_not_loaded": "Model has not been loaded\nLoad model first",
        "demix_error": "Error during demixing: {error}",
        "name_stems_list": "[file name, [[stem1, stem1_path], [stem2, stem2_path]]]",
        "stems_list": "[[stem1, stem1_path], [stem2, stem2_path]]",
        "stems_list_append_self": "[[stem1, stem1_path], [stem2, stem2_path]]",
        "name_stems_list_append_self": "[file name, [[stem1, stem1_path], [stem2, stem2_path]]]",
        "processing": "Processing",
        "loaded_mix": "Mix loaded",
        "array_shape": "Array shape",
        "reuse_btn": "Use again",
        "bands": "bands",
        "patches": "patches",
        "chunks": "chunks",
        "files": "files",
        "samples": "samples",
        "path_not_specified": "Specify the file path first",
        "path_not_exist": "This file does not exist",
        "file_is_not_audio": "This is not an audio file",
        "sr_required": "Sample rate not specified\nSpecify the sample rate in the sr parameter",
        "ffmpeg_error": "FFMPEG error: {error}",
        "ffmpeg_exit_code": "Exit code: {code}",
        "write_critical_error": "Critical error during writing: {error}",
        "write_error": "Error during writing: {error}",
        "no_files_written": "None of the audio arrays were written\nErrors:\n{errors}",
        "concatenate_complete": "Concatenation complete",
        "ensemble_complete": "Ensemble successfully created",
        "unknown_etype": "Unknown ensemble type: {alg}",
        "subtract_spectrogram": "Spectrogram subtraction",
        "subtract_phase": "Phase cancellation subtraction",
        "arrays": "audio arrays",
        "extending_progress": "Extending audio",
        "arrays_srs_mismatch": "Must have the same number of arrays and sample rates",
        "fitting_progress": "Fitting audio",
        "unknown_shape": "Unknown array shape",
        "unknown_var": "Unknown variant: {var}",
        "mid_side_var0": "Subtract side channel from original",
        "mid_side_var1": "Subtract mono channel from original",
        "mid_side_var2": "Subtract phantom center from original",
        "mid_side_var3": "Subtract phantom center from original (Audacity-like)",
        "mid_side_var4": "Subtract right channel from left",
        "unexpected_min_val": "Unexpected minimum value: {value}",
        "array_dim_error": "Unsupported number of axes: {axis}\nOnly 1D and 2D arrays are supported",
        "invalid_bitrate": "Invalid bitrate: {bitrate}",
        "multi_reading": "Multi-reading",
        "channels_read_error": "Error getting number of channels: {path}",
        "sr_read_error": "Error getting sample rate: {path}",
        "ffmpeg_found": "FFMPEG found",
        "ffmpeg_not_found": "FFMPEG not found",
        "ffprobe_found": "FFPROBE found",
        "ffprobe_not_found": "FFPROBE not found",
        "history": "History",
        "history_select_info": "When selecting a value in this list, players with stems from the selected past separation will be shown",
        "last_separations": "Last separations",
        "not_last_separation": "First separate at least one audio file",
        "separation_tab": "Separation",
        "ensemble_tab": "Ensemble",
        "auto_ensemble_tab": "Auto",
        "man_ensemble_tab": "Manual",
        "history_loaded": "History loaded",
        "input_base_loaded": "Input file list loaded",
        "input_base_cleared": "Input file list cleared",
        "primary_stem": "Primary stem",
        "invert": "Invert",
        "weights": "Weights",
        "delete": "Delete",
        "replace": "Replace",
        "add_model": "Add model",
        "insert": "Insert",
        "ae_added_model": "Model added",
        "ae_deleted_model": "Model deleted",
        "ae_replaced_model": "Model replaced",
        "ae_inserted_model": "Model inserted",
        "clear": "Clear",
        "ae_all_cleared": "All models cleared",
        "model_index": "Model index",
        "ensemble_result": "Ensemble result",
        "saved_primary_stems": "Saved primary stems",
        "enable_save_primary_stems": "Enable saving primary stems",
        "inverted_result": "Inversion",
        "error_occured_separation": "An error occurred during separation\nModel will be skipped",
        "reuse_output_btn": "Use result again",
        "reuse_invert_btn": "Use inversion again",
        "run_ensemble": "Create ensemble",
        "ensemble_models_count": "Models",
        "ensemble_settings": "Ensemble settings",
        "from_array": "from audio array",
        "not_separated": "Separate audio first",
        "ensemble_flow_saved": "Ensemble preset saved",
        "ensemble_flow_loaded": "Ensemble preset loaded",
        "preset_name": "Preset name",
        "load": "Load",
        "save": "Save",
        "import": "Import",
        "export": "Export",
        "weights_only_for_avg_fft": "Weights (required only for avg_fft type)",
        "presets": "Presets",
        "not_ensembled_with_primary_stems": "When creating an ensemble, enable saving primary stems to see them after ensemble creation",
        "paths_not_specified": "Specify file paths first",
        "weights_split": "Comma-separated split (e.g., 1.0,0.3,0.2)",
        "paths_not_exist": "These files do not exist",
        "files_is_not_audio": "These are not audio files",
        "original": "Original",
        "stem": "Stem",
        "subtract_tab": "Subtraction",
        "multi_writing": "Multi-writing",
        "ensemble_preset_settings": "Preset",
        "auto_ensemble_name_preset": "Preset name",
        "ensemble_flow_not_exist": "This ensemble preset does not exist",
        "name_not_specified": "Specify the name first",
        "subtract": "Subtract",
        "index_rate": "Index influence",
        "index_rate_info": "The lower the value, the more the voice resembles the original; the higher, the closer to the model",
        "volume_envelope": "Volume envelope ratio",
        "volume_envelope_info": "Replace or mix with the output volume envelope\nThe closer the values are to 1, the more the output envelope is used",
        "protect": "Consonant protection",
        "protect_info": "Prevents robotization of breath and consonants (May affect speech clarity)\nValue 0.5 provides full protection\n",
        "crepe_hop_length": "Hop length",
        "crepe_hop_length_info": "F0 extraction accuracy\nThe smaller the value, the more accurate the F0 extraction",
        "vbach_embedder": "Embedder",
        "vbach_embedder_info": "Model used for sound content analysis",
        "vbach_embedder_not": "This embedder does not exist",
        "stereo_mode": "Stereo mode",
        "stereo_mode_info": """<details>
<summary><b>Stereo modes</b></summary>

| Mode | Description |
|------|-------------|
| `mono` | Convert to mono signal (default) |
| `left/right` | Process left and right channels independently |
| `sim/dif` | Process phantom center and stereo base independently |

</details>""",
        "chunk_duration": "Chunk duration (sec)",
        "vbach_use_transformers": "Use Transformers stack",
        "pitch": "Pitch (semitones)",
        "speaker_id": "Speaker ID",
        "f0_method": "F0 extraction method",
        "using_vocoder": "Vocoder is used",
        "vbach_tab": "Vocal conversion",
        "convert": "Convert",
        "advanced_params": "Advanced parameters",
        "model_unloaded": "Model unloaded from memory",
        "model_not_selected": "Select model path first",
        "no_conversion_results": "Perform conversion first",
        "output_vbach_template_info": """<details>
<summary><b>Available keys</b></summary>

- `NAME` - input file name (without extension)
- `F0METHOD` - F0 extraction method (e.g., rmvpe+, fcpe)
- `MODEL` — model name (e.g., rvc_test)
- `PITCH` - pitch shift (e.g., 0, 12)

Example: `NAME_F0METHOD_PITCH` → `Song_rmvpe+_0`
        
</details>""",
        "model_path": "Model file path",
        "index_path": "Index file path",
        "extracting_f0": "Extracting F0...",
        "extracting_f0_success": "F0 extraction completed",
        "reading_faiss_index_error": "Error reading FAISS index: {error}",
        "mix_complete": "Mix created",
        "download_attempt_failed": "Attempt {attempt}/{retries} failed. Error: {error}",
        "all_download_attempts_failed": "All download attempts failed",
        "retrying": "Retrying...",
        "unknown_f0_method": "Unknown F0 extraction method: {method}",
        "models": "models",
        "f0_min": "Minimum F0 frequency",
        "f0_max": "Maximum F0 frequency",
        "download_from_internet": "Download from internet",
        "download_from_local_device": "Download from device",
        "supported_only_direct_links": "Only direct links are supported",
        "download_model_files_from_zip": "Download model files from ZIP archive",
        "download_model_files": "Download model files separately",
        "vbach_checkpoint_pth_placeholder": "Checkpoint (*.pth)",
        "vbach_index_file_placeholder": "Index file (*.index, optional)",
        "vbach_zip_placeholder": "Archive (*.zip)",
        "vbach_zip_link": "ZIP archive link",
        "vbach_pth_link": "Checkpoint link",
        "vbach_index_link": "Index file link",
        "download_and_unzip": "Download and unzip",
        "download_and_move_to_models_dir": "Download and move to models folder",
        "vbach_models_tab": "Vocal conversion models",
        "download_model": "Download model",
        "vbach_model_zip_unpacked": "ZIP archive unpacked",
        "vbach_model_zip_not_model_files": "No model files found in ZIP archive",
        "vbach_model_pths_uploaded": "Checkpoints uploaded",
        "vbach_model_indexes_uploaded": "Index files uploaded",
        "vbach_model_pth_uploaded": "Checkpoint uploaded",
        "vbach_model_index_uploaded": "Index file uploaded",
        "vbach_model_pth_downloaded": "Checkpoint downloaded",
        "vbach_model_index_downloaded": "Index file downloaded",
        "vbach_added_pths": "Checkpoints added",
        "vbach_added_indexes": "Index files added",
        "model_downloaded": "Model downloaded",
        "model_already_downloaded": "Model already downloaded",
        "download": "Download",
        "f0_curve_not_found": "No F0 curve found in the specified file",
        "importing_f0": "Importing F0",
        "importing_f0_success": "F0 import completed",
        "custom_f0": "Custom F0",
        "vbach_result": "Conversion result",
        "f0_extraction_tab": "F0 extraction",
        "extract_f0": "Extract F0",
        "output_path": "Save path",
        "f0_output_path_info": "Leave empty to automatically save next to the audio file",
        "f0_output_path_placeholder": "e.g., /path/to/output.json",
        "f0_extraction_results": "F0 extraction results",
        "f0_extracted_success": "F0 successfully extracted",
        "no_f0_extracted": "F0 not extracted yet",
        "download_f0_json": "Download JSON with F0",
        "no_audio_selected": "Select an audio file first",
        "f0_extraction_complete": "F0 extraction completed",
        "f0_json_file": "JSON file with F0 curve",
        "convert_custom_f0": "Convert with custom F0",
        "no_f0_file_selected": "Select a JSON file with F0 curve first",
        "conversion_results": "Conversion results",
        "download_audio": "Download audio",
        "conversion_complete": "Conversion completed",
        "output_vbach_custom_template_info": """<details>
<summary><b>Available keys</b></summary>

- `NAME` - input file name (without extension)
- `F0METHOD` - F0 extraction method (always custom)
- `MODEL` — model name (e.g., rvc_test)
- `PITCH` - pitch shift (e.g., 0, 12)

Example: `NAME_F0METHOD_PITCH` → `Song_custom_0`
        
</details>""",
        "status": "Status",
        "inference": "Inference",
        "vbach_inference_custom_f0": "Inference with custom F0",
        "model_info_updated": "Model information updated\nRestart the running application for changes to take effect",
        "model_info": "Model information",
        "output_stems": "Output stems",
        "table_model_info_installed_legend": "[green]installed[/]",
        "table_model_info_target_instrument_legend": "[green]target instrument[/]",
        "na": "n/a",
        "supported_yt_dlp_info": "YouTube, SoundCloud, TikTok and many other sites are supported (via yt-dlp)",
        "downloaded_and_uploaded": "Files downloaded and uploaded: {count}",
        "download_failed_no_file": "Failed to download audio. Check the link.",
        "bitrate": "Bitrate",
        "cookie_settings": "Cookie settings (for bypassing restrictions)",
        "cookie_explanation": """<details>
        <summary><b>📖 Why are cookies needed?</b></summary>
        <br>
        <b>Cookies are necessary in the following cases:</b>
        <ul>
        <li><b>Age restrictions:</b> YouTube requires authorization to access 18+ videos</li>
        <li><b>Regional blocks:</b> Some videos are unavailable in your country without authorization</li>
        <li><b>Premium content:</b> Some platforms require a subscription</li>
        <li><b>Anti-bot protection:</b> Some sites block mass requests</li>
        <li><b>Private videos:</b> Access to videos available only via link with authorization</li>
        </ul>

        <b>How to get a cookie file:</b>
        <ol>
        <li>Install a browser extension (e.g., Get cookies.txt LOCALLY for Chrome/Firefox)</li>
        <li>Log in to the desired site (YouTube, SoundCloud, etc.)</li>
        <li>Click the extension icon and export cookies in Netscape format</li>
        <li>Upload the resulting .txt file to the field above</li>
        </ol>

        ⚠️ <b>Important:</b> Cookie files contain your session data. Do not share them with third parties and delete after use.
        </details>""",
        "cookie_file": "Cookie file (Netscape format)",
        "cookie_status": "Cookie status",
        "cookie_loaded": "✅ Cookie loaded: {path}",
        "cookie_not_loaded": "❌ Cookie not loaded (access to restricted content may be impossible)",
        "custom_separation_tab": "Separation with custom model",
        "model_type": "Model type",
        "checkpoint_path": "Checkpoint path",
        "config_path": "Config path",
        "upload_audio": "Upload audio",
        "upload_from_zip": "Upload from ZIP archive",
        "upload_zip_placeholder": "",
        "uploaded_files_count": "Uploaded files: {count}",
        "extract_and_upload": "Extract and upload",
        "audio_url": "Audio link",
        "download_and_upload": "Download",
        "upload_from_path": "Upload by path",
        "folder_path": "File or folder path",
        "scan_and_upload": "Scan",
        "custom_separation_models_tab": "Custom separation models",
        "custom_checkpoint_link": "Checkpoint link",
        "custom_config_link": "Config link",
        "custom_checkpoint_placeholder": "Checkpoint (*.pth)",
        "custom_config_placeholder": "Config (*.yaml)",
        "custom_added_configs": "Configs added",
        "custom_added_checkpoints": "Checkpoints added",
        "custom_model_config_downloaded": "Config downloaded",
        "custom_model_checkpoint_downloaded": "Checkpoint downloaded",
        "upload_from_files": "Upload files",
        "upload": "Upload",
        "upload_from_url": "Upload from URL",
        "download_error": "Error downloading file: {error}",
        "download_start": "Download started, size:",
        "download_complete": "Download complete",
        "flows_imported": "Presets imported",
        "import": "Import",
        "export": "Export",
        "all_ensemble_flow_cleared": "All ensemble presets deleted",
        "f0_file_info": """<details><summary><b>Data format in F0 curve JSON file:</b></summary>

```python
{
    "method": "rmvpe+", # F0 extraction method
    "sample_rate": 16000, # Sample rate
    "window": 160, # Window size in samples
    "p_len": 500, # Expected number of frames
    "freqs": [0, 0, 120, 121.3, ...] # List of frequencies
}
```

</details>""",
        "arg_main_description": "MVSepless - tool for music source separation and creating ensembles from multiple models.",
        "arg_main_epilog": "Supports VR, MDX, MDXC, Demucs, Medley-Vox models. Documentation: https://github.com/noblebarkrr/mvsepless/tree/dzeta",
        "arg_subcommands_title": "subcommands",
        "arg_subcommands_description": "Available operating modes",
        "arg_subcommands_help": "Select action",
        "arg_separate_help": "Separate audio into stems using a built-in model",
        "arg_separate_description": "Separate audio files into stems (vocals, instrumental, drums, etc.) using preset models.",
        "arg_separate_epilog": "Example: python inference.py separate -i audio.mp3 -o output -of mp3 -mn bs_6stem -tm NAME_STEM",
        "arg_custom_separate_help": "Separate audio with a custom model (your own checkpoint and config)",
        "arg_custom_separate_description": "Separate audio files using your own model loaded from checkpoint and configuration file.",
        "arg_custom_separate_epilog": "Example: python inference.py custom_separate -i audio.mp3 -o output -ckpt model.ckpt -conf config.yaml -mt bs_roformer",
        "arg_info_help": "Information about available models",
        "arg_info_description": "Show the list of available separation models with filtering by stem, output limiting, and cache updating.",
        "arg_info_epilog": "Example: python inference.py info -limit 10 -stem vocals -oi",
        "arg_auto_ensemble_help": "Automatic ensemble of multiple models",
        "arg_auto_ensemble_description": "Create an ensemble of results from multiple separation models with automatic model loading by preset.",
        "arg_auto_ensemble_epilog": "Example: python inference.py auto_ensemble -i audio.mp3 -flow bs_6stem:vocals:True:1 mbr_inst1e_unwa:other:False:1 -type avg_fft",
        "arg_manual_ensemble_help": "Manual ensemble from ready-made audio files",
        "arg_manual_ensemble_description": "Create an ensemble from already prepared audio files (separation results) with the ability to specify weights.",
        "arg_manual_ensemble_epilog": "Example: python inference.py manual_ensemble -i audio1.wav audio2.wav -o output -w 1.0 0.5 -type median_fft",
        "arg_subtract_help": "Subtract a stem from the original (create instrumental)",
        "arg_subtract_description": "Subtract one audio file (stem) from another (original) to obtain an instrumental version or other combinations.",
        "arg_subtract_epilog": "Example: python inference.py subtract -i1 original.mp3 -i2 vocals.mp3 -o output -ispec",
        "arg_input_help": "Path to input audio file, folder with files, or list of paths",
        "arg_input_single_help": "Path to input audio file",
        "arg_output_dir_help": "Directory for saving results (default: current folder)",
        "arg_output_format_help": "Output audio format. Available: {formats} (default: {default})",
        "arg_template_help": "Output file name template. Available keys: {keys}. Example: {example}",
        "arg_extract_instrumental_help": "Create inversion of selected stems (instrumental) - stem named 'invert'",
        "arg_use_spec_invert_help": "Use spectrogram subtraction instead of phase cancellation when creating inversion",
        "arg_selected_stems_help": "List of stems to save (e.g., vocals drums). If not specified, all stems are saved",
        "arg_model_name_help": "Model name for separation (default: bs_6stem). Full list: python inference.py info",
        "arg_model_type_help": "Model type (e.g., bs_roformer, demucs, mdx). Default: bs_roformer",
        "arg_checkpoint_path_help": "Path to model checkpoint file (*.ckpt or *.pth)",
        "arg_config_path_help": "Path to model configuration file (*.yaml)",
        "arg_ensemble_type_help": "Ensemble type: avg_fft (average, weights required), median_fft (median, for 3+ audio), min_fft (minimum), max_fft (maximum). Default: avg_fft",
        "arg_save_primary_stems_help": "Save primary stems to output directory",
        "arg_flow_help": "Preset as a string: MODEL:PRIMARY_STEM:INVERSION:WEIGHT. Example: bs_6stem:vocals:True:1 mbr_inst1e_unwa:other:False:1",
        "arg_preset_json_help": "Path to JSON file with ensemble preset",
        "arg_weights_help": "Weights for each audio file (required for avg_fft type). Example: -w 1.0 0.5 0.2",
        "arg_input1_help": "Path to original audio file (from which we subtract)",
        "arg_input2_help": "Path to stem audio file (which we subtract)",
        "arg_update_help": "Update model information cache from repository",
        "arg_clear_cache_help": "Clear model information cache",
        "arg_download_help": "Download specified model (requires -mn)",
        "arg_limit_help": "Limit number of displayed models (0 or None - no limit)",
        "arg_stem_filter_help": "Show only models that contain the specified stem (e.g., vocals, drums)",
        "arg_only_installed_help": "Show only installed (downloaded) models",
        "arg_add_param_help": "Additional separation parameter",
        "vbach_main_description": "VBach - fork of PolGen for vocal conversion with pitch and timbre change",
        "vbach_main_epilog": "Supported F0 extraction methods: rmvpe+, hpa-rmvpe, fcpe, mangio-crepe, mangio-crepe-tiny, harvest, pm, pyin. Documentation: vbach_lib/README.md",
        "vbach_infer_help": "Vocal conversion (voice change) using a model",
        "vbach_infer_description": "Vocal conversion using RVC/PollGen model: pitch change, timbre change, application of index file to improve quality.",
        "vbach_infer_epilog": "Example: python infer.py infer -i audio.mp3 -o output -m model.pth -p 2 -f0m rmvpe+",
        "vbach_infer_custom_f0_help": "Vocal conversion using custom F0 curve from JSON file",
        "vbach_infer_custom_f0_description": "Vocal conversion with pre-extracted and saved F0 curve (allows precise melody control).",
        "vbach_infer_custom_f0_epilog": "Example: python infer.py infer_custom_f0 -i audio.mp3 -o output -m model.pth -f0f f0.json -p 0",
        "vbach_download_hubert_help": "Download embedder model (HuBERT) for vocal conversion",
        "vbach_download_hubert_description": "Download HuBERT embedder model (Fairseq or Transformers version) required for vocal conversion.",
        "vbach_download_hubert_epilog": "Example: python infer.py download_hubert -emb hubert_base -tf",
        "vbach_model_path_help": "Path to model checkpoint file (*.pth)",
        "vbach_index_path_help": "Path to FAISS index file (*.index) for quality improvement (optional)",
        "vbach_pitch_help": "Pitch shift in semitones (positive values - higher, negative - lower). Default: 0",
        "vbach_f0_method_help": "F0 extraction method. Available: rmvpe+, hpa-rmvpe, fcpe, mangio-crepe, mangio-crepe-tiny, harvest, pm, pyin. Default: rmvpe+",
        "vbach_index_rate_help": "Index influence (0-1). The lower the value, the more the voice resembles the original; the higher, the closer to the model. Default: 0.75",
        "vbach_volume_envelope_help": "Volume envelope ratio (0-1). Replaces or mixes with the output volume envelope. Default: 0.25",
        "vbach_protect_help": "Consonant protection (0-0.5). Prevents robotization of breath and consonants. Default: 0.33",
        "vbach_hop_length_help": "Hop length for F0 extraction with CREPE methods. The smaller the value, the more accurate. Default: 128",
        "vbach_embedder_help": "Embedder model: hubert_base or spin. Default: hubert_base",
        "vbach_use_transformers_help": "Use Transformers stack for embedder (instead of Fairseq)",
        "vbach_stereo_mode_help": "Stereo mode: mono (convert to mono signal), left/right (process channels independently), sim/dif (process center and stereo base independently). Default: mono",
        "vbach_f0_min_help": "Minimum F0 frequency in Hz. Default: 50",
        "vbach_f0_max_help": "Maximum F0 frequency in Hz. Default: 1100",
        "vbach_chunk_duration_help": "Chunk duration for processing in seconds. Default: 7",
        "vbach_f0_file_help": "Path to JSON file with custom F0 curve (obtained via f0_extractor.py)",
        "f0_extract_description": "Extract F0 curve (fundamental frequency) from audio file for later use in vocal conversion",
        "f0_extract_epilog": "Example: python f0_extractor.py -i audio.mp3 -f0m rmvpe+ -f0min 50 -f0max 1100 -o f0.json",
        "f0_extract_output_help": "Path to save JSON file with F0 curve. If not specified, saved next to the audio file",
        "app_description": "MVSepless Web-UI - graphical interface for music separation and vocal conversion",
        "app_epilog": "Example: python app.py --share --full --port 7860",
        "app_share_help": "Create a public link via Gradio Share (for internet access)",
        "app_port_help": "Port for server launch (default: 7860)",
        "app_full_help": "Launch full interface version (not Hugging Face Spaces mode)",
        "template_keys_separate": "NAME (file name without extension), STEM (stem name), MODEL (model name)",
        "template_keys_auto_ensemble": "NAME (file name), TYPE (ensemble type), COUNT (number of models)",
        "template_keys_manual_ensemble": "NAME (first file name), TYPE (ensemble type)",
        "template_keys_subtract": "NAME (original file name), TYPE (inversion type: waveform or spectrogram)",
        "template_keys_vbach": "NAME (file name), F0METHOD (F0 method), PITCH (pitch shift)",
        "stems": "Stems",
        "target_instrument": "Target instrument",
        "yes": "Yes",
        "no": "No",
        "zerogpu=true": "Runtime is ZeroGPU",
        "ensemble_processing": "Creating ensemble",
        "tracks": "tracks",
        "app_user_dir_help": "Path to directories for storing user files",
        "gdrive_mount_found": "Detected mounted Google Drive",
        "copy_to_gdrive": "Copying data to Google Drive",
        "dirs": "directories",
        "copy_to_gdrive_done": "Copy complete",
        "copied_dirs": "Directories copied",
        "copy_from_current_user_dir_to_gdrive": "Copy all user data to Google Drive",
        "google_drive": "Google Drive",
        "free_space": "Free",
        "used_space": "Used",
        "all_space": "All",
        "used_space_data_local": "User data space in runtime",
        "used_space_data_gdrive": "User data space on Google Drive",
        "added_files": "Added Files",
        "reuse_all_stem": "Reuse all {stem}",
        "reuse_all_stems": "Reuse all stems",
        "generate_zip_archive": "Generate ZIP",
        "download_zip_archive": "Download ZIP",
        "invert_plus": "When extracting instrumental, add all unselected stems",
        "invert_plus_info": "Reduces the remainder of selected stems when extracting a instrumental (relevant for Roformer models)",
        "invert_plus_applied": "Invert plus applied",
        "flow_empty": "Add at least one model to the preset",
        "iteration": "Iteration",
        "num_iters": "Number of iterations",
        "no_models_succeeded": "No models were used",
        "saved_file": "Saved file",
        "arg_iterative_ensemble_help": "Iterative ensemble for progressive separation improvement",
        "arg_iterative_ensemble_description": "Sequential application of a set of models to gradually improve separation quality (extracting residuals with each iteration).",
        "arg_iterative_ensemble_epilog": "Example: python inference.py iterative_ensemble -i audio.mp3 -o output -flow bs_6stem:vocals:True mbr_inst1e_unwa:other:False -n 4 -save_intermediate",
        "arg_iterative_flow_help": "Preset as a string: MODEL:PRIMARY_STEM:INVERSION. Example: bs_6stem:vocals:True mbr_inst1e_unwa:other:False",
        "arg_num_iters_help": "Number of iterations (default: 4). Each iteration applies the entire set of models to the residual from the previous iteration",
        "arg_save_intermediate_help": "Save intermediate results of each iteration",
        "template_keys_iterative_ensemble": "NAME (file name), ITER (iteration number)",
        "saved_intermediate_files": "Saved intermediate files",
        "iterative_ensemble_name_preset": "Iterative ensemble preset name",
        "run_iterative_ensemble": "Run iterative ensemble",
        "save_intermediate": "Save intermediate results",
        "intermediate_results": "Intermediate results",
        "no_intermediate_results": "No intermediate results",
        "output_iterative_template_info": """<details><summary><b>Available keys</b></summary>

- `NAME` — input file name (without extension)
- `ITER` — iteration number (iter_N)

Example: `NAME_ITER` → `Song_iter_3`

</details>""",
        "iterative_ensemble_tab": "Iterative",
        "model": "Model",
        "selected_stems": "Selected stems",
        "corrected_selected_stems": "Existing selected stems",
        "uncorrected_selected_stems": "Warning! Non-exist stems",
        "prefer_float": "Prefer high precision output",
        "current_device": "Current device: {device}",
        "no_has_taglib": "Dependency 'pytaglib' is not installed\nOutput audio files will be written without metadata",
        "write_metadata": "Write metadata",
        "write_metadata_info": "Adds processing information to the file's metadata.\nIf the original audio contains metadata, this information is written into the metadata alongside the original data.",
        "write_metadata_error": "Error writing metadata: {error}",
        "off_audio_players_output": "Off audio players in output",
        "off_audio_players_output_info": "Disables audio players in the output, replacing them with simple download buttons; this can speed up the interface when dealing with a large count of output files.",
        "add_uploaded_files_to_current_list": "Add uploaded files to the current list of files",
        "vr_aggr_and_post_process_not_applied_vr_6": "Aggressiveness and post-processing were not applied because the VR 6 model predicts two stems simultaneously."
    }
}

# Функция для получения перевода
def _i18n(key: str, **kwargs) -> str:
    """
    Получить перевод для текущего языка с возможностью форматирования
    
    Args:
        key: Ключ перевода
        **kwargs: Аргументы для форматирования строки
    
    Returns:
        Переведенная строка
    """
    translation = TRANSLATIONS[CURRENT_LANGUAGE].get(key, key)
    if kwargs:
        try:
            return translation.format(**kwargs)
        except (KeyError, IndexError) as e:
            # Если ключ форматирования не найден, возвращаем как есть
            return translation
    return translation

# Функция для смены языка
def set_language(lang: Language) -> None:
    """
    Установить язык
    
    Args:
        lang: Код языка ("ru" или "en")
    """
    global CURRENT_LANGUAGE
    if lang in TRANSLATIONS:
        CURRENT_LANGUAGE = lang
    else:
        CURRENT_LANGUAGE = "en"
