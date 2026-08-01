<div align="center">

# VBach
</div>

<h2> форк <u>PolGen 1.2.0</u> с большими изменениями</h2>

## Фишки

- Пакетная обработка по умолчанию
- Работа с двумя стеками эмбеддинга: `Fairseq` и `Transformers`
- Альтернативный пайлплайн, использующий чанки фиксированной длины с перекрытием
- История обработок (Только в Web-UI)
- Инференс с кастомной кривой F0
- Корректор кривой F0 (Только в Web-UI)

---

<details> <summary align="center"><b>Содержание</b></summary>

- [Шаблон имени файла](#шаблон-имени)
- [Доступные методы извлечения F0](#методы-f0)
- [Формат данных в JSON-файле кривой F0](#формат-данных-f0)
- [Стерео-режимы](#стерео-режимы-обработки)
- [Пример использования](#пример-использования)
    - [Извлечение F0 кривой](#извлечение-f0-кривой-использование)
    - [Преобразование вокала](#преобразование-вокала-использование)
    - [Преобразование вокала с кастомной F0 кривой](#преобразование-вокала-с-кастомной-f0-кривой-использование)
    - [Загрузка модели эмбеддера](#загрузка-модели-эмбеддера-использование)
- [Параметры методов](#параметры-методов)
    - [Извлечение F0 кривой](#извлечение-f0-кривой-метод)
    - [Преобразование вокала](#преобразование-вокала-метод)
    - [Преобразование вокала с кастомной F0 кривой](#преобразование-вокала-с-кастомной-f0-кривой-метод)
    - [Загрузка модели эмбеддера](#загрузка-модели-эмбеддера-метод)
- [Все параметры командной строки](#параметры-командной-строки)
    - [Извлечение F0 кривой](#извлечение-f0-кривой-cli)
    - [Преобразование вокала](#преобразование-вокала-cli)
    - [Преобразование вокала с кастомной F0 кривой](#преобразование-вокала-с-кастомной-f0-кривой-cli)
    - [Загрузка модели эмбеддера](#загрузка-модели-эмбеддера-cli)

</details>

---

## Шаблон имени файла <span id="шаблон-имени"></span>

Доступные ключи для шаблона:

- `NAME` - имя входного файла (без расширения)
- `F0METHOD` - метод извлечения F0 (например: rmvpe+, custom)
- `MODEL` — имя модели (например: rvc_test)
- `PITCH` - изменение высоты тона (например: 0, 12)

Пример: `NAME_F0METHOD_PITCH` → `Song_rmvpe+_0`

## Доступные методы извлечения F0 <span id="методы-f0"></span>

| Метод | Описание |
|-------|----------|
| `rmvpe` | Высокая точность, рекомендуется по умолчанию |
| `hpa-rmvpe` | Улучшенная версия RMVPE |
| `fcpe` | Fast Context-aware Pitch Extraction |
| `fcpe+unvoiced_rmvpe` | То же самое, что и `fcpe`, но если `rmvpe` считает участок глухим (unvoiced), частота `fcpe` принудительно зануляется |
| `mangio-crepe` | CREPE с высокой точностью |
| `mangio-crepe-tiny` | CREPE (облегченная версия) |
| `mangio-crepe+unvoiced_rmvpe` | То же самое, что и `mangio-crepe`, но если `rmvpe` считает участок глухим (unvoiced), частота `mangio-crepe` принудительно зануляется |
| `mangio-crepe-tiny+unvoiced_rmvpe` | То же самое, что и `mangio-crepe-tiny`, но если `rmvpe` считает участок глухим (unvoiced), частота `mangio-crepe-tiny` принудительно зануляется |
| `harvest` | Алгоритм из WORLD вокодера |
| `pm` | Parcelmouth (акустический анализ) |
| `pyin` | Probabilistic YIN (в Librosa)|

## Формат данных в JSON-файле кривой F0: <span id="формат-данных-f0"></spaN>

```python
{
    "method": "rmvpe+", # Метод извлечения F0
    "sample_rate": 16000, # Частота дискретизации
    "window": 160, # Размер окна в сэмплах
    "p_len": 500, # ожидаемое количество кадров
    "freqs": [0, 0, 120, 121.3, ...] # Список с частотами
}
```

## Стерео-режимы <span id="стерео-режимы-обработки"></span>

| Режим | Описание |
|-------|----------|
| `mono` | Преобразование моно-сигнала (по умолчанию) |
| `left/right` | Обработка левого и правого канала независимо |
| `sim/dif` | Обработка фантомного центра и стерео-базы независимо |

## 🎤 Пример использования <span id="пример-использования"></span>

### Извлечение F0 кривой <span id="извлечение-f0-кривой-использование"></span>

#### Через командную строку

```sh
python f0_extractor.py -i "audio.mp3" -f0m rmvpe+ -f0min 50 -f0max 1100 -o "f0.json"
```

#### Напрямую

```python
from vbach_lib.f0_extractor import f0_extract_and_write

# Извлечение F0 кривой из аудиофайла
output_path = f0_extract_and_write(
    "audio.mp3",           # Входной аудиофайл
    f0_method="rmvpe+",    # Метод извлечения F0
    f0_min=50,             # Минимальная частота F0
    f0_max=1100,           # Максимальная частота F0
    output_path="f0.json"  # Путь для сохранения (опционально)
)
```

### Преобразование вокала <span id="преобразование-вокала-использование"></span>

#### Через командную строку

```sh
python infer.py infer -i "audio.mp3" -o "output" -m "model.pth" -idx "index.index" -p 0 -f0m rmvpe+ -idxr 0.75 -ve 0.25 -pr 0.33 -of mp3 -tm "NAME_F0METHOD_PITCH"
```

#### Напрямую

```python
from vbach_lib.infer import VbachConverter

converter = VbachConverter()

results = converter.convert_audio(
    "audio.mp3",           # Входной файл/папка или список путей
    "output",              # Директория вывода
    "model.pth",           # Путь к чекпоинту модели
    "index.index",         # Путь к индекс-файлу (опционально)
    pitch=0,               # Изменение высоты тона (полутона)
    f0_method="rmvpe+",    # Метод извлечения F0
    index_rate=0.75,       # Влияние индекса
    volume_envelope=0.25,  # Соотношение огибающих громкости
    protect=0.33,          # Защита согласных
    hop_length=128,        # Длина шага для CREPE
    embedder_model="hubert_base",  # Модель эмбеддера
    use_transformers=False,        # Использовать стек Transformers
    output_format="mp3",   # Формат вывода
    stereo_mode="mono",    # Стерео-режим
    f0_min=50,             # Минимальная частота F0
    f0_max=1100,           # Максимальная частота F0
    chunk_duration=7,      # Длина чанка (сек)
    template="NAME_F0METHOD_PITCH"  # Шаблон имени файла
)
```

### Преобразование вокала с кастомной F0 кривой <span id="преобразование-вокала-с-кастомной-f0-кривой-использование"></span>

#### Через командную строку

```sh
python infer.py infer_custom_f0 -i "audio.mp3" -o "output" -m "model.pth" -f0f "f0.json" -p 0 -idxr 0.75 -ve 0.25 -pr 0.33 -of mp3 -tm "NAME_custom_PITCH"
```

#### Напрямую

```python
from vbach_lib.infer import VbachConverter

converter = VbachConverter()

result = converter.convert_audio_custom_f0(
    "audio.mp3",           # Входной аудиофайл
    "output",              # Директория вывода
    "model.pth",           # Путь к чекпоинту модели
    "index.index",         # Путь к индекс-файлу (опционально)
    pitch=0,               # Изменение высоты тона (полутона)
    f0_file="f0.json",     # Путь к JSON с F0 кривой
    index_rate=0.75,       # Влияние индекса
    volume_envelope=0.25,  # Соотношение огибающих громкости
    protect=0.33,          # Защита согласных
    embedder_model="hubert_base",  # Модель эмбеддера
    use_transformers=False,        # Использовать стек Transformers
    output_format="mp3",   # Формат вывода
    f0_min=50,             # Минимальная частота F0
    f0_max=1100,           # Максимальная частота F0
    chunk_duration=7,      # Длина чанка (сек)
    template="NAME_F0METHOD_PITCH"  # Шаблон имени файла
)
```

### Загрузка модели эмбеддера <span id="загрузка-модели-эмбеддера-использование"></span>

```python
from vbach_lib.infer import download_hubert

download_hubert(
    "hubert_base",         # Имя модели эмбеддера
    use_transformers=False # Использовать Transformers версию
)
```

## 📋 Параметры методов <span id="параметры-методов"></span>

### `f0_extract_and_write()` <span id="извлечение-f0-кривой-метод"></span>

| Параметр | Тип | Описание |
|----------|-----|----------|
| `input_audio` | `str \| Path` | Путь к входному аудиофайлу |
| `f0_method` | `str` | Метод извлечения F0 (`rmvpe+`, `hpa-rmvpe`, `fcpe`, `mangio-crepe`, `mangio-crepe-tiny`, `harvest`, `pm`, `pyin`) |
| `f0_min` | `int` | Минимальная частота F0 (Гц) |
| `f0_max` | `int` | Максимальная частота F0 (Гц) |
| `output_path` | `str \| Path` | Путь для сохранения JSON (опционально) |

**Возвращает**: `str` - путь к сохраненному JSON файлу

### `VbachConverter().convert_audio()` <span id="преобразование-вокала-метод"></span>

| Параметр | Тип | Описание |
|----------|-----|----------|
| `audio_input` | `str \| Path \| list` | Путь к файлу/папке или список путей |
| `output_dir` | `str \| Path` | Директория вывода |
| `model_path` | `str` | Путь к чекпоинту модели (`*.pth`) |
| `index_path` | `str` | Путь к индекс-файлу (`*.index`, опционально) |
| `pitch` | `int` | Изменение высоты тона (полутона) |
| `f0_method` | `str` | Метод извлечения F0 |
| `index_rate` | `float` | Влияние индекса (0-1) |
| `volume_envelope` | `float` | Соотношение огибающих громкости (0-1) |
| `protect` | `float` | Защита согласных (0-0.5) |
| `hop_length` | `int` | Длина шага для CREPE |
| `embedder_model` | `str` | Модель эмбеддера (`hubert_base`, `spin`) |
| `use_transformers` | `bool` | Использовать Transformers версию |
| `output_format` | `str` | Формат вывода (`mp3`, `wav`, `flac`, и т.д.) |
| `stereo_mode` | `str` | Стерео-режим (`mono`, `left/right`, `sim/dif`) |
| `f0_min` | `int` | Минимальная частота F0 (Гц) |
| `f0_max` | `int` | Максимальная частота F0 (Гц) |
| `chunk_duration` | `int` | Длина чанка для обработки (сек) |
| `template` | `str` | Шаблон имени файла (ключи: `NAME`, `F0METHOD`, `PITCH`) ([Подробнее](#шаблон-имени)) |

**Возвращает**: `list` - список путей к сохраненным аудиофайлам

### `VbachConverter().convert_audio_custom_f0()` <span id="преобразование-вокала-с-кастомной-f0-кривой-метод"></span>

| Параметр | Тип | Описание |
|----------|-----|----------|
| `audio_input` | `str \| Path` | Путь к входному аудиофайлу |
| `output_dir` | `str \| Path` | Директория вывода |
| `model_path` | `str` | Путь к чекпоинту модели |
| `index_path` | `str` | Путь к индекс-файлу (опционально) |
| `pitch` | `int` | Изменение высоты тона (полутона) |
| `f0_file` | `str \| Path` | Путь к JSON с F0 кривой ([Пример JSON-файла](#формат-данных-в-json-файле-кривой-f0))|
| `index_rate` | `float` | Влияние индекса |
| `volume_envelope` | `float` | Соотношение огибающих громкости |
| `protect` | `float` | Защита согласных |
| `embedder_model` | `str` | Модель эмбеддера |
| `use_transformers` | `bool` | Использовать Transformers версию |
| `output_format` | `str` | Формат вывода |
| `stereo_mode` | `str` | Стерео-режим |
| `f0_min` | `int` | Минимальная частота F0 |
| `f0_max` | `int` | Максимальная частота F0 |
| `chunk_duration` | `int` | Длина чанка (сек) |
| `template` | `str` | Шаблон имени файла (ключи: `NAME`, `F0METHOD`, `PITCH`) ([Подробнее](#шаблон-имени)) |

**Возвращает**: `str` - путь к сохраненному аудиофайлу

### `download_hubert()` <span id="загрузка-модели-эмбеддера-метод"></span>

| Параметр | Тип | Описание |
|----------|-----|----------|
| `name` | `str` | Модель эмбеддера |
| `use_transformers` | `bool` | Использовать Transformers версию |

## Все параметры командной строки <span id="параметры-командной-строки"></span>

### Извлечение F0 кривой <span id="извлечение-f0-кривой-cli"></span>

первичные аргументы - `f0_extractor.py`

| Аргументы | Параметр | Тип | Описание |
|-----------|----------|-----|----------|
| `-i`, `--i`, `-input`, `--input` | `input` | `str \| Path` | Путь к входному аудиофайлу |
| `-f0m`, `-f0_method`, `--f0_method`, `--f0-method` | `f0_method` | `str` | Метод извлечения F0: rmvpe+, hpa-rmvpe, fcpe, mangio-crepe, mangio-crepe-tiny, harvest, pm, pyin (по умолчанию: rmvpe+) |
| `-f0min`, `--f0_min`, `--f0-min` | `f0_min` | `int` | Минимальная частота F0 в Гц (по умолчанию: 50) |
| `-f0max`, `--f0_max`, `--f0-max` | `f0_max` | `int` | Максимальная частота F0 в Гц (по умолчанию: 1100) |
| `-o`, `-out`, `-output`, `--output`, `--output_path`, `--output-path` | `output_path` | `str \| Path` | Путь для сохранения JSON-файла с F0 кривой. Если не указан, сохраняется рядом с аудиофайлом |

### Преобразование вокала <span id="преобразование-вокала-cli"></span>

первичные аргументы - `infer.py infer`

| Аргументы | Параметр | Тип | Описание |
|-----------|----------|-----|----------|
| `-i`, `--i`, `-input`, `--input`, `--input_files`, `--input-files` | `input` | `str \| Path \| list` | Путь к входному аудиофайлу, папке с файлами или список путей |
| `-o`, `-out`, `-output`, `--output`, `--output_dir`, `--output-dir` | `output_dir` | `str \| Path` | Директория для сохранения результатов (по умолчанию: текущая папка) |
| `-m`, `-model`, `--model_path`, `--model-path` | `checkpoint_path` | `str` | Путь к файлу чекпоинта модели (*.pth) |
| `-idx`, `-index`, `--index_path`, `--index-path` | `index_path` | `str` | Путь к файлу индекса FAISS (*.index) для улучшения качества (опционально) |
| `-p`, `-pitch`, `--pitch` | `pitch` | `int` | Изменение высоты тона в полутонах (по умолчанию: 0) |
| `-f0m`, `-f0_method`, `--f0_method`, `--f0-method` | `f0_method` | `str` | Метод извлечения F0: rmvpe+, hpa-rmvpe, fcpe, mangio-crepe, mangio-crepe-tiny, harvest, pm, pyin (по умолчанию: rmvpe+) |
| `-idxr`, `-index_rate`, `--index_rate`, `--index-rate` | `index_rate` | `float` | Влияние индекса (0-1). Чем ниже значение, тем больше голос похож на исходный (по умолчанию: 0.75) |
| `-ve`, `-volume_envelope`, `--volume_envelope`, `--volume-envelope` | `volume_envelope` | `float` | Соотношение огибающих громкости (0-1) (по умолчанию: 0.25) |
| `-pr`, `-protect`, `--protect` | `protect` | `float` | Защита согласных (0-0.5). Предотвращает роботизацию (по умолчанию: 0.33) |
| `-hl`, `-hop_length`, `--hop_length`, `--hop-length` | `hop_length` | `int` | Длина шага для извлечения F0 методами CREPE (по умолчанию: 128) |
| `-emb`, `-embedder`, `--embedder_model`, `--embedder-model` | `embedder` | `str` | Модель эмбеддера: hubert_base или spin (по умолчанию: hubert_base) |
| `-tf`, `-use_transformers`, `--use_transformers`, `--use-transformers` | `use_transformers` | `bool` | Использовать стек Transformers для эмбеддера (вместо Fairseq) |
| `-of`, `-output_fmt`, `--output_format`, `--output-format` | `output_format` | `str` | Формат выходного аудио. Доступны: mp3, wav, flac и др. (по умолчанию: mp3) |
| `-stm`, `-stereo_mode`, `--stereo_mode`, `--stereo-mode` | `stereo_mode` | `str` | Стерео-режим: mono, left/right, sim/dif (по умолчанию: mono) |
| `-f0min`, `--f0_min`, `--f0-min` | `f0_min` | `int` | Минимальная частота F0 в Гц (по умолчанию: 50) |
| `-f0max`, `--f0_max`, `--f0-max` | `f0_max` | `int` | Максимальная частота F0 в Гц (по умолчанию: 1100) |
| `-chd`, `-chunk_duration`, `--chunk_duration`, `--chunk-duration` | `chunk_duration` | `int` | Длина чанка для обработки в секундах (по умолчанию: 7) |
| `-tm`, `-tmplt`, `--template` | `template` | `str` | Шаблон имени выходного файла. Доступные ключи: NAME, F0METHOD, PITCH. Пример: NAME_F0METHOD_PITCH ([Подробнее](#шаблон-имени)) |

### Преобразование вокала с кастомной F0 кривой <span id="преобразование-вокала-с-кастомной-f0-кривой-cli"></span>

первичные аргументы - `infer.py infer_custom_f0`

| Аргументы | Параметр | Тип | Описание |
|-----------|----------|-----|----------|
| `-i`, `--i`, `-input`, `--input` | `input` | `str \| Path` | Путь к входному аудиофайлу |
| `-o`, `-out`, `-output`, `--output`, `--output_dir`, `--output-dir` | `output_dir` | `str \| Path` | Директория для сохранения результатов (по умолчанию: текущая папка) |
| `-m`, `-model`, `--model_path`, `--model-path` | `checkpoint_path` | `str` | Путь к файлу чекпоинта модели (*.pth) |
| `-idx`, `-index`, `--index_path`, `--index-path` | `index_path` | `str` | Путь к файлу индекса FAISS (*.index) для улучшения качества (опционально) |
| `-p`, `-pitch`, `--pitch` | `pitch` | `int` | Изменение высоты тона в полутонах (по умолчанию: 0) |
| `-f0f`, `-f0_file`, `--f0_file`, `--f0-file` | `f0_file` | `str \| Path` | Путь к JSON-файлу с кастомной кривой F0 ([Пример JSON-файла](#формат-данных-в-json-файле-кривой-f0)) |
| `-idxr`, `-index_rate`, `--index_rate`, `--index-rate` | `index_rate` | `float` | Влияние индекса (0-1). Чем ниже значение, тем больше голос похож на исходный (по умолчанию: 0.75) |
| `-ve`, `-volume_envelope`, `--volume_envelope`, `--volume-envelope` | `volume_envelope` | `float` | Соотношение огибающих громкости (0-1) (по умолчанию: 0.25) |
| `-pr`, `-protect`, `--protect` | `protect` | `float` | Защита согласных (0-0.5). Предотвращает роботизацию (по умолчанию: 0.33) |
| `-emb`, `-embedder`, `--embedder_model`, `--embedder-model` | `embedder` | `str` | Модель эмбеддера: hubert_base или spin (по умолчанию: hubert_base) |
| `-tf`, `-use_transformers`, `--use_transformers`, `--use-transformers` | `use_transformers` | `bool` | Использовать стек Transformers для эмбеддера (вместо Fairseq) |
| `-of`, `-output_fmt`, `--output_format`, `--output-format` | `output_format` | `str` | Формат выходного аудио. Доступны: mp3, wav, flac и др. (по умолчанию: mp3) |
| `-stm`, `-stereo_mode`, `--stereo_mode`, `--stereo-mode` | `stereo_mode` | `str` | Стерео-режим: mono, left/right, sim/dif (по умолчанию: mono) |
| `-f0min`, `--f0_min`, `--f0-min` | `f0_min` | `int` | Минимальная частота F0 в Гц (по умолчанию: 50) |
| `-f0max`, `--f0_max`, `--f0-max` | `f0_max` | `int` | Максимальная частота F0 в Гц (по умолчанию: 1100) |
| `-chd`, `-chunk_duration`, `--chunk_duration`, `--chunk-duration` | `chunk_duration` | `int` | Длина чанка для обработки в секундах (по умолчанию: 7) |
| `-tm`, `-tmplt`, `--template` | `template` | `str` | Шаблон имени выходного файла. Доступные ключи: NAME, F0METHOD, PITCH. Пример: NAME_F0METHOD_PITCH ([Подробнее](#шаблон-имени)) |

### Загрузка модели эмбеддера <span id="загрузка-модели-эмбеддера-cli"></span>

первичные аргументы - `infer.py download_hubert`

| Аргументы | Параметр | Тип | Описание |
|-----------|----------|-----|----------|
| `-emb`, `-embedder`, `--embedder_model`, `--embedder-model` | `embedder` | `str` | Модель эмбеддера: hubert_base или spin (по умолчанию: hubert_base) |
| `-tf`, `-use_transformers`, `--use_transformers`, `--use-transformers` | `use_transformers` | `bool` | Использовать стек Transformers для эмбеддера (вместо Fairseq) |


