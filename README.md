<div align="center">

# MVSepless

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noblebarkrr/mvsepless/blob/dzeta/MVSepLess_Dzeta_Colab.ipynb)

[![Open In Huggingface](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/noblebarkrr/mvsepless_zero_gpu) (Приостановлено)

</div>

<h2> */непростая обёртка*/ для <s>audio-separator</s> и Music-Source-Separation-Training </h2>

## Фишки

- Извлечение инструментала (инверсии выбранных стемов)
- Встроенный авто-ансамбль, с возможностью выбрать основной стем для добавления в ансамбль
- Базовый итеративный ансамбль
- Дополнительные параметры разделения, как в UVR
- История обработок (Только в Web-UI)
- Пакетная обработка по умолчанию (кроме авто-ансамбля и вычитания)
- Язык интерфейса изменяется через переменную окружения `MVSEPLESS_LANGUAGE`

---

<details> <summary align="center"><b>Содержание</b></summary>

- [Установка](#️установка)
  - [Совместимость](#совместимость-кода)
  - [Подготовка среды выполнения](#подготовка-среды-выполнения-для-начала)
    - [Windows](#установка-windows)
    - [Linux](#установка-linux)
  - [Установка зависимостей](#установка-зависимостей-для-работы)
    - [Через uv](#установка-зависимостей-через-uv)
    - [Через pip](#установка-зависимостей-через-pip)
- [VBach - Форк PolGen](#vbach)
- [Специфичные атрибуты конфигов Roformer моделей](#specific_keys)
- [Шаблоны имен файлов](#шаблоны-имен)
    - [Разделние](#шаблон-разделение)
    - [Итеративный ансамбль](#шаблон-итеративный-ансамбль)
    - [Авто-ансамбль](#шаблон-авто-ансамбль)
    - [Ручной ансамбль](#шаблон-ручной-ансамбль)
    - [Вычитание](#шаблон-вычитание)
- [Типы ансамбля](#типы-ансамбля-инфо)
- [Пример пресета для авто-ансамбля](#пример-пресета-для-авто-ансамбля)
- [Пример пресета для итеративного ансамбля](#пример-итеративный-ансамбль)
- [Пример использования](#пример-использования-кода)
  - [Web-UI](#web-ui-использование)
  - [Информация о моделях](#информация-о-моделях-использование)
  - [Разделение](#разделение-использование)
  - [Разделение с кастомной моделью](#разделение-с-кастомной-моделью-использование)
  - [Итеративный ансамбль](#итеративный-ансамбль-использование)
  - [Авто-ансамбль](#авто-ансамбль-использование)
  - [Ручной ансамбль](#ручной-ансамбль-использование)
  - [Вычитание](#вычитание)
- [Параметры методов класса Separator](#параметры-методов)
  - [Информация о моделях](#информация-о-моделях-метод)
  - [Скачать модель](#скачать-модель-метод)
  - [Разделение](#разделение-separatorseparate)
  - [Разделение с кастомной моделью](#разделение-с-кастомной-моделью-метод)
  - [Авто-ансамбль](#авто-ансамбль-метод)
  - [Ручной ансамбль](#ручной-ансамбль-метод)
  - [Вычитание](#вычитание-метод)
- [Все параметры командной строки](#все-параметры-командной-строки)
  - [Информация о моделях](#информация-о-моделях-cli)
  - [Разделение](#разделение-cli)
  - [Разделение с кастомной моделью](#разделение-с-кастомной-моделью-cli)
  - [Итеративный ансамбль](#итеративный-ансамбль-cli)
  - [Авто-ансамбль](#авто-ансамбль-cli)
  - [Ручной ансамбль](#ручной-ансамбль-cli)
  - [Вычитание](#вычитание-cli)
  - [Web-UI](#web-ui-cli)

</details>

---

## 🛠️ Установка <span id="установка"></span>

### Совместимость: <span id="совместимость-кода"></span>

- *Python: 3.10-3.12* (на более новых версиях не тестировалось)
- *Pytorch: 1.13-latest*

### Подготовка среды выполнения <span id="подготовка-среды-выполнения-для-начала"></span>

#### **Windows:** <span id="установка-windows"></span>
1. Установите Python с включенными опциями `Add Python 3.x to PATH` и `Disable path length limit`: https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe
2. Скачайте архив с FFMPEG, распакуйте его и добавьте распакованную папку в переменную PATH: https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
3. Установите Microsoft Visual C++ 2015-2022 (x64): https://aka.ms/vs/17/release/vc_redist.x64.exe
4. Установите Microsoft C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
<br>Выберите `Desktop development with C++`
5. Установите PyTorch: https://pytorch.org/get-started/locally/
6. Скачайте и распакуйте архив:
https://github.com/noblebarkrr/mvsepless/archive/refs/heads/dzeta.zip
7. Откройте командную строку в распакованной папке

#### **Linux (Ubuntu/Debian):** <span id="установка-linux"></span>
```sh
apt update -y
apt upgrade -y
apt install -y wget curl git gcc libx11-dev ffmpeg build-essential cmake nano python3-full python3-dev
git clone https://github.com/noblebarkrr/mvsepless -b dzeta
cd mvsepless

# Если пакеты не устанваливаются без виртуального окружения
python3 -m venv env
source env/bin/activate

python3 -m pip install ninja cmake
```

### Установка зависимостей <span id="установка-зависимостей-для-работы"></span>
- через uv: <span id="установка-зависимостей-через-uv"></span>
```sh
pip install uv
# Обычные зависимости
uv pip install --no-cache-dir -r requirements.txt
# Устаревшие зависимости (только Python 3.10)
uv pip install --no-cache-dir -r requirements_old_torch_py310.txt
```

- через pip (медленее): <span id="установка-зависимостей-через-pip"></span>
```sh
# Обычные зависимости
pip install -r requirements.txt
# Устаревшие зависимости (только Python 3.10)
pip install -r requirements_old_torch_py310.txt
```

## VBach (форк PolGen-RVC) <span id="vbach"></span>

### [Документация](vbach_lib/README.md)

## Специфичные атрибуты конфигов Roformer моделей <span id="specific_keys"></span>

### Все атрибуты находятся [здесь](models/ROFORMER_SPECIFIC_KEYS.md)

## Шаблоны имен файлов <span id="шаблоны-имен"></span>

**Для разделения (`separate`, `custom_separate`):** <span id="шаблон-разделение"></span>
- `NAME` — имя входного файла (без расширения)
- `STEM` — название стема (например: vocals, instrumental)
- `MODEL` — имя модели (например: bs_6stem)

Пример: `NAME_STEM_MODEL` → `Song_vocals_bs_6stem`

---

**Для итеративного ансамбля:** <span id="шаблон-итеративный-ансамбль"></span>
- `NAME` — имя входного файла (без расширения)
- `ITER` — номер итерации (iter_N)

Пример: `NAME_ITER` → `Song_iter_3`

---

**Для авто-ансамбля (`auto_ensemble`):** <span id="шаблон-авто-ансамбль"></span>
- `NAME` — имя входного файла (без расширения)
- `TYPE` — тип ансамбля (например: min_fft, avg_fft)
- `COUNT` — количество используемых моделей/файлов

Пример: `NAME_COUNT_TYPE` → `Song_7_min_fft`

---

**Для ручного ансамбля (`manual_ensemble`):** <span id="шаблон-ручной-ансамбль"></span>
- `NAME` — имя входного файла (без расширения)
- `TYPE` — тип ансамбля (например: min_fft, avg_fft)

Пример: `NAME_TYPE` → `Song_min_fft`

---

**Для вычитания (`subtract`):** <span id="шаблон-вычитание"></span>
- `NAME` — имя входного файла (без расшириения)
- `TYPE` — тип инверсии (например: waveform, spectrogram)

Пример: `NAME_TYPE` → `Song_waveform`

## Типы ансамбля <span id="типы-ансамбля-инфо"></span>

| Тип | Описание |
|-----|----------|
| `avg_fft` | Среднее, стабильный результат (требуются веса) |
| `median_fft` | Медиана (эффективно от 3+ аудио) |
| `min_fft` | Минимум, более чистый результат |
| `max_fft` | Максимум, более полный, но "грязный" результат |

## Пример пресета для авто-ансамбля <span id="пример-авто-ансамбль"></span>
### Python <span id="пример-авто-ансамбль-python"></span>
```python
[
    ["имя_модели1", "основной_стем1", False, 1],
    ["имя_модели2", "основной_стем2", False, 2],
    ...
]
```
```python
[
    [str, str, bool, float],
    [str, str, bool, float],
    ...
]
```
### JSON <span id="пример-авто-ансамбль-json"></span>
```json
[
    ["имя_модели1", "основной_стем1", false, 1],
    ["имя_модели2", "основной_стем2", false, 2],
    ...
]
```

```python
[
    [str, str, bool, float],
    [str, str, bool, float],
    ...
]
```
### В командной строке <span id="пример-авто-ансамбль-cli"></span>
```sh
--flow имя_модели1:основной_стем_1:False:1 имя_модели2:основной_стем_2:False:2 ...
```

```python
--flow str:str:bool:float str:str:bool:float
```

---

## Пример пресета для итеративного ансамбля <span id="пример-итеративный-ансамбль"></span>
### Python <span id="пример-итеративного-ансамбля-python"></span>
```python
[
    ["имя_модели1", "основной_стем1", False],
    ["имя_модели2", "основной_стем2", False],
    ...
]
```

```python
[
    [str, str, bool],
    [str, str, bool],
    ...
]
```

### JSON <span id="пример-итеративного-ансамбля-json"></span>
```json
[
    ["имя_модели1", "основной_стем1", false],
    ["имя_модели2", "основной_стем2", false],
    ...
]
```

### В командной строке <span id="пример-итеративного-ансамбля-cli"></span>
```sh
--flow имя_модели1:основной_стем_1:False имя_модели2:основной_стем_2:False ...
```

```python
--flow str:str:bool str:str:bool
```


## 🚀 Пример использования <span id="пример-использования-кода"></span>

### Web-UI <span id="web-ui-использование"></span>

#### Через командную строку

```sh
# Если не указан флаг --full, то будет запущена версия для Hugging Face Spaces
python app.py --share --full
```
---
#### Напрямую

```python
from app import App

app = App()
app_ui = app.UI(theme, False)
app_ui.launch(
  allowed_paths=["/"],
  debug=True, 
  share=True, # Если нужно сделать общий доступ к Web-UI
  server_port=7860,
  server_name="0.0.0.0"
)
```

---

### Информация о моделях <span id="информация-о-моделях-использование"></span>

#### Через командную строку

```sh
# Список с фильтром
python inference.py info -limit 0 -stem "vocals"
# Полный список
python inference.py info
# Обновление информации о моделях
python inference.py info -update
```

#### Напрямую

```python
from inference import Separator

separator = Separator() # Инициализация класса
separator.update_info() # Обновление информации о моделях
separator.show_info(
  0, # Лимит количества отображаемых моделей
  "vocals", # Название стема (если указан, будут показаны только модели с данным стемом)
  False # Только установленные модели (при True)
)
```

---

### Разделение <span id="разделение-использование"></span>

#### Через командную строку

```sh
python inference.py separate -i "audio.mp3" -o "output" -of mp3 -mn "bs_6stem" -tm "NAME_STEM"
```

#### Напрямую

```python
from inference import Separator

separator = Separator()
model_name = "bs_6stem"
separator.download(model_name) # Скачивание модели без запуска инференса

results = separator.separate(
  "audio.mp3",           # Входной файл/папка
  "output",              # Директория вывода
  "mp3",                 # Формат вывода
  "NAME_STEM",           # Шаблон имени файла
  model_name             # Имя модели
)
```

---

### Разделение с кастомной моделью <span id="разделение-с-кастомной-моделью-использование"></span>

#### Через командную строку

```sh
python inference.py custom_separate -i "audio.mp3" -o "output" -of mp3 -ckpt "bs_6stem.ckpt" -conf "bs_6stem_config.yaml" -mt "bs_roformer" -tm "NAME_STEM"
```

#### Напрямую

```python
from inference import Separator

separator = Separator()

results = separator.custom_separate(
  "audio.mp3",                   # Входной файл/папка
  "output",                      # Директория вывода
  "mp3",                         # Формат вывода
  "NAME_STEM",                   # Шаблон имени файла
  "bs_roformer",                 # Тип модели
  "bs_6stem.ckpt",               # Путь к чекпоинту
  "bs_6stem_config.yaml"         # Путь к конфигу
)
```

---

### Итеративный ансамбль (пример использования) <span id="итеративный-ансамбль-использование"></span>

#### Через командную строку

[Пример пресета для итеративного ансамбля](#пример-итеративного-ансамбля-cli)

```sh
# Пресет в виде списка
python inference.py iterative_ensemble -i "audio.mp3" -o "output" -of mp3 -tm "NAME_ITER" -flow "bs_6stem:vocals:False" "mbr_inst1e_unwa:vocals:False" -n 4

# С сохранением промежуточных результатов
python inference.py iterative_ensemble -i "audio.mp3" -o "output" -of mp3 -tm "NAME_ITER" -flow "bs_6stem:vocals:False" "mbr_inst1e_unwa:vocals:False" -n 4 -save_intermediate

# Пресет в виде json-файла
python inference.py iterative_ensemble -i "audio.mp3" -o "output" -of mp3 -tm "NAME_ITER" -json "iterative_flow.json" -n 4
```

#### Напрямую

```python
from inference import Separator

separator = Separator()
flow = [
    ["bs_6stem", "vocals", True],      # [модель, осн. стем, инверсия]
    ["mbr_inst1e_unwa", "vocals", False]
]

result_path, intermediate_files = separator.iterative_ensemble(
    "audio.mp3",        # Входной файл
    "output",           # Директория вывода
    flow,               # Пресет ансамбля
    num_iters=4,        # Количество итераций
    output_format="mp3",# Формат вывода
    template="NAME_ITER", # Шаблон имени файла
    save_intermediate=True  # Сохранять промежуточные результаты
)

print(f"Финальный результат: {result_path}")
print(f"Промежуточные файлы: {intermediate_files}")
```

---

### Авто-ансамбль <span id="авто-ансамбль-использование"></span>

#### Через командную строку

[Пример пресета для авто-ансамбля](#пример-авто-ансамбль-cli)

```sh
# Пресет в виде списка списков
python inference.py auto_ensemble -i "audio.mp3" -o "output" -of mp3 -type avg_fft -tm "NAME_TYPE" -flow "bs_6stem:vocals:True:1" "mbr_inst1e_unwa:other:False:1"

# Пресет в виде json-файла
python inference.py auto_ensemble -i "audio.mp3" -o "output" -of mp3 -type avg_fft -tm "NAME_TYPE" -json "ensemble_flow.json"
```

#### Напрямую

```python
from inference import Separator

separator = Separator()
flow = [
    ["bs_6stem", "vocals", True, 1],      # [модель, осн. стем, инверсия, вес]
    ["mbr_inst1e_unwa", "other", False, 1]
]

result, invert_result = separator.auto_ensemble(
  "audio.mp3",           # Входной файл
  "output",              # Директория вывода
  flow,                  # Пресет ансамбля
  "NAME_TYPE",           # Шаблон имени файла
  "avg_fft",             # Тип ансамбля
  "mp3"                  # Формат вывода
)
```

---

### Ручной ансамбль <span id="ручной-ансамбль-использование"></span>

#### Через командную строку

```sh
python inference.py manual_ensemble -i "audio1.mp3" "audio2.mp3" -o "output" -of mp3 -type avg_fft -tm "NAME_TYPE" -w 1 1
```

#### Напрямую

```python
from inference import Separator

separator = Separator()

result = separator.manual_ensemble(
  ["audio1.mp3", "audio2.mp3"],  # Входные файлы
  "output",                      # Директория вывода
  [1, 1],                        # Веса
  "NAME_TYPE",                   # Шаблон имени файла
  "avg_fft",                     # Тип ансамбля
  "mp3"                          # Формат вывода
)
```

---

### Вычитание <span id="вычитание-использование"></span>

#### Через командную строку

```sh
python inference.py subtract -i1 "audio1.mp3" -i2 "audio2.mp3" -o "output" -of mp3 -tm "NAME_TYPE" 
```

#### Напрямую

```python
from inference import Separator

separator = Separator()

result = separator.manual_ensemble(
  "audio1.mp3",                  # Оригинал
  "audio2.mp3",                  # Стем
  "output",                      # Директория вывода
  "mp3"                          # Формат вывода
  [1, 1],                        # Веса
  False,                         # Использовать ли спектрограмму при вычитании?
  "avg_fft",                     # Тип ансамбля
)
```

## 📋 Параметры методов класса Separator <span id="параметры-методов"></span>

### Информация о моделях (`Separator().show_info()`) <span id="информация-о-моделях-метод"></span>

| Параметр | Тип | Описание |
|----------|-----|----------|
| `limit` | `int` | Лимит количества отображаемых моделей (0 или None — без лимита) |
| `stem` | `str` | Название стема — показывать только модели с данным стемом |
| `only_installed` | `bool` | Показывать только установленные модели |

### Скачать модель (`Separator().download()`)

| Параметр | Тип | Описание |
|----------|-----|----------|
| `model_name` | `str` | Имя модели для скачивания |

### Разделение (`Separator().separate()`) <span id="разделение-метод"></span>

| Параметр | Тип | Описание |
|----------|-----|----------|
| `input_files` | `str \| Path \| list` | Путь к файлу/папке или список путей |
| `output_dir` | `str \| Path` | Директория для сохранения результатов |
| `output_format` | `str` | Формат аудио (`mp3`, `wav`, `flac` и др.) |
| `template` | `str` | Шаблон имени файла (ключи: `NAME`, `STEM`, `MODEL`) ([Подробнее](#шаблон-разделение)) |
| `model_name` | `str` | Имя модели |
| `extract_instrumental` | `bool` | Создание инверсии выбранных стемов (стем `invert`) |
| `use_spec_invert` | `bool` | Вычитание из спектрограммы вместо противофазы |
| `invert_plus` | `bool` | Использует сложение невыбранных стемов при ссоздании инструментала |
| `selected_stems` | `list` | Список стемов для сохранения (если не указан — все) |
| `add_params` | `dict` | Дополнительные параметры разделения |

**Возвращает**: 
```python
[
  [имя_файла (str), [[стем (str), путь_к_файлу (str)], ...]],
  ...
]
```

### Разделение с кастомной моделью (`Separator().custom_separate()`) <span id="разделение-с-кастомной-моделью-метод"></span>

| Параметр | Тип | Описание |
|----------|-----|----------|
| `input_files` | `str \| Path \| list` | Путь к файлу/папке или список путей |
| `output_dir` | `str \| Path` | Директория для сохранения результатов |
| `output_format` | `str` | Формат аудио (`mp3`, `wav`, `flac` и др.) |
| `template` | `str` | Шаблон имени файла (ключи: `NAME`, `STEM`, `MODEL`) ([Подробнее](#шаблон-разделение)) |
| `model_type` | `str` | Тип модели (например, `bs_roformer`) |
| `ckpt` | `str \| Path` | Путь к чекпоинту (`*.ckpt`) |
| `conf` | `str \| Path` | Путь к YAML конфигу (`*.yaml`) |
| `extract_instrumental` | `bool` | Создание инверсии выбранных стемов |
| `use_spec_invert` | `bool` | Вычитание из спектрограммы вместо противофазы |
| `invert_plus` | `bool` | Использует сложение невыбранных стемов при ссоздании инструментала |
| `selected_stems` | `list` | Список стемов для сохранения |
| `add_params` | `dict` | Дополнительные параметры разделения |

**Возвращает**: аналогично `separate()`

### Итеративный ансамбль (`Separator().iterative_ensemble()`) <span id="итеративный-ансамбль-метод"></span>

| Параметр | Тип | Описание |
|----------|-----|----------|
| `input_file` | `str \| Path` | Путь к входному файлу |
| `output_dir` | `str \| Path` | Директория для сохранения результатов |
| `flow` | `list` | Пресет для ансамбля ([Пример](#пример-итеративного-ансамбля-python)) |
| `num_iters` | `int` | Количество итераций (по умолчанию: 4) |
| `output_format` | `str` | Формат аудио (`mp3`, `wav`, `flac` и др.) |
| `template` | `str` | Шаблон имени (ключи: `NAME`, `ITER`) ([Подробнее](#шаблон-итеративный-ансамбль)) |
| `save_intermediate` | `bool` | Сохранять промежуточные результаты каждой итерации |

**Возвращает**: `(путь_к_финальному_ансамблю, список_промежуточных_файлов)`

### Авто-ансамбль (`Separator().auto_ensemble()`) <span id="авто-ансамбль-метод"></span>

| Параметр | Тип | Описание |
|----------|-----|----------|
| `input_file` | `str \| Path` | Путь к входному файлу |
| `output_dir` | `str \| Path` | Директория для сохранения результатов |
| `flow` | `list` | Пресет для ансамбля ([Пример](#пример-авто-ансамбль-python)) |
| `template` | `str` | Шаблон имени (ключи: `NAME`, `TYPE`, `COUNT`) ([Подробнее](#шаблон-авто-ансамбль)) |
| `etype` | `str` | Тип ансамбля (`avg_fft`, `median_fft`, `min_fft`, `max_fft`) ([Подробнее](#типы-ансамбля-инфо)) |
| `output_format` | `str` | Формат аудио |
| `use_spec_invert` | `bool` | Вычитание из спектрограммы вместо противофазы |
| `save_primary_stems` | `bool` | Сохранять основные стемы в output_dir |

**Возвращает**: `(путь_к_ансамблю, путь_к_инверсии, список_основных_стемов)`

### Ручной ансамбль (`Separator().manual_ensemble()`) <span id="ручной-ансамбль-метод"></span>

| Параметр | Тип | Описание |
|----------|-----|----------|
| `input_files` | `list` | Список путей к аудиофайлам |
| `output_dir` | `str \| Path` | Директория для сохранения результата |
| `weights` | `list` | Веса файлов (обязательно для `avg_fft`) |
| `template` | `str` | Шаблон имени (ключи: `NAME`, `TYPE`) ([Подробнее](#шаблон-ручной-ансамбль)) |
| `etype` | `str` | Тип ансамбля (`avg_fft`, `median_fft`, `min_fft`, `max_fft`) ([Подробнее](#типы-ансамбля-инфо)) |
| `output_format` | `str` | Формат аудио |

**Возвращает**: `str` — путь к результату ансамбля

### Вычитание (`Separator().subtract()`) <span id="вычитание-метод"></span>

| Параметр | Тип | Описание |
|----------|-----|----------|
| `audio1` | `str \| Path` | Путь к оригинальному файлу |
| `audio2` | `str \| Path` | Путь к файлу стема|
| `output_dir` | `str \| Path` | Директория для сохранения результата |
| `output_format` | `str` | Формат аудио |
| `use_spec_invert` | `bool` | Вычитание из спектрограммы вместо противофазы |
| `template` | `str` | Шаблон имени (ключи: `NAME`, `TYPE`) ([Подробнее](#шаблон-вычитание)) |


**Возвращает**: `str` — путь к результату вычитания

## Все параметры командной строки <span id="все-параметры-cli"></span>

### Информация о моделях <span id="информация-о-моделях-cli"></span>

первичные аргументы - `inference.py info`

| Аргументы | Параметр | Тип | Описание |
|-----------|----------|-----|----------|
| `-u`, `-update`, `--update` | `update` | `bool` | Обновить кэш информации о моделях из репозитория |
| `-clear`, `-clear_cache`, `-clear-cache`, `--clear_cache`, `--clear-cache` | `clear_cache` | `bool` | Очистить кэш информации о моделях |
| `-mn`, `-model`, `--model_name`, `--model-name` | `model_name` | `str` | Имя модели для скачивания (по умолчанию: bs_6stem) |
| `-dw`, `-download`, `--download` | `download` | `bool` | Скачать указанную модель (требуется -mn) |
| `-l`, `-limit`, `--limit` | `limit` | `int` | Лимит количества отображаемых моделей (0 или None - без лимита) |
| `-s`, `-stem`, `--stem` | `stem` | `str` | Показать только модели, которые содержат указанный стем (например: vocals, drums) |
| `-oi`, `-installed`, `--only_installed`, `--only-installed` | `only_installed` | `bool` | Показывать только установленные (скачанные) модели |

### Разделение <span id="разделение-cli"></span>

первичные аргументы - `inference.py separate`

| Аргументы | Параметр | Тип | Описание |
|-----------|----------|-----|----------|
| `-i`, `--i`, `-input`, `--input`, `--input_files`, `--input-files` | `input` | `str \| Path \| list` | Путь к входному аудиофайлу, папке с файлами или список путей |
| `-o`, `-out`, `-output`, `--output`, `--output_dir`, `--output-dir` | `output_dir` | `str \| Path` | Директория для сохранения результатов (по умолчанию: текущая папка) |
| `-of`, `-output_fmt`, `--output_format`, `--output-format` | `output_format` | `str` | Формат выходного аудио. Доступны: mp3, wav, flac и др. (по умолчанию: mp3) |
| `-tm`, `-tmplt`, `--template` | `template` | `str` | Шаблон имени выходного файла. Доступные ключи: NAME, STEM, MODEL. Пример: NAME_STEM_MODEL ([Подробнее](#шаблон-разделение)) |
| `-mn`, `-model`, `--model_name`, `--model-name` | `model_name` | `str` | Имя модели для разделения (по умолчанию: bs_6stem) |
| `-inst`, `-ext_inst`, `-ext-inst`, `--extract_instrumental`, `--extract-instrumental` | `extract_instrumental` | `bool` | Создать инверсию выбранных стемов (инструментал) - стем с именем 'invert' |
| `-ispec`, `-spec_invert`, `-spec-invert`, `--use_spec_invert`, `--use-spec-invert` | `use_spec_invert` | `bool` | Использовать вычитание из спектрограммы вместо противофазы при создании инверсии |
| `-st`, `--st`, `-stems`, `--stems`, `--selected_stems`, `--selected-stems` | `selected_stems` | `list` | Список стемов для сохранения (например: vocals drums). Если не указаны - сохраняются все стемы |
| `--{param_name}` | `add_params.{param_name}` | `int/float/str/bool` | Дополнительный параметр разделения |

### Разделение с кастомной моделью <span id="разделение-с-кастомной-моделью-cli"></span>

первичные аргументы - `inference.py custom_separate`

| Аргументы | Параметр | Тип | Описание |
|-----------|----------|-----|----------|
| `-i`, `--i`, `-input`, `--input`, `--input_files`, `--input-files` | `input` | `str \| Path \| list` | Путь к входному аудиофайлу, папке с файлами или список путей |
| `-o`, `-out`, `-output`, `--output`, `--output_dir`, `--output-dir` | `output_dir` | `str \| Path` | Директория для сохранения результатов (по умолчанию: текущая папка) |
| `-of`, `-output_fmt`, `--output_format`, `--output-format` | `output_format` | `str` | Формат выходного аудио. Доступны: mp3, wav, flac и др. (по умолчанию: mp3) |
| `-tm`, `-tmplt`, `--template` | `template` | `str` | Шаблон имени выходного файла. Доступные ключи: NAME, STEM, MODEL. Пример: NAME_STEM_MODEL ([Подробнее](#шаблон-разделение)) |
| `-mt`, `-mtype`, `--model_type`, `--model-type` | `model_type` | `str` | Тип модели (например: bs_roformer, demucs, mdx). По умолчанию: bs_roformer |
| `-ckpt`, `--ckpt`, `-checkpoint`, `--checkpoint`, `--checkpoint_path`, `--checkpoint-path` | `checkpoint_path` | `str \| Path` | Путь к файлу чекпоинта модели (*.ckpt или *.pth) |
| `-conf`, `--conf`, `-config`, `--config`, `--config_path`, `--config-path` | `config_path` | `str \| Path` | Путь к конфигурационному файлу модели (*.yaml) |
| `-inst`, `-ext_inst`, `-ext-inst`, `--extract_instrumental`, `--extract-instrumental` | `extract_instrumental` | `bool` | Создать инверсию выбранных стемов (инструментал) - стем с именем 'invert' |
| `-ispec`, `-spec_invert`, `-spec-invert`, `--use_spec_invert`, `--use-spec-invert` | `use_spec_invert` | `bool` | Использовать вычитание из спектрограммы вместо противофазы при создании инверсии |
| `-st`, `--st`, `-stems`, `--stems`, `--selected_stems`, `--selected-stems` | `selected_stems` | `list` | Список стемов для сохранения (например: vocals drums). Если не указаны - сохраняются все стемы |
| `--{param_name}` | `add_params.{param_name}` | `int/float/str/bool` | Дополнительный параметр разделения |

### Итеративный ансамбль <span id="итеративный-ансамбль-cli"></span>

первичные аргументы - `inference.py iterative_ensemble`

| Аргументы | Параметр | Тип | Описание |
|-----------|----------|-----|----------|
| `-i`, `--i`, `-input`, `--input`, `--input_file`, `--input-file` | `input` | `str \| Path` | Путь к входному аудиофайлу |
| `-o`, `-out`, `-output`, `--output`, `--output_dir`, `--output-dir` | `output_dir` | `str \| Path` | Директория для сохранения результатов (по умолчанию: текущая папка) |
| `-of`, `-output_fmt`, `--output_format`, `--output-format` | `output_format` | `str` | Формат выходного аудио. Доступны: mp3, wav, flac и др. (по умолчанию: mp3) |
| `-tm`, `-tmplt`, `--template` | `template` | `str` | Шаблон имени выходного файла. Доступные ключи: NAME, ITER. Пример: NAME_ITER ([Подробнее](#шаблон-итеративный-ансамбль)) |
| `-n`, `-iters`, `--num_iters`, `--num-iters` | `num_iters` | `int` | Количество итераций (по умолчанию: 4). На каждой итерации применяется весь набор моделей к остатку от предыдущей итерации |
| `-save_intermediate`, `-save-intermediate`, `--save_intermediate`, `--save-intermediate` | `save_intermediate` | `bool` | Сохранять промежуточные результаты каждой итерации |
| `-flow`, `--flow` | `flow` | `list` | Пресет в виде строк: `МОДЕЛЬ`:`ОСНОВНОЙ_СТЕМ`:`ИНВЕРСИЯ`. ([Пример](#пример-итеративного-ансамбля-cli)) |
| `-json`, `-preset`, `-preset_json`, `-preset-json`, `--preset_json`, `--preset-json` | `preset` | `str` | Путь к JSON-файлу с пресетом ансамбля ([Пример](#пример-итеративного-ансамбля-json)) |

### Авто-ансамбль <span id="авто-ансамбль-cli"></span>

первичные аргументы - `inference.py auto_ensemble`

| Аргументы | Параметр | Тип | Описание |
|-----------|----------|-----|----------|
| `-i`, `--i`, `-input`, `--input`, `--input_file`, `--input-file` | `input` | `str \| Path` | Путь к входному аудиофайлу |
| `-o`, `-out`, `-output`, `--output`, `--output_dir`, `--output-dir` | `output_dir` | `str \| Path` | Директория для сохранения результатов (по умолчанию: текущая папка) |
| `-of`, `-output_fmt`, `--output_format`, `--output-format` | `output_format` | `str` | Формат выходного аудио. Доступны: mp3, wav, flac и др. (по умолчанию: mp3) |
| `-tm`, `-tmplt`, `--template` | `template` | `str` | Шаблон имени выходного файла. Доступные ключи: NAME, TYPE, COUNT. Пример: NAME_COUNT_TYPE ([Подробнее](#шаблон-авто-ансамбль)) |
| `-t`, `-type`, `-etype`, `--ensemble_type`, `--ensemble-type` | `ensemble_type` | `str` | Тип ансамбля: avg_fft, median_fft, min_fft, max_fft (по умолчанию: avg_fft; [Подробнее](#типы-ансамбля-инфо)) |
| `-ispec`, `-spec_invert`, `-spec-invert`, `--use_spec_invert`, `--use-spec-invert` | `use_spec_invert` | `bool` | Использовать вычитание из спектрограммы вместо противофазы при создании инверсии |
| `-save_stems`, `-save-stems`, `-save_primary_stems`, `--save-primary-stems` | `save_primary_stems` | `bool` | Сохранять основные стемы в выходную директорию |
| `-flow`, `--flow` | `flow` | `list` | Пресет в виде строк: `МОДЕЛЬ`:`ОСНОВНОЙ_СТЕМ`:`ИНВЕРСИЯ`:`ВЕС`. ([Пример](#пример-авто-ансамбль-cli)) |
| `-json`, `-preset`, `-preset_json`, `-preset-json`, `--preset_json`, `--preset-json` | `preset` | `str` | Путь к JSON-файлу с пресетом ансамбля ([Пример](#пример-авто-ансамбль-json)) |

### Ручной ансамбль <span id="ручной-ансамбль-cli"></span>

первичные аргументы - `inference.py manual_ensemble`

| Аргументы | Параметр | Тип | Описание |
|-----------|----------|-----|----------|
| `-i`, `--i`, `-input`, `--input`, `--input_files`, `--input-files` | `input` | `list` | Список путей к аудиофайлам |
| `-o`, `-out`, `-output`, `--output`, `--output_dir`, `--output-dir` | `output_dir` | `str \| Path` | Директория для сохранения результатов (по умолчанию: текущая папка) |
| `-of`, `-output_fmt`, `--output_format`, `--output-format` | `output_format` | `str` | Формат выходного аудио. Доступны: mp3, wav, flac и др. (по умолчанию: mp3) |
| `-tm`, `-tmplt`, `--template` | `template` | `str` | Шаблон имени выходного файла. Доступные ключи: NAME, TYPE. Пример: NAME_TYPE ([Подробнее](#шаблон-ручной-ансамбль)) |
| `-t`, `-type`, `-etype`, `--ensemble_type`, `--ensemble-type` | `ensemble_type` | `str` | Тип ансамбля: avg_fft, median_fft, min_fft, max_fft (по умолчанию: avg_fft; [Подробнее](#типы-ансамбля-инфо)) |
| `-w`, `-weights`, `--weights` | `weights` | `list` | Веса для каждого аудиофайла (обязательно для типа avg_fft). Пример: -w 1.0 0.5 0.2 |

### Вычитание <span id="вычитание-cli"></span>

первичные аргументы - `inference.py subtract`

| Аргументы | Параметр | Тип | Описание |
|-----------|----------|-----|----------|
| `-i1`, `--i1`, `-input1`, `--input1`, `--input_file1`, `--input-file1` | `input_1` | `str \| Path` | Путь к оригинальному аудиофайлу (из которого вычитаем) |
| `-i2`, `--i2`, `-input2`, `--input2`, `--input_file2`, `--input-file2` | `input_2` | `str \| Path` | Путь к аудиофайлу стема (который вычитаем) |
| `-o`, `-out`, `-output`, `--output`, `--output_dir`, `--output-dir` | `output_dir` | `str \| Path` | Директория для сохранения результатов (по умолчанию: текущая папка) |
| `-of`, `-output_fmt`, `--output_format`, `--output-format` | `output_format` | `str` | Формат выходного аудио. Доступны: mp3, wav, flac и др. (по умолчанию: mp3) |
| `-tm`, `-tmplt`, `--template` | `template` | `str` | Шаблон имени выходного файла. Доступные ключи: NAME, TYPE. Пример: NAME_TYPE ([Подробнее](#шаблон-вычитание)) |
| `-ispec`, `-spec_invert`, `-spec-invert`, `--use_spec_invert`, `--use-spec-invert` | `use_spec_invert` | `bool` | Использовать вычитание из спектрограммы вместо противофазы при создании инверсии |

##№ Web-UI <span id="web-ui-cli"></span>

первичные аргументы - `app.py`

| Аргументы | Параметр | Тип | Описание |
|-----------|----------|-----|----------|
| `-s`, `-share`, `--share`, `--public`, `--gradio_share`, `--gradio-share` | `share` | `bool` | Создать публичную ссылку через Gradio Share (для доступа из интернета) |
| `-p`, `-port`, `--port`, `--server_port`, `--server-port` | `port` | `int` | Порт для запуска сервера (по умолчанию: 7860) |
| `-f`, `-full`, `--full`, `--no_hf_mode`, `--no-hf-mode` | `full` | `bool` | Запустить полную версию интерфейса (не режим Hugging Face Spaces) |

