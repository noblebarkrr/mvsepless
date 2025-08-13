# MVSepLess - */непростая обёртка*/ для audio-separator и Music-Source-Separation-Training

Этот репозиторий работает только с Google Colab!

# Фишки:

- Динамическое отображение плееров с названиями стемов в Web-UI
- Возможность выбрать выходные стемы *(если у модели нет целевого инструмента)* ***(только через командную строку)***
- Инверсия выбранных стемов (при включенном "Извлечь инструментал") ***(только через командную строку)***
- Создание инструментала на 4/6 - стемных моделях, например: ***["bass", "drums", "vocals", "other", "piano", "guitar"]*** (при включенном "Извлечь инструментал")
- Результаты возвращаются в формате списка кортежей - **[("Стем", "Путь/к/файлу")]**
- Поддержка популярных форматов аудио (чтение и запись)
- Явное указание битрейта ***(только через командную строку)***
- Использование шаблона для именования выходных файлов ***(только через командную строку)***
- Локализованные имена стемов ***(только в Web-UI)***
- Встроенный audio-separator **(Только для VR и MDX моделей)**
- Загрузка аудио из интернета (yt-dlp) прямо в интерфейсе

# Доступные плагины:

**Vbach** - Форк ***PolGen-RVC 1.2.0***, адаптированный под MVSepLess.<br>Позволяет обрабатывать автоматически стерео файлы *(левый/правый)* или *(фантомный центр/стерео база)*<br>*[(ссылка на плагин)](https://huggingface.co/noblebarkrr/mvsepless_plugins/blob/main/vbach_for_mvsepless_gamma.py)*

**EnsembLess** - Приложение для создания автоматических и ручных ансамблей моделей для улучшения результатов разделения.<br>*(Встроено в репозиторий, загружать его отдельно не требуется)*

**Inverter** - Приложение для вычитания содержимого одного файла из другого посредством спектрограммы или противофазы.<br>*(Встроено в репозиторий, загружать его отдельно не требуется)*

# Рабочая среда

Google Colab - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noblebarkrr/mvsepless/blob/gamma/Mvsepless_Gamma.ipynb)

HF Spaces Zero GPU Gamma **(Без плагинов)** - [![Open In Huggingface](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/noblebarkrr/mvsepless_gamma)


### Требования для установки

- Python 3.11
- Актуальная версия Pytorch

# Использование

### В ячейке блокнота

##### Разделение

```python
# Импорт модуля из multi_inference.py
from multi_inference import MVSEPLESS

# Инициализация класса
mvsepless = MVSEPLESS()

# Создание разделения
output_files = mvsepless.separator(
     input_file="/content/test.mp3", # Путь к файлу
     output_dir="/content/output", # Выходная папка
     model_type="bs_roformer", # Тип модели
     model_name="BS-Roformer_SW", # Имя модели
     ext_inst=True, # "Извлечь инструментал"
     vr_aggr=5, # Агрессивность для VR ARCH моделей
     output_format="opus", # Формат вывода
     output_bitrate="320k", # Битрейт выходного аудио
     template="NAME (STEM)", # Шаблон имени выходного файла
     call_method="cli", # Метод вызова инференса в обёртке
     selected_stems=["drums"]) # Выбранные стемы

# Отображение списка кортежей
print(output_files) # --> [
# ("bass", "/content/output/test (bass).opus"),
# ("drums", "/content/output/test (drums).opus"),
# ("vocals", "/content/output/test (vocals).opus"),
# ("other", "/content/output/test (other).opus"),
# ("piano", "/content/output/test (piano).opus"),
# ("guitar", "/content/output/test (guitar).opus")
# ]

```

##### Получение информации о моделях

```python
# Импорт модуля из multi_inference.py
from multi_inference import MVSEPLESS

# Инициализация класса
mvsepless = MVSEPLESS()

# Получение типов моделей
mvsepless.get_mt() # --> ["model_type1", "model_type2"] Пример вывода

# Получение имени моделей
mvsepless.get_mn("model_type") # --> ["model_name1", "model_name2"] Пример вывода

# Получение списка стемов
mvsepless.get_stems("model_type", "model_name") # --> ["stem1", "stem2"] Пример вывода

# Проверка на наличие целевого инструмента
mvsepless.get_tgt_inst("model_type", "model_name") # --> "stem" Пример вывода

```

### В командной строке (CLI)

##### Разделение

```sh

использование: multi_inference.py separate [-i ВХОДНОЙ ФАЙЛ/ПАПКА] [-o ВЫХОДНАЯ ПАПКА] [-mt ТИП МОДЕЛИ] 
                                           [-mn ИМЯ МОДЕЛИ] [-inst] [-stems "СТЕМ" "СТЕМ" "СТЕМ"...] 
                                           [-bitrate БИТРЕЙТ] [--format ФОРМАТ ВЫВОДА] 
                                           [--template ШАБЛОН ИМЕНИ ВЫХОДНОГО ФАЙЛА]
                                           [-vr_aggr АГРЕССИВНОСТЬ ДЛЯ VR ARCH МОДЕЛЕЙ]
                                           [-l_out]

Входные аргументы:

-i ВХОДНОЙ ФАЙЛ/ПАПКА, --input ВХОДНОЙ ФАЙЛ/ПАПКА  Определяет путь к входному файлу [строка]
-mt ТИП МОДЕЛИ, --model_type ТИП МОДЕЛИ            Тип модели, используемой в разделении [строка]
-mn ИМЯ МОДЕЛИ, --model_name ИМЯ МОДЕЛИ            Имя модели, используемой в разделении [строка]

Настройки разделения:

-inst, --instrumental                              Инвертирует целевой инструмент/выбранные стемы (работает только с MSST) [булево значение]
-vr_aggr, --vr_arch_aggressive                     Интенсивность извлечения стема на VR ARCH моделях [число]

Выходные настройки:

-o ВЫХОДНАЯ ПАПКА, --output ВЫХОДНАЯ ПАПКА         Определяет директорию, куда будут сохранены стемы [строка]
-stems СТЕМ СТЕМ, --stems СТЕМ СТЕМ                Выбор стемов, которые будут сохранены в папке [список]
-bitrate БИТРЕЙТ, --bitrate БИТРЕЙТ                Битрейт выходных аудио (игнорируется если формат вывода WAV, FLAC и AIFF) [строка]
-of ФОРМАТ ВЫВОДА, --format ФОРМАТ ВЫВОДА          Формат выходного аудио [строка]
--template ШАБЛОН ИМЕНИ ВЫХОДНОГО ФАЙЛА            Форматирование имени выходного файла по шаблону [строка]
-l_out, --list_output                              Отображение результатов разделения [булево значение]


```


##### Полчуение информации о моделях

```sh
использование : multi_inference.py list [-l_filter СТЕМ]

Входные аргументы:


-l_filter СТЕМ, --list_filter СТЕМ   Показывает только те модели, в которых есть указанный стем [строка]

```


### В Web-UI

##### Запуск 

```sh
использование: app.py [--share] [--debug] [--port ПОРТ]           
                      [--theme НАЗВАНИЕ ТЕМЫ]       
                      [--lang ЯЗЫК]
 
Входные аргументы:

--share                      Общий доступ [булево значение]
--debug                      Отладка ( для gradio )
--port ПОРТ                  Порт сервера [число]
--theme НАЗВАНИЕ ТЕМЫ        Указывает, какую встроенную тему использовать в интерфейсе [строка]
--lang ЯЗЫК                  Язык интерфейса [строка]

```

##### Разделение

1. Загружаете аудио файл или указываете путь к нему
2. Ставите интересующие вас тип и имя модели
4. Нажимаете "Разделить"
5. На выходе должны появится плееры с названиями стемов, которые можете скачать и прослушать

# Плагины (Только для Web-UI)

### Шаблон плагина

```
plugin.py
```

```python

import gradio as gr

TRANSLATIONS = {
    "ru": {
        "test": "Тестовый текст"
    },
    "en": {
        "test": "Test text"
    }
}

CURRENT_LANG = "ru"

def t(key, **kwargs):
    """Функция для получения перевода с подстановкой значений"""
    lang = CURRENT_LANG
    translation = TRANSLATIONS.get(lang, {}).get(key, key)
    return translation.format(**kwargs) if kwargs else translation

def set_language(lang):
    global CURRENT_LANG
    CURRENT_LANG = lang

def plugin_name(): # Имя плагина (вкладки в интерфейсе)
    return "Plugin"

def plugin(lang): # аргумент lang используется для применения языка, такого же как и в интерфейсе
    set_language(lang)
    gr.Markdown(t("test"))


```













