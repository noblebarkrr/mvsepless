# MVSepLess - */непростая обёртка*/ для audio-separator и Music-Source-Separation-Training

Внимание! Данный репозиторий подходит только для Google Colab!

# Фишки:

- Динамическое отображение плееров с названиями стемов в Web-UI
- Возможность выбрать выходные стемы (если у модели нет целевого инструмента)
- Инверсия выбранных стемов (при включенном "Извлечь инструментал")
- Создание инструментала на 4/6 - стемных моделях, например: ["bass", "drums", "vocals", "other", "piano", "guitar"] (при включенном "Извлечь инструментал")
- Результаты возвращаются в формате списка кортежей - [("Стем", "Путь/к/файлу")]
- Поддержка популярных форматов аудио (чтение и запись)
- Явное указание битрейта
- Использование шаблона для именования выходных файлов
- Локализованные имена стемов (только в Web-UI)

# Проблемы

- Модель BS-Roformer Inst FNO от unwa запустится только в Google Colab!

# Рабочая среда

Google Colab - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noblebarkrr/mvsepless/blob/beta/Mvsepless_Beta.ipynb)

HF Spaces CPU (Без плагинов) - [![Open In Huggingface](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/noblebarkrr/mvsepless_cpu_beta)

HF Spaces Zero GPU (Без плагинов) - [![Open In Huggingface](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/noblebarkrr/mvsepless_zero_gpu)


### Требования для установки

- Python 3.11
- Актуальная версия Pytorch
- Исправленные/модифицированные библиотеки:
> - Исправленный fairseq для совместимости с RVC и Medley-Vox в актуальной среде выполнения выдаваемой Google Colab
> - Модифицированный audio-separator для запуска остальных моделей на VR ARCH, которых нет в оригинальной библиотеке

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
("bass", "/content/output/test (bass).opus"),
("drums", "/content/output/test (drums).opus"),
("vocals", "/content/output/test (vocals).opus"),
("other", "/content/output/test (other).opus"),
("piano", "/content/output/test (piano).opus"),
("guitar", "/content/output/test (guitar).opus")
]

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
использование: app.py [--share] [--port ПОРТ] 
                      [--ngrok_token ТОКЕН]           
                      [--font ПУТЬ К ФАЙЛУ ШРИФТА]       
                      [--lang ЯЗЫК]
 
Входные аргументы:

--share                      Общий доступ [булево значение]
--port ПОРТ                  Порт сервера [число]
--ngrok_token ТОКЕН          Токен авторизации для Ngrok [строка]
--font ПУТЬ К ФАЙЛУ ШРИФТА   Использует кастомный шрифт в интерфейсе по указанному пути [строка]
--lang ЯЗЫК                  Язык интерфейса [строка]

```

##### Разделение

1. Загружаете аудио файл или указываете путь к нему
2. Ставите нужные тип и имя модели
3. Выберите выходные стемы (на усмотрение)
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

### Доступные плагины

Скоро...











