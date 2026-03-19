## Установка

```sh
python install.py
```

## Веб-интерфейс

```sh
python separator.py app

--add_app      Включить дополнительные приложения (Ансамбль, Вычитание и Обработка аудио)
--use_plugins  Включить поддержку плагинов
--vbach        Включить Vbach
--port         Порт сервера
--share        Создать публичную ссылку
```

## Информация о моделях

```sh
python separator.py info

-limit     Лимит отображаемых моделей
-stem      Фильтр по стему
-c         Фильтр по категории
-t         Фильтр по типу модели
-oi        Только установленные модели

Уникальные параметры:

--update   Обновить информацию о моделях
-lc        Список категорий
-lt        Список типов моделей 
```

## Разделение через CLI

```sh
python separator.py separator -i file.mp3 -o test_output -mn bs_6stem --chunk_duration 300

-i               Входной файл/директория
-o               Директория вывода
-mn              Имя модели (например: bs_6stem)
-stem            Выбранные стемы (например: --stem vocals instrumental)
--chunk_duration Длина чанка для очень длинных файлов в секундах (например: 300)
--template       Шаблон имени (например: --template NAME_(STEM)_MODEL)
-dw              Только установка модели
```

## Переменные окружения

```sh
MVSEPLESS_LANGUAGE={ru|en} - Язык

MVSEPLESS_ECO_SEG={int} - Размер сегмента для эконом-режима при разделении (По умолчанию = 7)

VBACH_ALTPL_BASE_SEG={int} - Базовый размер сегмента для альтернативного пайплайна в Vbach (По умолчанию = 10)

VBACH_ALTPL_PREF_BASE_SEG={True|False} - Использовать ли базовый размер сегмента (По умолчанию = True)

MVSEPLESS_DPSS={True|False} - Использовать ли окно DPSS вмеcто окна Ханна? (По умолчанию = False)

MVSEPLESS_WRITE_ABS={True|False} - Возвращать ли абсюлотный путь после записи файла? (По умолчанию = False)
```
## Рабочая среда
- Google Colab  -  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noblebarkrr/mvsepless/blob/epsilon/MVSepLess_Epsilon_Colab_v2.ipynb)
- Hugging Face (ZeroGPU) - [![Open In Huggingface](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/noblebarkrr/mvsepless_zero_gpu)
