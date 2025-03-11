# mvsepless
CLI wrapper for MSST and python-audio-separator and medley-vox for Google Colab with selecting model, using the model code.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noblebarkrr/mvsepless/blob/test/mvsepless_cli_test_0_0_6.ipynb)


Обертка для MSST, python-audio-separator и medley-vox для Google Colab, с выбором модели по её коду.

No MVSep - No queue problem!

Нет MVSep - Нет очередей!

### Пример рабочей команды

```bash
!python code_infer.py -i input -o output -mcode 1130 -of wav -inst
```
Аргументы:

* '-i или --input' - Папка, откуда инференс будет брать файлы
* '-o или --output' - Папка, куда инференс будет сохранять файлы после разделения
* '-of или --output_foramt' - Формат вывода (не работает для medley-vox)
* '-mc или --modelcode' - Код модели который берёт нужную модель из списка в файле models_list.py
* '-tta или --use_tta' - Повышает качество разделения за счёт инвертирования полярности сигнала и двух каналов за каждый проход (работает только на MSST)
* '-inst или --instrum' - Сохранение инструментала (работает только на MSST)

### Требования:

Python = 3.11

Pip = последняя версия

Видеокарта = Nvidia Tesla T4

### Кредиты

## Credits 🙏

- [ZF Turbo](https://github.com/ZFTurbo) - Создатель репозитория для тренировки и инференса моделей для удаления вокала
- [beveradb](https://github.com/beveradb) - Создатель простого для использования обертки для Ultimate Vocal Remover
- [Cbeast25](https://github.com/Cbeast25) - Держит неофицальный репозиторий для medley-vox
