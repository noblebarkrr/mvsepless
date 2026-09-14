# Схема пресета

```json
{
    "nodes": {
        "node_1": {
            "id": "id_node", # ID ноды
            "type": "type", # тип ноды
            "x": 0, # расположение ноды по оси X
            "y": 0, # расположение ноды по оси Y 
            "params": {}, # словарь параметров ноды
            "ins": ["audio", ...], # может быть и пустым списком [], зависит от типа ноды и её параметров
            "outs": ["audio", ...], # может быть и пустым списком [], зависит от типа ноды и её параметров
        }, ... # node_2 и т.д.
    },
    "links": [
        {
            "fromNode": "node_1", # в какую ноду идут данные
            "fromPort": 0, # номер выхода
            "toNode": "node_2", # в какую ноду идут данные
            "toPort": 0 # номер входа
        }, ... # и т.д.
    ],
    "name": "presetless_name"
}
```

# Ноды

<span style="white-space: nowrap;"></span>

| Тип ноды `type` | Параметры `params` | Входы `ins` | Выходы `outs` | Функция |
| -------- | --------- | ----- | ------ | ------- |
| `input_file` | нет | [] | ["audio"] | Входной аудиофайл (один на весь пресет) |
| `output_file` | <span style="white-space: nowrap;">`str` \| `name_stem` - имя стема <br><br> `str` \| `output_format` - формат вывода <br><br> `bool` \| `prefer_float` - предочитать высокую точность</span> | <span style="white-space: nowrap;">["audio"]</span> | <span style="white-space: nowrap;">[]</span> | Выходной файл |
| `gain` | <span style="white-space: nowrap;">`float` \| `gain` - множитель громкости</span> | <span style="white-space: nowrap;">["audio"]</span> | <span style="white-space: nowrap;">["audio"]</span> | Изменение громкости |
| `normalize` | <span style="white-space: nowrap;">`float` \| `peak` - множитель громкости</span> | <span style="white-space: nowrap;">["audio"]</span> | <span style="white-space: nowrap;">["audio"]</span> | Нормализация уровня сигнала |
| `trim` | <span style="white-space: nowrap;">`float` \| `start` - начало (секунды) <br><br> `float` \| `end` - конец (секунды)</span> | <span style="white-space: nowrap;">["audio"]</span> | <span style="white-space: nowrap;">["audio"]</span> | Обрезка аудио по времени |
| `filter` | <span style="white-space: nowrap;">`str` \| `kind` - тип фильтра: <br>[`lp` - пропускает только низкие частоты] <br>[`hp` - пропускает только высокие частоты] <br><br> `bool` \| `fft_mode` - режим, который испольузет спектрограмму <br><br> `int` \| `cutoff` - частота среза (гц)</span> | <span style="white-space: nowrap;">["audio"]</span> | <span style="white-space: nowrap;">["audio"]</span> | Фильтр |
| `phase_shift` | <span style="white-space: nowrap;">`int` \| `degrees` - угол смещения фазы (градусы)</span> | <span style="white-space: nowrap;">["audio"]</span> | <span style="white-space: nowrap;">["audio"]</span> | Сдвиг фазы на указанное количество градусов. |
| `phase_correct` | <span style="white-space: nowrap;">`bool` \| `transfer_magnitude` - перенести магнитуду <br><br> `bool` \| `transfer_phase` - перенести фазу <br><br> `bool` \| `freq_blend_phases` - смешивать ли фазы по частотам <br><br> `int` \| `low_cutoff` - частота среза низких частот (гц) <br><br> `int` \| `high_cutoff` - частота среза высоких частот (гц)</span> | <span style="white-space: nowrap;">["target", "source"]</span> | <span style="white-space: nowrap;">["audio"]</span> | Коррекция фазы |
| `mix` | <span style="white-space: nowrap;">`int` \| `num_inputs` - количество входов</span> | <span style="white-space: nowrap;">["audio", ...]<br>Количество входов зависит от параметра `num_inputs`</span> | <span style="white-space: nowrap;">["audio"]</span> | Микширование нескольких входов в один |
| `ensemble` | <span style="white-space: nowrap;">`int` \| `num_inputs` - количество входов <br><br> `str` \| `type` - тип ансамбля: <br>[`avg_fft` - Среднее, стабильный результат] <br> [`median_fft` - Медиана (эффективно от 3+ аудио)] <br> [`min_fft` - Минимум, более чистый результат] <br> [`max_fft` - Максимум, более полный, но "грязный" результат]</span> | <span style="white-space: nowrap;">["audio", ...]<br>Количество входов зависит от параметра `num_inputs`</span> | <span style="white-space: nowrap;">["audio"]</span> | Ансамбль из нескольких входов (без ввода весов) |
| `split_stereo` | <span style="white-space: nowrap;">`str` \| `var` - вариант: <br>[`left/right` - делит аудио на левый и правый каналы] <br>[`mid/side` - делит аудио на моно и бока] <br>[`sim/dif` - делит аудио на фантомный центр и стерео-базу]</span> | <span style="white-space: nowrap;">["audio"]</span> | <span style="white-space: nowrap;">["left", "right"] - при `var` = "left/right" <br>["mid", "side"] - при `var` = "mid/side" <br>["sim", "dif"] - при `var` = "sim/dif"</span> | Разделение стерео на каналы |
| `join_stereo` | <span style="white-space: nowrap;">`str` \| `var` - вариант: <br>[`left/right` - объединяет левый и правый каналы в стерео] <br>[`mid/side` - объединяет моно и бока в стерео] <br>[`sim/dif` - объединяет фантомный центр и стерео-базу в стерео]</span> | <span style="white-space: nowrap;">["left", "right"] - при `var` = "left/right" <br>["mid", "side"] - при `var` = "mid/side" <br>["sim", "dif"] - при `var` = "sim/dif"</span> | <span style="white-space: nowrap;">["audio"]</span> | Объединение каналов обратно в стерео |
| `subtract` | <span style="white-space: nowrap;">`bool` \| `use_spectrogram` - использовать ли спектрограмму при вычитании</span> | <span style="white-space: nowrap;">["orig", "stem"]</span> | <span style="white-space: nowrap;">["subtracted"]</span> | Вычитание одного сигнала из другого |
| `invert` | нет | <span style="white-space: nowrap;">["audio"]</span> | <span style="white-space: nowrap;">["inverted"]</span> | Инверсия фазы сигнала |
| `separate` | `str` \| `model_name` - имя модели | <span style="white-space: nowrap;">["audio"]</span> | <span style="white-space: nowrap;">["stem1", "stem2", ...]<br>Количество и названия выходов зависят от параметра `model_name`<br>[Узнать стемы по имени модели](https://mvsepless-resources.github.io/model_info_page/)</span> | Разделение аудио на стемы |

