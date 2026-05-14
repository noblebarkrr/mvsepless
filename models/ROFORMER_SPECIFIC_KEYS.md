## Список атрибутов для запуска специфических архитектур семейства Roformer


mel_band_roformer
---
| Атрибут | Архитектура |
|:---:|:---:|
| `windowed` | [Windowed Sink Attention Mel-Band Roformer by SmuleLabs](https://github.com/smulelabs/windowed-roformer/blob/main/model.py) |
| `conformer` | Mel-Band Conformer |

bs_roformer
---
| Атрибут | Архитектура |
|:---:|:---:|
| `sw` | BS-Roformer SW |
| `fno` | BS-Roformer FNO by Unwa |
| `hyperace` | BS-Roformer HyperACE by Unwa |
| `hyperace2` | BS-Roformer HyperACE v2 by Unwa |
| `conformer` | BS-Conformer |
| `conditional` | BS-Roformer Conditional (from [YingMusic-SVC](https://github.com/GiantAILab/YingMusic-SVC/blob/main/accom_separation/models/bs_roformer/bs_roformer.py)) |
| `unwa_inst_large_2` | BS-Roformer Inst Large by Unwa |
| `siamese` | BS-Siamese-Roformer by Unwa |

## Пример использования атрибута в конфиге:

```yaml
hyperace2: true # если атрибут не добавлен, то будет запущена архитектура по умолчанию, важно его наличие, а не его значение

model: 
    ...
training:
    ...
inference:
    ...
audio:
    ...
```
