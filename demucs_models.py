# Убираем импорт из main.py
# from main import model_code  # Удалить эту строку

# Создаем словарь с привязкой model_code к model
models = [
    "drumsep", "htdemucs_ft.yaml", "htdemucs.yaml", 
    "hdemucs_mmi.yaml", "htdemucs_6s.yaml",
    "drumsep", "htdemucs_ft.yaml", "htdemucs.yaml",
    "hdemucs_mmi.yaml", "htdemucs_6s.yaml"
]

model_mapping = {code: model for code, model in zip(range(700, 700 + len(models)), models)}

# Функция для получения модели по коду
def get_demucs_code(model_code):
    """
    Возвращает модель по её коду.
    Если код не найден, возвращает сообщение об ошибке.
    """
    if model_code in model_mapping:
        return model_mapping[model_code]
    else:
        return "Ошибка: модель с таким кодом не найдена."