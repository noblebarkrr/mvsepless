# Убираем импорт из main.py
# from main import model_code  # Удалить эту строку

# Создаем словарь с привязкой model_code к model
models = [
    "1_HP-UVR.pth", "2_HP-UVR.pth",
    "3_HP-Vocal-UVR.pth", "4_HP-Vocal-UVR.pth", 
    "5_HP-Karaoke-UVR.pth", "6_HP-Karaoke-UVR.pth",
    "7_HP2-UVR.pth", "8_HP2-UVR.pth", 
    "9_HP2-UVR.pth", "10_SP-UVR-2B-32000-1.pth",
    "11_SP-UVR-2B-32000-2.pth", 
    "12_SP-UVR-3B-44100.pth",
    "13_SP-UVR-4B-44100-1.pth", 
    "14_SP-UVR-4B-44100-2.pth", 
    "15_SP-UVR-MID-44100-1.pth",
    "16_SP-UVR-MID-44100-2.pth",
    "17_HP-Wind_Inst-UVR.pth", 
    "UVR-De-Echo-Aggressive.pth",
    "UVR-De-Echo-Normal.pth",
    "UVR-DeEcho-DeReverb.pth", 
    "UVR-De-Reverb-aufr33-jarredou.pth",
    "UVR-DeNoise-Lite.pth", "UVR-DeNoise.pth", 
    "UVR-BVE-4B_SN-44100-1.pth",
    "MGM_HIGHEND_v4.pth", "MGM_LOWEND_A_v4.pth", 
    "MGM_LOWEND_B_v4.pth", "MGM_MAIN_v4.pth"
]

model_mapping = {code: model for code, model in zip(range(500, 500 + len(models)), models)}

# Функция для получения модели по коду
def get_model_by_code(model_code):
    """
    Возвращает модель по её коду.
    Если код не найден, возвращает сообщение об ошибке.
    """
    if model_code in model_mapping:
        return model_mapping[model_code]
    else:
        return "Ошибка: модель с таким кодом не найдена."