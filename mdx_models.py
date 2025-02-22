# Убираем импорт из main.py
# from main import model_code  # Удалить эту строку

# Создаем словарь с привязкой model_code к model
models = [
    "UVR-MDX-NET-Inst_full_292.onnx", "UVR-MDX-NET_Inst_187_beta.onnx",
    "UVR-MDX-NET_Inst_82_beta.onnx", "UVR-MDX-NET_Inst_90_beta.onnx",
    "UVR-MDX-NET_Main_340.onnx", "UVR-MDX-NET_Main_390.onnx",
    "UVR-MDX-NET_Main_406.onnx", "UVR-MDX-NET_Main_427.onnx",
    "UVR-MDX-NET_Main_438.onnx", "UVR-MDX-NET-Inst_HQ_1.onnx",
    "UVR-MDX-NET-Inst_HQ_2.onnx", "UVR-MDX-NET-Inst_HQ_3.onnx",
    "UVR-MDX-NET-Inst_HQ_4.onnx", "UVR-MDX-NET-Inst_HQ_5.onnx",
    "UVR_MDXNET_Main.onnx", "UVR-MDX-NET-Inst_Main.onnx",
    "UVR_MDXNET_1_9703.onnx", "UVR_MDXNET_2_9682.onnx",
    "UVR_MDXNET_3_9662.onnx", "UVR-MDX-NET-Inst_1.onnx",
    "UVR-MDX-NET-Inst_2.onnx", "UVR-MDX-NET-Inst_3.onnx",
    "UVR_MDXNET_KARA.onnx", "UVR_MDXNET_KARA_2.onnx",
    "UVR_MDXNET_9482.onnx", "UVR-MDX-NET-Voc_FT.onnx",
    "Kim_Vocal_1.onnx", "Kim_Vocal_2.onnx", "Kim_Inst.onnx",
    "Reverb_HQ_By_FoxJoy.onnx", "UVR-MDX-NET_Crowd_HQ_1.onnx",
    "kuielab_a_vocals.onnx", "kuielab_a_other.onnx",
    "kuielab_a_bass.onnx", "kuielab_a_drums.onnx",
    "kuielab_b_vocals.onnx", "kuielab_b_other.onnx",
    "kuielab_b_bass.onnx", "kuielab_b_drums.onnx"
]

model_mapping = {code: model for code, model in zip(range(400, 400 + len(models)), models)}

# Функция для получения модели по коду
def get_mdx_code(model_code):
    """
    Возвращает модель по её коду.
    Если код не найден, возвращает сообщение об ошибке.
    """
    if model_code in model_mapping:
        return model_mapping[model_code]
    else:
        return "Ошибка: модель с таким кодом не найдена."