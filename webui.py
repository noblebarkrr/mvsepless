#@title WEB-UI

import gradio as gr
import subprocess

# Функция для выполнения скрипта при нажатии кнопки "CONVERT"
def convert_audio(audio, split_type, model_type):
    if audio is None:
        return "Пожалуйста, загрузите аудиофайл.", None, None, None, None, None

    # Команды для разных значений model_type
    if model_type == "KARAOKE (MEL ROFORMER)":
        command = f"./melkaraoke.sh"
    elif model_type == "VOCALS (MEL ROFORMER)":
        command = f"./melvocals.sh"
    elif model_type == "VOCALS (BS ROFORMER)":
        command = f"./bsvocals.sh --split_type {split_type}"
    elif model_type == "DENOISE (MEL ROFORMER)":
        command = f"./meldenoise.sh"
    elif model_type == "UVR KARAOKE":
        command = f"./uvr_karaoke.sh"
    elif model_type == "UVR DEECHO":
        command = f"./uvr_deecho.sh"
    elif model_type == "UVR VOCALS":
        command = f"./uvr_vocals.sh"
    elif model_type == "UVR DENOISE":
        command = f"./uvr_denoise.sh"
    elif model_type == "MDX23C VOCALS":
        command = f"./mdx23c_vocals.sh"
    else:
        return f"Неизвестный тип модели: {model_type}", None, None, None, None, None

    # Выполнение команды
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            # Заглушки для путей к результатам конверсии
            if model_type == "KARAOKE (MEL ROFORMER)":
                return ("Успешно выполнено!", "output/instrumental.wav", "output/fullvocals.wav", "output/backvocals.wav", "output/lead_vocals.wav", None)
            else:
                return ("Успешно выполнено!", "output/instrumental.wav", "output/vocals.wav", None, None, None)
        else:
            return (f"Ошибка: {result.stderr}", None, None, None, None, None)
    except Exception as e:
        return (f"Ошибка: {str(e)}", None, None, None, None, None)

# Функция для обновления списка моделей в зависимости от типа разделения
def update_model_options(split_type):
    if split_type == "ROFORMER":
        return gr.update(choices=["VOCALS (MEL ROFORMER)", "VOCALS (BS ROFORMER)", "KARAOKE (MEL ROFORMER)", "DENOISE (MEL ROFORMER)"])
    elif split_type == "VR ARCH":
        return gr.update(choices=["UVR KARAOKE", "UVR DEECHO", "UVR VOCALS", "UVR DENOISE"])
    elif split_type == "MDX23C":
        return gr.update(choices=["MDX23C VOCALS"])
    else:
        return gr.update(choices=[])

# Функция для отображения аудиофайлов в зависимости от выбранной модели
def show_audio_outputs(model_type):
    if model_type == "KARAOKE (MEL ROFORMER)":
        return [gr.Audio(label="Instrumental", type="filepath"),
                gr.Audio(label="Full Vocals", type="filepath"),
                gr.Audio(label="Back Vocals", type="filepath"),
                gr.Audio(label="Lead Vocals", type="filepath")]
    else:
        return [gr.Audio(label="Instrumental", type="filepath"),
                gr.Audio(label="Vocals", type="filepath")]

# UI-компоненты
with gr.Blocks() as demo:
    with gr.Row():
        command = f"bash clear.sh"
        audio_input = gr.Audio(type="filepath", label="Загрузите аудио")
        command = f"cp /tmp/gradio/*/*.wav /content/input"

    with gr.Row():
        split_type = gr.Dropdown(choices=["ROFORMER", "VR ARCH", "MDX23C"], label="Тип разделения", value="ROFORMER")
        model_type = gr.Dropdown(choices=["VOCALS (MEL ROFORMER)", "VOCALS (BS ROFORMER)", "KARAOKE (MEL ROFORMER)", "DENOISE (MEL ROFORMER)"], label="Тип модели")

    # Обновление списка моделей при изменении типа разделения
    split_type.change(fn=update_model_options, inputs=split_type, outputs=model_type)

    with gr.Row():
        convert_button = gr.Button("CONVERT")
        status_output = gr.Textbox(label="Статус", interactive=False)

    # Поля для отображения аудиорезультатов
    instrumental_output = gr.Audio(label="Instrumental", type="filepath", visible=True)
    full_vocals_output = gr.Audio(label="Full Vocals", type="filepath", visible=True)
    back_vocals_output = gr.Audio(label="Back Vocals", type="filepath", visible=True)
    lead_vocals_output = gr.Audio(label="Lead Vocals", type="filepath", visible=True)
    vocals_output = gr.Audio(label="Vocals", type="filepath", visible=False)

    # Обработка события нажатия кнопки "CONVERT"
    def update_audio_outputs(status, instrumental, full_vocals, back_vocals, lead_vocals, vocals):
        if status.startswith("Ошибка"):
            return status, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        else:
            if full_vocals and back_vocals and lead_vocals:
                return status, gr.update(value=instrumental, visible=True), gr.update(value=full_vocals, visible=True), gr.update(value=back_vocals, visible=True), gr.update(value=lead_vocals, visible=True), gr.update(visible=False)
            else:
                return status, gr.update(value=instrumental, visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=vocals, visible=True)

    convert_button.click(
        fn=convert_audio,
        inputs=[audio_input, split_type, model_type],
        outputs=[status_output, instrumental_output, full_vocals_output, back_vocals_output, lead_vocals_output, vocals_output]
    )

    # Обновление полей для аудиовывода при изменении типа модели
    model_type.change(
        fn=show_audio_outputs,
        inputs=model_type,
        outputs=[instrumental_output, full_vocals_output, back_vocals_output, lead_vocals_output, vocals_output]
    )

# Запуск приложения
demo.launch()

