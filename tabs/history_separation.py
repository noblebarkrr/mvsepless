import gradio as gr
import os

# Константы лучше называть в верхнем регистре
OUTPUT_DIR_UVR = "/content/output"
MAX_AUDIO_FILES = 7  # Выносим максимальное количество файлов в константу

def update_audio_players(history_list):
    """Обновляет аудио-плееры на основе выбранной директории."""
    if not history_list:
        return [gr.update(visible=False)] * MAX_AUDIO_FILES
    
    # Формируем путь к папке с аудио
    audio_folder = os.path.join(OUTPUT_DIR_UVR, history_list)
    
    # Проверяем существование директории
    if not os.path.exists(audio_folder):
        return [gr.update(visible=False)] * MAX_AUDIO_FILES
    
    # Получаем список аудиофайлов с поддержкой разных расширений
    audio_extensions = (".wav", ".mp3", ".flac")
    audio_files = [
        os.path.join(audio_folder, f) 
        for f in sorted(os.listdir(audio_folder))  # Сортируем для предсказуемого порядка
        if f.lower().endswith(audio_extensions)
    ][:MAX_AUDIO_FILES]  # Сразу ограничиваем количество
    
    # Создаем список обновлений для компонентов
    updates = []
    for i in range(MAX_AUDIO_FILES):
        if i < len(audio_files):
            updates.append(gr.update(visible=True, value=audio_files[i]))
        else:
            updates.append(gr.update(visible=False))
    
    return updates

def get_history_list(output_dir):
    """Получение списка доступных директорий с результатами."""
    if not os.path.exists(output_dir):
        return []
    
    return [
        d for d in sorted(os.listdir(output_dir))  # Сортируем для удобства
        if os.path.isdir(os.path.join(output_dir, d))
    ]

def history_separations():
    """Создает интерфейс для просмотра истории разделения аудио."""
    with gr.Blocks() as demo:
        with gr.Column() as history_check:
            with gr.Row():
                history_list = gr.Dropdown(
                    choices=get_history_list(OUTPUT_DIR_UVR), 
                    label="Задание", 
                    interactive=True,
                    filterable=False
                )
            btn = gr.Button("Обновить список", variant="primary")
            show_audio = gr.Button("Показать аудио", variant="primary")
        
        with gr.Column() as history_players:
            stems = [
                gr.Audio(type="filepath", interactive=False, visible=False)
                for _ in range(MAX_AUDIO_FILES)
            ]
            
            # Обработчики событий
            show_audio.click(
                update_audio_players,
                inputs=[history_list],
                outputs=stems  # Используем список компонентов
            )
            
            # Добавим обработчик для обновления списка заданий
            btn.click(
                lambda: gr.Dropdown(choices=get_history_list(OUTPUT_DIR_UVR)),
                outputs=history_list
            )
    
    return demo