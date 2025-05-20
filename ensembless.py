import os
import tempfile
import gradio as gr

from model_list import models_data
from infer_utils.uvr_rename_stems import rename_stems
from infer_utils.audio_processing.invert import invert_and_overlay_wav
from multi_infer import audio_separation
from multi_infer import theme as mvsepless_theme
from ensem import ensemble_audio_files

from pydub import AudioSegment

# Экономный режим (для ансамбля больше трёх моделей)

def econom_ensemble(files, output, ensemble_type, weights):
    """
    Последовательно применяет ensemble_audio_files к каждому файлу с учетом весов
    """
    if len(files) != len(weights):
        raise ValueError(f"Ошибка! Файлов - {len(files)}, Весов - {len(weights)}")
    
    temp_dir = tempfile.mkdtemp()
    current_output = None
    
    try:
        for i, (file, weight) in enumerate(zip(files, weights)):
            # Создаем временный выходной файл для этого шага
            temp_output = os.path.join(temp_dir, f"step_{i}.wav")
            
            # Создаем список из weight копий текущего файла
            weighted_files = [file] * int(weight)
            
            # Первый файл - просто копируем (или обрабатываем если weight > 1)
            if i == 0:
                if len(weighted_files) == 1:
                    import shutil
                    shutil.copyfile(file, temp_output)
                else:
                    ensemble_audio_files(
                        files=weighted_files,
                        output=temp_output,
                        ensemble_type=ensemble_type,
                        weights=None
                    )
            else:
                # Для последующих файлов объединяем с предыдущим результатом
                prev_file = os.path.join(temp_dir, f"step_{i-1}.wav")
                ensemble_files = weighted_files + [prev_file]
                
                ensemble_audio_files(
                    files=ensemble_files,
                    output=temp_output,
                    ensemble_type=ensemble_type,
                    weights=None
                )
                
                # Удаляем предыдущий временный файл
                if os.path.exists(prev_file):
                    os.remove(prev_file)
        
        # Перемещаем финальный результат в нужное место
        if temp_output and os.path.exists(temp_output):
            os.rename(temp_output, output)
            return output
        return None
    
    finally:
        # Очищаем временную директорию
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)

# Ручной ансамбль

def manual_ensem(input_ensem_files, use_econom, method,  weights):
    if input_ensem_files != [] or input_ensem_files is not None:
        weights = [float(x) for x in weights.split(",")]
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, f"{method}_ensemble.wav")
        
        if use_econom:
            econom_ensemble(
                files=input_ensem_files,
                output=output_path,
                ensemble_type=method,
                weights=weights
            )
        else:
            ensemble_audio_files(
                files=input_ensem_files,
                output=output_path,
                ensemble_type=method,
                weights=weights
            )
            
        return output_path


# Подгонка частоты дискретизации оригинала под результат ансамбля для успешной инвертации результата:


def resample_for_invert(orig, result):
    filename = os.path.basename(orig)
    name_without_ext = os.path.splitext(filename)[0]
    folder_path = os.path.dirname(result)
    resampled_orig = os.path.join(folder_path, f"{name_without_ext}_resampled.wav")
    audio1 = AudioSegment.from_file(orig)
    audio2 = AudioSegment.from_file(result)
    target_frame_rate = audio2.frame_rate
    audio1_resampled = audio1.set_frame_rate(target_frame_rate)
    audio1_resampled.export(resampled_orig, format="wav")
    return resampled_orig


# Методы ансамбля


ensemble_methods = ["min_wave", "median_wave", "avg_wave", "max_wave", "min_fft", "median_fft", "avg_fft", "max_fft"]



# Простое инвертирование результата



def invert_result(result, orig):
    filename = os.path.basename(orig)
    name_without_ext = os.path.splitext(filename)[0]
    folder_path = os.path.dirname(result)
    inverted_file = os.path.join(folder_path, f"{name_without_ext}_inverted.wav")
    resampled_orig = resample_for_invert(orig, result)
    invert_and_overlay_wav(result, resampled_orig, inverted_file)
    return inverted_file


# Остальное



def get_model_options():
    options = []
    for model_type, model_dict in models_data.items():
        if model_type == "medley_vox":
            continue
        for model_name in model_dict:
            options.append(f"{model_type} / {model_name}")
    return options

def update_stems(model_label):
    try:
        model_type, model_name = model_label.split(" / ")
        return gr.update(choices=models_data[model_type][model_name]["stems"], value=[])
    except:
        return gr.update(choices=[], value=[])




# Процесс авто-ансамбля




def process_old(audio_path, method, use_econom, *inputs):
    temp_dir = tempfile.mkdtemp()
    try:
        file_name = os.path.splitext(os.path.basename(audio_path))[0]
        input_file_path = audio_path

        print("Информация об ансамбле")
        print(f"Папка с файлами ансамбля: {temp_dir}")
        print(f"Путь к входному файлу: {input_file_path}")
        print(f"Метод: {method}")
        print(f"Экономия при объединении: {use_econom}")




        input_ensem_files = []
        block_count = (len(inputs) - 1) // 2
        weights = [float(w.strip()) for w in inputs[-1].split(",")]

        for i in range(block_count):
            model_label = inputs[i]
            stems = inputs[block_count + i]

            if not model_label or not stems:
                continue

            model_type, model_name = model_label.split(" / ")
            output_dir = os.path.join(temp_dir, file_name, f"{model_type}_{model_name}")
            os.makedirs(output_dir, exist_ok=True)

            # progress_ensem += 1/block_count
            # gr.Progress(progress=progress_ensem, desc="Разделение аудио...", total=block_count)

            audio_separation(
                input_dir=input_file_path,
                output_dir=output_dir,
                instrum=True,
                model_name=model_name,
                model_type=model_type,
                output_format="wav",
                use_tta=False,
                batch=False,
                template="MODEL_STEM",
                selected_instruments=[]
            )

            for stem in stems:
                if model_type in ["vr_arch", "mdx_net"]:
                    stem_filename = f"{model_name}_{stem}.wav"
                else:  
                    stem_filename = f"{model_type}_{model_name}_{stem}.wav"

                if not stem_filename:
                    continue

                stem_path = os.path.join(output_dir, stem_filename)
                if os.path.exists(stem_path):
                    input_ensem_files.append(stem_path)
                    print(f"Added {stem_filename} to ensemble")

        if not input_ensem_files:
            return None

        # Подгонка результатов по длине для успешного ансамбля (актуально для audio-separator)

        import soundfile as sf
        import numpy as np

        audio_data = []
        max_length = 0
        for file in input_ensem_files:
            data, sr = sf.read(file)
            if data.ndim == 1:
                data = np.stack([data, data])
            elif data.shape[0] != 2:
                data = data.T
            audio_data.append(data)
            if data.shape[1] > max_length:
                max_length = data.shape[1]

        padded_files = []
        for i, (data, file) in enumerate(zip(audio_data, input_ensem_files)):
            if data.shape[1] < max_length:
                pad_width = ((0, 0), (0, max_length - data.shape[1]))
                padded_data = np.pad(data, pad_width, mode='constant')
            else:
                padded_data = data

            output_padded_path = os.path.join(temp_dir, f"padded_{i}.wav")
            sf.write(output_padded_path, padded_data.T, sr)
            padded_files.append(output_padded_path)

        input_ensem_files = padded_files
        # end padding

        output_path = os.path.join(temp_dir, f"{file_name}_{method}_ensemble.wav")
        
        if use_econom:
            econom_ensemble(
                files=input_ensem_files,
                output=output_path,
                ensemble_type=method,
                weights=weights
            )
        else:
            ensemble_audio_files(
                files=input_ensem_files,
                output=output_path,
                ensemble_type=method,
                weights=weights
            )
            
        return output_path
    except Exception as e:
        print(f"Ошибка: {e}")
        return None











def process(audio_path, method, use_econom, *inputs):
    temp_dir = tempfile.mkdtemp()
    try:
        file_name = os.path.splitext(os.path.basename(audio_path))[0]
        input_file_path = audio_path

        print("Информация об ансамбле")
        print(f"Папка с файлами ансамбля: {temp_dir}")
        print(f"Путь к входному файлу: {input_file_path}")
        print(f"Метод: {method}")
        print(f"Экономия при объединении: {use_econom}")

        input_ensem_files = []
        block_count = (len(inputs) - 1) // 2
        weights = [float(w.strip()) for w in inputs[-1].split(",")]

        # Создаем прогресс бар
        progress = gr.Progress()
        progress(0, desc="Начало обработки...")

        for i in range(block_count):
            model_label = inputs[i]
            stems = inputs[block_count + i]

            if not model_label or not stems:
                continue

            # Обновляем прогресс бар
            progress((i) / block_count, desc=f"Обработка модели {i+1} из {block_count}")

            model_type, model_name = model_label.split(" / ")
            output_dir = os.path.join(temp_dir, file_name, f"{model_type}_{model_name}")
            os.makedirs(output_dir, exist_ok=True)

            audio_separation(
                input_dir=input_file_path,
                output_dir=output_dir,
                instrum=True,
                model_name=model_name,
                model_type=model_type,
                output_format="wav",
                use_tta=False,
                batch=False,
                template="MODEL_STEM",
                selected_instruments=[]
            )

            for stem in stems:
                if model_type in ["vr_arch", "mdx_net"]:
                    stem_filename = f"{model_name}_{stem}.wav"
                else:  
                    stem_filename = f"{model_type}_{model_name}_{stem}.wav"

                if not stem_filename:
                    continue

                stem_path = os.path.join(output_dir, stem_filename)
                if os.path.exists(stem_path):
                    input_ensem_files.append(stem_path)
                    print(f"Added {stem_filename} to ensemble")

        if not input_ensem_files:
            return None

        # Обновляем прогресс бар для этапа выравнивания длин аудио
        progress(0.9, desc="Выравнивание длин аудио...")

        # Подгонка результатов по длине
        import soundfile as sf
        import numpy as np

        audio_data = []
        max_length = 0
        for file in input_ensem_files:
            data, sr = sf.read(file)
            if data.ndim == 1:
                data = np.stack([data, data])
            elif data.shape[0] != 2:
                data = data.T
            audio_data.append(data)
            if data.shape[1] > max_length:
                max_length = data.shape[1]

        padded_files = []
        for i, (data, file) in enumerate(zip(audio_data, input_ensem_files)):
            if data.shape[1] < max_length:
                pad_width = ((0, 0), (0, max_length - data.shape[1]))
                padded_data = np.pad(data, pad_width, mode='constant')
            else:
                padded_data = data

            output_padded_path = os.path.join(temp_dir, f"padded_{i}.wav")
            sf.write(output_padded_path, padded_data.T, sr)
            padded_files.append(output_padded_path)

        input_ensem_files = padded_files

        # Обновляем прогресс бар для этапа создания ансамбля
        progress(0.95, desc="Создание ансамбля...")

        output_path = os.path.join(temp_dir, f"{file_name}_{method}_ensemble.wav")
        
        if use_econom:
            econom_ensemble(
                files=input_ensem_files,
                output=output_path,
                ensemble_type=method,
                weights=weights
            )
        else:
            ensemble_audio_files(
                files=input_ensem_files,
                output=output_path,
                ensemble_type=method,
                weights=weights
            )
            
        progress(1.0, desc="Готово!")
        return output_path
    except Exception as e:
        print(f"Ошибка: {e}")
        return None











def create_ensembless_interface():
    with gr.Blocks(theme=mvsepless_theme) as ensembless:

        audio = gr.Audio(label="Загрузить аудио", type="filepath")
        method = gr.Dropdown(ensemble_methods, label="Метод", value="avg_wave")
        use_econom = gr.Checkbox(label="Экономия памяти", info="Экономия влияет только на объединение результатов моделей в ансамбль", value=False)
        weights = gr.Textbox(label="Весы", value="1.0,1.0")
        output_audio = gr.Audio(label="Результат", interactive=False, type="filepath")
        invert_btn = gr.Button("Инвертировать")
        inverted_output = gr.Audio(label="Инвертированный результат", interactive=False, type="filepath")
        invert_btn.click(
            fn=invert_result,
            inputs=[output_audio, audio],
            outputs=[inverted_output]
        )

        model_blocks = []
        model_dropdowns = []
        stem_dropdowns = []

        used_blocks = gr.State(value=[True, True] + [False] * 8)

        for i in range(10):
            with gr.Row(visible=(i < 2)) as row:
                model_dd = gr.Dropdown(get_model_options(), label="Тип модели / Имя модели", interactive=True)
                stem_dd = gr.Dropdown(choices=[], multiselect=True, label="Стемы", interactive=True, filterable=False)
                remove_btn = gr.Button("-", size="sm")

                model_dd.change(fn=update_stems, inputs=model_dd, outputs=stem_dd)

                def make_remove_fn(index):
                    def remove(used):
                        used[index] = False
                        return (
                            gr.update(value=used),
                            gr.update(visible=False), gr.update(value=None),
                            gr.update(visible=False), gr.update(value=None)
                        )
                    return remove

                remove_btn.click(
                    fn=make_remove_fn(i),
                    inputs=[used_blocks],
                    outputs=[used_blocks, row, model_dd, stem_dd]
                )

                model_blocks.append((row, model_dd, stem_dd))
                model_dropdowns.append(model_dd)
                stem_dropdowns.append(stem_dd)

        def show_next_block(used):
            for i, is_used in enumerate(used):
                if not is_used:
                    used[i] = True
                    updates = [gr.update(value=used)]
                    for j, (row, model_dd, stem_dd) in enumerate(model_blocks):
                        updates.extend([
                            gr.update(visible=used[j]),
                            gr.update(visible=used[j]),
                            gr.update(visible=used[j])
                        ])
                    return updates
            return [gr.update()] * (1 + 3 * 10)

        add_button = gr.Button("+")
        add_button.click(
            fn=show_next_block,
            inputs=[used_blocks],
            outputs=[used_blocks] + sum([[row, model_dd, stem_dd] for (row, model_dd, stem_dd) in model_blocks], [])
        )

        run_btn = gr.Button("Создать ансамбль")
        run_btn.click(
            fn=process,
            inputs=[audio, method, use_econom, *model_dropdowns, *stem_dropdowns, weights],
            outputs=[output_audio]
        )

    return ensembless

def manual_ensemble():
    with gr.Blocks(theme=mvsepless_theme) as ensembless_m:

        input_ensemble_files = gr.Files(label="Загрузить аудио", type="filepath", interactive=True)
        method_input = gr.Dropdown(ensemble_methods, label="Метод", value="avg_wave")
        use_econom_manual = gr.Checkbox(label="Экономия памяти", info="Экономия влияет только на объединение результатов моделей в ансамбль", value=False)
        weights_input = gr.Textbox(label="Весы", value="1.0,1.0")
        output_manual_audio = gr.Audio(label="Результат", interactive=False, type="filepath")
        run_manual_btn = gr.Button("Создать ансамбль")
        run_manual_btn.click(
            fn=manual_ensem,
            inputs=[input_ensemble_files, use_econom_manual, method_input, weights_input],
            outputs=[output_manual_audio]
        )

if __name__ == "__main__":
    ensembless = create_ensembless_interface()
    ensembless.launch(share=True)