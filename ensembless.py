import gradio as gr
import pandas as pd
import tempfile
import os
from separator.ensemble import ensemble_audio_files
from model_list import models_data
from pydub.utils import mediainfo
from pydub import AudioSegment
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from separator.audio_writer import write_audio_file
from multi_inference import single_multi_inference
from pydub.exceptions import CouldntDecodeError
from assets.translations import ENSEMBLESS_TRANSLATIONS as TRANSLATIONS

# Глобальная переменная для текущего языка
CURRENT_LANG = "ru"

def set_language(lang):
    global CURRENT_LANG
    CURRENT_LANG = lang

def t(key, **kwargs):
    """Функция для получения перевода с подстановкой значений"""
    translation = TRANSLATIONS[CURRENT_LANG].get(key, key)
    return translation.format(**kwargs) if kwargs else translation

def analyze_sample_rate(files):
    """
    Анализирует частоту дискретизации для списка аудиофайлов
    Возвращает форматированную строку с результатами
    """
    if not files:
        return t("error_no_files")
    
    results = []
    common_rate = None
    all_same = True
    
    for file_info in files:
        try:
            # Создаем аудиосегмент из файла
            audio = AudioSegment.from_file(file_info.name)
            rate = audio.frame_rate
            
            # Проверяем единообразие частоты
            if common_rate is None:
                common_rate = rate
            elif common_rate != rate:
                all_same = False
                
            results.append(f"{file_info.name.split('/')[-1]}: {rate} Hz")
            
        except CouldntDecodeError:
            results.append(f"{file_info.name.split('/')[-1]}: {t('error_unsupported_format')}")
        except Exception as e:
            results.append(f"{file_info.name.split('/')[-1]}: {t('error_general', error=str(e))}")
    
    # Форматируем итоговый результат
    header = t("analyze_title") + "\n" + "-" * 50 + "\n"
    body = "\n".join(results)
    footer = "\n" + "-" * 50 + "\n"
    
    if all_same and common_rate is not None:
        footer += f"\n{t('all_same_rate', rate=common_rate)}"
    elif common_rate is not None:
        footer += f"\n{t('different_rates')}"
    
    return header + body + footer


def manual_ensem(input_audios, method, weights, out_format):
    temp_dir = tempfile.mkdtemp()
    weights = [float(x) for x in weights.split(",")]
    padded_files = []

    audio_data = []
    max_length = 0
    for file in input_audios:
        
        data, sr = librosa.load(file, sr=None, mono=False)
        if data.ndim == 1:
            data = np.stack([data, data])
        elif data.shape[0] != 2:
            data = data.T
        audio_data.append([file, data])
        max_length = max(max_length, data.shape[1])
                          
    for i, [file, data] in enumerate(audio_data):
        if data.shape[1] < max_length:
            pad_width = ((0, 0), (0, max_length - data.shape[1]))
            padded_data = np.pad(data, pad_width, mode='constant')
        else:
            padded_data = data
        sf.write(f"{file}.wav", padded_data.T, sr)
        padded_files.append(f"{file}.wav")
    a1, a2 = ensemble_audio_files(padded_files, output=os.path.join(temp_dir, f"ensemble_{method}"), ensemble_type=method, weights=weights, out_format=out_format)
    return a1, a2
    

# Фиксированные параметры для STFT
N_FFT = 2048
WIN_LENGTH = 2048
HOP_LENGTH = WIN_LENGTH // 4


def load_audio(filepath):
    """Загрузка аудиофайла с помощью librosa"""
    if filepath is None:
        return None, None
    try:
        return librosa.load(filepath, sr=None, mono=False)
    except Exception as e:
        print(f"Ошибка загрузки аудио: {e}")
        return None, None

def process_channel(y1_ch, y2_ch, sr, method):
    """Обработка одного аудиоканала"""
    if method == "waveform":
        return y1_ch - y2_ch
    
    elif method == "spectrogram":
        # Вычисляем спектрограммы
        S1 = librosa.stft(y1_ch, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
        S2 = librosa.stft(y2_ch, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
        
        # Амплитудные спектрограммы
        mag1 = np.abs(S1)
        mag2 = np.abs(S2)
        
        # Спектральное вычитание
        mag_result = np.maximum(mag1 - mag2, 0)
        
        # Сохраняем фазовую информацию исходного сигнала
        phase = np.angle(S1)
        
        # Комбинируем амплитуду результата с фазой
        S_result = mag_result * np.exp(1j * phase)
        
        # Обратное преобразование
        return librosa.istft(
            S_result,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            length=len(y1_ch)
        )

def process_audio(audio1_path, audio2_path, out_format, method):
    # Загрузка аудиофайлов
    y1, sr1 = load_audio(audio1_path)
    y2, sr2 = load_audio(audio2_path)
    
    if sr1 is None or sr2 is None:
        raise gr.Error(t("error_both_audio"))
    
    # Определяем количество каналов
    channels1 = 1 if y1.ndim == 1 else y1.shape[0]
    channels2 = 1 if y2.ndim == 1 else y2.shape[0]
    
    # Преобразование в форму (samples, channels)
    if channels1 > 1:
        y1 = y1.T  # (channels, samples) -> (samples, channels)
    else:
        y1 = y1.reshape(-1, 1)
    
    if channels2 > 1:
        y2 = y2.T  # (channels, samples) -> (samples, channels)
    else:
        y2 = y2.reshape(-1, 1)
    
    # Ресемплинг до одинаковой частоты дискретизации
    if sr1 != sr2:
        if channels2 > 1:
            # Ресемплинг для каждого канала отдельно
            y2_resampled = np.zeros((len(y2), channels2), dtype=np.float32)
            for c in range(channels2):
                y2_resampled[:, c] = librosa.resample(
                    y2[:, c], 
                    orig_sr=sr2, 
                    target_sr=sr1
                )
            y2 = y2_resampled
        else:
            y2 = librosa.resample(y2[:, 0], orig_sr=sr2, target_sr=sr1)
            y2 = y2.reshape(-1, 1)
        sr2 = sr1
    
    # Приводим к одинаковой длине
    min_len = min(len(y1), len(y2))
    y1 = y1[:min_len]
    y2 = y2[:min_len]
    
    # Обрабатываем каждый канал отдельно
    result_channels = []
    
    # Если основной сигнал моно, а удаляемый стерео - преобразуем удаляемый в моно
    if channels1 == 1 and channels2 > 1:
        y2 = y2.mean(axis=1, keepdims=True)
        channels2 = 1
    
    for c in range(channels1):
        # Выбираем канал для основного сигнала
        y1_ch = y1[:, c]
        
        # Выбираем канал для удаляемого сигнала
        if channels2 == 1:
            y2_ch = y2[:, 0]
        else:
            # Если каналов удаляемого сигнала больше, используем соответствующий канал
            y2_ch = y2[:, min(c, channels2-1)]
        
        # Обрабатываем канал
        result_ch = process_channel(y1_ch, y2_ch, sr1, method)
        result_channels.append(result_ch)
    
    # Собираем каналы в один массив
    if len(result_channels) > 1:
        result = np.column_stack(result_channels)
    else:
        result = np.array(result_channels[0])
    
    # Нормализация (предотвращение клиппинга)
    if result.ndim > 1:
        # Для многоканального аудио нормализуем каждый канал отдельно
        for c in range(result.shape[1]):
            channel = result[:, c]
            max_val = np.max(np.abs(channel))
            if max_val > 0:
                result[:, c] = channel * 0.9 / max_val
    else:
        max_val = np.max(np.abs(result))
        if max_val > 0:
            result = result * 0.9 / max_val

    folder_path = os.path.dirname(audio2_path)

    # Сохраняем временный файл для вывода
    inverted_wav = os.path.join(folder_path, "inverted.wav")
    sf.write(inverted_wav, result, sr1)
    inverted = os.path.join(folder_path, f"inverted_ensemble.{out_format}")
    write_audio_file(inverted, result.T, sr1, out_format, "320k")
    return inverted, inverted_wav







def ensembless(input_audio, input_settings, type, out_format):

    progress = gr.Progress()
    progress(0, desc=f"{t('process1')}...")

    base_name = os.path.splitext(os.path.basename(input_audio))[0]
    temp_dir = tempfile.mkdtemp()
    source_files = []
    output_s_files = []
    output_s_weights = []
    block_count = len(input_settings)

    for i, (input_model, weight, s_stem) in enumerate(input_settings):
           
        progress(i / block_count, desc=f"{t('process2')} {i+1}/{block_count}")
    
        model_type, model_name = input_model.split(" / ")
        
        output_s_dir = os.path.join(temp_dir, f"{model_type}_{model_name}_s_stems")
        
        output = single_multi_inference(input_audio, output_s_dir, model_type, model_name, True, vr_aggr=10, output_format="wav", output_bitrate="320k", template="MODEL_STEM", call_method="cli", selected_stems=[])
        
        for stem, file in output:       
            source_files.append(file)
            if stem == s_stem:
               output_s_files.append(file)
               output_s_weights.append(weight)

    progress(0.9, desc=f"{t('process3')}...")
               
    padded_files = []

    audio_data = []
    max_length = 0
    for file in output_s_files:
        
        data, sr = sf.read(file)
        if data.ndim == 1:
            data = np.stack([data, data])
        elif data.shape[0] != 2:
            data = data.T
        audio_data.append([file, data])
        max_length = max(max_length, data.shape[1])
                          
    for i, [file, data] in enumerate(audio_data):
        if data.shape[1] < max_length:
            pad_width = ((0, 0), (0, max_length - data.shape[1]))
            padded_data = np.pad(data, pad_width, mode='constant')
        else:
            padded_data = data
        sf.write(file, padded_data.T, sr)
        padded_files.append(file)

    progress(0.95, desc=f"{t('process4')}...")
           
    output, output_wav = ensemble_audio_files(files=output_s_files, output=os.path.join(temp_dir, f"ensemble_{base_name}_{type}"), ensemble_type=type, weights=output_s_weights, out_format=out_format)

    return output, output_wav, source_files






def resample_audio(audio):
    original_name = os.path.splitext(os.path.basename(audio))[0]
    folder_path = os.path.dirname(audio)
    audio = AudioSegment.from_file(audio)
    audio_resampled = audio.set_frame_rate(44100)
    resampled_audio = os.path.join(folder_path, f"resampled_{original_name}.wav")
    audio_resampled.export(resampled_audio, format="wav")
    gr.Warning(message=t("resample_warning"))
    return resampled_audio

# Вспомогательные функции для обработки данных
def get_model_types():
    return list(models_data.keys())

def get_models_by_type(model_type):
    return list(models_data[model_type].keys()) if model_type in models_data else []

def get_stems_by_model(model_type, model_name):
    if model_type in models_data and model_name in models_data[model_type]:
        return models_data[model_type][model_name]['stems']
    return []

# Класс для управления состоянием ансамбля
class EnsembleManager:
    def __init__(self):
        self.models = []
    
    def add_model(self, model_type, model_name, stem, weight):
        model_info = {
            'type': model_type,
            'name': model_name,
            'stem': stem,
            'weight': float(weight)
        }
        self.models.append(model_info)
        return self.get_df()
    
    def remove_model(self, index):
        if 0 <= index < len(self.models):
            del self.models[index]
        return self.get_df()
    
    def clear_models(self):
        self.models = []
        return self.get_df()
    
    def get_df(self):
        if not self.models:
            columns = ["#", t("model_type"), t("model_name"), t("stem"), t("weight")]
            return pd.DataFrame(columns=columns)
        
        data = []
        for i, model in enumerate(self.models):
            data.append([
                f"{i+1}",
                model['type'],
                model['name'],
                model['stem'],
                model['weight']
            ])
        columns = ["#", t("model_type"), t("model_name"), t("stem"), t("weight")]
        return pd.DataFrame(data, columns=columns)
    
    def get_settings(self):
        return [(f"{m['type']} / {m['name']}", m['weight'], m['stem']) for m in self.models]

# Создаем экземпляр менеджера
manager = EnsembleManager()

# Функции обработчики для Gradio
def update_model_dropdown(model_type):
    models = get_models_by_type(model_type)
    return gr.Dropdown(choices=models, value=models[0] if models else None)

def update_stem_dropdown(model_type, model_name):
    stems = get_stems_by_model(model_type, model_name)
    return gr.Dropdown(choices=stems, value=stems[0] if stems else None)

def add_model(model_type, model_name, stem, weight):
    return manager.add_model(model_type, model_name, stem, weight)

def remove_model(index):
    if index >= 0:
        return manager.remove_model(index-1)  # Пользователь вводит начиная с 1, а индекс с 0
    return manager.get_df()

def clear_all_models():
    return manager.clear_models()

def run_ensemble(input_audio, ensemble_type, output_format):
    if not manager.models:
        raise gr.Error(t("error_no_models"))
        
    if not input_audio:
        raise gr.Error(t("error_no_audio"))
    
    input_settings = manager.get_settings()
    
    output, output_wav, result_source = ensembless(
        input_audio=input_audio,
        input_settings=input_settings,
        type=ensemble_type,
        out_format=output_format,
    )
    return output, output_wav, result_source

# Создаем интерфейс
def create_ensembless_app(lang):
    set_language(lang)
    with gr.Blocks(title=t("app_title")) as demo:
        # Добавляем переключатель языка
            
        with gr.Tabs():
            with gr.Tab(t("auto_ensemble")):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Секция добавления моделей
                        gr.Markdown(f"### {t('model_selection')}")
                        model_type = gr.Dropdown(
                            choices=get_model_types(),
                            label=t("model_type"),
                            value=get_model_types()[0] if get_model_types() else None,
                            filterable=False
                        )
                        model_name = gr.Dropdown(
                            choices=get_models_by_type(get_model_types()[0]),
                            label=t("model_name"),
                            interactive=True,
                            value=get_models_by_type(get_model_types()[0])[0],
                            filterable=False
                        )
                        stem = gr.Dropdown(
                            choices=get_stems_by_model(get_model_types()[0], get_models_by_type(get_model_types()[0])[0]),
                            label=t("stem_selection"),
                            interactive=True,
                            filterable=False
                        )
                        weight = gr.Slider(
                            label=t("weight"),
                            value=1.0,
                            minimum=0.1,
                            maximum=10.0,
                            step=0.1
                        )
                        add_btn = gr.Button(t("add_button"), variant="primary")
                        
                        # Обновляем модели и стемы при изменении типа
                        model_type.change(
                            update_model_dropdown,
                            inputs=model_type,
                            outputs=model_name
                        )
                        model_name.change(
                            update_stem_dropdown,
                            inputs=[model_type, model_name],
                            outputs=stem
                        )
            
                    with gr.Column(scale=2):
                        # Секция управления ансамблем
                        gr.Markdown(f"### {t('current_ensemble')}")
                        ensemble_df = gr.Dataframe(
                            value=manager.get_df(),
                            headers=["#", t("model_type"), t("model_name"), t("stem"), t("weight")],
                            datatype=["str", "str", "str", "str", "number"],
                            interactive=False
                        )
                        
                        with gr.Row():
                            remove_idx = gr.Number(
                                label=t("remove_index"),
                                precision=0,
                                minimum=1,
                                interactive=True
                            )
                            remove_btn = gr.Button(t("remove_button"), variant="stop")
                            clear_btn = gr.Button(t("clear_button"), variant="stop")
                
                # Секция запуска обработки
                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.Markdown(f"### {t('input_audio')}")
                        input_audio = gr.Audio(type="filepath", show_label=False)
                        input_audio_resampled = gr.Text(visible=False)
                        
                        gr.Markdown(f"### {t('settings')}")
                        ensemble_type = gr.Dropdown(
                            choices=['avg_wave', 'median_wave', 'min_wave', 'max_wave', 
                                     'avg_fft', 'median_fft', 'min_fft', 'max_fft'],
                            value='avg_fft',
                            label=t("method"),
                            filterable=False
                        )
                        output_format = gr.Dropdown(
                            choices=["wav", "mp3", "flac", "m4a", "aac", "ogg", "opus", "aiff"],
                            value="mp3",
                            label=t("output_format"),
                            filterable=False
                        )
                        run_btn = gr.Button(t("run_button"), variant="primary")

                    with gr.Tab(t('results')):
                    
                        with gr.Column():
                            output_audio = gr.Audio(label=t("results"), type="filepath", interactive=False, show_download_button=True)
                            output_wav = gr.Text(label="Результат в WAV", interactive=False, visible=False)
                
                            gr.Markdown(f"###### {t('inverted_result')}")
                
                            invert_method = gr.Radio(
                                choices=["waveform", "spectrogram"],
                                label=t("invert_method"),
                                value="waveform"
                            )
                            invert_btn = gr.Button(t("invert_button"))
                            inverted_output_audio = gr.Audio(label=t("inverted_result"), type="filepath", interactive=False, show_download_button=True)
                            inverted_wav = gr.Text(label="Инвертированный результат в WAV", interactive=False, visible=False)

                    with gr.Tab(t('result_source')):
                        result_source = gr.Files(interactive=False, label=t('result_source'))

                
                # Обработчики событий
            
                invert_btn.click(
                    process_audio,
                    inputs=[input_audio_resampled, output_wav, output_format, invert_method],
                    outputs=[inverted_output_audio, inverted_wav]
                )
                
                input_audio.upload(
                    resample_audio,
                    inputs=input_audio,
                    outputs=input_audio_resampled
                )
                
                add_btn.click(
                    add_model,
                    inputs=[model_type, model_name, stem, weight],
                    outputs=ensemble_df
                )
                
                remove_btn.click(
                    remove_model,
                    inputs=remove_idx,
                    outputs=ensemble_df
                )
                
                clear_btn.click(
                    clear_all_models,
                    outputs=ensemble_df
                )
                
                run_btn.click(
                    run_ensemble,
                    inputs=[input_audio_resampled, ensemble_type, output_format],
                    outputs=[output_audio, output_wav, result_source]
                )

            with gr.Tab(t("manual_ensemble")):
                with gr.Row(equal_height=True):
                    input_files = gr.Files(show_label=False, type="filepath", file_types=[".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff"])
                    with gr.Column():
                        info_audios = gr.Textbox(label="", interactive=False)
                        man_method = gr.Dropdown(
                            choices=['avg_wave', 'median_wave', 'min_wave', 'max_wave', 
                                     'avg_fft', 'median_fft', 'min_fft', 'max_fft'],
                            value='avg_fft',
                            label=t("method"),
                            filterable=False
                        )
                        
                        weights_input = gr.Textbox(label=t("weights_input"), value="1.0,1.0")
                        
                        output_man_format = gr.Dropdown(
                            choices=["wav", "mp3", "flac", "m4a", "aac", "ogg", "opus", "aiff"],
                            value="mp3",
                            label=t("output_format"),
                            filterable=False
                        )

                run_man_btn = gr.Button(t("run_button"), variant="primary")
                        
                output_man_audio = gr.Audio(label=t("results"), type="filepath", interactive=False, show_download_button=True)
                output_man_wav = gr.Text(label="Результат в WAV", interactive=False, visible=False)
                
                
                input_files.upload(
                    fn=analyze_sample_rate,
                    inputs=input_files,
                    outputs=info_audios
                )
                            
                
                run_man_btn.click(
                    manual_ensem,
                    inputs=[input_files, man_method, weights_input, output_man_format],
                    outputs=[output_man_audio, output_man_wav]               
                )
            with gr.Tab(t("inverter")):
                with gr.Row():
                    audio1 = gr.Audio(label=t("main_audio"), type="filepath")
                    audio2 = gr.Audio(label=t("audio_to_remove"), type="filepath")
                invert_man_method = gr.Radio(
                    choices=["waveform", "spectrogram"],
                    label=t("processing_method"),
                    value="waveform"
                )
                output_man_i_format = gr.Dropdown(
                    choices=["wav", "mp3", "flac", "m4a", "aac", "ogg", "opus", "aiff"],
                    value="mp3",
                    label=t("output_format"),
                    filterable=False
                )
                invert_man_btn = gr.Button(t("invert_button"))
                
                with gr.Column():
                    invert_man_output = gr.Audio(label=t("results"), interactive=False, show_download_button=True)
                    invert_man_output_wav = gr.Text(interactive=False, visible=False)
                    
                invert_man_btn.click(
                    process_audio,
                    inputs=[audio1, audio2, output_man_i_format, invert_man_method],
                    outputs=[invert_man_output, invert_man_output_wav]               
                )                              

        # Запускаем приложение
    return demo

if __name__ == "__main__":
    demo = create_ensembless_app()
    demo.launch(share=True)
