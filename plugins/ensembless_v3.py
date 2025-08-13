import gradio as gr
import json
import pandas as pd
import tempfile
import os
from separator.ensemble import ensemble_audio_files
from pydub.utils import mediainfo
from pydub import AudioSegment
from separator.audio_writer import write_audio_file
from multi_inference import MVSEPLESS
from pydub.exceptions import CouldntDecodeError
from utils.ensembless import ENSEMBLESS, INVERT_METHODS
from utils.inverter import Inverter
from utils.audio_editor import AudioEditor


audioeditor = AudioEditor()
ensembless = ENSEMBLESS()
inverter = Inverter()
mvsepless = MVSEPLESS()

TRANSLATIONS = {
    "ru": {
        "app_title": "EnsembLess",
        "auto_ensemble": "Авто-ансамбль",
        "invert_ensemble": "Инвертировать ансамбль",
        "give_name_preset": "Дайте имя пресету",
        "export": "Экспорт",
        "import": "Импорт",
        "manual_ensemble": "Ручной ансамбль",
        "inverter": "Инвертер",
        "model_selection": "Выберите модель для добавления в ансамбль",
        "model_type": "Тип модели",
        "model_name": "Имя модели",
        "stem_selection": "Стем, который будет использован в ансамбле",
        "weight": "Весы",
        "invert_weights": "Использовать перевернутые весы для инвертированного стема",
        "add_button": "➕ Добавить",
        "current_ensemble": "Текущий ансамбль",
        "remove_index": "Индекс модели, который хотите удалить (начинается с 1)",
        "remove_button": "❌ Удалить",
        "clear_button": "Очистить",
        "input_audio": "Входное аудио",
        "settings": "Настройки",
        "method": "Метод",
        "output_format": "Формат вывода",
        "run_button": "Создать ансамбль",
        "results": "Результаты",
        "inverted_result": "Инвертированный результат",
        "invert_method": "Метод инвертирования",
        "invert_button": "Инвертировать",
        "audio_files": "Аудио файлы",
        "weights_input": "Весы",
        "main_audio": "Основное аудио",
        "audio_to_remove": "Аудио для удаления",
        "processing_method": "Метод обработки",
        "analyze_title": "РЕЗУЛЬТАТЫ АНАЛИЗА:",
        "all_same_rate": "✅ ВСЕ ФАЙЛЫ имеют одинаковую частоту дискретизации: {rate} Hz",
        "different_rates": "⚠️ Файлы имеют РАЗНУЮ частоту дискретизации",
        "resample_warning": "К загруженному аудио автоматически применён ресэмплинг для лучшего инвертирования",
        "error_no_files": "Ошибка: файлы не загружены",
        "error_unsupported_format": "не поддерживаемый формат",
        "error_general": "ошибка ({error})",
        "error_no_models": "Добавьте хотя бы одну модель для создания ансамбля",
        "error_no_audio": "Сначала загрузите аудио",
        "error_both_audio": "Пожалуйста, загрузите оба аудиофайла",
        "language": "Язык",
        "batch_processing": "Пакетная обработка",
        "batch_info": "Позволяет загрузить сразу несколько файлов",
        "separation_info": "Информация о разделении",
        "vocal_separation": "Разделение вокалы",
        "stereo_mode": "Стерео режим",
        "stem": "Стем",
        "p_stem": "Основной стем",
        "s_stem": "Инвертированный стем",
        "vocal_multi_separation": "Мульти-вокал",
        "ensemble": "Ансамбль",
        "transform": "Преобразование",
        "algorithm": "Алгоритм: {model_fullname}",
        "output_format_info": "Формат выходных данных: {output_format}",
        "process1": "Начало обработки",
        "process2": "Модель",
        "process3": "Автоматическое выравнивание длин аудио",
        "process4": "Создание ансамбля",
        "result_source": "Промежуточные файлы",
        "local_path": "Указать путь к аудио локально",
        "resample": "Ресэмпл"
    },
    "en": {
        "app_title": "EnsembLess",
        "auto_ensemble": "Auto-Ensemble",
        "invert_ensemble": "Invert ensemble",
        "give_name_preset": "Give name of preset",
        "export": "Export",
        "import": "Import",
        "manual_ensemble": "Manual Ensemble",
        "inverter": "Inverter",
        "model_selection": "Select a model to add to the ensemble",
        "model_type": "Model Type",
        "model_name": "Model Name",
        "stem_selection": "Stem to use in the ensemble",
        "weight": "Weights",
        "invert_weights": "Use inverted weights for inverted stem",
        "add_button": "➕ Add",
        "current_ensemble": "Current Ensemble",
        "remove_index": "Index of model to remove (starts from 1)",
        "remove_button": "❌ Remove",
        "clear_button": "Clear",
        "input_audio": "Input Audio",
        "settings": "Settings",
        "method": "Method",
        "output_format": "Output Format",
        "run_button": "Create Ensemble",
        "results": "Results",
        "inverted_result": "Inverted Result",
        "invert_method": "Inversion Method",
        "invert_button": "Invert",
        "audio_files": "Audio Files",
        "weights_input": "Weights",
        "main_audio": "Main Audio",
        "audio_to_remove": "Audio to Remove",
        "processing_method": "Processing Method",
        "analyze_title": "ANALYSIS RESULTS:",
        "all_same_rate": "✅ ALL FILES have the same sample rate: {rate} Hz",
        "different_rates": "⚠️ Files have DIFFERENT sample rates",
        "resample_warning": "Resampling applied automatically for better inversion",
        "error_no_files": "Error: no files uploaded",
        "error_unsupported_format": "unsupported format",
        "error_general": "error ({error})",
        "error_no_models": "Add at least one model to create an ensemble",
        "error_no_audio": "Please upload audio first",
        "error_both_audio": "Please upload both audio files",
        "language": "Language",
        "batch_processing": "Batch Processing",
        "batch_info": "Allows uploading multiple files at once",
        "separation_info": "Separation Info",
        "vocal_separation": "Vocal Separation",
        "stereo_mode": "Stereo Mode",
        "stem": "Stem",
        "p_stem": "Primary stem",
        "s_stem": "Secondary stem",
        "vocal_multi_separation": "Multi-Vocal",
        "ensemble": "Ensemble",
        "transform": "Transform",
        "algorithm": "Algorithm: {model_fullname}",
        "output_format_info": "Output format: {output_format}",
        "process1": "Start process",
        "process2": "Model",
        "process3": "Auto post-padding audios",
        "process4": "Build ensemble",
        "result_source": "Intermediate files",
        "local_path": "Specify path to audio locally",
        "resample": "Resample"
    }
}

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



class EnsembleManager:
    def __init__(self):
        self.models = []
        self.presets_dir = os.path.join(os.getcwd(), "presets")
        os.makedirs(self.presets_dir, exist_ok=True)
        
    def export_preset(self, name):
        if not name:
            name = "ensembless_preset"
        filepath = os.path.join(self.presets_dir, f"{name}.json")
        with open(filepath, 'w') as f:
            json.dump(self.models, f)
        return filepath

    def import_preset(self, filepath):
        with open(filepath, 'r') as f:
            self.models = json.load(f)
        return self.get_df()
    
    def add_model(self, model_type, model_name, p_stem, s_stem, weight):
        model_info = {
            'type': model_type,
            'name': model_name,
            'p_stem': p_stem,
            's_stem': s_stem,
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
            columns = ["#", t("model_type"), t("model_name"), t("p_stem"), t("s_stem"), t("weight")]
            return pd.DataFrame(columns=columns)
        
        data = []
        for i, model in enumerate(self.models):
            data.append([
                f"{i+1}",
                model['type'],
                model['name'],
                model['p_stem'],
                model['s_stem'],
                model['weight']
            ])
        columns = ["#", t("model_type"), t("model_name"), t("p_stem"), t("s_stem"), t("weight")]
        return pd.DataFrame(data, columns=columns)
    
    def get_settings(self):
        return [(f"{m['type']} / {m['name']}", m['weight'], m['p_stem'], m['s_stem']) for m in self.models]

manager = EnsembleManager()

class EnsembLess_ui_updates:

    def update_model_dropdown(self, model_type):
        models = ensembless.get_models_by_type(model_type)
        return gr.Dropdown(choices=models, value=models[0] if models else None)
    
    def update_stem_dropdown(self, model_type, model_name):
        stems = ensembless.get_stems_by_model(model_type, model_name)
        return gr.Dropdown(choices=stems, value=stems[0] if stems else None)

    def update_invert_stem_dropdown(self, model_type, model_name, primary_stem):
        stems = ensembless.get_invert_stems_by_model(model_type, model_name, primary_stem)
        return gr.Dropdown(choices=stems, value=stems[0] if stems else None)
    
    def add_model(self, model_type, model_name, p_stem, s_stem, weight):
        return manager.add_model(model_type, model_name, p_stem, s_stem, weight)
    
    def remove_model(self, index):
        if index >= 0:
            return manager.remove_model(index-1)  # Пользователь вводит начиная с 1, а индекс с 0
        return manager.get_df()
    
    def clear_all_models(self):
        return manager.clear_models()
    
    def run_ensemble(self, input_audio, ensemble_type, output_format, invert_weights, invert_ensemble):
        if not manager.models:
            raise gr.Error(t("error_no_models"))
            
        if not input_audio:
            raise gr.Error(t("error_no_audio"))
        
        input_settings = manager.get_settings()
        
        o, o_wav, i, i_wav, result_source = ensembless.auto_ensemble(
            input_audio=input_audio,
            input_settings=input_settings,
            type=ensemble_type,
            out_format=output_format,
            invert_weights=invert_weights,
            invert_ensemble=invert_ensemble,
        )
        return o, o_wav, i, i_wav, result_source

ensembless_ui = EnsembLess_ui_updates()

def ensembless_plugin_name():
    return "EnsembLess"

# Создаем интерфейс
def ensembless_plugin(lang):
    set_language(lang)

    with gr.Tabs():
        with gr.Tab(t("auto_ensemble")):
            with gr.Row():
                with gr.Column(scale=1):
                    # Секция добавления моделей
                    gr.Markdown(f"### {t('model_selection')}")
                    with gr.Group():

                        model_type = gr.Dropdown(
                            choices=ensembless.get_model_types(),
                            label=t("model_type"),
                            value=ensembless.get_model_types()[0] if ensembless.get_model_types() else None,
                            filterable=False
                        )
                        model_name = gr.Dropdown(
                            choices=ensembless.get_models_by_type(ensembless.get_model_types()[0]),
                            label=t("model_name"),
                            interactive=True,
                            value=ensembless.get_models_by_type(ensembless.get_model_types()[0])[0],
                            filterable=False
                        )
                        stem = gr.Dropdown(
                            choices=ensembless.get_stems_by_model(ensembless.get_model_types()[0], ensembless.get_models_by_type(ensembless.get_model_types()[0])[0]),
                            label=t("p_stem"),
                            interactive=True,
                            filterable=False
                        )
                        invert_stem = gr.Dropdown(
                            choices=ensembless.get_invert_stems_by_model(ensembless.get_model_types()[0], ensembless.get_models_by_type(ensembless.get_model_types()[0])[0], "vocals"),
                            label=t("s_stem"),
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
                    
                with gr.Column(scale=2):
                    # Секция управления ансамблем
                    gr.Markdown(f"### {t('current_ensemble')}")
                    ensemble_df = gr.Dataframe(
                        value=manager.get_df(),
                        headers=["#", t("model_type"), t("model_name"), t("p_stem"), t("s_stem"), t("weight")],
                        datatype=["str", "str", "str", "str", "str", "number"],
                        interactive=False
                    )
                    with gr.Group():
                        with gr.Row(equal_height=True):
                            export_preset_name = gr.Textbox(label=t("give_name_preset"), interactive=True, value="ensembless_preset")
                            with gr.Column():
                                export_btn = gr.DownloadButton(t("export"), variant="secondary")
                                import_btn = gr.UploadButton(t("import"), file_types=[".json"], file_count="single")
                        with gr.Row(equal_height=True):
                            remove_idx = gr.Number(
                                label=t("remove_index"),
                                precision=0,
                                minimum=1,
                                interactive=True
                            )
                            with gr.Column():
                                remove_btn = gr.Button(t("remove_button"), variant="stop")
                                clear_btn = gr.Button(t("clear_button"), variant="stop")
            
            # Секция запуска обработки
            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"### {t('input_audio')}")
                    input_audio = gr.File(type="filepath", interactive=True, show_label=False, file_count="single")
                    input_audio_resampled = gr.Text(visible=False)
                    
                    gr.Markdown(f"### {t('settings')}")
                    with gr.Group():
                        ensemble_type = gr.Dropdown(
                            choices=['avg_wave', 'median_wave', 'min_wave', 'max_wave', 
                                    'avg_fft', 'median_fft', 'min_fft', 'max_fft'],
                            value='avg_fft',
                            label=t("method"),
                            filterable=False
                        )
                        invert_ensem = gr.Checkbox(label=t("invert_ensemble"))
                        invert_weights = gr.Checkbox(label=t("invert_weights"))
                        output_format = gr.Dropdown(
                            choices=["wav", "mp3", "flac", "m4a", "aac", "ogg", "opus", "aiff"],
                            value="mp3",
                            label=t("output_format"),
                            filterable=False
                        )
                        run_btn = gr.Button(t("run_button"), variant="primary")

                with gr.Column():
                    with gr.Tab(t('results')):
                    
                        with gr.Column():
                            output_audio = gr.Audio(label=t("results"), type="filepath", interactive=False, show_download_button=True)
                            output_wav = gr.Text(label="Результат в WAV", interactive=False, visible=False)
                
                            gr.Markdown(f"###### {t('inverted_result')}")
                            with gr.Group():
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
    
            stem.change(ensembless_ui.update_invert_stem_dropdown, inputs=[model_type, model_name, stem], outputs=invert_stem)

            model_type.change(
                ensembless_ui.update_model_dropdown,
                inputs=model_type,
                outputs=model_name
            )
            model_name.change(
                ensembless_ui.update_stem_dropdown,
                inputs=[model_type, model_name],
                outputs=stem
            )

            ensemble_df.change(
                manager.export_preset,
                inputs=export_preset_name,
                outputs=export_btn
            )
            
            export_preset_name.change(
                manager.export_preset,
                inputs=export_preset_name,
                outputs=export_btn
            )
            
            import_btn.upload(
                manager.import_preset,
                inputs=import_btn,
                outputs=ensemble_df
            )

            invert_btn.click(
                inverter.process_audio,
                inputs=[input_audio_resampled, output_wav, output_format, invert_method],
                outputs=[inverted_output_audio, inverted_wav]
            )
            
            input_audio.upload(
                audioeditor.resample_audio,
                inputs=input_audio,
                outputs=input_audio_resampled
            )
            
            add_btn.click(
                ensembless_ui.add_model,
                inputs=[model_type, model_name, stem, invert_stem, weight],
                outputs=ensemble_df
            )
            
            remove_btn.click(
                ensembless_ui.remove_model,
                inputs=remove_idx,
                outputs=ensemble_df
            )
            
            clear_btn.click(
                ensembless_ui.clear_all_models,
                outputs=ensemble_df
            )
            
            run_btn.click(
                ensembless_ui.run_ensemble,
                inputs=[input_audio_resampled, ensemble_type, output_format, invert_weights, invert_ensem],
                outputs=[output_audio, output_wav, inverted_output_audio, inverted_wav, result_source]
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
                ensembless.manual_ensemble,
                inputs=[input_files, man_method, weights_input, output_man_format],
                outputs=[output_man_audio, output_man_wav]               
            )                  
