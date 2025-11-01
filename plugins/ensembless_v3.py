import gradio as gr
import json
import pandas as pd
import tempfile
import os
from separator.ensemble import ensemble_audio_files
from multi_inference import MVSEPLESS, clean_filename
from separator.audio_utils import Audio
from pydub.exceptions import CouldntDecodeError
from utils.ensembless import ENSEMBLESS, INVERT_METHODS
from utils.inverter import Inverter

audio = Audio()
ensembless = ENSEMBLESS()
inverter = Inverter()
output_formats = audio.output_formats
input_formats = audio.input_formats

TRANSLATIONS = {
    "ru": {
        "auto_ensemble": "Авто-ансамбль",
        "invert_ensemble": "Инвертировать ансамбль",
        "give_name_preset": "Имя пресета",
        "export": "Экспорт",
        "import": "Импорт",
        "manual_ensemble": "Ручной ансамбль",
        "model_type": "Тип модели",
        "model_name": "Имя модели",
        "weight": "Весы",
        "invert_weights": "Использовать перевернутые весы для инвертированного стема",
        "add_button": "➕ Добавить",
        "remove_index": "Индекс удаляемой модели",
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
        "weights_input": "Весы",
        "analyze_title": "РЕЗУЛЬТАТЫ АНАЛИЗА:",
        "all_same_rate": "✅ ВСЕ ФАЙЛЫ имеют одинаковую частоту дискретизации: {rate} Hz",
        "different_rates": "⚠️ Файлы имеют РАЗНУЮ частоту дискретизации",
        "error_no_files": "Ошибка: файлы не загружены",
        "error_unsupported_format": "не поддерживаемый формат",
        "error_general": "ошибка ({error})",
        "error_no_models": "Добавьте хотя бы одну модель для создания ансамбля",
        "error_no_audio": "Сначала загрузите аудио",
        "p_stem": "Основной стем",
        "s_stem": "Инвертированный стем",
        "result_source": "Промежуточные файлы",
        "output_name": "Имя выходного файла",
        "preset": "Пресет",
        "ensemble_setting": "Настройка текущего ансамбля",
    },
    "en": {
        "auto_ensemble": "Auto-Ensemble",
        "invert_ensemble": "Invert ensemble",
        "give_name_preset": "Preset name",
        "export": "Export",
        "import": "Import",
        "manual_ensemble": "Manual Ensemble",
        "model_type": "Model Type",
        "model_name": "Model Name",
        "weight": "Weights",
        "invert_weights": "Use inverted weights for inverted stem",
        "add_button": "➕ Add",
        "remove_index": "Index of the model being deleted",
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
        "weights_input": "Weights",
        "analyze_title": "ANALYSIS RESULTS:",
        "all_same_rate": "✅ ALL FILES have the same sample rate: {rate} Hz",
        "different_rates": "⚠️ Files have DIFFERENT sample rates",
        "error_no_files": "Error: no files uploaded",
        "error_unsupported_format": "unsupported format",
        "error_general": "error ({error})",
        "error_no_models": "Add at least one model to create an ensemble",
        "error_no_audio": "Please upload audio first",
        "p_stem": "Primary stem",
        "s_stem": "Secondary stem",
        "result_source": "Intermediate files",
        "output_name": "Name output file",
        "preset": "Preset",
        "ensemble_setting": "Setting up the current ensemble",
    },
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
            info = audio.get_info(file_info)
            stream_0 = info.get(0, None)
            if stream_0:
                rate = stream_0["sample_rate"]
            else:
                rate = 0

            # Проверяем единообразие частоты
            if common_rate is None:
                common_rate = rate
            elif common_rate != rate:
                all_same = False

            results.append(f"{file_info.name.split('/')[-1]}: {rate} Hz")

        except Exception as e:
            results.append(
                f"{file_info.name.split('/')[-1]}: {t('error_general', error=str(e))}"
            )

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
        self.presets_dir = os.path.join(tempfile.tempdir, "presets")
        os.makedirs(self.presets_dir, exist_ok=True)

    def export_preset(self, name):
        if not name:
            name = "ensembless_preset"
        filepath = os.path.join(self.presets_dir, f"{clean_filename(name)}.json")
        with open(filepath, "w") as f:
            json.dump(self.models, f)
        return filepath

    def import_preset(self, filepath):
        with open(filepath, "r") as f:
            self.models = json.load(f)
        return self.get_df()

    def add_model(self, model_type, model_name, p_stem, s_stem, weight):
        model_info = {
            "type": model_type,
            "name": model_name,
            "p_stem": p_stem,
            "s_stem": s_stem,
            "weight": float(weight),
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

    def refresh_df(self):
        return self.get_df()

    def get_df(self):
        if not self.models:
            columns = [
                "#",
                t("model_type"),
                t("model_name"),
                t("p_stem"),
                t("s_stem"),
                t("weight"),
            ]
            return pd.DataFrame(columns=columns)

        data = []
        for i, model in enumerate(self.models):
            data.append(
                [
                    f"{i+1}",
                    model["type"],
                    model["name"],
                    model["p_stem"],
                    model["s_stem"],
                    model["weight"],
                ]
            )
        columns = [
            "#",
            t("model_type"),
            t("model_name"),
            t("p_stem"),
            t("s_stem"),
            t("weight"),
        ]
        return pd.DataFrame(data, columns=columns)

    def get_settings(self):
        return [
            (f"{m['type']} / {m['name']}", m["weight"], m["p_stem"], m["s_stem"])
            for m in self.models
        ]


manager = EnsembleManager()


class EnsembLess_ui_updates:

    def update_model_dropdown(self, model_type):
        models = ensembless.get_models_by_type(model_type)
        return gr.Dropdown(choices=models, value=models[0] if models else None)

    def update_stem_dropdown(self, model_type, model_name):
        stems = ensembless.get_stems_by_model(model_type, model_name)
        return gr.Dropdown(choices=stems, value=stems[0] if stems else None)

    def update_invert_stem_dropdown(self, model_type, model_name, primary_stem):
        stems = ensembless.get_invert_stems_by_model(
            model_type, model_name, primary_stem
        )
        return gr.Dropdown(choices=stems, value=stems[0] if stems else None)

    def add_model(self, model_type, model_name, p_stem, s_stem, weight):
        return manager.add_model(model_type, model_name, p_stem, s_stem, weight)

    def remove_model(self, index):
        if index >= 0:
            return manager.remove_model(
                index - 1
            )  # Пользователь вводит начиная с 1, а индекс с 0
        return manager.get_df()

    def clear_all_models(self):
        return manager.clear_models()

    def refresh_df(self):
        return manager.get_df()

    def run_ensemble(
        self, input_audio, ensemble_type, output_format, invert_weights, invert_ensemble
    ):
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
        return (
            gr.update(value=o),
            gr.update(value=o_wav),
            gr.update(value=i),
            gr.update(value=i_wav),
            gr.update(value=result_source),
        )


ensembless_ui = EnsembLess_ui_updates()

default_model = {
    "mt": ensembless.get_model_types(),
    "mn": ensembless.get_models_by_type(ensembless.get_model_types()[0]),
    "stem": ensembless.get_stems_by_model(
        ensembless.get_model_types()[0],
        ensembless.get_models_by_type(ensembless.get_model_types()[0])[0],
    ),
    "invert_stem": ensembless.get_invert_stems_by_model(
        ensembless.get_model_types()[0],
        ensembless.get_models_by_type(ensembless.get_model_types()[0])[0],
        "vocals",
    ),
    "weight": 1,
}


def ensembless_plugin_name():
    return "EnsembLess"

# Создаем интерфейс
def ensembless_plugin(lang):
    set_language(lang)

    with gr.Tabs():
        with gr.Tab(t("auto_ensemble")):
            gr.Markdown(f"### {t('preset')}")
            with gr.Group():
                with gr.Row(equal_height=True):
                    auto_ensemble_export_preset_name = gr.Textbox(
                        label=t("give_name_preset"),
                        interactive=True,
                        value="ensembless_preset",
                    )
                    with gr.Column():
                        auto_ensemble_export_btn = gr.DownloadButton(t("export"), variant="secondary")
                        auto_ensemble_import_btn = gr.UploadButton(
                            t("import"), file_types=[".json"], file_count="single"
                        )
            gr.Markdown(f"### {t('ensemble_setting')}")
            with gr.Group():

                auto_ensemble_model_type = gr.Dropdown(
                    choices=default_model["mt"],
                    label=t("model_type"),
                    value=default_model["mt"][0],
                    filterable=False,
                )
                auto_ensemble_model_name = gr.Dropdown(
                    choices=default_model["mn"],
                    label=t("model_name"),
                    interactive=True,
                    value=default_model["mn"][0],
                    filterable=False,
                )
                auto_ensemble_stem = gr.Dropdown(
                    choices=default_model["stem"],
                    label=t("p_stem"),
                    interactive=True,
                    filterable=False,
                )
                auto_ensemble_invert_stem = gr.Dropdown(
                    choices=default_model["invert_stem"],
                    label=t("s_stem"),
                    interactive=True,
                    filterable=False,
                )

                auto_ensemble_weight = gr.Slider(
                    label=t("weight"),
                    value=default_model["weight"],
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                )
                auto_ensemble_add_btn = gr.Button(t("add_button"), variant="primary")

                auto_ensemble_df = gr.Dataframe(
                    value=manager.get_df(),
                    headers=[
                        "#",
                        t("model_type"),
                        t("model_name"),
                        t("p_stem"),
                        t("s_stem"),
                        t("weight"),
                    ],
                    datatype=["str", "str", "str", "str", "str", "number"],
                    interactive=False,
                )
                with gr.Row(equal_height=True):
                    auto_ensemble_remove_idx = gr.Number(
                        label=t("remove_index"),
                        precision=1,
                        value=1,
                        minimum=1,
                        interactive=True,
                    )
                    with gr.Column():
                        auto_ensemble_remove_btn = gr.Button(t("remove_button"), variant="stop")
                        auto_ensemble_clear_btn = gr.Button(t("clear_button"), variant="stop")

            # Секция запуска обработки
            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"### {t('input_audio')}")
                    auto_ensemble_input_audio = gr.File(
                        type="filepath",
                        interactive=True,
                        show_label=False,
                        file_count="single",
                        file_types=[f".{of}" for of in input_formats]
                    )

                    gr.Markdown(f"### {t('settings')}")
                    with gr.Group():
                        auto_ensemble_ensemble_type = gr.Dropdown(
                            choices=[
                                "avg_wave",
                                "median_wave",
                                "min_wave",
                                "max_wave",
                                "avg_fft",
                                "median_fft",
                                "min_fft",
                                "max_fft",
                            ],
                            value="avg_fft",
                            label=t("method"),
                            filterable=False,
                        )
                        auto_ensemble_invert_ensem = gr.Checkbox(label=t("invert_ensemble"))
                        auto_ensemble_invert_weights = gr.Checkbox(label=t("invert_weights"))
                        auto_ensemble_output_format = gr.Dropdown(
                            choices=output_formats,
                            value="mp3",
                            label=t("output_format"),
                            filterable=False,
                        )
                        auto_ensemble_run_btn = gr.Button(t("run_button"), variant="primary")

                with gr.Column():
                    with gr.Tab(t("results")):

                        with gr.Column():
                            auto_ensemble_output_audio = gr.Audio(
                                label=t("results"),
                                type="filepath",
                                interactive=False,
                                show_download_button=True,
                            )
                            auto_ensemble_output_wav = gr.Text(
                                label="Результат в WAV",
                                interactive=False,
                                visible=False,
                            )

                            gr.Markdown(f"###### {t('inverted_result')}")
                            with gr.Group():
                                auto_ensemble_invert_method = gr.Radio(
                                    choices=["waveform", "spectrogram"],
                                    label=t("invert_method"),
                                    value="waveform",
                                )
                                auto_ensemble_invert_btn = gr.Button(t("invert_button"))
                            auto_ensemble_inverted_output_audio = gr.Audio(
                                label=t("inverted_result"),
                                type="filepath",
                                interactive=False,
                                show_download_button=True,
                            )
                            auto_ensemble_inverted_wav = gr.Text(
                                label="Инвертированный результат в WAV",
                                interactive=False,
                                visible=False,
                            )

                    with gr.Tab(t("result_source")):
                        auto_ensemble_result_source = gr.Files(
                            interactive=False, label=t("result_source")
                        )

                gr.on(
                    fn=ensembless_ui.refresh_df,
                    inputs=None,
                    outputs=auto_ensemble_df
                )

            auto_ensemble_stem.change(
                ensembless_ui.update_invert_stem_dropdown,
                inputs=[auto_ensemble_model_type, auto_ensemble_model_name, auto_ensemble_stem],
                outputs=auto_ensemble_invert_stem,
            )

            auto_ensemble_model_type.change(
                ensembless_ui.update_model_dropdown,
                inputs=auto_ensemble_model_type,
                outputs=auto_ensemble_model_name,
            )
            auto_ensemble_model_name.change(
                ensembless_ui.update_stem_dropdown,
                inputs=[auto_ensemble_model_type, auto_ensemble_model_name],
                outputs=auto_ensemble_stem,
            )

            auto_ensemble_df.change(
                manager.export_preset, inputs=auto_ensemble_export_preset_name, outputs=auto_ensemble_export_btn
            )

            auto_ensemble_export_preset_name.change(
                manager.export_preset, inputs=auto_ensemble_export_preset_name, outputs=auto_ensemble_export_btn
            )

            auto_ensemble_import_btn.upload(
                manager.import_preset, inputs=auto_ensemble_import_btn, outputs=auto_ensemble_df
            )

            auto_ensemble_invert_btn.click(
                inverter.process_audio,
                inputs=[auto_ensemble_input_audio, auto_ensemble_output_wav, auto_ensemble_output_format, auto_ensemble_invert_method],
                outputs=[auto_ensemble_inverted_output_audio, auto_ensemble_inverted_wav],
            )

            auto_ensemble_add_btn.click(
                ensembless_ui.add_model,
                inputs=[auto_ensemble_model_type, auto_ensemble_model_name, auto_ensemble_stem, auto_ensemble_invert_stem, auto_ensemble_weight],
                outputs=auto_ensemble_df,
            )

            auto_ensemble_remove_btn.click(
                ensembless_ui.remove_model, inputs=auto_ensemble_remove_idx, outputs=auto_ensemble_df
            )

            auto_ensemble_clear_btn.click(ensembless_ui.clear_all_models, outputs=auto_ensemble_df)

            auto_ensemble_run_btn.click(
                ensembless_ui.run_ensemble,
                inputs=[
                    auto_ensemble_input_audio,
                    auto_ensemble_ensemble_type,
                    auto_ensemble_output_format,
                    auto_ensemble_invert_weights,
                    auto_ensemble_invert_ensem,
                ],
                outputs=[
                    auto_ensemble_output_audio,
                    auto_ensemble_output_wav,
                    auto_ensemble_inverted_output_audio,
                    auto_ensemble_inverted_wav,
                    auto_ensemble_result_source,
                ],
            )

        with gr.Tab(t("manual_ensemble")):
            with gr.Row(equal_height=True):
                manual_ensemble_input_files = gr.Files(
                    show_label=False,
                    type="filepath",
                    file_types=[f".{of}" for of in input_formats],
                )
                with gr.Column():
                    manual_ensemble_info_audios = gr.Textbox(label="", interactive=False)
                    manual_ensemble_man_method = gr.Dropdown(
                        choices=[
                            "avg_wave",
                            "median_wave",
                            "min_wave",
                            "max_wave",
                            "avg_fft",
                            "median_fft",
                            "min_fft",
                            "max_fft",
                        ],
                        value="avg_fft",
                        label=t("method"),
                        filterable=False,
                    )

                    manual_ensemble_weights_input = gr.Textbox(
                        label=t("weights_input"), value="1.0,1.0"
                    )

                    manual_ensemble_output_file_name = gr.Textbox(
                        label=t("output_name"), value="ensemble"
                    )

                    manual_ensemble_output_man_format = gr.Dropdown(
                        choices=output_formats,
                        value="mp3",
                        label=t("output_format"),
                        filterable=False,
                    )

            manual_ensemble_run_btn = gr.Button(t("run_button"), variant="primary")

            manual_ensemble_output_man_audio = gr.Audio(
                label=t("results"),
                type="filepath",
                interactive=False,
                show_download_button=True,
            )
            manual_ensemble_output_man_wav = gr.Text(
                label="Результат в WAV", interactive=False, visible=False
            )

            manual_ensemble_input_files.upload(
                fn=analyze_sample_rate, inputs=manual_ensemble_input_files, outputs=manual_ensemble_info_audios
            )

            manual_ensemble_run_btn.click(
                lambda a, b, c, d, e: ensembless.manual_ensemble(
                    a, b, c, d, clean_filename(e)
                ),
                inputs=[
                    manual_ensemble_input_files,
                    manual_ensemble_man_method,
                    manual_ensemble_weights_input,
                    manual_ensemble_output_man_format,
                    manual_ensemble_output_file_name,
                ],
                outputs=[manual_ensemble_output_man_audio, manual_ensemble_output_man_wav],
            )
