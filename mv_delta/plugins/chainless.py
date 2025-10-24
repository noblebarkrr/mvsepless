import os
import sys
import ast
import tempfile
import json
import shutil
import pandas as pd
import time
import gradio as gr
from datetime import datetime
from multi_inference import MVSEPLESS, clean_filename
from separator.audio_utils import Audio

output_formats = Audio().output_formats
input_formats = Audio().input_formats
from utils.inverter import Inverter

mvsepless = MVSEPLESS()

TRANSLATIONS = {
    "ru": {
        "preset": "Пресет",
        "give_name_preset": "Имя пресета",
        "export": "Экспорт",
        "import": "Импорт",
        "chain_setting": "Настройка текущей цепи",
        "model_type": "Тип модели",
        "model_name": "Имя модели",
        "selected_stems": "Выбранные стемы",
        "output_stems": "Выходные стемы",
        "intermediate_stem": "Промежуточный стем",
        "add_button": "➕ Добавить",
        "selected_output_stems": "Выбранные выходные стемы",
        "remove_index": "Индекс удаляемой модели",
        "remove_button": "❌ Удалить",
        "clear_button": "Очистить",
        "input_audio": "Входное аудио",
        "settings": "Настройки",
        "save_only_last_intermediate_stem": "Сохранить только последний промежуточный стем",
        "output_format": "Формат вывода",
        "run_button": "Создать цепь разделений",
        "results": "Результаты",
        "last_intermediate_stem": "Последний промежуточный стем",
        "inverted_result": "Инвертированный последний промежуточный стем",
        "invert_method": "Метод инвертирования",
        "invert_button": "Инвертировать",
        "error_no_models": "Добавьте хотя бы одну модель для создания ансамбля",
        "error_no_audio": "Сначала загрузите аудио",
        "output_stems_info": "(Выберите как минимум один стем)",
    },
    "en": {
        "preset": "Preset",
        "give_name_preset": "Preset name",
        "export": "Export",
        "import": "Import",
        "chain_setting": "Setting up the current chain",
        "model_type": "Model Type",
        "model_name": "Model Name",
        "selected_stems": "Selected stems",
        "output_stems": "Output stems",
        "intermediate_stem": "Intermediate stem",
        "add_button": "➕ Add",
        "selected_output_stems": "Selected output stems",
        "remove_index": "Index of the model being deleted",
        "remove_button": "❌ Remove",
        "clear_button": "Clear",
        "input_audio": "Input Audio",
        "settings": "Settings",
        "save_only_last_intermediate_stem": "Save only the last intermediate stem",
        "output_format": "Output Format",
        "run_button": "Create chain",
        "results": "Results",
        "last_intermediate_stem": "Last intermediate stem",
        "inverted_result": "Inverted last intermediate stem",
        "invert_method": "Inversion Method",
        "invert_button": "Invert",
        "error_no_models": "Add at least one model to create an ensemble",
        "error_no_audio": "Please upload audio first",
        "output_stems_info": "(Select at least one stem)",
    },
}

CURRENT_LANG = "ru"


def set_language(lang):
    global CURRENT_LANG
    CURRENT_LANG = lang


def t(key, **kwargs):
    """Функция для получения перевода с подстановкой значений"""
    translation = TRANSLATIONS[CURRENT_LANG].get(key, key)
    return translation.format(**kwargs) if kwargs else translation


inverter = Inverter()


class CHAINNESS:
    def __init__(self):
        self.mvsepless = MVSEPLESS()
        self.output_app_base_dir = os.environ.get(
            "CHAINNESS_OUTPUT_DIR", os.path.join(os.getcwd(), "chain_output")
        )
        os.makedirs(self.output_app_base_dir, exist_ok=True)

    def mt(self):
        ls = self.mvsepless.model_manager.get_mt()
        return gr.update(choices=ls, value=ls[0])

    def mn(self, model_type):
        ls = self.mvsepless.model_manager.get_mn(model_type)
        return gr.update(choices=ls, value=ls[0])

    def mt2(self):
        ls = self.mvsepless.model_manager.get_mt()
        return ls

    def mn2(self, model_type):
        ls = self.mvsepless.model_manager.get_mn(model_type)
        return ls

    def stems(self, model_type, model_name):
        all_stems = []
        stems = self.mvsepless.model_manager.get_stems(model_type, model_name)
        for stem in stems:
            all_stems.append(stem)
        return gr.update(
            choices=all_stems,
            value=[],
            interactive=(
                True
                if not self.mvsepless.model_manager.get_tgt_inst(model_type, model_name)
                and model_type not in ["vr", "mdx"]
                else False
            ),
        )

    def stems2(self, model_type, model_name):
        all_stems = []
        stems = self.mvsepless.model_manager.get_stems(model_type, model_name)
        for stem in stems:
            all_stems.append(stem)
        return all_stems

    def gen_out_stems(self, model_type, model_name, selected_stems):
        output_stems = []
        stems = self.mvsepless.model_manager.get_stems(model_type, model_name)
        if selected_stems == None:
            selected_stems = []
        for stem in stems:
            if self.mvsepless.model_manager.get_tgt_inst(model_type, model_name):
                output_stems.append(stem)
            else:
                if len(selected_stems) != 0:
                    if len(selected_stems) == len(stems):
                        output_stems.append(stem)
                    else:
                        if stem in selected_stems:
                            output_stems.append(stem)
                else:
                    output_stems.append(stem)

        if not self.mvsepless.model_manager.get_tgt_inst(
            model_type, model_name
        ) and model_type not in ["vr", "mdx"]:
            if len(selected_stems) != 0:
                if len(selected_stems) == len(stems) - 1:
                    pass
                elif len(selected_stems) == len(stems):
                    pass
                else:
                    output_stems.append("inverted +")
                output_stems.append("inverted -")

            elif len(selected_stems) == 0:
                if (
                    set(stems) == {"bass", "drums", "vocals", "other"}
                    or set(stems)
                    == {"bass", "drums", "vocals", "other", "piano", "guitar"}
                    and not self.mvsepless.model_manager.get_tgt_inst(
                        model_type, model_name
                    )
                ):
                    output_stems.append("instrumental +")
                    output_stems.append("instrumental -")

        return output_stems

    def out_stems(self, model_type, model_name, selected_stems):
        output_stems = self.gen_out_stems(
            model_type=model_type, model_name=model_name, selected_stems=selected_stems
        )

        int_stems = len(output_stems)
        int_stems_2 = int_stems - 1

        return gr.update(
            choices=output_stems, value=[output_stems[0]], max_choices=int_stems_2
        )

    def intout_stems(
        self, model_type, model_name, selected_stems, selected_output_stems
    ):
        output_stems = self.gen_out_stems(
            model_type=model_type, model_name=model_name, selected_stems=selected_stems
        )
        intermediate_stems = []

        for stem in output_stems:
            if stem not in selected_output_stems:
                intermediate_stems.append(stem)

        return gr.update(
            choices=intermediate_stems,
            value=intermediate_stems[0] if len(output_stems) <= 2 else None,
        )

    def intout_stems2(
        self, model_type, model_name, selected_stems, selected_output_stems
    ):
        output_stems = self.gen_out_stems(
            model_type=model_type, model_name=model_name, selected_stems=selected_stems
        )
        intermediate_stems = []

        for stem in output_stems:
            if stem not in selected_output_stems:
                intermediate_stems.append(stem)

        return intermediate_stems

    def chain(
        self,
        input_audio,
        input_settings,
        output_format,
        save_only_last_intermediate_stem,
    ):

        output_dir = os.path.join(
            self.output_app_base_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        os.makedirs(output_dir, exist_ok=True)

        progress = gr.Progress()
        progress(0, desc=None)

        base_name = os.path.splitext(os.path.basename(input_audio))[0]
        temp_dir = tempfile.mkdtemp()
        last_intermediate_stem = None
        last_intermediate_stem_name = None
        output_stems = []
        output_stems_moved = []

        block_count = len(input_settings)

        for i, (
            input_model,
            selected_stems,
            selected_output_stems,
            intermediate_stem,
        ) in enumerate(input_settings):
            selected_stems = ast.literal_eval(selected_stems)
            selected_output_stems = ast.literal_eval(selected_output_stems)
            progress(i / block_count, desc=f"{i+1}/{block_count}")
            model_type, model_name = input_model.split(" / ")
            output_dir_p = os.path.join(temp_dir, f"{model_type}_{model_name}")

            output_p = self.mvsepless.separator.base(
                input_file=(
                    input_audio
                    if not last_intermediate_stem
                    else last_intermediate_stem
                ),
                output_dir=output_dir_p,
                model_type=model_type,
                model_name=model_name,
                ext_inst=True,
                vr_aggr=10,
                output_format=output_format,
                template=f"{i+1}_({f'{i}{str(last_intermediate_stem_name)}_' if last_intermediate_stem_name else ''})MODEL_STEM",
                call_method="cli",
                selected_stems=selected_stems,
            )
            for stem, file in output_p:
                if stem in selected_output_stems:
                    output_stems.append(file)
                elif stem == intermediate_stem:
                    last_intermediate_stem = file
                    last_intermediate_stem_name = stem
                    output_stems.append(file)

        if not save_only_last_intermediate_stem:
            for path in output_stems:
                name, ext = os.path.splitext(os.path.basename(path))
                new_path = os.path.join(output_dir, f"{clean_filename(base_name, length=60)}_{name}{ext}")
                shutil.copy(src=path, dst=new_path)
                output_stems_moved.append(new_path)

        progress(0.95, desc=None)
        return last_intermediate_stem, output_stems_moved


class ChainManager:
    def __init__(self):
        self.models = []
        self.presets_dir = os.path.join(tempfile.tempdir, "presets")
        os.makedirs(self.presets_dir, exist_ok=True)

    def export_preset(self, name):
        if not name:
            name = "chain_preset"
        filepath = os.path.join(self.presets_dir, f"{clean_filename(name)}.json")
        with open(filepath, "w") as f:
            json.dump(self.models, f)
        return filepath

    def import_preset(self, filepath):
        with open(filepath, "r") as f:
            self.models = json.load(f)
        return self.get_df()

    def add_model(self, model_type, model_name, selected_stems, stem_a, stem_b):
        if len(stem_a) == 0:
            return self.get_df()
        if stem_b == "" or not stem_b:
            return self.get_df()
        model_info = {
            "type": model_type,
            "name": model_name,
            "selected_stems": str(selected_stems),
            "stems_a": str(stem_a),
            "stem_b": stem_b,
        }
        self.models.append(model_info)
        return self.get_df()

    def refresh_df(self):
        return self.get_df()

    def remove_model(self, index):
        index = index - 1
        if 0 <= index < len(self.models):
            del self.models[index]
        return self.get_df()

    def clear_models(self):
        self.models = []
        return self.get_df()

    def get_df(self):
        if not self.models:
            columns = [
                "#",
                t("model_type"),
                t("model_name"),
                t("selected_stems"),
                t("selected_output_stems"),
                t("intermediate_stem"),
            ]
            return pd.DataFrame(columns=columns)

        data = []
        for i, model in enumerate(self.models):
            data.append(
                [
                    f"{i+1}",
                    model["type"],
                    model["name"],
                    model["selected_stems"],
                    model["stems_a"],
                    model["stem_b"],
                ]
            )
        columns = [
            "#",
            t("model_type"),
            t("model_name"),
            t("selected_stems"),
            t("selected_output_stems"),
            t("intermediate_stem"),
        ]
        return pd.DataFrame(data, columns=columns)

    def get_settings(self):
        return [
            (
                f"{m['type']} / {m['name']}",
                m["selected_stems"],
                m["stems_a"],
                m["stem_b"],
            )
            for m in self.models
        ]

    def run_chain(self, input_audio, output_format, save_only_last_intermediate_stem):
        if not self.models:
            raise gr.Error(t("error_no_models"))

        if not input_audio:
            raise gr.Error(t("error_no_audio"))

        input_settings = self.get_settings()

        last_intermediate_stem, all_stems = chain.chain(
            input_audio=input_audio,
            input_settings=input_settings,
            output_format=output_format,
            save_only_last_intermediate_stem=save_only_last_intermediate_stem,
        )
        return gr.update(value=last_intermediate_stem), gr.update(value=all_stems)


chain = CHAINNESS()
chain_manager = ChainManager()

default_model = {
    "mt": chain.mt2(),
    "mn": chain.mn2(chain.mt2()[0]),
    "s_stems": chain.stems2(
        model_type=chain.mt2()[0], model_name=chain.mn2(chain.mt2()[0])[0]
    ),
    "stems_a": chain.gen_out_stems(
        model_type=chain.mt2()[0],
        model_name=chain.mn2(chain.mt2()[0])[0],
        selected_stems=[],
    ),
    "stems_a_value": chain.gen_out_stems(
        model_type=chain.mt2()[0],
        model_name=chain.mn2(chain.mt2()[0])[0],
        selected_stems=[],
    )[0],
    "stem_b": chain.intout_stems2(
        chain.mt2()[0],
        chain.mn2(chain.mt2()[0])[0],
        [],
        [
            chain.stems2(
                model_type=chain.mt2()[0], model_name=chain.mn2(chain.mt2()[0])[0]
            )[0]
        ],
    ),
}


def plugin_name():
    return "ChainLess"


def plugin(lang):
    set_language(lang)
    gr.Markdown(f"### {t('preset')}")
    with gr.Group():
        with gr.Row(equal_height=True):
            chainless_export_preset_name = gr.Textbox(
                label=t("give_name_preset"), interactive=True, value="chain_preset"
            )
            with gr.Column():
                chainless_export_btn = gr.DownloadButton(t("export"), variant="secondary")
                chainless_import_btn = gr.UploadButton(
                    t("import"), file_types=[".json"], file_count="single"
                )

    gr.Markdown(f"### {t('chain_setting')}")
    with gr.Group():
        chainless_c_mt = gr.Dropdown(
            label=t("model_type"),
            multiselect=False,
            interactive=True,
            filterable=False,
            choices=default_model["mt"],
            value=default_model["mt"][0],
        )
        chainless_c_mn = gr.Dropdown(
            label=t("model_name"),
            multiselect=False,
            interactive=True,
            filterable=False,
            choices=default_model["mn"],
            value=default_model["mn"][0],
        )
        chainless_c_s_stems = gr.Dropdown(
            label=t("selected_stems"),
            multiselect=True,
            interactive=False,
            filterable=False,
            choices=default_model["s_stems"],
            value=[],
        )
        chainless_c_a_stems = gr.Dropdown(
            label=t("output_stems"),
            info=t("output_stems_info"),
            multiselect=True,
            interactive=True,
            filterable=False,
            choices=default_model["stems_a"],
            value=default_model["stems_a_value"],
            max_choices=1,
        )
        chainless_c_b_stem = gr.Dropdown(
            label=t("intermediate_stem"),
            info=t("output_stems_info"),
            multiselect=False,
            interactive=True,
            filterable=False,
            choices=default_model["stem_b"],
            value=default_model["stem_b"][0],
        )
        chainless_add_btn = gr.Button(t("add_button"), variant="primary")
        chain_df = gr.Dataframe(
            value=chain_manager.get_df(),
            headers=[
                "#",
                t("model_type"),
                t("model_name"),
                t("selected_stems"),
                t("selected_output_stems"),
                t("intermediate_stem"),
            ],
            datatype=["str", "str", "str", "str", "str", "str"],
            interactive=False,
        )
        with gr.Row(equal_height=True):
            chainless_remove_idx = gr.Number(
                label=t("remove_index"),
                precision=1,
                value=1,
                minimum=1,
                interactive=True,
            )
            with gr.Column():
                chainless_remove_btn = gr.Button(t("remove_button"), variant="stop")
                chainless_clear_btn = gr.Button(t("clear_button"), variant="stop")

    with gr.Row():
        with gr.Column():
            gr.Markdown(f"### {t('input_audio')}")
            chainless_input_file = gr.File(
                type="filepath", interactive=True, show_label=False, file_count="single", file_types=[f".{of}" for of in input_formats]
            )
            gr.Markdown(f"### {t('settings')}")
            with gr.Group():
                chainless_save_only_last_intermediate_stem = gr.Checkbox(
                    label=t("save_only_last_intermediate_stem"), value=False
                )
                chainless_output_format = gr.Dropdown(
                    choices=output_formats,
                    value="mp3",
                    label=t("output_format"),
                    filterable=False,
                )
                chainless_run_btn = gr.Button(t("run_button"), variant="primary")
        with gr.Column():
            with gr.Tab(t("results")):
                with gr.Column():
                    chainless_last_intermediate_stem = gr.Audio(
                        label=t("last_intermediate_stem"),
                        type="filepath",
                        interactive=False,
                        show_download_button=True,
                    )
                    gr.Markdown(f"###### {t('inverted_result')}")
                    with gr.Group():
                        chainless_invert_method = gr.Radio(
                            choices=["waveform", "spectrogram"],
                            label=t("invert_method"),
                            value="waveform",
                        )
                        chainless_invert_btn = gr.Button(t("invert_button"))
                    chainless_inverted_output_audio = gr.Audio(
                        label=t("inverted_result"),
                        type="filepath",
                        interactive=False,
                        show_download_button=True,
                    )
            with gr.Tab(t("output_stems")):
                result_source = gr.Files(interactive=False, label=t("output_stems"))

    chain_df.change(
        chain_manager.export_preset, inputs=chainless_export_preset_name, outputs=chainless_export_btn
    )

    chainless_export_preset_name.change(
        chain_manager.export_preset, inputs=chainless_export_preset_name, outputs=chainless_export_btn
    )

    chainless_import_btn.upload(chain_manager.import_preset, inputs=chainless_import_btn, outputs=chain_df)

    chainless_add_btn.click(
        chain_manager.add_model,
        inputs=[chainless_c_mt, chainless_c_mn, chainless_c_s_stems, chainless_c_a_stems, chainless_c_b_stem],
        outputs=chain_df,
    )

    chainless_remove_btn.click(chain_manager.remove_model, inputs=chainless_remove_idx, outputs=chain_df)

    chainless_clear_btn.click(chain_manager.clear_models, outputs=chain_df)

    chainless_c_mt.change(chain.mn, inputs=chainless_c_mt, outputs=chainless_c_mn)
    chainless_c_mn.change(chain.stems, inputs=[chainless_c_mt, chainless_c_mn], outputs=chainless_c_s_stems).then(
        chain.out_stems, inputs=[chainless_c_mt, chainless_c_mn, chainless_c_s_stems], outputs=chainless_c_a_stems
    ).then(
        chain.intout_stems, inputs=[chainless_c_mt, chainless_c_mn, chainless_c_s_stems, chainless_c_a_stems], outputs=chainless_c_b_stem
    )
    chainless_c_s_stems.change(
        chain.out_stems, inputs=[chainless_c_mt, chainless_c_mn, chainless_c_s_stems], outputs=chainless_c_a_stems
    ).then(
        chain.intout_stems, inputs=[chainless_c_mt, chainless_c_mn, chainless_c_s_stems, chainless_c_a_stems], outputs=chainless_c_b_stem
    )
    chainless_c_a_stems.change(
        chain.intout_stems, inputs=[chainless_c_mt, chainless_c_mn, chainless_c_s_stems, chainless_c_a_stems], outputs=chainless_c_b_stem
    )

    chainless_invert_btn.click(
        inverter.process_audio,
        inputs=[chainless_input_file, chainless_last_intermediate_stem, chainless_output_format, chainless_invert_method],
        outputs=[chainless_inverted_output_audio, gr.State()],
    )

    chainless_run_btn.click(
        chain_manager.run_chain,
        inputs=[chainless_input_file, chainless_output_format, chainless_save_only_last_intermediate_stem],
        outputs=[chainless_last_intermediate_stem, result_source],
    )

    gr.on(
        fn=chain_manager.refresh_df,
        inputs=None,
        outputs=chain_df
    )