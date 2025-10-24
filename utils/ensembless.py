import os
import sys
import tempfile
import json
import gradio as gr
from datetime import datetime
from multi_inference import MVSEPLESS, clean_filename
from separator.ensemble import ensemble_audio_files

INVERT_METHODS = {
    "min_fft": "max_fft",
    "max_fft": "min_fft",
    "min_wave": "max_wave",
    "max_wave": "min_wave",
    "median_fft": "median_fft",
    "median_wave": "median_wave",
    "avg_fft": "avg_fft",
    "avg_wave": "avg_wave",
}


class ENSEMBLESS:
    def __init__(self):
        self.mvsepless = MVSEPLESS()
        self.output_app_base_dir = os.environ.get(
            "ENSEMBLESS_OUTPUT_DIR", os.path.join(os.getcwd(), "ensembless_output")
        )
        os.makedirs(self.output_app_base_dir, exist_ok=True)

    def get_model_types(self):
        return self.mvsepless.model_manager.get_mt()

    def get_models_by_type(self, model_type):
        return self.mvsepless.model_manager.get_mn(model_type)

    def get_stems_by_model(self, model_type, model_name):
        all_stems = []
        stems = self.mvsepless.model_manager.get_stems(model_type, model_name)
        for stem in stems:
            all_stems.append(stem)
        if (
            set(stems) == {"bass", "drums", "vocals", "other"}
            or set(stems) == {"bass", "drums", "vocals", "other", "piano", "guitar"}
            and not self.mvsepless.model_manager.get_tgt_inst(model_type, model_name)
        ):
            all_stems.append("instrumental +")
            all_stems.append("instrumental -")
        return all_stems

    def get_invert_stems_by_model(self, model_type, model_name, primary_stem):
        invert_stems = []
        stems = self.mvsepless.model_manager.get_stems(model_type, model_name)
        for stem in stems:
            if stem != primary_stem:
                invert_stems.append(stem)

        if (
            not self.mvsepless.model_manager.get_tgt_inst(model_type, model_name)
            and model_type not in ["vr", "mdx"]
            and primary_stem not in ["instrumental -", "instrumental +"]
        ):
            if len(stems) > 2:
                invert_stems.append("inverted +")
                invert_stems.append("inverted -")

        return invert_stems

    def invert_weights(self, weights):
        total_weight = sum(weights)
        return [total_weight - w for w in weights]

    def manual_ensemble(
        self, input_audios, method, weights, out_format, output_name=None
    ):
        temp_dir = tempfile.mkdtemp()
        weights = [float(x) for x in weights.split(",")]
        if not output_name or output_name == "":
            output_name = f"ensemble_{method}"
        output_path = os.path.join(temp_dir, output_name)
        a1, a2 = ensemble_audio_files(
            input_audios,
            output=output_path,
            ensemble_type=method,
            weights=weights,
            out_format=out_format,
        )
        return a1, a2

    def auto_ensemble(
        self,
        input_audio,
        input_settings,
        type,
        out_format,
        invert_weights,
        invert_ensemble,
    ):

        output_dir = os.path.join(
            self.output_app_base_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        os.makedirs(output_dir, exist_ok=True)

        progress = gr.Progress()
        progress(0, desc=None)

        base_name = os.path.splitext(os.path.basename(input_audio))[0]
        temp_dir = tempfile.mkdtemp()
        source_files = []
        output_p_files = []
        output_s_files = []
        output_p_weights = []

        block_count = len(input_settings)

        for i, (input_model, weight, p_stem, s_stem) in enumerate(input_settings):
            output_s_files.append(None)
            progress(i / block_count, desc=f"{i+1}/{block_count}")
            model_type, model_name = input_model.split(" / ")
            output_dir_p = os.path.join(temp_dir, f"{model_type}_{model_name}_p_stems")
            output_p = self.mvsepless.separator.base(
                input_file=input_audio,
                output_dir=output_dir_p,
                model_type=model_type,
                model_name=model_name,
                ext_inst=True,
                vr_aggr=10,
                output_format="wav",
                template="MODEL_STEM",
                call_method="cli",
            )
            for stem, file in output_p:
                source_files.append(file)
                if stem == p_stem:
                    output_p_files.append(file)
                    output_p_weights.append(weight)
                elif invert_ensemble:
                    if stem == s_stem:
                        output_s_files[i] = file

            if invert_ensemble:
                if not output_s_files[i]:

                    output_dir_s = os.path.join(
                        temp_dir, f"{model_type}_{model_name}_s_stems"
                    )
                    output_s = self.mvsepless.separator.base(
                        input_file=input_audio,
                        output_dir=output_dir_s,
                        model_type=model_type,
                        model_name=model_name,
                        ext_inst=True,
                        vr_aggr=10,
                        output_format="wav",
                        template="MODEL_STEM",
                        call_method="cli",
                        selected_stems=[
                            (
                                p_stem
                                if not self.mvsepless.model_manager.get_tgt_inst(
                                    model_type, model_name
                                )
                                else "both"
                            )
                        ],
                    )
                    for stem, file in output_s:
                        source_files.append(file)
                        if stem == s_stem:
                            output_s_files[i] = file
                            source_files.append(file)

        progress(0.95, desc=None)
        if invert_ensemble:
            if invert_weights:
                output_s_weights = self.invert_weights(output_p_weights)
            else:
                output_s_weights = output_p_weights
            output_s, output_wav_s = ensemble_audio_files(
                files=output_s_files,
                output=os.path.join(temp_dir, f"ensemble_invert_{base_name}_{type}"),
                ensemble_type=INVERT_METHODS[type],
                weights=output_s_weights,
                out_format=out_format,
            )
        else:
            output_s, output_wav_s = None, None

        output_p, output_wav_p = ensemble_audio_files(
            files=output_p_files,
            output=os.path.join(temp_dir, f"ensemble_{clean_filename(base_name, length=120)}_{type}"),
            ensemble_type=type,
            weights=output_p_weights,
            out_format=out_format,
        )

        return output_p, output_wav_p, output_s, output_wav_s, source_files
