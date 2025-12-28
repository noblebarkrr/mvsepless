import os
import subprocess
import gradio as gr
import soundfile as sf
import librosa
import numpy as np
from pedalboard import Pedalboard, Compressor, Reverb, Delay, NoiseGate, Chorus
import tempfile
import hashlib
import json
from pathlib import Path
import datetime
from audio import check, read, write, output_formats
from namer import Namer
from separator import Separator
from vbach import f0_methods
from downloader import dw_yt_dlp

class VbachGen(Separator):
    def __init__(self, model_manager, input_files, upload_files, user_directory, vbach_inference):
        super().__init__()
        self.namer = Namer()
        self.processing_data = {}
        self.separation_stages = {}
        self.conversion_cache = {}
        self.vbach_model_manager = model_manager
        self.vbach_inference = vbach_inference
        self.input_files = input_files
        self.upload_files = upload_files
        self.user_directory = user_directory
        
        # Создаем базовую директорию для хранения результатов
        self.output_base_directory = os.path.join(self.user_directory.path, "vbachgen")
        os.makedirs(self.output_base_directory, exist_ok=True)
        
        self.fairseq_embedders = list(
            self.vbach_model_manager.huberts_fairseq_dict.keys()
        )
        self.transformers_embedders = list(
            self.vbach_model_manager.huberts_transformers_dict.keys()
        )

    def get_output_directory(self, input_audio_path, stage, model_name=""):
        """Создает путь для сохранения файлов в постоянной директории"""
        basename = os.path.splitext(os.path.basename(input_audio_path))[0]
        # Заменяем недопустимые символы в имени файла
        basename = "".join(c if c.isalnum() or c in " _-" else "_" for c in basename)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if model_name:
            model_suffix = f"_{model_name}"
        else:
            model_suffix = ""
            
        output_dir = os.path.join(
            self.output_base_directory, 
            basename, 
            f"{timestamp}_{stage}{model_suffix}"
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def get_cache_key(self, params_dict):
        params_str = json.dumps(params_dict, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()

    def parse_voice_models_actual(self):
        return self.vbach_model_manager.parse_voice_models()

    def find_file_from_stem(self, results, stem_names=["Vocals", "vocals"]):
        for stem_name, stem_file in results:
            if stem_name in stem_names:
                return stem_file
        return None

    def extract_inst_voc(self, input_audio, model_name, progress=None):
        key = f"inst_voc_{hashlib.md5(input_audio.encode()).hexdigest()}_{model_name}"

        if key in self.separation_stages:
            print(
                f"Пропускаем разделение на инструментал и вокал (уже выполнено с моделью: {model_name})"
            )
            inst_file = self.separation_stages[key]["inst_file"]
            voc_file = self.separation_stages[key]["voc_file"]
            return inst_file, voc_file

        if progress:
            progress(0.2, desc="Извлечение инструментала и вокала...")

        # Используем постоянную директорию вместо временной
        output_dir = self.get_output_directory(input_audio, "inst_voc", model_name)
        
        inst_output = self.separate(
            input=input_audio,
            output_dir=output_dir,
            model_name=model_name,
            ext_inst=True,
            output_format="flac",
            template="VbachGen_NAME_STEM",
            add_settings={
                "add_single_sep_text_progress": "Извлечение инструментала и вокала..."
            },
            progress=progress,
        )

        inst_file = self.find_file_from_stem(
            inst_output,
            [
                "Instrument",
                "instrument",
                "Instrumental",
                "instrumental",
                "other",
                "Other",
            ],
        )
        voc_file = self.find_file_from_stem(inst_output, ["Vocals", "vocals"])

        self.separation_stages[key] = {
            "inst_file": inst_file,
            "voc_file": voc_file,
            "model": model_name,
            "input_file": input_audio,
        }

        return inst_file, voc_file

    def extract_lead_back(self, vocals_file, model_name, progress=None):
        if not vocals_file:
            return None, None

        key = f"lead_back_{hashlib.md5(vocals_file.encode()).hexdigest()}_{model_name}"

        if key in self.separation_stages:
            print(
                f"Пропускаем разделение на лид и бэк-вокал (уже выполнено с моделью: {model_name})"
            )
            lead_file = self.separation_stages[key]["lead_file"]
            back_file = self.separation_stages[key]["back_file"]
            return lead_file, back_file

        if progress:
            progress(0.4, desc="Извлечение лид и бэк-вокала...")

        # Используем постоянную директорию
        output_dir = self.get_output_directory(vocals_file, "lead_back", model_name)
        
        karaoke_output = self.separate(
            input=vocals_file,
            output_dir=output_dir,
            model_name=model_name,
            ext_inst=True,
            output_format="flac",
            template="karaoke_NAME_STEM",
            add_settings={
                "add_single_sep_text_progress": "Извлечение лид и бэк-вокала..."
            },
            progress=progress,
        )

        back_file = self.find_file_from_stem(
            karaoke_output,
            ["Instrumental", "instrumental", "other", "Other", "back", "Back"],
        )
        lead_file = self.find_file_from_stem(
            karaoke_output, ["Vocals", "vocals", "karaoke", "lead", "Lead"]
        )

        self.separation_stages[key] = {
            "lead_file": lead_file,
            "back_file": back_file,
            "model": model_name,
            "input_file": vocals_file,
        }

        return lead_file, back_file

    def clear_vocals(self, vocals_file, model_name, progress=None, vocal_type="vocals"):
        if not vocals_file:
            return None

        key = f"clear_{vocal_type}_{hashlib.md5(vocals_file.encode()).hexdigest()}_{model_name}"

        if key in self.separation_stages:
            print(
                f"Пропускаем очистку {vocal_type} (уже выполнено с моделью: {model_name})"
            )
            return self.separation_stages[key]["cleared_file"]

        if progress:
            progress(0.6, desc=f"Очистка {vocal_type}...")

        # Используем постоянную директорию
        output_dir = self.get_output_directory(vocals_file, f"clear_{vocal_type}", model_name)
        
        clear_output = self.separate(
            input=vocals_file,
            output_dir=output_dir,
            model_name=model_name,
            ext_inst=True,
            output_format="flac",
            template="precleared_NAME_STEM",
            add_settings={"add_single_sep_text_progress": f"Очистка {vocal_type}..."},
            progress=progress,
        )

        cleared_file = self.find_file_from_stem(
            clear_output, ["No Echo", "No Reverb", "Dry"]
        )

        self.separation_stages[key] = {
            "cleared_file": cleared_file,
            "model": model_name,
            "input_file": vocals_file,
            "vocal_type": vocal_type,
        }

        return cleared_file

    def separation_only(
        self,
        input_audio,
        anti_instrum_model,
        karaoke_model,
        dereverb_model,
        karaoke_check,
        preclear_vocals_check,
        progress=None,
    ):
        if progress is None:
            progress = gr.Progress(track_tqdm=True)

        progress(0, desc="Начало разделения...")

        progress(0.1, desc="Проверка предыдущих результатов разделения...")

        inst_file = None
        full_vocals_file = None
        back_vocals_file = None
        lead_vocals_file = None

        inst_voc_key = f"inst_voc_{hashlib.md5(input_audio.encode()).hexdigest()}_{anti_instrum_model}"

        if inst_voc_key in self.separation_stages:
            print("Используем ранее извлеченные инструментал и вокал")
            inst_file = self.separation_stages[inst_voc_key]["inst_file"]
            full_vocals_file = self.separation_stages[inst_voc_key]["voc_file"]
            progress(0.2, desc="Пропуск: инструментал и вокал уже извлечены")
        else:
            progress(0.2, desc="Извлечение инструментала и вокала...")
            inst_file, full_vocals_file = self.extract_inst_voc(
                input_audio, anti_instrum_model, progress
            )

        if karaoke_check and full_vocals_file:
            lead_back_key = f"lead_back_{hashlib.md5(full_vocals_file.encode()).hexdigest()}_{karaoke_model}"

            if lead_back_key in self.separation_stages:
                print("Используем ранее извлеченные лид и бэк-вокалы")
                lead_vocals_file = self.separation_stages[lead_back_key]["lead_file"]
                back_vocals_file = self.separation_stages[lead_back_key]["back_file"]
                progress(0.4, desc="Пропуск: лид и бэк-вокалы уже извлечены")
            else:
                progress(0.4, desc="Извлечение лид и бэк-вокала...")
                lead_vocals_file, back_vocals_file = self.extract_lead_back(
                    full_vocals_file, karaoke_model, progress
                )

        cleared_vocals = []
        if preclear_vocals_check:
            if full_vocals_file:
                clear_key = f"clear_vocals_{hashlib.md5(full_vocals_file.encode()).hexdigest()}_{dereverb_model}"

                if clear_key in self.separation_stages:
                    print("Используем ранее очищенный полный вокал")
                    cleared_full_vocals = self.separation_stages[clear_key][
                        "cleared_file"
                    ]
                    cleared_vocals.append(("full_vocals", cleared_full_vocals))
                    progress(0.6, desc="Пропуск: полный вокал уже очищен")
                else:
                    progress(0.6, desc="Очистка полного вокала...")
                    cleared_full_vocals = self.clear_vocals(
                        full_vocals_file, dereverb_model, progress, vocal_type="vocals"
                    )
                    if cleared_full_vocals:
                        cleared_vocals.append(("full_vocals", cleared_full_vocals))

            if lead_vocals_file:
                clear_key = f"clear_lead_{hashlib.md5(lead_vocals_file.encode()).hexdigest()}_{dereverb_model}"

                if clear_key in self.separation_stages:
                    print("Используем ранее очищенный лид-вокал")
                    cleared_lead_vocals = self.separation_stages[clear_key][
                        "cleared_file"
                    ]
                    cleared_vocals.append(("lead_vocals", cleared_lead_vocals))
                else:
                    cleared_lead_vocals = self.clear_vocals(
                        lead_vocals_file, dereverb_model, progress, vocal_type="lead"
                    )
                    if cleared_lead_vocals:
                        cleared_vocals.append(("lead_vocals", cleared_lead_vocals))

            if back_vocals_file:
                clear_key = f"clear_back_{hashlib.md5(back_vocals_file.encode()).hexdigest()}_{dereverb_model}"

                if clear_key in self.separation_stages:
                    print("Используем ранее очищенный бэк-вокал")
                    cleared_back_vocals = self.separation_stages[clear_key][
                        "cleared_file"
                    ]
                    cleared_vocals.append(("back_vocals", cleared_back_vocals))
                else:
                    cleared_back_vocals = self.clear_vocals(
                        back_vocals_file, dereverb_model, progress, vocal_type="back"
                    )
                    if cleared_back_vocals:
                        cleared_vocals.append(("back_vocals", cleared_back_vocals))

        list_vocals = [
            ("full_vocals", full_vocals_file),
            ("back_vocals", back_vocals_file),
            ("lead_vocals", lead_vocals_file),
        ]

        for cleared_name, cleared_file in cleared_vocals:
            for i, (name, file) in enumerate(list_vocals):
                if name == cleared_name:
                    list_vocals[i] = (name, cleared_file)
                    break

        generated_files = []
        if inst_file:
            generated_files.append(inst_file)
        for name, file in list_vocals:
            if file:
                generated_files.append(file)

        progress(1.0, desc="Разделение завершено")

        # Создаем директорию для результатов разделения
        separation_dir = self.get_output_directory(input_audio, "separation_results")
        
        return {
            "inst_file": inst_file,
            "list_vocals": list_vocals,
            "temp_dir": separation_dir,  # Используем постоянную директорию
            "generated_files": generated_files,
        }

    def clear_separation_cache(self):
        self.separation_stages.clear()
        print("Кэш разделения очищен")

    def conversion_only(
        self,
        separation_result,
        voice_name,
        conversion_mode,
        pitch1_val,
        pitch2_val,
        method_pitch,
        index_rate,
        fr,
        rms,
        protect,
        hop_mangio_crepe,
        f0_max,
        output_format,
        unconv_vocals_check,
        use_effects,
        instrumental_gain,
        vocal1_gain,
        vocal2_gain,
        echo_delay,
        echo_feedback,
        echo_mix,
        reverb_rm_size,
        reverb_width,
        reverb_wet,
        reverb_dry,
        reverb_damping,
        chorus_rate_hz,
        chorus_depth,
        chorus_centre_delay_ms,
        chorus_feedback,
        chorus_mix,
        compressor_ratio,
        compressor_threshold,
        compressor_attack,
        compressor_release,
        noise_gate_threshold,
        noise_gate_ratio,
        noise_gate_attack,
        noise_gate_release,
        embedder_name=None,
        transformers_mode=False,
        input_audio=None,
        progress=None,
        always_new_conversion=True,
    ):
        conversion_params = {
            "voice_name": voice_name,
            "conversion_mode": conversion_mode,
            "pitch1_val": pitch1_val,
            "pitch2_val": pitch2_val,
            "method_pitch": method_pitch,
            "index_rate": index_rate,
            "fr": fr,
            "rms": rms,
            "protect": protect,
            "hop_mangio_crepe": hop_mangio_crepe,
            "f0_max": f0_max,
            "output_format": output_format,
            "unconv_vocals_check": unconv_vocals_check,
            "use_effects": use_effects,
            "instrumental_gain": instrumental_gain,
            "vocal1_gain": vocal1_gain,
            "vocal2_gain": vocal2_gain,
            "echo_delay": echo_delay,
            "echo_feedback": echo_feedback,
            "echo_mix": echo_mix,
            "reverb_rm_size": reverb_rm_size,
            "reverb_width": reverb_width,
            "reverb_wet": reverb_wet,
            "reverb_dry": reverb_dry,
            "reverb_damping": reverb_damping,
            "chorus_rate_hz": chorus_rate_hz,
            "chorus_depth": chorus_depth,
            "chorus_centre_delay_ms": chorus_centre_delay_ms,
            "chorus_feedback": chorus_feedback,
            "chorus_mix": chorus_mix,
            "compressor_ratio": compressor_ratio,
            "compressor_threshold": compressor_threshold,
            "compressor_attack": compressor_attack,
            "compressor_release": compressor_release,
            "noise_gate_threshold": noise_gate_threshold,
            "noise_gate_ratio": noise_gate_ratio,
            "noise_gate_attack": noise_gate_attack,
            "noise_gate_release": noise_gate_release,
            "embedder_name": embedder_name,
            "transformers_mode": transformers_mode,
        }
        cache_key = self.get_cache_key(conversion_params)

        if not always_new_conversion:
            if cache_key in self.conversion_cache:
                print("Используем кэшированные результаты преобразования")
                return self.conversion_cache[cache_key]

        if progress is None:
            progress = gr.Progress(track_tqdm=True)

        progress(0, desc="Начало преобразования...")

        inst_path = separation_result["inst_file"]
        list_vocals = separation_result["list_vocals"]
        temp_dir = separation_result["temp_dir"]

        rvc_params = {
            "model_name": voice_name,
            "pitch1": pitch1_val,
            "pitch2": pitch2_val,
            "f0_method": method_pitch,
            "index_rate": index_rate,
            "filter_radius": fr,
            "rms": rms,
            "protect": protect,
            "hop_length": hop_mangio_crepe,
            "f0_max": f0_max,
            "embedder_name": embedder_name,
            "transformers_mode": transformers_mode,
        }

        params = {"output_format": output_format, "conversion_mode": conversion_mode}

        mix_params = {
            "add_unconverted_vocals_to_instrumental": unconv_vocals_check,
            "use_effects": use_effects,
            "gain": {
                "instrum": instrumental_gain,
                "vocals1": vocal1_gain,
                "vocals2": vocal2_gain,
            },
            "pedalboard_settings": {
                "echo": {
                    "delay": echo_delay,
                    "feedback": echo_feedback,
                    "mix": echo_mix,
                },
                "reverb": {
                    "room_size": reverb_rm_size,
                    "wet": reverb_wet,
                    "dry": reverb_dry,
                    "damping": reverb_damping,
                    "width": reverb_width,
                },
                "compressor": {
                    "ratio": compressor_ratio,
                    "threshold": compressor_threshold,
                    "attack": compressor_attack,
                    "release": compressor_release,
                },
                "noise_gate": {
                    "threshold": noise_gate_threshold,
                    "ratio": noise_gate_ratio,
                    "attack": noise_gate_attack,
                    "release": noise_gate_release,
                },
                "chorus": {
                    "rate": chorus_rate_hz,
                    "depth": chorus_depth,
                    "center_delay": chorus_centre_delay_ms,
                    "feedback": chorus_feedback,
                    "mix": chorus_mix,
                },
            },
        }

        progress(0.3, desc="Преобразование вокалов...")

        converted_vocals_list = []
        
        # Создаем директорию для преобразованных файлов
        conversion_dir = self.get_output_directory(input_audio, "converted", voice_name)
        
        full_vocals_file = next(
            (f[1] for f in list_vocals if f[0] == "full_vocals"), None
        )
        back_vocals_file = next(
            (f[1] for f in list_vocals if f[0] == "back_vocals"), None
        )
        lead_vocals_file = next(
            (f[1] for f in list_vocals if f[0] == "lead_vocals"), None
        )

        stack = "transformers" if transformers_mode else "fairseq"

        if conversion_mode == "full" and full_vocals_file:
            full_vocals_converted_path = self.vbach_inference(
                input_file=full_vocals_file,
                output_dir=conversion_dir,
                model_name=rvc_params["model_name"],
                format_name=False,
                output_name=f"full_vocals_converted-{self.namer.short(os.path.splitext(os.path.basename(input_audio))[0], length=60)}",
                pitch=rvc_params["pitch1"],
                method_pitch=rvc_params["f0_method"],
                output_bitrate=320,
                output_format="flac",
                pipeline_mode="alt",
                embedder_name=embedder_name,
                stack=stack,
                add_params={
                    "index_rate": rvc_params["index_rate"],
                    "filter_radius": rvc_params["filter_radius"],
                    "protect": rvc_params["protect"],
                    "rms": rvc_params["rms"],
                    "mangio_crepe_hop_length": rvc_params["hop_length"],
                    "f0_min": 50,
                    "f0_max": rvc_params["f0_max"],
                    "stereo_mode": "mono",
                },
            )
            converted_vocals_list.append(full_vocals_converted_path)

        elif conversion_mode == "lead/back" and lead_vocals_file and back_vocals_file:
            lead_vocals_converted_path = self.vbach_inference(
                input_file=lead_vocals_file,
                output_dir=conversion_dir,
                model_name=rvc_params["model_name"],
                format_name=False,
                output_name=f"lead_vocals_converted-{self.namer.short(os.path.splitext(os.path.basename(input_audio))[0], length=60)}",
                pitch=rvc_params["pitch1"],
                method_pitch=rvc_params["f0_method"],
                output_bitrate=320,
                output_format="flac",
                pipeline_mode="alt",
                embedder_name=embedder_name,
                stack=stack,
                add_params={
                    "index_rate": rvc_params["index_rate"],
                    "filter_radius": rvc_params["filter_radius"],
                    "protect": rvc_params["protect"],
                    "rms": rvc_params["rms"],
                    "mangio_crepe_hop_length": rvc_params["hop_length"],
                    "f0_min": 50,
                    "f0_max": rvc_params["f0_max"],
                    "stereo_mode": "mono",
                },
            )

            back_vocals_converted_path = self.vbach_inference(
                input_file=back_vocals_file,
                output_dir=conversion_dir,
                model_name=rvc_params["model_name"],
                format_name=False,
                output_name=f"back_vocals_converted-{self.namer.short(os.path.splitext(os.path.basename(input_audio))[0], length=60)}",
                pitch=rvc_params["pitch2"],
                method_pitch=rvc_params["f0_method"],
                output_bitrate=320,
                output_format="flac",
                pipeline_mode="alt",
                embedder_name=embedder_name,
                stack=stack,
                add_params={
                    "index_rate": rvc_params["index_rate"],
                    "filter_radius": rvc_params["filter_radius"],
                    "protect": rvc_params["protect"],
                    "rms": rvc_params["rms"],
                    "mangio_crepe_hop_length": rvc_params["hop_length"],
                    "f0_min": 50,
                    "f0_max": rvc_params["f0_max"],
                    "stereo_mode": "mono",
                },
            )

            converted_vocals_list.append(back_vocals_converted_path)
            converted_vocals_list.append(lead_vocals_converted_path)

        elif conversion_mode == "back" and back_vocals_file:
            back_vocals_converted_path = self.vbach_inference(
                input_file=back_vocals_file,
                output_dir=conversion_dir,
                model_name=rvc_params["model_name"],
                format_name=False,
                output_name=f"back_vocals_converted-{self.namer.short(os.path.splitext(os.path.basename(input_audio))[0], length=60)}",
                pitch=rvc_params["pitch2"],
                method_pitch=rvc_params["f0_method"],
                output_bitrate=320,
                output_format="flac",
                pipeline_mode="alt",
                embedder_name=embedder_name,
                stack=stack,
                add_params={
                    "index_rate": rvc_params["index_rate"],
                    "filter_radius": rvc_params["filter_radius"],
                    "protect": rvc_params["protect"],
                    "rms": rvc_params["rms"],
                    "mangio_crepe_hop_length": rvc_params["hop_length"],
                    "f0_min": 50,
                    "f0_max": rvc_params["f0_max"],
                    "stereo_mode": "mono",
                },
            )
            converted_vocals_list.append(back_vocals_converted_path)

        elif conversion_mode == "lead" and lead_vocals_file:
            lead_vocals_converted_path = self.vbach_inference(
                input_file=lead_vocals_file,
                output_dir=conversion_dir,
                model_name=rvc_params["model_name"],
                format_name=False,
                output_name=f"lead_vocals_converted-{self.namer.short(os.path.splitext(os.path.basename(input_audio))[0], length=60)}",
                pitch=rvc_params["pitch1"],
                method_pitch=rvc_params["f0_method"],
                output_bitrate=320,
                output_format="flac",
                pipeline_mode="alt",
                embedder_name=embedder_name,
                stack=stack,
                add_params={
                    "index_rate": rvc_params["index_rate"],
                    "filter_radius": rvc_params["filter_radius"],
                    "protect": rvc_params["protect"],
                    "rms": rvc_params["rms"],
                    "mangio_crepe_hop_length": rvc_params["hop_length"],
                    "f0_min": 50,
                    "f0_max": rvc_params["f0_max"],
                    "stereo_mode": "mono",
                },
            )
            converted_vocals_list.append(lead_vocals_converted_path)

        generated_files = []
        if inst_path:
            generated_files.append(inst_path)
        for name, file in list_vocals:
            if file:
                generated_files.append(file)
        generated_files.extend(converted_vocals_list)

        self.processing_data = {
            "inst_path": inst_path,
            "list_vocals": list_vocals,
            "converted_vocals_list": converted_vocals_list,
            "params": params,
            "rvc_params": rvc_params,
            "input_audio": input_audio,
        }

        progress(0.7, desc="Сведение итогового кавера...")

        final_path = self.mix_and_save(
            inst_path,
            list_vocals,
            converted_vocals_list,
            mix_params,
            params,
            rvc_params,
            temp_dir,
            input_audio,
        )

        generated_files.append(final_path)

        result = {
            "generated_files": generated_files,
            "final_path": final_path,
            "converted_vocals_list": converted_vocals_list,
        }
        self.conversion_cache[cache_key] = result

        progress(1.0, desc="Преобразование завершено")

        return result

    def mix_and_save(
        self,
        inst_path,
        list_vocals,
        converted_vocals_list,
        mix_params,
        params,
        rvc_params,
        temp_dir,
        input_audio,
    ):
        final_audio = None
        samplerate = 44100
        
        # Функция для выравнивания длин аудио
        def align_audio_length(audio1, audio2):
            if audio1 is None:
                return audio2
            if audio2 is None:
                return audio1
                
            # Получаем минимальную длину
            min_len = min(audio1.shape[1], audio2.shape[1])
            
            # Обрезаем оба массива до минимальной длины
            if audio1.shape[1] > min_len:
                audio1 = audio1[:, :min_len]
            if audio2.shape[1] > min_len:
                audio2 = audio2[:, :min_len]
                
            return audio1, audio2

        if inst_path and os.path.exists(inst_path):
            inst_data, samplerate = read(
                path=inst_path, mono=False, sr=None, dtype="float32"
            )
            inst_gain = 10 ** (mix_params["gain"]["instrum"] / 20.0)
            inst_data *= inst_gain
            final_audio = inst_data.copy()

        if mix_params["add_unconverted_vocals_to_instrumental"]:
            full_vocals_file = next(
                (f[1] for f in list_vocals if f[0] == "full_vocals"), None
            )
            back_vocals_file = next(
                (f[1] for f in list_vocals if f[0] == "back_vocals"), None
            )
            lead_vocals_file = next(
                (f[1] for f in list_vocals if f[0] == "lead_vocals"), None
            )

            conversion_mode = params.get("conversion_mode", "full")
            
            if conversion_mode == "lead" and back_vocals_file:
                back_vocals, sr = read(
                    path=back_vocals_file, mono=False, sr=samplerate, dtype="float32"
                )
                back_vocals *= (10 ** (mix_params["gain"]["vocals2"] / 20.0))
                
                if final_audio is None:
                    final_audio = back_vocals
                else:
                    # Выравниваем длины перед сложением
                    final_audio, back_vocals = align_audio_length(final_audio, back_vocals)
                    final_audio = final_audio + back_vocals
                    
            elif conversion_mode == "back" and lead_vocals_file:
                lead_vocals, sr = read(
                    path=lead_vocals_file, mono=False, sr=samplerate, dtype="float32"
                )
                lead_vocals *= (10 ** (mix_params["gain"]["vocals1"] / 20.0))
                
                if final_audio is None:
                    final_audio = lead_vocals
                else:
                    # Выравниваем длины перед сложением
                    final_audio, lead_vocals = align_audio_length(final_audio, lead_vocals)
                    final_audio = final_audio + lead_vocals
                    
            elif conversion_mode == "lead/back":
                # В режиме lead/back НЕ добавляем непереобразованные вокалы
                # так как мы преобразовываем и лид, и бэк отдельно
                pass

        # Определяем порядок вокалов в зависимости от режима преобразования
        conversion_mode = params.get("conversion_mode", "full")
        
        for i, vocal_path in enumerate(converted_vocals_list):
            if not vocal_path or not os.path.exists(vocal_path):
                continue

            vocal_data, sr = read(
                path=vocal_path, mono=False, sr=samplerate, dtype="float32"
            )

            if mix_params.get("use_effects", False):
                board = Pedalboard()
                effects = mix_params.get("pedalboard_settings", {})

                if "compressor" in effects:
                    comp = effects["compressor"]
                    board.append(
                        Compressor(
                            ratio=comp["ratio"],
                            threshold_db=comp["threshold"],
                            attack_ms=comp["attack"],
                            release_ms=comp["release"],
                        )
                    )

                if "noise_gate" in effects:
                    ng = effects["noise_gate"]
                    board.append(
                        NoiseGate(
                            threshold_db=ng["threshold"],
                            ratio=ng["ratio"],
                            attack_ms=ng["attack"],
                            release_ms=ng["release"],
                        )
                    )

                if "echo" in effects:
                    echo = effects["echo"]
                    board.append(
                        Delay(
                            delay_seconds=echo["delay"],
                            feedback=echo["feedback"],
                            mix=echo["mix"],
                        )
                    )

                if "reverb" in effects:
                    rev = effects["reverb"]
                    board.append(
                        Reverb(
                            room_size=rev["room_size"],
                            dry_level=rev["dry"],
                            wet_level=rev["wet"],
                            damping=rev["damping"],
                            width=rev["width"],
                        )
                    )

                if "chorus" in effects:
                    chorus = effects["chorus"]
                    board.append(
                        Chorus(
                            rate_hz=chorus["rate"],
                            depth=chorus["depth"],
                            centre_delay_ms=chorus["center_delay"],
                            feedback=chorus["feedback"],
                            mix=chorus["mix"],
                        )
                    )

                vocal_data = board(vocal_data, samplerate)

            # Определяем какой gain использовать для каждого вокала
            if conversion_mode == "lead/back":
                # В режиме lead/back первый вокал в списке - бэк-вокал, второй - лид-вокал
                # Это соответствует порядку в методе conversion_only
                if "back_vocals_converted" in os.path.basename(vocal_path) or i == 0:
                    gain_db = mix_params["gain"]["vocals2"]  # Бэк-вокал
                else:
                    gain_db = mix_params["gain"]["vocals1"]  # Лид-вокал
            else:
                # Для других режимов используем стандартную логику
                gain_db = (
                    mix_params["gain"]["vocals1"]
                    if i == 0
                    else mix_params["gain"]["vocals2"]
                )
            
            vocal_data *= 10 ** (gain_db / 20.0)

            if final_audio is None:
                final_audio = vocal_data.copy()
            else:
                # Выравниваем длины перед сложением
                final_audio, vocal_data = align_audio_length(final_audio, vocal_data)
                final_audio = final_audio + vocal_data

        # Убеждаемся, что final_audio не None
        if final_audio is None:
            # Создаем пустой аудиофайл с минимальной длиной
            final_audio = np.zeros((2, samplerate), dtype=np.float32)

        # Нормализация
        max_amplitude = np.max(np.abs(final_audio))
        if max_amplitude > 0:
            final_audio = final_audio / max_amplitude
            
        filename = (
            f"{rvc_params['model_name']} - {self.namer.short(os.path.splitext(os.path.basename(input_audio))[0], length=60)}.{params['output_format']}"
            if input_audio
            else f"remixed.{params['output_format']}"
        )
        
        # Сохраняем в постоянной директории
        final_dir = self.get_output_directory(input_audio, "final_cover")
        final_path = os.path.join(final_dir, filename)
        
        final_path = write(
            final_path,
            final_audio,
            samplerate,
            "320k",
        )

        return final_path

    def gen_cover(
        self,
        input_audio,
        anti_instrum_model,
        karaoke_model,
        dereverb_model,
        output_format,
        karaoke_check,
        conversion_mode,
        preclear_vocals_check,
        voice_name,
        pitch1_val,
        pitch2_val,
        method_pitch,
        index_rate,
        fr,
        rms,
        protect,
        hop_mangio_crepe,
        f0_max,
        unconv_vocals_check,
        use_effects,
        instrumental_gain,
        vocal1_gain,
        vocal2_gain,
        echo_delay,
        echo_feedback,
        echo_mix,
        reverb_rm_size,
        reverb_width,
        reverb_wet,
        reverb_dry,
        reverb_damping,
        chorus_rate_hz,
        chorus_depth,
        chorus_centre_delay_ms,
        chorus_feedback,
        chorus_mix,
        compressor_ratio,
        compressor_threshold,
        compressor_attack,
        compressor_release,
        noise_gate_threshold,
        noise_gate_ratio,
        noise_gate_attack,
        noise_gate_release,
        embedder_name,
        transformers_mode,
    ):
        if not input_audio:
            raise gr.Error("Сначала загрузите аудио")

        if not voice_name:
            raise gr.Error("Сначала выберите модель")

        progress = gr.Progress(track_tqdm=True)
        progress(0, desc="Начало обработки...")

        progress(0.1, desc="Этап разделения...")
        separation_result = self.separation_only(
            input_audio=input_audio,
            anti_instrum_model=anti_instrum_model,
            karaoke_model=karaoke_model,
            dereverb_model=dereverb_model,
            karaoke_check=karaoke_check,
            preclear_vocals_check=preclear_vocals_check,
            progress=progress,
        )

        progress(0.5, desc="Этап преобразования и сведения...")
        conversion_result = self.conversion_only(
            separation_result=separation_result,
            voice_name=voice_name,
            conversion_mode=conversion_mode,
            pitch1_val=pitch1_val,
            pitch2_val=pitch2_val,
            method_pitch=method_pitch,
            index_rate=index_rate,
            fr=fr,
            rms=rms,
            protect=protect,
            hop_mangio_crepe=hop_mangio_crepe,
            f0_max=f0_max,
            output_format=output_format,
            unconv_vocals_check=unconv_vocals_check,
            use_effects=use_effects,
            instrumental_gain=instrumental_gain,
            vocal1_gain=vocal1_gain,
            vocal2_gain=vocal2_gain,
            echo_delay=echo_delay,
            echo_feedback=echo_feedback,
            echo_mix=echo_mix,
            reverb_rm_size=reverb_rm_size,
            reverb_width=reverb_width,
            reverb_wet=reverb_wet,
            reverb_dry=reverb_dry,
            reverb_damping=reverb_damping,
            chorus_rate_hz=chorus_rate_hz,
            chorus_depth=chorus_depth,
            chorus_centre_delay_ms=chorus_centre_delay_ms,
            chorus_feedback=chorus_feedback,
            chorus_mix=chorus_mix,
            compressor_ratio=compressor_ratio,
            compressor_threshold=compressor_threshold,
            compressor_attack=compressor_attack,
            compressor_release=compressor_release,
            noise_gate_threshold=noise_gate_threshold,
            noise_gate_ratio=noise_gate_ratio,
            noise_gate_attack=noise_gate_attack,
            noise_gate_release=noise_gate_release,
            embedder_name=embedder_name,
            transformers_mode=transformers_mode,
            input_audio=input_audio,
            progress=progress,
        )

        return conversion_result["generated_files"], conversion_result["final_path"]

    def regenerate_conversion(
        self,
        voice_name,
        conversion_mode,
        pitch1_val,
        pitch2_val,
        method_pitch,
        index_rate,
        fr,
        rms,
        protect,
        hop_mangio_crepe,
        f0_max,
        output_format,
        unconv_vocals_check,
        use_effects,
        instrumental_gain,
        vocal1_gain,
        vocal2_gain,
        echo_delay,
        echo_feedback,
        echo_mix,
        reverb_rm_size,
        reverb_width,
        reverb_wet,
        reverb_dry,
        reverb_damping,
        chorus_rate_hz,
        chorus_depth,
        chorus_centre_delay_ms,
        chorus_feedback,
        chorus_mix,
        compressor_ratio,
        compressor_threshold,
        compressor_attack,
        compressor_release,
        noise_gate_threshold,
        noise_gate_ratio,
        noise_gate_attack,
        noise_gate_release,
        embedder_name,
        transformers_mode,
    ):
        if not self.processing_data:
            raise gr.Error("Сначала выполните полную генерацию!")

        progress = gr.Progress(track_tqdm=True)
        progress(0, desc="Перегенерация преобразования...")

        separation_result = {
            "inst_file": self.processing_data["inst_path"],
            "list_vocals": self.processing_data["list_vocals"],
            "temp_dir": tempfile.mkdtemp(),
        }

        conversion_result = self.conversion_only(
            separation_result=separation_result,
            voice_name=voice_name,
            conversion_mode=conversion_mode,
            pitch1_val=pitch1_val,
            pitch2_val=pitch2_val,
            method_pitch=method_pitch,
            index_rate=index_rate,
            fr=fr,
            rms=rms,
            protect=protect,
            hop_mangio_crepe=hop_mangio_crepe,
            f0_max=f0_max,
            output_format=output_format,
            unconv_vocals_check=unconv_vocals_check,
            use_effects=use_effects,
            instrumental_gain=instrumental_gain,
            vocal1_gain=vocal1_gain,
            vocal2_gain=vocal2_gain,
            echo_delay=echo_delay,
            echo_feedback=echo_feedback,
            echo_mix=echo_mix,
            reverb_rm_size=reverb_rm_size,
            reverb_width=reverb_width,
            reverb_wet=reverb_wet,
            reverb_dry=reverb_dry,
            reverb_damping=reverb_damping,
            chorus_rate_hz=chorus_rate_hz,
            chorus_depth=chorus_depth,
            chorus_centre_delay_ms=chorus_centre_delay_ms,
            chorus_feedback=chorus_feedback,
            chorus_mix=chorus_mix,
            compressor_ratio=compressor_ratio,
            compressor_threshold=compressor_threshold,
            compressor_attack=compressor_attack,
            compressor_release=compressor_release,
            noise_gate_threshold=noise_gate_threshold,
            noise_gate_ratio=noise_gate_ratio,
            noise_gate_attack=noise_gate_attack,
            noise_gate_release=noise_gate_release,
            embedder_name=embedder_name,
            transformers_mode=transformers_mode,
            input_audio=self.processing_data.get("input_audio"),
            progress=progress,
        )

        return conversion_result["generated_files"], conversion_result["final_path"]

    def remix_cover(
        self,
        use_effects,
        instrumental_gain,
        vocal1_gain,
        vocal2_gain,
        echo_delay,
        echo_feedback,
        echo_mix,
        reverb_rm_size,
        reverb_width,
        reverb_wet,
        reverb_dry,
        reverb_damping,
        chorus_rate_hz,
        chorus_depth,
        chorus_centre_delay_ms,
        chorus_feedback,
        chorus_mix,
        compressor_ratio,
        compressor_threshold,
        compressor_attack,
        compressor_release,
        noise_gate_threshold,
        noise_gate_ratio,
        noise_gate_attack,
        noise_gate_release,
    ):
        if not self.processing_data:
            raise gr.Error("Сначала сгенерируйте кавер хотя бы один раз!")

        data = self.processing_data
        temp_dir = tempfile.mkdtemp()

        mix_params = {
            "add_unconverted_vocals_to_instrumental": True,
            "use_effects": use_effects,
            "gain": {
                "instrum": instrumental_gain,
                "vocals1": vocal1_gain,
                "vocals2": vocal2_gain,
            },
            "pedalboard_settings": {
                "echo": {
                    "delay": echo_delay,
                    "feedback": echo_feedback,
                    "mix": echo_mix,
                },
                "reverb": {
                    "room_size": reverb_rm_size,
                    "wet": reverb_wet,
                    "dry": reverb_dry,
                    "damping": reverb_damping,
                    "width": reverb_width,
                },
                "compressor": {
                    "ratio": compressor_ratio,
                    "threshold": compressor_threshold,
                    "attack": compressor_attack,
                    "release": compressor_release,
                },
                "noise_gate": {
                    "threshold": noise_gate_threshold,
                    "ratio": noise_gate_ratio,
                    "attack": noise_gate_attack,
                    "release": noise_gate_release,
                },
                "chorus": {
                    "rate": chorus_rate_hz,
                    "depth": chorus_depth,
                    "center_delay": chorus_centre_delay_ms,
                    "feedback": chorus_feedback,
                    "mix": chorus_mix,
                },
            },
        }

        final_path = self.mix_and_save(
            data["inst_path"],
            data["list_vocals"],
            data["converted_vocals_list"],
            mix_params,
            data["params"],
            data["rvc_params"],
            temp_dir,
            data["input_audio"],
        )

        return final_path

    def UI(self):

        with gr.Row(equal_height=False, variant="panel"):
            with gr.Group():
                model_name = gr.Dropdown(
                    label="Имя модели", interactive=True, filterable=False, scale=6
                )
                model_update_btn = gr.Button(
                    "Обновить", variant="primary", scale=3, size="lg"
                )
            with gr.Row(min_height=150):
                with gr.Group():
                    with gr.Group():
                        upload = gr.File(show_label=False, type="filepath", interactive=True)
                        refresh_input_btn = gr.Button("Обновить", variant="primary", interactive=True)
                        list_input_files = gr.Dropdown(
                            label="Загрузить файлы",
                            choices=self.input_files,
                            value=None,
                            multiselect=False,
                            interactive=True,
                            filterable=False, scale=15
                        )
                        refresh_input_btn.click(lambda: gr.update(choices=reversed(self.input_files), value=None), outputs=list_input_files)
                            
                        @upload.upload(inputs=[upload], outputs=[list_input_files, upload])
                        def upload_files(input_file):
                            files = self.upload_files([input_file])
                            return gr.update(
                                choices=reversed(self.input_files), value=files[0]
                            ), gr.update(value=None)

        with gr.Row():
            with gr.Column():

                with gr.Tab("Разделение"):
                    preclear_vocals_check = gr.Checkbox(
                        label="Очистить вокал от реверба/эха", value=False
                    )
                    karaoke_check = gr.Checkbox(
                        label="Разделить вокал на лид/бэк-вокалы", value=False
                    )

                    with gr.Column(variant="panel"):
                        with gr.Group() as extract_vocals_group:
                            anti_instrum_model = gr.Dropdown(
                                label="Вокальная модель",
                                choices=self.get_list_mn_from_category(
                                    ["Инструментал", "Вокал", "Инструментал и вокал"],
                                    [
                                        "mel_band_roformer",
                                        "bs_roformer",
                                        "mdx23c",
                                        "mdxnet",
                                        "htdemucs",
                                    ],
                                ),
                                interactive=True,
                                filterable=False,
                            )

                        with gr.Group(visible=False) as deecho_group:
                            dereverb_model = gr.Dropdown(
                                label="Dereverb/Deecho модель",
                                choices=self.get_list_mn_from_category(
                                    ["Реверб и эхо", "Реверб", "Эхо"], ["vr"]
                                ),
                                interactive=True,
                                filterable=False,
                            )

                        with gr.Group(visible=False) as karaoke_group:
                            karaoke_model = gr.Dropdown(
                                label="Караоке модель",
                                choices=self.get_list_mn_from_category(["Караоке"]),
                                interactive=True,
                                filterable=False,
                            )

                    separate_only_btn = gr.Button(
                        "Только разделение", variant="secondary"
                    )
                    clear_cache_btn = gr.Button(
                        "Очистить кэш разделения", variant="secondary", size="sm"
                    )
                    separation_status = gr.Textbox(
                        label="Статус разделения", interactive=False
                    )

                with gr.Tab("Настройки преобразования голоса"):
                    conversion_mode = gr.Dropdown(
                        label="Режим преобразования",
                        choices=["lead", "back", "lead/back", "full"],
                        value="full",
                        filterable=False,
                        visible=False,
                        info="lead - только основной вокал\nback - только бэк-вокал\nlead/back - основной и бэк-вокалы\nfull - весь вокал",
                    )
                    with gr.Row():
                        pitch1 = gr.Slider(
                            -48,
                            48,
                            value=0,
                            step=12,
                            label="Высота тона вокала",
                            interactive=True,
                        )
                        pitch2 = gr.Slider(
                            -48,
                            48,
                            value=0,
                            step=12,
                            label="Высота тона бэк-вокала",
                            visible=False,
                            interactive=True,
                        )
                    with gr.Row():
                        method_pitch = gr.Dropdown(
                            label="Метод извлечения тона",
                            choices=f0_methods,
                            value=f0_methods[0],
                            interactive=True,
                            filterable=False,
                        )
                        f0_max = gr.Slider(
                            50,
                            2000,
                            value=1100,
                            step=50,
                            label="Верхний лимит определения высоты тона",
                            interactive=True,
                        )
                    with gr.Row():
                        with gr.Column(scale=1):
                            index_rate = gr.Slider(
                                0,
                                1,
                                value=0,
                                step=0.05,
                                label="Влияние индекса",
                                interactive=True,
                            )
                            fr = gr.Slider(
                                0,
                                7,
                                value=3,
                                step=1,
                                label="Радиус фильтра",
                                interactive=True,
                            )
                        with gr.Column(scale=1):
                            rms = gr.Slider(
                                0,
                                1,
                                value=0.25,
                                step=0.05,
                                label="Огибающая громкости",
                                interactive=True,
                            )
                            protect = gr.Slider(
                                minimum=0,
                                maximum=0.5,
                                step=0.01,
                                value=0.33,
                                label="Защита согласных",
                                interactive=True,
                            )
                    hop_mangio_crepe = gr.Slider(
                        8,
                        512,
                        value=128,
                        step=8,
                        label="Длина шага",
                        interactive=True,
                        visible=False,
                    )
                    with gr.Accordion(label="Эмбеддер", open=False):
                        embedder_name = gr.Radio(
                            label="Модель Hubert",
                            choices=self.fairseq_embedders,
                            value=self.fairseq_embedders[0],
                        )
                        transformers_mode = gr.Checkbox(
                            label="Использовать стек Transformers",
                            value=False,
                            interactive=True,
                        )

                        @transformers_mode.change(
                            inputs=[transformers_mode], outputs=[embedder_name]
                        )
                        def change_embedders(tr_m):
                            if tr_m:
                                return gr.update(
                                    value=self.transformers_embedders[0],
                                    choices=self.transformers_embedders,
                                )
                            else:
                                return gr.update(
                                    choices=self.fairseq_embedders,
                                    value=self.fairseq_embedders[0],
                                )

                with gr.Tab("Настройки сведения аудио"):
                    gr.Markdown("<center><h2>Изменение громкости</h2></center>")
                    with gr.Row(variant="panel"):
                        vocal1_gain = gr.Slider(
                            -30,
                            30,
                            value=-3,
                            step=1,
                            label="Вокал",
                            scale=3,
                            interactive=True,
                        )
                        vocal2_gain = gr.Slider(
                            -30,
                            30,
                            value=-3,
                            step=1,
                            label="Бэк-вокал",
                            scale=3,
                            visible=False,
                            interactive=True,
                        )
                        instrumental_gain = gr.Slider(
                            -30,
                            30,
                            value=0,
                            step=1,
                            label="Инструментал",
                            scale=3,
                            interactive=True,
                        )

                    output_format = gr.Dropdown(
                        label="Формат вывода",
                        choices=output_formats,
                        value=output_formats[0],
                        interactive=True,
                        filterable=False,
                    )
                    unconv_vocals_check = gr.Checkbox(
                        label="Добавить к инструменталу непреобразованный вокал",
                        visible=False,
                    )
                    use_effects = gr.Checkbox(
                        label="Добавить эффекты на голос", value=True
                    )
                    with gr.Column(variant="panel", visible=True) as effects_accordion:
                        with gr.Tab("Эффекты"):
                            with gr.Tab("Эхо"):
                                with gr.Group():
                                    with gr.Column(variant="panel"):
                                        with gr.Row():
                                            echo_delay = gr.Slider(
                                                0,
                                                3,
                                                value=0,
                                                label="Время задержки (сек)",
                                                interactive=True,
                                            )
                                            echo_feedback = gr.Slider(
                                                0,
                                                1,
                                                value=0,
                                                label="Обратная связь",
                                                interactive=True,
                                            )
                                            echo_mix = gr.Slider(
                                                0,
                                                1,
                                                value=0,
                                                label="Смешение",
                                                interactive=True,
                                            )

                            with gr.Tab("Реверберация"):
                                with gr.Group():
                                    with gr.Column(variant="panel"):
                                        with gr.Row():
                                            reverb_rm_size = gr.Slider(
                                                0,
                                                1,
                                                value=0.1,
                                                label="Размер комнаты",
                                                interactive=True,
                                            )
                                            reverb_width = gr.Slider(
                                                0,
                                                1,
                                                value=1.0,
                                                label="Ширина реверберации",
                                                interactive=True,
                                            )
                                        with gr.Row():
                                            reverb_wet = gr.Slider(
                                                0,
                                                1,
                                                value=0.3,
                                                label="Уровень влажности",
                                                interactive=True,
                                            )
                                            reverb_dry = gr.Slider(
                                                0,
                                                1,
                                                value=0.8,
                                                label="Уровень сухости",
                                                interactive=True,
                                            )
                                        with gr.Row():
                                            reverb_damping = gr.Slider(
                                                0,
                                                1,
                                                value=0.9,
                                                label="Уровень демпфирования",
                                                interactive=True,
                                            )

                            with gr.Tab("Хорус"):
                                with gr.Group():
                                    with gr.Column(variant="panel"):
                                        with gr.Row():
                                            chorus_rate_hz = gr.Slider(
                                                0,
                                                10,
                                                value=0,
                                                label="Скорость хоруса",
                                                interactive=True,
                                            )
                                            chorus_depth = gr.Slider(
                                                0,
                                                1,
                                                value=0,
                                                label="Глубина хоруса",
                                                interactive=True,
                                            )
                                        with gr.Row():
                                            chorus_centre_delay_ms = gr.Slider(
                                                0,
                                                50,
                                                value=0,
                                                label="Задержка центра (мс)",
                                                interactive=True,
                                            )
                                            chorus_feedback = gr.Slider(
                                                0,
                                                1,
                                                value=0,
                                                label="Обратная связь",
                                                interactive=True,
                                            )
                                        with gr.Row():
                                            chorus_mix = gr.Slider(
                                                0,
                                                1,
                                                value=0,
                                                label="Смешение",
                                                interactive=True,
                                            )

                        with gr.Tab("Обработка"):
                            with gr.Tab("Компрессор"):
                                with gr.Row(variant="panel"):
                                    compressor_ratio = gr.Slider(
                                        1,
                                        20,
                                        value=16,
                                        label="Соотношение",
                                        interactive=True,
                                    )
                                    compressor_threshold = gr.Slider(
                                        -60,
                                        0,
                                        value=-16,
                                        label="Порог",
                                        interactive=True,
                                    )
                                    compressor_attack = gr.Slider(
                                        0,
                                        2000,
                                        value=40,
                                        label="Время атаки (мс)",
                                        interactive=True,
                                    )
                                    compressor_release = gr.Slider(
                                        0,
                                        2000,
                                        value=100,
                                        label="Время спада (мс)",
                                        interactive=True,
                                    )

                            with gr.Tab("Подавление шума"):
                                with gr.Group():
                                    with gr.Column(variant="panel"):
                                        with gr.Row():
                                            noise_gate_threshold = gr.Slider(
                                                -60,
                                                0,
                                                value=-40,
                                                label="Порог",
                                                interactive=True,
                                            )
                                            noise_gate_ratio = gr.Slider(
                                                1,
                                                20,
                                                value=8,
                                                label="Соотношение",
                                                interactive=True,
                                            )
                                        with gr.Row():
                                            noise_gate_attack = gr.Slider(
                                                0,
                                                100,
                                                value=10,
                                                label="Время атаки (мс)",
                                                interactive=True,
                                            )
                                            noise_gate_release = gr.Slider(
                                                0,
                                                1000,
                                                value=100,
                                                label="Время спада (мс)",
                                                interactive=True,
                                            )

            with gr.Column(variant="panel"):
                final_ai_cover = gr.Audio(
                    label="Финальный результат",
                    interactive=False,
                    show_download_button=True,
                )
                generated_files_list = gr.Files(
                    label="Промежуточные файлы", interactive=False, type="filepath"
                )

                with gr.Row(equal_height=True):
                    generate_btn = gr.Button("Сгенерировать кавер", variant="primary")
                    regenerate_btn = gr.Button(
                        "Перегенерировать вокал", variant="secondary"
                    )
                    remix_btn = gr.Button("Пересвести кавер", variant="secondary")
        status_text = gr.Textbox(label="Статус", interactive=False)

        method_pitch.change(
            fn=lambda x: gr.update(
                visible=(
                    True
                    if x in ["mangio-crepe", "mangio-crepe-tiny", "pyin"]
                    else False
                )
            ),
            inputs=method_pitch,
            outputs=hop_mangio_crepe,
        )

        @model_update_btn.click(inputs=None, outputs=model_name)
        def update_voice_models():
            models = self.parse_voice_models_actual()
            first_model = None
            if len(models) > 0:
                first_model = models[0]
            return gr.update(choices=models, value=first_model)

        @gr.on(fn="decorator", inputs=None, outputs=model_name)
        def update_voice_models():
            models = self.parse_voice_models_actual()
            first_model = None
            if len(models) > 0:
                first_model = models[0]
            return gr.update(choices=models, value=first_model)

        use_effects.change(
            fn=lambda x: gr.update(visible=x),
            inputs=use_effects,
            outputs=effects_accordion,
        )

        karaoke_check.change(
            fn=lambda x: gr.update(visible=x),
            inputs=karaoke_check,
            outputs=karaoke_group,
        ).then(
            fn=lambda x: gr.update(value="full", visible=x),
            inputs=karaoke_check,
            outputs=conversion_mode,
        ).then(
            fn=lambda x: gr.update(
                visible=True if x in ["back", "lead"] else False, value=False
            ),
            inputs=conversion_mode,
            outputs=unconv_vocals_check,
        )

        preclear_vocals_check.change(
            fn=lambda x: gr.update(visible=x),
            inputs=preclear_vocals_check,
            outputs=deecho_group,
        )

        conversion_mode.change(
            fn=lambda mode: (
                gr.update(visible=mode in ["lead", "lead/back"]),
                gr.update(visible=mode in ["back", "lead/back"]),
                gr.update(visible=mode in ["lead/back"]),
            ),
            inputs=conversion_mode,
            outputs=[vocal1_gain, vocal2_gain, pitch2],
        ).then(
            fn=lambda x: gr.update(
                visible=True if x in ["back", "lead"] else False, value=False
            ),
            inputs=conversion_mode,
            outputs=unconv_vocals_check,
        )

        separate_only_btn.click(
            fn=lambda audio, a_model, k_model, d_model, k_check, p_check: (
                (
                    self.separation_only(
                        input_audio=audio,
                        anti_instrum_model=a_model,
                        karaoke_model=k_model,
                        dereverb_model=d_model,
                        karaoke_check=k_check,
                        preclear_vocals_check=p_check,
                    )["generated_files"],
                    "Разделение завершено!",
                )
                if audio and a_model
                else (gr.update(), "Загрузите аудио и выберите модель!")
            ),
            inputs=[
                list_input_files,
                anti_instrum_model,
                karaoke_model,
                dereverb_model,
                karaoke_check,
                preclear_vocals_check,
            ],
            outputs=[generated_files_list, separation_status],
        )

        clear_cache_btn.click(
            fn=lambda: (self.clear_separation_cache(), "Кэш разделения очищен!"),
            inputs=None,
            outputs=[separation_status],
        )

        generate_btn.click(
            fn=self.gen_cover,
            inputs=[
                list_input_files,
                anti_instrum_model,
                karaoke_model,
                dereverb_model,
                output_format,
                karaoke_check,
                conversion_mode,
                preclear_vocals_check,
                model_name,
                pitch1,
                pitch2,
                method_pitch,
                index_rate,
                fr,
                rms,
                protect,
                hop_mangio_crepe,
                f0_max,
                unconv_vocals_check,
                use_effects,
                instrumental_gain,
                vocal1_gain,
                vocal2_gain,
                echo_delay,
                echo_feedback,
                echo_mix,
                reverb_rm_size,
                reverb_width,
                reverb_wet,
                reverb_dry,
                reverb_damping,
                chorus_rate_hz,
                chorus_depth,
                chorus_centre_delay_ms,
                chorus_feedback,
                chorus_mix,
                compressor_ratio,
                compressor_threshold,
                compressor_attack,
                compressor_release,
                noise_gate_threshold,
                noise_gate_ratio,
                noise_gate_attack,
                noise_gate_release,
                embedder_name,
                transformers_mode,
            ],
            outputs=[generated_files_list, final_ai_cover],
        )

        regenerate_btn.click(
            fn=self.regenerate_conversion,
            inputs=[
                model_name,
                conversion_mode,
                pitch1,
                pitch2,
                method_pitch,
                index_rate,
                fr,
                rms,
                protect,
                hop_mangio_crepe,
                f0_max,
                output_format,
                unconv_vocals_check,
                use_effects,
                instrumental_gain,
                vocal1_gain,
                vocal2_gain,
                echo_delay,
                echo_feedback,
                echo_mix,
                reverb_rm_size,
                reverb_width,
                reverb_wet,
                reverb_dry,
                reverb_damping,
                chorus_rate_hz,
                chorus_depth,
                chorus_centre_delay_ms,
                chorus_feedback,
                chorus_mix,
                compressor_ratio,
                compressor_threshold,
                compressor_attack,
                compressor_release,
                noise_gate_threshold,
                noise_gate_ratio,
                noise_gate_attack,
                noise_gate_release,
                embedder_name,
                transformers_mode,
            ],
            outputs=[generated_files_list, final_ai_cover],
        )

        remix_btn.click(
            fn=self.remix_cover,
            inputs=[
                use_effects,
                instrumental_gain,
                vocal1_gain,
                vocal2_gain,
                echo_delay,
                echo_feedback,
                echo_mix,
                reverb_rm_size,
                reverb_width,
                reverb_wet,
                reverb_dry,
                reverb_damping,
                chorus_rate_hz,
                chorus_depth,
                chorus_centre_delay_ms,
                chorus_feedback,
                chorus_mix,
                compressor_ratio,
                compressor_threshold,
                compressor_attack,
                compressor_release,
                noise_gate_threshold,
                noise_gate_ratio,
                noise_gate_attack,
                noise_gate_release,
            ],
            outputs=[final_ai_cover],
        )