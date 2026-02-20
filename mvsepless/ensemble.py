from audio import output_formats, check, write, multiread, ensemble
from namer import Namer
from gradio_helper import tz
from separator import Separator, script_dir
import time
import argparse
from datetime import datetime
import gradio as gr
import os, json

namer = Namer()

def ensemble_audio_files(
    files: list,
    weights: list,
    output_name: str,
    ensemble_type: str,
    out_format="mp3",
    add_wav=False
):
    arrays, srs = multiread(files)
    results, max_sr = ensemble(arrays, srs, weights, ensemble_type)
    
    if add_wav:
        print(f"Запись в файлы: {output_name}.{out_format} и {output_name}_orig.wav")
        return write(f"{output_name}.{out_format}", results, max_sr), write(f"{output_name}_orig.wav", results, max_sr)
    else:
        print(f"Запись в файл: {output_name}.{out_format}")
        return write(f"{output_name}.{out_format}", results, max_sr)

ensemble_methods = ("min_fft", "max_fft", "avg_fft", "median_fft")

ensemble_invert_methods_map = {
    "min_fft": "max_fft",
    "max_fft": "min_fft",
    "avg_fft": "avg_fft",
    "median_fft": "median_fft",
}

def auto_ensemble_run(
    input_file,
    ensemble_state: list[list[str, str, str, int]],
    output_dir,
    method,
    out_format,
    invert_ensemble,
    progress=gr.Progress(track_tqdm=True),
):
    separator = Separator()
    ensemble_state = ensemble_state
    invert_methods_map = ensemble_invert_methods_map
    if not input_file:
        return None, None, None, []
    if not os.path.exists(input_file):
        return None, None, None, []
    if not check(input_file):
        return None, None, None, []
    
    o = output_dir
    os.makedirs(o, exist_ok=True)

    basename = os.path.splitext(os.path.basename(input_file))[0]

    def invert_weights(weights):
        total_weight = sum(weights)
        return [total_weight - w for w in weights]

    success_separations = []
    ensemble_sources_list = []
    if ensemble_state:
        total_ensemble_models = len(ensemble_state)
        for i, model in enumerate(ensemble_state, start=1):

            ens_mn = model[0]
            ens_s_stem = model[1]
            ens_i_stem = model[2]
            weight = model[3]

            s_stem = None
            i_stem = None

            try:
                result_seped_auto_ensemble = separator.separate(
                    input=input_file,
                    output_dir=os.path.join(o, ens_mn),
                    model_name=ens_mn,
                    ext_inst=True,
                    template="NAME - MODEL - STEM",
                    output_format="wav",
                    add_settings={
                        "add_single_sep_text_progress": f"{i} из {total_ensemble_models}"
                    },
                    progress=progress,
                )
                if result_seped_auto_ensemble:
                    for stem, path in result_seped_auto_ensemble:
                        ensemble_sources_list.append(path)
                        if stem == ens_s_stem:
                            s_stem = path
                        elif stem == ens_i_stem:
                            i_stem = path

                if invert_ensemble:
                    if not i_stem:
                        result_seped_auto_ensemble_invert = separator.separate(
                            input=input_file,
                            output_dir=os.path.join(o, f"{ens_mn}_invert"),
                            model_name=ens_mn,
                            ext_inst=True,
                            template="NAME - MODEL - STEM",
                            output_format="wav",
                            selected_stems=[ens_s_stem],
                            add_settings={
                                "add_single_sep_text_progress": f"{i} из {total_ensemble_models} (инверт.)"
                            },
                            progress=progress,
                        )
                        if result_seped_auto_ensemble_invert:
                            for stem, path in result_seped_auto_ensemble_invert:
                                if stem == ens_i_stem:
                                    i_stem = path
                                    ensemble_sources_list.append(path)

            except Exception as e:
                print(f"\nПроизошла ошибка при разделении: {e}")
                progress(
                    0,
                    desc="Произошла ошибка при разделении, модель пропускается...",
                )
                continue
            finally:
                if s_stem:
                    success_separations.append((ens_mn, s_stem, i_stem, weight))

    ensemble_sources_stems = []
    ensemble_sources_invert_stems = []
    weights = []

    for out_mn, out_s_stem, out_i_stem, out_weight in success_separations:
        ensemble_sources_stems.append(out_s_stem)
        ensemble_sources_invert_stems.append(out_i_stem)
        weights.append(out_weight)

    auto_ensemble_invout_file = None
    auto_ensemble_invout_file_wav = None

    if not ensemble_sources_stems:
        return None, None, None, []
    auto_ensemble_output_name = f"ensembless_{namer.short(basename, length=50)}_{len(ensemble_sources_stems)}_{method}"
    auto_ensemble_inverted_output_name = f"ensembless_{namer.short(basename, length=50)}_{len(ensemble_sources_stems)}_{invert_methods_map[method]}_invert"
    auto_ensemble_out_file, auto_ensemble_out_file_wav = ensemble_audio_files(
        files=ensemble_sources_stems,
        weights=weights,
        output_name=os.path.join(o, auto_ensemble_output_name),
        ensemble_type=method,
        out_format=out_format,
        add_wav=True,
    )

    if invert_ensemble:
        if ensemble_sources_invert_stems:
            auto_ensemble_invout_file, auto_ensemble_invout_file_wav = (
                ensemble_audio_files(
                    files=ensemble_sources_invert_stems,
                    weights=invert_weights(weights),
                    output_name=os.path.join(o, auto_ensemble_inverted_output_name),
                    ensemble_type=invert_methods_map[method],
                    out_format=out_format,
                    add_wav=True,
                )
            )
    return (
        auto_ensemble_out_file,
        auto_ensemble_out_file_wav,
        auto_ensemble_invout_file,
        ensemble_sources_list,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Программа для работы с ансамблями")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Режим работы")

    # --- Команда AUTO (разделение + ансамбль) ---
    auto_parser = subparsers.add_parser("auto", help="Автоматическое разделение и сборка ансамбля")
    auto_parser.add_argument("--input_file", type=str, required=True, help="Путь к входному аудио")
    auto_parser.add_argument("--output_dir", type=str, default="results", help="Папка для сохранения")
    auto_parser.add_argument("--method", type=str, default="avg_fft", choices=ensemble_methods)
    auto_parser.add_argument("--output_format", type=str, default="wav", choices=output_formats)
    auto_parser.add_argument("--invert", action="store_true", help="Включить инверсию")
    
    # Модели для auto (либо JSON, либо список)
    auto_group = auto_parser.add_mutually_exclusive_group(required=True)
    auto_group.add_argument('--model_list', nargs='+', metavar='MODEL,PRIMARY_STEM,SECONDARY_STEM,WEIGHT', help="Список моделей через запятую")
    auto_group.add_argument("--json", type=str, help="Путь к JSON-файлу")

    # --- Команда MANUAL (только ансамбль готовых файлов) ---
    manual_parser = subparsers.add_parser("manual", help="Сборка ансамбля из готовых файлов")
    manual_parser.add_argument("--input_files", nargs='+', required=True, help="Список путей к файлам")
    manual_parser.add_argument("--weights", nargs='+', type=float, help="Веса для каждого файла (по умолчанию 1.0)")
    manual_parser.add_argument("--output_name", type=str, required=True, help="Имя выходного файла (без расширения)")
    manual_parser.add_argument("--method", type=str, default="avg_fft", choices=ensemble_methods)
    manual_parser.add_argument("--output_format", type=str, default="wav", choices=output_formats)

    args = parser.parse_args()

    if args.command == "auto":
        ensemble_state = []
        errors = []

        if args.json:
            with open(args.json, "r", encoding="utf-8") as f:
                exported_model_list = json.load(f)
            for i, item in enumerate(exported_model_list, start=1):
                if isinstance(item, list) and len(item) == 4:
                    item[3] = float(item[3])
                    ensemble_state.append(item)
                else:
                    errors.append(f"  #{i} - Неверный формат JSON")

        elif args.model_list:
            for i, item in enumerate(args.model_list, start=1):
                parts = item.split(',')
                if len(parts) == 4:
                    try:
                        parts[3] = float(parts[3])
                        ensemble_state.append(parts)
                    except ValueError:
                        errors.append(f"  #{i} - Вес должен быть числом: {item}")
                else:
                    errors.append(f"  #{i} - Нужно 4 значения: {item}")

        if errors:
            raise ValueError("Ошибки в описании моделей:\n" + "\n".join(errors))

        auto_ensemble_run(
            input_file=args.input_file,
            ensemble_state=ensemble_state,
            output_dir=args.output_dir,
            method=args.method,
            out_format=args.output_format,
            invert_ensemble=args.invert
        )

    elif args.command == "manual":
        weights: list = args.weights if args.weights else [1.0] * len(args.input_files)
        total_files, total_weights = len(args.input_files), len(weights)
        if total_weights < total_files:
            weights = weights + ([1.0] * (total_files - total_weights))
        elif total_weights > total_files:
            weights = weights[:total_files]

        print(f"Запуск ручного ансамбля ({args.method})...")
        ensemble_audio_files(
            files=args.input_files,
            weights=weights,
            output_name=args.output_name,
            ensemble_type=args.method,
            out_format=args.output_format
        )


    
        
