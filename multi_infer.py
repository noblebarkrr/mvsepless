import os
import shutil
import argparse
import tempfile
import gradio as gr


# Импорт audio-separator для поддержки моделей на архитектуре VR ARCH и MDX-NET 

from audio_separator.separator import Separator

# Модифицированный Music-Source-Separation-Training (в виде функции)

from inference import mvsep_offline

# Импорт списка моделей

from model_list import models_data

# Загрузчик моделей для разделения аудио

from infer_utils.download_models import download_model

# Инструмент для переименования стемов в audio-separator
from infer_utils.uvr_rename_stems import rename_stems

# Редактирование конфига модели для Music-Source-Separation-Training

from infer_utils.preedit_config import conf_editor

# Удаление всех моделей в папке ckpts
def del_all_models():
    shutil.rmtree("ckpts")
    return 

# Добавление Ensembless для создания ансамблей

def blessensem():
    from ensembless import create_ensembless_interface as ensem
    from ensembless import manual_ensemble
    with gr.TabItem("Ансамбль"):
        with gr.TabItem("Aвто-ансамбль"):
            ensem()
        with gr.TabItem("Ручной ансамбль"):
            manual_ensemble()


# Функции для обновления интерфейса разделения аудио


def update_model_names(model_type):
    if model_type in models_data:
        model_names = list(models_data.get(model_type, {}).keys())
        if not model_names:
            return (
                gr.Dropdown(choices=[], value=None),  # No models available
                gr.CheckboxGroup(choices=[], value=[], interactive=False),
                gr.Checkbox(visible=False, value=False)
            )
        
        model_name = model_names[0]
        stems, ext_inst = update_stems_ui(model_type, model_name, extract_checked=False)
        
        return (
            gr.Dropdown(choices=model_names, value=model_name),  # Updated dropdown
            stems,  # Updated checkbox group
            ext_inst  # Updated extract instrumental checkbox
        )
    
    return (
        gr.Dropdown(choices=[], value=None),
        gr.CheckboxGroup(choices=[], value=[], interactive=False),
        gr.Checkbox(visible=False, value=False)
    )

def get_stems_from_model(model_type, model_name):
    if not model_type or not model_name:
        return []
    
    model_info = models_data.get(model_type, {}).get(model_name, {})
    return model_info.get("stems", [])


def update_stems_ui(model_type, model_name, extract_checked):
    stems = get_stems_from_model(model_type, model_name)
    model_info = models_data.get(model_type, {}).get(model_name, {})
    target_instrument = model_info.get("target_instrument", "No")
    
    if target_instrument != "No":
        if target_instrument == "voxes":
            return gr.CheckboxGroup(
                label="Medley Vox не поддерживает выбор стемов",
                info="Это связано с особенностями архитектуры",
                choices=stems,
                value=[],
                interactive=False
           ), gr.Checkbox(visible=False, value=False)
        return gr.CheckboxGroup(
            label=f"Целевой инструмент - {target_instrument}, для извлечения второго стема включите 'Извлечь инструментал'",
            info="Выбор стемов недоступен",
            choices=stems,
            value=[],
            interactive=False
        ), gr.Checkbox(visible=True)
    elif model_type == "vr_arch" or model_type == "mdx_net" or model_type == "medley_vox":
        return gr.CheckboxGroup(
            label="Извлечение инструментала недоступно, из-за особенностей audio-separator",
            info="Можно выбрать только один стем",
            choices=stems,
            interactive=True, 
            value=[]
        ), gr.Checkbox(visible=False, value=False)
    elif extract_checked:
        return gr.CheckboxGroup(
            label="Извлечение инструментала включено",
            info="Если из этого списка будет выбран хотя бы один стем, то появится стем 'inverted' в результатах разделения.",
            choices=stems,
            interactive=True, 
            value=[]
        ), gr.Checkbox(visible=True)
    else:
        return gr.CheckboxGroup(
            choices=stems,
            value=[],
            label="Выберите стемы",
            info="Для извлечения остатков, включите 'Извлечь инструментал'",
            interactive=bool(stems)
        ), gr.Checkbox(visible=True)

def on_extract_change(checked, model_type, model_name):
    print(f"Извлечь инструментал: {checked}")
    return update_stems_ui(model_type, model_name, checked)



# Medley-Vox (работает только через командную строку)


def medley_inference(input, output, model_dir, model_name, output_format, batch): 
    command = (
        f"python -m models.medley_vox.svs.inference "
        f"--inference_data_dir '{input}' "
        f"--results_save_dir {output} "
        f"--model_dir {model_dir} "
        f"--exp_name {model_name} "
        f"--use_overlapadd=ola "
        f"--output_format {output_format} "
        f"{'--batch' if batch else ''}"
    )
    os.system(command)
    return







# Функция для разделения аудио



def audio_separation(input_dir, output_dir="", instrum=False, model_name="", model_type="", output_format='wav', use_tta=False, batch=False, template=None, selected_instruments=[], gradio=False, progress=gr.Progress(track_tqdm=True)):
    os.makedirs(output_dir, exist_ok=True)
    
    if gradio:
        
        output_dir = tempfile.mkdtemp(prefix="mvsepless_")
        print(f"Результаты будут сохранены в {output_dir}")
        progress(0, desc="Начало разделения...")

    # Использование моделей на MDX-NET и VR ARCH в audio-separator

    if model_type == "vr_arch" or model_type == "mdx_net":

            separator = Separator(
                    output_dir=output_dir,
                    output_format=output_format,
                    output_single_stem=(selected_instruments[0] if len(selected_instruments) == 1 else None),
                    vr_params={"batch_size": 1, "window_size": 512, "aggression": 100, "enable_tta": use_tta, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False},
                    mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": True}
            )
            if gradio:
                progress(0.2, desc="Загрузка модели...")
            separator.load_model(model_filename=model_name)
            if batch:
                for filename in os.listdir(input_dir):
                    input_file = os.path.join(input_dir, filename)
                    if os.path.isfile(input_file):
                        output_names = rename_stems(input_file, template, model_name)
                        uvr_output = separator.separate(input_file, output_names)
            else:
                output_names = rename_stems(input_dir, template, model_name)
                if gradio:
                    progress(0.5, desc="Разделение аудио...")
                uvr_output = separator.separate(input_dir, output_names)


    # Использование моделей на популярных архитектурах в модифицированном Music-Source-Separation-Training


    elif model_type == "mel_band_roformer" or model_type == "bs_roformer" or model_type == "mdx23c" or model_type == "scnet" or model_type == "htdemucs":

        model_paths = "ckpts"
        config_url = models_data[model_type][model_name]["config_url"]

        checkpoint_url = models_data[model_type][model_name]["checkpoint_url"]
        if gradio:
            progress(0.2, desc="Загрузка модели")
        conf, ckpt = download_model(model_paths, model_name, model_type, checkpoint_url, config_url)
   
        print(selected_instruments)
        
        if model_type != "htdemucs":
            conf_editor(conf)
        if gradio:
            progress(0.5, desc="Разделение аудио...")
        mvsep_offline(input_dir, output_dir, model_type, conf, ckpt, instrum, output_format, model_name, template, 0, use_tta=use_tta, batch=batch, selected_instruments=selected_instruments) 

    # Разделение вокалов в Medley-Vox

    elif model_type == "medley_vox":

        model_paths = "ckpts"
        config_url = models_data[model_type][model_name]["config_url"]

        checkpoint_url = models_data[model_type][model_name]["checkpoint_url"]
        if gradio:
            progress(0.2, desc="Загрузка модели")
        medley_vox_model_dir = download_model(model_paths, model_name, model_type, checkpoint_url, config_url)
        if gradio:
            progress(0.5, desc="Разделение вокалов...")
        medley_inference(input_dir, output_dir, medley_vox_model_dir, model_name, output_format, batch)

    if gradio: # Отображение выходных файлов (упрощенная версия)
        audio_folder = output_dir
        audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith((".wav", ".mp3", ".flac"))][:20]
    
        results = []
        for i in range(20):
            visible = i < len(audio_files)
            results.append(gr.update(visible=visible, value=audio_files[i] if visible else None))
        return tuple(results)
    return



# Интерфейс для MVSEPLESS

## Кастомная тема

theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Montserrat"), "Arial", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Roboto Mono"), "Courier New", "monospace"]
).set(

    button_primary_background_fill="#3a7bd5",
    button_primary_background_fill_hover="#2c65c0",
    button_primary_text_color="#ffffff",
    input_background_fill="#ffffff",
    input_border_color="#d0d0d6",
    block_background_fill="#ffffff",
    border_color_primary="#d0d0d6"
)

def mvsepless_non_cli(vbach):
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("Разделение"):
                # State variables
                extract_state = gr.State(False)
                
                with gr.Row():
                    input_file = gr.Audio(label="Загрузить аудио", type="filepath", visible=True)

                with gr.Row():
                    model_type_dropdown = gr.Dropdown(
                        choices=list(models_data.keys()),
                        label="Тип модели",
                        value=list(models_data.keys())[0],  # Set default to first model type
                        interactive=True,
                        filterable=False
                    )
                    model_name_dropdown = gr.Dropdown(
                        label="Имя модели",
                        interactive=True,
                        filterable=False,
                        choices=list(models_data[list(models_data.keys())[0]].keys()),
                        value="unwa_instrumental_v1e"
                    )
        
                default_model_type = list(models_data.keys())[0]
                default_model_name = list(models_data[default_model_type].keys())[0]
                ext_instrum, initial_stems = get_stems_from_model(default_model_type, default_model_name)
                
                stems_checkbox = gr.CheckboxGroup(
                    label="Целевой инструмент - other, для извлечения второго стема включите 'Извлечь инструментал'",
                    choices=["vocals", "other"],
                    value=[],
                    interactive=False)
                
                extract_checkbox = gr.Checkbox(
                    label="Извлечь инструментал",
                    value=False,
                    visible=True
                )

                with gr.Row():
                    output_format = gr.Radio(
                        label="Формат результата",
                        choices=["wav", "mp3", "flac"],
                        value="flac",
                        visible=True
                    )

                separate_btn = gr.Button("Разделить", variant="primary", visible=True)

                stems = [gr.Audio(visible=(i == 0)) for i in range(20)]



                # Model type change updates both name dropdown and stems
                model_type_dropdown.change(
                    fn=update_model_names,
                    inputs=model_type_dropdown,
                    outputs=[model_name_dropdown, stems_checkbox, extract_checkbox]
                )

                # Model name change updates stems
                model_name_dropdown.change(
                    fn=lambda model_type, model_name, extract_checked: update_stems_ui(model_type, model_name, extract_checked),
                    inputs=[model_type_dropdown, model_name_dropdown, extract_checkbox],
                    outputs=[stems_checkbox, extract_checkbox]
                )

                # Extract checkbox change updates stems
                extract_checkbox.change(
                    fn=on_extract_change,
                    inputs=[extract_checkbox, model_type_dropdown, model_name_dropdown],
                    outputs=[stems_checkbox, extract_state]
                )

                # Debug handler for stems selection
                stems_checkbox.change(
                    lambda x: print(f"Выбранные стемы: {x}"),
                    inputs=stems_checkbox,
                    outputs=None
                )

            # Ensembless (Experimental)

            blessensem()

            # Vbach NON-CLI

            if vbach:
                from vbach import conversion, url_download, zip_upload, files_upload, voice_conversion
                with gr.TabItem("Преобразование"):
                    with gr.TabItem("Инференс"):
                        conversion()
                    with gr.TabItem("Загрузка моделей"):
                        url_download()                            
                        zip_upload()
                        files_upload()

            with gr.TabItem("Настройки"):
                with gr.Row():
                    clear_btn = gr.Button("Удалить все модели в MSST")
                    clear_btn.click(
                        fn=del_all_models,
                        inputs=None,
                        outputs=None
                    )

                with gr.Accordion("Формат имени результатов", open=False):
                    
                   gr.Markdown(
                       """
                       > Формат имени результатов в мульти-инференсе.

                       > Доступные ключи для формата имени стемов:
                       > (изменить формат имени стемов можно в здесь)
                       > * **NAME** - Имя входного файла
                       > * **STEM** - Название стема (например, vocals, drums, bass)
                       > * **MODEL** - Имя модели (например, bs_roformer_zf_turbo_4_stems, UVR-MDX-NET-Inst_HQ_3.onnx)

                       > Пример:
                       > * **Шаблон:** NAME_STEM_MODEL
                       > * **Результат:** test_vocals_bs_roformer_zf_turbo_4_stems

                       > Доступные ключи для формата имени преобразованного голоса:
                       > (изменить формат имени преобразованного вокала можно в вкладке Преобразование --> Инференс)
                       > * **NAME** - Имя входного файла
                       > * **DATETIME** - Дата и время создания результата (например, 20250518_155312)
                       > * **MODEL** - Имя модели (указанное в списке загруженных моделей, например, test_model)
                       > * **F0METHOD** - Метод извлечения высоты тона (например, rmvpe+, fcpe)
                       > * **PITCH** - Высоты тона (например, 0, 12, -12)

                       > Пример:
                       > * **Шаблон:** NAME_MODEL_F0METHOD_PITCH
                       > * **Результат:** test_senko_rmvpe+_12
    
                       <div style="color: red; font-weight: bold; background-color: #ffecec; padding: 10px; border-left: 3px solid red; margin: 10px 0;">

                       Используйте ТОЛЬКО указанные ключи (NAME, STEM, MODEL, DATETIME, PITCH, F0METHOD) во избежание повреждения файлов. 

                       НЕ добавляйте дополнительный текст или символы вне этих ключей, либо делайте это с осторожностью.

                       </div>
                       """
                   )


                   template_stem = gr.Text(label="Формат имени стемов", value="NAME_STEM_MODEL", visible=True)
                   # if vbach:
                       # template_voice = gr.Text(label="Формат имени преобразованного вокала", value="NAME_MODEL_F0METHOD_PITCH", visible=True)

            separate_btn.click(
                fn=audio_separation,
                inputs=[
                    input_file, gr.State("/tmp/mvsepless"), extract_state, 
                    model_name_dropdown, model_type_dropdown, 
                    output_format, gr.State(False), gr.State(False),
                    template_stem, stems_checkbox, gr.State(True)],
                outputs=[*stems]
            )


def should_show_extract(model_type, model_name):
    """Helper to determine if extract checkbox should be visible"""
    model_info = models_data.get(model_type, {}).get(model_name, {})
    target_instrument = model_info.get("target_instrument", "No")
    return target_instrument == "No" and model_type not in ["vr_arch", "mdx_net", "medley_vox"]



                    

def code_infer():
    parser = argparse.ArgumentParser(description="Мульти инференс для разделения аудио")
    parser.add_argument("-i", "--input", type=str, help="Входной файл/директория")
    parser.add_argument("-o", "--output", default="", type=str, help="Выходной файл/директория")
    parser.add_argument("-inst", "--instrum", action='store_true', help="Извлечь инструментал")
    
    parser.add_argument("-mn", "--model_name", type=str, help="Имя модели")
    parser.add_argument("-mt", "--model_type", type=str, help="Тип модели, т.е. архитектура")
    
    parser.add_argument("-of", "--output_format", type=str, choices=['mp3', 'wav', 'flac'], default='wav', help="Формат результатов")
    parser.add_argument("-tta", "--use_tta", action='store_true', help="Использование TTA")
    parser.add_argument("-b", "--batch", action='store_true', help="Пакетная обработка")
    parser.add_argument("-tmpl", "--template", type=str, default='NAME_MODEL_STEM', help="Формат имени стемов")
    parser.add_argument("--select", nargs='+', help="Выбор стемов")
    parser.add_argument("-gr", "--gradio", action='store_true', help="Запуск интерфейса")
    parser.add_argument("-grvc", "--gradiovbach", action='store_true', help="Запуск интерфейса с Vbach")
    parser.add_argument("-hface", "--hf", action='store_true', help="Для запуска на HuggingFace Spaces")

    args = parser.parse_args()

    if args.model_name and not args.model_type:
        parser.error("При указании --model_name необходимо указать и --model_arch")

    if args.gradio or args.gradiovbach:
        vbach = args.gradiovbach
        with gr.Blocks(title="Music & Voice Separation", theme=theme) as demo:
            gr.HTML("<h1><center> MVSEPLESS </center></h1>")
            mvsepless_non_cli(vbach)
        if args.hf:
            demo.queue().launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["/content"])
        else:
            demo.launch(share=True, allowed_paths=["/content"])
    else:

        audio_separation(
            input_dir=args.input,
            output_dir=args.output,
            instrum=args.instrum,
            model_name=args.model_name,
            model_type=args.model_type,
            output_format=args.output_format,
            use_tta=args.use_tta,
            batch=args.batch,
            template=args.template,
            selected_instruments=args.select
        )

if __name__ == "__main__":
    code_infer()
