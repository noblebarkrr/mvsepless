import os
def output_file_template(template, input_file_name, stem, model_name, id):
    template_name = (
        template
        .replace("NAME", f"{input_file_name}")
        .replace("MODEL", f"{model_name}")
        .replace("STEM", f"{stem}")
        .replace("ID", f"{id}")
    )
    output_name = f"{template_name}"
    return output_name

def audio_separator_rename_stems(audio, template, name_model, id):
    base_name = os.path.splitext(os.path.basename(audio))[0]
    stems = {
        "Bass": template.replace("NAME", base_name).replace("STEM", "Bass").replace("MODEL", name_model).replace("ID", f"{id}"),
        "Crowd": template.replace("NAME", base_name).replace("STEM", "Crowd").replace("MODEL", name_model).replace("ID", f"{id}"),
        "Drums": template.replace("NAME", base_name).replace("STEM", "Drums").replace("MODEL", name_model).replace("ID", f"{id}"),
        "Dry": template.replace("NAME", base_name).replace("STEM", "Dry").replace("MODEL", name_model).replace("ID", f"{id}"),
        "Breath": template.replace("NAME", base_name).replace("STEM", "Breath").replace("MODEL", name_model).replace("ID", f"{id}"),
        "Echo": template.replace("NAME", base_name).replace("STEM", "Echo").replace("MODEL", name_model).replace("ID", f"{id}"),
        "Instrumental": template.replace("NAME", base_name).replace("STEM", "Instrumental").replace("MODEL", name_model).replace("ID", f"{id}"),
        "No Bass": template.replace("NAME", base_name).replace("STEM", "No Bass").replace("MODEL", name_model).replace("ID", f"{id}"),
        "No Crowd": template.replace("NAME", base_name).replace("STEM", "No Crowd").replace("MODEL", name_model).replace("ID", f"{id}"),
        "No Drums": template.replace("NAME", base_name).replace("STEM", "No Drums").replace("MODEL", name_model).replace("ID", f"{id}"),
        "No Dry": template.replace("NAME", base_name).replace("STEM", "No Dry").replace("MODEL", name_model).replace("ID", f"{id}"),
        "No Echo": template.replace("NAME", base_name).replace("STEM", "No Echo").replace("MODEL", name_model).replace("ID", f"{id}"),
        "No Noise": template.replace("NAME", base_name).replace("STEM", "No Noise").replace("MODEL", name_model).replace("ID", f"{id}"),
        "No Other": template.replace("NAME", base_name).replace("STEM", "No Other").replace("MODEL", name_model).replace("ID", f"{id}"),
        "No Breath": template.replace("NAME", base_name).replace("STEM", "No Breath").replace("MODEL", name_model).replace("ID", f"{id}"),
        "No Reverb": template.replace("NAME", base_name).replace("STEM", "No Reverb").replace("MODEL", name_model).replace("ID", f"{id}"),
        "No Woodwinds": template.replace("NAME", base_name).replace("STEM", "No Woodwinds").replace("MODEL", name_model).replace("ID", f"{id}"),
        "Noise": template.replace("NAME", base_name).replace("STEM", "Noise").replace("MODEL", name_model).replace("ID", f"{id}"),
        "Other": template.replace("NAME", base_name).replace("STEM", "Other").replace("MODEL", name_model).replace("ID", f"{id}"),
        "Reverb": template.replace("NAME", base_name).replace("STEM", "Reverb").replace("MODEL", name_model).replace("ID", f"{id}"),
        "Vocals": template.replace("NAME", base_name).replace("STEM", "Vocals").replace("MODEL", name_model).replace("ID", f"{id}"),
        "Woodwinds": template.replace("NAME", base_name).replace("STEM", "Woodwinds").replace("MODEL", name_model).replace("ID", f"{id}"),
        "Guitar": template.replace("NAME", base_name).replace("STEM", "Guitar").replace("MODEL", name_model).replace("ID", f"{id}"),
        "Piano": template.replace("NAME", base_name).replace("STEM", "Piano").replace("MODEL", name_model).replace("ID", f"{id}")
    }
    return stems

def audio_separator_vr_rename_stems(audio, template, name_model, primary_stem, id):
    base_name = os.path.splitext(os.path.basename(audio))[0]
    stems = {
        f"{primary_stem}": template.replace("NAME", base_name).replace("STEM", primary_stem).replace("MODEL", name_model).replace("ID", f"{id}"),
        f"No {primary_stem}": template.replace("NAME", base_name).replace("STEM", f"No {primary_stem}").replace("MODEL", name_model).replace("ID", f"{id}")
    }
    return stems
