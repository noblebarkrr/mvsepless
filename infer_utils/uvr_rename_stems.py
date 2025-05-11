import os

def rename_stems(audio, template, name_model):
    base_name = os.path.splitext(os.path.basename(audio))[0]
    stems = {
        "Bass": template.replace("NAME", base_name).replace("STEM", "Bass").replace("MODEL", name_model),
        "Crowd": template.replace("NAME", base_name).replace("STEM", "Crowd").replace("MODEL", name_model),
        "Drums": template.replace("NAME", base_name).replace("STEM", "Drums").replace("MODEL", name_model),
        "Dry": template.replace("NAME", base_name).replace("STEM", "Dry").replace("MODEL", name_model),
        "Echo": template.replace("NAME", base_name).replace("STEM", "Echo").replace("MODEL", name_model),
        "Instrumental": template.replace("NAME", base_name).replace("STEM", "Instrumental").replace("MODEL", name_model),
        "No Bass": template.replace("NAME", base_name).replace("STEM", "No Bass").replace("MODEL", name_model),
        "No Crowd": template.replace("NAME", base_name).replace("STEM", "No Crowd").replace("MODEL", name_model),
        "No Drums": template.replace("NAME", base_name).replace("STEM", "No Drums").replace("MODEL", name_model),
        "No Dry": template.replace("NAME", base_name).replace("STEM", "No Dry").replace("MODEL", name_model),
        "No Echo": template.replace("NAME", base_name).replace("STEM", "No Echo").replace("MODEL", name_model),
        "No Noise": template.replace("NAME", base_name).replace("STEM", "No Noise").replace("MODEL", name_model),
        "No Other": template.replace("NAME", base_name).replace("STEM", "No Other").replace("MODEL", name_model),
        "No Reverb": template.replace("NAME", base_name).replace("STEM", "No Reverb").replace("MODEL", name_model),
        "No Woodwinds": template.replace("NAME", base_name).replace("STEM", "No Woodwinds").replace("MODEL", name_model),
        "Noise": template.replace("NAME", base_name).replace("STEM", "Noise").replace("MODEL", name_model),
        "Other": template.replace("NAME", base_name).replace("STEM", "Other").replace("MODEL", name_model),
        "Reverb": template.replace("NAME", base_name).replace("STEM", "Reverb").replace("MODEL", name_model),
        "Vocals": template.replace("NAME", base_name).replace("STEM", "Vocals").replace("MODEL", name_model),
        "Woodwinds": template.replace("NAME", base_name).replace("STEM", "Woodwinds").replace("MODEL", name_model),
        "Guitar": template.replace("NAME", base_name).replace("STEM", "Guitar").replace("MODEL", name_model),
        "Piano": template.replace("NAME", base_name).replace("STEM", "Piano").replace("MODEL", name_model),
    }
    return stems
