import os

def rename_stems(audio, template, name_model):
    base_name = os.path.splitext(os.path.basename(audio))[0]
    stems = {
        "Bass": template.replace("NAME", base_name).replace("STEM", "bass").replace("MODEL", name_model),
        "Crowd": template.replace("NAME", base_name).replace("STEM", "crowd").replace("MODEL", name_model),
        "Drums": template.replace("NAME", base_name).replace("STEM", "drums").replace("MODEL", name_model),
        "Dry": template.replace("NAME", base_name).replace("STEM", "dry").replace("MODEL", name_model),
        "Echo": template.replace("NAME", base_name).replace("STEM", "echo").replace("MODEL", name_model),
        "Instrumental": template.replace("NAME", base_name).replace("STEM", "instrumental").replace("MODEL", name_model),
        "No Bass": template.replace("NAME", base_name).replace("STEM", "no_bass").replace("MODEL", name_model),
        "No Crowd": template.replace("NAME", base_name).replace("STEM", "no_crowd").replace("MODEL", name_model),
        "No Drums": template.replace("NAME", base_name).replace("STEM", "no_drums").replace("MODEL", name_model),
        "No Dry": template.replace("NAME", base_name).replace("STEM", "no_dry").replace("MODEL", name_model),
        "No Echo": template.replace("NAME", base_name).replace("STEM", "no_echo").replace("MODEL", name_model),
        "No Noise": template.replace("NAME", base_name).replace("STEM", "no_noise").replace("MODEL", name_model),
        "No Other": template.replace("NAME", base_name).replace("STEM", "no_other").replace("MODEL", name_model),
        "No Reverb": template.replace("NAME", base_name).replace("STEM", "no_reverb").replace("MODEL", name_model),
        "No Woodwinds": template.replace("NAME", base_name).replace("STEM", "no_wind").replace("MODEL", name_model),
        "Noise": template.replace("NAME", base_name).replace("STEM", "noise").replace("MODEL", name_model),
        "Other": template.replace("NAME", base_name).replace("STEM", "other").replace("MODEL", name_model),
        "Reverb": template.replace("NAME", base_name).replace("STEM", "reverb").replace("MODEL", name_model),
        "Vocals": template.replace("NAME", base_name).replace("STEM", "vocals").replace("MODEL", name_model),
        "Woodwinds": template.replace("NAME", base_name).replace("STEM", "wind").replace("MODEL", name_model),
        "Guitar": template.replace("NAME", base_name).replace("STEM", "guitar").replace("MODEL", name_model),
        "Piano": template.replace("NAME", base_name).replace("STEM", "piano").replace("MODEL", name_model),
    }
    return stems