from model_list import models_data

def model_info(model_type, model_name=None):
    if model_name is not None:
        model_info = models_data[model_type][model_name]
        print("Full name: ",model_info["full_name"])
        print("Stems: ",model_info["stems"])
        print("Target instrumental: ",model_info["target_instrument"])
        print("Info: ",model_info["information"])
    else:
        for model in models_data[model_type]:
            if model_type == "vr_arch" or model_type == "mdx_net":
                print("")
                print("Model :", model)
            else:
                print("")
                print("Model :", model, "    Target stem :", models_data[model_type][model]["target_instrument"])
