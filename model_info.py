from model_list import models_data

def model_info(model_type, model_name=None):
    if model_name is not None:
        model_info = models_data[model_type][model_name]
        print("Stems: ",model_info["stems"])
        print("Target instrumental: ",model_info["target_instrument"])
    else:
        for model in models_data[model_type]: 
            print("")
            print("Model :", model)
            print("Stems :", models_data[model_type][model]["stems"])
            print("Target stem :", models_data[model_type][model]["target_instrument"])
