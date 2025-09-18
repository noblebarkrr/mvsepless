import os
import sys
import tempfile
import json
import gradio as gr
from multi_inference import MVSEPLESS, OUTPUT_FORMATS
from separator.ensemble import ensemble_audio_files
from utils.inverter import Inverter

inverter = Inverter()

INVERT_METHODS = {
    "min_fft": "max_fft",
    "max_fft": "min_fft",
    "min_wave": "max_wave",
    "max_wave": "min_wave",
    "median_fft": "median_fft",
    "median_wave": "median_wave",
    "avg_fft": "avg_fft",
    "avg_wave": "avg_wave"
}

class ModelParser:
    def __init__(self):
        with open("models.json", "r", encoding="utf-8") as f:
            self.models_data = json.load(f) 

    def get_mt(self):
        return list(self.models_data.keys())
       
    def get_mn(self, model_type):
        return list(self.models_data[model_type].keys())
       
    def get_stems(self, model_type, model_name):
        stems = self.models_data[model_type][model_name]["stems"]
        return stems

    def get_id(self, model_type, model_name):
        id = self.models_data[model_type][model_name]["id"]
        return id

    def get_tgt_inst(self, model_type, model_name):
        target_instrument = self.models_data[model_type][model_name]["target_instrument"]
        return target_instrument

class ENSEMBLESS:
    def __init__(self):
        self.mvsepless = MVSEPLESS()
        self.mp = ModelParser()

    def get_model_types(self):
        return self.mp.get_mt()
    
    def get_models_by_type(self, model_type):
        return self.mp.get_mn(model_type)
    
    def get_stems_by_model(self, model_type, model_name):
        all_stems = []
        stems = self.mp.get_stems(model_type, model_name)
        for stem in stems:
            all_stems.append(stem)
        if set(stems) == {"bass", "drums", "vocals", "other"} or set(stems) == {"bass", "drums", "vocals", "other", "piano", "guitar"} and not self.mp.get_tgt_inst(model_type, model_name):
            all_stems.append("instrumental +")
            all_stems.append("instrumental -")
        return all_stems
        
    def get_invert_stems_by_model(self, model_type, model_name, primary_stem):
        invert_stems = []
        stems = self.mp.get_stems(model_type, model_name)
        for stem in stems:
            if stem != primary_stem:
                invert_stems.append(stem)
          
        if not self.mp.get_tgt_inst(model_type, model_name) and model_type not in ["vr", "mdx"] and primary_stem not in ["instrumental -", "instrumental +"]:
        
            invert_stems.append("inverted +")
            invert_stems.append("inverted -")
            
        return invert_stems   
        
    def invert_weights(self, weights):
        total_weight = sum(weights)
        return [total_weight - w for w in weights]

    def manual_ensemble(self, input_audios, method, weights, out_format):
        temp_dir = tempfile.mkdtemp()
        weights = [float(x) for x in weights.split(",")]
        # padded_files = self.maximize_length_audio(input_audios)
        a1, a2 = ensemble_audio_files(input_audios, output=os.path.join(temp_dir, f"ensemble_{method}"), ensemble_type=method, weights=weights, out_format=out_format)
        return a1, a2

    def auto_ensemble(self, input_audio, input_settings, type, out_format, invert_weights, invert_ensemble):

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
            output_p = self.mvsepless.separator(input_file=input_audio, output_dir=output_dir_p, model_type=model_type, model_name=model_name, ext_inst=True, vr_aggr=10, output_format="wav", template="MODEL_STEM", call_method="cli")           
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
                
                    output_dir_s = os.path.join(temp_dir, f"{model_type}_{model_name}_s_stems")
                    output_s = self.mvsepless.separator(input_file=input_audio, output_dir=output_dir_s, model_type=model_type, model_name=model_name, ext_inst=True, vr_aggr=10, output_format="wav", template="MODEL_STEM", call_method="cli", selected_stems=[p_stem if not mvsepless.get_tgt_inst(model_type, model_name) else "both"])
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
            output_s, output_wav_s = ensemble_audio_files(files=output_s_files, output=os.path.join(temp_dir, f"ensemble_invert_{base_name}_{type}"), ensemble_type=INVERT_METHODS[type], weights=output_s_weights, out_format=out_format)
        else:
            output_s, output_wav_s = None, None
            
        output_p, output_wav_p = ensemble_audio_files(files=output_p_files, output=os.path.join(temp_dir, f"ensemble_{base_name}_{type}"), ensemble_type=type, weights=output_p_weights, out_format=out_format)
            
        return output_p, output_wav_p, output_s, output_wav_s, source_files
