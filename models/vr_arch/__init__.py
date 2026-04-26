import os
import math
import sys
import json
import torch
import torch.nn as nn
import librosa
import numpy as np

from . import spec_utils, nets, nets_new
from .model_param_init import ModelParameters

VOCAL_STEM = "vocals"
INST_STEM = "instrumental"
OTHER_STEM = "other"
BASS_STEM = "bass"
DRUM_STEM = "drums"
GUITAR_STEM = "guitar"
PIANO_STEM = "piano"
SYNTH_STEM = "synthesizer"
STRINGS_STEM = "strings"
WOODWINDS_STEM = "woodwinds"
BRASS_STEM = "brass"
WIND_INST_STEM = "wind_inst"

NON_ACCOM_STEMS = (
    VOCAL_STEM,
    OTHER_STEM,
    BASS_STEM,
    DRUM_STEM,
    GUITAR_STEM,
    PIANO_STEM,
    SYNTH_STEM,
    STRINGS_STEM,
    WOODWINDS_STEM,
    BRASS_STEM,
    WIND_INST_STEM,
)

def get_model(config):
    model_params = ModelParameters(dict(config.model.model_params))
    nn_arch_sizes = [
        31191,
        33966,
        56817,
        123821,
        123812,
        129605,
        218409,
        537238,
        537227,
    ]
    vr_5_1_models = [56817, 218409]
    is_vr_51_model = False
    model_capacity = 32, 128
    if config.model.nout and config.model.nout_lstm:
        model_capacity = config.model.nout, config.model.nout_lstm
        is_vr_51_model = True
    if config.model.nn_arch_size in vr_5_1_models or is_vr_51_model:
        model = nets_new.CascadedNet(
            model_params.param["bins"] * 2,
            config.model.nn_arch_size,
            nout=model_capacity[0],
            nout_lstm=model_capacity[1],
        )
    else:
        model = nets.determine_model_capacity(
            model_params.param["bins"] * 2, config.model.nn_arch_size
        )
    model.model_params = model_params
    return model
