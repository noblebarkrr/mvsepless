import os
import math
import sys
import json
import torch
import torch.nn as nn
import librosa
import numpy as np
from tqdm import tqdm

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


class VRNet:
    def __init__(
        self,
        model_params={},
        nout=None,
        nout_lstm=None,
    ):
        self.torch_device_mps = torch.backends.mps.is_available()
        self.enable_post_process = False
        self.post_process_threshold = 0.2
        self.batch_size = 1
        self.window_size = 512
        self.high_end_process = False
        self.primary_stem = "Instrumental"
        self.secondary_stem = "Vocals"
        self.model_capacity = 32, 128
        self.is_vr_51_model = False
        if nout and nout_lstm:
            self.model_capacity = nout, nout_lstm
            self.is_vr_51_model = True
        self.model_params = ModelParameters(model_params)
        self.input_high_end_h = None
        self.input_high_end = None
        self.enable_tta = False
        self.model_samplerate = self.model_params.param["sr"]
        self.model_run = lambda *args, **kwargs: print(
            "Model run method is not initialised yet."
        )

    def load_checkpoint(self, checkpoint_path: str, device: torch.device):
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
        model_size = math.ceil(os.stat(checkpoint_path).st_size / 1024)
        nn_arch_size = min(nn_arch_sizes, key=lambda x: abs(x - model_size))

        if nn_arch_size in vr_5_1_models or self.is_vr_51_model:
            self.model_run = nets_new.CascadedNet(
                self.model_params.param["bins"] * 2,
                nn_arch_size,
                nout=self.model_capacity[0],
                nout_lstm=self.model_capacity[1],
            )
            self.is_vr_51_model = True
        else:
            self.model_run = nets.determine_model_capacity(
                self.model_params.param["bins"] * 2, nn_arch_size
            )

        self.model_run.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model_run.to(device)

    def loading_mix(self, numpy_array, orig_sr=44100):
        X_wave, X_spec_s = {}, {}

        bands_n = len(self.model_params.param["band"])

        audio_file = numpy_array

        for d in tqdm(range(bands_n, 0, -1)):
            bp = self.model_params.param["band"][d]

            wav_resolution = bp["res_type"]

            if self.torch_device_mps:
                wav_resolution = "polyphase"

            if d == bands_n:
                X_wave[d], _ = librosa.resample(
                    y=numpy_array,
                    orig_sr=orig_sr,
                    target_sr=bp["sr"],
                    res_type=wav_resolution,
                )
                X_spec_s[d] = spec_utils.wave_to_spectrogram(
                    X_wave[d],
                    bp["hl"],
                    bp["n_fft"],
                    self.model_params,
                    band=d,
                    is_v51_model=self.is_vr_51_model,
                )

                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
            else:
                X_wave[d] = librosa.resample(
                    X_wave[d + 1],
                    orig_sr=self.model_params.param["band"][d + 1]["sr"],
                    target_sr=bp["sr"],
                    res_type=wav_resolution,
                )
                X_spec_s[d] = spec_utils.wave_to_spectrogram(
                    X_wave[d],
                    bp["hl"],
                    bp["n_fft"],
                    self.model_params,
                    band=d,
                    is_v51_model=self.is_vr_51_model,
                )

            if d == bands_n and self.high_end_process:
                self.input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (
                    self.model_params.param["pre_filter_stop"]
                    - self.model_params.param["pre_filter_start"]
                )
                self.input_high_end = X_spec_s[d][
                    :, bp["n_fft"] // 2 - self.input_high_end_h : bp["n_fft"] // 2, :
                ]

        X_spec = spec_utils.combine_spectrograms(
            X_spec_s, self.model_params, is_v51_model=self.is_vr_51_model
        )

        del X_wave, X_spec_s

        return X_spec

    def inference_vr(self, X_spec, device, aggressiveness):
        def _execute(X_mag_pad, roi_size):
            X_dataset = []
            patches = (X_mag_pad.shape[2] - 2 * self.model_run.offset) // roi_size
            total = patches
            for i in tqdm(range(patches)):
                processed = min(i + self.batch_size, patches)
                sys.stdout.write(
                    json.dumps(
                        {"processing": {"processed": processed, "total": total}},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                start = i * roi_size
                X_mag_window = X_mag_pad[:, :, start : start + self.window_size]
                X_dataset.append(X_mag_window)

            total_iterations = (
                patches // self.batch_size
                if not self.enable_tta
                else (patches // self.batch_size) * 2
            )

            X_dataset = np.asarray(X_dataset)
            self.model_run.eval()
            with torch.no_grad():
                mask = []

                for i in tqdm(range(0, patches, self.batch_size)):
                    processed = min(i + self.batch_size, patches)
                    sys.stdout.write(
                        json.dumps(
                            {"processing": {"processed": processed, "total": total}},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    X_batch = X_dataset[i : i + self.batch_size]
                    X_batch = torch.from_numpy(X_batch).to(device)
                    pred = self.model_run.predict_mask(X_batch)
                    if not pred.size()[3] > 0:
                        raise ValueError(
                            f"Window size error: h1_shape[3] must be greater than h2_shape[3]"
                        )
                    pred = pred.detach().cpu().numpy()
                    pred = np.concatenate(pred, axis=2)
                    mask.append(pred)
                if len(mask) == 0:
                    raise ValueError(
                        f"Window size error: h1_shape[3] must be greater than h2_shape[3]"
                    )

                mask = np.concatenate(mask, axis=2)
            return mask

        def postprocess(mask, X_mag, X_phase):
            is_non_accom_stem = False
            for stem in NON_ACCOM_STEMS:
                if stem == self.primary_stem:
                    is_non_accom_stem = True

            mask = spec_utils.adjust_aggr(mask, is_non_accom_stem, aggressiveness)

            if self.enable_post_process:
                mask = spec_utils.merge_artifacts(
                    mask, thres=self.post_process_threshold
                )

            y_spec = mask * X_mag * np.exp(1.0j * X_phase)
            v_spec = (1 - mask) * X_mag * np.exp(1.0j * X_phase)

            return y_spec, v_spec

        X_mag, X_phase = spec_utils.preprocess(X_spec)
        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = spec_utils.make_padding(
            n_frame, self.window_size, self.model_run.offset
        )
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")
        X_mag_pad /= X_mag_pad.max()
        mask = _execute(X_mag_pad, roi_size)

        if self.enable_tta:
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")
            X_mag_pad /= X_mag_pad.max()
            mask_tta = _execute(X_mag_pad, roi_size)
            mask_tta = mask_tta[:, :, roi_size // 2 :]
            mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5
        else:
            mask = mask[:, :, :n_frame]

        y_spec, v_spec = postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec

    def spec_to_wav(self, spec):
        if (
            self.high_end_process
            and isinstance(self.input_high_end, np.ndarray)
            and self.input_high_end_h
        ):
            input_high_end_ = spec_utils.mirroring(
                "mirroring", spec, self.input_high_end, self.model_params
            )
            wav = spec_utils.cmb_spectrogram_to_wave(
                spec,
                self.model_params,
                self.input_high_end_h,
                input_high_end_,
                is_v51_model=self.is_vr_51_model,
            )
        else:
            wav = spec_utils.cmb_spectrogram_to_wave(
                spec, self.model_params, is_v51_model=self.is_vr_51_model
            )

        return wav

    def settings(
        self,
        enable_post_process=False,
        post_process_threshold=0.2,
        batch_size=1,
        window_size=512,
        high_end_process=False,
        primary_stem="Instrumental",
        secondary_stem="Vocals",
    ):
        self.enable_post_process = enable_post_process
        self.post_process_threshold = post_process_threshold
        self.batch_size = batch_size
        self.window_size = window_size
        self.high_end_process = high_end_process
        self.primary_stem = primary_stem
        self.secondary_stem = secondary_stem

    def demix(self, numpy_array, sr, device, aggression):
        aggr = float(int(aggression) / 100)
        aggressiveness = {
            "value": aggr,
            "split_bin": self.model_params.param["band"][1]["crop_stop"],
            "aggr_correction": self.model_params.param.get("aggr_correction"),
        }
        y_spec, v_spec = self.inference_vr(
            self.loading_mix(numpy_array, sr), device, aggressiveness
        )
        y_spec = np.nan_to_num(y_spec, nan=0.0, posinf=0.0, neginf=0.0)
        v_spec = np.nan_to_num(v_spec, nan=0.0, posinf=0.0, neginf=0.0)
        primary_stem_array = self.spec_to_wav(y_spec).T
        primary_stem_array = librosa.resample(
            primary_stem_array.T,
            orig_sr=self.model_samplerate,
            target_sr=44100,
        ).T
        secondary_stem_array = self.spec_to_wav(v_spec).T
        secondary_stem_array = librosa.resample(
            secondary_stem_array.T,
            orig_sr=self.model_samplerate,
            target_sr=44100,
        ).T
        return {
            self.primary_stem: primary_stem_array,
            self.secondary_stem: secondary_stem_array,
        }
