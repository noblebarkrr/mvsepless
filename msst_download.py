model_code = 100


# Mel Band Roformer

if model_code == 1000: # Kimberly Jensen / Vocals
    ckpt_url = "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt?download=true"
    conf_url = "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml"

#   Unwa

elif model_code == 1001: # Unwa / Kim FT v1
    ckpt_url = "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft.ckpt?download=true"
    conf_url = "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml?download=true"
elif model_code == 1002: # Unwa / Instrumental v1e
    ckpt_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e.ckpt?download=true"
    conf_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml?download=true"
elif model_code == 1003: # Unwa / Vocals Big Beta v5e
    ckpt_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.ckpt?download=true"
    conf_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.yaml?download=true"
elif model_code == 1004: # Unwa / InstVoc Duality v1
    ckpt_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvoc_duality_v1.ckpt?download=true"
    conf_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/config_melbandroformer_instvoc_duality.yaml?download=true"
elif model_code == 1005: # Unwa / InstVoc Duality v2
    ckpt_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvox_duality_v2.ckpt?download=true"
    conf_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/config_melbandroformer_instvoc_duality.yaml?download=true"
elif model_code == 1006: # Unwa / Kim FT v2
    ckpt_url = "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2.ckpt?download=true"
    conf_url = "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml?download=true"
elif model_code == 1007: # Unwa / Kim FT v2 Bleedless
    ckpt_url = "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2_bleedless.ckpt?download=true"
    conf_url = "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml?download=true"
elif model_code == 1008: # Unwa / Big Beta v1
    ckpt_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta1.ckpt?download=true"
    conf_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/config_melbandroformer_big.yaml?download=true"
elif model_code == 1009: # Unwa / Big Beta v2
    ckpt_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta2.ckpt?download=true"
    conf_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/config_melbandroformer_big.yaml?download=true"
elif model_code == 1010: # Unwa / Big Beta v3
    ckpt_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta3.ckpt?download=true"
    conf_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/config_melbandroformer_big.yaml?download=true"
elif model_code == 1011: # Unwa / Big Beta v4
    ckpt_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta4.ckpt?download=true"
    conf_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/config_melbandroformer_big_beta4.yaml?download=true"
elif model_code == 1012: # Unwa / Big Beta v6
    ckpt_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6.ckpt?download=true"
    conf_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6.yaml?download=true"
elif model_code == 1013: # Unwa / Instrumental v1
    ckpt_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v1.ckpt?download=true"
    conf_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml?download=true"
elif model_code == 1014: # Unwa / Instrumental v2
    ckpt_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v2.ckpt?download=true"
    conf_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst_v2.yaml?download=true"
elif model_code == 1015: # Unwa / Small v1
    ckpt_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-small/resolve/main/melband_roformer_small_v1.ckpt?download=true"
    conf_url = "https://huggingface.co/pcunwa/Mel-Band-Roformer-small/resolve/main/config_melbandroformer_small.yaml?download=true"

#   Becruily

elif model_code == 1016: # Becruily / Instrumental
    ckpt_url = "https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt?download=true"
    conf_url = "https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/config_instrumental_becruily.yaml?download=true"
elif model_code == 1017: # Becruily / Vocals
    ckpt_url = "https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt?download=true"
    conf_url = "https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/config_vocals_becruily.yaml?download=true"

#   SYH99999

elif model_code == 1018: # SYH99999 / SYHFT v1
    ckpt_url = "https://huggingface.co/SYH99999/MelBandRoformerSYHFT/resolve/main/MelBandRoformerSYHFT.ckpt?download=true"
    conf_url = "https://huggingface.co/SYH99999/MelBandRoformerSYHFT/resolve/main/config_vocals_mel_band_roformer_ft.yaml?download=true"
elif model_code == 1019: # SYH99999 / SYHFT v2
    ckpt_url = "https://huggingface.co/SYH99999/MelBandRoformerSYHFTV2/resolve/main/MelBandRoformerSYHFTV2.ckpt?download=true"
    conf_url = "https://huggingface.co/SYH99999/MelBandRoformerSYHFT/resolve/main/config_vocals_mel_band_roformer_ft.yaml?download=true"
elif model_code == 1020: # SYH99999 / SYHFT v2.5
    ckpt_url = "https://huggingface.co/SYH99999/MelBandRoformerSYHFTV2.5/resolve/main/MelBandRoformerSYHFTV2.5.ckpt/MelBandRoformerSYHFTV2.5.ckpt?download=true"
    conf_url = "https://huggingface.co/SYH99999/MelBandRoformerSYHFT/resolve/main/config_vocals_mel_band_roformer_ft.yaml?download=true"
elif model_code == 1021: # SYH99999 / SYHFT v3
    ckpt_url = "https://huggingface.co/SYH99999/MelBandRoformerSYHFTV3Epsilon/resolve/main/MelBandRoformerSYHFTV3Epsilon.ckpt?download=true"
    conf_url = "https://huggingface.co/SYH99999/MelBandRoformerSYHFT/resolve/main/config_vocals_mel_band_roformer_ft.yaml?download=true"
elif model_code == 1022: # SYH99999 / Big SYHFT v1 FAST
    ckpt_url = "https://huggingface.co/SYH99999/MelBandRoformerBigSYHFTV1Fast/resolve/main/MelBandRoformerBigSYHFTV1.ckpt?download=true"
    conf_url = "https://huggingface.co/SYH99999/MelBandRoformerBigSYHFTV1Fast/resolve/main/config.yaml?download=true"
elif model_code == 1023: # SYH99999 / Merged Beta v1
    ckpt_url = "https://huggingface.co/SYH99999/MelBandRoformerMergedSYHFTBeta1/resolve/main/merge_syhft.ckpt?download=true"
    conf_url = "https://huggingface.co/SYH99999/MelBandRoformerSYHFT/resolve/main/config_vocals_mel_band_roformer_ft.yaml?download=true"
elif model_code == 1024: # SYH99999 / SYHFT B1
    ckpt_url = "https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model3.ckpt?download=true"
    conf_url = "https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml?download=true"
elif model_code == 1025: # SYH99999 / 4 stems FT Large
    ckpt_url = "https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/MelBandRoformer4StemFTLarge.ckpt?download=true"
    conf_url = "https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/config.yaml?download=true"

#   Gabox

#   Instrumental Fullness 1100-1130

elif model_code == 1100: # GaboxR67 / Instrumental Fv1
    ckpt_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv1.ckpt?download=true"
    conf_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
elif model_code == 1101: # GaboxR67 / Instrumental Fv2
    ckpt_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv2.ckpt?download=true"
    conf_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
elif model_code == 1102: # GaboxR67 / Instrumental Fv3
    ckpt_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv3.ckpt?download=true"
    conf_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
elif model_code == 1103: # GaboxR67 / Instrumental Fv4 Noise
    ckpt_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_Fv4Noise.ckpt?download=true"
    conf_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
elif model_code == 1104: # GaboxR67 / Instrumental Fv5
    ckpt_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV5.ckpt?download=true"
    conf_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
elif model_code == 1105: # GaboxR67 / Instrumental Fv5 Noise
    ckpt_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV5N.ckpt?download=true"
    conf_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
elif model_code == 1106: # GaboxR67 / Instrumental Fv6
    ckpt_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6.ckpt?download=true"
    conf_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
elif model_code == 1107: # GaboxR67 / Instrumental Fv6 Noise
    ckpt_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6N.ckpt?download=true"
    conf_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
elif model_code == 1108: # GaboxR67 / Instrumental Fv7
    ckpt_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxV7.ckpt?download=true"
    conf_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
elif model_code == 1109: # GaboxR67 / Instrumental Fv7 Noise
    ckpt_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV7N.ckpt?download=true"
    conf_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"

#   Instrumental Fullness X 1130

elif model_code == 1130: # GaboxR67 / Instrumental FvX
    ckpt_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxFVX.ckpt?download=true"
    conf_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"

#   Instrumental Bleedless 1150-1169

elif model_code == 1150: # GaboxR67 / Instrumental Bv1
    ckpt_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxBv1.ckpt?download=true"
    conf_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
elif model_code == 1151: # GaboxR67 / Instrumental Bv2
    ckpt_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxBv2.ckpt?download=true"
    conf_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"

#   Vocals Fullness 1170-1189   

elif model_code == 1170: # GaboxR67 / Vocals Fv1
    ckpt_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gaboxFv1.ckpt?download=true"
    conf_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml?download=true"
elif model_code == 1171: # GaboxR67 / Vocals Fv2
    ckpt_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gaboxFv2.ckpt?download=true"
    conf_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml?download=true"
elif model_code == 1172: # GaboxR67 / Vocals Fv3
    ckpt_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_Fv3.ckpt?download=true"
    conf_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml?download=true"
elif model_code == 1173: # GaboxR67 / Vocals Fv4
    ckpt_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_fv4.ckpt?download=true"
    conf_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml?download=true"

#   Karaoke (Experimental) 1190-1195

elif model_code == 1190: # GaboxR67 / Karaoke 25.02.2025
    ckpt_url = "https://huggingface.co/noblebarkrr/all_models_for_mel_band_roformer/resolve/main/gabox_karaoke_25_02_2025.ckpt?download=true"
    conf_url = "https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/config_mel_band_roformer_karaoke.yaml?download=true"
elif model_code == 1191: # GaboxR67 / Karaoke 28.02.2025
    ckpt_url = "https://huggingface.co/noblebarkrr/all_models_for_mel_band_roformer/resolve/main/gabox_karaoke_28_02_2025.ckpt?download=true"
    conf_url = "https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/config_mel_band_roformer_karaoke.yaml?download=true"

elif model_code == 1199: # GaboxR67 / Denoise Debleed
    ckpt_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/denoisedebleed.ckpt?download=true"
    conf_url = "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"

#   Anvuew 1030 - 1039

elif model_code == 1030: # Anvuew / Dereverb
    ckpt_url = "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt?download=true"
    conf_url = "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml?download=true"
elif model_code == 1031: # Anvuew / Dereverb Aggressive
    ckpt_url = "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt?download=true"
    conf_url = "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml?download=true"
elif model_code == 1032: # Anvuew / Dereverb Mono
    ckpt_url = "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt?download=true"
    conf_url = "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml?download=true"

#   Sucial 1040-1049

elif model_code == 1040: # Sucial / Aspiration
    ckpt_url = "https://huggingface.co/Sucial/Aspiration_Mel_Band_Roformer/resolve/main/aspiration_mel_band_roformer_less_aggr_sdr_18.1201.ckpt?download=true"
    conf_url = "https://huggingface.co/Sucial/Aspiration_Mel_Band_Roformer/resolve/main/config_aspiration_mel_band_roformer.yaml?download=true"
elif model_code == 1041: # Sucial / Dereverb Deecho
    ckpt_url = "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt?download=true"
    conf_url = "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb-echo_mel_band_roformer.yaml?download=true"
elif model_code == 1042: # Sucial / Big Dereverb
    ckpt_url = "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/de_big_reverb_mbr_ep_362.ckpt?download=true"
    conf_url = "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml?download=true"
elif model_code == 1043: # Sucial / Super Big Dereverb
    ckpt_url = "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/de_super_big_reverb_mbr_ep_346.ckpt?download=true"
    conf_url = "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml?download=true"
elif model_code == 1044: # Sucial / Dereverb Deecho MBR Fused v1
    ckpt_url = "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt?download=true"
    conf_url = "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml?download=true"
elif model_code == 1045: # Sucial / Dereverb Deecho MBR v2
    ckpt_url = "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb_echo_mbr_v2_sdr_dry_13.4843.ckpt?download=true"
    conf_url = "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml?download=true"

#   Aufr33 & ViperX 1050-1054

elif model_code == 1050: # Aufr33 & ViperX / Karaoke
    ckpt_url = "https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt?download=true"
    conf_url = "https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/config_mel_band_roformer_karaoke.yaml?download=true"
elif model_code == 1051: # Aufr33 / Denoise
    ckpt_url = "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt?download=true"
    conf_url = "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml?download=true"
elif model_code == 1052: # Aufr33 / Denoise Aggresive
    ckpt_url = "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt?download=true"
    conf_url = "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml?download=true"
elif model_code == 1053: # Aufr33 & ViperX / Decrowd
    ckpt_url = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt"
    conf_url = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/model_mel_band_roformer_crowd.yaml"
elif model_code == 1054: # ViperX / Vocals
    ckpt_url = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"
    conf_url = "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_mel_band_roformer_ep_3005_sdr_11.4360.yaml"



# BS Roformer

#   Vocals / Instrumental

elif model_code == 200: # Unwa / Instrumental / EXP Value Residual
    ckpt_url = "https://huggingface.co/pcunwa/BS-Roformer-Inst-EXP-Value-Residual/resolve/main/BS_Inst_EXP_VRL.ckpt?download=true"
    conf_url = "https://huggingface.co/pcunwa/BS-Roformer-Inst-EXP-Value-Residual/resolve/main/BS_Inst_EXP_VRL.yaml?download=true"
elif model_code == 201: # ViperX / Vocals / 1296
    ckpt_url = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt"
    conf_url = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml"
elif model_code == 202: # ViperX / Vocals / 1297
    ckpt_url = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    conf_url = "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml"
elif model_code == 203: # Gabox /Vocals
    ckpt_url = "https://huggingface.co/GaboxR67/BSRoformerVocTest/resolve/main/voc_gaboxBSR.ckpt?download=true"
    conf_url = "https://huggingface.co/GaboxR67/BSRoformerVocTest/resolve/main/voc_gaboxBSroformer.yaml?download=true"

# 4 stems

elif model_code == 204: # ZF Turbo / 4 stems
    ckpt_url = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt"
    conf_url = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/config_bs_roformer_384_8_2_485100.yaml"
elif model_code == 205: # SYH99999 / 4 stems /
    ckpt_url = "https://huggingface.co/SYH99999/bs_roformer_4stems_ft/resolve/main/bs_roformer_4stems_ft.pth?download=true"
    conf_url = "https://huggingface.co/SYH99999/bs_roformer_4stems_ft/resolve/main/config.yaml?download=true"

#   Male/Female

elif model_code == 206: # Sucial / Male-Female / 146 epochs
    ckpt_url = "https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/model_chorus_bs_roformer_ep_146_sdr_23.8613.ckpt?download=true"
    conf_url = "https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/config_chorus_male_female_bs_roformer.yaml?download=true"
elif model_code == 207: # Sucial / Male-Female / 267 epochs
    ckpt_url = "https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt?download=true"
    conf_url = "https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/config_chorus_male_female_bs_roformer.yaml?download=true"
elif model_code == 208: # Aufr33 / Male-Female
    ckpt_url = "https://huggingface.co/RareSirMix/AIModelRehosting/resolve/main/bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt"
    conf_url = "https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/config_chorus_male_female_bs_roformer.yaml"

#   Dereverb

elif model_code == 209: # Anvuew / Dereverb / 256 Dim 8 Depth
    ckpt_url = "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_256dim_8depth.ckpt?download=true"
    conf_url = "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_256dim_8depth.yaml?download=true"
elif model_code == 210: # Anvuew / Dereverb / 384 Dim 10 Depth
    ckpt_url = "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_384dim_10depth.ckpt?download=true"
    conf_url = "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_384dim_10depth.yaml?download=true"

# MDX23C

elif model_code == 300: # Inst Voc HQ
    ckpt_url = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt"
    conf_url = "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_mdx23c.yaml"
elif model_code == 301: # Aufr33 & Jarredou / Drumsep / 6 stems
    ckpt_url = "https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt"
    conf_url = "https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml"
elif model_code == 302: # Aufr33 & Jarredou / Dereverb
    ckpt_url = "https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/dereverb_mdx23c_sdr_6.9096.ckpt"
    conf_url = "https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/config_dereverb_mdx23c.yaml"
elif model_code == 303: # Wesleyr36 / Mid-Side
    ckpt_url = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.10/model_mdx23c_ep_271_l1_freq_72.2383.ckpt"
    conf_url = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.10/config_mdx23c_similarity.yaml"
elif model_code == 304: # ZF Turbo / 4 stems
    ckpt_url = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.1/model_mdx23c_ep_168_sdr_7.0207.ckpt"
    conf_url = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.1/config_musdb18_mdx23c.yaml"

# SCNET

elif model_code == 800: # ZF Turbo / 4 stems
    ckpt_url = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/SCNet-large_starrytong_fixed.ckpt"
    conf_url = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/config_musdb18_scnet_large_starrytong.yaml"
elif model_code == 801: # Starrytong / 4 stems / Large
    ckpt_url = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/model_scnet_ep_54_sdr_9.8051.ckpt"
    conf_url = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/config_musdb18_scnet_xl.yaml"
elif model_code == 802: # ZF Turbo / 4 stems / XL
    ckpt_url = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/scnet_checkpoint_musdb18.ckpt"
    conf_url = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/config_musdb18_scnet.yaml"





