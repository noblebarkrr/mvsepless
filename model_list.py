models_data = {

    # Music-Source-Separation-Training models

    "mel_band_roformer": {
        "kimberlyjsn_vocals": {
            "full_name": "Mel-Band Roformer Vocals by KimberleyJSN",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "First open-source model for Mel-Band Roformer",
            "checkpoint_url": "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt?download=true",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml"
        },
        "unwa_kim_ft_v1": {
            "full_name": "Mel-Band Roformer Kim FT v1 by Unwa",
            "category": "Vocals",
            "stems": ["Vocals", "other"],
            "target_instrument": "Vocals",
            "information": "Fine-tuned version of KimberleyJSN's model",
            "checkpoint_url": "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml?download=true"
        },
        "unwa_instrumental_v1e": {
            "full_name": "Mel-Band Roformer Instrumental v1e by Unwa",
            "category": "Instrumental",
            "stems": ["vocals", "other"],
            "target_instrument": "other",
            "information": "Instrumental separation model",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml?download=true"
        },
        "unwa_vocals_big_beta_v5e": {
            "full_name": "Mel-Band Roformer Vocals Big Beta v5e by Unwa",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Large-scale vocal separation model",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.yaml?download=true"
        },
        "unwa_instvoc_duality_v1": {
            "full_name": "Mel-Band Roformer InstVoc Duality v1 by Unwa",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No",
            "information": "Duality model for vocal and instrumental separation",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvoc_duality_v1.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/config_melbandroformer_instvoc_duality.yaml?download=true"
        },
        "unwa_instvoc_duality_v2": {
            "full_name": "Mel-Band Roformer InstVoc Duality v2 by Unwa",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No",
            "information": "Improved duality model for vocal and instrumental separation",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvox_duality_v2.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/config_melbandroformer_instvoc_duality.yaml?download=true"
        },
        "unwa_kim_ft_v2": {
            "full_name": "Mel-Band Roformer Kim FT v2 by Unwa",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Second version of fine-tuned KimberleyJSN's model",
            "checkpoint_url": "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml?download=true"
        },
        "unwa_kim_ft_v2_bleedless": {
            "full_name": "Mel-Band Roformer Kim FT v2 Bleedless by Unwa",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Bleedless version of fine-tuned KimberleyJSN's model",
            "checkpoint_url": "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2_bleedless.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml?download=true"
        },
        "unwa_big_beta_v1": {
            "full_name": "Mel-Band Roformer Big Beta v1 by Unwa",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "First version of large-scale vocal separation model",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta1.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/config_melbandroformer_big.yaml?download=true"
        },
        "unwa_big_beta_v2": {
            "full_name": "Mel-Band Roformer Big Beta v2 by Unwa",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Second version of large-scale vocal separation model",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta2.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/config_melbandroformer_big.yaml?download=true"
        },
        "unwa_big_beta_v3": {
            "full_name": "Mel-Band Roformer Big Beta v3 by Unwa",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Third version of large-scale vocal separation model",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta3.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/config_melbandroformer_big.yaml?download=true"
        },
        "unwa_big_beta_v4": {
            "full_name": "Mel-Band Roformer Big Beta v4 by Unwa",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Fourth version of large-scale vocal separation model",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta4.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/config_melbandroformer_big_beta4.yaml?download=true"
        },
        "unwa_big_beta_v6": {
            "full_name": "Mel-Band Roformer Big Beta v6 by Unwa",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Sixth version of large-scale vocal separation model",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6.yaml?download=true"
        },
        "unwa_instrumental_v1": {
            "full_name": "Mel-Band Roformer Instrumental v1 by Unwa",
            "category": "Instrumental",
            "stems": ["vocals", "other"],
            "target_instrument": "other",
            "information": "First version of instrumental separation model",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v1.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml?download=true"
        },
        "unwa_instrumental_v2": {
            "full_name": "Mel-Band Roformer Instrumental v2 by Unwa",
            "category": "Instrumental",
            "stems": ["vocals", "other"],
            "target_instrument": "other",
            "information": "Second version of instrumental separation model",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v2.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst_v2.yaml?download=true"
        },
        "unwa_small_v1": {
            "full_name": "Mel-Band Roformer Small v1 by Unwa",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Small version of vocal separation model",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-small/resolve/main/melband_roformer_small_v1.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-small/resolve/main/config_melbandroformer_small.yaml?download=true"
        },
        "becruily_instrumental": {
            "full_name": "Mel-Band Roformer Instrumental by Becruily",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "information": "Instrumental separation model by Becruily",
            "checkpoint_url": "https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt?download=true",
            "config_url": "https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/config_instrumental_becruily.yaml?download=true"
        },
        "becruily_karaoke": {
            "full_name": "Mel-Band Roformer Karaoke by Becruily",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No",
            "information": "Karaoke separation model by Becruily",
            "checkpoint_url": "https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/mel_band_roformer_karaoke_becruily.ckpt?download=true",
            "config_url": "https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/config_karaoke_becruily.yaml?download=true"
        },
        "becruily_vocals": {
            "full_name": "Mel-Band Roformer Vocals by Becruily",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Vocal separation model by Becruily",
            "checkpoint_url": "https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt?download=true",
            "config_url": "https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/config_vocals_becruily.yaml?download=true"
        },
        "syh99999_syhft_v1": {
            "full_name": "Mel-Band Roformer SYHFT v1 by SYH99999",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "First version of fine-tuned vocal separation model by SYH99999",
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFT/resolve/main/MelBandRoformerSYHFT.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFT/resolve/main/config_vocals_mel_band_roformer_ft.yaml?download=true"
        },
        "syh99999_syhft_v2": {
            "full_name": "Mel-Band Roformer SYHFT v2 by SYH99999",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Second version of fine-tuned vocal separation model by SYH99999",
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFTV2/resolve/main/MelBandRoformerSYHFTV2.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFT/resolve/main/config_vocals_mel_band_roformer_ft.yaml?download=true"
        },
        "syh99999_syhft_v2.5": {
            "full_name": "Mel-Band Roformer SYHFT v2.5 by SYH99999",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Intermediate version of fine-tuned vocal separation model by SYH99999",
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFTV2.5/resolve/main/MelBandRoformerSYHFTV2.5.ckpt/MelBandRoformerSYHFTV2.5.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFT/resolve/main/config_vocals_mel_band_roformer_ft.yaml?download=true"
        },
        "syh99999_syhft_v3": {
            "full_name": "Mel-Band Roformer SYHFT v3 by SYH99999",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Third version of fine-tuned vocal separation model by SYH99999",
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFTV3Epsilon/resolve/main/MelBandRoformerSYHFTV3Epsilon.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFT/resolve/main/config_vocals_mel_band_roformer_ft.yaml?download=true"
        },
        "syh99999_big_syhft_v1_fast": {
            "full_name": "Mel-Band Roformer Big SYHFT v1 Fast by SYH99999",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Fast version of large-scale fine-tuned vocal separation model by SYH99999",
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformerBigSYHFTV1Fast/resolve/main/MelBandRoformerBigSYHFTV1.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformerBigSYHFTV1Fast/resolve/main/config.yaml?download=true"
        },
        "syh99999_merged_beta_v1": {
            "full_name": "Mel-Band Roformer Merged Beta v1 by SYH99999",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Merged version of fine-tuned vocal separation models by SYH99999",
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformerMergedSYHFTBeta1/resolve/main/merge_syhft.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFT/resolve/main/config_vocals_mel_band_roformer_ft.yaml?download=true"
        },
        "syh99999_syhft_b1": {
            "full_name": "Mel-Band Roformer SYHFT B1 by SYH99999",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Beta version of fine-tuned vocal separation model by SYH99999",
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model3.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml?download=true"
        },
        "syh99999_4_stems_ft_large": {
            "full_name": "Mel-Band Roformer 4 Stems FT Large by SYH99999",
            "category": "Splitter",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": "No",
            "information": "Large-scale 4-stems separation model by SYH99999",
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/MelBandRoformer4StemFTLarge.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/config.yaml?download=true"
        },
        "syh99999_4_stems_ft_large_v2": {
            "full_name": "Mel-Band Roformer 4 Stems FT Large v2 by SYH99999",
            "category": "Splitter",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": "No",
            "information": "Improved large-scale 4-stems separation model by SYH99999",
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/ver2.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/config.yaml?download=true"
        },
        "gaboxr67_instrumental_fv1": {
            "full_name": "Mel-Band Roformer Instrumental Fv1 by GaboxR67",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "information": "First version of instrumental separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv1.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },
        "gaboxr67_instrumental_fv2": {
            "full_name": "Mel-Band Roformer Instrumental Fv2 by GaboxR67",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "information": "Second version of instrumental separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv2.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },
        "gaboxr67_instrumental_fv3": {
            "full_name": "Mel-Band Roformer Instrumental Fv3 by GaboxR67",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "information": "Third version of instrumental separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv3.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },
        "gaboxr67_instrumental_fv4_noise": {
            "full_name": "Mel-Band Roformer Instrumental Fv4 Noise by GaboxR67",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "information": "Noise-optimized version of instrumental separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_Fv4Noise.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },
        "gaboxr67_instrumental_fv5": {
            "full_name": "Mel-Band Roformer Instrumental Fv5 by GaboxR67",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "information": "Fifth version of instrumental separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV5.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },
        "gaboxr67_instrumental_fv5_noise": {
            "full_name": "Mel-Band Roformer Instrumental Fv5 Noise by GaboxR67",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "information": "Noise-optimized fifth version of instrumental separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV5N.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },
        "gaboxr67_instrumental_fv6": {
            "full_name": "Mel-Band Roformer Instrumental Fv6 by GaboxR67",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "information": "Sixth version of instrumental separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },
        "gaboxr67_instrumental_fv6_noise": {
            "full_name": "Mel-Band Roformer Instrumental Fv6 Noise by GaboxR67",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "information": "Noise-optimized sixth version of instrumental separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6N.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },
        "gaboxr67_instrumental_fv7": {
            "full_name": "Mel-Band Roformer Instrumental Fv7 by GaboxR67",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "information": "Seventh version of instrumental separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxV7.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },
        "gaboxr67_instrumental_fv7_noise": {
            "full_name": "Mel-Band Roformer Instrumental Fv7 Noise by GaboxR67",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "information": "Noise-optimized seventh version of instrumental separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV7N.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },
        "gaboxr67_instrumental_fv8": {
            "full_name": "Mel-Band Roformer Instrumental Fv8 by GaboxR67",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "information": "Experimental eighth version of instrumental separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/INSTV8.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },
        "gaboxr67_instrumental_fv9": {
            "full_name": "Mel-Band Roformer Instrumental Fv9 by GaboxR67",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "information": "Experimental ninth version of instrumental separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/INSTV9.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },
        "gaboxr67_instrumental_fv10": {
            "full_name": "Mel-Band Roformer Instrumental Fv10 by GaboxR67",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "information": "Experimental tenth version of instrumental separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/INSTV10.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },
        "gaboxr67_instrumental_fvx": {
            "full_name": "Mel-Band Roformer Instrumental FvX by GaboxR67",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "information": "Extended version of instrumental separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxFVX.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },
        "gaboxr67_instrumental_bv1": {
            "full_name": "Mel-Band Roformer Instrumental Bv1 by GaboxR67",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "information": "First beta version of instrumental separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxBv1.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },
        "gaboxr67_instrumental_bv2": {
            "full_name": "Mel-Band Roformer Instrumental Bv2 by GaboxR67",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "information": "Second beta version of instrumental separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxBv2.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },
        "gaboxr67_vocals_fv1": {
            "full_name": "Mel-Band Roformer Vocals Fv1 by GaboxR67",
            "category": "Vocals",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Vocals",
            "information": "First version of vocal separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gaboxFv1.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml?download=true"
        },
        "gaboxr67_vocals_fv2": {
            "full_name": "Mel-Band Roformer Vocals Fv2 by GaboxR67",
            "category": "Vocals",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Vocals",
            "information": "Second version of vocal separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gaboxFv2.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml?download=true"
        },
        "gaboxr67_vocals_fv3": {
            "full_name": "Mel-Band Roformer Vocals Fv3 by GaboxR67",
            "category": "Vocals",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "vocals",
            "information": "Third version of vocal separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_Fv3.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml?download=true"
        },
        "gaboxr67_vocals_fv4": {
            "full_name": "Mel-Band Roformer Vocals Fv4 by GaboxR67",
            "category": "Vocals",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Vocals",
            "information": "Fourth version of vocal separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_fv4.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml?download=true"
        },
        "gaboxr67_karaoke_25_02_2025": {
            "full_name": "Mel-Band Roformer Karaoke 25-02-2025 by GaboxR67",
            "category": "Karaoke",
            "stems": ["karaoke", "other"],
            "target_instrument": "karaoke",
            "information": "Karaoke separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/noblebarkrr/all_models_for_mel_band_roformer/resolve/main/gabox_karaoke_25_02_2025.ckpt?download=true",
            "config_url": "https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/config_mel_band_roformer_karaoke.yaml?download=true"
        },
        "gaboxr67_karaoke_28_02_2025": {
            "full_name": "Mel-Band Roformer Karaoke 28-02-2025 by GaboxR67",
            "category": "Karaoke",
            "stems": ["karaoke", "other"],
            "target_instrument": "karaoke",
            "information": "Improved karaoke separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/noblebarkrr/all_models_for_mel_band_roformer/resolve/main/gabox_karaoke_28_02_2025.ckpt?download=true",
            "config_url": "https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/config_mel_band_roformer_karaoke.yaml?download=true"
        },
        "gaboxr67_denoise_debleed": {
            "full_name": "Mel-Band Roformer Denoise Debleed by GaboxR67",
            "category": "Denoise",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "information": "Denoise and debleed model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/denoisedebleed.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },
        "anvuew_dereverb": {
            "full_name": "Mel-Band Roformer Dereverb by Anvuew",
            "category": "Dereverb",
            "stems": ["reverb", "noreverb"],
            "target_instrument": "noreverb",
            "information": "Dereverb model by Anvuew",
            "checkpoint_url": "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt?download=true",
            "config_url": "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml?download=true"
        },
        "anvuew_dereverb_aggressive": {
            "full_name": "Mel-Band Roformer Dereverb Aggressive by Anvuew",
            "category": "Dereverb",
            "stems": ["reverb", "noreverb"],
            "target_instrument": "noreverb",
            "information": "Aggressive dereverb model by Anvuew",
            "checkpoint_url": "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt?download=true",
            "config_url": "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml?download=true"
        },
        "anvuew_dereverb_mono": {
            "full_name": "Mel-Band Roformer Dereverb Mono by Anvuew",
            "category": "Dereverb",
            "stems": ["reverb", "noreverb"],
            "target_instrument": "noreverb",
            "information": "Mono dereverb model by Anvuew",
            "checkpoint_url": "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt?download=true",
            "config_url": "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml?download=true"
        },
        "sucial_aspiration": {
            "full_name": "Mel-Band Roformer Aspiration by Sucial",
            "category": "Aspiration",
            "stems": ["aspiration", "other"],
            "target_instrument": "No",
            "information": "Aspiration removal model by Sucial",
            "checkpoint_url": "https://huggingface.co/Sucial/Aspiration_Mel_Band_Roformer/resolve/main/aspiration_mel_band_roformer_less_aggr_sdr_18.1201.ckpt?download=true",
            "config_url": "https://huggingface.co/Sucial/Aspiration_Mel_Band_Roformer/resolve/main/config_aspiration_mel_band_roformer.yaml?download=true"
        },
        "sucial_dereverb_deecho": {
            "full_name": "Mel-Band Roformer Dereverb-Echo by Sucial",
            "category": "Dereverb",
            "stems": ["dry", "other"],
            "target_instrument": "No",
            "information": "Dereverb and de-echo model by Sucial",
            "checkpoint_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt?download=true",
            "config_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb-echo_mel_band_roformer.yaml?download=true"
        },
        "sucial_big_dereverb": {
            "full_name": "Mel-Band Roformer Big Dereverb by Sucial",
            "category": "Dereverb",
            "stems": ["dry", "other"],
            "target_instrument": "dry",
            "information": "Large-scale dereverb model by Sucial",
            "checkpoint_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/de_big_reverb_mbr_ep_362.ckpt?download=true",
            "config_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml?download=true"
        },
        "sucial_super_big_dereverb": {
            "full_name": "Mel-Band Roformer Super Big Dereverb by Sucial",
            "category": "Dereverb",
            "stems": ["dry", "other"],
            "target_instrument": "dry",
            "information": "Extra large-scale dereverb model by Sucial",
            "checkpoint_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/de_super_big_reverb_mbr_ep_346.ckpt?download=true",
            "config_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml?download=true"
        },
        "sucial_dereverb_deecho_mbr_fused_v1": {
            "full_name": "Mel-Band Roformer Dereverb-Echo Fused v1 by Sucial",
            "category": "Dereverb",
            "stems": ["dry", "other"],
            "target_instrument": "dry",
            "information": "Fused dereverb and de-echo model by Sucial",
            "checkpoint_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt?download=true",
            "config_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml?download=true"
        },
        "sucial_dereverb_deecho_mbr_v2": {
            "full_name": "Mel-Band Roformer Dereverb-Echo v2 by Sucial",
            "category": "Dereverb",
            "stems": ["dry", "other"],
            "target_instrument": "dry",
            "information": "Improved dereverb and de-echo model by Sucial",
            "checkpoint_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb_echo_mbr_v2_sdr_dry_13.4843.ckpt?download=true",
            "config_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml?download=true"
        },
        "aufr33_viperx_karaoke": {
            "full_name": "Mel-Band Roformer Karaoke by Aufr33 & ViperX",
            "category": "Karaoke",
            "stems": ["karaoke", "other"],
            "target_instrument": "karaoke",
            "information": "Karaoke separation model by Aufr33 & ViperX",
            "checkpoint_url": "https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt?download=true",
            "config_url": "https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/config_mel_band_roformer_karaoke.yaml?download=true"
        },
        "aufr33_denoise": {
            "full_name": "Mel-Band Roformer Denoise by Aufr33",
            "category": "Denoise",
            "stems": ["dry", "other"],
            "target_instrument": "dry",
            "information": "Denoise model by Aufr33",
            "checkpoint_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt?download=true",
            "config_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml?download=true"
        },
        "aufr33_denoise_aggressive": {
            "full_name": "Mel-Band Roformer Denoise Aggressive by Aufr33",
            "category": "Denoise",
            "stems": ["dry", "other"],
            "target_instrument": "dry",
            "information": "Aggressive denoise model by Aufr33",
            "checkpoint_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt?download=true",
            "config_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml?download=true"
        },
        "aufr33_viperx_decrowd": {
            "full_name": "Mel-Band Roformer Decrowd by Aufr33 & ViperX",
            "category": "Decrowd",
            "stems": ["crowd", "other"],
            "target_instrument": "crowd",
            "information": "Crowd noise removal model by Aufr33 & ViperX",
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt",
            "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/model_mel_band_roformer_crowd.yaml"
        },
        "viperx_vocals": {
            "full_name": "Mel-Band Roformer Vocals by ViperX",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Vocal separation model by ViperX",
            "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_mel_band_roformer_ep_3005_sdr_11.4360.yaml"
        },
        "unwa_instrumental_v1_plus": {
            "full_name": "Mel-Band Roformer Instrumental v1 Plus by Unwa",
            "category": "Instrumental",
            "stems": ["vocals", "other"],
            "target_instrument": "other",
            "information": "Improved version of instrumental separation model by Unwa",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1_plus_test.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml?download=true"
        },
        "unwa_instrumental_v1e_plus": {
            "full_name": "Mel-Band Roformer Instrumental v1e Plus by Unwa",
            "category": "Instrumental",
            "stems": ["vocals", "other"],
            "target_instrument": "other",
            "information": "Extended version of instrumental separation model by Unwa",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e_plus.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml?download=true"
        },
        "unwa_big_beta_v6x": {
            "full_name": "Mel-Band Roformer Big Beta v6x by Unwa",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Extended version of large-scale vocal separation model by Unwa",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6x.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6x.yaml?download=true"
        },
        "unwa_kim_ft_v3": {
            "full_name": "Mel-Band Roformer Kim FT v3 by Unwa",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Third version of fine-tuned KimberleyJSN's model by Unwa",
            "checkpoint_url": "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft3_prev.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml?download=true"
        },
        "aname_vocals_fullness": {
            "full_name": "Mel-Band Roformer Vocals Fullness by Aname",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Vocal separation model with fullness by Aname",
            "checkpoint_url": "https://huggingface.co/Aname-Tommy/MelBandRoformers/resolve/main/FullnessVocalModel.ckpt?download=true",
            "config_url": "https://huggingface.co/Aname-Tommy/MelBandRoformers/resolve/main/config.yaml?download=true"
        },
        "gaboxr67_instrumental_donotusepls": {
            "full_name": "Mel-Band Roformer Instrumental DoNotUsePls by GaboxR67",
            "category": "Instrumental",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "information": "Experimental instrumental separation model by GaboxR67 (not recommended)",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/donotusepls.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },
        "aname_4_stems_large": {
            "full_name": "Mel-Band Roformer 4 Stems Large by Aname",
            "category": "Splitter",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": "No",
            "information": "Large 4-stems separation model by Aname",
            "checkpoint_url": "https://huggingface.co/Aname-Tommy/melbandroformer4stems/resolve/main/mel_band_roformer_4stems_large_ver1.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/config.yaml?download=true"
        },
        "aname_4_stems_xl": {
            "full_name": "Mel-Band Roformer 4 Stems XL by Aname",
            "category": "Splitter",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": "No",
            "information": "Extra large 4-stems separation model by Aname",
            "checkpoint_url": "https://huggingface.co/Aname-Tommy/melbandroformer4stems/resolve/main/mel_band_roformer_4stems_xl_ver1.ckpt?download=true",
            "config_url": "https://huggingface.co/Aname-Tommy/melbandroformer4stems/resolve/main/config_xl.yaml?download=true"
        },
        "exp_drums": {
            "full_name": "Mel-Band Roformer Drums Experimental",
            "category": "Drums",
            "stems": ["percussions", "other"],
            "target_instrument": "percussions",
            "information": "Experimental drums separation model",
            "checkpoint_url": "https://huggingface.co/am2460162/msst_failed_failed_test/resolve/main/model_mel_band_roformer_ep_11_sdr_7.6853.ckpt?download=true",
            "config_url": "https://huggingface.co/am2460162/msst_failed_failed_test/resolve/main/config_drums_musdb18_moises_mel_band_roformer.yaml?download=true"
        },
        "mesk_metal_inst_prev": {
            "full_name": "Mel-Band Roformer Metal Inst Preview by Mesk",
            "category": "Instrumental",
            "stems": ["vocals", "other"],
            "target_instrument": "other",
            "information": "Preview version of metal instrumental separation model by Mesk",
            "checkpoint_url": "https://huggingface.co/meskvlla33/metal_roformer_preview/resolve/main/metal_roformer_inst_mesk_preview.ckpt?download=true",
            "config_url": "https://huggingface.co/meskvlla33/metal_roformer_preview/resolve/main/config_inst_metal_roformer_mesk.yaml?download=true"
        }
    },
    "bs_roformer": {
        "exp_drums": {
            "full_name": "BS Roformer Drums Experimental",
            "category": "Drums",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": "drums",
            "information": "Experimental drums separation model for BS Roformer",
            "checkpoint_url": "https://huggingface.co/am2460162/msst_failed_failed_test/resolve/main/model_drums_bs_roformer_ep_12_sdr_7.2279.ckpt?download=true",
            "config_url": "https://huggingface.co/am2460162/msst_failed_failed_test/resolve/main/config_4_drums_bs_roformer.yaml?download=true"
        },
        "exp_bass": {
            "full_name": "BS Roformer Bass Experimental",
            "category": "Bass",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": "bass",
            "information": "Experimental bass separation model for BS Roformer",
            "checkpoint_url": "https://huggingface.co/am2460162/msst_failed_failed_test/resolve/main/model_bass_bs_roformer_ep_10_sdr_5.7862.ckpt?download=true",
            "config_url": "https://huggingface.co/am2460162/msst_failed_failed_test/resolve/main/config_4_bass_bs_roformer.yaml?download=true"
        },
        "viperx_vocals_1296": {
            "full_name": "BS Roformer Vocals 12.96 by ViperX",
            "category": "Vocals",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Vocals",
            "information": "Vocal separation model with SDR 12.96 by ViperX",
            "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt",
            "config_url": "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml"
        },
        "viperx_vocals_1297": {
            "full_name": "BS Roformer Vocals 12.97 by ViperX",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "information": "Vocal separation model with SDR 12.97 by ViperX",
            "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml"
        },
        "gabox_vocals": {
            "full_name": "BS Roformer Vocals by GaboxR67",
            "category": "Vocals",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "vocals",
            "information": "Vocal separation model by GaboxR67",
            "checkpoint_url": "https://huggingface.co/GaboxR67/BSRoformerVocTest/resolve/main/voc_gaboxBSR.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/BSRoformerVocTest/resolve/main/voc_gaboxBSroformer.yaml?download=true"
        },
        "zf_turbo_4_stems": {
            "full_name": "BS Roformer 4 Stems by ZFTurbo",
            "category": "Splitter",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": "No",
            "information": "4-stems separation model by ZFTurbo",
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt",
            "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/config_bs_roformer_384_8_2_485100.yaml"
        },
        "syh99999_4_stems_ft": {
            "full_name": "BS Roformer 4 Stems FT by SYH99999",
            "category": "Splitter",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": "No",
            "information": "Fine-tuned 4-stems separation model by SYH99999",
            "checkpoint_url": "https://huggingface.co/SYH99999/bs_roformer_4stems_ft/resolve/main/bs_roformer_4stems_ft.pth?download=true",
            "config_url": "https://huggingface.co/SYH99999/bs_roformer_4stems_ft/resolve/main/config.yaml?download=true"
        },
        "sucial_male_female_146": {
            "full_name": "BS Roformer Male-Female by Sucial",
            "category": "Male-Female",
            "stems": ["male", "female"],
            "target_instrument": "No",
            "information": "Male-Female vocal separation model by Sucial (ep 146)",
            "checkpoint_url": "https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/model_chorus_bs_roformer_ep_146_sdr_23.8613.ckpt?download=true",
            "config_url": "https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/config_chorus_male_female_bs_roformer.yaml?download=true"
        },
        "sucial_male_female_267": {
            "full_name": "BS Roformer Male-Female by Sucial",
            "category": "Male-Female",
            "stems": ["male", "female"],
            "target_instrument": "No",
            "information": "Male-Female vocal separation model by Sucial (ep 267)",
            "checkpoint_url": "https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt?download=true",
            "config_url": "https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/config_chorus_male_female_bs_roformer.yaml?download=true"
        },
        "aufr33_male_female": {
            "full_name": "BS Roformer Male-Female by Aufr33",
            "category": "Male-Female",
            "stems": ["male", "female"],
            "target_instrument": "No",
            "information": "Male-Female vocal separation model by Aufr33",
            "checkpoint_url": "https://huggingface.co/RareSirMix/AIModelRehosting/resolve/main/bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt",
            "config_url": "https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/config_chorus_male_female_bs_roformer.yaml"
        },
        "anvuew_deverb_256_8": {
            "full_name": "BS Roformer Deverb 256-8 by Anvuew",
            "category": "Dereverb",
            "stems": ["reverb", "noreverb"],
            "target_instrument": "noreverb",
            "information": "Dereverb model with 256 dim and 8 depth by Anvuew",
            "checkpoint_url": "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_256dim_8depth.ckpt?download=true",
            "config_url": "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_256dim_8depth.yaml?download=true"
        },
        "anvuew_deverb_384_10": {
            "full_name": "BS Roformer Deverb 384-10 by Anvuew",
            "category": "Dereverb",
            "stems": ["reverb", "noreverb"],
            "target_instrument": "noreverb",
            "information": "Dereverb model with 384 dim and 10 depth by Anvuew",
            "checkpoint_url": "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_384dim_10depth.ckpt?download=true",
            "config_url": "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_384dim_10depth.yaml?download=true"
        },
        "aname_4_stems": {
            "full_name": "BS Roformer 4 stems by Aname",
            "category": "Splitter",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": "No",
            "information": "4 stem model for BS Roformer by Aname",
            "checkpoint_url": "https://huggingface.co/RareSirMix/AIModelRehosting/resolve/main/Amane4stem_bs_roformer.ckpt",
            "config_url": "https://huggingface.co/RareSirMix/AIModelRehosting/resolve/main/Amane4stem_bs_roformer.yaml"
        }
    },
    "mdx23c": {
        "inst_voc_hq": {
            "full_name": "MDX23C Inst-Voc HQ by ZFTurbo",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "No",
            "information": "High-quality instrumental-vocal separation model by ZFTurbo",
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_mdx23c.yaml"
        },
        "aufr33_jarredou_drumsep": {
            "full_name": "MDX23C DrumSep by Aufr33 & Jarredou",
            "category": "Drumsep",
            "stems": ["kick", "snare", "toms", "hh", "ride", "crash"],
            "target_instrument": "No",
            "information": "Drum separation model by Aufr33 & Jarredou",
            "checkpoint_url": "https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt",
            "config_url": "https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml"
        },
        "aufr33_jarredou_dereverb": {
            "full_name": "MDX23C Dereverb by Aufr33 & Jarredou",
            "category": "Dereverb",
            "stems": ["dry", "other"],
            "target_instrument": "dry",
            "information": "Dereverb model by Aufr33 & Jarredou",
            "checkpoint_url": "https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/dereverb_mdx23c_sdr_6.9096.ckpt",
            "config_url": "https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/config_dereverb_mdx23c.yaml"
        },
        "wesleyr36_mid_side": {
            "full_name": "MDX23C Mid-Side by WesleyR36",
            "category": "Mid-Side",
            "stems": ["similarity", "difference"],
            "target_instrument": "similarity",
            "information": "Mid-Side separation model by WesleyR36",
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.10/model_mdx23c_ep_271_l1_freq_72.2383.ckpt",
            "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.10/config_mdx23c_similarity.yaml"
        },
        "zf_turbo_4_stems": {
            "full_name": "MDX23C 4 Stems by ZFTurbo",
            "category": "Splitter",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": "No",
            "information": "4-stems separation model by ZFTurbo",
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.1/model_mdx23c_ep_168_sdr_7.0207.ckpt",
            "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.1/config_musdb18_mdx23c.yaml"
        },
        "exp_orch": {
            "full_name": "MDX23C Orchestra Experimental",
            "category": "Orchestra",
            "stems": ["inst", "orch"],
            "target_instrument": "orch",
            "information": "Experimental orchestra separation model",
            "checkpoint_url": "https://huggingface.co/am2460162/msst_failed_failed_test/resolve/main/model_mdx23c_ep_120_sdr_4.4174.ckpt?download=true",
            "config_url": "https://huggingface.co/am2460162/msst_failed_failed_test/resolve/main/config_orchestra_mdx23c.yaml?download=true"
        }
    },
    "scnet": {
        "zf_turbo_4_stems": {
            "full_name": "SCNet 4 Stems by ZFTurbo",
            "category": "Splitter",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": "No",
            "information": "4-stems separation model by ZFTurbo",
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/SCNet-large_starrytong_fixed.ckpt",
            "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/config_musdb18_scnet_large_starrytong.yaml"
        },
        "starrytong_4_stems_xl": {
            "full_name": "SCNet 4 Stems XL by StarryTong",
            "category": "Splitter",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": "No",
            "information": "Extra large 4-stems separation model by StarryTong",
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/model_scnet_ep_54_sdr_9.8051.ckpt",
            "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/config_musdb18_scnet_xl.yaml"
        },
        "zf_turbo_4_stems_xl": {
            "full_name": "SCNet 4 Stems XL by ZFTurbo",
            "category": "Splitter",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": "No",
            "information": "Extra large 4-stems separation model by ZFTurbo",
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/scnet_checkpoint_musdb18.ckpt",
            "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/config_musdb18_scnet.yaml"
        }
    },

    # UVR models

    "vr_arch": {
        "1_HP-UVR.pth": {
            "full_name": "VR Arch Single Model v5: 1_HP-UVR",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "2_HP-UVR.pth": {
            "full_name": "VR Arch Single Model v5: 2_HP-UVR",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "3_HP-Vocal-UVR.pth": {
            "full_name": "VR Arch Single Model v5: 3_HP-Vocal-UVR",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No"
        },
        "4_HP-Vocal-UVR.pth": {
            "full_name": "VR Arch Single Model v5: 4_HP-Vocal-UVR",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No"
        },
        "5_HP-Karaoke-UVR.pth": {
            "full_name": "VR Arch Single Model v5: 5_HP-Karaoke-UVR",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "6_HP-Karaoke-UVR.pth": {
            "full_name": "VR Arch Single Model v5: 6_HP-Karaoke-UVR",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "7_HP2-UVR.pth": {
            "full_name": "VR Arch Single Model v5: 7_HP2-UVR",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "8_HP2-UVR.pth": {
            "full_name": "VR Arch Single Model v5: 8_HP2-UVR",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "9_HP2-UVR.pth": {
            "full_name": "VR Arch Single Model v5: 9_HP2-UVR",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "10_SP-UVR-2B-32000-1.pth": {
            "full_name": "VR Arch Single Model v5: 10_SP-UVR-2B-32000-1",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "11_SP-UVR-2B-32000-2.pth": {
            "full_name": "VR Arch Single Model v5: 11_SP-UVR-2B-32000-2",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "12_SP-UVR-3B-44100.pth": {
            "full_name": "VR Arch Single Model v5: 12_SP-UVR-3B-44100",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "13_SP-UVR-4B-44100-1.pth": {
            "full_name": "VR Arch Single Model v5: 13_SP-UVR-4B-44100-1",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "14_SP-UVR-4B-44100-2.pth": {
            "full_name": "VR Arch Single Model v5: 14_SP-UVR-4B-44100-2",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "15_SP-UVR-MID-44100-1.pth": {
            "full_name": "VR Arch Single Model v5: 15_SP-UVR-MID-44100-1",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "16_SP-UVR-MID-44100-2.pth": {
            "full_name": "VR Arch Single Model v5: 16_SP-UVR-MID-44100-2",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "17_HP-Wind_Inst-UVR.pth": {
            "full_name": "VR Arch Single Model v5: 17_HP-Wind_Inst-UVR",
            "stems": ["No Woodwinds", "Woodwinds"],
            "target_instrument": "No"
        },
        "UVR-De-Echo-Aggressive.pth": {
            "full_name": "VR Arch Single Model v5: UVR-De-Echo-Aggressive by FoxJoy",
            "stems": ["No Echo", "Echo"],
            "target_instrument": "No"
        },
        "UVR-De-Echo-Normal.pth": {
            "full_name": "VR Arch Single Model v5: UVR-De-Echo-Normal by FoxJoy",
            "stems": ["No Echo", "Echo"],
            "target_instrument": "No"
        },
        "UVR-DeEcho-DeReverb.pth": {
            "full_name": "VR Arch Single Model v5: UVR-DeEcho-DeReverb by FoxJoy",
            "stems": ["No Reverb", "Reverb"],
            "target_instrument": "No"
        },
        "UVR-DeNoise-Lite.pth": {
            "full_name": "VR Arch Single Model v5: UVR-DeNoise-Lite by FoxJoy",
            "stems": ["Noise", "No Noise"],
            "target_instrument": "No"
        },
        "UVR-DeNoise.pth": {
            "full_name": "VR Arch Single Model v5: UVR-DeNoise by FoxJoy",
            "stems": ["Noise", "No Noise"],
            "target_instrument": "No"
        },
        "UVR-BVE-4B_SN-44100-1.pth": {
            "full_name": "VR Arch Single Model v5: UVR-BVE-4B_SN-44100-1",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No"
        },
        "MGM_HIGHEND_v4.pth": {
            "full_name": "VR Arch Single Model v4: MGM_HIGHEND_v4",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "MGM_LOWEND_A_v4.pth": {
            "full_name": "VR Arch Single Model v4: MGM_LOWEND_A_v4",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "MGM_LOWEND_B_v4.pth": {
            "full_name": "VR Arch Single Model v4: MGM_LOWEND_B_v4",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "MGM_MAIN_v4.pth": {
            "full_name": "VR Arch Single Model v4: MGM_MAIN_v4",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "UVR-De-Reverb-aufr33-jarredou.pth": {
            "full_name": "VR Arch Single Model v4: UVR-De-Reverb by aufr33-jarredou",
            "stems": ["Dry", "No Dry"],
            "target_instrument": "No"
        }
    },
    "mdx_net": {
        "UVR-MDX-NET-Inst_HQ_1.onnx": {
            "full_name": "MDX-Net Model: UVR-MDX-NET Inst HQ 1",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "UVR-MDX-NET-Inst_HQ_2.onnx": {
            "full_name": "MDX-Net Model: UVR-MDX-NET Inst HQ 2",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "UVR-MDX-NET-Inst_HQ_3.onnx": {
            "full_name": "MDX-Net Model: UVR-MDX-NET Inst HQ 3",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "UVR-MDX-NET-Inst_HQ_4.onnx": {
            "full_name": "MDX-Net Model: UVR-MDX-NET Inst HQ 4",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "UVR-MDX-NET-Inst_HQ_5.onnx": {
            "full_name": "MDX-Net Model: UVR-MDX-NET Inst HQ 5",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "UVR_MDXNET_Main.onnx": {
            "full_name": "MDX-Net Model: UVR-MDX-NET Main",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No"
        },
        "UVR-MDX-NET-Inst_Main.onnx": {
            "full_name": "MDX-Net Model: UVR-MDX-NET Inst Main",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "UVR_MDXNET_1_9703.onnx": {
            "full_name": "MDX-Net Model: UVR-MDX-NET 1",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No"
        },
        "UVR_MDXNET_2_9682.onnx": {
            "full_name": "MDX-Net Model: UVR-MDX-NET 2",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No"
        },
        "UVR_MDXNET_3_9662.onnx": {
            "full_name": "MDX-Net Model: UVR-MDX-NET 3",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No"
        },
        "UVR-MDX-NET-Inst_1.onnx": {
            "full_name": "MDX-Net Model: UVR-MDX-NET Inst 1",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "UVR-MDX-NET-Inst_2.onnx": {
            "full_name": "MDX-Net Model: UVR-MDX-NET Inst 2",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "UVR-MDX-NET-Inst_3.onnx": {
            "full_name": "MDX-Net Model: UVR-MDX-NET Inst 3",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "UVR_MDXNET_KARA.onnx": {
            "full_name": "MDX-Net Model: UVR-MDX-NET Karaoke",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No"
        },
        "UVR_MDXNET_KARA_2.onnx": {
            "full_name": "MDX-Net Model: UVR-MDX-NET Karaoke 2",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "UVR_MDXNET_9482.onnx": {
            "full_name": "MDX-Net Model: UVR_MDXNET_9482",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No"
        },
        "UVR-MDX-NET-Voc_FT.onnx": {
            "full_name": "MDX-Net Model: UVR-MDX-NET Voc FT",
            "stems": ["Vocals", "Instrumental"]
        },
        "Kim_Vocal_1.onnx": {
            "full_name": "MDX-Net Model: Kim Vocal 1",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No"
        },
        "Kim_Vocal_2.onnx": {
            "full_name": "MDX-Net Model: Kim Vocal 2",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No"
        },
        "Kim_Inst.onnx": {
            "full_name": "MDX-Net Model: Kim Inst",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "Reverb_HQ_By_FoxJoy.onnx": {
            "full_name": "MDX-Net Model: Reverb HQ By FoxJoy",
            "stems": ["Reverb", "No Reverb"],
            "target_instrument": "No"
        },
        "UVR-MDX-NET_Crowd_HQ_1.onnx": {
            "full_name": "MDX-Net Model: UVR-MDX-NET Crowd HQ 1 By Aufr33",
            "stems": ["No Crowd", "Crowd"],
            "target_instrument": "No"
        },
        "kuielab_a_vocals.onnx": {
            "full_name": "MDX-Net Model: kuielab_a_vocals",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No"
        },
        "kuielab_a_other.onnx": {
            "full_name": "MDX-Net Model: kuielab_a_other",
            "stems": ["Other", "No Other"],
            "target_instrument": "No"
        },
        "kuielab_a_bass.onnx": {
            "full_name": "MDX-Net Model: kuielab_a_bass",
            "stems": ["Bass", "No Bass"],
            "target_instrument": "No"
        },
        "kuielab_a_drums.onnx": {
            "full_name": "MDX-Net Model: kuielab_a_drums",
            "stems": ["Drums", "No Drums"],
            "target_instrument": "No"
        },
        "kuielab_b_vocals.onnx": {
            "full_name": "MDX-Net Model: kuielab_b_vocals",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No"
        },
        "kuielab_b_other.onnx": {
            "full_name": "MDX-Net Model: kuielab_b_other",
            "stems": ["Other", "No Other"],
            "target_instrument": "No"
        },
        "kuielab_b_bass.onnx": {
            "full_name": "MDX-Net Model: kuielab_b_bass",
            "stems": ["Bass", "No Bass"],
            "target_instrument": "No"
        },
        "kuielab_b_drums.onnx": {
            "full_name": "MDX-Net Model: kuielab_b_drums",
            "stems": ["Drums", "No Drums"],
            "target_instrument": "No"
        },
        "UVR-MDX-NET_Main_340.onnx": {
            "full_name": "MDX-Net Model VIP: UVR-MDX-NET_Main_340",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No"
        },
        "UVR-MDX-NET_Main_390.onnx": {
            "full_name": "MDX-Net Model VIP: UVR-MDX-NET_Main_390",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No"
        },
        "UVR-MDX-NET_Main_406.onnx": {
            "full_name": "MDX-Net Model VIP: UVR-MDX-NET_Main_406",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No"
        },
        "UVR-MDX-NET_Main_427.onnx": {
            "full_name": "MDX-Net Model VIP: UVR-MDX-NET_Main_427",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No"
        },
        "UVR-MDX-NET_Main_438.onnx": {
            "full_name": "MDX-Net Model VIP: UVR-MDX-NET_Main_438",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "No"
        },
        "UVR-MDX-NET_Inst_82_beta.onnx": {
            "full_name": "MDX-Net Model VIP: UVR-MDX-NET_Inst_82_beta",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "UVR-MDX-NET_Inst_90_beta.onnx": {
            "full_name": "MDX-Net Model VIP: UVR-MDX-NET_Inst_90_beta",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "UVR-MDX-NET_Inst_187_beta.onnx": {
            "full_name": "MDX-Net Model VIP: UVR-MDX-NET_Inst_187_beta",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        },
        "UVR-MDX-NET-Inst_full_292.onnx": {
            "full_name": "MDX-Net Model VIP: UVR-MDX-NET-Inst_full_292",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": "No"
        }
    },
    "htdemucs": {
        "v4_mvsep_vocals": {
            "full_name": "HT Demucs v4 Vocals (MVSep finetuned)",
            "category": "Vocals",
            "stems": ["vocals", "other"],
            "target_instrument": "No",
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_htdemucs_sdr_8.78.ckpt",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_htdemucs.yaml"
        },
        "v4_4stems": {
            "full_name": "HT Demucs v4 4 stems",
            "category": "Splitter",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": "No",
            "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml"
        },
        "v4_6stems": {
            "full_name": "HT Demucs v4 6 stems",
            "category": "Splitter",
            "stems": ["vocals", "drums", "bass", "other", "guitar", "piano"],
            "target_instrument": "No",
            "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_htdemucs_6stems.yaml"
        },
        "v3_mmi_4_stems": {
            "full_name": "HT Demucs v3 MMI 4 stems",
            "category": "Splitter",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": "No",
            "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/75fc33f5-1941ce65.th",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_demucs3_mmi.yaml"
        },
        "v4_ft_bass": {
            "full_name": "HT Demucs v4 FT Bass",
            "category": "Bass",
            "stems": ["bass"],
            "target_instrument": "No",
            "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/d12395a8-e57c48e6.th",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml"
        },
        "v4_ft_drums": {
            "full_name": "HT Demucs v4 FT Drums",
            "category": "Drums",
            "stems": ["drums"],
            "target_instrument": "No",
            "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml"
        },
        "v4_ft_vocals": {
            "full_name": "HT Demucs v4 FT Vocals",
            "category": "Vocals",
            "stems": ["vocals"],
            "target_instrument": "No",
            "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/04573f0d-f3cf25b2.th",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml"
        },
    }
}

