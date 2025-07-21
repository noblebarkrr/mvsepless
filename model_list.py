models_data = {

    "mel_band_roformer": {

        "Mel-Band-Roformer_Vocals_kimberley_jensen": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Vocals by Kimberley Jensen",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt?download=true",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml"
        },

        "Mel-Band-Roformer_InstVoc_Duality_v1_unwa": {
            "category": "Инструментал и вокал",
            "full_name": "Mel-Band Roformer InstVoc Duality v1 by Unwa",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvoc_duality_v1.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/config_melbandroformer_instvoc_duality.yaml?download=true"
        },

        "Mel-Band-Roformer_InstVoc_Duality_v2_unwa": {
            "category": "Инструментал и вокал",
            "full_name": "Mel-Band Roformer InstVoc Duality v2 by Unwa",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvox_duality_v2.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/config_melbandroformer_instvoc_duality.yaml?download=true"
        },

        "Mel-Band-Roformer_Kim_FT_v1_unwa": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Kim FT v1 by Unwa",
            "stems": ["Vocals", "other"],
            "target_instrument": "Vocals",
            "checkpoint_url": "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml?download=true"
        },

        "Mel-Band-Roformer_Kim_FT_v2_unwa": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Kim FT v2 by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml?download=true"
        },

        "Mel-Band-Roformer_Kim_FT_v2_bleedless_unwa": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Kim FT v2 Bleedless by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2_bleedless.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml?download=true"
        },

        "Mel-Band-Roformer_Kim_FT_v3_prev_unwa": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Kim FT v3 preview by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft3_prev.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml?download=true"
        },

        "Mel-Band-Roformer_Big_Beta_v1_unwa": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Big Beta v1 by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta1.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/config_melbandroformer_big.yaml?download=true"
        },

        "Mel-Band-Roformer_Big_Beta_v2_unwa": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Big Beta v2 by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta2.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/config_melbandroformer_big.yaml?download=true"
        },

        "Mel-Band-Roformer_Big_Beta_v3_unwa": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Big Beta v3 by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta3.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/config_melbandroformer_big.yaml?download=true"
        },

        "Mel-Band-Roformer_Big_Beta_v4_unwa": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Big Beta v4 by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta4.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/config_melbandroformer_big_beta4.yaml?download=true"
        },

        "Mel-Band-Roformer_Big_Beta_v5e_unwa": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Vocals Big Beta v5e by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.yaml?download=true"
        },

        "Mel-Band-Roformer_Big_Beta_v6_unwa": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Big Beta v6 by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6.yaml?download=true"
        },

        "Mel-Band-Roformer_Big_Beta_v6x_unwa": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Big Beta v6x by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6x.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6x.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_v1_unwa": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental v1 by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "other",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v1.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_v1_plus_unwa": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental v1+ by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "other",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1_plus_test.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_v1e_unwa": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental v1e by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "other",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_v1e_plus_unwa": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental v1e Plus by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "other",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e_plus.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_v2_unwa": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental v2 by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "other",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v2.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst_v2.yaml?download=true"
        },

        "Mel-Band-Roformer_Small_v1_unwa": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Small v1 by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-small/resolve/main/melband_roformer_small_v1.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-small/resolve/main/config_melbandroformer_small.yaml?download=true"
        },

        "Mel-Band-Roformer_Bleed_Suppressor_v1_unwa_97chris": {
            "category": "Шум",
            "full_name": "Mel-Band Roformer Bleed Suppressor v1 by Unwa / 97chris",
            "stems": ["Instrumental", "Bleed"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/jarredou/bleed_suppressor_melband_rofo_by_unwa_97chris/resolve/main/bleed_suppressor_v1.ckpt?download=true",
            "config_url": "https://huggingface.co/jarredou/bleed_suppressor_melband_rofo_by_unwa_97chris/resolve/main/config_bleed_suppressor_v1.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_becruliy": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental by Becruily",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt?download=true",
            "config_url": "https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/config_instrumental_becruily.yaml?download=true"
        },

        "Mel-Band-Roformer_Guitar_becruily": {
            "category": "Гитара",
            "full_name": "Mel-Band Roformer Instrumental by Becruily",
            "stems": ["Guitar", "Other"],
            "target_instrument": "Guitar",
            "checkpoint_url": "https://huggingface.co/becruily/mel-band-roformer-guitar/resolve/main/becruily_guitar.ckpt?download=true",
            "config_url": "https://huggingface.co/becruily/mel-band-roformer-guitar/resolve/main/config_guitar_becruily.yaml?download=true"
        },

        "Mel-Band-Roformer_Karaoke_becruily": {
            "category": "Караоке",
            "full_name": "Mel-Band Roformer Karaoke by Becruily",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/mel_band_roformer_karaoke_becruily.ckpt?download=true",
            "config_url": "https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/config_karaoke_becruily.yaml?download=true"
        },

        "Mel-Band-Roformer_Vocals_becruily": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Vocals by Becruily",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt?download=true",
            "config_url": "https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/config_vocals_becruily.yaml?download=true"
        },

        "Mel-Band-Roformer_SYHFT_v1_syh99999": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer SYHFT v1 by SYH99999",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFT/resolve/main/MelBandRoformerSYHFT.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFT/resolve/main/config_vocals_mel_band_roformer_ft.yaml?download=true"
        },

        "Mel-Band-Roformer_SYHFT_v2_syh99999": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer SYHFT v2 by SYH99999",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFTV2/resolve/main/MelBandRoformerSYHFTV2.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFT/resolve/main/config_vocals_mel_band_roformer_ft.yaml?download=true"
        },

        "Mel-Band-Roformer_SYHFT_v2.5_syh99999": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer SYHFT v2.5 by SYH99999",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFTV2.5/resolve/main/MelBandRoformerSYHFTV2.5.ckpt/MelBandRoformerSYHFTV2.5.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFT/resolve/main/config_vocals_mel_band_roformer_ft.yaml?download=true"
        },

        "Mel-Band-Roformer_SYHFT_v3_syh99999": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer SYHFT v3 by SYH99999",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFTV3Epsilon/resolve/main/MelBandRoformerSYHFTV3Epsilon.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFT/resolve/main/config_vocals_mel_band_roformer_ft.yaml?download=true"
        },

        "Mel-Band-Roformer_BIG_SYHFT_v1_Fast_syh99999": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Big SYHFT v1 Fast by SYH99999",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformerBigSYHFTV1Fast/resolve/main/MelBandRoformerBigSYHFTV1.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformerBigSYHFTV1Fast/resolve/main/config.yaml?download=true"
        },

        "Mel-Band-Roformer_Merged_SYHFT_Beta_v1_syh99999": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Merged Beta v1 by SYH99999",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformerMergedSYHFTBeta1/resolve/main/merge_syhft.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFT/resolve/main/config_vocals_mel_band_roformer_ft.yaml?download=true"
        },

        "Mel-Band-Roformer_SYHFT_B1_model1_syh99999": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer SYHFT B1 1 by SYH99999",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml?download=true"
        },

        "Mel-Band-Roformer_SYHFT_B1_model2_syh99999": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer SYHFT B1 2 by SYH99999",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model2.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml?download=true"
        },

        "Mel-Band-Roformer_SYHFT_B1_model3_syh99999": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer SYHFT B1 3 by SYH99999",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model3.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml?download=true"
        },

        "Mel-Band-Roformer_4_stems_FT_Large_v1_syh99999": {
            "category": "4 стема",
            "full_name": "Mel-Band Roformer 4 Stems FT Large v1 by SYH99999",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/MelBandRoformer4StemFTLarge.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/config.yaml?download=true"
        },

        "Mel-Band-Roformer_4_stems_FT_Large_v2_syh99999": {
            "category": "4 стема",
            "full_name": "Mel-Band Roformer 4 Stems FT Large v2 by SYH99999",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/ver2.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/config.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_1652_essid": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental by Essid (sdr 16.52)",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/Essid/Essid-MelBandRoformer/resolve/3960860f7895c87a12707ca6b378df2b3c68e2c0/model_mel_band_roformer_ep_17_sdr_16.5244.ckpt?download=true",
            "config_url": "https://huggingface.co/Essid/Essid-MelBandRoformer/resolve/4768859bd59bc699d33f4567e82082993dde7eb9/config_vocals_mel_band_roformer_essid.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_1681_essid": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental by Essid (sdr 16.81)",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/Essid/Essid-MelBandRoformer/resolve/main/essid_mel_inst_old.ckpt?download=true",
            "config_url": "https://huggingface.co/Essid/Essid-MelBandRoformer/resolve/4768859bd59bc699d33f4567e82082993dde7eb9/config_vocals_mel_band_roformer_essid.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_Fv1_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Fv1 by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv1.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_Fv2_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Fv2 by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv2.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_Fv3_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Fv3 by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv3.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_Fv4N_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Fv4 Noise by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_Fv4Noise.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_Fv5_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Fv5 by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV5.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_Fv5N_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Fv5 Noise by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV5N.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_Fv6_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Fv6 by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_Fv6N_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Fv6 Noise by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6N.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_Fv7_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Fv7 by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxV7.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_Fv7N_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Fv7 Noise by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV7N.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_Fv7_plus_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Fv7+ by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/5fba9605d4b6bc1a31c04c50d08d757c5107d23f/melbandroformers/experimental/instv7plus.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_Fv7z_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Fv7z by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxFv7z.ckpt?download=true?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },


        "Mel-Band-Roformer_Instrumental_Fv8_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Fv8 by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxFv8.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },
        
        "Mel-Band-Roformer_Instrumental_Fv8b_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Fv8b by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/Inst_FV8b.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },


        "Mel-Band-Roformer_Instrumental_Fv9_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Fv9 by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/Inst_Fv9.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_Fv10_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Fv10 by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/INSTV10.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_FvX_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental FvX by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxFVX.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_Bv1_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Bv1 by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxBv1.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_Bv2_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Bv2 by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxBv2.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_Bv3_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Bv3 by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxBv3.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_small_gabox": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Instrumental Small by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/small_inst.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },


        "Mel-Band-Roformer_Vocals_Fv1_gabox": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Vocals Fv1 by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Vocals",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gaboxFv1.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Vocals_Fv2_gabox": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Vocals Fv2 by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Vocals",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gaboxFv2.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Vocals_Fv3_gabox": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Vocals Fv3 by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_Fv3.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Vocals_Fv4_gabox": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Vocals Fv4 by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Vocals",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_fv4.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Vocals_Fv5_gabox": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Vocals Fv5 by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Vocals",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_fv5.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Karaoke_25_02_2025_gabox": {
            "category": "Караоке",
            "full_name": "Mel-Band Roformer Karaoke 25-02-2025 by GaboxR67",
            "stems": ["karaoke", "other"],
            "target_instrument": "karaoke",
            "checkpoint_url": "https://huggingface.co/noblebarkrr/all_models_for_mel_band_roformer/resolve/main/gabox_karaoke_25_02_2025.ckpt?download=true",
            "config_url": "https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/config_mel_band_roformer_karaoke.yaml?download=true"
        },

        "Mel-Band-Roformer_Karaoke_28_02_2025_gabox": {
            "category": "Караоке",
            "full_name": "Mel-Band Roformer Karaoke 28-02-2025 by GaboxR67",
            "stems": ["karaoke", "other"],
            "target_instrument": "karaoke",
            "checkpoint_url": "https://huggingface.co/noblebarkrr/all_models_for_mel_band_roformer/resolve/main/gabox_karaoke_28_02_2025.ckpt?download=true",
            "config_url": "https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/config_mel_band_roformer_karaoke.yaml?download=true"
        },

        "Mel-Band-Roformer_Karaoke_v1_gabox": {
            "category": "Караоке",
            "full_name": "Mel-Band Roformer Karaoke v1 by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Vocals",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/27e73ca2beec0ab7daa46e366159753a166612e1/melbandroformers/karaoke/Karaoke_GaboxV1.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/27e73ca2beec0ab7daa46e366159753a166612e1/melbandroformers/karaoke/karaokegabox_1750911344.yaml?download=true"
        },

        "Mel-Band-Roformer_Denoise_DeBleed_gabox": {
            "category": "Шум",
            "full_name": "Mel-Band Roformer Denoise DeBleed by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Instrumental",
            "checkpoint_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/denoisedebleed.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml?download=true"
        },

        "Mel-Band-Roformer_Karaoke_Fusion_gonzaluigi": {
            "category": "Караоке",
            "full_name": "Mel-Band Roformer Karaoke Fusion by Gonzaluigi",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Vocals",
            "checkpoint_url": "https://huggingface.co/Gonzaluigi/Mel-Band-Karaoke-Fusion/resolve/main/mel_band_karaoke_fusion_standard.ckpt?download=true",
            "config_url": "https://huggingface.co/Gonzaluigi/Mel-Band-Karaoke-Fusion/resolve/main/melband_karaokefusion_gonza.yaml?download=true"
        },

        "Mel-Band-Roformer_Karaoke_Fusion_Aggr_gonzaluigi": {
            "category": "Караоке",
            "full_name": "Mel-Band Roformer Karaoke Fusion Aggressive by Gonzaluigi",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Vocals",
            "checkpoint_url": "https://huggingface.co/Gonzaluigi/Mel-Band-Karaoke-Fusion/resolve/main/mel_band_karaoke_fusion_aggressive.ckpt?download=true",
            "config_url": "https://huggingface.co/Gonzaluigi/Mel-Band-Karaoke-Fusion/resolve/main/melband_karaokefusion_gonza.yaml?download=true"
        },







        "Mel-Band-Roformer_DeReverb_anvuew": {
            "category": "Реверб",
            "full_name": "Mel-Band Roformer DeReverb by Anvuew",
            "stems": ["reverb", "noreverb"],
            "target_instrument": "noreverb",
            "checkpoint_url": "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt?download=true",
            "config_url": "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml?download=true"
        },

        "Mel-Band-Roformer_DeReverb_Less_Aggr_anvuew": {
            "category": "Реверб",
            "full_name": "Mel-Band Roformer DeReverb Less Aggressive by Anvuew",
            "stems": ["reverb", "noreverb"],
            "target_instrument": "noreverb",
            "checkpoint_url": "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt?download=true",
            "config_url": "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml?download=true"
        },

        "Mel-Band-Roformer_DeReverb_Mono_anvuew": {
            "category": "Реверб",
            "full_name": "Mel-Band Roformer DeReverb Mono by Anvuew",
            "stems": ["reverb", "noreverb"],
            "target_instrument": "noreverb",
            "checkpoint_url": "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt?download=true",
            "config_url": "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml?download=true"
        },

        "Mel-Band-Roformer_Aspiration_sucial": {
            "category": "Дыхание",
            "full_name": "Mel-Band Roformer Aspiration by Sucial",
            "stems": ["aspiration", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/Sucial/Aspiration_Mel_Band_Roformer/resolve/main/aspiration_mel_band_roformer_less_aggr_sdr_18.1201.ckpt?download=true",
            "config_url": "https://huggingface.co/Sucial/Aspiration_Mel_Band_Roformer/resolve/main/config_aspiration_mel_band_roformer.yaml?download=true"
        },

        "Mel-Band-Roformer_DeReverb-Echo_v1_sucial": {
            "category": "Реверб и эхо",
            "full_name": "Mel-Band Roformer DeReverb-Echo by Sucial",
            "stems": ["dry", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt?download=true",
            "config_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb-echo_mel_band_roformer.yaml?download=true"
        },

        "Mel-Band-Roformer_DeBigReverb_sucial": {
            "category": "Реверб",
            "full_name": "Mel-Band Roformer DeBigReverb by Sucial",
            "stems": ["dry", "other"],
            "target_instrument": "dry",
            "checkpoint_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/de_big_reverb_mbr_ep_362.ckpt?download=true",
            "config_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml?download=true"
        },

        "Mel-Band-Roformer_DeSuperBigReverb_sucial": {
            "category": "Реверб",
            "full_name": "Mel-Band Roformer Super Big DeReverb by Sucial",
            "stems": ["dry", "other"],
            "target_instrument": "dry",
            "checkpoint_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/de_super_big_reverb_mbr_ep_346.ckpt?download=true",
            "config_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml?download=true"
        },

        "Mel-Band-Roformer_DeReverb-Echo_Fused_sucial": {
            "category": "Реверб и эхо",
            "full_name": "Mel-Band Roformer DeReverb-Echo Fused by Sucial",
            "stems": ["dry", "other"],
            "target_instrument": "dry",
            "checkpoint_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt?download=true",
            "config_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml?download=true"
        },

        "Mel-Band-Roformer_DeReverb-Echo_v2_sucial": {
            "category": "Реверб и эхо",
            "full_name": "Mel-Band Roformer DeReverb-Echo v2 by Sucial",
            "stems": ["dry", "other"],
            "target_instrument": "dry",
            "checkpoint_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb_echo_mbr_v2_sdr_dry_13.4843.ckpt?download=true",
            "config_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml?download=true"
        },

        "Mel-Band-Roformer_Karaoke_aufr33_viperx": {
            "category": "Караоке",
            "full_name": "Mel-Band Roformer Karaoke by Aufr33 & ViperX",
            "stems": ["karaoke", "other"],
            "target_instrument": "karaoke",
            "checkpoint_url": "https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt?download=true",
            "config_url": "https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/config_mel_band_roformer_karaoke.yaml?download=true"
        },

        "Mel-Band-Roformer_DeNoise_aufr33": {
            "category": "Шум",
            "full_name": "Mel-Band Roformer DeNoise by Aufr33",
            "stems": ["dry", "other"],
            "target_instrument": "dry",
            "checkpoint_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt?download=true",
            "config_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml?download=true"
        },

        "Mel-Band-Roformer_Denoise_Aggr_aufr33": {
            "category": "Шум",
            "full_name": "Mel-Band Roformer DeNoise Aggressive by Aufr33",
            "stems": ["dry", "other"],
            "target_instrument": "dry",
            "checkpoint_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt?download=true",
            "config_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml?download=true"
        },

        "Mel-Band-Roformer_Crowd_aufr33_viperx": {
            "category": "Звуки толпы",
            "full_name": "Mel-Band Roformer Crowd by Aufr33 & ViperX",
            "stems": ["crowd", "other"],
            "target_instrument": "crowd",
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt",
            "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/model_mel_band_roformer_crowd.yaml"
        },

        "Mel-Band-Roformer_Vocals_viperx": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Vocals by ViperX",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_mel_band_roformer_ep_3005_sdr_11.4360.yaml"
        },

        "Mel-Band-Roformer_Vocals_Fullness_aname": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Vocals Fullness by Aname",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/Aname-Tommy/MelBandRoformers/resolve/main/FullnessVocalModel.ckpt?download=true",
            "config_url": "https://huggingface.co/Aname-Tommy/MelBandRoformers/resolve/main/config.yaml?download=true"
        },

        "Mel-Band-Roformer_Kim_FT_v1_aname": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Kim FT v1 by Aname",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/Aname-Tommy/Test/resolve/main/model_kim.ckpt?download=true",
            "config_url": "https://huggingface.co/Aname-Tommy/MelBandRoformers/resolve/main/config.yaml?download=true"
        },

        "Mel-Band-Roformer_Kim_FT_v2_aname": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Kim FT v2 by Aname",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/Aname-Tommy/Test/resolve/main/model_kim_2.ckpt?download=true",
            "config_url": "https://huggingface.co/Aname-Tommy/MelBandRoformers/resolve/main/config.yaml?download=true"
        },

        "Mel-Band-Roformer_Kim_FT_v2_Fullness_aname": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Kim FT v2 Fullness by Aname",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/Aname-Tommy/Test/resolve/main/model_kim_2_fullness.ckpt?download=true",
            "config_url": "https://huggingface.co/Aname-Tommy/MelBandRoformers/resolve/main/config.yaml?download=true"
        },
        
        "Mel-Band-Roformer_Kim_FT_v3_aname": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Kim FT v3 by Aname",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/Aname-Tommy/Test/resolve/main/model_kim_3.ckpt?download=true",
            "config_url": "https://huggingface.co/Aname-Tommy/MelBandRoformers/resolve/main/config.yaml?download=true"
        },

        "Mel-Band-Roformer_kapm_aname": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer kapm by Aname",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/Aname-Tommy/Test/resolve/main/kapm.ckpt?download=true",
            "config_url": "https://huggingface.co/Aname-Tommy/Test/resolve/main/kapm.yaml?download=true"
        },
        
        "Mel-Band-Roformer_Small_aname": {
            "category": "Вокал",
            "full_name": "Mel-Band Roformer Small by Aname",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/Aname-Tommy/Mel_Band_Roformer_small/resolve/main/mel_band_roformer_small.ckpt?download=true",
            "config_url": "https://huggingface.co/Aname-Tommy/Mel_Band_Roformer_small/resolve/main/config.yaml?download=true"
        },

        "Mel-Band-Roformer_4_stems_Large_aname": {
            "category": "4 стема",
            "full_name": "Mel-Band Roformer 4 Stems Large by Aname",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/Aname-Tommy/melbandroformer4stems/resolve/main/mel_band_roformer_4stems_large_ver1.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/config.yaml?download=true"
        },
        
        "Mel-Band-Roformer_4_stems_v2_Large_aname": {
            "category": "4 стема",
            "full_name": "Mel-Band Roformer 4 Stems v2 Large by Aname",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/Aname-Tommy/Test/resolve/main/4stemsver2.ckpt?download=true",
            "config_url": "https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/config.yaml?download=true"
        },

        "Mel-Band-Roformer_4_stems_XL_aname": {
            "category": "4 стема",
            "full_name": "Mel-Band Roformer 4 Stems XL by Aname",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/Aname-Tommy/melbandroformer4stems/resolve/main/mel_band_roformer_4stems_xl_ver1.ckpt?download=true",
            "config_url": "https://huggingface.co/Aname-Tommy/melbandroformer4stems/resolve/main/config_xl.yaml?download=true"
        },
        
        "Mel-Band-Roformer_Drums_yolkispaliks": {
            "category": "Ударные",
            "full_name": "Mel-Band Roformer Drums Experimental by yolkispalkis",
            "stems": ["percussions", "other"],
            "target_instrument": "percussions",
            "checkpoint_url": "https://huggingface.co/am2460162/msst_failed_failed_test/resolve/main/model_mel_band_roformer_ep_11_sdr_7.6853.ckpt?download=true",
            "config_url": "https://huggingface.co/am2460162/msst_failed_failed_test/resolve/main/config_drums_musdb18_moises_mel_band_roformer.yaml?download=true"
        },

        "Mel-Band-Roformer_Instrumental_Metal_Preview_meskvlla33": {
            "category": "Инструментал",
            "full_name": "Mel-Band Roformer Metal Inst Preview by Mesk",
            "stems": ["vocals", "other"],
            "target_instrument": "other",
            "checkpoint_url": "https://huggingface.co/meskvlla33/metal_roformer_preview/resolve/main/metal_roformer_inst_mesk_preview.ckpt?download=true",
            "config_url": "https://huggingface.co/meskvlla33/metal_roformer_preview/resolve/main/config_inst_metal_roformer_mesk.yaml?download=true"
        }

    },

    "bs_roformer": {

        "BS-Roformer_Drums_beatloo_labs": {
            "category": "Ударные",
            "full_name": "BS Roformer Drums Experimental by BeatLoo Labs",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": "drums",
            "checkpoint_url": "https://huggingface.co/am2460162/msst_failed_failed_test/resolve/main/model_drums_bs_roformer_ep_12_sdr_7.2279.ckpt?download=true",
            "config_url": "https://huggingface.co/am2460162/msst_failed_failed_test/resolve/main/config_4_drums_bs_roformer.yaml?download=true"
        },

        "BS-Roformer_Bass_beatloo_labs": {
            "category": "Басс",
            "full_name": "BS Roformer Bass Experimental by BeatLoo Labs",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": "bass",
            "checkpoint_url": "https://huggingface.co/am2460162/msst_failed_failed_test/resolve/main/model_bass_bs_roformer_ep_10_sdr_5.7862.ckpt?download=true",
            "config_url": "https://huggingface.co/am2460162/msst_failed_failed_test/resolve/main/config_4_bass_bs_roformer.yaml?download=true"
        },

        "BS-Roformer_Vocals_1296_viperx": {
            "category": "Вокал",
            "full_name": "BS Roformer Vocals (sdr 12.96) by ViperX",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "Vocals",
            "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt",
            "config_url": "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml"
        },
        
        "BS-Roformer_Other_viperx": {
            "category": "Прочее",
            "full_name": "BS Roformer Other by ViperX",
            "stems": ["vocals", "other"],
            "target_instrument": "other",
            "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_937_sdr_10.5309.ckpt",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_937_sdr_10.5309.yaml"
        },

        "BS-Roformer_Revive_v1_unwa": {
            "category": "Вокал",
            "full_name": "BS Roformer Vocals Revive v1 by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/config.yaml?download=true"
        },

        "BS-Roformer_Revive_v2_unwa": {
            "category": "Вокал",
            "full_name": "BS Roformer Vocals Revive v2 by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive2.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/config.yaml?download=true"
        },

        "BS-Roformer_Revive_v3e_unwa": {
            "category": "Вокал",
            "full_name": "BS Roformer Vocals Revive v3e by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive3e.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/config.yaml?download=true"
        },


        "BS-Roformer_Resurrection_unwa": {
            "category": "Вокал",
            "full_name": "BS Roformer Vocals Resurrection by Unwa",
            "stems": ["vocals", "other"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection.ckpt?download=true",
            "config_url": "https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Config.yaml?download=true"
        },


        "BS-Roformer_VocTest_gabox": {
            "category": "Вокал",
            "full_name": "BS Roformer Vocals by GaboxR67",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": "vocals",
            "checkpoint_url": "https://huggingface.co/GaboxR67/BSRoformerVocTest/resolve/main/voc_gaboxBSR.ckpt?download=true",
            "config_url": "https://huggingface.co/GaboxR67/BSRoformerVocTest/resolve/main/voc_gaboxBSroformer.yaml?download=true"
        },

        "BS-Roformer_SW": {
            "category": "6 стемов",
            "full_name": "BS Roformer SW",
            "stems": ["bass", "drums", "other", "piano", "guitar", "vocals"],
            "target_instrument": None,
            "checkpoint_url": "https://github.com/undef13/splifft/releases/download/v0.0.1/roformer-fp16.pt",
            "config_url": "https://huggingface.co/noblebarkrr/all_models_for_mel_band_roformer/resolve/main/BS-Roformer_SW_config.yaml?download=true"
        },

        "BS-Roformer_4_stems_zfturbo": {
            "category": "4 стема",
            "full_name": "BS Roformer 4 Stems by ZFTurbo",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt",
            "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/config_bs_roformer_384_8_2_485100.yaml"
        },

        "BS-Roformer_4_stems_FT_syh99999": {
            "category": "4 стема",
            "full_name": "BS Roformer 4 Stems FT by SYH99999",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/SYH99999/bs_roformer_4stems_ft/resolve/main/bs_roformer_4stems_ft.pth?download=true",
            "config_url": "https://huggingface.co/SYH99999/bs_roformer_4stems_ft/resolve/main/config.yaml?download=true"
        },

        "BS-Roformer_Chorus_Male-Female_146_sucial": {
            "category": "Мужской/Женский вокал",
            "full_name": "BS Roformer Male-Female (ep 146) by Sucial",
            "stems": ["male", "female"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/model_chorus_bs_roformer_ep_146_sdr_23.8613.ckpt?download=true",
            "config_url": "https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/config_chorus_male_female_bs_roformer.yaml?download=true"
        },

        "BS-Roformer_Chorus_Male-Female_267_sucial": {
            "category": "Мужской/Женский вокал",
            "full_name": "BS Roformer Male-Female (ep 267) by Sucial",
            "stems": ["male", "female"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt?download=true",
            "config_url": "https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/config_chorus_male_female_bs_roformer.yaml?download=true"
        },

        "BS-Roformer_Male-Female_aufr33": {
            "category": "Мужской/Женский вокал",
            "full_name": "BS Roformer Male-Female by Aufr33",
            "stems": ["male", "female"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/RareSirMix/AIModelRehosting/resolve/main/bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt",
            "config_url": "https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/config_chorus_male_female_bs_roformer.yaml"
        },

        "BS-Roformer_Deverb_256_8_anvuew": {
            "category": "Реверб",
            "full_name": "BS Roformer Deverb 256-8 by Anvuew",
            "stems": ["reverb", "noreverb"],
            "target_instrument": "noreverb",
            "checkpoint_url": "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_256dim_8depth.ckpt?download=true",
            "config_url": "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_256dim_8depth.yaml?download=true"
        },

        "BS-Roformer_Deverb_384_10_anvuew": {
            "category": "Реверб",
            "full_name": "BS Roformer Deverb 384-10 by Anvuew",
            "stems": ["reverb", "noreverb"],
            "target_instrument": "noreverb",
            "checkpoint_url": "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_384dim_10depth.ckpt?download=true",
            "config_url": "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_384dim_10depth.yaml?download=true"
        },

        "BS-Roformer_4_stems_aname": {
            "category": "4 стема",
            "full_name": "BS Roformer 4 stems by Aname",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/RareSirMix/AIModelRehosting/resolve/main/Amane4stem_bs_roformer.ckpt",
            "config_url": "https://huggingface.co/RareSirMix/AIModelRehosting/resolve/main/Amane4stem_bs_roformer.yaml"
        }
    },

    "mdx23c": {

        "MDX23C_InstVoc_HQ_zfturbo": {
            "category": "Инструментал и вокал",
            "full_name": "MDX23C Inst-Voc HQ by ZFTurbo",
            "stems": ["vocals", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_mdx23c.yaml"
        },

        "MDX23C_8kFFT_InstVoc_HQ_v1": {
            "category": "Инструментал и вокал",
            "full_name": "MDX23C 8k FFT Inst-Voc HQ v1",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDX23C/MDX23C-8KFFT-InstVoc_HQ.ckpt?download=true",
            "config_url": "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDX23C/model_2_stem_full_band_8k.yaml?download=true"
        },

        "MDX23C_8kFFT_InstVoc_HQ_v2": {
            "category": "Инструментал и вокал",
            "full_name": "MDX23C 8k FFT Inst-Voc HQ v2",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDX23C/MDX23C-8KFFT-InstVoc_HQ_2.ckpt?download=true",
            "config_url": "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDX23C/model_2_stem_full_band_8k.yaml?download=true"
        },

        "MDX23C_D1581": {
            "category": "Инструментал и вокал",
            "full_name": "MDX23C D1581",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDX23C/MDX23C_D1581.ckpt?download=true",
            "config_url": "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDX23C/model_2_stem_061321.yaml?download=true"
        },


        "MDX23C_DrumSep_aufr33_jarredou": {
            "category": "DrumSep",
            "full_name": "MDX23C DrumSep by Aufr33 & Jarredou",
            "stems": ["kick", "snare", "toms", "hh", "ride", "crash"],
            "target_instrument": None,
            "checkpoint_url": "https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt",
            "config_url": "https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml"
        },

        "MDX23C_DeReverb_aufr33_jarredou": {
            "category": "Реверб",
            "full_name": "MDX23C DeReverb by Aufr33 & Jarredou",
            "stems": ["dry", "other"],
            "target_instrument": "dry",
            "checkpoint_url": "https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/dereverb_mdx23c_sdr_6.9096.ckpt",
            "config_url": "https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/config_dereverb_mdx23c.yaml"
        },

        "MDX23C_Mid_Side_wesleyr36": {
            "category": "Фантомный центр",
            "full_name": "MDX23C Mid-Side by WesleyR36",
            "stems": ["similarity", "difference"],
            "target_instrument": "similarity",
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.10/model_mdx23c_ep_271_l1_freq_72.2383.ckpt",
            "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.10/config_mdx23c_similarity.yaml"
        },

        "MDX23C_4_stems_zfturbo": {
            "category": "4 стема",
            "full_name": "MDX23C 4 Stems by ZFTurbo",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.1/model_mdx23c_ep_168_sdr_7.0207.ckpt",
            "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.1/config_musdb18_mdx23c.yaml"
        },

        "MDX23C_Orchestra_verosment": {
            "category": "Оркестр",
            "full_name": "MDX23C Orchestra Experimental by Verosment",
            "stems": ["inst", "orch"],
            "target_instrument": "orch",
            "checkpoint_url": "https://huggingface.co/am2460162/msst_failed_failed_test/resolve/main/model_mdx23c_ep_120_sdr_4.4174.ckpt?download=true",
            "config_url": "https://huggingface.co/am2460162/msst_failed_failed_test/resolve/main/config_orchestra_mdx23c.yaml?download=true"
        }

    },

    "scnet": {

        "SCNet_4_stems_zfturbo": {
            "category": "4 стема",
            "full_name": "SCNet 4 Stems by ZFTurbo",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/SCNet-large_starrytong_fixed.ckpt",
            "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/config_musdb18_scnet_large_starrytong.yaml"
        },

        "SCNet_XL_IHF_4_stems_zfturbo": {
            "category": "4 стема",
            "full_name": "SCNet XL IHF 4 Stems by ZFTurbo",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.15/model_scnet_ep_36_sdr_10.0891.ckpt",
            "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.15/config_musdb18_scnet_xl_more_wide_v5.yaml"
        },


        "SCNet_XL_4_stems_starrytong": {
            "category": "4 стема",
            "full_name": "SCNet 4 Stems XL by StarryTong",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/model_scnet_ep_54_sdr_9.8051.ckpt",
            "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/config_musdb18_scnet_xl.yaml"
        },

        "SCNet_XL_4_stems_zftrubo": {
            "category": "4 стема",
            "full_name": "SCNet 4 Stems XL by ZFTurbo",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/scnet_checkpoint_musdb18.ckpt",
            "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/config_musdb18_scnet.yaml"
        },

        "SCNet_Large_Jazz_4_stems_jorisvaneyghen": {
            "category": "4 стема",
            "full_name": "SCNet Large Jazz model by Joris Vaneyghen",
            "stems": ["piano", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/jorisvaneyghen/SCNet/resolve/main/model_jazz_scnet_large.ckpt?download=true",
            "config_url": "https://huggingface.co/spaces/jorisvaneyghen/jazz_playalong/resolve/main/configs/config_jazz_scnet_large.yaml?download=true"
        },

        "SCNet_XL_Jazz_4_stems_jorisvaneyghen": {
            "category": "4 стема",
            "full_name": "SCNet XL Jazz model by Joris Vaneyghen",
            "stems": ["piano", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/jorisvaneyghen/SCNet/resolve/main/model_jazz_scnet_xl.ckpt?download=true",
            "config_url": "https://huggingface.co/spaces/jorisvaneyghen/jazz_playalong/resolve/main/configs/config_jazz_scnet_xl.yaml?download=true"
        }

    },

    "vr": {

        "1_HP-UVR": {
            "category": "Инструментал",
            "full_name": "VR Arch Single Model v5: 1_HP-UVR",
            "stems": ["Instrumental", "Vocals"],
            "custom_vr": False,
            "target_instrument": None
        },

        "2_HP-UVR": {
            "category": "Инструментал",
            "full_name": "VR Arch Single Model v5: 2_HP-UVR",
            "stems": ["Instrumental", "Vocals"],
            "custom_vr": False,
            "target_instrument": None
        },

        "3_HP-Vocal-UVR": {
            "category": "Вокал",
            "full_name": "VR Arch Single Model v5: 3_HP-Vocal-UVR",
            "stems": ["Vocals", "Instrumental"],
            "custom_vr": False,
            "target_instrument": None
        },

        "4_HP-Vocal-UVR": {
            "category": "Вокал",
            "full_name": "VR Arch Single Model v5: 4_HP-Vocal-UVR",
            "stems": ["Vocals", "Instrumental"],
            "custom_vr": False,
            "target_instrument": None
        },

        "5_HP-Karaoke-UVR": {
            "category": "Караоке",
            "full_name": "VR Arch Single Model v5: 5_HP-Karaoke-UVR",
            "stems": ["Instrumental", "Vocals"],
            "custom_vr": False,
            "target_instrument": None
        },

        "6_HP-Karaoke-UVR": {
            "category": "Караоке",
            "full_name": "VR Arch Single Model v5: 6_HP-Karaoke-UVR",
            "stems": ["Instrumental", "Vocals"],
            "custom_vr": False,
            "target_instrument": None
        },

        "7_HP2-UVR": {
            "category": "Инструментал и вокал",
            "full_name": "VR Arch Single Model v5: 7_HP2-UVR",
            "stems": ["Instrumental", "Vocals"],
            "custom_vr": False,
            "target_instrument": None
        },

        "8_HP2-UVR": {
            "category": "Инструментал и вокал",
            "full_name": "VR Arch Single Model v5: 8_HP2-UVR",
            "stems": ["Instrumental", "Vocals"],
            "custom_vr": False,
            "target_instrument": None
        },

        "9_HP2-UVR": {
            "category": "Инструментал и вокал",
            "full_name": "VR Arch Single Model v5: 9_HP2-UVR",
            "stems": ["Instrumental", "Vocals"],
            "custom_vr": False,
            "target_instrument": None
        },

        "10_SP-UVR-2B-32000-1": {
            "category": "Инструментал и вокал",
            "full_name": "VR Arch Single Model v5: 10_SP-UVR-2B-32000-1",
            "stems": ["Instrumental", "Vocals"],
            "custom_vr": False,
            "target_instrument": None
        },

        "11_SP-UVR-2B-32000-2": {
            "category": "Инструментал и вокал",
            "full_name": "VR Arch Single Model v5: 11_SP-UVR-2B-32000-2",
            "stems": ["Instrumental", "Vocals"],
            "custom_vr": False,
            "target_instrument": None
        },

        "12_SP-UVR-3B-44100": {
            "category": "Инструментал и вокал",
            "full_name": "VR Arch Single Model v5: 12_SP-UVR-3B-44100",
            "stems": ["Instrumental", "Vocals"],
            "custom_vr": False,
            "target_instrument": None
        },

        "13_SP-UVR-4B-44100-1": {
            "category": "Инструментал и вокал",
            "full_name": "VR Arch Single Model v5: 13_SP-UVR-4B-44100-1",
            "stems": ["Instrumental", "Vocals"],
            "custom_vr": False,
            "target_instrument": None
        },

        "14_SP-UVR-4B-44100-2": {
            "category": "Инструментал и вокал",
            "full_name": "VR Arch Single Model v5: 14_SP-UVR-4B-44100-2",
            "stems": ["Instrumental", "Vocals"],
            "custom_vr": False,
            "target_instrument": None
        },

        "15_SP-UVR-MID-44100-1": {
            "category": "Инструментал и вокал",
            "full_name": "VR Arch Single Model v5: 15_SP-UVR-MID-44100-1",
            "stems": ["Instrumental", "Vocals"],
            "custom_vr": False,
            "target_instrument": None
        },

        "16_SP-UVR-MID-44100-2": {
            "category": "Инструментал и вокал",
            "full_name": "VR Arch Single Model v5: 16_SP-UVR-MID-44100-2",
            "stems": ["Instrumental", "Vocals"],
            "custom_vr": False,
            "target_instrument": None
        },

        "17_HP-Wind_Inst-UVR": {
            "category": "Деревянные духовые",
            "full_name": "VR Arch Single Model v5: 17_HP-Wind_Inst-UVR",
            "stems": ["No Woodwinds", "Woodwinds"],
            "custom_vr": False,
            "target_instrument": None
        },

        "UVR-De-Echo-Aggressive": {
            "category": "Эхо",
            "full_name": "VR Arch Single Model v5: UVR-De-Echo-Aggressive by FoxJoy",
            "stems": ["No Echo", "Echo"],
            "custom_vr": False,
            "target_instrument": None
        },

        "UVR-De-Echo-Normal": {
            "category": "Эхо",
            "full_name": "VR Arch Single Model v5: UVR-De-Echo-Normal by FoxJoy",
            "stems": ["No Echo", "Echo"],
            "custom_vr": False,
            "target_instrument": None
        },

        "UVR-DeEcho-DeReverb": {
            "category": "Реверб",
            "full_name": "VR Arch Single Model v5: UVR-DeEcho-DeReverb by FoxJoy",
            "stems": ["No Reverb", "Reverb"],
            "custom_vr": False,
            "target_instrument": None
        },

        "UVR-DeNoise-Lite": {
            "category": "Шум",
            "full_name": "VR Arch Single Model v5: UVR-DeNoise-Lite by FoxJoy",
            "stems": ["Noise", "No Noise"],
            "custom_vr": False,
            "target_instrument": None
        },

        "UVR-DeNoise": {
            "category": "Шум",
            "full_name": "VR Arch Single Model v5: UVR-DeNoise by FoxJoy",
            "stems": ["Noise", "No Noise"],
            "custom_vr": False,
            "target_instrument": None
        },

        "UVR-BVE-4B_SN-44100-1": {
            "category": "Караоке",
            "full_name": "VR Arch Single Model v5: UVR-BVE-4B_SN-44100",
            "stems": ["Vocals", "Instrumental"],
            "custom_vr": False,
            "target_instrument": None
        },

        "UVR-BVE-v2-4B-SN-44100": {
            "category": "Караоке",
            "full_name": "VR Arch Single Model v4: UVR-BVE-v2-4B-SN-44100",
            "stems": ["Vocals", "Instrumental"],
            "custom_vr": True,
            "primary_stem": "Vocals",
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/RareSirMix/AIModelRehosting/resolve/main/UVR-5-1_4band_v4_ms_fullband_BVE_v2_by_aufr33.pth?download=true",
            "config_url": "https://huggingface.co/RareSirMix/AIModelRehosting/resolve/main/4band_v4_ms_fullband.yaml?download=true"
        },

        "MGM-v5-KAROKEE-32000-BETA1": {
            "category": "Караоке",
            "full_name": "VR Arch Single Model v5: MGM-v5-KAROKEE-32000-BETA1",
            "stems": ["Vocals", "Instrumental"],
            "custom_vr": True,
            "primary_stem": "Instrumental",
            "target_instrument": None,
            "checkpoint_url": "https://github.com/lucassantilli/UVR-Colab-GUI/releases/download/m5.1/MGM-v5-KAROKEE-32000-BETA1.pth",
            "config_url": "https://github.com/lucassantilli/UVR-Colab-GUI/raw/refs/heads/main/modelparams/2band_32000.json"
        },

        "MGM-v5-KAROKEE-32000-BETA2-AGR": {
            "category": "Караоке",
            "full_name": "VR Arch Single Model v5: MGM-v5-KAROKEE-32000-BETA2-AGR.pth",
            "stems": ["Vocals", "Instrumental"],
            "custom_vr": True,
            "primary_stem": "Instrumental",
            "target_instrument": None,
            "checkpoint_url": "https://github.com/lucassantilli/UVR-Colab-GUI/releases/download/m5.1/MGM-v5-KAROKEE-32000-BETA2-AGR.pth",
            "config_url": "https://github.com/lucassantilli/UVR-Colab-GUI/raw/refs/heads/main/modelparams/2band_32000_agr.json"
        },




        "MGM_HIGHEND_v4": {
            "category": "Инструментал и вокал",
            "full_name": "VR Arch Single Model v4: MGM_HIGHEND_v4",
            "stems": ["Instrumental", "Vocals"],
            "custom_vr": False,
            "target_instrument": None
        },

        "MGM_LOWEND_A_v4": {
            "category": "Инструментал и вокал",
            "full_name": "VR Arch Single Model v4: MGM_LOWEND_A_v4",
            "stems": ["Instrumental", "Vocals"],
            "custom_vr": False,
            "target_instrument": None
        },

        "MGM_LOWEND_B_v4": {
            "category": "Инструментал и вокал",
            "full_name": "VR Arch Single Model v4: MGM_LOWEND_B_v4",
            "stems": ["Instrumental", "Vocals"],
            "custom_vr": False,
            "target_instrument": None
        },

        "MGM_MAIN_v4": {
            "category": "Инструментал и вокал",
            "full_name": "VR Arch Single Model v4: MGM_MAIN_v4",
            "stems": ["Instrumental", "Vocals"],
            "custom_vr": False,
            "target_instrument": None
        },

        "UVR-De-Reverb-aufr33-jarredou": {
            "category": "Реверб",
            "full_name": "VR Arch Single Model v4: UVR-De-Reverb by aufr33-jarredou",
            "stems": ["Dry", "No Dry"],
            "custom_vr": False,
            "target_instrument": None
        },

        "UVR-De-Breath-sucial-v1": {
            "category": "Дыхание",
            "full_name": "VR Arch Single Model v4: UVR-De-Breath v1 by Sucial",
            "stems": ["Breath", "No Breath"],
            "custom_vr": True,
            "primary_stem": "Breath",
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/Sucial/De-Breathe-Models/resolve/main/UVR_De-Breathe_1band_sr44100_hl1024_Sucial_v1.pth?download=true",
            "config_url": "https://huggingface.co/Sucial/De-Breathe-Models/resolve/main/1band_sr44100_hl1024.json?download=true"
        },

        "UVR-De-Breath-sucial-v2": {
            "category": "Дыхание",
            "full_name": "VR Arch Single Model v4: UVR-De-Breath v2 by Sucial",
            "stems": ["Breath", "No Breath"],
            "custom_vr": True,
            "primary_stem": "Breath",
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/Sucial/De-Breathe-Models/resolve/main/UVR_De-Breathe_1band_sr44100_hl1024_Sucial_v2.pth?download=true",
            "config_url": "https://huggingface.co/Sucial/De-Breathe-Models/resolve/main/1band_sr44100_hl1024.json?download=true"
        },

        "VR_Harmonic_Noise_Sep": {
            "category": "Дыхание",
            "full_name": "VR Arch Single Model v5: Harmonic_Noise_Sep",
            "stems": ["Noise", "No Noise"],
            "custom_vr": True,
            "primary_stem": "Noise",
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/Sucial/MSST-WebUI/resolve/main/All_Models/VR_Models/Harmonic_Noise_Separation_yxlllc.pth?download=true",
            "config_url": "https://github.com/SUC-DriverOld/MSST-WebUI/raw/refs/heads/main/configs_backup/vr_modelparams/1band_sr44100_hl1024.json"
        }

    },

    "mdx": {

        "UVR-MDX-NET-Inst_HQ_1": {
            "category": "Инструментал",
            "full_name": "MDX-Net Model: UVR-MDX-NET Inst HQ 1",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": None
        },

        "UVR-MDX-NET-Inst_HQ_2": {
            "category": "Инструментал",
            "full_name": "MDX-Net Model: UVR-MDX-NET Inst HQ 2",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": None
        },

        "UVR-MDX-NET-Inst_HQ_3": {
            "category": "Инструментал",
            "full_name": "MDX-Net Model: UVR-MDX-NET Inst HQ 3",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": None
        },

        "UVR-MDX-NET-Inst_HQ_4": {
            "category": "Инструментал",
            "full_name": "MDX-Net Model: UVR-MDX-NET Inst HQ 4",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": None
        },

        "UVR-MDX-NET-Inst_HQ_5": {
            "category": "Инструментал",
            "full_name": "MDX-Net Model: UVR-MDX-NET Inst HQ 5",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": None
        },

        "UVR_MDXNET_Main": {
            "category": "Инструментал и вокал",
            "full_name": "MDX-Net Model: UVR-MDX-NET Main",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None
        },

        "UVR-MDX-NET-Inst_Main": {
            "category": "Инструментал",
            "full_name": "MDX-Net Model: UVR-MDX-NET Inst Main",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": None
        },

        "UVR_MDXNET_1_9703": {
            "category": "Инструментал и вокал",
            "full_name": "MDX-Net Model: UVR-MDX-NET 1",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None
        },

        "UVR_MDXNET_2_9682": {
            "category": "Инструментал и вокал",
            "full_name": "MDX-Net Model: UVR-MDX-NET 2",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None
        },

        "UVR_MDXNET_3_9662": {
            "category": "Инструментал и вокал",
            "full_name": "MDX-Net Model: UVR-MDX-NET 3",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None
        },

        "UVR-MDX-NET-Inst_1": {
            "category": "Инструментал",
            "full_name": "MDX-Net Model: UVR-MDX-NET Inst 1",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": None
        },

        "UVR-MDX-NET-Inst_2": {
            "category": "Инструментал",
            "full_name": "MDX-Net Model: UVR-MDX-NET Inst 2",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": None
        },

        "UVR-MDX-NET-Inst_3": {
            "category": "Инструментал",
            "full_name": "MDX-Net Model: UVR-MDX-NET Inst 3",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": None
        },

        "UVR_MDXNET_KARA": {
            "category": "Караоке",
            "full_name": "MDX-Net Model: UVR-MDX-NET Karaoke",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None
        },

        "UVR_MDXNET_KARA_2": {
            "category": "Караоке",
            "full_name": "MDX-Net Model: UVR-MDX-NET Karaoke 2",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": None
        },

        "UVR_MDXNET_9482": {
            "category": "Инструментал и вокал",
            "full_name": "MDX-Net Model: UVR_MDXNET_9482",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None
        },

        "UVR-MDX-NET-Voc_FT": {
            "category": "Вокал",
            "full_name": "MDX-Net Model: UVR-MDX-NET Voc FT",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None
        },

        "Kim_Vocal_1": {
            "category": "Вокал",
            "full_name": "MDX-Net Model: Kim Vocal 1",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None
        },

        "Kim_Vocal_2": {
            "category": "Вокал",
            "full_name": "MDX-Net Model: Kim Vocal 2",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None
        },

        "Kim_Inst": {
            "category": "Инструментал",
            "full_name": "MDX-Net Model: Kim Inst",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": None
        },

        "Reverb_HQ_By_FoxJoy": {
            "category": "Реверб",
            "full_name": "MDX-Net Model: Reverb HQ By FoxJoy",
            "stems": ["Reverb", "No Reverb"],
            "target_instrument": None
        },

        "UVR-MDX-NET_Crowd_HQ_1": {
            "category": "Звуки толпы",
            "full_name": "MDX-Net Model: UVR-MDX-NET Crowd HQ 1 By Aufr33",
            "stems": ["No Crowd", "Crowd"],
            "target_instrument": None
        },

        "kuielab_a_vocals": {
            "category": "Вокал",
            "full_name": "MDX-Net Model: kuielab_a_vocals",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None
        },

        "kuielab_a_other": {
            "category": "Прочее",
            "full_name": "MDX-Net Model: kuielab_a_other",
            "stems": ["Other", "No Other"],
            "target_instrument": None
        },

        "kuielab_a_bass": {
            "category": "Басс",
            "full_name": "MDX-Net Model: kuielab_a_bass",
            "stems": ["Bass", "No Bass"],
            "target_instrument": None
        },

        "kuielab_a_drums": {
            "category": "Ударные",
            "full_name": "MDX-Net Model: kuielab_a_drums",
            "stems": ["Drums", "No Drums"],
            "target_instrument": None
        },

        "kuielab_b_vocals": {
            "category": "Вокал",
            "full_name": "MDX-Net Model: kuielab_b_vocals",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None
        },

        "kuielab_b_other": {
            "category": "Прочее",
            "full_name": "MDX-Net Model: kuielab_b_other",
            "stems": ["Other", "No Other"],
            "target_instrument": None
        },

        "kuielab_b_bass": {
            "category": "Басс",
            "full_name": "MDX-Net Model: kuielab_b_bass",
            "stems": ["Bass", "No Bass"],
            "target_instrument": None
        },

        "kuielab_b_drums": {
            "category": "Ударные",
            "full_name": "MDX-Net Model: kuielab_b_drums",
            "stems": ["Drums", "No Drums"],
            "target_instrument": None
        },

        "UVR-MDX-NET_Main_340": {
            "category": "Инструментал и вокал",
            "full_name": "MDX-Net Model VIP: UVR-MDX-NET_Main_340",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None
        },

        "UVR-MDX-NET_Main_390": {
            "category": "Инструментал и вокал",
            "full_name": "MDX-Net Model VIP: UVR-MDX-NET_Main_390",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None
        },

        "UVR-MDX-NET_Main_406": {
            "category": "Инструментал и вокал",
            "full_name": "MDX-Net Model VIP: UVR-MDX-NET_Main_406",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None
        },

        "UVR-MDX-NET_Main_427": {
            "category": "Инструментал и вокал",
            "full_name": "MDX-Net Model VIP: UVR-MDX-NET_Main_427",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None
        },

        "UVR-MDX-NET_Main_438": {
            "category": "Инструментал и вокал",
            "full_name": "MDX-Net Model VIP: UVR-MDX-NET_Main_438",
            "stems": ["Vocals", "Instrumental"],
            "target_instrument": None
        },

        "UVR-MDX-NET_Inst_82_beta": {
            "category": "Инструментал",
            "full_name": "MDX-Net Model VIP: UVR-MDX-NET_Inst_82_beta",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": None
        },

        "UVR-MDX-NET_Inst_90_beta": {
            "category": "Инструментал",
            "full_name": "MDX-Net Model VIP: UVR-MDX-NET_Inst_90_beta",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": None
        },

        "UVR-MDX-NET_Inst_187_beta": {
            "category": "Инструментал",
            "full_name": "MDX-Net Model VIP: UVR-MDX-NET_Inst_187_beta",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": None
        },

        "UVR-MDX-NET-Inst_full_292": {
            "category": "Инструментал",
            "full_name": "MDX-Net Model VIP: UVR-MDX-NET-Inst_full_292",
            "stems": ["Instrumental", "Vocals"],
            "target_instrument": None
        }

    },

    "htdemucs": {

        "HTDemucs4_MVSep_vocals": {
            "category": "Вокал",
            "full_name": "HTDemucs4 (MVSep finetuned)",
            "stems": ["vocals", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_htdemucs_sdr_8.78.ckpt",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_htdemucs.yaml"
        },

        "HTDemucs4": {
            "category": "4 стема",
            "full_name": "HTDemucs4",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml"
        },

        "HTDemucs4_6s": {
            "category": "6 стемов",
            "full_name": "HTDemucs4 (6 stems)",
            "stems": ["vocals", "drums", "bass", "other", "guitar", "piano"],
            "target_instrument": None,
            "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_htdemucs_6stems.yaml"
        },

        "Demucs3_mmi": {
            "category": "4 стема",
            "full_name": "Demucs3 mmi",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/75fc33f5-1941ce65.th",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_demucs3_mmi.yaml"
        },

        "HTDemucs4_FT_Bass": {
            "category": "Басс",
            "full_name": "HTDemucs4 FT Bass",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/d12395a8-e57c48e6.th",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml"
        },

        "HTDemucs4_FT_Drums": {
            "category": "Ударные",
            "full_name": "HTDemucs4 FT Drums",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml"
        },

        "HTDemucs4_FT_Vocals": {
            "category": "Вокал",
            "full_name": "HTDemucs4 FT Vocals",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/04573f0d-f3cf25b2.th",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml"
        },

        "HTDemucs4_FT_Other": {
            "category": "Прочее",
            "full_name": "HTDemucs4 FT Other",
            "stems": ["vocals", "drums", "bass", "other"],
            "target_instrument": None,
            "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/92cfc3b6-ef3bcb9c.th",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml"
        },

        "HTDemucs4_Mid_Side_wesleyr36": {
            "category": "Фантомный центр",
            "full_name": "HTDemucs4 MId-Side by wesleyr36",
            "stems": ["similarity", "difference"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/jarredou/HTDemucs_Similarity_Extractor_by_wesleyr36/resolve/main/model_htdemucs_ep_21_sdr_13.6970.ckpt?download=true",
            "config_url": "https://huggingface.co/jarredou/HTDemucs_Similarity_Extractor_by_wesleyr36/resolve/main/config_htdemucs_similarity.yaml?download=true"
        }

    },

    "bandit": {

        "Bandit_Plus": {
            "category": "Кинематограф",
            "full_name": "Bandit Plus: Cinematic Bandit Plus (by kwatcharasupat)",
            "stems": ["speech", "music", "effects"],
            "target_instrument": None,
            "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/model_bandit_plus_dnr_sdr_11.47.chpt",
            "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/config_dnr_bandit_bsrnn_multi_mus64.yaml"
        },

    },

    "bandit_v2": {

        "Bandit_v2_Multi": {
            "category": "Кинематограф",
            "full_name": "Bandit v2: Cinematic Bandit v2 Multilang (by kwatcharasupat)",
            "stems": ["speech", "music", "sfx"],
            "target_instrument": None,
            "checkpoint_url": "https://huggingface.co/jarredou/banditv2_state_dicts_only/resolve/main/checkpoint-multi_state_dict.ckpt",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/refs/heads/main/configs/config_dnr_bandit_v2_mus64.yaml"
        },

    }

}

medley_vox_models = {

    "multi_singing_librispeech": {
        "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/multi_singing_librispeech/vocals.pth?download=true",
        "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/multi_singing_librispeech/vocals.json?download=true"
    },

    "multi_singing_librispeech_138": {
        "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/multi_singing_librispeech_138/vocals.pth?download=true",
        "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/multi_singing_librispeech_138/vocals.json?download=true"
    },

    "singing_librispeech_ft_isrnet": {
        "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/singing_librispeech_ft_iSRNet/vocals.pth?download=true",
        "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/singing_librispeech_ft_iSRNet/vocals.json?download=true"
    },

    "singing_librispeech_isrnet": {
        "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/singing_librispeech_iSRNet/vocals.pth?download=true",
        "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/singing_librispeech_iSRNet/vocals.json?download=true"
    },

    "vocal_231": {
        "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocal%20231/vocals.pth?download=true",
        "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocal%20231/vocals.json?download=true"
    },

    "vocals_135": {
        "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20135/vocals.pth?download=true",
        "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20135/vocals.json?download=true"
    },

    "vocals_163": {
        "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20163/vocals.pth?download=true",
        "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20163/vocals.json?download=true"
    },

    "vocals_188": {
        "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20188/vocals.pth?download=true",
        "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20188/vocals.json?download=true"
    },

    "vocals_200": {
        "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20200/vocals.pth?download=true",
        "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20200/vocals.json?download=true"
    },

    "vocals_238": {
        "checkpoint_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20238/vocals.pth?download=true",
        "config_url": "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20238/vocals.json?download=true"
    }

}











