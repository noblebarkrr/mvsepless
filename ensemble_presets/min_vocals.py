# Type ensemble

def preset():

    type = "min_fft"

    # Weights

    weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    # List models in ensemble

    list_models = [
        ('mel_band_roformer', 'viperx_vocals'),
        ('mel_band_roformer', 'kimberlyjsn_vocals'),
        ('mel_band_roformer', 'unwa_instrumental_v1e'),
        ('mel_band_roformer', 'unwa_vocals_big_beta_v5e'),
        ('bs_roformer', 'unwa_kim_ft_v2_bleedless'),
        ('bs_roformer', 'gaboxr67_vocals_fv4'),
        ('bs_roformer', 'aname_vocals_fullness')
    ]

    # Select stem

    target_stem = "vocals"
    
    ext_inst = False

    return target_stem, weights, list_models, type, ext_inst