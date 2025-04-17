# Type ensemble

def preset():

    type = "max_fft"

    # Weights

    weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # List models in ensemble

    list_models = [
        ('mel_band_roformer', 'aname_4_stems_xl'),
        ('mel_band_roformer', 'aname_4_stems_large'),
        ('mel_band_roformer', 'syh99999_4_stems_ft_large'),
        ('mel_band_roformer', 'syh99999_4_stems_ft_large_v2'),
        ('bs_roformer', 'zf_turbo_4_stems'),
        ('bs_roformer', 'aname_4_stems'),
        ('bs_roformer', 'syh99999_4_stems_ft')
    ]

    # Select stem

    target_stem = "bass"
    
    ext_inst = False

    return target_stem, weights, list_models, type, ext_inst