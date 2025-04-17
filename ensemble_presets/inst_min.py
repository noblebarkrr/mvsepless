# Type ensemble

def preset():

    type = "min_fft"

    # Weights

    weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    # List models in ensemble

    list_models = [
        ('mel_band_roformer', 'unwa_instrumental_v1e_plus'),
        ('mel_band_roformer', 'gaboxr67_instrumental_fvx'),
        ('mel_band_roformer', 'unwa_instrumental_v1e'),
        ('mel_band_roformer', 'becruily_instrumental'),
        ('mel_band_roformer', 'unwa_instrumental_v2'),
        ('mel_band_roformer', 'gaboxr67_instrumental_fv6_noise'),
        ('mel_band_roformer', 'gaboxr67_instrumental_bv2')
    ]

    # Select stem

    target_stem = "instrumental"
    
    ext_inst = False

    return target_stem, weights, list_models, type, ext_inst