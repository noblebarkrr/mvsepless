model_code = 600

# Multi Singing Librispeech
if model_code == 600:
    pth_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/multi_singing_librispeech/vocals.pth?download=true"
    json_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/multi_singing_librispeech/vocals.json?download=true"

# Multi Singing Librispeech 138
elif model_code == 601:
    pth_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/multi_singing_librispeech_138/vocals.pth?download=true"
    json_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/multi_singing_librispeech_138/vocals.json?download=true"

# Singing Librispeech FT ISRNET
elif model_code == 602:
    pth_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/singing_librispeech_ft_iSRNet/vocals.pth?download=true"
    json_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/singing_librispeech_ft_iSRNet/vocals.json?download=true"

# Singing Librispeech ISRNET
elif model_code == 603:
    pth_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/singing_librispeech_iSRNet/vocals.pth?download=true"
    json_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/singing_librispeech_iSRNet/vocals.json?download=true"

# Vocal 231
elif model_code == 604:
    pth_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocal%20231/vocals.pth?download=true"
    json_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocal%20231/vocals.json?download=true"

# Vocals 135
elif model_code == 605:
    pth_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20135/vocals.pth?download=true"
    json_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20135/vocals.json?download=true"

# Vocals 163
elif model_code == 606:
    pth_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20163/vocals.pth?download=true"
    json_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20163/vocals.json?download=true"

# Vocals 188
elif model_code == 607:
    pth_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20188/vocals.pth?download=true"
    json_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20188/vocals.json?download=true"

# Vocals 200
elif model_code == 608:
    pth_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20200/vocals.pth?download=true"
    json_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20200/vocals.json?download=true"

# Vocals 238
elif model_code == 609:
    pth_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20238/vocals.pth?download=true"
    json_url = "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20238/vocals.json?download=true"

# Вывод URL-адресов для загрузки
print(f"Model Code: {model_code}")
print(f"PTH URL: {pth_url}")
print(f"JSON URL: {json_url}")