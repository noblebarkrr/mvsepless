from separator import MvseplessModelManager
from downloader import dw_file
file_path = MvseplessModelManager().models_info_path
url_link = "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/models.json?download=true"
dw_file(url_link, file_path, retries=999999)
