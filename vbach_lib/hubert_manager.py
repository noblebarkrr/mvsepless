from pathlib import Path
from extra_utils import dw_file
from i18n import _i18n

BASE_DIR = Path(__file__).resolve().parent / "embedders"
BASE_DIR.mkdir(parents=True, exist_ok=True)
FAIRSEQ_DIR = BASE_DIR / "fairseq"
TRANSFORMERS_DIR = BASE_DIR / "transformers"

class HubertModelNotExist(Exception): pass

huberts_fairseq_dict = {
    "hubert_base": {
        "url": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/fairseq/hubert_base.pt?download=true",
        "local_path": FAIRSEQ_DIR / "hubert_base.pt"
    },
    "contentvec_base": {
        "url": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/fairseq/contentvec_base.pt?download=true",
        "local_path": FAIRSEQ_DIR / "contentvec_base.pt"
    },
    "korean_hubert_base": {
        "url": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/fairseq/korean_hubert_base.pt?download=true",
        "local_path": FAIRSEQ_DIR / "korean_hubert_base.pt"
    },
    "chinese_hubert_base": {
        "url": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/fairseq/chinese_hubert_base.pt?download=true",
        "local_path": FAIRSEQ_DIR / "chinese_hubert_base.pt"
    },
    "portuguese_hubert_base": {
        "url": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/fairseq/portuguese_hubert_base.pt?download=true",
        "local_path": FAIRSEQ_DIR / "portuguese_hubert_base.pt"
    },
    "japanese_hubert_base": {
        "url": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/fairseq/japanese_hubert_base.pt?download=true",
        "local_path": FAIRSEQ_DIR / "japanese_hubert_base.pt"
    }
}

huberts_transformers_dict = {
    "contentvec": {
        "base_dir": TRANSFORMERS_DIR / "contentvec",
        "url_bin": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/contentvec/pytorch_model.bin?download=true",
        "url_json": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/contentvec/config.json?download=true"
    },
    "spin": {
        "base_dir": TRANSFORMERS_DIR / "spin",
        "url_bin": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/spin/pytorch_model.bin?download=true",
        "url_json": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/spin/config.json?download=true"
    },
    "spin-v2": {
        "base_dir": TRANSFORMERS_DIR / "spinv2",
        "url_bin": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/spinv2/pytorch_model.bin?download=true",
        "url_json": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/spinv2/config.json?download=true"
    },
    "chinese-hubert-base": {
        "base_dir": TRANSFORMERS_DIR / "chinese_hubert_base",
        "url_bin": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/chinese_hubert_base/pytorch_model.bin?download=true",
        "url_json": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/chinese_hubert_base/config.json?download=true"
    },
    "japanese-hubert-base": {
        "base_dir": TRANSFORMERS_DIR / "japanese_hubert_base",
        "url_bin": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/japanese_hubert_base/pytorch_model.bin?download=true",
        "url_json": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/japanese_hubert_base/config.json?download=true"
    },
    "korean-hubert-base": {
        "base_dir": TRANSFORMERS_DIR / "korean_hubert_base",
        "url_bin": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/korean_hubert_base/pytorch_model.bin?download=true",
        "url_json": "https://huggingface.co/noblebarkrr/vbach_resources/resolve/main/transformers/korean_hubert_base/config.json?download=true"
    },
}

huberts_fairseq = list(huberts_fairseq_dict.keys())
huberts_transformers = list(huberts_transformers_dict.keys())

def download_hubert(name: str, use_transformers: bool = False):
    if use_transformers:
        info = huberts_transformers_dict.get(name, {})
        if not info:
            raise HubertModelNotExist(_i18n("vbach_embedder_not"))
        base_dir = info["base_dir"]
        base_dir.mkdir(parents=True, exist_ok=True)
        
        urls = (info["url_bin"], info["url_json"])
        paths = (base_dir / "pytorch_model.bin", base_dir / "config.json")
        
        # Проверяем оба файла
        need_download = [not p.exists() for p in paths]
        if any(need_download):
            for url, path, need in zip(urls, paths, need_download):
                if need:
                    dw_file(url, path)
    else:
        info = huberts_fairseq_dict.get(name, {})
        if not info:
            HubertModelNotExist(_i18n("vbach_embedder_not"))
        url = info.get("url")
        path = info.get("local_path")
        if not path.exists():
            dw_file(url, path)

def get_hubert(name: str, use_transformers: bool = False):
    if use_transformers:
        info = huberts_transformers_dict.get(name, {})
        if not info:
            HubertModelNotExist(_i18n("vbach_embedder_not"))
        download_hubert(name, True)
        return info.get("base_dir")
    else:
        info = huberts_fairseq_dict.get(name, {})
        if not info:
            HubertModelNotExist(_i18n("vbach_embedder_not"))
        download_hubert(name, False)
        return info.get("local_path")