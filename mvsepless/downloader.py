import os
import sys
import json
import yaml
from tabulate import tabulate
import shutil
from tqdm import tqdm
import urllib.request
import tempfile
import argparse
import time
import yt_dlp
from typing import Dict, Any

script_dir = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.environ.get(
    "MVSEPLESS_DOWNLOAD_DIR", os.path.join(os.getcwd(), "downloaded")
)

def dw_file(url_model: str, local_path: str, retries: int = 180):
    dir_name = os.path.dirname(local_path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)

    class TqdmUpTo(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    for attempt in range(retries):
        try:
            with TqdmUpTo(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc=os.path.basename(local_path),
            ) as t:
                urllib.request.urlretrieve(
                    url_model, local_path, reporthook=t.update_to
                )
            break
        except Exception as e:
            print(f"Попытка {attempt + 1}/{retries} не удалась. Ошибка: {e}")
            if attempt < retries - 1:
                print("Повторная попытка...")
                time.sleep(2)
            else:
                print("Все попытки загрузки завершились неудачно")
                raise

def dw_yt_dlp(
    url,
    output_dir=None,
    cookie=None,
    output_format="mp3",
    output_bitrate="320",
    title=None,
):
    outtmpl = "%(title)s.%(ext)s" if title is None else f"{title}.%(ext)s"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(
            DOWNLOAD_DIR if not output_dir else output_dir, outtmpl
        ),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": output_format,
                "preferredquality": output_bitrate,
            }
        ],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    if cookie and os.path.exists(cookie):
        ydl_opts["cookiefile"] = cookie

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            if "_type" in info and info["_type"] == "playlist":
                entry = info["entries"][0]
                filename = ydl.prepare_filename(entry)
            else:
                filename = ydl.prepare_filename(info)

            base, _ = os.path.splitext(filename)
            audio_file = base + f".{output_format}"

            return os.path.join(DOWNLOAD_DIR, audio_file)
        except Exception as e:
            return None