import os
import requests
from mimetypes import guess_extension
from pathlib import Path
from filecmp import dircmp
import yt_dlp
import magic
import random
import gradio as gr
import string
import shutil
import re
    
    
def download_audio_from_url(url_or_text, type_url, cookies, output_dir="downloads"):
    def extract_url_from_text(text):
        """Ищет ссылку в произвольном тексте"""
        url_regex = r"https?://[^\s]+"
        match = re.search(url_regex, text)
        if match:
            return match.group(0)
        return None
    if type_url == "YT Music, Soundcloud, Tiktok":
        def is_supported_url(url):
            """Проверка, что это ссылка с нужного сайта"""
            return any(domain in url for domain in ["soundcloud.com", "youtube.com", "youtu.be", "tiktok.com"])
        def download_track(url_or_text, cookies, output_dir="downloads"):
            """Скачивает трек из TikTok, YouTube или SoundCloud"""
            os.makedirs(output_dir, exist_ok=True)
            # Проверка и извлечение URL
            if "http" not in url_or_text:
                gr.Warning("Нет ссылки в строке")
                return None
            url = extract_url_from_text(url_or_text)
            if not url:
                gr.Warning("Ссылка не найдена")
                return None
            if not is_supported_url(url):
                gr.Warning(f"Сайт не поддерживается: {url}")
                return None
            
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": f"{output_dir}/%(title)s.%(ext)s",
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                }],
                "quiet": True,
                "cookiefile": cookies,  # Укажите путь к вашему файлу с куками
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                final_path = filename.rsplit(".", 1)[0] + ".mp3"
                gr.Warning("Трек успешно скачан")
                return final_path
        url = extract_url_from_text(url_or_text)
        audio = download_track(url, cookies, output_dir)
        return audio
    if type_url == "Прямая ссылка":

        def create_unique_file_name(prefix="song_", length=15):
            while True:
                random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
                folder_name = prefix + random_part
                return folder_name

        def download_file(url, out_dir):
            try:
                save_path = os.path.join(out_dir, f"{create_unique_file_name()}.bin")
                
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Проверяем, что запрос выполнен успешно
                
                # Записываем содержимое в файл
                with open(save_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                
                def get_correct_extension(file_path):
                    # Определяем MIME-тип
                    mime = magic.Magic(mime=True)
                    mime_type = mime.from_file(file_path)
                    
                    # Получаем расширение из MIME-типа
                    extension = guess_extension(mime_type)
                    return extension if extension else ".bin"  # если тип неизвестен → .bin
                
                def is_audio_file(file_path):
                    audio_extensions = {'.mp3', '.wav', '.flac', '.aiff', '.ogg', '.opus', '.m4a', '.aac'}
                    _, ext = os.path.splitext(file_path)
                    return ext.lower() in audio_extensions
                
                def rename_file_with_proper_extension(file_path):
                    dirname = os.path.dirname(file_path)
                    basename = os.path.basename(file_path)
                    name_without_ext = os.path.splitext(basename)[0]
                    
                    # Получаем правильное расширение
                    correct_ext = get_correct_extension(file_path)
                    
                    # Новое имя файла
                    new_name = f"{name_without_ext}{correct_ext}"
                    new_path = os.path.join(dirname, new_name)
                    
                    # Переименовываем
                    os.rename(file_path, new_path)
                    return new_path
                
                save_path = rename_file_with_proper_extension()
                is_audio = is_audio_file(save_path)
                if is_audio == False:
                    gr.Warning("Скачанный файл не является аудиофайлом")
                    return None
                gr.Warning(f"Файл успешно скачан и сохранен как {save_path}")
                return save_path
            except Exception as e:
                gr.Warning(f"Произошла ошибка при скачивании файла: {e}")
                return None
        url = extract_url_from_text(url_or_text)
        audio = download_file(url, output_dir)
        return audio
