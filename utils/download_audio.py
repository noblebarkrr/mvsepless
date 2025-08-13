import os
import yt_dlp
import urllib.request
import gradio as gr
CURRENT_DIR = os.getcwd()

class Downloader:
    def __init__(self, output_dir="downloaded"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def dw_yt_dlp(self, url, cookie=None, output_format="mp3", output_bitrate="320", title=None):
        # Подготовка шаблона имени файла
        outtmpl = "%(title)s.%(ext)s" if title is None else f"{title}.%(ext)s"
        
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(self.output_dir, outtmpl),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": output_format,
                    "preferredquality": output_bitrate,
                }
            ],
            "noplaylist": True,  # Скачивать только одно видео, не плейлист
            "quiet": True,  # Отключить вывод в консоль
            "no_warnings": True,  # Скрыть предупреждения
        }

        # Добавляем cookies если указаны
        if cookie and os.path.exists(cookie):
            ydl_opts["cookiefile"] = cookie

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
                if "_type" in info and info["_type"] == "playlist":
                    # Для плейлистов берем первое видео
                    entry = info["entries"][0]
                    filename = ydl.prepare_filename(entry)
                else:
                    # Для одиночного видео
                    filename = ydl.prepare_filename(info)

                # Заменяем оригинальное расширение на выбранный формат
                base, _ = os.path.splitext(filename)
                audio_file = base + f".{output_format}"

                return os.path.abspath(audio_file)
            except Exception as e:
                print(e)
                gr.Warning(e)
                return url
            
    def dw_from_url(self, url, title=None):
        try:
            response = urllib.request.urlopen(url)
            content_type = response.info().get_content_type()
            if "audio" in content_type:
                filename = os.path.join(self.output_dir, title or "downloaded_audio")
                with open(filename, 'wb') as f:
                    f.write(response.read())
                return filename
            else:
                raise ValueError("URL does not point to an audio file.")
        except Exception as e:
            print(e)
            gr.Warning(e)
            return url
