import os, sys, gradio as gr, tempfile, zipfile, ast, json, pickle, argparse, shutil, re, subprocess
from datetime import datetime, timezone, timedelta

tz = timezone(timedelta(hours=3))

class GradioHelper:

    def return_list(self, list, none=False, **kwargs):
        if list:
            return gr.update(choices=list, value=list[0] if not none else None, **kwargs)
        else:
            return gr.update(choices=[], value=None, **kwargs)

    def return_audio(self, label, path):
        return gr.update(label=label, value=path)

    def get_file_size(self, path):
        if path:
            if os.path.exists(path):
                size_bytes = os.path.getsize(path)
            else:
                return "[Указанного файла не существует]"
        else:
            return ""

        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        
        if size_bytes == 0:
            return "[0 B]"
        
        i = 0
        while size_bytes >= 1024 and i < len(units) - 1:
            size_bytes /= 1024
            i += 1
        
        return f"[{size_bytes:.1f} {units[i]}]" if i > 0 else f"[{size_bytes} {units[i]}]"

    def return_audio_with_size(self, *args, **kwargs):
        if "label" in kwargs and "value" in kwargs:
            kwargs["label"] = f"{self.get_file_size(kwargs['value'])} {kwargs['label']}"
        elif not "label" in kwargs and "value" in kwargs:
            kwargs["label"] = f"{self.get_file_size(kwargs['value'])}"
        return gr.update(**kwargs)

    def define_audio_with_size(self, *args, **kwargs):
        if "label" in kwargs and "value" in kwargs:
            kwargs["label"] = f"{self.get_file_size(kwargs['value'])} {kwargs['label']}"
        elif not "label" in kwargs and "value" in kwargs:
            kwargs["label"] = f"{self.get_file_size(kwargs['value'])}"
        return gr.Audio(**kwargs)

    def create_archive_advanced(self, file_list, archive_name="archive.zip"):
        try:
            print("Генерация ZIP-архива с результатами разделения...")
            with zipfile.ZipFile(
                archive_name, "w", zipfile.ZIP_DEFLATED
            ) as zipf:
                successful_files = 0

                for basename, stems in file_list:
                    for stem_name, stem_path in stems:
                        try:
                            if os.path.exists(stem_path) and os.path.isfile(
                                stem_path
                            ):
                                basename_ = os.path.basename(stem_path)
                                zipf.write(stem_path, basename_)
                                successful_files += 1
                                print(
                                    f"✓ Добавлен: {stem_path} -> {basename}"
                                )
                            else:
                                print(
                                    f"✗ Файл не найден или не является файлом: {stem_path}"
                                )

                        except Exception as e:
                            print(
                                f"✗ Ошибка при добавлении {stem_path}: {e}"
                            )

                print(f"\nАрхив создан: {archive_name}")
                print(f"Успешно добавлено файлов: {successful_files}")
                return os.path.abspath(archive_name)

        except Exception as e:
            print(f"Ошибка при создании архива: {e}")

    def extract_zip(self, zip_file_path, output_dir=None):
        """
        Распаковывает ZIP-архив с обработкой ошибок
        
        Args:
            zip_file_path: путь к ZIP-архиву
            output_dir: папка для распаковки (по умолчанию - текущая директория)
        """
        
        if output_dir is None:
            output_dir = os.path.splitext(zip_file_path)[0]
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                if zip_ref.testzip() is not None:
                    print("Предупреждение: архив может быть поврежден")
                
                zip_ref.extractall(output_dir)
                
                file_count = len(zip_ref.namelist())
                print(f"Успешно распаковано {file_count} файлов в {output_dir}")                
        except zipfile.BadZipFile:
            print("Ошибка: файл не является ZIP-архивом или поврежден")
            return []
        except PermissionError:
            print("Ошибка: нет прав на запись в указанную директорию")
            return []
        except Exception as e:
            print(f"Неизвестная ошибка: {e}")
            return []
        finally:
            input_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    input_files.append(os.path.join(root, file))
            return input_files