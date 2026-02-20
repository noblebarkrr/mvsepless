import argparse
import os
from audio import read, multiread, write, trim, concatenate, check

def trim_(i, s=0, e=-1, output_path="./trimmed.mp3"):
    y, sr = read(i)
    end_sample = int(e * sr) if e != -1 else -1
    y = trim(y, int(s * sr), end_sample)
    return write(output_path, y, sr)

def concat(files, output_path):
    # Фильтруем файлы с помощью функции check
    valid_files = [f for f in files if check(f)]
    
    if not valid_files:
        print("Ошибка: Не найдено подходящих аудиофайлов для склейки.")
        return

    print(f"Обработка {len(valid_files)} файлов...")
    arrays, srs = multiread(valid_files)
    full_audio, max_sr = concatenate(arrays, srs, dtype="float32")
    return write(output_path, full_audio, max_sr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Утилита для обработки аудио")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # TRIM
    trim_parser = subparsers.add_parser("trim")
    trim_parser.add_argument("--input", type=str, required=True, help="Путь к входному файлу")
    trim_parser.add_argument("-s", "--start", type=float, default=0)
    trim_parser.add_argument("-e", "--end", type=float, default=-1)
    trim_parser.add_argument("-o", "--output", default="./trimmed.mp3")

    # CONCAT
    concat_parser = subparsers.add_parser("concat")
    # Добавляем аргумент, который может быть и списком файлов, и директорией
    concat_parser.add_argument("--path", nargs="+", help="Список файлов или путь к папке")
    concat_parser.add_argument("-o", "--output", required=True)

    args = parser.parse_args()

    if args.command == "trim":
        trim_(args.input, args.start, args.end, args.output)
        print(f"Готово: {args.output}")

    elif args.command == "concat":
        target_files = []
        
        for p in args.path:
            if os.path.isdir(p):
                # Если это папка, берем все файлы внутри
                files_in_dir = [os.path.join(p, f) for f in os.listdir(p) 
                                if os.path.isfile(os.path.join(p, f))]
                # Сортируем по имени, чтобы склейка была предсказуемой
                target_files.extend(sorted(files_in_dir))
            else:
                target_files.append(p)
        
        concat(target_files, args.output)
        print(f"Готово! Результат: {args.output}")