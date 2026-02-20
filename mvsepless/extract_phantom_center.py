from audio import read, split_mid_side, write
import argparse

def extract_phantom_center(i, output_path_mid, output_path_side):
    y, sr = read(i)
    mid, side = split_mid_side(y, var=3, sr=sr)
    print(f"Запись в файлы: {output_path_mid} и {output_path_side}")
    return write(output_path_mid, mid, sr), write(output_path_side, side, sr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Разделение аудио на фантомный центр и стерео-базу"
    )
    parser.add_argument("--input", type=str, required=True, help="Путь к входному файлу")
    parser.add_argument("--output_path_mid", type=str, required=True, help="Путь к выходному файлу центра")
    parser.add_argument("--output_path_side", type=str, required=True, help="Путь к выходному файлу стерео-базы")
    args = parser.parse_args()
    extract_phantom_center(args.input, args.output_path_mid, args.output_path_side)