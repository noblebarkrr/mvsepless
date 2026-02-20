from audio import read, substractor, write
import argparse

class Inverter:
    def __init__(self):
        self.test = "test"
        self.w_types = (
            "boxcar",
            "triang",
            "blackman",
            "hamming",
            "hann",
            "bartlett",
            "flattop",
            "parzen",
            "bohman",
            "blackmanharris",
            "nuttall",
            "barthann",
            "cosine",
            "exponential",
            "tukey",
            "taylor",
            "lanczos",
        )
        self.methods = ("waveform", "spectrogram")

    def process_audio(
        self,
        audio1_path,
        audio2_path,
        method,
        output_path="./inverted.mp3",
    ):
        y1, sr1 = read(audio1_path)
        y2, sr2 = read(audio2_path)
        inverted, min_sr = substractor(y1, y2, sr1, sr2, spectrogram=method == "spectrogram")
        print(f"Запись в файл: {output_path}")
        return write(output_path, inverted, min_sr)
    
if __name__ == "__main__":
    inverter = Inverter()
    parser = argparse.ArgumentParser(
        description="Вычитание сигнала стема из оригинала"
    )
    parser.add_argument("--original", type=str, required=True, help="Путь к оригиналу")
    parser.add_argument("--stem", type=str, required=True, help="Путь к стему")
    parser.add_argument("--method", type=str, choices=inverter.methods, default=inverter.methods[0], help="Метод вычитания")
    parser.add_argument("--output_path", type=str, required=True, help="Путь к выходному файлу")
    args = parser.parse_args()
    inverter.process_audio(args.original, args.stem, args.method, args.output_path)