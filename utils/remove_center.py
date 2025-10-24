import numpy as np
from datetime import datetime
from scipy import signal
import os
import tempfile
from multi_inference import clean_filename
from separator.audio_utils import Audio
audio = Audio()

class PhantomCenterExtractor:

    def __init__(self, base_dir=tempfile.mkdtemp(prefix="rmv_center")):
        self.output_base_dir = base_dir
        self.w_types = [
            "boxcar",  # Прямоугольное окно
            "triang",  # Треугольное окно
            "blackman",  # Окно Блэкмана
            "hamming",  # Окно Хэмминга
            "hann",  # Окно Ханна
            "bartlett",  # Окно Бартлетта
            "flattop",  # Окно с плоской вершиной
            "parzen",  # Окно Парзена
            "bohman",  # Окно Бохмана
            "blackmanharris",  # Окно Блэкмана-Харриса
            "nuttall",  # Окно Нуттала
            "barthann",  # Окно Бартлетта-Ханна
            "cosine",  # Косинусное окно
            "exponential",  # Экспоненциальное окно
            "tukey",  # Окно Туки
            "taylor",  # Окно Тейлора
            "lanczos",  # Окно Ланцоша
        ]

    def remove_center(
        self,
        input_file,
        output_format="flac",
        rdf=0.99999,
        window_size=4096,
        overlap=2,
        window_type="blackman",
        stereo_mode="stereo",
    ):
        output_dir = os.path.join(
            self.output_base_dir, datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(output_dir, exist_ok=True)
        base_name_file = os.path.splitext(os.path.basename(input_file))[0]
        base_name_file = clean_filename(base_name_file, length=120)
        output_file = os.path.join(
            output_dir, f"{base_name_file}_other.{output_format}"
        )
        output_center_file = os.path.join(
            output_dir, f"{base_name_file}_center.{output_format}"
        )
        data, samplerate = audio.read(i=input_file, mono=False, sr=None)

        if data.ndim != 2 or data.shape[0] != 2:
            raise ValueError("Требуется стереофайл (2 канала)")

        left = data[0, :]
        right = data[1, :]
        mono = left * 0.5 + right * 0.5

        nperseg = window_size  # Размер окна
        noverlap = nperseg // overlap  # Перекрытие окон

        f, t, Z_left = signal.stft(
            left, fs=samplerate, nperseg=nperseg, noverlap=noverlap, window=window_type
        )
        f, t, Z_right = signal.stft(
            right, fs=samplerate, nperseg=nperseg, noverlap=noverlap, window=window_type
        )
        f, t, Z_mono = signal.stft(
            mono, fs=samplerate, nperseg=nperseg, noverlap=noverlap, window=window_type
        )
        if stereo_mode == "mono":
            Z_common_left = np.minimum(np.abs(Z_left), np.abs(Z_right)) * np.exp(
                1j * np.angle(Z_mono)
            )
            Z_common_right = np.minimum(np.abs(Z_left), np.abs(Z_right)) * np.exp(
                1j * np.angle(Z_mono)
            )
        else:
            Z_common_left = np.minimum(np.abs(Z_left), np.abs(Z_right)) * np.exp(
                1j * np.angle(Z_right)
            )
            Z_common_right = np.minimum(np.abs(Z_left), np.abs(Z_right)) * np.exp(
                1j * np.angle(Z_left)
            )

        reduction_factor = rdf

        Z_new_left = Z_left - Z_common_left * reduction_factor
        Z_new_right = Z_right - Z_common_right * reduction_factor

        _, new_left = signal.istft(
            Z_new_left,
            fs=samplerate,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window_type,
        )
        _, new_right = signal.istft(
            Z_new_right,
            fs=samplerate,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window_type,
        )

        _, common_signal_left = signal.istft(
            Z_common_left,
            fs=samplerate,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window_type,
        )
        _, common_signal_right = signal.istft(
            Z_common_right,
            fs=samplerate,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window_type,
        )

        new_left = new_left[: len(left)]
        new_right = new_right[: len(right)]
        common_signal_left = common_signal_left[: len(left)]
        common_signal_right = common_signal_right[: len(right)]

        peak = np.max([np.abs(new_left).max(), np.abs(new_right).max()])
        if peak > 1.0:
            new_left = new_left / peak
            new_right = new_right / peak

        audio.write(
            o=output_file,
            array=np.column_stack((new_left, new_right)),
            sr=samplerate,
            of=output_format,
            br="320k",
        )

        inverted_center_left = -common_signal_left
        inverted_center_right = -common_signal_right

        mixed_left = left + inverted_center_left
        mixed_right = right + inverted_center_right

        peak_mixed = np.max([np.abs(mixed_left).max(), np.abs(mixed_right).max()])
        if peak_mixed > 1.0:
            mixed_left = mixed_left / peak_mixed
            mixed_right = mixed_right / peak_mixed

        audio.write(
            o=output_center_file,
            array=np.column_stack((common_signal_left, common_signal_right)),
            sr=samplerate,
            of=output_format,
            br="320k",
        )

        return (output_file, output_center_file)
