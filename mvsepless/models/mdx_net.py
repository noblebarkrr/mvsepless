import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
from typing import Dict, Any, List
import torch.nn.functional as F
import sys
import json


class MDXNet(nn.Module):
    def __init__(
        self,
        dim_f: int,
        dim_t: int,
        n_fft: int,
        hop_length: int,
        primary_stem: str,
        compensation: float = 1.0,
    ):
        super().__init__()
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.n_fft = n_fft
        self.dim_c = 4
        self.hop_length = hop_length
        self.primary_stem = primary_stem
        self.compensation = compensation

        self.internal_chunk_size = self.hop_length * (self.dim_t - 1)
        self.n_bins = self.n_fft // 2 + 1

        self.ort_session = None

    def init_onnx_session(self, onnx_model_path: str, device: torch.device, device_ids: list):
        if device.type == "cuda":
            providers = ["CUDAExecutionProvider"]
        elif device.type == "mps":
            if "CoreMLExecutionProvider" in ort.get_available_providers():
                providers = ["CoreMLExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.ort_session = ort.InferenceSession(onnx_model_path, providers=providers, provider_options={"device_id": device_ids[0]} if device_ids else None)

        self.ort_session.run(
            None,
            {"input": torch.rand(1, 4, self.dim_f, self.dim_t).numpy()},
        )

        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)

        out_c = self.dim_c

        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t])

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        window = self.window.to(x.device)

        x = x.reshape([-1, self.internal_chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 4, self.n_bins, self.dim_t]
        )
        return x[:, :, : self.dim_f]

    def istft(self, x: torch.Tensor) -> torch.Tensor:
        window = self.window.to(x.device)
        freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1]).to(x.device)

        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
        )
        return x.reshape([-1, 2, self.internal_chunk_size])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ort_session is None:
            raise ValueError(
                "ONNX session not initialized. Call init_onnx_session first."
            )

        x_np = x.cpu().numpy()
        output = self.ort_session.run(None, {"input": x_np})[0]
        return torch.from_numpy(output).to(x.device)

    def process_wave(
        self,
        wave: torch.Tensor,
        device: torch.device,
        num_overlap: int,
        pbar: bool = False,
    ) -> torch.Tensor:
        wave = wave.to(device)

        chunk_size = self.internal_chunk_size
        fade_size = chunk_size // 10
        step = chunk_size // num_overlap
        border = chunk_size - step

        length_init = wave.shape[-1]

        if length_init > 2 * border and border > 0:
            wave = nn.functional.pad(wave, (border, border), mode="reflect")

        window = self._get_windowing_array(chunk_size, fade_size).to(device)

        batch_size = 1

        with torch.no_grad():
            result = torch.zeros_like(wave, device=device)
            counter = torch.zeros_like(wave, device=device)

            i = 0
            batch_data = []
            batch_locations = []

            total_chunks = 0
            temp_i = 0
            while temp_i < wave.shape[1]:
                total_chunks += 1
                temp_i += step

            processed_chunks = 0

            while i < wave.shape[1]:
                part = wave[:, i : i + chunk_size]
                chunk_len = part.shape[-1]

                if chunk_len < chunk_size:
                    pad_mode = "reflect" if chunk_len > chunk_size // 2 else "constant"
                    part = nn.functional.pad(
                        part, (0, chunk_size - chunk_len), mode=pad_mode, value=0
                    )

                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += step

                if len(batch_data) >= batch_size or i >= wave.shape[1]:
                    arr = torch.stack(batch_data, dim=0)

                    for j, (start, seg_len) in enumerate(batch_locations):
                        spec = self.stft(arr[j : j + 1])
                        processed_spec = self(spec)
                        processed_wav = self.istft(processed_spec)

                        window_segment = window[..., :seg_len]
                        result[:, start : start + seg_len] += (
                            processed_wav[0, :, :seg_len] * window_segment
                        )
                        counter[:, start : start + seg_len] += window_segment

                    processed_chunks += len(batch_data)
                    if pbar:
                        progress_data = {
                            "processing": {
                                "processed": min(i, wave.shape[1]),
                                "total": wave.shape[1],
                            }
                        }
                        sys.stdout.write(
                            json.dumps(progress_data, ensure_ascii=False) + "\n"
                        )
                        sys.stdout.flush()

                    batch_data.clear()
                    batch_locations.clear()

            estimated_sources = result / counter

            if length_init > 2 * border and border > 0:
                estimated_sources = estimated_sources[..., border:-border]

            return estimated_sources

    def _get_windowing_array(self, window_size: int, fade_size: int) -> torch.Tensor:
        fadein = torch.linspace(0, 1, fade_size)
        fadeout = torch.linspace(1, 0, fade_size)

        window = torch.ones(window_size)
        window[-fade_size:] = fadeout
        window[:fade_size] = fadein
        return window
