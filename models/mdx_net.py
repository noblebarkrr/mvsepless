import torch
import onnx
import platform
import onnx2torch
import numpy as np
from typing import Dict, Any, List

class MDXNet():
    def __init__(
        self,
        dim_f: int,
        n_fft: int,
        hop_length: int,
        compensation: float = 1.0,
        **kwargs
    ):
        self.dim_f = dim_f
        self.n_fft = n_fft
        self.dim_c = 4
        self.compensation = compensation
        self.n_bins = self.n_fft // 2 + 1
        self.onnx_model_path = None
        self.providers = []
        self.provider_options = None

    def init_onnx_session(self, onnx_model_path: str, device: torch.device, device_ids: list):
        self.onnx_model_path = onnx_model_path
        if platform.system() == 'Windows':
            onnx_model = onnx.load(self.onnx_model_path)
            self.model = onnx2torch.convert(onnx_model)
        else:
            self.model = onnx2torch.convert(self.onnx_model_path)
        self.model.to(device).eval()
        
    def post_init(self, dim_t, device):
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        out_c = self.dim_c
        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, dim_t])

    def stft(self, x: torch.Tensor, chunk_size: int, hop_length: int, dim_t: int) -> torch.Tensor:
        window = self.window.to(x.device)
        x = x.reshape([-1, chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=hop_length,
            window=window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, dim_t]).reshape(
            [-1, 4, self.n_bins, dim_t]
        )
        return x[:, :, : self.dim_f]

    def istft(self, x: torch.Tensor, chunk_size: int, hop_length: int, dim_t: int) -> torch.Tensor:
        window = self.window.to(x.device)
        freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1]).to(x.device)

        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1, 2, 2, self.n_bins, dim_t]).reshape(
            [-1, 2, self.n_bins, dim_t]
        )
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=hop_length,
            window=window,
            center=True,
        )
        return x.reshape([-1, 2, chunk_size])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
