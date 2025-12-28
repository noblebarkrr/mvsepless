import math
from typing import Optional

import numpy as np
import torch
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint

LRELU_SLOPE = 0.1


class MRFLayer(torch.nn.Module):

    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = weight_norm(
            torch.nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=(kernel_size * dilation - dilation) // 2,
                dilation=dilation,
            )
        )
        self.conv2 = weight_norm(
            torch.nn.Conv1d(
                channels, channels, kernel_size, padding=kernel_size // 2, dilation=1
            )
        )

    def forward(self, x: torch.Tensor):
        y = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
        y = self.conv1(y)
        y = torch.nn.functional.leaky_relu(y, LRELU_SLOPE)
        y = self.conv2(y)
        return x + y

    def remove_weight_norm(self):
        remove_weight_norm(self.conv1)
        remove_weight_norm(self.conv2)


class MRFBlock(torch.nn.Module):

    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for dilation in dilations:
            self.layers.append(MRFLayer(channels, kernel_size, dilation))

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def remove_weight_norm(self):
        for layer in self.layers:
            layer.remove_weight_norm()


class SineGenerator(torch.nn.Module):

    def __init__(
        self,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super(SineGenerator, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0: torch.Tensor):
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values: torch.Tensor):
        rad_values = (f0_values / self.sampling_rate) % 1

        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)

        return sines

    def forward(self, f0: torch.Tensor):
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            sine_waves = self._f02sine(f0_buf) * self.sine_amp

            uv = self._f02uv(f0)

            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):

    def __init__(
        self,
        sampling_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        self.l_sin_gen = SineGenerator(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold
        )

        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor):
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        return sine_merge, None, None


class HiFiGANMRFGenerator(torch.nn.Module):

    def __init__(
        self,
        in_channel: int,
        upsample_initial_channel: int,
        upsample_rates: list[int],
        upsample_kernel_sizes: list[int],
        resblock_kernel_sizes: list[int],
        resblock_dilations: list[list[int]],
        gin_channels: int,
        sample_rate: int,
        harmonic_num: int,
        checkpointing: bool = False,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.checkpointing = checkpointing

        self.f0_upsample = torch.nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num)

        self.conv_pre = weight_norm(
            torch.nn.Conv1d(
                in_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3
            )
        )
        self.upsamples = torch.nn.ModuleList()
        self.noise_convs = torch.nn.ModuleList()

        stride_f0s = [
            math.prod(upsample_rates[i + 1 :]) if i + 1 < len(upsample_rates) else 1
            for i in range(len(upsample_rates))
        ]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            if u % 2 == 0:
                padding = (k - u) // 2
            else:
                padding = u // 2 + u % 2

            self.upsamples.append(
                weight_norm(
                    torch.nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=k,
                        stride=u,
                        padding=padding,
                        output_padding=u % 2,
                    )
                )
            )
            """ handling odd upsampling rates
            #  s   k   p
            # 40  80  20
            # 32  64  16
            #  4   8   2
            #  2   3   1
            # 63 125  31
            #  9  17   4
            #  3   5   1
            #  1   1   0
            """
            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            padding = 0 if stride == 1 else (kernel - stride) // 2

            self.noise_convs.append(
                torch.nn.Conv1d(
                    1,
                    upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )
        self.mrfs = torch.nn.ModuleList()
        for i in range(len(self.upsamples)):
            channel = upsample_initial_channel // (2 ** (i + 1))
            self.mrfs.append(
                torch.nn.ModuleList(
                    [
                        MRFBlock(channel, kernel_size=k, dilations=d)
                        for k, d in zip(resblock_kernel_sizes, resblock_dilations)
                    ]
                )
            )
        self.conv_post = weight_norm(
            torch.nn.Conv1d(channel, 1, kernel_size=7, stride=1, padding=3)
        )
        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(
        self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None
    ):
        f0 = self.f0_upsample(f0[:, None, :]).transpose(-1, -2)
        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(-1, -2)
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        for ups, mrf, noise_conv in zip(self.upsamples, self.mrfs, self.noise_convs):
            x = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)

            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False)
                x = x + noise_conv(har_source)
                xs = sum([checkpoint(layer, x, use_reentrant=False) for layer in mrf])
            else:
                x = ups(x)
                x = x + noise_conv(har_source)
                xs = sum([layer(x) for layer in mrf])
            x = xs / self.num_kernels

        x = torch.nn.functional.leaky_relu(x)
        x = torch.tanh(self.conv_post(x))

        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for up in self.upsamples:
            remove_weight_norm(up)
        for mrf in self.mrfs:
            mrf.remove_weight_norm()
        remove_weight_norm(self.conv_post)
