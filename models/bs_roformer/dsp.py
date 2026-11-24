from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scipy.signal import firwin, firwin2
from torch import Tensor
import sys
import platform
from functools import lru_cache, partial
from pathlib import Path


# =============================================================================
# utils.py
# =============================================================================

def fix_length(x: Tensor, start: int, length: int) -> Tensor:
    end = start + length
    return x[..., start : end]


def shift_frequency(x: Tensor, f: Tensor, sr: int, n=1) -> Tensor:
    r"""Shift frequency to -ω0.

    X(j(ω-ω0)) = e^{j(ω0)n}X(jω)

    k: n_bands
    l: audio_samples

    Args:
        x: (any, k, l)
        f: (k,)

    Returns:
        out: (any, k, l)
    """
    if f.dtype in {torch.float16, torch.bfloat16}:
        f = f.float()
    omega = f / (sr / 2) * math.pi  # (k,)
    t = torch.arange(x.shape[-1], device=x.device) * n  # (l,)
    a = torch.exp(1.j * omega[:, None] * t[None, :])
    out = x * a
    return out


# =============================================================================
# analytic.py
# =============================================================================

def real_to_analytic(x: Tensor) -> Tensor:
    r"""Convert real signal into analytic signal.

    Args:
        x: (any, l), real signal

    Returns:
        out: (any, l), complex signal
    """
    L = x.shape[-1]
    x = torch.fft.fft(x)
    x[..., 1 : L // 2] *= 2
    x[..., L // 2 + 1 :] = 0
    return torch.fft.ifft(x)


def analytic_to_real(x: Tensor) -> Tensor:
    r"""Convert analytic singal into real signal.

    Args:
        x: (any,), complex signal

    Returns:
        out: (any,), real signal
    """
    return x.real


# =============================================================================
# banks.py
# =============================================================================

def linear_banks(
    sr: int,
    n_bands: int,
) -> list[tuple[float, float]]:
    r"""Linear banks.

    Returns:
        (n_banks, 2)
    """
    freqs = np.linspace(0, sr / 2, n_bands + 1)
    banks = [[freqs[i].item(), freqs[i + 1].item()] for i in range(len(freqs) - 1)]  # (k1+k2, 2)
    return banks


def mel_linear_banks(
    sr: int,
    n_bands: int,
    max_bandwidth: float
) -> list[tuple[float, float]]:
    r"""Mel bank in low frequency and linear band in high frequency.

    Returns:
        (n_banks, 2)
    """
    freqs = np.linspace(0, hz_to_mel(sr / 2), n_bands + 1)
    freqs = mel_to_hz(freqs)

    if max(np.diff(freqs)) >= max_bandwidth:
        idx = np.argmax(np.diff(freqs) >= max_bandwidth)  # (k1,)
        mel_part = freqs[: idx + 1]  # (k1,)
        linear_part = np.arange(mel_part[-1] + max_bandwidth, sr//2 + max_bandwidth, max_bandwidth)  # (k2,)
        freqs = np.concatenate([mel_part, linear_part])  # (k1+k2,)
        freqs[-1] = sr // 2

    banks = [[freqs[i].item(), freqs[i + 1].item()] for i in range(len(freqs) - 1)]  # (k1+k2, 2)
    return banks


def mel_linear_banks_triangle(
    sr: int,
    n_bands: int,
    max_bandwidth: float
) -> list[tuple[float, float]]:
    r"""Mel bank in low frequency and linear band in high frequency.

    Returns:
        (n_banks, 2)
    """
    freqs = np.linspace(0, hz_to_mel(sr / 2), n_bands + 1)
    freqs = mel_to_hz(freqs)

    if max(np.diff(freqs)) >= max_bandwidth:
        idx = np.argmax(np.diff(freqs) >= max_bandwidth)  # (k1,)
        mel_part = freqs[: idx + 1]  # (k1,)
        linear_part = np.arange(mel_part[-1] + max_bandwidth, sr//2 + max_bandwidth, max_bandwidth)  # (k2,)
        freqs = np.concatenate([mel_part, linear_part])  # (k1+k2,)
        freqs[-1] = sr // 2

    banks = [[freqs[0].item(), freqs[1].item()]]
    banks += [[freqs[i].item(), freqs[i + 1].item(), freqs[i + 2].item()] for i in range(len(freqs) - 2)]
    banks += [[freqs[-2].item(), freqs[-1].item()]]
    return banks


def erb_linear_banks(
    sr: int,
    n_bands: int,
    max_bandwidth: float,
) -> list[tuple[float, float]]:
    r"""ERB bank in low frequency and linear band in high frequency.

    Returns:
        (n_banks, 2)
    """
    freqs = np.linspace(0, hz_to_erb(sr / 2), n_bands + 1)
    freqs = erb_to_hz(freqs)

    if max(np.diff(freqs)) >= max_bandwidth:
        idx = np.argmax(np.diff(freqs) >= max_bandwidth)  # (k1,)
        mel_part = freqs[: idx + 1]  # (k1,)
        linear_part = np.arange(mel_part[-1] + max_bandwidth, sr//2 + max_bandwidth, max_bandwidth)  # (k2,)
        freqs = np.concatenate([mel_part, linear_part])  # (k1+k2,)
        freqs[-1] = sr // 2

    banks = [[freqs[i].item(), freqs[i + 1].item()] for i in range(len(freqs) - 1)]  # (k1+k2, 2)
    return banks


def erb_linear_banks_overlap(
    sr: int,
    n_bands: int,
    max_bandwidth: float,
) -> list[tuple[float, float]]:
    r"""ERB bank in low frequency and linear band in high frequency.

    Returns:
        (n_banks, 2)
    """
    freqs = np.linspace(0, hz_to_erb(sr / 2), n_bands + 1)
    freqs = erb_to_hz(freqs)

    if max(np.diff(freqs)) >= max_bandwidth:
        idx = np.argmax(np.diff(freqs) >= max_bandwidth)  # (k1,)
        mel_part = freqs[: idx + 1]  # (k1,)
        linear_part = np.arange(mel_part[-1] + max_bandwidth, sr//2 + max_bandwidth, max_bandwidth)  # (k2,)
        freqs = np.concatenate([mel_part, linear_part])  # (k1+k2,)
        freqs[-1] = sr // 2

    banks = [[freqs[0].item(), freqs[1].item()]]
    banks += [[freqs[i].item(), freqs[i + 2].item()] for i in range(len(freqs) - 2)]  # (k1+k2, 2)
    banks += [[freqs[-2].item(), freqs[-1].item()]]
    return banks


def erb_linear_banks_triangle(
    sr: int,
    n_bands: int,
    max_half_bandwidth: float,
) -> list[tuple[float, float]]:
    r"""ERB bank in low frequency and linear band in high frequency.

    Returns:
        (n_banks, 2)
    """
    freqs = np.linspace(0, hz_to_erb(sr / 2), n_bands + 1)
    freqs = erb_to_hz(freqs)

    if max(np.diff(freqs)) >= max_half_bandwidth:
        idx = np.argmax(np.diff(freqs) >= max_half_bandwidth)  # (k1,)
        mel_part = freqs[: idx + 1]  # (k1,)
        linear_part = np.arange(mel_part[-1] + max_half_bandwidth, sr//2 + max_half_bandwidth, max_half_bandwidth)  # (k2,)
        freqs = np.concatenate([mel_part, linear_part])  # (k1+k2,)
        freqs[-1] = sr // 2

    banks = [[freqs[0].item(), freqs[1].item()]]
    banks += [[freqs[i].item(), freqs[i + 1].item(), freqs[i + 2].item()] for i in range(len(freqs) - 2)]
    banks += [[freqs[-2].item(), freqs[-1].item()]]

    return banks


def erb_linear_ex_banks_triangle(
    sr: int,
    n_bands: int,
    max_half_bandwidth: float,
    a: float,
    b: float,
) -> list[tuple[float, float]]:
    r"""ERB bank in low frequency and linear band in high frequency.

    Returns:
        (n_banks, 2)
    """
    freqs = np.linspace(0, hz_to_erb_ex(sr / 2, a, b), n_bands + 1)
    freqs = erb_to_hz_ex(freqs, a, b)

    if max(np.diff(freqs)) >= max_half_bandwidth:
        idx = np.argmax(np.diff(freqs) >= max_half_bandwidth)  # (k1,)
        mel_part = freqs[: idx + 1]  # (k1,)
        linear_part = np.arange(mel_part[-1] + max_half_bandwidth, sr//2 + max_half_bandwidth, max_half_bandwidth)  # (k2,)
        freqs = np.concatenate([mel_part, linear_part])  # (k1+k2,)
        freqs[-1] = sr // 2

    banks = [[freqs[0].item(), freqs[1].item()]]
    banks += [[freqs[i].item(), freqs[i + 1].item(), freqs[i + 2].item()] for i in range(len(freqs) - 2)]
    banks += [[freqs[-2].item(), freqs[-1].item()]]

    return banks


def exp_linear_banks(
    sr: int,
    n_bands: int,
) -> list[tuple[float, float]]:
    r"""ERB bank in low frequency and linear band in high frequency.

    Returns:
        (n_banks, 2)
    """

    # Low freq
    freqs = [0, 25, 30]

    # Middle freq
    r = 2
    for i in range(68):
        freqs.append(freqs[-1] + r)
        r = r ** 1.032

    # High freq
    n_linear_bands = (n_bands - len(freqs))
    bw = ((sr / 2) - freqs[-1]) / n_linear_bands

    for _ in range(n_linear_bands):
        freqs.append(freqs[-1] + bw)

    freqs[-1] = sr / 2

    # Banks
    banks = [[freqs[0], freqs[1]]]
    banks += [[freqs[i], freqs[i + 1], freqs[i + 2]] for i in range(len(freqs) - 2)]
    banks += [[freqs[-2], freqs[-1]]]

    return banks


def exp_linear_banks2(
    sr: int,
    n_bands: int,
    max_bandwidth: float,
    q: float
) -> list[tuple[float, float]]:
    r"""ERB bank in low frequency and linear band in high frequency.

    Returns:
        (n_banks, 2)
    """

    n_bands -= 2
    max_bw = max_bandwidth

    a1 = (max_bw * q) / (q - 1) - sr / 2 + n_bands * max_bw
    a1 = a1 / max_bw - 1
    exp = q ** a1
    f0 = max_bw / (q - 1) / exp

    n = 1 + math.log((max_bw / f0) / (q - 1)) / math.log(q)
    n = math.ceil(n)

    # Calibriate f0
    f0 = max_bw / (q ** n - q ** (n-1))

    freqs = [0]
    for i in range(n+1):
        freqs.append(f0 * q**i)

    n_linear_bands = n_bands - n
    bw = (sr / 2 - f0 * q ** n) / n_linear_bands
    for i in range(n_linear_bands):
        freqs.append(freqs[-1] + bw)

    freqs[-1] = sr / 2

    # Banks
    banks = [[freqs[0], freqs[1]]]
    banks += [[freqs[i], freqs[i + 1], freqs[i + 2]] for i in range(len(freqs) - 2)]
    banks += [[freqs[-2], freqs[-1]]]

    return banks


def linear_banks_triangle(
    sr: int,
    n_bands: int,
) -> list[tuple[float, float]]:
    r"""ERB bank in low frequency and linear band in high frequency.

    Returns:
        (n_banks, 2)
    """

    freqs = np.linspace(0, sr / 2, n_bands)
    banks = [[freqs[0].item(), freqs[1].item()]]
    banks += [[freqs[i].item(), freqs[i + 1].item(), freqs[i + 2].item()] for i in range(len(freqs) - 2)]
    banks += [[freqs[-2].item(), freqs[-1].item()]]

    return banks


def hz_to_erb(f):
    return 21.4 * np.log10(1 + 0.00437 * f)


def erb_to_hz(erb):
    return 1 / 0.00437 * (10 ** (erb / 21.4) - 1)


def hz_to_erb_ex(f, a, b):
    return a * np.log10(1 + b * f)


def erb_to_hz_ex(erb, a, b):
    return 1 / b * (10 ** (erb / a) - 1)


def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


# =============================================================================
# convolve.py
# =============================================================================

def fftconvolve(x: Tensor, h: Tensor, use_complex_fft: bool, mode="same") -> Tensor:
    """Convolution with FFT.

    Args:
        x: (b, i, l), signal
        h: (o, i, l), signal

    Returns:
        out: (b, o, l)
    """

    if x.dtype in {torch.float16, torch.bfloat16}:
        x = x.float()
    if h.dtype in {torch.float16, torch.bfloat16}:
        h = h.float()
    L1 = x.shape[-1]
    L2 = h.shape[-1]
    L = L1 + L2 - 1  # full convolution length

    if use_complex_fft:
        fft = torch.fft.fft
        ifft = torch.fft.ifft
    else:
        fft = torch.fft.rfft
        ifft = torch.fft.irfft

    # pad both sequences to length L
    X = fft(torch.nn.functional.pad(x, (0, L - L1)))
    H = fft(torch.nn.functional.pad(h, (0, L - L2)))

    # multiply in frequency domain and IFFT
    Y = torch.einsum('bil,oil->bol', X, H)  # (b, o, l)
    y = ifft(Y, n=L)

    if mode == "full":
        return y

    elif mode == "same":
        start = (L2 - 1) // 2
        end = start + L1
        return y[:, :, start:end]  # (b, o, l)

    else:
        raise ValueError(mode)


def polyphase_fftconvolve(x: Tensor, w: Tensor, stride: int) -> Tensor:
    r"""Strided convolution with polylphase decomposition and FFT.

    b: batch_size
    k: n_banks
    l: audio_samples
    n: filter_len

    Args:
        x: (b, l_up)
        w: (k, n)
        stride: int

    Returns:
        out: (b, k, l_down)
    """
    assert torch.all(w[:, 0] == 0)
    x = rearrange(x, 'b (t1 t2) -> b t2 t1', t2=stride)  # (b, t2, t1)
    w = rearrange(w, 'k (n1 n2) -> k n2 n1', n2=stride)  # (b, n2, n1)

    L = x.shape[-1]
    N = w.shape[-1]
    assert N % 2 == 0

    w = torch.flip(w, dims=[2])  # (b, n2, n1)
    x = fftconvolve(x, w, use_complex_fft=True, mode="full")  # (b, k, l_down)
    out = fix_length(x, N // 2 - 1, L)  # (b, k, l_down)

    return out


# =============================================================================
# upsample.py
# =============================================================================

def polyphase_fftupsample(x: Tensor, w: Tensor, stride: int) -> Tensor:
    r"""Convolution with polylphase.

    b: batch_size
    l: audio_samples
    n: filter_len

    Args:
        x: (b, l_down)
        w: (n,)

    Returns:
        out: (b, l_up)
    """

    L = x.shape[-1] * stride
    N = w.shape[-1]
    x = rearrange(x, 'b l -> b 1 l')
    w = rearrange(w, '(n1 n2) -> n2 1 n1', n2=stride)  # (b, n2, n1)

    x = fftconvolve(x, w, use_complex_fft=True, mode="full")  # (b, k, l_down)
    x = rearrange(x, 'b t2 t1 -> b (t1 t2)')
    out = fix_length(x, N // 2, L)

    return out


# =============================================================================
# overlap_add.py
# =============================================================================

def overlap_add(x: Tensor, hop_length: int, window: Tensor | None) -> Tensor:
    r"""Overlap-add.

    b: batch_size
    t: n_frames
    n: frame_samples
    l: audio_samples

    Args:
        x: (b, t, n)

    Returns:
        x: (b, l)
    """

    n_frames, frame_length = x.shape[-2:]  # (t, n)
    L = frame_length + (n_frames - 1) * hop_length

    # Overlap-add
    x = F.fold(
        input=rearrange(x, 'b t n -> b n t'),  # (b, n, t)
        output_size=(1, L),
        kernel_size=(1, frame_length),
        stride=(1, hop_length)
    )  # (b, 1, 1, l)
    out = x.squeeze(dim=[1, 2])  # (b, l)

    # Divide overlap-add window
    if window is not None:
        win_norm = F.fold(
            window[None, :, None].repeat(1, 1, n_frames),  # (1, n, t),
            output_size=(1, L),
            kernel_size=(1, frame_length),
            stride=(1, hop_length)
        ).squeeze(dim=(0, 1, 2))  # (l,)

        out /= torch.clamp(win_norm, 1e-8)  # (b, l)

    return out


# =============================================================================
# subband_fast_triangle.py
# =============================================================================

class SubbandFilter(nn.Module):
    r"""Save memory version. Split signal into subbands."""

    def __init__(
        self,
        sr: int,
        banks: list[tuple[int, int]],
        factor: int,
        chunk_size = 4,
        bandpass_filter_len = 48000,
        upsample_filter_len = 12000
    ):
        r"""
        k: n_bands
        m: filter_len
        """

        super().__init__()

        self.sr = sr
        self.banks = banks
        self.window_type = "hamming"
        self.factor = factor
        self.chunk_size = chunk_size
        self.bandpass_filter_len = bandpass_filter_len
        self.upsample_filter_len = upsample_filter_len

        # Bandpass filter
        n_banks = len(banks)
        N = self.bandpass_filter_len - 1
        w = torch.zeros((n_banks, self.bandpass_filter_len))  # (k, n)
        for i in range(n_banks):
            if i == 0:
                w[i, 1 : ] = self.lowpass(banks[i], N)
            elif i == n_banks - 1:
                w[i, 1 : ] = self.highpass(banks[i], N)
            else:
                w[i, 1 : ] = self.bandpass(banks[i], N)
        self.register_buffer("w", w)  # (k, n)

        # Check Nyquist sampling rate
        bandwidths = [bank[-1] - bank[0] for bank in banks]  # (k,)
        assert max(bandwidths) <= sr / self.factor

        # Center frequency
        self.register_buffer("f_center", Tensor([np.mean(bank) for bank in banks]))  # (k,)

        # Upsample filter
        up = torch.zeros(self.upsample_filter_len)
        up[1 :] = torch.from_numpy(firwin(
            numtaps=self.upsample_filter_len - 1,
            cutoff=1. / self.factor,
            pass_zero="lowpass",
            window="hamming"
        ))
        self.register_buffer("up", up)

    def analysis(self, x: Tensor) -> Tensor:
        r"""Split signal into subbands.

        b: batch_size
        c: audio_channels
        l: audio_samples
        k: n_bands

        Args:
            x: (b, c, l)

        Returns:
            out: (b, c, k, l)
        """

        # self.factor = 10
        B, C, L = x.shape
        x = real_to_analytic(x)  # (b, c, l_up)

        x = bandpass_demodulate_downsample(
            x=rearrange(x, 'b c l -> (b c) l'),
            h=self.w,
            sr=self.sr,
            freq=-self.f_center,
            factor=self.factor,
            chunk_size=self.chunk_size
        )  # (b*c, k, l_down)
        out = rearrange(x, '(b c) k l -> b c k l', b=B)

        return out


    def synthesis(self, x: Tensor) -> Tensor:
        r"""Sum subband signals into original signal.

        b: batch_size
        c: audio_channels
        l: audio_samples
        k: n_bands

        Args:
            x: (b, c, k, l)

        Returns:
            out: (b, c, l)
        """
        B = x.shape[0]

        x = upsample_modulate_sum(
            x=rearrange(x, 'b c k l -> (b c) k l'),
            up=self.up,
            sr=self.sr,
            freq=self.f_center,
            factor=self.factor,
            chunk_size=self.chunk_size
        )
        x = rearrange(x, '(b c) l -> b c l', b=B)
        out = analytic_to_real(x)  # (b, c, l)

        return out

    def lowpass(self, banks: list[float], n: int) -> Tensor:
        h = firwin2(
            numtaps=n,
            freq=banks + [self.sr / 2],
            gain=[1., 0., 0.],
            fs=self.sr,
            window=self.window_type
        )
        return torch.from_numpy(h)  # (n,)

    def bandpass(self, banks: list[float], n: int) -> Tensor:
        if banks[0] == 0:
            h = firwin2(
                numtaps=n,
                freq=banks + [self.sr / 2],
                gain=[0., 1., 0., 0.],
                fs=self.sr,
                window=self.window_type
            )
        elif banks[-1] == self.sr / 2:
            h = firwin2(
                numtaps=n,
                freq=[0.] + banks,
                gain=[0., 0., 1., 0.],
                fs=self.sr,
                window=self.window_type
            )
        else:
            h = firwin2(
                numtaps=n,
                freq=[0.] + banks + [self.sr / 2],
                gain=[0., 0., 1., 0., 0.],
                fs=self.sr,
                window=self.window_type
            )
        return torch.from_numpy(h)  # (n,)

    def highpass(self, banks: list[float], n: int) -> Tensor:
        h = firwin2(
            numtaps=n,
            freq=[0.] + banks,
            gain=[0., 0., 1.],
            fs=self.sr,
            window=self.window_type
        )
        return torch.from_numpy(h)  # (n,)


def bandpass_demodulate_downsample(
    x: Tensor,
    h: Tensor,
    sr: int,
    freq: Tensor,
    factor: int,
    chunk_size: int
) -> Tensor:
    r"""
    1. Bandpass: Split signal into subbands
    2. Demodulate: Shift the spectrum to baseband ω=0
    3. Downsample.

    b: batch_size
    l: audio_samples
    k: n_bands
    n: filter_len

    Args:
        x: (b, l)
        h: (k, n)
        sr: int
        freq: (k,)
        factor: int

    Returns:
        out: (b, k, l_down)
    """
    B = x.shape[0]
    K = h.shape[0]
    L = math.ceil(x.shape[1] / factor)

    out = torch.empty(B, K, L, dtype=x.dtype, device=x.device)  # (b, k, l_out)
    k = 0

    while k < K:

        # Split into bands
        e = polyphase_fftconvolve(x, h[k : k + chunk_size, :], factor)  # (b, s, l_up)

        # Demodulate
        e = shift_frequency(e, freq[k : k + chunk_size], sr, factor)  # (b, s, l_up)

        # Downsample
        out[:, k : k + chunk_size, :] = e  # (b, s, l_down)

        k += chunk_size

    return out


def upsample_modulate_sum(
    x: Tensor,
    up: Tensor,
    sr: int,
    freq: Tensor,
    factor: int,
    chunk_size: int
):
    r"""
    1. Upsample: Use sinc function to upsample a signal
    2. Modulate: Shift the baseband signal to ω0
    3. Sum: Sum subband signals

    b: batch_size
    l: audio_samples
    k: n_bands
    n: filter_len

    Args:
        x: (b, k, l_down)
        up: (n,)

    Returns:
        out: (b, l)
    """

    B, K = x.shape[0 : 2]
    L = x.shape[2] * factor

    out = torch.zeros(B, L, dtype=x.dtype, device=x.device)  # (b, l)
    k = 0

    while k < K:

        # Upsample
        e = x[:, k : k + chunk_size, :]
        e = rearrange(e, 'b s l -> (b s) l')  # (b*s, 1, l)
        e = polyphase_fftupsample(e * factor, up, factor)  # (b*s, l)
        e = rearrange(e, '(b s) l -> b s l', b=B)

        # # Modulate
        e = shift_frequency(e, freq[k : k + chunk_size], sr)  # (b, s, l)

        # # Sum
        out.add_(e.sum(dim=1))  # (b, l)

        k += chunk_size

    return out


# =============================================================================
# inference_optimized.py
# =============================================================================

OPTIMIZATION_STEPS = {
    1: "Validate fixed filters once instead of every band group",
    2: "Reuse the analysis input FFT across band groups",
    3: "Cache analysis and synthesis filter FFTs",
    4: "Cache analysis and synthesis modulation tables",
}


class InferenceSubbandFilter(SubbandFilter):
    """Inference-only DSP replacement; parameters and state-dict keys are unchanged."""

    def __init__(self, source: SubbandFilter, level: int):
        torch.nn.Module.__init__(self)
        for name in (
            "sr", "banks", "window_type", "factor", "chunk_size",
            "bandpass_filter_len", "upsample_filter_len",
        ):
            setattr(self, name, getattr(source, name))
        for name in ("w", "f_center", "up"):
            self.register_buffer(name, getattr(source, name))
        self.level = level
        self._cache_key = None
        self._filter_spectra = {}
        self._modulations = {}
        self.register_load_state_dict_post_hook(self._after_load)

    def _after_load(self, module, incompatible_keys):
        self.clear_cache()

    def clear_cache(self):
        self._cache_key = None
        self._filter_spectra.clear()
        self._modulations.clear()

    def cache_bytes(self):
        return sum(
            value.numel() * value.element_size()
            for cache in (self._filter_spectra, self._modulations)
            for value in cache.values()
        )

    def _apply(self, function, recurse=True):
        self.clear_cache()
        return super()._apply(function, recurse=recurse)

    def train(self, mode=True):
        if mode:
            self.clear_cache()
        return super().train(mode)

    def _prepare_cache(self, samples, inputs):
        key = (
            samples, inputs.device, inputs.dtype, self.factor, self.chunk_size, self.sr,
            tuple((id(value), value._version) for value in (self.w, self.up, self.f_center)),
        )
        if key != self._cache_key:
            self.clear_cache()
            assert torch.all(self.w[:, 0] == 0)
            self._cache_key = key

    def _shift(self, inputs, frequencies, stride, key):
        if self.level < 4:
            return shift_frequency(inputs, frequencies, self.sr, stride)
        modulation = self._modulations.get(key)
        if modulation is None:
            if frequencies.dtype in {torch.float16, torch.bfloat16}:
                frequencies = frequencies.float()
            omega = frequencies / (self.sr / 2) * math.pi
            positions = torch.arange(inputs.shape[-1], device=inputs.device) * stride
            modulation = torch.exp(1.j * omega[:, None] * positions[None, :])
            self._modulations[key] = modulation
        return inputs * modulation

    def analysis(self, inputs):
        if self.training or torch.is_grad_enabled() or any(
            torch.is_inference(value) for value in (self.w, self.up, self.f_center)
        ):
            return super().analysis(inputs)
        batch_size = inputs.shape[0]
        inputs = real_to_analytic(inputs)
        self._prepare_cache(inputs.shape[-1], inputs)
        inputs = rearrange(inputs, "b c l -> (b c) l")
        samples = inputs.shape[-1] // self.factor
        output = inputs.new_empty(inputs.shape[0], len(self.banks), samples)
        phases = rearrange(inputs, "b (t1 t2) -> b t2 t1", t2=self.factor)
        filter_samples = self.w.shape[-1] // self.factor
        fft_samples = samples + filter_samples - 1
        input_spectrum = None
        if self.level >= 2:
            input_spectrum = torch.fft.fft(F.pad(phases, (0, fft_samples - samples)))
        for start in range(0, len(self.banks), self.chunk_size):
            weights = self.w[start:start + self.chunk_size]
            weights = rearrange(weights, "k (n1 n2) -> k n2 n1", n2=self.factor)
            assert filter_samples % 2 == 0
            weights = torch.flip(weights, dims=[2])
            if input_spectrum is None:
                filtered = fftconvolve(phases, weights, use_complex_fft=True, mode="full")
            else:
                filter_spectrum = self._filter_spectra.get(("analysis", start))
                if filter_spectrum is None:
                    if weights.dtype in {torch.float16, torch.bfloat16}:
                        weights = weights.float()
                    filter_spectrum = torch.fft.fft(F.pad(weights, (0, fft_samples - filter_samples)))
                    if self.level >= 3:
                        self._filter_spectra[("analysis", start)] = filter_spectrum
                product = torch.einsum("bil,oil->bol", input_spectrum, filter_spectrum)
                filtered = torch.fft.ifft(product, n=fft_samples)
            filtered = filtered[..., filter_samples // 2 - 1:filter_samples // 2 - 1 + samples]
            filtered = self._shift(
                filtered, -self.f_center[start:start + self.chunk_size], self.factor, ("analysis", start),
            )
            output[:, start:start + self.chunk_size] = filtered
        return rearrange(output, "(b c) k l -> b c k l", b=batch_size)


    def synthesis(self, inputs):
        if self.training or torch.is_grad_enabled() or self.level < 3 or any(
            torch.is_inference(value) for value in (self.w, self.up, self.f_center)
        ):
            return super().synthesis(inputs)
        batch_size = inputs.shape[0]
        inputs = rearrange(inputs, "b c k l -> (b c) k l")
        samples = inputs.shape[-1] * self.factor
        self._prepare_cache(samples, inputs)
        weights = rearrange(self.up, "(n1 n2) -> n2 1 n1", n2=self.factor)
        fft_samples = inputs.shape[-1] + weights.shape[-1] - 1
        filter_spectrum = self._filter_spectra.get(("synthesis", 0))
        if filter_spectrum is None:
            if weights.dtype in {torch.float16, torch.bfloat16}:
                weights = weights.float()
            filter_spectrum = torch.fft.fft(F.pad(weights, (0, fft_samples - weights.shape[-1])))
            self._filter_spectra[("synthesis", 0)] = filter_spectrum
        output = inputs.new_zeros(inputs.shape[0], samples)
        for start in range(0, len(self.banks), self.chunk_size):
            chunk = rearrange(inputs[:, start:start + self.chunk_size], "b s l -> (b s) l")
            chunk = rearrange(chunk * self.factor, "b l -> b 1 l")
            spectrum = torch.fft.fft(F.pad(chunk, (0, fft_samples - chunk.shape[-1])))
            product = torch.einsum("bil,oil->bol", spectrum, filter_spectrum)
            expanded = torch.fft.ifft(product, n=fft_samples)
            expanded = rearrange(expanded, "b t2 t1 -> b (t1 t2)")
            expanded = expanded[..., self.up.shape[-1] // 2:self.up.shape[-1] // 2 + samples]
            expanded = rearrange(expanded, "(b s) l -> b s l", b=inputs.shape[0])
            expanded = self._shift(expanded, self.f_center[start:start + self.chunk_size], 1, ("synthesis", start))
            output.add_(expanded.sum(dim=1))
        return rearrange(output, "(b c) l -> b c l", b=batch_size).real


# =============================================================================
# __init__.py
# =============================================================================
# =============================================================================
# Fixed-point subband filtering (opt-in) — utils.py (dsp3)
# =============================================================================

_PHASE_CHUNK_ELEMENTS = 1 << 20


def _phase_steps(x: Tensor, f: Tensor, sr: int) -> Tensor:
    phase_scale = 1 << 32
    phase_half = phase_scale >> 1
    phase_dtype = torch.float64 if x.dtype in (torch.float64, torch.complex128) else torch.float32
    frequencies = f.to(device=x.device, dtype=phase_dtype)
    scaled_frequencies = frequencies.abs().mul(phase_scale).round().to(torch.int64)
    steps = torch.div(scaled_frequencies + sr // 2, sr, rounding_mode="floor")
    steps.mul_(frequencies.sign().to(torch.int64))
    return steps.add_(phase_half).bitwise_and_(phase_scale - 1).sub_(phase_half)


def _shift_frequency_reference(x: Tensor, f: Tensor, sr: int, n: int = 1) -> Tensor:
    r"""Shift frequency to -ω0.

    X(j(ω-ω0)) = e^{j(ω0)n}X(jω)

    k: n_bands
    l: audio_samples

    Args:
        x: (any, k, l)
        f: (k,)

    Returns:
        out: (any, k, l)

    Frequencies are fixed, non-learnable values quantized to sr / 2**32 Hz.
    Integer phase wraps before conversion to radians, so float32 phase
    evaluation stays within [-pi, pi) regardless of signal length. Phase
    workspace is tiled; float32/complex64 inputs require no float64 tensors.
    """
    phase_scale = 1 << 32
    phase_half = phase_scale >> 1
    phase_mask = phase_scale - 1
    phase_dtype = torch.float64 if x.dtype in (torch.float64, torch.complex128) else torch.float32
    oscillator_dtype = torch.complex128 if phase_dtype == torch.float64 else torch.complex64
    phase_steps = _phase_steps(x, f, sr)

    length = x.shape[-1]
    oscillator = torch.empty((f.numel(), length), dtype=oscillator_dtype, device=x.device)
    chunk_length = max(1, _PHASE_CHUNK_ELEMENTS // max(1, f.numel()))
    for start in range(0, length, chunk_length):
        end = min(start + chunk_length, length)
        sample_indices = torch.arange(start, end, dtype=torch.int64, device=x.device)
        sample_indices.mul_(n).bitwise_and_(phase_mask)
        phase_words = phase_steps[:, None] * sample_indices[None, :]
        phase_words.add_(phase_half).bitwise_and_(phase_mask).sub_(phase_half)
        phase = phase_words.to(phase_dtype).mul_(2 * math.pi / phase_scale)
        torch.polar(torch.ones_like(phase), phase, out=oscillator[:, start:end])
        del sample_indices, phase_words, phase

    return x * oscillator


class _ShiftFrequency(torch.autograd.Function):
    @staticmethod
    def forward(ctx, signal, steps, stride, kernel, conjugate):
        ctx.save_for_backward(steps)
        ctx.stride = stride
        ctx.kernel = kernel
        ctx.conjugate = conjugate
        signal = signal.resolve_conj().resolve_neg()
        if not signal.is_contiguous() and not (signal.ndim in (2, 3) and signal.stride(-1) == 1):
            signal = signal.contiguous()
        return kernel(signal, steps, stride, conjugate)

    @staticmethod
    def backward(ctx, gradient):
        steps, = ctx.saved_tensors
        result = _ShiftFrequency.apply(
            gradient, steps, ctx.stride, ctx.kernel, not ctx.conjugate,
        )
        return result, None, None, None, None


def shift_frequency_fixed(x: Tensor, f: Tensor, sr: int, n: int = 1) -> Tensor:
    """Fixed-point modulation with fused CPU/CUDA/MPS kernels when available.

    Frequencies are non-learnable. Other devices, real inputs, and broadcast
    band shapes retain the original PyTorch implementation.
    """
    if x.dtype not in (torch.complex64, torch.complex128) or x.ndim < 2 or f.ndim != 1 or x.shape[-2] != f.numel():
        return _shift_frequency_reference(x, f, sr, n)
    if x.device.type == "cpu":
        is_available = is_available_cpu

        kernel = shift_frequency_cpu
    elif x.device.type == "cuda":
        is_available = is_available_cuda

        kernel = partial(shift_frequency_cuda, sample_rate=sr)
    elif x.device.type == "mps" and x.dtype == torch.complex64:
        is_available = is_available_mps

        kernel = partial(shift_frequency_mps, sample_rate=sr)
    else:
        return _shift_frequency_reference(x, f, sr, n)
    if not is_available():
        return _shift_frequency_reference(x, f, sr, n)
    if x.device.type == "cuda":
        values = f.to(device=x.device).contiguous()
    elif x.device.type == "mps":
        values = f.to(device=x.device, dtype=torch.float32).contiguous()
    else:
        values = _phase_steps(x, f, sr)
    return _ShiftFrequency.apply(x, values, n % (1 << 32), kernel, False)


# =============================================================================
# cpu_kernel.py
# =============================================================================

@lru_cache(maxsize=1)
def _load_cpu_extension():
    from torch.utils.cpp_extension import load

    flags = ["-O3", "-ffp-contract=off"]
    if sys.platform == "linux" and torch.backends.openmp.is_available():
        flags.append("-fopenmp")
    if platform.machine() in ("x86_64", "AMD64") and torch.backends.cpu.get_cpu_capability() in ("AVX2", "AVX512"):
        flags.extend(["-mavx2", "-mfma", "-DCPU_CAPABILITY_AVX2", "-DCPU_CAPABILITY=AVX2"])
    try:
        return load(
            name="subroformer_dsp3_native_cpu",
            sources=[str(Path(__file__).parent / "csrc" / "shift_frequency_cpu.cpp")],
            extra_cflags=flags,
            with_cuda=False,
            is_python_module=False,
            verbose=False,
        )
    except Exception as error:
        raise RuntimeError(
            "The DSP3 CPU kernel requires a working C++ compiler and Ninja. "
            "Select the reference backend if native compilation is unavailable."
        ) from error


@lru_cache(maxsize=1)
def is_available_cpu() -> bool:
    """Check whether the native extension can be loaded, compiling if needed."""
    try:
        _load_cpu_extension()
    except RuntimeError:
        return False
    return True


def shift_frequency_cpu(
    x: Tensor, phase_steps: Tensor, n: int = 1, conjugate: bool = False
) -> Tensor:
    """Modulate contiguous or row-contiguous CPU input from signed phase steps."""
    import torch

    _load_cpu_extension()
    return torch.ops.subroformer_dsp3_native.shift_frequency_cpu(
        x, phase_steps, n % (1 << 32), conjugate
    )


# =============================================================================
# cuda_kernel.py
# =============================================================================

try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None


def is_available_cuda() -> bool:
    """Return whether the optional Triton implementation can be imported."""
    return triton is not None


if triton is not None:

    @triton.jit
    def _shift_frequency_kernel(
        source,
        steps,
        target,
        elements: tl.constexpr,
        length: tl.constexpr,
        bands: tl.constexpr,
        band_stride: tl.constexpr,
        batch_stride: tl.constexpr,
        stride: tl.constexpr,
        double: tl.constexpr,
        conjugate: tl.constexpr,
        sample_rate: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
        valid = offsets < elements
        band_indices = (offsets // length) % bands
        sample_indices = (offsets % length).to(tl.uint32)
        values = tl.load(steps + band_indices, valid, other=0)
        if sample_rate is not None:
            if double:
                frequencies = values.to(tl.float64)
            else:
                frequencies = values.to(tl.float32)
            scaled = libdevice.nearbyint(tl.abs(frequencies) * 4294967296.0).to(tl.int64)
            magnitude = (scaled + sample_rate // 2) // sample_rate
            phase_steps = tl.where(frequencies < 0, -magnitude, magnitude).to(tl.uint32)
        else:
            phase_steps = values.to(tl.uint32)
        phase_words = (phase_steps * sample_indices * stride).to(tl.int32)
        # 2 * pi / 2**32 as a literal: torch.compile (inductor) re-generates user triton kernels
        # without the module globals, so `math.pi` here fails with "NameError: math is not defined"
        if double:
            phase = phase_words.to(tl.float64) * tl.full((), 1.4629180792671596e-09, tl.float64)
        else:
            phase = phase_words.to(tl.float32) * 1.4629180792671596e-09
        if double:
            real_oscillator = libdevice.cos(phase)
            imag_oscillator = libdevice.sin(phase)
        else:
            real_oscillator = tl.cos(phase)
            imag_oscillator = tl.sin(phase)
        if conjugate:
            imag_oscillator = -imag_oscillator
        source_offsets = (offsets // (length * bands)) * batch_stride + band_indices * band_stride + offsets % length
        real_source = tl.load(source + 2 * source_offsets, valid, other=0)
        imag_source = tl.load(source + 2 * source_offsets + 1, valid, other=0)
        real_target = real_source * real_oscillator - imag_source * imag_oscillator
        imag_target = real_source * imag_oscillator + imag_source * real_oscillator
        tl.store(target + 2 * offsets, real_target, valid)
        tl.store(target + 2 * offsets + 1, imag_target, valid)


def shift_frequency_cuda(
    x: Tensor, phase_steps: Tensor, n: int = 1, conjugate: bool = False,
    *, sample_rate: int | None = None,
) -> Tensor:
    """Modulate complex CUDA input from phase steps or frequencies in Hz.

    With sample_rate set, quantization is fused into the modulation kernel.
    """
    if not is_available_cuda():
        raise RuntimeError("The DSP3 CUDA kernel requires Triton.")
    if x.device.type != "cuda" or x.dtype not in (torch.complex64, torch.complex128):
        raise ValueError("Expected a complex64 or complex128 CUDA input.")
    if x.ndim < 2 or not (x.is_contiguous() or (x.ndim <= 3 and x.stride(-1) == 1)):
        raise ValueError("Expected a contiguous input with at least two dimensions.")
    if sample_rate is not None and (not isinstance(sample_rate, int) or sample_rate <= 0):
        raise ValueError("Expected a positive integer sample rate.")
    if (
        phase_steps.device != x.device
        or (sample_rate is None and phase_steps.dtype != torch.int64)
        or phase_steps.ndim != 1
        or not phase_steps.is_contiguous()
        or phase_steps.numel() != x.shape[-2]
    ):
        raise ValueError("Expected contiguous phase steps (int64) or frequencies matching the input bands and device.")
    result = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if x.numel() == 0:
        return result
    block = 256
    with torch.cuda.device(x.device):
        _shift_frequency_kernel[(triton.cdiv(x.numel(), block),)](
            torch.view_as_real(x),
            phase_steps,
            torch.view_as_real(result),
            x.numel(),
            x.shape[-1],
            x.shape[-2],
            x.stride(-2),
            x.stride(-3) if x.ndim == 3 else x.shape[-2] * x.shape[-1],
            n & ((1 << 32) - 1),
            x.dtype == torch.complex128,
            conjugate,
            sample_rate,
            block,
            enable_fp_fusion=False,
        )
    return result


# =============================================================================
# mps_kernel.py
# =============================================================================

def is_available_mps() -> bool:
    """Return whether this runtime supports compiling MPS shaders."""
    return torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")


@lru_cache(maxsize=1)
def _load_mps_library():
    source = Path(__file__).parent / "csrc" / "shift_frequency_mps.metal"
    return torch.mps.compile_shader(source.read_text())


def shift_frequency_mps(
    x: Tensor, phase_steps: Tensor, n: int = 1, conjugate: bool = False,
    *, sample_rate: int | None = None,
) -> Tensor:
    """Modulate row-contiguous complex64 MPS input from steps or float32 Hz."""
    if not is_available_mps():
        raise RuntimeError("The DSP3 MPS kernel requires torch.mps.compile_shader and MPS.")
    if x.device.type != "mps" or x.dtype != torch.complex64:
        raise ValueError("Expected a complex64 MPS input.")
    if x.ndim < 2 or not (x.is_contiguous() or (x.ndim <= 3 and x.stride(-1) == 1)):
        raise ValueError("Expected a contiguous input with at least two dimensions.")
    if sample_rate is not None and (not isinstance(sample_rate, int) or sample_rate <= 0):
        raise ValueError("Expected a positive integer sample rate.")
    if (
        phase_steps.device != x.device
        or phase_steps.dtype != (torch.int64 if sample_rate is None else torch.float32)
        or phase_steps.ndim != 1
        or not phase_steps.is_contiguous()
        or phase_steps.numel() != x.shape[-2]
    ):
        raise ValueError("Expected contiguous phase steps (int64) or frequencies (float32) matching the input bands and device.")
    result = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if x.numel() == 0:
        return result
    library = _load_mps_library()
    if sample_rate is not None:
        steps = torch.empty_like(phase_steps, dtype=torch.int64)
        library.quantize_frequencies(phase_steps, steps, sample_rate, threads=phase_steps.numel())
        phase_steps = steps
    library.shift_frequency(
        torch.view_as_real(x), phase_steps, torch.view_as_real(result),
        x.shape[-1], x.shape[-2], x.stride(-2),
        x.stride(-3) if x.ndim == 3 else x.shape[-2] * x.shape[-1],
        n % (1 << 32), int(conjugate),
        threads=(x.shape[-1], x.shape[-2], x.numel() // (x.shape[-1] * x.shape[-2])),
        group_size=(256, 1, 1),
    )
    return result


# =============================================================================
# Fixed-point SubbandFilter (subband_fast_triangle.py — DSP3)
# =============================================================================

class FixedPointSubbandFilter(SubbandFilter):
    """Reuse the original filters, replacing only the modulation path."""

    def analysis(self, x: Tensor) -> Tensor:
        """Convert [batch, channels, samples] to decimated complex subbands."""
        batch, channels, length = x.shape
        analytic = real_to_analytic(x).reshape(batch * channels, length)
        band_count = len(self.banks)
        downsampled_length = math.ceil(length / self.factor)
        output = torch.empty(
            batch * channels, band_count, downsampled_length,
            dtype=analytic.dtype, device=analytic.device,
        )
        for start in range(0, band_count, self.chunk_size):
            end = min(start + self.chunk_size, band_count)
            subbands = polyphase_fftconvolve(analytic, self.w[start:end], self.factor)
            output[:, start:end] = shift_frequency_fixed(
                subbands, -self.f_center[start:end], self.sr, self.factor,
            )
        return output.reshape(batch, channels, band_count, downsampled_length)

    def synthesis(self, x: Tensor) -> Tensor:
        """Reconstruct [batch, channels, samples] from complex subbands."""
        batch, channels, band_count, downsampled_length = x.shape
        subbands = x.reshape(batch * channels, band_count, downsampled_length)
        length = downsampled_length * self.factor
        output = torch.zeros(batch * channels, length, dtype=x.dtype, device=x.device)
        for start in range(0, band_count, self.chunk_size):
            end = min(start + self.chunk_size, band_count)
            band_chunk = subbands[:, start:end].reshape(-1, downsampled_length)
            interpolated = polyphase_fftupsample(band_chunk * self.factor, self.up, self.factor)
            interpolated = interpolated.reshape(batch * channels, end - start, length)
            modulated = shift_frequency_fixed(interpolated, self.f_center[start:end], self.sr)
            output.add_(modulated.sum(dim=1))
        return analytic_to_real(output.reshape(batch, channels, length))


# =============================================================================
# inference_optimized.py (DSP3 — MPS FFT cache retention)
# =============================================================================

class InferenceFixedPointSubbandFilter(InferenceSubbandFilter, FixedPointSubbandFilter):
    """Reuse cache invalidation and FFTs; uncached paths resolve to DSP3 via MRO."""

    def __init__(self, source: SubbandFilter):
        super().__init__(source, level=3)

    def _shift(self, inputs, frequencies, stride, key):
        return shift_frequency_fixed(inputs, frequencies, self.sr, stride)

    def analysis(self, inputs):
        if inputs.device.type != "mps":
            return FixedPointSubbandFilter.analysis(self, inputs)
        return super().analysis(inputs)

    def synthesis(self, inputs):
        if inputs.device.type != "mps":
            return FixedPointSubbandFilter.synthesis(self, inputs)
        return super().synthesis(inputs)
