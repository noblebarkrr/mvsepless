from functools import wraps
from packaging import version
from collections import namedtuple

import os
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce


def _print_once(msg):
    printed = False

    def inner():
        nonlocal printed
        if not printed:
            print(msg)
            printed = True

    return inner


# Проверяем доступность SageAttention
try:
    from sageattention import sageattn
    _has_sage_attention = True
except ImportError:
    _has_sage_attention = False
    _print_sage_not_found = _print_once(
        "SageAttention not found. Will fall back to PyTorch SDPA (if available) or manual einsum."
    )
    _print_sage_not_found()


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


class Attend(nn.Module):
    def __init__(self, dropout=0.0, flash=False, scale=None):
        super().__init__()
        self.scale = scale
        self.dropout = dropout

        self.use_sage = flash and _has_sage_attention
        self.use_pytorch_sdpa = False
        self._sdpa_checked = False
        self.flash = flash
        
        # Инициализируем сообщения
        self._init_messages = False
        
        if flash and not self.use_sage:
            if not self._sdpa_checked:
                if version.parse(torch.__version__) >= version.parse("2.0.0"):
                    self.use_pytorch_sdpa = True
                self._sdpa_checked = True

        self.attn_dropout = nn.Dropout(dropout)

    def _print_init_messages(self):
        """Печатаем сообщения инициализации один раз"""
        if self._init_messages:
            return
            
        if self.flash:
            if self.use_sage:
                print_once = _print_once("Using SageAttention backend.")
                print_once()
            elif self.use_pytorch_sdpa:
                print_once = _print_once(
                    "Using PyTorch SDPA backend (FlashAttention-2, Memory-Efficient, or Math)."
                )
                print_once()
            else:
                print_once = _print_once(
                    "Flash attention requested but Pytorch < 2.0 and SageAttention not found. Falling back to einsum."
                )
                print_once()
        
        self._init_messages = True

    def forward(self, q, k, v):
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        # Печатаем сообщения инициализации при первом вызове
        self._print_init_messages()

        # Пробуем SageAttention если доступен
        if self.use_sage and self.flash:
            try:
                # Исправленный вызов: убрали повторный try-except
                out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
                return out
            except Exception as e:
                print(f"SageAttention failed with error: {e}. Falling back.")
                self.use_sage = False
                if not self._sdpa_checked:
                    if version.parse(torch.__version__) >= version.parse("2.0.0"):
                        self.use_pytorch_sdpa = True
                        print_once = _print_once(
                            "Falling back to PyTorch SDPA."
                        )
                        print_once()
                    else:
                        print_once = _print_once("Falling back to einsum.")
                        print_once()
                    self._sdpa_checked = True

        # Пробуем PyTorch SDPA если доступен
        if self.use_pytorch_sdpa and self.flash:
            try:
                # Для PyTorch >= 2.0
                if version.parse(torch.__version__) >= version.parse("2.0.0"):
                    with torch.backends.cuda.sdp_kernel(
                        enable_flash=True, enable_math=True, enable_mem_efficient=True
                    ):
                        out = F.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            attn_mask=None,
                            dropout_p=self.dropout if self.training else 0.0,
                            is_causal=False,
                        )
                    return out
            except Exception as e:
                print(f"PyTorch SDPA failed with error: {e}. Falling back to einsum.")
                self.use_pytorch_sdpa = False

        # Fallback на einsum (работает в PyTorch 1.13+)
        scale = default(self.scale, q.shape[-1] ** -0.5)

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out