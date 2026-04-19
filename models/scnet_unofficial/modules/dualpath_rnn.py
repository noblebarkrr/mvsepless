import torch
import torch.nn as nn
import torch.nn.functional as Func


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return Func.normalize(x, dim=-1) * self.scale * self.gamma


class MambaModule(nn.Module):
    def __init__(self, d_model, d_state, d_conv, d_expand):
        super().__init__()
        self.norm = RMSNorm(dim=d_model)
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, d_expand=d_expand
        )

    def forward(self, x):
        x = x + self.mamba(self.norm(x))
        return x


class RNNModule(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, bidirectional: bool = True):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups=1, num_channels=input_dim)
        self.rnn = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.groupnorm(x)
        x = x.transpose(1, 2)

        x, (hidden, _) = self.rnn(x)
        x = self.fc(x)
        return x


class RFFTModule(nn.Module):

    def __init__(self, inverse: bool = False):
        super().__init__()
        self.inverse = inverse

    def forward(self, x: torch.Tensor, time_dim: int) -> torch.Tensor:
        dtype = x.dtype
        B, F, T, D = x.shape

        x = x.float()

        if not self.inverse:
            x = torch.fft.rfft(x, dim=2)
            x = torch.view_as_real(x)
            x = x.reshape(B, F, T // 2 + 1, D * 2)
        else:
            x = x.reshape(B, F, T, D // 2, 2)
            x = torch.view_as_complex(x)
            x = torch.fft.irfft(x, n=time_dim, dim=2)

        x = x.to(dtype)
        return x

    def extra_repr(self) -> str:
        return f"inverse={self.inverse}"


class DualPathRNN(nn.Module):

    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        hidden_dim: int,
        use_mamba: bool = False,
        d_state: int = 16,
        d_conv: int = 4,
        d_expand: int = 2,
    ):
        super().__init__()

        if use_mamba:
            from mamba_ssm.modules.mamba_simple import Mamba

            net = MambaModule
            dkwargs = {
                "d_model": input_dim,
                "d_state": d_state,
                "d_conv": d_conv,
                "d_expand": d_expand,
            }
            ukwargs = {
                "d_model": input_dim * 2,
                "d_state": d_state,
                "d_conv": d_conv,
                "d_expand": d_expand * 2,
            }
        else:
            net = RNNModule
            dkwargs = {"input_dim": input_dim, "hidden_dim": hidden_dim}
            ukwargs = {"input_dim": input_dim * 2, "hidden_dim": hidden_dim * 2}

        self.layers = nn.ModuleList()
        for i in range(1, n_layers + 1):
            kwargs = dkwargs if i % 2 == 1 else ukwargs
            layer = nn.ModuleList(
                [
                    net(**kwargs),
                    net(**kwargs),
                    RFFTModule(inverse=(i % 2 == 0)),
                ]
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        time_dim = x.shape[2]

        for time_layer, freq_layer, rfft_layer in self.layers:
            B, F, T, D = x.shape

            x = x.reshape((B * F), T, D)
            x = time_layer(x)
            x = x.reshape(B, F, T, D)
            x = x.permute(0, 2, 1, 3)

            x = x.reshape((B * T), F, D)
            x = freq_layer(x)
            x = x.reshape(B, T, F, D)
            x = x.permute(0, 2, 1, 3)

            x = rfft_layer(x, time_dim)

        return x
