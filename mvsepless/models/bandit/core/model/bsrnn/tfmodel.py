import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import rnn

import torch.backends.cuda


class TimeFrequencyModellingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class ResidualRNN(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        rnn_dim: int,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
        use_batch_trick: bool = True,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm = nn.LayerNorm(emb_dim)
        else:
            self.norm = nn.GroupNorm(num_groups=emb_dim, num_channels=emb_dim)

        self.rnn = rnn.__dict__[rnn_type](
            input_size=emb_dim,
            hidden_size=rnn_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.fc = nn.Linear(
            in_features=rnn_dim * (2 if bidirectional else 1), out_features=emb_dim
        )

        self.use_batch_trick = use_batch_trick
        if not self.use_batch_trick:
            warnings.warn("NOT USING BATCH TRICK IS EXTREMELY SLOW!!")

    def forward(self, z):

        z0 = torch.clone(z)

        if self.use_layer_norm:
            z = self.norm(z)
        else:
            z = torch.permute(z, (0, 3, 1, 2))

            z = self.norm(z)

            z = torch.permute(z, (0, 2, 3, 1))

        batch, n_uncrossed, n_across, emb_dim = z.shape

        if self.use_batch_trick:
            z = torch.reshape(z, (batch * n_uncrossed, n_across, emb_dim))

            z = self.rnn(z.contiguous())[0]

            z = torch.reshape(z, (batch, n_uncrossed, n_across, -1))
        else:
            zlist = []
            for i in range(n_uncrossed):
                zi = self.rnn(z[:, i, :, :])[0]
                zlist.append(zi)

            z = torch.stack(zlist, dim=1)

        z = self.fc(z)

        z = z + z0

        return z


class SeqBandModellingModule(TimeFrequencyModellingModule):
    def __init__(
        self,
        n_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
        parallel_mode=False,
    ) -> None:
        super().__init__()
        self.seqband = nn.ModuleList([])

        if parallel_mode:
            for _ in range(n_modules):
                self.seqband.append(
                    nn.ModuleList(
                        [
                            ResidualRNN(
                                emb_dim=emb_dim,
                                rnn_dim=rnn_dim,
                                bidirectional=bidirectional,
                                rnn_type=rnn_type,
                            ),
                            ResidualRNN(
                                emb_dim=emb_dim,
                                rnn_dim=rnn_dim,
                                bidirectional=bidirectional,
                                rnn_type=rnn_type,
                            ),
                        ]
                    )
                )
        else:

            for _ in range(2 * n_modules):
                self.seqband.append(
                    ResidualRNN(
                        emb_dim=emb_dim,
                        rnn_dim=rnn_dim,
                        bidirectional=bidirectional,
                        rnn_type=rnn_type,
                    )
                )

        self.parallel_mode = parallel_mode

    def forward(self, z):

        if self.parallel_mode:
            for sbm_pair in self.seqband:
                sbm_t, sbm_f = sbm_pair[0], sbm_pair[1]
                zt = sbm_t(z)
                zf = sbm_f(z.transpose(1, 2))
                z = zt + zf.transpose(1, 2)
        else:
            for sbm in self.seqband:
                z = sbm(z)
                z = z.transpose(1, 2)

        q = z
        return q


class ResidualTransformer(nn.Module):
    def __init__(
        self,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.tf = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=4, dim_feedforward=rnn_dim, batch_first=True
        )

        self.is_causal = not bidirectional
        self.dropout = dropout

    def forward(self, z):
        batch, n_uncrossed, n_across, emb_dim = z.shape
        z = torch.reshape(z, (batch * n_uncrossed, n_across, emb_dim))
        z = self.tf(z, is_causal=self.is_causal)
        z = torch.reshape(z, (batch, n_uncrossed, n_across, emb_dim))

        return z


class TransformerTimeFreqModule(TimeFrequencyModellingModule):
    def __init__(
        self,
        n_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.seqband = nn.ModuleList([])

        for _ in range(2 * n_modules):
            self.seqband.append(
                ResidualTransformer(
                    emb_dim=emb_dim,
                    rnn_dim=rnn_dim,
                    bidirectional=bidirectional,
                    dropout=dropout,
                )
            )

    def forward(self, z):
        z = self.norm(z)

        for sbm in self.seqband:
            z = sbm(z)
            z = z.transpose(1, 2)

        q = z
        return q


class ResidualConvolution(nn.Module):
    def __init__(
        self,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(emb_dim, affine=True)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=emb_dim,
                out_channels=rnn_dim,
                kernel_size=(3, 3),
                padding="same",
                stride=(1, 1),
            ),
            nn.Tanhshrink(),
        )

        self.is_causal = not bidirectional
        self.dropout = dropout

        self.fc = nn.Conv2d(
            in_channels=rnn_dim,
            out_channels=emb_dim,
            kernel_size=(1, 1),
            padding="same",
            stride=(1, 1),
        )

    def forward(self, z):

        z0 = torch.clone(z)

        z = self.norm(z)
        z = self.conv(z)
        z = self.fc(z)
        z = z + z0

        return z


class ConvolutionalTimeFreqModule(TimeFrequencyModellingModule):
    def __init__(
        self,
        n_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seqband = torch.jit.script(
            nn.Sequential(
                *[
                    ResidualConvolution(
                        emb_dim=emb_dim,
                        rnn_dim=rnn_dim,
                        bidirectional=bidirectional,
                        dropout=dropout,
                    )
                    for _ in range(2 * n_modules)
                ]
            )
        )

    def forward(self, z):

        z = torch.permute(z, (0, 3, 1, 2))

        z = self.seqband(z)

        z = torch.permute(z, (0, 2, 3, 1))

        return z
