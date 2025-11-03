import torch
import torch.nn as nn

import torch

from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack, reduce, repeat
from librosa import filters
from .modules import BandSplit, MaskEstimator, Transformer, pack_one, unpack_one
from functools import partial

class MelBandRoformerWSA(nn.Module):
    def __init__(
            self,
            dim: int = 384,
            depth: int = 6,
            num_stems: int = 1,
            num_bands: int = 60,
            dim_head: int = 64,
            heads: int = 8,
            attn_dropout: float = 0.0,
            ff_dropout: float = 0.0,
            sample_rate: int = 44100,
            stft_n_fft: int = 2048,
            stft_hop_length: int = 441,
            stft_win_length: int = 2048,
            stft_normalized: bool = False,
            wsa_window_len: int = 10,
            num_sink_tokens: int = 8,
            n_wsa_sinks: int = 8,
            dim_freqs_in: int = 1025,
            flash_attn: bool = True,
            use_flex_attention: bool = True,
            freq_transformer_depth: int = 1,
            mask_estimator_depth=1,
            multi_stft_resolution_loss_weight=1.0,
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (
              4096,
              2048,
              1024,
              512,
              256,
            ),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn: Callable = torch.hann_window,
            match_input_audio_length=False,  # if True, pad output tensor to match length of input tensor
            stereo=True,
            time_transformer_depth=1,
            window_size=10
            
    ):
        super().__init__()
        
        # Store configuration parameters
        self.audio_channels = 2
        self.num_stems = num_stems
        self.n_wsa_sinks = num_sink_tokens
        self.dim = dim
        self.depth = depth
        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length
        self.stft_win_length = stft_win_length
        self.stft_normalized = stft_normalized
        self.wsa_window_len = wsa_window_len
        
        # Store transformer configuration
        self.time_transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            wsa_window_len=wsa_window_len,
            n_wsa_sinks=n_wsa_sinks
        )
        self.freq_transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )

        # Build components
        self._build_stft_config()
        self._build_mel_filter_bank()
        self._build_transformer_layers()
        self._build_band_split_and_mask_estimators()
        self._build_sink_tokens()
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for linear and conv layers"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _build_stft_config(self):
        """Build STFT configuration and window function"""
        self.stft_window_fn = partial(torch.hann_window, self.stft_win_length)
        self.stft_kwargs = dict(
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop_length,
            win_length=self.stft_win_length,
            normalized=self.stft_normalized
        )

    def _build_mel_filter_bank(self):
        """Build mel filter bank and frequency indices"""
        # Get number of frequencies from STFT
        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, window=torch.ones(self.stft_n_fft), return_complex=True).shape[1]

        # Create mel filter bank with librosa.filters.mel as in section 2 of paper
        mel_filter_bank_numpy = filters.mel(sr=self.sample_rate, n_fft=self.stft_n_fft, n_mels=self.num_bands)
        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

        # Fix edge cases
        mel_filter_bank[0][0] = 1.  # First frequency
        mel_filter_bank[-1, -1] = 1.  # Last frequency

        # Binary mask as in paper (estimated masks are averaged for overlapping regions)
        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(dim=0).all(), 'all frequencies need to be covered by all bands for now'

        # Create frequency indices for band splitting
        repeated_freq_indices = repeat(torch.arange(freqs), 'f -> b f', b=self.num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        # if self.stereo:
        freq_indices = repeat(freq_indices, 'f -> f s', s=2)
        freq_indices = freq_indices * 2 + torch.arange(2)
        freq_indices = rearrange(freq_indices, 'f s -> (f s)')

        # Register buffers
        self.register_buffer('freq_indices', freq_indices, persistent=False)
        self.register_buffer('freqs_per_band', freqs_per_band, persistent=False)

        # Calculate frequency statistics
        num_freqs_per_band = reduce(freqs_per_band, 'b f -> b', 'sum')
        num_bands_per_freq = reduce(freqs_per_band, 'b f -> f', 'sum')

        self.register_buffer('num_freqs_per_band', num_freqs_per_band, persistent=False)
        self.register_buffer('num_bands_per_freq', num_bands_per_freq, persistent=False)

    def _build_transformer_layers(self):
        """Build transformer layers with rotary embeddings"""
        self.layers = nn.ModuleList([])
        
        time_rotary_embed = RotaryEmbedding(dim=self.time_transformer_kwargs['dim_head'])
        freq_rotary_embed = RotaryEmbedding(dim=self.freq_transformer_kwargs['dim_head'])

        for _ in range(self.depth):
            tran_modules = []
            tran_modules.append(
                Transformer(rotary_embed=time_rotary_embed, **self.time_transformer_kwargs)
            )
            tran_modules.append(
                Transformer(rotary_embed=freq_rotary_embed, **self.freq_transformer_kwargs)
            )
            self.layers.append(nn.ModuleList(tran_modules))

    def _build_band_split_and_mask_estimators(self):
        """Build band split module and mask estimators"""
        # Calculate input dimensions for each band
        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in self.num_freqs_per_band.tolist())

        # Band split module
        self.band_split = BandSplit(
            dim=self.dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        # Mask estimators
        self.mask_estimators = nn.ModuleList([])
        for _ in range(1):
            mask_estimator = MaskEstimator(
                dim=self.dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=2,
                mlp_expansion_factor=4,
            )
            self.mask_estimators.append(mask_estimator)

    def _build_sink_tokens(self):
        """Build learnable sink tokens for efficient attention"""
        if self.n_wsa_sinks > 0:
            self.sink_tokens = nn.Parameter(
                torch.randn(self.n_wsa_sinks, self.num_bands, self.dim)
            )
            print(f"Using {self.n_wsa_sinks} sink tokens for attention")
        else:
            self.sink_tokens = None


    def forward(self, raw_audio):
        """
        Main forward pass for audio source separation
        
        einops notation:
        b - batch
        f - freq
        t - time
        s - audio channel (2 for stereo)
        n - number of 'stems'
        c - complex (2)
        d - feature dimension
        """
        # Preprocess input audio
        raw_audio, batch_info = self._preprocess_audio(raw_audio)
        
        # Convert to STFT representation
        stft_repr = self._audio_to_stft(raw_audio, batch_info)
        
        # Extract features and apply band splitting
        features = self._extract_features(stft_repr, batch_info)
        
        # Apply transformer layers with sink tokens
        processed_features = self._apply_transformer_layers(features)
        
        # Generate masks for each stem
        masks = self._generate_masks(processed_features)
        
        # Apply masks and reconstruct audio
        recon_audio = self._reconstruct_audio(stft_repr, masks, batch_info)
        
        return recon_audio

    def _preprocess_audio(self, raw_audio):
        """Preprocess input audio and validate dimensions"""
        device = raw_audio.device

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        batch, channels, raw_audio_length = raw_audio.shape

        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')
        
        batch_info = {
            'batch_size': batch,
            'channels': channels,
            'device': device,
            'packed_shape': batch_audio_channel_packed_shape
        }
        
        return raw_audio, batch_info

    def _audio_to_stft(self, raw_audio, batch_info):
        """Convert audio to STFT representation"""
        stft_window = self.stft_window_fn(device=batch_info['device'])
        
        stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        stft_repr = torch.view_as_real(stft_repr)
        
        stft_repr = unpack_one(stft_repr, batch_info['packed_shape'], '* f t c')
        
        # Merge stereo/mono into frequency dimension for band splitting
        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')
        
        return stft_repr

    def _extract_features(self, stft_repr, batch_info):
        """Extract features using band splitting"""
        batch_arange = torch.arange(batch_info['batch_size'], device=batch_info['device'])[..., None]
        
        # Index frequencies for band splitting
        x = stft_repr[batch_arange, self.freq_indices]
        
        # Fold complex dimensions into frequency dimension
        x = rearrange(x, 'b f t c -> b t (f c)')
        
        # Apply band split
        x = self.band_split(x)
        
        return x

    def _apply_transformer_layers(self, features):
        """Apply transformer layers with sink tokens"""
        x = features
        
        # Add sink tokens at the beginning of the time dimension
        if self.sink_tokens is not None:
            batch_size = x.shape[0]
            sinks = repeat(self.sink_tokens, 'n f d -> b n f d', b=batch_size)
            x = torch.cat([sinks, x], dim=1)
        
        # Apply axial/hierarchical attention
        for transformer_block in self.layers:
            time_transformer, freq_transformer = transformer_block
            
            # Time transformer
            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')
            x = time_transformer(x)
            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            
            # Frequency transformer
            x, ps = pack([x], '* f d')
            x = freq_transformer(x)
            x, = unpack(x, ps, '* f d')
        
        # Remove sink tokens before mask estimation
        if self.sink_tokens is not None:
            x = x[:, self.n_wsa_sinks:, :, :]
        
        return x

    def _generate_masks(self, processed_features):
        """Generate masks for each stem"""
        masks = torch.stack([fn(processed_features) for fn in self.mask_estimators], dim=1)
        masks = rearrange(masks, 'b n t (f c) -> b n f t c', c=2)
        return masks

    def _reconstruct_audio(self, stft_repr, masks, batch_info):
        """Apply masks and reconstruct audio"""
        batch = batch_info['batch_size']
        channels = batch_info['channels']
        device = batch_info['device']
        num_stems = len(self.mask_estimators)
        
        # Prepare STFT for modulation
        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')
        
        # Convert to complex representation
        stft_repr = torch.view_as_complex(stft_repr)
        masks = torch.view_as_complex(masks)
        masks = masks.type(stft_repr.dtype)
        
        # Average masks for overlapping frequencies
        scatter_indices = repeat(self.freq_indices, 'f -> b n f t', b=batch, n=num_stems, t=stft_repr.shape[-1])
        stft_repr_expanded_stems = repeat(stft_repr, 'b 1 ... -> b n ...', n=num_stems)
        masks_summed = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(2, scatter_indices, masks)
        
        denom = repeat(self.num_bands_per_freq, 'f -> (f r) 1', r=channels)
        masks_averaged = masks_summed / denom.clamp(min=1e-8)
        
        # Apply masks to STFT
        stft_repr = stft_repr * masks_averaged
        
        # Convert back to real representation for ISTFT
        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)
        
        # Zero out DC component
        stft_repr = stft_repr.index_fill(1, torch.tensor(0, device=device), 0.)
        
        # ISTFT reconstruction
        stft_window = self.stft_window_fn(device=device)
        recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False, length=None)
        
        # Reshape output
        recon_audio = rearrange(recon_audio, '(b s) t -> b s t', b=batch, s=self.audio_channels)
        
        return recon_audio