import functools
from itertools import cycle
from pathlib import Path

from functools import partial, wraps
from itertools import zip_longest

import torch
from torch import nn, einsum
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
from torch.linalg import vector_norm
from torchaudio.functional import resample

import torchaudio.transforms as T

from einops import rearrange, reduce, pack, unpack

from vector_quantize_pytorch import ResidualVQ

from local_attention import LocalMHA
from local_attention.transformer import FeedForward

from mega_pytorch import MultiHeadedEMA

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(t, l = 1):
    return ((t,) * l) if not isinstance(t, tuple) else t

# gan losses

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def gradient_penalty(wave, output, weight = 10):
    batch_size, device = wave.shape[0], wave.device

    gradients = torch_grad(
        outputs = output,
        inputs = wave,
        grad_outputs = torch.ones_like(output),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((vector_norm(gradients, dim = 1) - 1) ** 2).mean()

def round_down_nearest_multiple(num, divisor):
    return num // divisor * divisor

def curtail_to_multiple(t, mult):
    data_len = t.shape[-1]
    return t[..., :round_down_nearest_multiple(data_len, mult)]

# discriminators

class MultiScaleDiscriminator(nn.Module):
    def __init__(
        self,
        channels = 16,
        layers = 4,
        groups = 4,
        chan_max = 1024,
        input_channels = 1
    ):
        super().__init__()
        self.init_conv = nn.Conv1d(input_channels, channels, 7)
        self.conv_layers = nn.ModuleList([])

        curr_channels = channels

        for _ in range(layers):
            chan_out = min(curr_channels * 4, chan_max)

            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(curr_channels, chan_out, 8, stride = 4, padding = 4, groups = groups),
                leaky_relu()
            ))

            curr_channels = chan_out

        self.final_conv = nn.Sequential(
            nn.Conv1d(curr_channels, curr_channels, 3),
            leaky_relu(),
            nn.Conv1d(curr_channels, 1, 1),
        )

    def forward(self, x, return_intermediates = False):
        x = self.init_conv(x)

        intermediates = []

        for layer in self.conv_layers:
            x = layer(x)
            intermediates.append(x)

        out = self.final_conv(x)

        if not return_intermediates:
            return out

        return out, intermediates

# complex stft discriminator

class ModReLU(nn.Module):
    """
    https://arxiv.org/abs/1705.09792
    https://github.com/pytorch/pytorch/issues/47052#issuecomment-718948801
    """
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return F.relu(torch.abs(x) + self.b) * torch.exp(1.j * torch.angle(x))

class ComplexConv2d(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        kernel_size,
        stride = 1,
        padding = 0
    ):
        super().__init__()
        conv = nn.Conv2d(dim, dim_out, kernel_size, dtype = torch.complex64)
        self.weight = nn.Parameter(torch.view_as_real(conv.weight)).contiguous()
        self.bias = nn.Parameter(torch.view_as_real(conv.bias)).contiguous()

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # print(x.shape , x.dtype)
        weight, bias = map(torch.view_as_complex, (self.weight, self.bias))
        return F.conv2d(x, weight, bias, stride = self.stride, padding = self.padding)

def ComplexSTFTResidualUnit(chan_in, chan_out, strides):
    kernel_sizes = tuple(map(lambda t: t + 2, strides))
    paddings = tuple(map(lambda t: t // 2, kernel_sizes))

    return nn.Sequential(
        ComplexConv2d(chan_in, chan_in, 3, padding = 1),
        ModReLU(),
        ComplexConv2d(chan_in, chan_out, kernel_sizes, stride = strides, padding = paddings)
    )

class ComplexSTFTDiscriminator(nn.Module):
    def __init__(
        self,
        *,
        channels = 32,
        strides = ((1, 2), (2, 2), (1, 2), (2, 2), (1, 2), (2, 2)),
        chan_mults = (1, 2, 4, 4, 8, 8),
        input_channels = 1,
        n_fft = 1024,
        hop_length = 256,
        win_length = 1024,
        stft_normalized = False
    ):
        super().__init__()
        self.init_conv = ComplexConv2d(input_channels, channels, 7, padding = 3)
        layer_channels = tuple(map(lambda mult: mult * channels, chan_mults))
        layer_channels = (channels, *layer_channels)
        layer_channels_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        curr_channels = channels

        self.layers = nn.ModuleList([])

        for layer_stride, (chan_in, chan_out) in zip(strides, layer_channels_pairs):
            self.layers.append(ComplexSTFTResidualUnit(chan_in, chan_out, layer_stride))

        self.final_conv = ComplexConv2d(layer_channels[-1], 1, (16, 1)) # todo: remove hardcoded 16

        # stft settings

        self.stft_normalized = stft_normalized

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, x, return_intermediates = False):
        x = rearrange(x, 'b 1 n -> b n')

        '''
        reference: The content of the paper( https://arxiv.org/pdf/2107.03312.pdf)is as follows:
        The STFT-based discriminator is illustrated in Figure 4
        and operates on a single scale, computing the STFT with a
        window length of W = 1024 samples and a hop length of
        H = 256 samples
        '''
        # print("in for of disc" , x.shape , x.dtype)

        x = torch.stft(
            x,
            self.n_fft,
            hop_length = self.hop_length,
            win_length = self.win_length,
            normalized = self.stft_normalized,
            return_complex = True
        )

        # print("after stft" , x.shape , x.dtype)

        x = rearrange(x, 'b ... -> b 1 ...')

        intermediates = []

        x = self.init_conv(x)

        # print("after init conv" , x.shape , x.dtype)
        intermediates.append(x)

        for layer in self.layers:
            x = layer(x)
            # print("going through residual" , x.shape , x.dtype)
            intermediates.append(x)

        complex_logits = self.final_conv(x)
        # print("after final conv" , x.shape , x.dtype)

        complex_logits_abs = torch.abs(complex_logits)

        if not return_intermediates:
            return complex_logits_abs

        return complex_logits_abs, intermediates

# learned EMA blocks

class MultiHeadEMABlock(nn.Module):
    def __init__(
        self,
        dim,
        **kwargs
    ):
        super().__init__()
        self.prenorm = nn.LayerNorm(dim)
        self.mhema = MultiHeadedEMA(dim = dim, **kwargs)

    def forward(self, x):
        residual = x.clone()
        x = rearrange(x, 'b c n -> b n c')
        x = self.prenorm(x)
        x = self.mhema(x)
        x = rearrange(x, 'b n c -> b c n')
        return x + residual

# sound stream

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class CausalConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, **kwargs):
        super().__init__()
        kernel_size = kernel_size
        dilation = kwargs.get('dilation', 1)
        self.causal_padding = dilation * (kernel_size - 1)

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, **kwargs)

    def forward(self, x):
        x = F.pad(x, (self.causal_padding, 0))
        return self.conv(x)

class CausalConvTranspose1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride, **kwargs):
        super().__init__()
        self.upsample_factor = stride
        self.padding = kernel_size - 1
        self.conv = nn.ConvTranspose1d(chan_in, chan_out, kernel_size, stride, **kwargs)

    def forward(self, x):
        n = x.shape[-1]

        out = self.conv(x)
        out = out[..., :(n * self.upsample_factor)]

        return out

def ResidualUnit(chan_in, chan_out, dilation, kernel_size = 7):
    return Residual(nn.Sequential(
        CausalConv1d(chan_in, chan_out, kernel_size, dilation = dilation),
        nn.ELU(),
        CausalConv1d(chan_out, chan_out, 1),
        nn.ELU()
    ))

def EncoderBlock(chan_in, chan_out, stride, cycle_dilations = (1, 3, 9)):
    it = cycle(cycle_dilations)
    return nn.Sequential(
        ResidualUnit(chan_in, chan_in, next(it)),
        ResidualUnit(chan_in, chan_in, next(it)),
        ResidualUnit(chan_in, chan_in, next(it)),
        CausalConv1d(chan_in, chan_out, 2 * stride, stride = stride)
    )

def DecoderBlock(chan_in, chan_out, stride, cycle_dilations = (1, 3, 9)):
    even_stride = (stride % 2 == 0)
    padding = (stride + (0 if even_stride else 1)) // 2
    output_padding = 0 if even_stride else 1

    it = cycle(cycle_dilations)
    return nn.Sequential(
        CausalConvTranspose1d(chan_in, chan_out, 2 * stride, stride = stride),
        ResidualUnit(chan_out, chan_out, next(it)),
        ResidualUnit(chan_out, chan_out, next(it)),
        ResidualUnit(chan_out, chan_out, next(it)),
    )

class LocalTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        **kwargs
    ):
        super().__init__()
        self.attn = LocalMHA(dim = dim, qk_rmsnorm = True, **kwargs)
        self.ff = FeedForward(dim = dim)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x

class SoundStream(nn.Module):
    def __init__(
        self,
        *,
        channels = 32,
        strides = (2, 4, 5, 8),
        channel_mults = (2, 4, 8, 16),
        codebook_dim = 512,
        codebook_size = 1024,
        rq_num_quantizers = 8,
        rq_commitment_weight = 1.,
        rq_ema_decay = 0.95,
        input_channels = 1,
        discr_multi_scales = (1, 0.5, 0.25),
        stft_normalized = True,
        enc_cycle_dilations = (1, 3, 9),
        dec_cycle_dilations = (1, 3, 9),
        multi_spectral_window_powers_of_two = tuple(range(6, 12)),
        multi_spectral_n_ffts = 512,
        multi_spectral_n_mels = 64,
        recon_loss_weight = 1.,
        multi_spectral_recon_loss_weight = 1.,
        adversarial_loss_weight = 1.,
        feature_loss_weight = 100,
        quantize_dropout_cutoff_index = 1,
        target_sample_hz = 24000,
        use_local_attn = True,
        use_mhesa = True,
        mhesa_heads = 4,
        mhesa_dim_head = 32,
        attn_window_size = 128,
        attn_dim_head = 64,
        attn_heads = 8,
        attn_depth = 1
    ):
        super().__init__()
        self.target_sample_hz = target_sample_hz # for resampling on the fly

        self.single_channel = input_channels == 1
        self.strides = strides

        layer_channels = tuple(map(lambda t: t * channels, channel_mults))
        layer_channels = (channels, *layer_channels)
        chan_in_out_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        encoder_blocks = []

        for ((chan_in, chan_out), layer_stride) in zip(chan_in_out_pairs, strides):
            encoder_blocks.append(EncoderBlock(chan_in, chan_out, layer_stride, enc_cycle_dilations))

            if not use_mhesa:
                continue

            encoder_blocks.append(MultiHeadEMABlock(chan_out, dim_head = mhesa_dim_head, heads = mhesa_heads))

        self.encoder = nn.Sequential(
            CausalConv1d(input_channels, channels, 7),
            *encoder_blocks,
            CausalConv1d(layer_channels[-1], codebook_dim, 3)
        )

        attn_kwargs = dict(
            dim = codebook_dim,
            dim_head = attn_dim_head,
            heads = attn_heads,
            window_size = attn_window_size,
            prenorm = True,
            causal = True
        )

        self.encoder_attn = nn.Sequential(*[LocalTransformerBlock(**attn_kwargs) for _ in range(attn_depth)]) if use_local_attn else None

        self.rq = ResidualVQ(
            dim = codebook_dim,
            num_quantizers = rq_num_quantizers,
            codebook_size = codebook_size,
            decay = rq_ema_decay,
            commitment_weight = rq_commitment_weight,
            kmeans_init = True,
            threshold_ema_dead_code = 2,
            quantize_dropout = True,
            quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        )

        self.decoder_attn = nn.Sequential(*[LocalTransformerBlock(**attn_kwargs) for _ in range(attn_depth)]) if use_local_attn else None

        decoder_blocks = []

        for ((chan_in, chan_out), layer_stride) in zip(reversed(chan_in_out_pairs), reversed(strides)):
            decoder_blocks.append(DecoderBlock(chan_out, chan_in, layer_stride, dec_cycle_dilations))

            if not use_mhesa:
                continue

            decoder_blocks.append(MultiHeadEMABlock(chan_in, dim_head = mhesa_dim_head, heads = mhesa_heads))

        self.decoder = nn.Sequential(
            CausalConv1d(codebook_dim, layer_channels[-1], 7),
            *decoder_blocks,
            CausalConv1d(channels, input_channels, 7)
        )

        # discriminators

        self.discr_multi_scales = discr_multi_scales
        self.discriminators = nn.ModuleList([MultiScaleDiscriminator() for _ in range(len(discr_multi_scales))])
        discr_rel_factors = [int(s1 / s2) for s1, s2 in zip(discr_multi_scales[:-1], discr_multi_scales[1:])]
        self.downsamples = nn.ModuleList([nn.Identity()] + [nn.AvgPool1d(2 * factor, stride = factor, padding = factor) for factor in discr_rel_factors])

        self.stft_discriminator = ComplexSTFTDiscriminator(
            stft_normalized = stft_normalized
        )

        # multi spectral reconstruction

        self.mel_spec_transforms = nn.ModuleList([])
        self.mel_spec_recon_alphas = []

        num_transforms = len(multi_spectral_window_powers_of_two)
        multi_spectral_n_ffts = cast_tuple(multi_spectral_n_ffts, num_transforms)
        multi_spectral_n_mels = cast_tuple(multi_spectral_n_mels, num_transforms)

        for powers, n_fft, n_mels in zip_longest(multi_spectral_window_powers_of_two, multi_spectral_n_ffts, multi_spectral_n_mels):
            win_length = 2 ** powers
            alpha = (win_length / 2) ** 0.5

            calculated_n_fft = default(max(n_fft, win_length), win_length)  # @AndreyBocharnikov said this is usually win length, but overridable

            # if any audio experts have an opinion about these settings, please submit a PR

            melspec_transform = T.MelSpectrogram(
                sample_rate = target_sample_hz,
                n_fft = calculated_n_fft,
                win_length = win_length,
                hop_length = win_length // 4,
                n_mels = n_mels,
                normalized = stft_normalized
            )

            self.mel_spec_transforms.append(melspec_transform)
            self.mel_spec_recon_alphas.append(alpha)

        # loss weights

        self.recon_loss_weight = recon_loss_weight
        self.multi_spectral_recon_loss_weight = multi_spectral_recon_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.feature_loss_weight = feature_loss_weight

    def decode_from_codebook_indices(self, quantized_indices):
        codes = self.rq.get_codes_from_indices(quantized_indices)
        x = reduce(codes, 'q ... -> ...', 'sum')

        x = self.decoder_attn(x)
        x = rearrange(x, 'b n c -> b c n')
        return self.decoder(x)

    def save(self, path):
        path = Path(path)
        torch.save(self.state_dict(), str(path))

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path))

        # some hacky logic to remove confusion around loading trainer vs main model

        maybe_trainer_pkg = len(pkg.keys()) < 15
        if maybe_trainer_pkg:
            self.load_from_trainer_saved_obj(str(path))
            return

        self.load_state_dict()

    def load_from_trainer_saved_obj(self, path):
        path = Path(path)
        assert path.exists()
        obj = torch.load(str(path))
        self.load_state_dict(obj['model'])

    def non_discr_parameters(self):
        return [
            *self.encoder.parameters(),
            *self.decoder.parameters(),
            *(self.encoder_attn.parameters() if exists(self.encoder_attn) else []),
            *(self.decoder_attn.parameters() if exists(self.decoder_attn) else [])
        ]

    @property
    def seq_len_multiple_of(self):
        return functools.reduce(lambda x, y: x * y, self.strides)

    def forward(
        self,
        x,
        return_encoded = False,
        return_discr_loss = False,
        return_discr_losses_separately = False,
        return_loss_breakdown = False,
        return_recons_only = False,
        input_sample_hz = None,
        apply_grad_penalty = False
    ):
        x, ps = pack([x], '* n')

        if exists(input_sample_hz):
            x = resample(x, input_sample_hz, self.target_sample_hz)

        x = curtail_to_multiple(x, self.seq_len_multiple_of)

        if x.ndim == 2:
            x = rearrange(x, 'b n -> b 1 n')

        orig_x = x.clone()

        x = self.encoder(x)

        x = rearrange(x, 'b c n -> b n c')

        if exists(self.encoder_attn):
            x = self.encoder_attn(x)

        x, indices, commit_loss = self.rq(x)

        if exists(self.decoder_attn):
            x = self.decoder_attn(x)

        x = rearrange(x, 'b n c -> b c n')

        if return_encoded:
            return x, indices, commit_loss

        recon_x = self.decoder(x).float()

        if return_recons_only:
            recon_x, = unpack(recon_x, ps, '* c n')
            return recon_x

        # multi-scale discriminator loss

        if return_discr_loss:
            real, fake = orig_x, recon_x.detach()
            # print("in multi-scale losses" , real.shape, real.dtype , fake.dtype)

            stft_discr_loss = None
            stft_grad_penalty = None
            discr_losses = []
            discr_grad_penalties = []

            if self.single_channel:
                real, fake = orig_x.clone(), recon_x.detach()
                stft_real_logits, stft_fake_logits = map(self.stft_discriminator, (real.requires_grad_(), fake))
                stft_discr_loss = hinge_discr_loss(stft_fake_logits, stft_real_logits)

                if apply_grad_penalty:
                    stft_grad_penalty = gradient_penalty(real, stft_discr_loss)

            scaled_real, scaled_fake = real, fake
            for discr, downsample in zip(self.discriminators, self.downsamples):
                scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))

                real_logits, fake_logits = map(discr, (scaled_real.requires_grad_(), scaled_fake))
                one_discr_loss = hinge_discr_loss(fake_logits, real_logits)

                discr_losses.append(one_discr_loss)
                if apply_grad_penalty:
                    discr_grad_penalties.append(gradient_penalty(scaled_real, one_discr_loss))

            if not return_discr_losses_separately:
                all_discr_losses = torch.stack(discr_losses).mean()

                if exists(stft_discr_loss):
                    all_discr_losses = all_discr_losses + stft_discr_loss

                if exists(stft_grad_penalty):
                    all_discr_losses = all_discr_losses + stft_grad_penalty

                return all_discr_losses

            # return a list of discriminator losses with List[Tuple[str, Tensor]]

            discr_losses_pkg = []

            discr_losses_pkg.extend([(f'scale:{scale}', multi_scale_loss) for scale, multi_scale_loss in zip(self.discr_multi_scales, discr_losses)])

            discr_losses_pkg.extend([(f'scale_grad_penalty:{scale}', discr_grad_penalty) for scale, discr_grad_penalty in zip(self.discr_multi_scales, discr_grad_penalties)])

            if exists(stft_discr_loss):
                discr_losses_pkg.append(('stft', stft_discr_loss))

            if exists(stft_grad_penalty):
                discr_losses_pkg.append(('stft_grad_penalty', stft_grad_penalty))

            return discr_losses_pkg

        # recon loss

        recon_loss = F.mse_loss(orig_x, recon_x)

        # multispectral recon loss - eq (4) and (5) in https://arxiv.org/abs/2107.03312

        multi_spectral_recon_loss = 0

        if self.multi_spectral_recon_loss_weight > 0:
            for mel_transform, alpha in zip(self.mel_spec_transforms, self.mel_spec_recon_alphas):
                orig_mel, recon_mel = map(mel_transform, (orig_x, recon_x))
                log_orig_mel, log_recon_mel = map(log, (orig_mel, recon_mel))

                l1_mel_loss = (orig_mel - recon_mel).abs().sum(dim = -2).mean()
                l2_log_mel_loss = alpha * vector_norm(log_orig_mel - log_recon_mel, dim = -2).mean()

                multi_spectral_recon_loss = multi_spectral_recon_loss + l1_mel_loss + l2_log_mel_loss

        # adversarial loss

        adversarial_losses = []

        discr_intermediates = []

        # adversarial loss for multi-scale discriminators

        real, fake = orig_x, recon_x

        # features from stft
        # print("in features" , real.shape, real.dtype , fake.dtype)


        (stft_real_logits, stft_real_intermediates), (stft_fake_logits, stft_fake_intermediates) = map(partial(self.stft_discriminator, return_intermediates=True), (real, fake))
        discr_intermediates.append((stft_real_intermediates, stft_fake_intermediates))

        scaled_real, scaled_fake = real, fake
        for discr, downsample in zip(self.discriminators, self.downsamples):
            scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))

            (real_logits, real_intermediates), (fake_logits, fake_intermediates) = map(partial(discr, return_intermediates = True), (scaled_real, scaled_fake))

            discr_intermediates.append((real_intermediates, fake_intermediates))

            one_adversarial_loss = hinge_gen_loss(fake_logits)
            adversarial_losses.append(one_adversarial_loss)

        feature_losses = []

        for real_intermediates, fake_intermediates in discr_intermediates:
            losses = [F.l1_loss(real_intermediate, fake_intermediate) for real_intermediate, fake_intermediate in zip(real_intermediates, fake_intermediates)]
            feature_losses.extend(losses)

        feature_loss = torch.stack(feature_losses).mean()

        # adversarial loss for stft discriminator

        adversarial_losses.append(hinge_gen_loss(stft_fake_logits))
        adversarial_loss = torch.stack(adversarial_losses).mean()

        # sum commitment loss

        all_commitment_loss = commit_loss.sum()

        total_loss = recon_loss * self.recon_loss_weight + multi_spectral_recon_loss * self.multi_spectral_recon_loss_weight + adversarial_loss * self.adversarial_loss_weight + feature_loss * self.feature_loss_weight + all_commitment_loss

        if return_loss_breakdown:
            return total_loss, (recon_loss, multi_spectral_recon_loss, adversarial_loss, feature_loss, all_commitment_loss)

        return total_loss
