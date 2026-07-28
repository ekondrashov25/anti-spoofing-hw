import torch
import torchaudio
from torch import nn


class LogSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate,
        n_fft,
        win_length_s,
        hop_length_s,
        fixed_length,
        n_filter_banks,
        normalized=False,
        eps=1e-6,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = round(win_length_s * sample_rate)
        self.hop_length = round(hop_length_s * sample_rate)
        self.normalized = normalized
        self.register_buffer("window", torch.hann_window(self.win_length))

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_filter_banks,
            sample_rate=sample_rate,
            n_stft=n_fft // 2 + 1,
        )

        self.fixed_length = fixed_length
        self.eps = eps

    def forward(self, waveform):
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=self.normalized,
            center=True,
            return_complex=True,
        )
        power_spec = stft.abs() ** 2
        mel_spec = self.mel_scale(power_spec)

        spec = torch.log(mel_spec + self.eps)

        if spec.shape[-1] < self.fixed_length:
            n_reps = (self.fixed_length + spec.shape[-1] - 1) // spec.shape[-1]
            spec = spec.repeat(1, 1, n_reps)

        spec = spec[..., : self.fixed_length]

        return spec
