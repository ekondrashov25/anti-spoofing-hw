import torch
import torchaudio
from torch import nn


class LogSpectrogram(nn.Module):
    def __init__(self, n_fft, hop_length, window_length, fixed_length, eps=1e-6):
        super().__init__()

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, win_length=window_length, power=2
        )

        self.fixed_length = fixed_length
        self.eps = eps

    def forward(self, waveform):
        spec = self.spectrogram(waveform)

        spec = torch.log(spec + self.eps)

        if spec.shape[-1] < self.fixed_length:
            n_reps = (self.fixed_length + spec.shape[-1] - 1) // spec.shape[-1]
            spec = spec.repeat(1, 1, n_reps)

        spec = spec[..., : self.fixed_length]

        return spec
