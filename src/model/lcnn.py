import torch
from torch import nn

from src.model.mfm import MFM


class LCNN(nn.Module):
    def __init__(self, n_freq_bins, fixed_length, dropout_prob=0.75, num_classes=2):
        super().__init__()

        self.nn = nn.Sequential(
            # Block 1
            MFM(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            MFM(in_channels=32, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(num_features=32),
            # Block 3
            MFM(in_channels=32, out_channels=48, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=48),
            # Block 4
            MFM(in_channels=48, out_channels=48, kernel_size=1),
            nn.BatchNorm2d(num_features=48),
            MFM(in_channels=48, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            MFM(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            MFM(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            MFM(in_channels=32, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(num_features=32),
            MFM(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.nn.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_freq_bins, fixed_length)
            flatten_size = self.nn(dummy_input).flatten(1).shape[1]
        self.nn.train()

        self.fc1 = nn.Linear(flatten_size, 160)
        self.bn = nn.BatchNorm1d(num_features=80)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(80, num_classes)

    def _mfm_1d(self, x, out_features):
        first_half = x[:, :out_features]
        second_half = x[:, out_features:]

        return torch.max(first_half, second_half)

    def forward(self, data_object, **batch):
        data_object = self.nn(data_object)
        data_object = torch.flatten(data_object, start_dim=1)

        data_object = self.fc1(data_object)
        data_object = self._mfm_1d(data_object, out_features=80)
        data_object = self.dropout(data_object)
        data_object = self.bn(data_object)

        logits = self.fc2(data_object)

        return {"logits": logits}
