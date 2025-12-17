import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, 2*channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.res = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        y = self.conv(x)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        return self.res(y) + x


class DiffWave(nn.Module):
    def __init__(self, residual_layers, residual_channels, dilation_cycle):
        super().__init__()

        self.input = nn.Conv1d(1, residual_channels, kernel_size=1)

        self.blocks = nn.ModuleList()
        for i in range(residual_layers):
            dilation = 2 ** (i % dilation_cycle)
            self.blocks.append(ResidualBlock(residual_channels, dilation))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(residual_channels, residual_channels, 1),
            nn.ReLU(),
            nn.Conv1d(residual_channels, 1, 1)
        )

    def forward(self, audio_noise, mel):
        x = self.input(audio_noise)
        for block in self.blocks:
            x = block(x)
        return self.output(x)
