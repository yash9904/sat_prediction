from typing import Optional, Tuple, Union

import torch
from torch import nn


class DilatedInceptionModule(nn.Module):

    def __init__(
        self,
        input_channels,
        n_conv1,
        n_conv3,
        n_conv5,
        n_conv7,
        skip_channels: Optional[int] = None,
        downsample: Optional[bool] = None,
    ) -> None:
        super(DilatedInceptionModule, self).__init__()
        input_channels = (
            input_channels if not skip_channels else input_channels + skip_channels
        )
        self.downsample = downsample
        self.conv0 = nn.Sequential(
            nn.Conv2d(input_channels, n_conv1, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(n_conv1),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, n_conv1, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(n_conv1),
            nn.Conv2d(n_conv1, n_conv3, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(n_conv3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channels, n_conv1, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(n_conv1),
            nn.Conv2d(n_conv1, n_conv5, kernel_size=(3, 3), padding="same", dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(n_conv5),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(input_channels, n_conv1, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(n_conv1),
            nn.Conv2d(n_conv1, n_conv7, kernel_size=(3, 3), padding="same", dilation=3),
            nn.ReLU(),
            nn.BatchNorm2d(n_conv7),
        )
        self.out_channels = n_conv1 + n_conv3 + n_conv5 + n_conv7
        if self.downsample:
            self.out_conv = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        elif self.downsample is None:
            self.out_conv = nn.Identity()
        else:
            self.out_conv = nn.ConvTranspose2d(
                self.out_channels, self.out_channels, kernel_size=(2, 2), stride=2
            )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        out = torch.cat([x0, x1, x2, x3], dim=1)

        res = self.out_conv(out)

        return res, out

    def __call__(self, x) -> torch.Tensor:
        return self.forward(x)

    def n_out(self):
        return self.out_channels


class DilatedInceptionUNet(nn.Module):
    def __init__(self, input_channels):
        super(DilatedInceptionUNet, self).__init__()
        self.eb1 = DilatedInceptionModule(
            input_channels, 16, 16, 16, 16, downsample=True
        )
        self.eb2 = DilatedInceptionModule(
            self.eb1.n_out(),
            32,
            32,
            32,
            32,
            downsample=True,
        )
        self.eb3 = DilatedInceptionModule(
            self.eb2.n_out(),
            32,
            64,
            64,
            64,
            downsample=True,
        )
        self.eb4 = DilatedInceptionModule(
            self.eb3.n_out(),
            64,
            128,
            128,
            128,
            downsample=True,
        )

        self.bottleneck = DilatedInceptionModule(
            self.eb4.n_out(), 64, 256, 256, 256, downsample=False
        )

        self.db1 = DilatedInceptionModule(
            self.bottleneck.n_out() + self.eb4.n_out(),
            64,
            128,
            128,
            128,
            downsample=False,
        )
        self.db2 = DilatedInceptionModule(
            self.db1.n_out() + self.eb3.n_out(),
            32,
            64,
            64,
            64,
            downsample=False,
        )
        self.db3 = DilatedInceptionModule(
            self.db2.n_out() + self.eb2.n_out(),
            32,
            32,
            32,
            32,
            downsample=False,
        )
        self.db4 = DilatedInceptionModule(
            self.db3.n_out() + self.eb1.n_out(),
            16,
            16,
            16,
            16,
            downsample=None,
        )
        self.final_conv = nn.Conv2d(self.db4.n_out(), 1, kernel_size=(1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        out1, res1 = self.eb1(x)
        out2, res2 = self.eb2(out1)
        out3, res3 = self.eb3(out2)
        out4, res4 = self.eb4(out3)

        out5, _ = self.bottleneck(out4)
        out6, _ = self.db1(torch.cat([out5, res4], dim=1))
        out7, _ = self.db2(torch.cat([out6, res3], dim=1))
        out8, _ = self.db3(torch.cat([out7, res2], dim=1))
        out9, _ = self.db4(torch.cat([out8, res1], dim=1))
        out10 = self.final_conv(out9)
        return self.relu(out10)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DilatedInceptionUNet(48).to(device)
    test_input = torch.randn(12, 48, 128, 128).to(device)
    res = model(test_input)
    print(res.shape)
