import torch
import torch.nn as nn

from config import (
    IN_CHANNELS,
    CONV1_OUT,
    CONV2_OUT,
    CONV3_OUT,
    KERNEL_SIZE,
    PADDING,
)


class CastaMaskFullScanCNN(nn.Module):
    """
    Small fully-convolutional temporal-angular model for per-beam prediction.

    Expected input:
        x: [B, K, T, N]
    where:
        B = batch size
        K = number of feature channels
        T = temporal window length
        N = number of beams (360)

    Output:
        logits: [B, N]
    one logit per beam for the current frame.
    """

    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(
                in_channels=IN_CHANNELS,
                out_channels=CONV1_OUT,
                kernel_size=KERNEL_SIZE,
                padding=PADDING,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=CONV1_OUT,
                out_channels=CONV2_OUT,
                kernel_size=KERNEL_SIZE,
                padding=PADDING,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=CONV2_OUT,
                out_channels=CONV3_OUT,
                kernel_size=KERNEL_SIZE,
                padding=PADDING,
            ),
            nn.ReLU(inplace=True),
        )

        self.temporal_pool = nn.AdaptiveAvgPool2d((1, None))
        self.head = nn.Conv1d(
            in_channels=CONV3_OUT,
            out_channels=1,
            kernel_size=1,
            padding=0,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)      # [B, C3, T, N]
        x = self.temporal_pool(x) # [B, C3, 1, N]
        x = x.squeeze(2)          # [B, C3, N]
        x = self.head(x)          # [B, 1, N]
        x = x.squeeze(1)          # [B, N]
        return x


if __name__ == "__main__":
    model = CastaMaskFullScanCNN()
    x = torch.randn(4, IN_CHANNELS, 7, 360)
    y = model(x)
    print("input shape :", x.shape)
    print("output shape:", y.shape)