"""models/change_detector.py — Siamese U-Net change-detection model (LEVIR-CD).

A true shared-weight Siamese design (FC-Siam-diff style): both dates pass
through the *same* ImageNet-pretrained ResNet-34 encoder, the absolute
difference of the features at every scale is fed to a U-Net decoder, and the
head outputs a single-channel change logit per pixel.
"""
import inspect

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SiameseChangeDetector(nn.Module):
    def __init__(self, encoder_name: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
        )
        # segmentation_models_pytorch <0.4 takes decoder(*features); newer takes decoder(features).
        params = list(inspect.signature(self.unet.decoder.forward).parameters.values())
        self._decoder_takes_list = not any(p.kind == p.VAR_POSITIONAL for p in params)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        f1 = self.unet.encoder(x1)
        f2 = self.unet.encoder(x2)
        diff = [torch.abs(a - b) for a, b in zip(f1, f2)]
        decoded = self.unet.decoder(diff) if self._decoder_takes_list else self.unet.decoder(*diff)
        return self.unet.segmentation_head(decoded).squeeze(1)
