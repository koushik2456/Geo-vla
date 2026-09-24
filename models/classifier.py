"""models/classifier.py — ResNet-50 scene classifier (EuroSAT, 10 classes)."""
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from models import EUROSAT_CLASSES


class SceneClassifier(nn.Module):
    """ImageNet-pretrained ResNet-50 with the final FC layer replaced for
    10-way EuroSAT land-cover classification.

    Pass pretrained=False when loading a fine-tuned checkpoint, so inference
    does not download ImageNet weights that are immediately overwritten.
    """

    def __init__(self, num_classes: int = len(EUROSAT_CLASSES), pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.model = backbone

    def forward(self, x):
        return self.model(x)
