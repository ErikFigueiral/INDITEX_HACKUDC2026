import torch
import torch.nn as nn
import torchvision.models as models


IMG_SIZE = 224


class EmbeddingModel(nn.Module):

    def __init__(self, trainable=False):
        super().__init__()

        base = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if not trainable:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):

        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        # L2 normalize
        x = nn.functional.normalize(x, dim=1)

        return x