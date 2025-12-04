import torch
import torch.nn as nn
import timm
from typing import Optional


class DurianConvNeXt(nn.Module):
    """ConvNeXt model for Durian Leaf Disease Classification"""

    def __init__(
        self,
        model_name: str = 'convnext_tiny',
        num_classes: int = 4,
        pretrained: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0
    ):
        """
        Args:
            model_name: Name of ConvNeXt variant (convnext_tiny, convnext_small, convnext_base, etc.)
            num_classes: Number of disease classes
            pretrained: Whether to use pretrained weights
            drop_rate: Dropout rate
            drop_path_rate: Stochastic depth rate
        """
        super(DurianConvNeXt, self).__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def create_model(
    model_name: str = 'convnext_tiny',
    num_classes: int = 4,
    pretrained: bool = True,
    drop_rate: float = 0.2,
    drop_path_rate: float = 0.1
) -> nn.Module:
    """Factory function to create ConvNeXt model"""
    return DurianConvNeXt(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    )


def load_model(
    checkpoint_path: str,
    model_name: str = 'convnext_tiny',
    num_classes: int = 4,
    device: str = 'cuda'
) -> nn.Module:
    """Load model from checkpoint"""
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing"""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss
