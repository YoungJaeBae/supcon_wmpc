from typing import Union, List, Dict, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_
from torchvision.models import vgg16_bn, VGG16_BN_Weights

__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]


class VGG(nn.Module):
    def __init__(self, features: nn.Module, init_weights: bool = True) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(
    cfg: List[Union[str, int]], batch_norm: bool = False, pretrained: bool = False
) -> nn.Sequential:
    layers: List[nn.Module] = []
    if pretrained:
        in_channels = 3
    else:
        in_channels = 1
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(cfg: str, batch_norm: bool = True, pretrained: bool = False) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, pretrained=pretrained))
    return model


def vgg11(pretrained: bool = False) -> VGG:
    return _vgg("A", pretrained=pretrained)


def vgg13(pretrained: bool = False) -> VGG:
    return _vgg("B", pretrained=pretrained)


def vgg16(pretrained: bool = False) -> VGG:
    return _vgg("D", pretrained=pretrained)


def vgg19(pretrained: bool = False) -> VGG:
    return _vgg("E", pretrained=pretrained)


model_dict = {
    "vgg11": [vgg11, 512],
    "vgg13": [vgg13, 512],
    "vgg16": [vgg16, 512],
    "vgg19": [vgg19, 512],
}


class SupConVGG(nn.Module):
    """backbone + projection head"""

    def __init__(self, name="vgg16", head="linear", feat_dim=128):
        super(SupConVGG, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.head = None

        if head == "linear":
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
            )

        else:
            raise NotImplementedError("head not supported: {}".format(head))

    def forward(self, x):
        feat = self.encoder(x)
        projected = F.normalize(self.head(feat), dim=1)
        return feat, projected


class VGGLinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, encoder_name="vgg16", num_classes=9):
        super(VGGLinearClassifier, self).__init__()
        _, feat_dim = model_dict[encoder_name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        output = self.fc(x)
        return output
