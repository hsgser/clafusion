"""VGG11/13/16/19 in Pytorch."""
import logging as logger
import sys

import torch.nn as nn


sys.path.insert(0, "../..")


# Taken from https://github.com/kuangliu/pytorch-cifar

cfg = {
    "VGG8": [64, "M", 128, "M", 256, "M", 512, "M", 512, "M"],
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG11_quad": [64, "M", 512, "M", 1024, 1024, "M", 2048, 2048, "M", 2048, 512, "M"],
    "VGG11_doub": [64, "M", 256, "M", 512, 512, "M", 1024, 1024, "M", 1024, 512, "M"],
    "VGG11_half": [64, "M", 64, "M", 128, 128, "M", 256, 256, "M", 256, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13_quad": [64, 256, "M", 512, 512, "M", 1024, 1024, "M", 2048, 2048, "M", 2048, 512, "M"],
    "VGG13_doub": [64, 128, "M", 256, 256, "M", 512, 512, "M", 1024, 1024, "M", 1024, 512, "M"],
    "VGG13_half": [64, 32, "M", 64, 64, "M", 128, 128, "M", 256, 256, "M", 256, 512, "M"],
    "VGG13_student": [64, 64, "M", 64, 64, "M", 128, 128, "M", 256, 256, "M", 256, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, model_config, num_classes=10, batch_norm=True, bias=True, relu_inplace=True):
        super(VGG, self).__init__()
        self.batch_norm = batch_norm
        self.bias = bias
        self.features = self._make_layers(model_config, relu_inplace=relu_inplace)
        self.classifier = nn.Linear(512, num_classes, bias=self.bias)
        logger.info("Relu Inplace is {}".format(relu_inplace))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, relu_inplace=True):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.batch_norm:
                    layers += [
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=relu_inplace),
                    ]
                else:
                    layers += [
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias),
                        nn.ReLU(inplace=relu_inplace),
                    ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        logger.info("in _make_layers")
        for layer in layers:
            logger.info(layer)
        return nn.Sequential(*layers)
