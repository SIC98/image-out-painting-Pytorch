
from torchvision.models import resnet50
import torch.nn as nn
import torch


def get_model(n_classes, image_channels=4):
    model = resnet50(pretrained=False)

    inft = model.fc.in_features
    model.fc = nn.Linear(in_features=inft, out_features=n_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)

    return model


def get_pretrained_model(n_classes, image_channels=4):
    model = resnet50(pretrained=True)
    weight = model.conv1.weight.clone()

    model.conv1 = nn.Conv2d(4, 64, kernel_size=7,
                            stride=2, padding=3, bias=False)

    with torch.no_grad():
        model.conv1.weight[:, :3] = weight
        model.conv1.weight[:, 3] = model.conv1.weight[:, 0]

    inft = model.fc.in_features
    model.fc = nn.Linear(in_features=inft, out_features=n_classes)

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    return model


if __name__ == "__main__":
    model = get_model(2, 4)
    pretrained_model = get_pretrained_model(2, 4)
