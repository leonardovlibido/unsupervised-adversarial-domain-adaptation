import torch
from torch import nn
from torch.autograd import Function
from torchvision.models import resnet18, mobilenet_v2
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights


class GradientReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def grad_reverse(x, alpha):
    return GradientReverseLayer.apply(x, alpha)


class SVHNCustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(SVHNCustomCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.25),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.25),
        )
        # 32 -> 28 -> 14

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.5),
        )

        self.classification_head = nn.Sequential(
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

        self.domain_head = nn.Sequential(
            nn.Linear(128, 1000),
            nn.BatchNorm1d(1000),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1000, 1),
        )

    def forward(self, x, alpha=1.0):
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        block3_out = self.block3(block2_out)
        features = block3_out.view(-1, 128)

        classification_out = self.classification_head(features)

        grl_out = grad_reverse(features, alpha)
        domain_out = self.domain_head(grl_out)
        # domain_out = torch.zeros(1)

        return classification_out, domain_out


class MNISTCustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(MNISTCustomCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(32, affine=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(48, affine=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
        )

        self.classification_head = nn.Sequential(
            nn.Linear(768, 100),
            nn.BatchNorm1d(100, affine=False),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100, affine=False),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

        self.domain_head = nn.Sequential(
            nn.Linear(768, 100),
            nn.BatchNorm1d(100),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x, alpha=1.0):
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        features = block2_out.view(-1, 768)

        classification_out = self.classification_head(features)

        grl_out = grad_reverse(features, alpha)
        domain_out = self.domain_head(grl_out)
        # domain_out = torch.zeros(1)

        return classification_out, domain_out


def initialize_classification_model(model_name, num_classes):
    # TODO: create an enum ModelName if needed

    if model_name == 'resnet':
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == 'mobilenetv2':
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        num_ftrs: int = model.classifier[-1].in_features  # type: ignore
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == 'mnist_custom_cnn':
        model = MNISTCustomCNN(num_classes=num_classes)
        input_size = 28
    elif model_name == 'svhn_custom_cnn':
        model = SVHNCustomCNN(num_classes=num_classes)
        input_size = 32
    else:
        raise ValueError(f'unknown model_name {model_name}')

    return model, input_size
