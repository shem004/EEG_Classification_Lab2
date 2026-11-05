import torch
import torch.nn as nn


def make_activation(activation='elu', alpha=1.0, negative_slope=0.01):
    """Creates the specified activation layer."""
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=negative_slope)
    else:
        return nn.ELU(alpha=alpha)


class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs.
    Input shape: (B, 1, 2, 750)
    Output shape: (B, 2)
    """
    def __init__(
        self,
        activation='elu',
        dropout=0.25,
        alpha=1.0,
        negative_slope=0.01,
        F1=16,
        D=2,
        F2=32,
        kernel_length=51
    ):
        super(EEGNet, self).__init__()

        if isinstance(dropout, (tuple, list)):
            drop1, drop2 = dropout[0], dropout[1]
        else:
            drop1 = drop2 = dropout

        # 1. Temporal Convolution: (B, 1, 2, 750) -> (B, F1, 2, ~750)
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), stride=(1, 1), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        # 2. Depthwise Spatial Convolution: (B, F1, 2, ~750) -> (B, F1*D, 1, 187)
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(2, 1), stride=(1, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            make_activation(activation, alpha=alpha, negative_slope=negative_slope),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=drop1)
        )

        # 3. Separable Convolution: (B, F1*D, 1, 187) -> (B, F2, 1, 23)
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(F2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            make_activation(activation, alpha=alpha, negative_slope=negative_slope),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=drop2)
        )

        # Dynamically compute flattened feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 2, 750)
            out = self.separableConv(self.depthwiseConv(self.firstconv(dummy)))
            flatten_dim = out.view(1, -1).size(1)

        # 4. Classification Head
        self.classify = nn.Sequential(
            nn.Linear(in_features=flatten_dim, out_features=2, bias=True)
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x


class DeepConvNet(nn.Module):
    """
    DeepConvNet architecture for EEG classification.
    Input shape: (B, 1, 2, 750)
    Output shape: (B, 2)
    """
    def __init__(self, activation='elu', dropout=0.5, alpha=1.0, negative_slope=0.01):
        super(DeepConvNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), bias=True),
            nn.Conv2d(25, 25, kernel_size=(2, 1), bias=False),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            make_activation(activation, alpha, negative_slope),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=dropout)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            make_activation(activation, alpha, negative_slope),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=dropout)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            make_activation(activation, alpha, negative_slope),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=dropout)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            make_activation(activation, alpha, negative_slope),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=dropout)
        )
        self.classify = nn.Linear(in_features=8600, out_features=2, bias=True)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x