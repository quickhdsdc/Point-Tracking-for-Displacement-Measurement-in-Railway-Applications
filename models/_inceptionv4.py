import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, bias=bias)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class SepconvResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SepconvResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        self.use_res_connect = stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            Conv2d(in_channels, out_channels*4, kernel_size=1, padding=0, stride=1),
            SeparableConv2d(out_channels*4, out_channels, kernel_size=kernel_size)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Inceptionv4_stem(nn.Module):
    def __init__(self):
        super(Inceptionv4_stem, self).__init__()
        self.conv2d_1a_3x3 = Conv2d(3, 32, 3, stride=2, padding=0, bias=False)

        self.conv2d_2a_3x3 = Conv2d(32, 32, 3, stride=1, padding=0, bias=False)
        self.conv2d_2b_3x3 = Conv2d(32, 64, 3, stride=1, padding=1, bias=False)

        self.mixed_3a_branch_0 = nn.MaxPool2d(3, stride=2, padding=0)
        self.mixed_3a_branch_1 = Conv2d(64, 96, 3, stride=2, padding=0, bias=False)

        self.mixed_4a_branch_0 = nn.Sequential(
            Conv2d(160, 64, 1, stride=1, padding=0, bias=False),
            Conv2d(64, 96, 3, stride=1, padding=0, bias=False),
        )
        self.mixed_4a_branch_1 = nn.Sequential(
            Conv2d(160, 64, 1, stride=1, padding=0, bias=False),
            Conv2d(64, 64, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(64, 64, (7, 1), stride=1, padding=(3, 0), bias=False),
            Conv2d(64, 96, 3, stride=1, padding=0, bias=False)
        )

        self.mixed_5a_branch_0 = Conv2d(192, 192, 3, stride=2, padding=0, bias=False)
        self.mixed_5a_branch_1 = nn.MaxPool2d(3, stride=2, padding=0)
        
        # self.sepconv_residual = SepconvResidual(384, 512, 3, 1)
        self.conv2d_last = Conv2d(384, 512, 3, 2, 2)

    def forward(self, x):
        x = self.conv2d_1a_3x3(x) # 126 x 126 x 32
        x = self.conv2d_2a_3x3(x) # 124 x 124 x 32
        x = self.conv2d_2b_3x3(x) # 124 x 124 x 64
        x0 = self.mixed_3a_branch_0(x)
        x1 = self.mixed_3a_branch_1(x)
        x = torch.cat((x0, x1), dim=1) # 62 x 62 x 160
        x0 = self.mixed_4a_branch_0(x)
        x1 = self.mixed_4a_branch_1(x)
        x = torch.cat((x0, x1), dim=1) # 60 x 60 x 192
        x0 = self.mixed_5a_branch_0(x)
        x1 = self.mixed_5a_branch_1(x)
        x = torch.cat((x0, x1), dim=1) # 29 x 29 x 384
        x = self.conv2d_last(x)   # 16 x 16 x 512
        return x