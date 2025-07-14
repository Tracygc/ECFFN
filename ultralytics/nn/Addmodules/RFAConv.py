from torch import nn
import math
from einops import rearrange
from .SPDConv import *
from pytorch_wavelets import DWTForward
# from ultralytics.nn.modules.block import Bottleneck, C3


# 2D:
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2
    from torch.fft import rfft2

    def rfft(x, d):
        t = rfft2(x, dim=(-d, -1))
        return torch.stack((t.real, t.imag), -1)

    def irfft(x, d, signal_sizes):
        return irfft2(torch.complex(x[:, :, 0], x[:, :, 1]), s=signal_sizes, dim=(-d, -1))

# ##1D:
# try:
#     from torch import irfft
#     from torch import rfft
# except ImportError:
#     def rfft(x, d):
#         t = torch.fft.fft(x, dim = (-d))
#         r = torch.stack((t.real, t.imag), -1)
#         return r
#     def irfft(x, d):
#         t = torch.fft.ifft(torch.complex(x[:,:,0], x[:,:,1]), dim = (-d))
#         return t.real


class RFAConv(nn.Module):  # 基于Unfold实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2)
        self.get_weights = nn.Sequential(
            nn.Conv2d(in_channel * (kernel_size ** 2), in_channel * (kernel_size ** 2), kernel_size=1,
                      groups=in_channel),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)))

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=0, stride=kernel_size)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU()

    def forward(self, x):
        b, c, h, w = x.shape
        unfold_feature = self.unfold(x)  # 获得感受野空间特征  b c*kernel**2,h*w
        x = unfold_feature
        data = unfold_feature.unsqueeze(-1)
        weight = self.get_weights(data).view(b, c, self.kernel_size ** 2, h, w).permute(0, 1, 3, 4, 2).softmax(-1)
        weight_out = rearrange(weight, 'b c h w (n1 n2) -> b c (h n1) (w n2)', n1=self.kernel_size,
                               n2=self.kernel_size)  # b c h w k**2 -> b c h*k w*k
        receptive_field_data = rearrange(x, 'b (c n1) l -> b c n1 l', n1=self.kernel_size ** 2).permute(0, 1, 3,
                                                                                                        2).reshape(b, c,
                                                                                                                   h, w,
                                                                                                                   self.kernel_size ** 2)  # b c*kernel**2,h*w ->  b c h w k**2
        data_out = rearrange(receptive_field_data, 'b c h w (n1 n2) -> b c (h n1) (w n2)', n1=self.kernel_size,
                             n2=self.kernel_size)  # b c h w k**2 -> b c h*k w*k
        conv_data = data_out * weight_out
        conv_out = self.conv(conv_data)
        return self.act(self.bn(conv_out))

    def __str__(self):
        return 'RFAConv'


class RFAConv_G(nn.Module):  # 基于Group Conv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
                                                  groups=in_channel, bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)


class RFAConv_yolov8(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, g=1, dilation=1):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, k=1)

        self.RFAConv = RFAConv(out_channels, out_channels, kernel_size=3)

        self.bn = nn.BatchNorm2d(out_channels)

        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)

        x = self.RFAConv(x)

        x = self.gelu(self.bn(x))
        return x




class RFA_SPDConv(nn.Module):  # 基于SPDConv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
        #                                 SPDConv(in_channel, in_channel * (kernel_size ** 2)))
        self.get_weight = nn.Sequential(SPDConv(in_channel, in_channel * (kernel_size ** 2), 1),
                                        nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride))
        # self.generate_feature = nn.Sequential(
        #     SPDConv(in_channel, in_channel * (kernel_size ** 2)),
        #     nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
        #     nn.ReLU())
        self.generate_feature = nn.Sequential(
            SPDConv(in_channel, in_channel * (kernel_size ** 2), kernel_size, p=kernel_size// 2),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              autopad(kernel_size, p=kernel_size, d=1), bias=False)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)


class RFA_SPD_FCConv(nn.Module):  # 基于SPDConv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
        #                                 SPDConv(in_channel, in_channel * (kernel_size ** 2)))
        self.get_weight = nn.Sequential(SPDConv_FFC(in_channel, in_channel * (kernel_size ** 2), 1),
                                        nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride))
        # self.generate_feature = nn.Sequential(
        #     SPDConv(in_channel, in_channel * (kernel_size ** 2)),
        #     nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
        #     nn.ReLU())
        self.generate_feature = nn.Sequential(
            SPDConv_FFC(in_channel, in_channel * (kernel_size ** 2), kernel_size),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              autopad(kernel_size, p=kernel_size, d=1), bias=False)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)



# 尝试把卷积kernal的平方去掉，发现卷积参数量很大，所以不考虑
# 把AvgPool2d改成HWD  # 没跑出来
class RFA_SPDConv_HWD(nn.Module):  # 基于SPDConv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # self.get_weight = nn.Sequential(SPDConv(in_channel, in_channel * (kernel_size ** 2), 1),
        #                                 nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride))
        self.get_weight = nn.Sequential(SPDConv(in_channel, in_channel * (kernel_size ** 2), 1),
                                        # nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride)
                                        HWD_ADown(in_channel, in_channel * (kernel_size ** 2)))
        self.generate_feature = nn.Sequential(
            SPDConv(in_channel, in_channel * (kernel_size ** 2), kernel_size, p=kernel_size// 2),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              autopad(kernel_size, p=kernel_size, d=1), bias=False)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)

# 把AvgPool2d改成HWD，对接使用CARAFE上采样 #跑通了
class RFA_SPDConv_HWD_CARAFE(nn.Module):  # 基于SPDConv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # self.get_weight = nn.Sequential(SPDConv(in_channel, in_channel * (kernel_size ** 2), 1),
        #                                 nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride))
        self.get_weight = nn.Sequential(SPDConv(in_channel, in_channel * (kernel_size ** 2), 1),
                                        # nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride)
                                        HWD_ADown_CARAFE(in_channel, in_channel * (kernel_size ** 2)))
        self.generate_feature = nn.Sequential(
            SPDConv(in_channel, in_channel * (kernel_size ** 2), kernel_size, p=kernel_size// 2),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              autopad(kernel_size, p=kernel_size, d=1), bias=False)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)

# 把AvgPool2d改成HWD，对接使用CARAFE上采样
class RFA_SPDConv_HWD_CARAFE_SDI(nn.Module):  # 基于SPDConv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # self.get_weight = nn.Sequential(SPDConv(in_channel, in_channel * (kernel_size ** 2), 1),
        #                                 nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride))
        self.get_weight = nn.Sequential(SPDConv(in_channel, in_channel * (kernel_size ** 2), 1),
                                        # nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride)
                                        HWD_ADown_CARAFE_SDI(in_channel, in_channel * (kernel_size ** 2)))
        self.generate_feature = nn.Sequential(
            SPDConv(in_channel, in_channel * (kernel_size ** 2), kernel_size, p=kernel_size// 2),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              autopad(kernel_size, p=kernel_size, d=1), bias=False)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)

# 没跑出来
class RFA_SPDConv_Fourier(nn.Module):  # 基于SPDConv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # self.get_weight = nn.Sequential(SPDConv(in_channel, in_channel * (kernel_size ** 2), 1),
        #                                 nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride))
        self.get_weight = nn.Sequential(SPDConv(in_channel, in_channel * (kernel_size ** 2), 1),
                                        # nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride)
                                        Fourier_ADown(in_channel, in_channel * (kernel_size ** 2)))
        self.generate_feature = nn.Sequential(
            SPDConv(in_channel, in_channel * (kernel_size ** 2), kernel_size, p=kernel_size// 2),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              autopad(kernel_size, p=kernel_size, d=1), bias=False)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)


# 把AvgPool2d改成Fourier，对接使用CARAFE上采样
# 把val去掉可以能跑得
class RFA_SPDConv_Fourier_CARAFE(nn.Module):  # 基于SPDConv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # self.get_weight = nn.Sequential(SPDConv(in_channel, in_channel * (kernel_size ** 2), 1),
        #                                 nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride))
        self.get_weight = nn.Sequential(SPDConv(in_channel, in_channel * (kernel_size ** 2), 1),
                                        # nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride)
                                        Fourier_ADown_CARAFE(in_channel, in_channel * (kernel_size ** 2)))
        self.generate_feature = nn.Sequential(
            SPDConv(in_channel, in_channel * (kernel_size ** 2), kernel_size, p=kernel_size// 2),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              autopad(kernel_size, p=kernel_size, d=1), bias=False)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)

# 把val去掉可以能跑得
# 把AvgPool2d改成Fourier，对接使用CARAFE上采样 拼接使用SDI
class RFA_SPDConv_Fourier_CARAFE_SDI(nn.Module):  # 基于SPDConv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # self.get_weight = nn.Sequential(SPDConv(in_channel, in_channel * (kernel_size ** 2), 1),
        #                                 nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride))
        self.get_weight = nn.Sequential(SPDConv(in_channel, in_channel * (kernel_size ** 2), 1),
                                        # nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride)
                                        Fourier_ADown_CARAFE_SDI(in_channel, in_channel * (kernel_size ** 2)))
        self.generate_feature = nn.Sequential(
            SPDConv(in_channel, in_channel * (kernel_size ** 2), kernel_size, p=kernel_size// 2),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              autopad(kernel_size, p=kernel_size, d=1), bias=False)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)

## 没有跑通
class RFA_SPDConv_Fourier_SDI(nn.Module):  # 基于SPDConv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # self.get_weight = nn.Sequential(SPDConv(in_channel, in_channel * (kernel_size ** 2), 1),
        #                                 nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride))
        self.get_weight = nn.Sequential(SPDConv(in_channel, in_channel * (kernel_size ** 2), 1),
                                        # nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride)
                                        Fourier_SDIModel(in_channel, in_channel * (kernel_size ** 2)))
        self.generate_feature = nn.Sequential(
            SPDConv(in_channel, in_channel * (kernel_size ** 2), kernel_size),
            Fourier_SDIModel(in_channel, in_channel * (kernel_size ** 2)),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Conv2d(in_channel** 2, out_channel, kernel_size,
                              autopad(kernel_size, p=kernel_size, d=1), bias=False)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c** 2, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w

        feature = self.generate_feature(x).view(b, c** 2, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)


### 可以跑通的SPDConv_FFC类似上面SPDConv加入HWD和SDI
class RFA_SPD_FCConv_HWD_ADown_CARAFE(nn.Module):  # 基于SPDConv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
        #                                 SPDConv(in_channel, in_channel * (kernel_size ** 2)))
        self.get_weight = nn.Sequential(SPDConv_FFC(in_channel, in_channel * (kernel_size ** 2), 1),
                                        HWD_ADown_CARAFE(in_channel, in_channel * (kernel_size ** 2)))
        # self.generate_feature = nn.Sequential(
        #     SPDConv(in_channel, in_channel * (kernel_size ** 2)),
        #     nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
        #     nn.ReLU())
        self.generate_feature = nn.Sequential(
            SPDConv_FFC(in_channel, in_channel * (kernel_size ** 2), kernel_size),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              autopad(kernel_size, p=kernel_size, d=1), bias=False)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)

class RFA_SPD_FCConv_HWD_ADown_CARAFE_SDI(nn.Module):  # 基于SPDConv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
        #                                 SPDConv(in_channel, in_channel * (kernel_size ** 2)))
        self.get_weight = nn.Sequential(SPDConv_FFC(in_channel, in_channel * (kernel_size ** 2), 1),
                                        HWD_ADown_CARAFE_SDI(in_channel, in_channel * (kernel_size ** 2)))
        # self.generate_feature = nn.Sequential(
        #     SPDConv(in_channel, in_channel * (kernel_size ** 2)),
        #     nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
        #     nn.ReLU())
        self.generate_feature = nn.Sequential(
            SPDConv_FFC(in_channel, in_channel * (kernel_size ** 2), kernel_size),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              autopad(kernel_size, p=kernel_size, d=1), bias=False)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)

##### Oct里面能放到RFAConv里面可以跑的
class RFA_SPD_Oct_CARAFE_HWDConv(nn.Module):  # 基于SPDConv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
        #                                 SPDConv(in_channel, in_channel * (kernel_size ** 2)))
        self.get_weight = nn.Sequential(SPDConv_Oct_CARAFE_HWD(in_channel, in_channel * (kernel_size ** 2), 1),
                                        nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride))
        # self.generate_feature = nn.Sequential(
        #     SPDConv(in_channel, in_channel * (kernel_size ** 2)),
        #     nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
        #     nn.ReLU())
        self.generate_feature = nn.Sequential(
            SPDConv_Oct_CARAFE_HWD(in_channel, in_channel * (kernel_size ** 2), kernel_size),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              autopad(kernel_size, p=kernel_size, d=1), bias=False)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)



#####################没跑通
#### 还在尝试中
class RFA_SPD_OctConv(nn.Module):  # 基于SPDConv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
        #                                 SPDConv(in_channel, in_channel * (kernel_size ** 2)))
        self.get_weight = nn.Sequential(SPDConv_Oct(in_channel, in_channel * (kernel_size ** 2), 1),
                                        nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride))
        # self.generate_feature = nn.Sequential(
        #     SPDConv(in_channel, in_channel * (kernel_size ** 2)),
        #     nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
        #     nn.ReLU())
        self.generate_feature = nn.Sequential(
            SPDConv_Oct(in_channel, in_channel * (kernel_size ** 2), kernel_size),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              autopad(kernel_size, p=kernel_size, d=1), bias=False)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)

class RFA_SPD_Oct_CARAFEConv(nn.Module):  # 基于SPDConv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
        #                                 SPDConv(in_channel, in_channel * (kernel_size ** 2)))
        self.get_weight = nn.Sequential(SPDConv_Oct_CARAFE(in_channel, in_channel * (kernel_size ** 2), 1),
                                        nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride))
        # self.generate_feature = nn.Sequential(
        #     SPDConv(in_channel, in_channel * (kernel_size ** 2)),
        #     nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
        #     nn.ReLU())
        self.generate_feature = nn.Sequential(
            SPDConv_Oct_CARAFE(in_channel, in_channel * (kernel_size ** 2), kernel_size),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              autopad(kernel_size, p=kernel_size, d=1), bias=False)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)




class RFA_SPD_Oct_CARAFE_HWD_ADownConv(nn.Module):  # 基于SPDConv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
        #                                 SPDConv(in_channel, in_channel * (kernel_size ** 2)))
        self.get_weight = nn.Sequential(SPDConv_Oct_CARAFE_HWD_ADown(in_channel, in_channel * (kernel_size ** 2), 1),
                                        nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride))
        # self.generate_feature = nn.Sequential(
        #     SPDConv(in_channel, in_channel * (kernel_size ** 2)),
        #     nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
        #     nn.ReLU())
        self.generate_feature = nn.Sequential(
            SPDConv_Oct_CARAFE_HWD_ADown(in_channel, in_channel * (kernel_size ** 2), kernel_size),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              autopad(kernel_size, p=kernel_size, d=1), bias=False)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)


### 出错
class RFA_FFConv(nn.Module):  # 基于FFConv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
        #                                 SPDConv(in_channel, in_channel * (kernel_size ** 2)))
        self.get_weight = nn.Sequential(FFC_BN_ACT(in_channel, out_channel, kernel_size,
                                                   ratio_gin=0.5, ratio_gout=0.5),
                                        nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride))
        # self.generate_feature = nn.Sequential(
        #     SPDConv(in_channel, in_channel * (kernel_size ** 2)),
        #     nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
        #     nn.ReLU())
        self.generate_feature = nn.Sequential(
            FFC_BN_ACT(in_channel, out_channel, kernel_size,  ratio_gin=0.5, ratio_gout=0.5 ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU())

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              autopad(kernel_size, p=kernel_size, d=1), bias=False)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)

class RFA_OctConv(nn.Module):  # 基于FFConv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
        #                                 SPDConv(in_channel, in_channel * (kernel_size ** 2)))
        self.get_weight = nn.Sequential(OctConv(in_channel, out_channel, kernel_size),
                                        nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride))
        # self.generate_feature = nn.Sequential(
        #     SPDConv(in_channel, in_channel * (kernel_size ** 2)),
        #     nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
        #     nn.ReLU())
        self.generate_feature = nn.Sequential(
            OctConv(in_channel, out_channel, kernel_size ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU())

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              autopad(kernel_size, p=kernel_size, d=1), bias=False)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)



class RFA_SPDConv_FFC_ADown_CARAFE(nn.Module):  # 基于SPDConv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # self.get_weight = nn.Sequential(SPDConv(in_channel, in_channel * (kernel_size ** 2), 1),
        #                                 nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride))
        self.get_weight = nn.Sequential(SPDConv(in_channel, in_channel * (kernel_size ** 2), 1),
                                        # nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride)
                                        FFC_ADown_CARAFE(in_channel, in_channel * (kernel_size ** 2)))
        self.generate_feature = nn.Sequential(
            SPDConv(in_channel, in_channel * (kernel_size ** 2), kernel_size, p=kernel_size// 2),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              autopad(kernel_size, p=kernel_size, d=1), bias=False)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)

class RFA_SPDConv_OctC_CARAFE(nn.Module):  # 基于SPDConv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # self.get_weight = nn.Sequential(SPDConv(in_channel, in_channel * (kernel_size ** 2), 1),
        #                                 nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride))
        self.get_weight = nn.Sequential(SPDConv(in_channel, in_channel * (kernel_size ** 2), 1),
                                        # nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size// 2, stride=stride)
                                        OctC_ADown_CARAFE(in_channel, in_channel * (kernel_size ** 2)))
        self.generate_feature = nn.Sequential(
            SPDConv(in_channel, in_channel * (kernel_size ** 2), kernel_size, p=kernel_size// 2),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              autopad(kernel_size, p=kernel_size, d=1), bias=False)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)


#####################没跑通


class HWD_ADown_CARAFE(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        # self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv1 = HWD(c1 // 2, self.c)
        # self.cv1 = HWD_pytorch(c1 // 2, self.c)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
        self.cv3 = CARAFE(c2)
        # self.cv3 = Conv(c1 // 2, self.c, 1, 1, 0)


    def forward(self, x):
        x = nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        x3 = torch.cat((x1, x2), 1)
        x3 = self.cv3(x3)
        # return torch.cat((x1, x2), 1)
        return x3

class HWD_ADown_CARAFE_SDI(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        # self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv1 = HWD_SDI(c1 // 2, self.c)
        # self.cv1 = HWD_pytorch(c1 // 2, self.c)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
        self.cv3 = CARAFE(c2)
        # self.cv3 = Conv(c1 // 2, self.c, 1, 1, 0)


    def forward(self, x):
        x = nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        x3 = torch.cat((x1, x2), 1)
        x3 = self.cv3(x3)
        # return torch.cat((x1, x2), 1)
        return x3

class Fourier_ADown_CARAFE(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        # self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        # self.cv1 = HWD(c1 // 2, self.c)
        self.cv1 = FourierModel(c1 // 2, self.c)
        # self.cv1 = HWD_pytorch(c1 // 2, self.c)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
        self.cv3 = CARAFE(c2)
        # self.cv3 = Conv(c1 // 2, self.c, 1, 1, 0)


    def forward(self, x):
        x = nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        x3 = torch.cat((x1, x2), 1)
        x3 = self.cv3(x3)
        # return torch.cat((x1, x2), 1)
        return x3


class Fourier_ADown_CARAFE_SDI(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        # self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        # self.cv1 = HWD(c1 // 2, self.c)
        self.cv1 = Fourier_SDIModel(c1 // 2, self.c)
        # self.cv1 = HWD_pytorch(c1 // 2, self.c)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
        self.cv3 = CARAFE(c2)
        # self.cv3 = Conv(c1 // 2, self.c, 1, 1, 0)


    def forward(self, x):
        x = nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        # print(x1.shape,111)
        # print(x2.shape,222)
        x3 = torch.cat((x1, x2), 1)
        x3 = self.cv3(x3)
        # return torch.cat((x1, x2), 1)
        return x3

class FFC_ADown_CARAFE(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        # self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        # self.cv1 = HWD(c1 // 2, self.c)
        # self.cv1 = FFC_BN_ACT(c1 // 2, self.c,kernel_size=1, ratio_gin=0.5, ratio_gout=0.5)
        self.cv1 = FFC_BN_ACT(c1 // 2, self.c, kernel_size=1, ratio_gin=0.5, ratio_gout=0.5)
        # self.cv1 = HWD_pytorch(c1 // 2, self.c)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
        self.cv3 = CARAFE(c2)
        # self.cv3 = Conv(c1 // 2, self.c, 1, 1, 0)


    def forward(self, x):
        x = nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1_1, x1_2= x1.chunk(2, 1)
        x1_tuple = (x1_1, x1_2)
        x1 = self.cv1(x1_tuple)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        x3 = torch.cat((x1, x2), 1)
        x3 = self.cv3(x3)
        # return torch.cat((x1, x2), 1)
        return x3

class OctC_ADown_CARAFE(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        # self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        # self.cv1 = HWD(c1 // 2, self.c)
        # self.cv1 = FFC_BN_ACT(c1 // 2, self.c,kernel_size=1, ratio_gin=0.5, ratio_gout=0.5)
        self.cv1 = OctConv(c1 // 2, self.c, kernel_size=1)
        # self.cv1 = HWD_pytorch(c1 // 2, self.c)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
        self.cv3 = CARAFE(c2)
        # self.cv3 = Conv(c1 // 2, self.c, 1, 1, 0)


    def forward(self, x):
        x = nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1_1, x1_2= x1.chunk(2, 1)
        x1_tuple = (x1_1, x1_2)
        x1 = self.cv1(x1_tuple)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        x3 = torch.cat((x1, x2), 1)
        x3 = self.cv3(x3)
        # return torch.cat((x1, x2), 1)
        return x3

class HWD_ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        # self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv1 = HWD(c1 // 2, self.c)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)

class Fourier_ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        # self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv1 = FourierModel(c1 // 2, self.c)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class HWD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()

        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv = Conv(in_ch * 4, out_ch, 1, 1)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv(x)
        return x

# 源代码
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x


class FourierModel(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FourierModel, self).__init__()
        self.ft = Fourierforward(J=1)
        self.conv = Conv(in_ch * 4, out_ch, 1, 1)

    def forward(self, x):
        yL, yH = self.ft(x)
        # y_HL = yH[0][:, :, 0, ::]
        # y_LH = yH[0][:, :, 1, ::]
        # y_HH = yH[0][:, :, 2, ::]
        y_HL = yH[0]
        y_LH = yH[1]
        y_HH = yH[2]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv(x)
        return x

class HWD_SDI(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWD_SDI, self).__init__()
        from pytorch_wavelets import DWTForward
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.sdi = SDI1([out_ch])
        self.conv = Conv(in_ch * 4, out_ch, 1, 1)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        # x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        xL = self.sdi(yL)
        x_HL = self.sdi(y_HL)
        x_LH = self.sdi(y_LH)
        x_HH = self.sdi(y_HH)
        x = torch.cat([xL, x_HL, x_LH, x_HH], dim=1)
        x = self.conv(x)
        return x


class Fourier_SDIModel(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Fourier_SDIModel, self).__init__()
        self.ft = Fourierforward(J=1)
        self.sdi = SDI1([out_ch])
        self.conv = Conv(in_ch * 4, out_ch, 1)

    def forward(self, x):
        yL, yH = self.ft(x)
        # y_HL = yH[0][:, :, 0, ::]
        # y_LH = yH[0][:, :, 1, ::]
        # y_HH = yH[0][:, :, 2, ::]
        y_HL = yH[0]
        y_LH = yH[1]
        y_HH = yH[2]

        # x = self.sdi([y_HL, y_LH, y_HH], yL)
        # xL = self.sdi(yL).unsqueeze(0)
        # x_HL = self.sdi(y_HL).unsqueeze(0)
        # x_LH = self.sdi(y_LH).unsqueeze(0)
        # x_HH = self.sdi(y_HH).unsqueeze(0)
        xL = self.sdi(yL)
        x_HL = self.sdi(y_HL)
        x_LH = self.sdi(y_LH)
        x_HH = self.sdi(y_HH)
        x = torch.cat([xL, x_HL, x_LH, x_HH], dim=1)
        x = self.conv(x)
        return x

# import pytorch_wavelets.dwt.lowlevel as lowlevel
import torch.nn.functional as F
import numpy as np
class Fourierforward(nn.Module):
    def __init__(self, J=1):
        super().__init__()
        self.J = J
        # self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT using Fourier Transform.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x
        # mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):

            # 计算每个维度需要补充的零数
            # signal_size = ll.shape
            # padded_size = [2 ** n for n in range(10) if 2 ** n >= signal_size[-2]]
            # padded_size_y = [2 ** n for n in range(10) if 2 ** n >= signal_size[-1]]
            #
            # # 零填充
            # ll_padded = torch.nn.functional.pad(ll, (0, padded_size[-1] - signal_size[0], 0, padded_size_y[-1] - signal_size[1]))

            # 定义填充后的尺寸  # 补充2的幂次 但是后面的变成（1，8，159，159）本身只要到160就可，但是补充到了256
            padded_size = [2 ** int(np.ceil(np.log2(ll.shape[-2]))), 2 ** int(np.ceil(np.log2(ll.shape[-1])))]

            # 0填充张量
            ll_padded = F.pad(ll, (0, padded_size[-1] - ll.shape[-1], 0, padded_size[-2] - ll.shape[-2]),
                              mode='constant', value=0)
            # Do 1 level of the transform using Fourier Transform
            # 进行傅里叶变换
            ll_freq = torch.fft.rfft2(ll_padded, dim=(-2, -1))  # 官方文档说是用来rfft处理都是实数的输入。one-side输出
            # ll_freq = torch.fft.fft2(ll_padded, dim=(-2, -1))  # two-side输出
            # ll_freq = torch.fft.rfft2(ll, dim=(-2, -1))
            h, w = ll.size(-2), ll.size(-1)

            # h, w = ll_padded.size(-2), ll_padded.size(-1)

            ll_low_freq = ll_freq[:, :, :h//2+1, :w//2+1]  # Get low frequency components
            # ll_low_freq = ll_freq[:, :, :h//2, :w//2] # Get high frequency components
            # ll = torch.fft.irfft2(ll_low_freq, s=(h, w), dim=(-2, -1))
            ll = ll_low_freq.real

            # high_freq = ll_freq - ll_low_freq
            high_freq = ll_freq[:, :, h//2+1:h+1, :w//2+1] # Get high frequency components
            # high_freq = ll_freq[:, :, h//2:h , :w//2]  # Get high frequency components
            y1 = ll_low_freq.imag
            y2 = high_freq.real
            y3 = high_freq.imag

            # 添加到一个列表中
            yh = [y1, y2, y3]
            # high = torch.cat([high_freq.real.unsqueeze(-1), high_freq.imag.unsqueeze(-1)], dim=-1)
            # yh.append(high)

        return ll, yh


################# CARAFE上采样模块 ##############################################################
class CARAFE(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = torch.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X
########################################### CARAFE上采样模块 ######################################

#############SDI
##### U.-NET V2: RETHINKING THE SKIP CONNECTIONS OF U-NET FOR MEDICAL IMAGE
class SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(c, channel[0], kernel_size=3, stride=1, padding=1) for c in channel])

    def forward(self, xs):
        ans = torch.ones_like(xs[0])
        target_size = xs[0].shape[-2:]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[0]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[0]:
                x = F.interpolate(x, size=(target_size[0], target_size[1]),
                                  mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x)

        return ans

class SDI1(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(c, channel[0], kernel_size=3, stride=1, padding=1) for c in channel])

    def forward(self, xs):
        ans = torch.ones_like(xs[0])
        target_size = xs[0].shape[-2:]
        ans_list = []
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[0]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[0]:
                x = F.interpolate(x, size=(target_size[0], target_size[1]),
                                  mode='bilinear', align_corners=True)
            # 如果有batchsize 所以用list # 多个batch就联立起来
            ans = ans * self.convs[0](x)
            ans_list.append(ans)
        ans = torch.stack(ans_list, dim=0)
        return ans

class SDI2(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(c, channel[0], kernel_size=3, stride=1, padding=1) for c in channel])

    def forward(self, xs):
        ans= torch.ones_like(xs[0])
        for i, x in enumerate(xs):
            ans = torch.ones_like(xs[i])
            target_size = xs[i].shape[-2:]
            if x.shape[-1] > target_size[0]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[0]:
                x = F.interpolate(x, size=(target_size[0], target_size[1]),
                                  mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x)

        return ans
############# SDI


#####This module implements the OctConv paper
from functools import partial

class OctConv(torch.nn.Module):
    """
    This module implements the OctConv paper https://arxiv.org/pdf/1904.05049v1.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, alpha_in=0.5, alpha_out=0.5):
        super(OctConv, self).__init__()
        self.alpha_in, self.alpha_out, self.kernel_size = alpha_in, alpha_out, kernel_size
        self.H2H, self.L2L, self.H2L, self.L2H = None, None, None, None
        if not (alpha_in == 0.0 and alpha_out == 0.0):
            self.L2L = torch.nn.Conv2d(int(alpha_in * in_channels),
                                       int(alpha_out * out_channels),
                                       kernel_size, stride, kernel_size//2)
        if not (alpha_in == 0.0 and alpha_out == 1.0):
            self.L2H = torch.nn.Conv2d(int(alpha_in * in_channels),
                                       out_channels - int(alpha_out * out_channels),
                                       kernel_size, stride, kernel_size//2)
        if not (alpha_in == 1.0 and alpha_out == 0.0):
            self.H2L = torch.nn.Conv2d(in_channels - int(alpha_in * in_channels),
                                       int(alpha_out * out_channels),
                                       kernel_size, stride, kernel_size//2)
        if not (alpha_in == 1.0 and alpha_out == 1.0):
            self.H2H = torch.nn.Conv2d(in_channels - int(alpha_in * in_channels),
                                       out_channels - int(alpha_out * out_channels),
                                       kernel_size, stride, kernel_size//2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.avg_pool = partial(torch.nn.functional.avg_pool2d, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        hf, lf = x
        h2h, l2l, h2l, l2h = None, None, None, None
        if self.H2H is not None:
            h2h = self.H2H(hf)
        if self.L2L is not None:
            l2l = self.L2L(lf)
        if self.H2L is not None:
            h2l = self.H2L(self.avg_pool(hf))
        if self.L2H is not None:
            l2h = self.upsample(self.L2H(lf))
        hf_, lf_ = 0, 0
        for i in [h2h, l2h]:
            if i is not None:
                hf_ = hf_ + i
        for i in [l2l, h2l]:
            if i is not None:
                lf_ = lf_ + i
        return hf_, lf_
#####This module implements the OctConv paper


##########Fast Fourier Convolution
class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * \
            self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
            self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        # (batch, c, h, w/2+1, 2)
        # ffted = torch.rfft(x, signal_ndim=2, normalized=True)
        # ffted = torch.fft.rfft(x, signal_ndim=2, normalized=True)
        ffted = rfft(x, 2)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)

        # output = torch.irfft(ffted, signal_ndim=2,
        #                      signal_sizes=r_size[2:], normalized=True)
        # output = torch.fft.irfft(ffted, signal_ndim=2,
        #                      signal_sizes=r_size[2:], normalized=True)
        output = irfft(ffted, d=2, signal_sizes=r_size[2:])

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output

class SpectralTransform1(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransform1, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w// split_no
            if c==2:
                xs1= torch.split(x[:, :c // 2], split_s_h, dim=-2)[0]
                xs2 = torch.split(x[:, :c // 2], split_s_h, dim=-2)[1]
                xs3 = torch.cat([xs1,xs2], dim=1)
                xs =xs3.contiguous()

                xss1= torch.split(xs, split_s_w, dim=-1)[0]
                xss2 = torch.split(xs, split_s_w, dim=-1)[1]
                xss3 = torch.cat([xss1, xss2], dim=1)
                xs =xss3.contiguous()
            else:
                xs1 = torch.split(x[:, :c // 4], split_s_h, dim=-2)[0]
                xs2 = torch.split(x[:, :c // 4], split_s_h, dim=-2)[1]
                xs3 = torch.cat([xs1, xs2], dim=1)
                xs = xs3.contiguous()

                xss1 = torch.split(xs, split_s_w, dim=-1)[0]
                xss2 = torch.split(xs, split_s_w, dim=-1)[1]
                xss3 = torch.cat([xss1, xss2], dim=1)
                xs = xss3.contiguous()
            # xs = torch.cat(torch.split(
            #     x[:, :c // 2], split_s_h, dim=-2), dim=1).contiguous()
            # xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
            #                dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output

class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform1
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 enable_lfu=True):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)))
        self.bn_g = gnorm(int(out_channels * ratio_gout))

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

##########Fast Fourier Convolution


## V11的改进
class Bottleneck_RFAConv(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = RFAConv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_RFAConv(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_RFAConv(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3_RFAConv(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck_RFAConv(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k_RFAConv(C3_RFAConv):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_RFAConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_RFAConv(C2f_RFAConv):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_RFAConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_RFAConv(self.c, self.c, shortcut, g) for _ in range(n)
        )



### 加入注意力机制 CBAM CA
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SE(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(SE, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channel, ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(ratio, in_channel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[0:2]
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class RFCBAMConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        if kernel_size % 2 == 0:
            assert ("the kernel_size must be  odd.")
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU()
            )
        self.get_weight = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False), nn.Sigmoid())
        self.se = SE(in_channel)

        # self.conv = nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size,stride=kernel_size),nn.BatchNorm2d(out_channel),nn.ReLu())
        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)

    def forward(self, x):
        b, c = x.shape[0:2]
        channel_attention = self.se(x)
        generate_feature = self.generate(x)

        h, w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b, c, self.kernel_size ** 2, h, w)

        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                                     n2=self.kernel_size)

        unfold_feature = generate_feature * channel_attention
        max_feature, _ = torch.max(generate_feature, dim=1, keepdim=True)
        mean_feature = torch.mean(generate_feature, dim=1, keepdim=True)
        receptive_field_attention = self.get_weight(torch.cat((max_feature, mean_feature), dim=1))
        conv_data = unfold_feature * receptive_field_attention
        return self.conv(conv_data)


class RFCAConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride=1, reduction=32):
        super(RFCAConv, self).__init__()
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(nn.Conv2d(inp, inp * (kernel_size ** 2), kernel_size, padding=kernel_size // 2,
                                                stride=stride, groups=inp,
                                                bias=False),
                                      nn.BatchNorm2d(inp * (kernel_size ** 2)),
                                      nn.ReLU()
                                      )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, kernel_size, stride=kernel_size))

    def forward(self, x):
        b, c = x.shape[0:2]
        generate_feature = self.generate(x)
        h, w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b, c, self.kernel_size ** 2, h, w)

        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                                     n2=self.kernel_size)

        x_h = self.pool_h(generate_feature)
        x_w = self.pool_w(generate_feature).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        h, w = generate_feature.shape[2:]
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return self.conv(generate_feature * a_w * a_h)


####  RFCBAMConv的Bottleneck， C3， C2f， C3k2变体
class Bottleneck_RFCBAMConv(nn.Module):
    """Standard bottleneck with RFCBAMConv."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = RFCBAMConv(c_, c2, 3)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3_RFCBAMConv(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_RFCBAMConv(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C2f_RFCBAMConv(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_RFCBAMConv(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k_RFCBAMConv(C3_RFCBAMConv):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_RFCBAMConv(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class C3k2_RFCBAMConv(C2f_RFCBAMConv):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_RFCBAMConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_RFCBAMConv(self.c, self.c, shortcut, g) for _ in range(n)
        )

####  RFCBAMConv的Bottleneck， C3， C2f， C3k2变体


####  RFCAonv的Bottleneck， C3， C2f， C3k2变体
class Bottleneck_RFCAConv(nn.Module):
    """Standard bottleneck with RFCBAMConv."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = RFCAConv(c_, c2, 3)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3_RFCAConv(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_RFCAConv(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C2f_RFCAConv(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_RFCAConv(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k_RFCAConv(C3_RFCAConv):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_RFCAConv(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class C3k2_RFCAConv(C2f_RFCAConv):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_RFCAConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_RFCAConv(self.c, self.c, shortcut, g) for _ in range(n)
        )


# class DWT_FFT_ADown_CARAFE(nn.Module):
#     def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
#         super().__init__()
#         self.c = c2 // 2
#         # self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
#         # self.cv1 = HWD(c1 // 2, self.c)
#         self.cv1 = dwt2fftModel(c1 // 2, self.c)
#         # self.cv1 = HWD_pytorch(c1 // 2, self.c)
#         self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
#         self.cv3 = CARAFE(c2)
#         # self.cv3 = Conv(c1 // 2, self.c, 1, 1, 0)
#
#
#     def forward(self, x):
#         x = nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
#         x1, x2 = x.chunk(2, 1)
#         x1 = self.cv1(x1)
#         x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
#         x2 = self.cv2(x2)
#         x3 = torch.cat((x1, x2), 1)
#         x3 = self.cv3(x3)
#         # return torch.cat((x1, x2), 1)
#         return x3

# # WT转换为可逆变换（如CDF9/7小波变换），然后将其与FFT结合使用
#
# import pywt
#
# class dwt2fftModel(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(dwt2fftModel, self).__init__()
#         self.conv = Conv(in_ch * 4, out_ch, 1, 1)
#
#     def forward(self, x):
#         yL, yH = self.dwt_to_fft(x)
#         y_HL = yH[0][:, :, 0, ::]
#         y_LH = yH[0][:, :, 1, ::]
#         y_HH = yH[0][:, :, 2, ::]
#         # y_HL = yH[0]
#         # y_LH = yH[1]
#         # y_HH = yH[2]
#         x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
#         x = self.conv(x)
#         return x
#
#     def dwt_to_fft(self, x, wavelet ='cdf9/7'):
#         # Compute DWT coefficients
#         coeffs = pywt.wavedecn(x, wavelet)
#
#         # Convert DWT coefficients to frequency domain
#         for i in range(len(coeffs)):
#             if i == 0:
#                 continue
#             for j in range(len(coeffs[i])):
#                 coeffs[i][j] = np.fft.fft2(coeffs[i][j])
#
#         # Perform FFT on frequency domain coefficients
#         for i in range(len(coeffs)):
#             if i == 0:
#                 continue
#             for j in range(len(coeffs[i])):
#                 coeffs[i][j] = np.fft.fftshift(coeffs[i][j])
#                 coeffs[i][j] = np.fft.fft2(coeffs[i][j])
#                 coeffs[i][j] = np.fft.ifftshift(coeffs[i][j])
#
#         # Convert frequency domain coefficients back to time domain
#         for i in range(len(coeffs)):
#             if i == 0:
#                 continue
#             for j in range(len(coeffs[i])):
#                 coeffs[i][j] = np.fft.ifft2(coeffs[i][j]).real
#
#         # Reconstruct image from lowpass and bandpass parts
#         yl = pywt.waverecn(coeffs, wavelet)
#         yh = [coeffs[i][j] for i in range(1,len(coeffs)) for j in range(len(coeffs[i]))]
#
#         return yl, yh



# # 下面尝试的都是出错的
# # 转换成pytorch的版本，不然测试会有问题，但是下面代码吗还是有问题
# class HWD_pytorch(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(HWD_pytorch, self).__init__()
#         # self.wt = haar_wavelet_transform(J=1, mode='zero', wave='haar')
#         self.conv = Conv(in_ch * 4, out_ch, 1, 1)
#
#     def forward(self, x):
#         # yL, yH = self.wt(x)
#         yL, yH = haar_wavelet_transform(x)
#         y_HL = yH[0][:, :, 0, ::]
#         y_LH = yH[0][:, :, 1, ::]
#         y_HH = yH[0][:, :, 2, ::]
#         x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
#         x = self.conv(x)
#         return x
#
# # 手动实现小波变换HWD  出错了
# def haar_wavelet_transform(x):
#     length = x.size(-1)
#     # 确保信号长度为 2 的幂次方，如果不是，则补零
#     if length % 2 != 0:
#         x = torch.nn.functional.pad(x, (0, 1))
#         length += 1
#
#     # 小波滤波器
#     h0 = torch.tensor([1, 1]) / torch.sqrt(torch.tensor(2.0))
#     h1 = torch.tensor([-1, 1]) / torch.sqrt(torch.tensor(2.0))
#
#     # 奇数索引和偶数索引系数
#     even = x[..., ::2]
#     odd = x[..., 1::2]
#
#     # 进行卷积
#     # cA = torch.nn.functional.conv1d(even, h0.unsqueeze(0), padding=0)
#     # cD = torch.nn.functional.conv1d(odd, h1.unsqueeze(0), padding=0)
#     cA = h0(even)
#     cD = h1(even)
#     return cA, cD

#
# # 傅里叶变换
# class FtModel(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(FtModel, self).__init__()
#         self.conv = Conv(in_ch * 4, out_ch, 1, 1)
#
#     def forward(self, x):
#         yL, yH = self.ft(x)
#         y_HL = yH[0][:, :, 0, ::]
#         y_LH = yH[0][:, :, 1, ::]
#         y_HH = yH[0][:, :, 2, ::]
#         x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
#         x = self.conv(x)
#         return x
#
#     def ft(self, x):
#         # 对输入图像进行二维傅里叶变换
#         Fx = torch.fft.fftn(x, dim=(-2, -1))
#
#         # 获取图像的中心点坐标
#         h, w = x.shape[-2:]
#         cy, cx = h // 2, w // 2
#
#         # 创建一个与输入图像相同大小的掩码，将图像中心以外的频率置为零
#         mask = torch.zeros_like(x)
#         mask[:, :, cy - 1:cy + 2, cx - 1:cx + 2] = 1
#
#         # 将傅里叶变换后的图像乘以掩码，保留低频信息
#         lowpass = Fx * mask
#
#         # 将高频信息设置为零
#         highpass = Fx * (1 - mask)
#
#         # 将处理后的频谱转换回空间域
#         yL = torch.fft.ifftn(lowpass, dim=(-2, -1))
#         yH = torch.fft.ifftn(highpass, dim=(-2, -1))
#
#         # 返回低通信息和带通信息的元组
#         return yL, yH
#
#
# # 尝试的，会出错
# class FourierModel1(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(FourierModel1, self).__init__()
#         self.ft = Fourierforward(J=1, mode='zero')
#         self.conv = Conv(in_ch * 4, out_ch, 1, 1)
#
#     def ft1(self, x):
#         # 执行傅里叶变换
#         # Separate low frequency (DC component) and high frequency components
#         yf = torch.fft.fftn(x, dim=(-2, -1))
#         # yl = yf[..., 0:1, 0:1]
#         # Apply low-pass filter
#         # yl = torch.fft.ifftn(yf * self.filter.reshape((1, 1) + self.filter.shape), dim=(-2, -1)).unsqueeze(1)
#         yl = torch.stack((yf[..., 0:1, 0:1], torch.roll(yf[..., 0:1, 0:1], shifts=(-1, -1), dims=(-2, -1))))
#         yh = torch.stack((yf[..., 1:, :], torch.roll(yf[..., 1:, :], shifts=(-1, -1), dims=(-2, -1))), dim=3)
#
#         # Inverse FFT to obtain lowpass and bandpass coefficients
#         yl = torch.fft.ifftn(yl, dim=(-2, -1)).real
#         yh = torch.fft.ifftn(yh, dim=(-2, -1)).real
#
#         return yl, yh
#
#     def ft2(self, x):
#         # 获取图像的高度和宽度
#         height, width = x.shape[-2], x.shape[-1]
#
#         # 对图像进行二维傅里叶变换
#         f = torch.fft.fftn(x, dim=(-2, -1))
#
#         # 计算低频部分和带通部分
#         # 保留中心部分以获取低频信息
#         low_freq = f[:, :8, :height // 2 + 1, :width // 2 + 1].real
#         # low_freq = f[:, :8, :height, :width].real
#         # Inverse FFT to obtain lowpass and bandpass coefficients
#         # yl = torch.fft.ifftn(low_freq, dim=(-2, -1)).real
#
#         # 带通信息包括右上、左下和右下象限
#         # 假设 high_freq 是带通信息的高频部分
#         right_top = f[:, :6, height // 2:, width // 2:].real
#         left_top = f[:, :6, :height // 2 + 1, width // 2:].real
#         right_bottom = f[:, :6, height // 2:, :width // 2 + 1].real
#         left_bottom = f[:, :6, :height // 2+ 1, :width // 2+ 1].real
#
#         # 将四个添加到一个列表中
#         high_freq = [right_top, left_top, right_bottom, left_bottom]
#         # high_freq = f[:, :, height // 2:, width // 2:]
#         # high_freq = torch.stack((f[:, :,  height // 2:, width // 2:],
#         #                         torch.roll(f[:, :,  height // 2:, width // 2:], shifts=(-1, -1), dims=(-2, -1))),
#         #                         dim=3)
#
#         # 返回低通和带通信息作为元组
#         return low_freq, high_freq
#
#     def forward(self, x):
#         yL, yH = self.ft2(x)
#         # y_HL = yH[:, :, :, 0, ::]
#         # y_LH = yH[:, :, :, 1, ::]
#         # y_HH = yH[:, :, :, 2, ::]
#         # x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
#         yH0 = yH[0]
#         yH1 = yH[1]
#         yH2 = yH[2]
#         yH3 = yH[3]
#         x = torch.cat([yL, yH0, yH1, yH2, yH3], dim=1)
#         x = self.conv(x)
#         return x

    # def forward(self, x):
    #     yL, yH = self.ft(x)
    #     # y_HL = yH[0][:, :, 0, ::]
    #     # y_LH = yH[0][:, :, 1, ::]
    #     # y_HH = yH[0][:, :, 2, ::]
    #     y_HL = yH[0]
    #     y_LH = yH[1]
    #     y_HH = yH[2]
    #     x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
    #     x = self.conv(x)
    #     return x


