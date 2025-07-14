import torch
import torch.nn as nn
from .ffc import *


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class SPDConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        c1 = c1 * 4
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.act(self.conv(x))

    def __str__(self):
        return 'SPDConv'

class SPDConv1(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x

## FFC放入SPDConv可以跑的
class SPDConv_FFC(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = FFC(inc*4, ouc, kernel_size=3, ratio_gin=0.5, ratio_gout=0.5, padding=1)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x_1, x_2= x.chunk(2, 1)
        x_tuple = (x_1, x_2)
        x = self.conv(x_tuple)
        # # 按第 0 维度拼接特征图
        x = torch.cat([fm for fm in x], dim=1)
        return x

### Oct里面能放到RFAConv里面可以跑的
class SPDConv_Oct_CARAFE_HWD(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = OctConv_CARAFE_HWD(inc*4, ouc, kernel_size= 1)
        # kernel_size是1，如果是3，后面的self.avg_pool得到的结果就是尺寸缩小为[1, c, h//kernel_size, h//kernel_size]，通道对不准

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x_1, x_2= x.chunk(2, 1)
        x_tuple = (x_1, x_2)
        x = self.conv(x_tuple)
        # # 按第 0 维度拼接特征图
        x = torch.cat([fm for fm in x], dim=1)
        return x

## 按放到SPDConv里面跑
class SPDConv_Oct_CARAFE_HWD_Conv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = OctConv_CARAFE_HWD_Conv(inc*4, ouc, kernel_size= 1)
        # kernel_size是1，如果是3，后面的self.avg_pool得到的结果就是尺寸缩小为[1, c, h//kernel_size, h//kernel_size]，通道对不准

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x_1, x_2= x.chunk(2, 1)
        x_tuple = (x_1, x_2)
        x = self.conv(x_tuple)
        # # 按第 0 维度拼接特征图
        x = torch.cat([fm for fm in x], dim=1)
        return x

## 按放到SPDConv里面跑
class SPDConv_Oct_CARAFE_HWD_ADown(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = OctConv_CARAFE_HWD_ADown(inc*4, ouc, kernel_size= 1)
        # kernel_size是1，如果是3，后面的self.avg_pool得到的结果就是尺寸缩小为[1, c, h//kernel_size, h//kernel_size]，通道对不准

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x_1, x_2= x.chunk(2, 1)
        x_tuple = (x_1, x_2)
        x = self.conv(x_tuple)
        # # 按第 0 维度拼接特征图
        x = torch.cat([fm for fm in x], dim=1)

        return x


### 下面放到RFAConv里面不能跑。直接当卷积也不行
class SPDConv_Oct(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = OctConv(inc*4, ouc, kernel_size=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x_1, x_2= x.chunk(2, 1)
        x_tuple = (x_1, x_2)
        x = self.conv(x_tuple)
        # # 按第 0 维度拼接特征图
        x = torch.cat([fm for fm in x], dim=1)

        return x

class SPDConv_Oct_CARAFE(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = OctConv_CARAFE(inc*4, ouc, kernel_size= 1)
        # kernel_size是1，如果是3，后面的self.avg_pool得到的结果就是尺寸缩小为[1, c, h//kernel_size, h//kernel_size]，通道对不准

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x_1, x_2= x.chunk(2, 1)
        x_tuple = (x_1, x_2)
        x = self.conv(x_tuple)
        # # 按第 0 维度拼接特征图
        x = torch.cat([fm for fm in x], dim=1)
        return x

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


class OctConv_CARAFE(torch.nn.Module):
    """
    This module implements the OctConv paper https://arxiv.org/pdf/1904.05049v1.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, alpha_in=0.5, alpha_out=0.5):
        super(OctConv_CARAFE, self).__init__()
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
        self.cv1 = CARAFE(out_channels - int(alpha_out * out_channels))
        self.cv2 = CARAFE(int(alpha_out * out_channels))
        # self.cv2 = HWD_ADown(in_channels - int(alpha_in * in_channels),int(alpha_out * out_channels))

    def forward(self, x):
        hf, lf = x
        h2h, l2l, h2l, l2h = None, None, None, None
        if self.H2H is not None:
            h2h = self.H2H(hf)
            h2h = self.cv1(h2h)
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
        lf_=self.cv2(lf_)
        return hf_, lf_

####为RFA_SPD_OctConv_CARAFE_HWD准备的，HWD参数要除以4
class OctConv_CARAFE_HWD(torch.nn.Module):
    """
    This module implements the OctConv paper https://arxiv.org/pdf/1904.05049v1.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, alpha_in=0.5, alpha_out=0.5):
        super(OctConv_CARAFE_HWD, self).__init__()
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
        self.cv1 = CARAFE(out_channels - int(alpha_out * out_channels))
        # self.cv2 = CARAFE(int(alpha_out * out_channels))
        # self.cv3 = HWD_ADown((in_channels - int(alpha_in * in_channels))//16, int(alpha_out * out_channels))
        self.cv2 = HWD((in_channels - int(alpha_in * in_channels)) // 4, int(alpha_out * out_channels))

    def forward(self, x):
        hf, lf = x
        h2h, l2l, h2l, l2h = None, None, None, None
        if self.H2H is not None:
            h2h = self.H2H(hf)
            h2h = self.cv1(h2h)
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
        hf_ = self.cv2(hf_)
        return hf_, lf_

####为SPD_OctConv_CARAFE_HWD_Conv准备的，HWD参数要除以2
class OctConv_CARAFE_HWD_Conv(torch.nn.Module):
    """
    This module implements the OctConv paper https://arxiv.org/pdf/1904.05049v1.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, alpha_in=0.5, alpha_out=0.5):
        super(OctConv_CARAFE_HWD_Conv, self).__init__()
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
        self.cv1 = CARAFE(out_channels - int(alpha_out * out_channels))
        # self.cv2 = CARAFE(int(alpha_out * out_channels))
        # self.cv3 = HWD_ADown((in_channels - int(alpha_in * in_channels))//16, int(alpha_out * out_channels))
        self.cv2 = HWD((in_channels - int(alpha_in * in_channels)) // 2, int(alpha_out * out_channels))

    def forward(self, x):
        hf, lf = x
        h2h, l2l, h2l, l2h = None, None, None, None
        if self.H2H is not None:
            h2h = self.H2H(hf)
            h2h = self.cv1(h2h)
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
        hf_ = self.cv2(hf_)
        return hf_, lf_

####为SPD_OctConv_CARAFE_HWD_ADown准备的，HWD参数要除以2
class OctConv_CARAFE_HWD_ADown(torch.nn.Module):
    """
    This module implements the OctConv paper https://arxiv.org/pdf/1904.05049v1.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, alpha_in=0.5, alpha_out=0.5):
        super(OctConv_CARAFE_HWD_ADown, self).__init__()
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
        self.cv1 = CARAFE(out_channels - int(alpha_out * out_channels))
        # self.cv2 = CARAFE(int(alpha_out * out_channels))
        self.cv3 = HWD_ADown((in_channels - int(alpha_in * in_channels))//2, int(alpha_out * out_channels))

    def forward(self, x):
        hf, lf = x
        h2h, l2l, h2l, l2h = None, None, None, None
        if self.H2H is not None:
            h2h = self.H2H(hf)
            h2h = self.cv1(h2h)
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
        # lf_=self.cv2(lf_)
        hf_ = self.cv3(hf_)
        return hf_, lf_


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

class HWD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        from pytorch_wavelets import DWTForward
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



###V11
class Bottleneck_SPDConv(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = SPDConv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f_SPDConv(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_SPDConv(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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


class C3_SPDConv(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck_SPDConv(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k_SPDConv(C3_SPDConv):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_SPDConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_SPDConv(C2f_SPDConv):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_SPDConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_SPDConv(self.c, self.c, shortcut, g) for _ in range(n)
        )