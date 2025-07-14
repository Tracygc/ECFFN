import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import DFL
from ultralytics.nn.modules.conv import Conv
from ultralytics.utils.tal import dist2bbox, make_anchors
import torch.utils.checkpoint as cp

from ultralytics.nn.Addmodules.Recursive_AFPN import ASPP, Bottleneck, Upsample, BasicConv, BlockBody


__all__ = ("Recursive_ASFF2_BiFPN_Add2", "Recursive_ASFF3_BiFPN_Add3")


class BiFPN_Add2(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add2, self).__init__()
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1]))


class BiFPN_Add3(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add3, self).__init__()
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        # Fast normalized fusion
        return self.conv(
            self.silu(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2])
        )


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Bi_FPN(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(length, dtype=torch.float32), requires_grad=True)
        self.swish = swish()
        self.epsilon = 0.0001

    def forward(self, x):
        weights = self.weight / (torch.sum(self.swish(self.weight), dim=0) + self.epsilon)  # 权重归一化处理
        weighted_feature_maps = [weights[i] * x[i] for i in range(len(x))]
        stacked_feature_maps = torch.stack(weighted_feature_maps, dim=0)
        result = torch.sum(stacked_feature_maps, dim=0)
        return result


class RecursiveFPN(nn.Module):
    def __init__(self,
                 out_indices=(0, 1, 2, 3),
                 rfp_steps=2,
                 rfp_sharing=False,
                 stage_with_rfp=(False, True, True, True),
                 neck_out_channels=128):
        super().__init__()
        self.rfp_steps = rfp_steps
        self.rfp_sharing = rfp_sharing
        self.stage_with_rfp = stage_with_rfp

        self.out_indices = out_indices
        if not self.rfp_sharing:
            self.rfp_modules = torch.nn.ModuleList()
        self.rfp_aspp = ASPP(neck_out_channels, neck_out_channels // 4)
        self.rfp_weight = torch.nn.Conv2d(
            neck_out_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.rfp_weight.weight.data.fill_(0)
        self.rfp_weight.bias.data.fill_(0)

        self.conv1 = Conv(neck_out_channels, neck_out_channels, 3, 1)
        # Batch Normalization (BatchNorm2d): 适用于大多数任务，可以有效地提高训练速度和稳定性。
        # Layer Normalization (LayerNorm): 更适用于序列模型或小批量数据。
        # Instance Normalization (InstanceNorm2d): 常用于图像风格转换任务。
        # Group Normalization (GroupNorm): 对小批量数据效果较好，特别是在内存有限的情况下。

        self.norm1 = nn.BatchNorm2d( neck_out_channels)
        # self.norm1 = nn.LayerNorm([64, 32, 32])  # 根据输入的尺寸调整
        # self.norm1 = nn.InstanceNorm2d(64)
        # self.norm1 = nn.GroupNorm(num_groups=32, num_channels=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.res_layers = nn.ModuleList([
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        # ])

        self.res_layers = nn.ModuleList([
            Bottleneck(32, 8),
            Bottleneck(64, 16),
            Bottleneck(128, 32),
            Bottleneck(256, 64),
        ])

        self.res_layer =  Bottleneck( neck_out_channels,  neck_out_channels//4)
        self.upsample= Upsample( neck_out_channels ,  neck_out_channels)


    def rfp_forward(self, x, rfp_feats):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = self.res_layer.rfp_forward1(x,rfp_feats)
        # 使用双线性插值进行上采样，将张量大小调整到 (1, 32, 32, 32)
        # x_upsampled = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        outs =  self.upsample(outs)
        return outs



    def forward(self, x):
        # x = self.backbone(img)
        # x = self.neck(x)
        for rfp_idx in range(self.rfp_steps - 1):
            rfp_feats = tuple(self.rfp_aspp(x))
            #rfp_feats = tuple(self.rfp_aspp(x[i]) if self.stage_with_rfp[i] else x[i]
             #                 for i in range(len(self.stage_with_rfp)))
            x_idx = self.rfp_forward(x, rfp_feats)
            # 调整张量大小
            if x.size() != x_idx.size():
                x_idx = F.interpolate(x_idx, size=x.size()[2:], mode='bilinear', align_corners=False)

            add_weight = torch.sigmoid(self.rfp_weight(x_idx))
            x_new =  add_weight * x_idx+ (1 - add_weight) * x
            x = x_new
        return x


class Recursive_BiFPN(nn.Module):
      "The code will be released soon."

class Recursive_ASFF2_BiFPN_Add2(nn.Module):
    "The code will be released soon."



class Recursive_ASFF3_BiFPN_Add3(nn.Module):
    "The code will be released soon."