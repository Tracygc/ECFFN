import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import DFL
from ultralytics.nn.modules.conv import Conv
from ultralytics.utils.tal import dist2bbox, make_anchors
import torch.utils.checkpoint as cp


__all__ = ['Recursive_ASFF2', 'Recursive_ASFF3']

def BasicConv(filter_in, filter_out, kernel_size, stride=1, pad=None):
    if not pad:
        pad = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        pad = pad
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU(inplace=True)),
    ]))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, filter_in, filter_out):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(filter_in, filter_out, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filter_out, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filter_out, filter_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filter_out, momentum=0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )

    def forward(self, x):
        x = self.upsample(x)

        return x


class Downsample_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x2, self).__init__()

        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 2, 2, 0)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x


class Downsample_x4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x4, self).__init__()

        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 4, 4, 0)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x


class Downsample_x8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x8, self).__init__()

        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 8, 8, 0)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x


class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_2, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2):
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]

        out = self.conv(fused_out_reduced)

        return out


class ASFF_3(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_3, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = BasicConv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2, input3):
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :] + \
                            input3 * levels_weight[:, 2:, :, :]

        out = self.conv(fused_out_reduced)

        return out


class ASFF_4(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_4, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_0 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input0, input1, input2):
        level_0_weight_v = self.weight_level_0(input0)
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input0 * levels_weight[:, 0:1, :, :] + \
                            input1 * levels_weight[:, 1:2, :, :] + \
                            input2 * levels_weight[:, 2:3, :, :]

        out = self.conv(fused_out_reduced)

        return out


class BlockBody(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512]):
        super(BlockBody, self).__init__()

        self.blocks_scalezero1 = nn.Sequential(
            BasicConv(channels[0], channels[0], 1),
        )
        self.blocks_scaleone1 = nn.Sequential(
            BasicConv(channels[1], channels[1], 1),
        )
        self.blocks_scaletwo1 = nn.Sequential(
            BasicConv(channels[2], channels[2], 1),
        )

        self.downsample_scalezero1_2 = Downsample_x2(channels[0], channels[1])
        self.upsample_scaleone1_2 = Upsample(channels[1], channels[0], scale_factor=2)

        self.asff_scalezero1 = ASFF_2(inter_dim=channels[0])
        self.asff_scaleone1 = ASFF_2(inter_dim=channels[1])

        self.blocks_scalezero2 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone2 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )

        self.downsample_scalezero2_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero2_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scaleone2_2 = Downsample_x2(channels[1], channels[2])
        self.upsample_scaleone2_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.upsample_scaletwo2_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.upsample_scaletwo2_4 = Upsample(channels[2], channels[0], scale_factor=4)

        self.asff_scalezero2 = ASFF_3(inter_dim=channels[0])
        self.asff_scaleone2 = ASFF_3(inter_dim=channels[1])
        self.asff_scaletwo2 = ASFF_3(inter_dim=channels[2])

        self.blocks_scalezero3 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone3 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo3 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )

        self.downsample_scalezero3_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero3_4 = Downsample_x4(channels[0], channels[2])
        self.upsample_scaleone3_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.downsample_scaleone3_2 = Downsample_x2(channels[1], channels[2])
        self.upsample_scaletwo3_4 = Upsample(channels[2], channels[0], scale_factor=4)
        self.upsample_scaletwo3_2 = Upsample(channels[2], channels[1], scale_factor=2)

        self.asff_scalezero3 = ASFF_4(inter_dim=channels[0])
        self.asff_scaleone3 = ASFF_4(inter_dim=channels[1])
        self.asff_scaletwo3 = ASFF_4(inter_dim=channels[2])

        self.blocks_scalezero4 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone4 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo4 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )

    def forward(self, x):
        x0, x1, x2 = x

        x0 = self.blocks_scalezero1(x0)
        x1 = self.blocks_scaleone1(x1)
        x2 = self.blocks_scaletwo1(x2)

        scalezero = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1))
        scaleone = self.asff_scaleone1(self.downsample_scalezero1_2(x0), x1)

        x0 = self.blocks_scalezero2(scalezero)
        x1 = self.blocks_scaleone2(scaleone)

        scalezero = self.asff_scalezero2(x0, self.upsample_scaleone2_2(x1), self.upsample_scaletwo2_4(x2))
        scaleone = self.asff_scaleone2(self.downsample_scalezero2_2(x0), x1, self.upsample_scaletwo2_2(x2))
        scaletwo = self.asff_scaletwo2(self.downsample_scalezero2_4(x0), self.downsample_scaleone2_2(x1), x2)

        x0 = self.blocks_scalezero3(scalezero)
        x1 = self.blocks_scaleone3(scaleone)
        x2 = self.blocks_scaletwo3(scaletwo)

        scalezero = self.asff_scalezero3(x0, self.upsample_scaleone3_2(x1), self.upsample_scaletwo3_4(x2))
        scaleone = self.asff_scaleone3(self.downsample_scalezero3_2(x0), x1, self.upsample_scaletwo3_2(x2))
        scaletwo = self.asff_scaletwo3(self.downsample_scalezero3_4(x0), self.downsample_scaleone3_2(x1), x2)

        scalezero = self.blocks_scalezero4(scalezero)
        scaleone = self.blocks_scaleone4(scaleone)
        scaletwo = self.blocks_scaletwo4(scaletwo)

        return scalezero, scaleone, scaletwo


class AFPN(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=128):
        super(AFPN, self).__init__()

        self.fp16_enabled = False

        self.conv0 = BasicConv(in_channels[0], in_channels[0] // 8, 1)
        self.conv1 = BasicConv(in_channels[1], in_channels[1] // 8, 1)
        self.conv2 = BasicConv(in_channels[2], in_channels[2] // 8, 1)
        # self.conv3 = BasicConv(in_channels[3], in_channels[3] // 8, 1)

        self.body = nn.Sequential(
            BlockBody([in_channels[0] // 8, in_channels[1] // 8, in_channels[2] // 8])
        )

        self.conv00 = BasicConv(in_channels[0] // 8, out_channels, 1)
        self.conv11 = BasicConv(in_channels[1] // 8, out_channels, 1)
        self.conv22 = BasicConv(in_channels[2] // 8, out_channels, 1)
        # self.conv33 = BasicConv(in_channels[3] // 8, out_channels, 1)
        # self.conv44 = nn.MaxPool2d(kernel_size=1, stride=2)

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x0, x1, x2 = x

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        # x3 = self.conv3(x3)

        out0, out1, out2 = self.body([x0, x1, x2])

        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)

        return out0, out1, out2



#### RECURsive
class ASPP(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        kernel_sizes = [1, 3, 3, 1]
        dilations = [1, 3, 6, 1]
        paddings = [0, 3, 6, 0]
        self.aspp = torch.nn.ModuleList()
        for aspp_idx in range(len(kernel_sizes)):
            conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_sizes[aspp_idx],
                stride=1,
                dilation=dilations[aspp_idx],
                padding=paddings[aspp_idx],
                bias=True)
            self.aspp.append(conv)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.aspp_num = len(kernel_sizes)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(self.aspp_num):
            inp = avg_x if (aspp_idx == self.aspp_num - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 gcb=None,
                 sac=None,
                 rfp=None,
                 gen_attention=None):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert gcb is None or isinstance(gcb, dict)
        assert sac is None or isinstance(sac, dict)
        assert gen_attention is None or isinstance(gen_attention, dict)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.gcb = gcb
        self.with_gcb = gcb is not None
        self.sac = sac
        self.with_sac = sac is not None
        self.gen_attention = gen_attention
        self.with_gen_attention = gen_attention is not None

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        # Batch Normalization (BatchNorm2d): 适用于大多数任务，可以有效地提高训练速度和稳定性。
        # Layer Normalization (LayerNorm): 更适用于序列模型或小批量数据。
        # Instance Normalization (InstanceNorm2d): 常用于图像风格转换任务。
        # Group Normalization (GroupNorm): 对小批量数据效果较好，特别是在内存有限的情况下。

        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)
        self.norm3 = nn.BatchNorm2d(planes * self.expansion)
        # self.norm1 = nn.LayerNorm([64, 32, 32])  # 根据输入的尺寸调整
        # self.norm1 = nn.InstanceNorm2d(64)
        # self.norm1 = nn.GroupNorm(num_groups=32, num_channels=64)



        # self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        # self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        # self.norm3_name, norm3 = build_norm_layer(
        #     norm_cfg, planes * self.expansion, postfix=3)

        # self.conv1 = build_conv_layer(
        #     conv_cfg,
        #     inplanes,
        #     planes,
        #     kernel_size=1,
        #     stride=self.conv1_stride,
        #     bias=False)

        self.conv1 = Conv(inplanes, planes, 1, self.conv1_stride)
        self.add_module('norm1',self.norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)

        if self.with_sac:
            # self.conv2 = build_conv_layer(
            #     sac,
            #     planes,
            #     planes,
            #     kernel_size=3,
            #     stride=self.conv2_stride,
            #     padding=dilation,
            #     dilation=dilation,
            #     bias=False)
            self.conv2 = Conv(planes, planes, 3, self.conv2_stride, p=dilation, d=dilation)
        elif not self.with_dcn or fallback_on_stride:
            # self.conv2 = build_conv_layer(
            #     conv_cfg,
            #     planes,
            #     planes,
            #     kernel_size=3,
            #     stride=self.conv2_stride,
            #     padding=dilation,
            #     dilation=dilation,
            #     bias=False)
            self.conv2 = Conv(planes, planes, 3, self.conv2_stride, p=dilation, d=dilation)
        else:
            assert self.conv_cfg is None, 'conv_cfg cannot be None for DCN'
            # self.conv2 = build_conv_layer(
            #     dcn,
            #     planes,
            #     planes,
            #     kernel_size=3,
            #     stride=self.conv2_stride,
            #     padding=dilation,
            #     dilation=dilation,
            #     bias=False)
            self.conv2 = Conv(planes, planes, 3, self.conv2_stride, p=dilation, d=dilation)
        self.add_module('norm2',self.norm2)

        # self.conv3 = build_conv_layer(
        #     conv_cfg,
        #     planes,
        #     planes * self.expansion,
        #     kernel_size=1,
        #     bias=False)
        self.conv3 = Conv(planes, planes * self.expansion,1)
        self.add_module('norm3',self.norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        # if self.with_gcb:
        #     gcb_inplanes = planes * self.expansion
        #     self.context_block = ContextBlock(inplanes=gcb_inplanes, **gcb)
        #
        # # gen_attention
        # if self.with_gen_attention:
        #     self.gen_attention_block = GeneralizedAttention(
        #         planes, **gen_attention)

        # recursive feature pyramid
        self.rfp = rfp
        if self.rfp:
            self.rfp_conv = torch.nn.Conv2d(
                self.rfp,
                planes * self.expansion,
                kernel_size=1,
                stride=1,
                bias=True)
            self.rfp_conv.weight.data.fill_(0)
            self.rfp_conv.bias.data.fill_(0)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            # if self.with_gen_attention:
            #     out = self.gen_attention_block(out)

            out = self.conv3(out)
            out = self.norm3(out)

            # if self.with_gcb:
            #     out = self.context_block(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

    def rfp_forward1(self, x, rfp_feat):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_gen_attention:
                out = self.gen_attention_block(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_gcb:
                out = self.context_block(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        if self.rfp:
            rfp_feat = self.rfp_conv(rfp_feat)
            out = out + rfp_feat

        out = self.relu(out)

        return out

### 自己修改的 去掉backbone 和neck
class RecursiveFPN(nn.Module):
    def __init__(self,
                 # num_stages,
                 # semantic_roi_extractor=None,
                 # semantic_head=None,
                 # semantic_fusion=('bbox', 'mask'),
                 # interleaved=True,
                 # mask_info_flow=True,
                 out_indices=(0, 1, 2, 3),
                 rfp_steps=2,
                 rfp_sharing=False,
                 stage_with_rfp=(False, True, True, True),
                 neck_out_channels=128):
        super().__init__()
        self.rfp_steps = rfp_steps
        self.rfp_sharing = rfp_sharing
        self.stage_with_rfp = stage_with_rfp
        # self.stage_with_rfp = [False, True, True, True]

        # neck_out_channels = kwargs["neck"]["out_channels"]

        # super().__init__(
        #          num_stages,
        #          semantic_roi_extractor,
        #          semantic_head,
        #          semantic_fusion,
        #          interleaved,
        #          mask_info_flow,
        #          )
        self.out_indices = out_indices
        if not self.rfp_sharing:
            self.rfp_modules = torch.nn.ModuleList()
            # for rfp_idx in range(1, rfp_steps):
            #     rfp_module = builder.build_backbone(backbone)
            #     rfp_module.init_weights(kwargs["pretrained"])
            #     self.rfp_modules.append(rfp_module)
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

        # outs = []
        # for i, layer_name in enumerate(self.res_layers):
        #     res_layer = Bottleneck(self, layer_name)
        #     rfp_feat = None
        #     if self.stage_with_rfp[i]:
        #         rfp_feat = rfp_feats[i]
        #     for layer in res_layer:
        #         x = layer.rfp_forward1(x,rfp_feat)
        #     if i in self.out_indices:
        #         outs.append(x)
        # return tuple(outs)
        #
        # for i, res_layer in enumerate(self.res_layers):

        #     # rfp_feat =  rfp_feat
        #     # if self.stage_with_rfp[i]:
        #     #     rfp_feat = rfp_feats[i]
        #     x = res_layer.rfp_forward1(x, rfp_feats)
        #     if i in self.out_indices:
        #         outs.append(x)
        # return tuple(outs)


    def forward(self, x):
        # x = self.backbone(img)
        # x = self.neck(x)
        for rfp_idx in range(self.rfp_steps - 1):
            rfp_feats = tuple(self.rfp_aspp(x))
            #rfp_feats = tuple(self.rfp_aspp(x[i]) if self.stage_with_rfp[i] else x[i]
             #                 for i in range(len(self.stage_with_rfp)))
            x_idx = self.rfp_forward(x, rfp_feats)
            # if self.rfp_sharing:
            #     x_idx = self.backbone.rfp_forward(x, rfp_feats)
            # else:
            #     x_idx = self.rfp_modules[rfp_idx].rfp_forward(x, rfp_feats)
            # x_idx = self.neck(x_idx)


            # # x_new = []
            # for ft_idx in range(len(x_idx)):
            #     add_weight = torch.sigmoid(self.rfp_weight(x_idx[ft_idx]))
            #     x_new.append(add_weight * x_idx[ft_idx] + (1 - add_weight) * x[ft_idx])
            # x = x_new
            # x = torch.tensor(x_new)
            # 调整张量大小
            if x.size() != x_idx.size():
                x_idx = F.interpolate(x_idx, size=x.size()[2:], mode='bilinear', align_corners=False)

            add_weight = torch.sigmoid(self.rfp_weight(x_idx))
            x_new =  add_weight * x_idx+ (1 - add_weight) * x
            x = x_new
        return x


class Recursive_ASFF2(nn.Module):
    """ASFF2 module for YOLO AFPN head https://arxiv.org/abs/2306.15988"""

    def __init__(self, c1, c2, level=0):
        super().__init__()
        c1_l, c1_h = c1[0], c1[1]
        self.level = level
        self.dim = c1_l, c1_h
        self.inter_dim = self.dim[self.level]
        compress_c = 8

        if level == 0:
            self.stride_level_1 = Upsample(c1_h, self.inter_dim)
        if level == 1:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 2, 2, 0)  # downsample 2x

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weights_levels = nn.Conv2d(
            compress_c * 2, 2, kernel_size=1, stride=1, padding=0
        )
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)


        # self.extract_feat = RecursiveFPN(neck_out_channels=self.inter_dim)
        self.layer = RecursiveFPN(neck_out_channels=self.inter_dim, stage_with_rfp=(True) )

    def forward(self, x):
        x_level_0, x_level_1 = x[0], x[1]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), 1)
        levels_weight = self.weights_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = (
            level_0_resized * levels_weight[:, 0:1]
            + level_1_resized * levels_weight[:, 1:2]
        )

        out = self.conv(fused_out_reduced)

        # extract_feat = RecursiveFPN(neck_out_channels=self.inter_dim,stage_with_rfp=(True),)
        rout0 = self.layer(out)

        return rout0





class Recursive_ASFF3(nn.Module):
    """ASFF3 module for YOLO AFPN head https://arxiv.org/abs/2306.15988"""

    def __init__(self, c1, c2, level=0):
        super().__init__()
        c1_l, c1_m, c1_h = c1[0], c1[1], c1[2]
        self.level = level
        self.dim = c1_l, c1_m, c1_h
        self.inter_dim = self.dim[self.level]
        compress_c = 8

        if level == 0:
            self.stride_level_1 = Upsample(c1_m, self.inter_dim)
            self.stride_level_2 = Upsample(c1_h, self.inter_dim, scale_factor=4)

        if level == 1:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 2, 2, 0)  # downsample 2x
            self.stride_level_2 = Upsample(c1_h, self.inter_dim)

        if level == 2:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 4, 4, 0)  # downsample 4x
            self.stride_level_1 = Conv(c1_m, self.inter_dim, 2, 2, 0)  # downsample 2x

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weights_levels = nn.Conv2d(
            compress_c * 3, 3, kernel_size=1, stride=1, padding=0
        )
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)
        self.layer = RecursiveFPN(neck_out_channels=self.inter_dim, stage_with_rfp=(True) )



    def forward(self, x):
        x_level_0, x_level_1, x_level_2 = x[0], x[1], x[2]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)

        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)

        elif self.level == 2:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1
        )
        w = self.weights_levels(levels_weight_v)
        w = F.softmax(w, dim=1)

        fused_out_reduced = (
            level_0_resized * w[:, :1]
            + level_1_resized * w[:, 1:2]
            + level_2_resized * w[:, 2:]
        )
        # return self.conv(fused_out_reduced)
        out = self.conv(fused_out_reduced)

        # extract_feat = RecursiveFPN(neck_out_channels=self.inter_dim, stage_with_rfp=(True), )
        # rout0 = extract_feat(out)

        rout0 = self.layer(out)

        return rout0

class RecursiveAFPN(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=128,
                 mask_info_flow=True,
                 rfp_steps=2,
                 rfp_sharing=False,
                 stage_with_rfp=(False, True, True, True),

                 ):
        super(RecursiveAFPN, self).__init__()

        self.fp16_enabled = False

        self.conv0 = BasicConv(in_channels[0], in_channels[0] // 8, 1)
        self.conv1 = BasicConv(in_channels[1], in_channels[1] // 8, 1)
        self.conv2 = BasicConv(in_channels[2], in_channels[2] // 8, 1)
        # self.conv3 = BasicConv(in_channels[3], in_channels[3] // 8, 1)

        self.body = nn.Sequential(
            BlockBody([in_channels[0] // 8, in_channels[1] // 8, in_channels[2] // 8])
        )

        self.conv00 = BasicConv(in_channels[0] // 8, out_channels, 1)
        self.conv11 = BasicConv(in_channels[1] // 8, out_channels, 1)
        self.conv22 = BasicConv(in_channels[2] // 8, out_channels, 1)
        # self.conv33 = BasicConv(in_channels[3] // 8, out_channels, 1)
        # self.conv44 = nn.MaxPool2d(kernel_size=1, stride=2)

        self.rfp_steps = rfp_steps
        self.rfp_sharing = rfp_sharing
        self.stage_with_rfp = stage_with_rfp
        # backbone["rfp"] = None
        # backbone["stage_with_rfp"] = stage_with_rfp
        # neck_out_channels = kwargs["neck"]["out_channels"]
        # if rfp_sharing:
        #     backbone["rfp"] = neck_out_channels
        if not self.rfp_sharing:
            # backbone["rfp"] = neck_out_channels
            self.rfp_modules = torch.nn.ModuleList()
            # for rfp_idx in range(1, rfp_steps):
                # rfp_module = builder.build_backbone(backbone)
                # rfp_module.init_weights(kwargs["pretrained"])
                # self.rfp_modules.append(rfp_module)
        self.rfp_aspp = ASPP( out_channels,  out_channels// 4)
        self.rfp_weight = torch.nn.Conv2d(
            out_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.rfp_weight.weight.data.fill_(0)
        self.rfp_weight.bias.data.fill_(0)

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def rfp_forward(self, x, rfp_feats):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            rfp_feat = None
            if self.stage_with_rfp[i]:
                rfp_feat = rfp_feats[i]
            for layer in res_layer:
                x = layer.rfp_forward1(x, rfp_feat)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def rfp_forward1(self, x, rfp_feat):
        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_gen_attention:
                out = self.gen_attention_block(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_gcb:
                out = self.context_block(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        if self.rfp:
            rfp_feat = self.rfp_conv(rfp_feat)
            out = out + rfp_feat

        out = self.relu(out)

        return out

    def extract_feat(self, x):
        # x = self.backbone(img)
        # x = self.neck(x)
        for rfp_idx in range(self.rfp_steps - 1):
            rfp_feats = tuple(self.rfp_aspp(x[i]) if self.stage_with_rfp[i] else x[i]
                              for i in range(len(self.stage_with_rfp)))

            x_idx = self.rfp_forward(x, rfp_feats)
            # if self.rfp_sharing:
            #     x_idx = self.rfp_forward(x, rfp_feats)
            # else:
            #     x_idx = self.rfp_modules[rfp_idx].rfp_forward(x, rfp_feats)
            # x_idx = self.neck(x_idx)
            x_new = []
            for ft_idx in range(len(x_idx)):
                add_weight = torch.sigmoid(self.rfp_weight(x_idx[ft_idx]))
                x_new.append(add_weight * x_idx[ft_idx] + (1 - add_weight) * x[ft_idx])
            x = x_new
        return x

    def forward(self, x):
        x0, x1, x2 = x

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        # x3 = self.conv3(x3)

        out0, out1, out2 = self.body([x0, x1, x2])

        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)

        rout0 = self.extract_feat(out0)
        rout1 = self.extract_feat(out1)
        rout2 = self.extract_feat(out2)

        # return out0, out1, out2

        return rout0, rout1, rout2

