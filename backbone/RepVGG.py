# --------------------------------------------------------
# RepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
# Github source: https://github.com/DingXiaoH/RepVGG
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch.nn as nn
import numpy as np
import torch
import copy
# from se_block import SEBlock
import torch.utils.checkpoint as checkpoint


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    # 该模块负责建立一个卷积层和BN层
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    # 该模块用来产生RepVGGBlock，当deploy=False时，产生三个分支，当deploy=True时，产生一个结构重参数化后的卷积和偏置
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        # assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = nn.Identity()
            #   Note that RepVGG-D2se uses SE before nonlinearity. But RepVGGplus models uses SE after nonlinearity.
            # self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            # 当deploy=True时，产生一个结构重参数化后的卷积和偏置
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            # 当deploy=False时，产生三个分支
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)

    def forward(self, inputs):
        # 当结构重参数化时，卷积和偏置之后跟上一个SE模块和非线性激活模块
        if hasattr(self, 'rbr_reparam'):
            return self.se(self.rbr_reparam(inputs))
        # 如果没有线性映射shortcut时，则第三个分支输出为0
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        # 训练时输出为三个分支输出结果相加，再加上SE模块和非线性激活
        return self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        # 用来将三个分支中的卷积算子和BN算子都转化为3x3卷积算子和偏置，然后将3x3卷积核参数相加，偏置相加
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        # 输出将三个分支转化后的的3x3卷积核参数相加，偏置相加
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        # 将第二个分支中的1x1卷积核padding为3x3的卷积核
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        # 将BN层的算子转化为卷积核的乘积和偏置
        if branch is None:
            return 0, 0
        # 当输入的分支是序列时，记录该分支的卷积核参数、BN的均值、方差、gamma、beta和eps（一个非常小的数）
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        # 当输入是第三个分支只有BN层时，添加一个只进行线性映射的3x3卷积核和一个偏置
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        # 输出新的卷积核参数（kernel * t），新的偏置（beta - running_mean * gamma / std）
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        # 该模块用来进行结构重参数化，输出由三个分支重参数化后的只含有主分支的block
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        # 用self.__delattr__删除掉之前的旧的三个分支
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True



def repvgg_model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model