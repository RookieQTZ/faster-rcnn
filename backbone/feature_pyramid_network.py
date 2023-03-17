from collections import OrderedDict

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from torch.jit.annotations import Tuple, List, Dict


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()

        # 遍历模型子模块按顺序存入有序字典
        # 只保存layer4及其之前的结构，舍去之后不用的结构
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        # 依次遍历模型的所有子模块，并进行正向传播，
        # 收集layer1, layer2, layer3, layer4的输出
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.
    The feature maps are currently supposed to be in increasing depth
    order.
    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.
    Arguments:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names
    """

    def __init__(self, in_channels_list, out_channels,
                 channel_attention=None, spatial_attention=None,
                 extra_blocks=None):
        super().__init__()
        # 用来调整resnet特征矩阵(layer1,2,3,4)的channel（kernel_size=1）
        self.inner_blocks = nn.ModuleList()
        # fixme： inner2
        self.inner_blocks2 = nn.ModuleList()
        self.inner_top_downs = nn.ModuleList()
        # 对调整后的特征矩阵使用3x3的卷积核来得到对应的预测特征矩阵
        self.layer_blocks = nn.ModuleList()
        for i, in_channels in enumerate(in_channels_list):
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            inner_block_module2 = nn.Conv2d(out_channels, out_channels, 1)
            # 三个下采样模块
            if i < len(in_channels_list) - 1:
                inner_top_down_module = nn.Conv2d(out_channels, out_channels, 2, stride=2)
                # inner_top_down_module = nn.MaxPool2d(kernel_size=2)
                # inner_top_down_module = nn.AvgPool2d(kernel_size=2)
                self.inner_top_downs.append(inner_top_down_module)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.inner_blocks2.append(inner_block_module2)
            self.layer_blocks.append(layer_block_module)

        if channel_attention is None:
            self.channel_blocks = nn.ModuleList()
            for in_channels in in_channels_list:
                if in_channels == 0:
                    continue
                channel_block_module = ChannelAttention(in_channels)
                self.channel_blocks.append(channel_block_module)

        if spatial_attention is None:
            self.spatial_attention = SpatialAttention()

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_inner_blocks2(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks2)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks2:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_top_down_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_top_downs)
        if idx < 0:
            idx += num_blocks
        i = 1
        out = x
        for module in self.inner_top_downs:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_channel_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.channel_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.channel_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_channel_result(self, x):
        # unpack OrderedDict into two lists for easier handling
        # names = list(x.keys())
        # x = list(x.values())
        #
        # # result中保存着每个预测特征层
        results = []

        for idx in range(len(x) - 1, -1, -1):
            channel_result = self.get_result_from_channel_blocks(x[idx], idx)
            results.insert(0, channel_result)

        # make it back an OrderedDict
        # out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return results

    def get_attention_result(self, x):
        # unpack OrderedDict into two lists for easier handling
        # names = list(x.keys())
        # x = list(x.values())

        # result中保存着每个预测特征层
        results = []

        for idx in range(len(x) - 1, -1, -1):
            results.insert(0, self.spatial_attention(x[idx]))

        # make it back an OrderedDict
        # out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return results

    def forward(self, x: Dict[str, Tensor], cbam: bool, double_fusion: bool) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.
        Arguments:
            x (OrderedDict[Tensor]): feature maps for each feature level.
            cbam
            double_fusion
        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        # fixme: 注意力机制
        if cbam:
            # 此时，x为resnet50 layer1 layer2 layer3 layer4的输出
            # 通道注意力机制
            x = self.get_channel_result(x)

            # 空间注意力机制
            x = self.get_attention_result(x)

        # 将resnet layer4的channel调整到指定的out_channels
        # last_inner = self.inner_blocks[-1](x[-1])
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        # result中保存着每个预测特征层
        results = []
        # 将layer4调整channel后的特征矩阵，通过3x3卷积后得到对应的预测特征矩阵
        # results.append(self.layer_blocks[-1](last_inner))
        if double_fusion:
            results.append(last_inner)
        else:
            results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_down_top = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_down_top
            if double_fusion:
                results.insert(0, last_inner)
            else:
                results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        # fixme: 注意力机制

        if double_fusion:
            # 残差下采样fpn -> 3*3卷积
            # results2 中保存着每个预测特征层
            results2 = []
            last_inner = results[0]
            results2.append(self.get_result_from_layer_blocks(last_inner, 0))
            for idx in range(1, len(results)):
                inner_lateral = self.get_result_from_inner_blocks2(results[idx], idx)
                # 下采样
                inner_top_down = self.get_result_from_top_down_inner_blocks(last_inner, idx)
                last_inner = inner_lateral + inner_top_down + results[idx]
                results2.append(self.get_result_from_layer_blocks(last_inner, idx))

            # 在layer4对应的预测特征层基础上生成预测特征矩阵5
            if self.extra_blocks is not None:
                results, names = self.extra_blocks(results2, x, names)
        else:
            # 在layer4对应的预测特征层基础上生成预测特征矩阵5
            if self.extra_blocks is not None:
                results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxPool(torch.nn.Module):
    """
    Applies a max_pool2d on top of the last feature map
    """

    def forward(self, x: List[Tensor], y: List[Tensor], names: List[str]) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))  # input, kernel_size, stride, padding
        return x, names


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        extra_blocks: ExtraFPNBlock
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self,
                 backbone: nn.Module,
                 return_layers=None,
                 in_channels_list=None,
                 out_channels=256,
                 extra_blocks=None,
                 re_getter=True,
                 ):
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        if re_getter is True:
            assert return_layers is not None
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        else:
            self.body = backbone

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )

        self.out_channels = out_channels

    def forward(self, x, cbam, double_fusion):
        # body: orderDict
        x = self.body(x)

        x = self.fpn(x, cbam, double_fusion)
        return x


class UlPoolings(nn.Module):
    '''
    3, h, w
    conv(3, 1, size=1, strides=1)
    1, h, w
    pooling(size=k, strides=s)
    '''
    def __init__(self, h, w):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((h, w))
        self.pool2 = nn.AdaptiveAvgPool2d((h // 2, w // 2))
        self.pool3 = nn.AdaptiveAvgPool2d((h // 4, w // 4))
        self.pool4 = nn.AdaptiveAvgPool2d((h // 8, w // 8))

    def forward(self, x):
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.pool4(x)

        return x


# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out) + x  # 残差


# 空间注意力机制，先通道注意力机制、后空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1):
        '''
        :param x1: origin feature maps
        :param x2: enhanced images (etc. ul images)
        :return:
        '''
        avg_out = torch.mean(x1, dim=1, keepdim=True)
        max_out, _ = torch.max(x1, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        out = self.sigmoid(x) * x1
        return out + x1  # 残差
