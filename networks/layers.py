import torch
import torch.nn as nn
from collections import OrderedDict


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(
        kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


class BasicUnit(nn.Module):

    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


class BasicLayer(BasicUnit):

    def __init__(
            self,
            in_channels,
            out_channels,
            use_bn=True,
            act_func='relu',
            dropout_rate=0,
            ops_order='weight_bn_act'):
        super(BasicLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ add modules """
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                self.bn = nn.BatchNorm2d(in_channels)
            else:
                self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        # activation
        if act_func == 'relu':
            if self.ops_list[0] == 'act':
                self.activation = nn.ReLU(inplace=False)
            else:
                self.activation = nn.ReLU(inplace=True)
        elif act_func == 'relu6':
            if self.ops_list[0] == 'act':
                self.activation = nn.ReLU6(inplace=False)
            else:
                self.activation = nn.ReLU6(inplace=True)
        else:
            self.activation = None
        # dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            self.dropout = None

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def weight_call(self, x):
        raise NotImplementedError

    def forward(self, x):
        for op in self.ops_list:
            if op == 'weight':
                # dropout before weight operation
                if self.dropout is not None:
                    x = self.dropout(x)
                x = self.weight_call(x)
            elif op == 'bn':
                if self.bn is not None:
                    x = self.bn(x)
            elif op == 'act':
                if self.activation is not None:
                    x = self.activation(x)
            else:
                raise ValueError('Unrecognized op: %s' % op)
        return x

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    @staticmethod
    def is_zero_layer():
        return False


class ConvLayer(BasicLayer):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            groups=1,
            bias=False,
            has_shuffle=False,
            use_bn=True,
            act_func='relu',
            dropout_rate=0,
            ops_order='weight_bn_act',
            f_size=None):
        super(
            ConvLayer,
            self).__init__(
            in_channels,
            out_channels,
            use_bn,
            act_func,
            dropout_rate,
            ops_order)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle
        self.f_size = f_size

        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation
        # `kernel_size`, `stride`, `padding`, `dilation` can either be `int` or `tuple` of int

        if self.f_size is not None:
            self.upsample = nn.Upsample(size=self.f_size, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias)

    def weight_call(self, x):
        if self.f_size is not None:
            x = self.upsample(x)
        x = self.conv(x)
        return x

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)


class DepthConvLayer(BasicLayer):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            groups=1,
            bias=False,
            has_shuffle=False,
            use_bn=True,
            act_func='relu',
            dropout_rate=0,
            ops_order='weight_bn_act'):
        super(
            DepthConvLayer,
            self).__init__(
            in_channels,
            out_channels,
            use_bn,
            act_func,
            dropout_rate,
            ops_order)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation
        # `kernel_size`, `stride`, `padding`, `dilation` can either be `int` or `tuple` of int
        self.depth_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=in_channels,
            bias=False)
        self.point_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=self.groups,
            bias=self.bias)

    def weight_call(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

    @staticmethod
    def build_from_config(config):
        return DepthConvLayer(**config)


class IdentityLayer(BasicLayer):

    def __init__(
            self,
            in_channels,
            out_channels,
            use_bn=False,
            act_func=None,
            dropout_rate=0,
            ops_order='weight_bn_act'):
        super(
            IdentityLayer,
            self).__init__(
            in_channels,
            out_channels,
            use_bn,
            act_func,
            dropout_rate,
            ops_order)

    def weight_call(self, x):
        return x

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)


class PPMOutConvLayer(BasicUnit):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, expand_ratio=6, mid_channels=None, f_size=None):
        super(PPMOutConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.f_size = f_size
        if self.f_size is not None:
            self.upsample = nn.Upsample(size=self.f_size, mode='bilinear', align_corners=True)

        self.project = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('act', nn.ReLU6(inplace=False)),
        ]))

        if self.kernel_size > 1:
            if self.mid_channels is None:
                feature_dim = round(self.out_channels * self.expand_ratio)
            else:
                feature_dim = self.mid_channels

            if self.expand_ratio == 1:
                self.inverted_bottleneck = None
            else:
                self.inverted_bottleneck = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(self.out_channels, feature_dim, 1, 1, 0, bias=False)),
                    ('bn', nn.BatchNorm2d(feature_dim)),
                    ('act', nn.ReLU6(inplace=False)),
                ]))

            pad = get_same_padding(self.kernel_size)
            self.depth_conv = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, groups=feature_dim, bias=False)),
                ('bn', nn.BatchNorm2d(feature_dim)),
                ('act', nn.ReLU6(inplace=False)),
            ]))

            self.point_linear = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(feature_dim, self.out_channels, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(self.out_channels)),
            ]))

    def forward(self, x):
        x = self.project(x)
        if self.f_size is not None:
            x = self.upsample(x)
        if self.kernel_size > 1:
            if self.inverted_bottleneck:
                x = self.inverted_bottleneck(x)
            x = self.depth_conv(x)
            x = self.point_linear(x)
        return x

    @staticmethod
    def build_from_config(config):
        return PPMOutConvLayer(**config)

    @staticmethod
    def is_zero_layer():
        return False


class MBInvertedConvLayer(BasicUnit):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            expand_ratio=6,
            mid_channels=None,
            f_size=None):
        super(MBInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels # not used
        self.f_size = f_size
        if self.f_size is not None:
            self.upsample = nn.Upsample(size=self.f_size, mode='bilinear', align_corners=True)

        if self.expand_ratio > 1:
            feature_dim = round(in_channels * self.expand_ratio)
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(feature_dim)),
                ('relu', nn.ReLU6(inplace=True)),
            ]))
        else:
            feature_dim = in_channels
            self.inverted_bottleneck = None

        # depthwise convolution
        pad = get_same_padding(self.kernel_size)
        self.depth_conv = nn.Sequential(
            OrderedDict(
                [
                    ('conv',
                     nn.Conv2d(
                         feature_dim,
                         feature_dim,
                         kernel_size,
                         stride,
                         pad,
                         groups=feature_dim,
                         bias=False)),
                    ('bn',
                     nn.BatchNorm2d(feature_dim)),
                    ('relu',
                     nn.ReLU6(
                         inplace=True)),
                ]))

        # pointwise linear
        self.point_linear = OrderedDict([
            ('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
        ])

        self.point_linear = nn.Sequential(self.point_linear)

    def forward(self, x):
        if self.f_size is not None:
            x = self.upsample(x)
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @staticmethod
    def build_from_config(config):
        return MBInvertedConvLayer(**config)

    @staticmethod
    def is_zero_layer():
        return False


class PPM(BasicUnit):
    def __init__(self, in_channels, out_channels, f_size=None, pool_sizes=(1,3,5)):
        super(PPM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.f_size = f_size
        self.pool_sizes = pool_sizes

        self.down = ConvLayer(
                in_channels, out_channels, kernel_size=1, stride=1, use_bn=True, act_func=None, ops_order='weight_bn_act'
            )
        self.ppms = nn.ModuleList()
        for pool_size in pool_sizes:
            self.ppms.append(nn.Sequential(OrderedDict([
                ('pool', nn.AdaptiveAvgPool2d(pool_size)),
                ('conv', nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(out_channels)),
                ('upsample', nn.Upsample(size=self.f_size, mode='bilinear', align_corners=True)),
            ])))
        self.sum = DepthConvLayer(
                out_channels, out_channels, kernel_size=3, stride=1, 
                use_bn=True, act_func='relu6', ops_order='weight_bn_act'
            )
        self.relu = nn.ReLU6(inplace=False)

    def forward(self, x):
        x = self.down(x)
        res = x 
        for ppm in self.ppms:
            res = torch.add(res, ppm(x))
        res = self.relu(res)
        res = self.sum(res)
        return res

    @staticmethod
    def build_from_config(config):
        return PPM(**config)

    @staticmethod
    def is_zero_layer():
        return False


class ZeroLayer(BasicUnit):

    def __init__(self, in_channels, out_channels, stride, f_size=None):
        super(ZeroLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.f_size = f_size
        if self.f_size is not None:
            self.upsample = nn.Upsample(size=self.f_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        n, c, h, w = x.size()
        h = int(h // self.stride)
        w = int(w // self.stride)
        device = x.get_device() if x.is_cuda else torch.device('cpu')
        # noinspection PyUnresolvedReferences
        padding = torch.zeros(n, self.out_channels, h, w, device=device, requires_grad=False)
        if self.f_size is not None:
            padding = self.upsample(padding)
        x_mean = torch.mean(x, [1,2,3], keepdim=True)
        return padding * x_mean.expand_as(padding)
        # return padding

    @staticmethod
    def build_from_config(config):
        return ZeroLayer(**config)

    @staticmethod
    def is_zero_layer():
        return True