import math

from .layers import *


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        IdentityLayer.__name__: IdentityLayer,
        PPMOutConvLayer.__name__: PPMOutConvLayer,
        MBInvertedConvLayer.__name__: MBInvertedConvLayer,
        PPM.__name__: PPM,
        ZeroLayer.__name__: ZeroLayer,
        MobileInvertedResidualBlock.__name__: MobileInvertedResidualBlock,
    }

    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


class MobileInvertedResidualBlock(BasicUnit):
    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv.is_zero_layer():
            res = self.shortcut(x)
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.mobile_inverted_conv(x)
        else:
            conv_x = self.mobile_inverted_conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(
            config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)


class NASALNets(BasicUnit):
    def __init__(self, first_conv, down_blocks, down_flags, ppm, ppm_out, bridges, up_blocks, up_flags, predictor):
        super(NASALNets, self).__init__()

        self.first_conv = first_conv
        self.down_blocks = nn.ModuleList(down_blocks)
        self.down_flags = down_flags
        self.ppm = ppm
        self.ppm_out = nn.ModuleList(ppm_out)
        self.bridges = nn.ModuleList(bridges)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.up_flags = up_flags
        self.predictor = predictor
        self.relu = nn.ReLU6(inplace=False)

    def forward(self, x, depth=None):
        x_size = x.size()
        if depth is not None:
            x = torch.cat((x, depth), dim=1)
        x = self.first_conv(x)
        down_features = []
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            if i in self.down_flags:
                down_features.append(x)
        down_features.reverse()
        x = self.ppm(x)
        ppm_feature = x
        stage_cnt = 0
        for i, block in enumerate(self.up_blocks):
            x = block(x)
    
            if i in self.up_flags:
                assert x.size(3) == down_features[stage_cnt].size(3) or x.size(3) * 2 == down_features[stage_cnt].size(3), \
                '{} should be the same or half size of {} at stage {}.'.format(x.size(3), down_features[stage_cnt].size(3), stage_cnt)
                if x.size(3) != down_features[stage_cnt].size(3):
                    x = torch.nn.functional.interpolate(x, down_features[stage_cnt].size()[2:], mode='bilinear', align_corners=True)
                x = x + self.bridges[stage_cnt](down_features[stage_cnt])

                x = x + self.ppm_out[stage_cnt](ppm_feature)

                x = self.relu(x)
                stage_cnt += 1 
        x = self.predictor(x)
        x = torch.nn.functional.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)

        return x

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        bridges = []
        for bridge_config in config['bridges']:
            bridges.append(set_layer_from_config(bridge_config))
        predictor = set_layer_from_config(config['predictor'])
        down_blocks, up_blocks = [], []
        for block_config in config['down_blocks']:
            down_blocks.append(set_layer_from_config(block_config))
        down_flags = config['down_flags']
        ppm = set_layer_from_config(config['ppm'])
        ppm_out = []
        for block_config in config['ppm_out']:
            ppm_out.append(set_layer_from_config(block_config))
        for block_config in config['up_blocks']:
            up_blocks.append(set_layer_from_config(block_config))
        up_flags = config['up_flags']

        return NASALNets(first_conv, down_blocks, down_flags, ppm, ppm_out, bridges, up_blocks, up_flags, predictor)

    def init_model(self, model_init, init_div_groups=True):
        print('Init model Conv2d with {}......'.format(model_init))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'normal':
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()