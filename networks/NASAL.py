from functools import partial
import json

from .nas_sod_modules import NASALNets

def NASAL(net_config=None):
    assert net_config is not None, "Please input a network config"
    net_config_json = json.load(open(net_config, 'r'))
    net = NASALNets.build_from_config(net_config_json)
    net.init_model(model_init='normal', init_div_groups=False)
    return net

def get_model(net_config):
    return NASAL(net_config)
