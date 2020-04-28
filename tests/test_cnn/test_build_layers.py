import pytest
import torch
import torch.nn as nn

from mmcv.cnn.bricks import (ACTIVATION_LAYERS, CONV_LAYERS, NORM_LAYERS,
                             PADDING_LAYERS, build_activation_layer,
                             build_conv_layer, build_norm_layer,
                             build_padding_layer, build_upsample_layer)
from mmcv.cnn.bricks.norm import infer_abbr
from mmcv.cnn.bricks.upsample import PixelShufflePack


def test_build_conv_layer():
    with pytest.raises(TypeError):
        # cfg must be a dict
        cfg = 'Conv2d'
        build_conv_layer(cfg)

    with pytest.raises(KeyError):
        # `type` must be in cfg
        cfg = dict(kernel_size=3)
        build_conv_layer(cfg)

    with pytest.raises(KeyError):
        # unsupported conv type
        cfg = dict(type='FancyConv')
        build_conv_layer(cfg)

    kwargs = dict(
        in_channels=4, out_channels=8, kernel_size=3, groups=2, dilation=2)
    cfg = None
    layer = build_conv_layer(cfg, **kwargs)
    assert isinstance(layer, nn.Conv2d)
    assert layer.in_channels == kwargs['in_channels']
    assert layer.out_channels == kwargs['out_channels']
    assert layer.kernel_size == (kwargs['kernel_size'], kwargs['kernel_size'])
    assert layer.groups == kwargs['groups']
    assert layer.dilation == (kwargs['dilation'], kwargs['dilation'])

    cfg = dict(type='Conv')
    layer = build_conv_layer(cfg, **kwargs)
    assert isinstance(layer, nn.Conv2d)
    assert layer.in_channels == kwargs['in_channels']
    assert layer.out_channels == kwargs['out_channels']
    assert layer.kernel_size == (kwargs['kernel_size'], kwargs['kernel_size'])
    assert layer.groups == kwargs['groups']
    assert layer.dilation == (kwargs['dilation'], kwargs['dilation'])

    for type_name, module in CONV_LAYERS.module_dict.items():
        cfg = dict(type=type_name)
        layer = build_conv_layer(cfg, **kwargs)
        assert isinstance(layer, module)
        assert layer.in_channels == kwargs['in_channels']
        assert layer.out_channels == kwargs['out_channels']


def test_infer_abbr():
    with pytest.raises(TypeError):
        # class_type must be a class
        infer_abbr(0)

    class MyNorm:

        abbr = 'mn'

    assert infer_abbr(MyNorm) == 'mn'

    class FancyBatchNorm:
        pass

    assert infer_abbr(FancyBatchNorm) == 'bn'

    class FancyInstanceNorm:
        pass

    assert infer_abbr(FancyInstanceNorm) == 'in'

    class FancyLayerNorm:
        pass

    assert infer_abbr(FancyLayerNorm) == 'ln'

    class FancyGroupNorm:
        pass

    assert infer_abbr(FancyGroupNorm) == 'gn'

    class FancyNorm:
        pass

    assert infer_abbr(FancyNorm) == 'norm'


def test_build_norm_layer():
    with pytest.raises(TypeError):
        # cfg must be a dict
        cfg = 'BN'
        build_norm_layer(cfg, 3)

    with pytest.raises(KeyError):
        # `type` must be in cfg
        cfg = dict()
        build_norm_layer(cfg, 3)

    with pytest.raises(KeyError):
        # unsupported norm type
        cfg = dict(type='FancyNorm')
        build_norm_layer(cfg, 3)

    with pytest.raises(AssertionError):
        # postfix must be int or str
        cfg = dict(type='BN')
        build_norm_layer(cfg, 3, postfix=[1, 2])

    with pytest.raises(AssertionError):
        # `num_groups` must be in cfg when using 'GN'
        cfg = dict(type='GN')
        build_norm_layer(cfg, 3)

    # test each type of norm layer in norm_cfg
    abbr_mapping = {
        'BN': 'bn',
        'BN1d': 'bn',
        'BN2d': 'bn',
        'BN3d': 'bn',
        'SyncBN': 'bn',
        'GN': 'gn',
        'LN': 'ln',
        'IN': 'in',
        'IN1d': 'in',
        'IN2d': 'in',
        'IN3d': 'in',
    }
    for type_name, module in NORM_LAYERS.module_dict.items():
        for postfix in ['_test', 1]:
            cfg = dict(type=type_name)
            if type_name == 'GN':
                cfg['num_groups'] = 2
            name, layer = build_norm_layer(cfg, 3, postfix=postfix)
            assert name == abbr_mapping[type_name] + str(postfix)
            assert isinstance(layer, module)
            if type_name == 'GN':
                assert layer.num_channels == 3
                assert layer.num_groups == cfg['num_groups']
            elif type_name != 'LN':
                assert layer.num_features == 3


def test_build_activation_layer():
    with pytest.raises(TypeError):
        # cfg must be a dict
        cfg = 'ReLU'
        build_activation_layer(cfg)

    with pytest.raises(KeyError):
        # `type` must be in cfg
        cfg = dict()
        build_activation_layer(cfg)

    with pytest.raises(KeyError):
        # unsupported activation type
        cfg = dict(type='FancyReLU')
        build_activation_layer(cfg)

    # test each type of activation layer in activation_cfg
    for type_name, module in ACTIVATION_LAYERS.module_dict.items():
        cfg['type'] = type_name
        layer = build_activation_layer(cfg)
        assert isinstance(layer, module)


def test_build_padding_layer():
    with pytest.raises(TypeError):
        # cfg must be a dict
        cfg = 'reflect'
        build_padding_layer(cfg)

    with pytest.raises(KeyError):
        # `type` must be in cfg
        cfg = dict()
        build_padding_layer(cfg)

    with pytest.raises(KeyError):
        # unsupported activation type
        cfg = dict(type='FancyPad')
        build_padding_layer(cfg)

    for type_name, module in PADDING_LAYERS.module_dict.items():
        cfg['type'] = type_name
        layer = build_padding_layer(cfg, 2)
        assert isinstance(layer, module)

    input_x = torch.randn(1, 2, 5, 5)
    cfg = dict(type='reflect')
    padding_layer = build_padding_layer(cfg, 2)
    res = padding_layer(input_x)
    assert res.shape == (1, 2, 9, 9)


def test_upsample_layer():
    with pytest.raises(TypeError):
        # cfg must be a dict
        cfg = 'bilinear'
        build_upsample_layer(cfg)

    with pytest.raises(KeyError):
        # `type` must be in cfg
        cfg = dict()
        build_upsample_layer(cfg)

    with pytest.raises(KeyError):
        # unsupported activation type
        cfg = dict(type='FancyUpsample')
        build_upsample_layer(cfg)

    for type_name in ['nearest', 'bilinear']:
        cfg['type'] = type_name
        layer = build_upsample_layer(cfg)
        assert isinstance(layer, nn.Upsample)
        assert layer.mode == type_name

    cfg = dict(
        type='deconv', in_channels=3, out_channels=3, kernel_size=3, stride=2)
    layer = build_upsample_layer(cfg)
    assert isinstance(layer, nn.ConvTranspose2d)

    cfg = dict(
        type='pixel_shuffle',
        in_channels=3,
        out_channels=3,
        scale_factor=2,
        upsample_kernel=3)
    layer = build_upsample_layer(cfg)

    assert isinstance(layer, PixelShufflePack)
    assert layer.scale_factor == 2
    assert layer.upsample_kernel == 3


def test_pixel_shuffle_pack():
    x_in = torch.rand(2, 3, 10, 10)
    pixel_shuffle = PixelShufflePack(3, 3, scale_factor=2, upsample_kernel=3)
    assert pixel_shuffle.upsample_conv.kernel_size == (3, 3)
    x_out = pixel_shuffle(x_in)
    assert x_out.shape == (2, 3, 20, 20)
