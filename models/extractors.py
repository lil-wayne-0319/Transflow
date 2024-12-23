from .resnet import *
from .mae import *


def build_extractor_single_branch(c):
    if c.extractor == 'resnet18':
        extractor = resnet18(pretrained=True, progress=True)  # llw 24/11/22 改为随机初始化
    elif c.extractor == 'resnet34':
        extractor = resnet34(pretrained=True, progress=True)
    elif c.extractor == 'resnet50':
        extractor = resnet50(pretrained=True, progress=True)
    elif c.extractor == 'resnext50_32x4d':
        extractor = resnext50_32x4d(pretrained=True, progress=True)
    elif c.extractor == 'wide_resnet50_2':
        # llw pretrained->False, use random init
        extractor = wide_resnet50_2(pretrained=True, progress=True)
    elif c.extractor == 'wide_resnet50_2_single_branch':
        extractor = wide_resnet50_2_single_branch(pretrained=True, progress=True)
    elif c.extractor == 'deit_base_distilled_patch16_384':
        extractor = deit_base_distilled_patch16_384(pretrained=True)

    output_channels = []
    if 'wide' in c.extractor:
        for i in range(3):
            output_channels.append(eval('extractor.layer{}[-1].conv3.out_channels'.format(i + 1)))  # 256,512,1024
    else:
        for i in range(3):
            output_channels.append(extractor.eval('layer{}'.format(i + 1))[-1].conv2.out_channels)

    print("Channels of extracted features:", output_channels)
    return extractor, output_channels


def build_extractor(c):
    if   c.extractor == 'resnet18':
        extractor = resnet18(pretrained=True, progress=True)    # llw 24/11/22 改为随机初始化
    elif c.extractor == 'resnet34':
        extractor = resnet34(pretrained=True, progress=True)
    elif c.extractor == 'resnet50':
        extractor = resnet50(pretrained=True, progress=True)
    elif c.extractor == 'resnext50_32x4d':
        extractor = resnext50_32x4d(pretrained=True, progress=True)
    elif c.extractor == 'wide_resnet50_2':
        # llw pretrained->False, use random init
        extractor = wide_resnet50_2(pretrained=True, progress=True)
    elif c.extractor == 'mae_visualize_vit_base':
        extractor = mae_visualize_vit_base(c.chkpt_dir, pretrained=True)

    output_channels = []
    if 'wide' in c.extractor:
        for i in range(3):
            output_channels.append(eval('extractor.layer{}[-1].conv3.out_channels'.format(i+1)))  # 256,512,1024
    elif 'mae' in c.extractor:
        pass
    else:
        for i in range(3):
            output_channels.append(extractor.eval('layer{}'.format(i+1))[-1].conv2.out_channels)
            
    print("Channels of extracted features:", output_channels)
    return extractor, output_channels