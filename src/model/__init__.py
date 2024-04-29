# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# from logging import getLogger
from torchvision import models
from torch import nn


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.conv5 = conv_block(512, 1028, pool=True)
        self.res3 = nn.Sequential(conv_block(1028, 1028), conv_block(1028, 1028))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(2), 
                                        nn.Flatten(), 
                                        nn.Linear(1028, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out


EMBEDDING_SIZE = {
    "resnet18": 512,
    "resnet50": 2048
}

# logger = getLogger()

def check_model_params(params):
    if hasattr(params, "train_path") and params.train_path == "none":
        params.train_path = ""

    if hasattr(params, "from_ckpt") and params.from_ckpt == "none":
        params.from_ckpt = ""


def build_model(params):
    vision_models = [name for name in dir(models) if name.islower() and not name.startswith("__") and callable(models.__dict__[name])]
    if params.architecture in vision_models:
        model = models.__dict__[params.architecture](num_classes=params.num_classes)
        if params.exp_code.startswith('cifar'):
            if params.architecture.startswith("resnet"):
                model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1), bias=False)
    elif params.architecture == 'resnet9':
        model = ResNet9(num_classes=params.num_classes)
    else:
        assert False, "Architecture not recognized"

    print("Model: {}".format(model))

    return model


def build_baseline_model(params):
    vision_models = [name for name in dir(models) if name.islower() and not name.startswith("__") and callable(models.__dict__[name])]
    if params.architecture in vision_models:
        
        if params.architecture.startswith("resnet"):
            model = models.__dict__[params.architecture](pretrained=True)
            for p in model.parameters():
                p.requires_grad = False

        elif params.architecture.startswith("vgg"):
            model = models.__dict__[params.architecture](pretrained=True)
            for p in model.parameters():
                p.requires_grad = False

        elif params.architecture.startswith("densenet"):
            model = models.__dict__[params.architecture](pretrained=True)
            for p in model.parameters():
                p.requires_grad = False
        else:
            assert False, "Architecture not recognized"
    else:
        assert False, "Architecture not recognized"

    # print("Baseline Model: {}".format(model))

    return model
