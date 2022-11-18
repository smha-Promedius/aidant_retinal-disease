#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : model
# @Date : 2022-11-18-15-07
# @Project : aidant_retinal-disease
# @Author : seungmin

import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, model_arch, sub_model, classes, pretrain):
        super(ResNet, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=pretrain),
                            "resnet34": models.resnet34(pretrained=pretrain),
                            "resnet50": models.resnet50(pretrained=pretrain),
                            "resnet101": models.resnet101(pretrained=pretrain),
                            "resnet152": models.resnet152(pretrained=pretrain),
                            "resnext50_32x4d": models.resnext50_32x4d(pretrained=pretrain),
                            "resnext101_32x8d": models.resnext101_32x8d(pretrained=pretrain),
                            "wide_resnet50_2": models.wide_resnet50_2(pretrained=pretrain),
                            "wide_resnet101_2": models.wide_resnet101_2(pretrained=pretrain)}

        print("Model architecture:", model_arch)
        resnet = self._get_submodel(sub_model)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.linear = nn.Linear(resnet.fc.in_features, classes)

    def _get_submodel(self, feature_extractor):
        try:
            model = self.resnet_dict[feature_extractor]
            print("Feature extractor:", feature_extractor)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet~, resnext~ or wide_resnet~.")

    def forward(self, x):
        h = self.features(x)
        h = self.linear(h.squeeze())
        return h


class MobileNet(nn.Module):
    def __init__(self, model_arch, sub_model, classes, pretrain):
        super(MobileNet, self).__init__()
        self.mobilenet_dict = {'mobilenet_v2' : models.mobilenet_v2(pretrained=pretrain),
                               'mobilenet_v3_small' : models.mobilenet_v3_small(pretrained=pretrain),
                               'mobilenet_v3_large' : models.mobilenet_v3_large(pretrained=pretrain)}

        print("Model architecture:", model_arch)
        self.mobilenet = self._get_submodel(sub_model)
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.classifier[1].in_features, classes)

    def _get_submodel(self, feature_extractor):
        try:
            model = self.mobilenet_dict[feature_extractor]
            print("Feature extractor:", feature_extractor)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet~, resnext~ or wide_resnet~.")

    def forward(self, x):
        h = self.mobilenet(x)
        return h