#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : inference.py
# @Date : 2022-11-18-12-51
# @Project : aidant_retinal-disease
# @Author : seungmin

import os, yaml, itertools
import torch
import numpy as np
from PIL import Image

import torchvision.transforms as T

import matplotlib.pyplot as plt
import scikitplot as skplt
from utils.dataloader_multi import MyInferDatasetWrapper
from utils.cam import GradCAM
from utils.visualize import visualize, reverse_normalize

from model import *


## load model
def _get_model(base_model):
    model_dict = {"resnet": ResNet,
                  "mobilenet": MobileNet}

    try:
        model = model_dict[base_model]
        return model
    except:
        raise ("Invalid model name. Pass one of the model dictionary.")


## main
def main(model_name, model_dir, image):
    # 학습시에 yaml 파일과 모델을 이 폴더로부터 복사하여 저장함. 가장 최신 파일.

    checkpoints_folder = os.path.join('./weights/experiments/', str(model_name) + '_checkpoints')
    timestamp = model_dir
    checkpoint = torch.load(os.path.join(checkpoints_folder, timestamp, 'model.pt'))

    config = yaml.load(open(os.path.join(checkpoints_folder, timestamp, 'resnet.yaml'), "r"), Loader=yaml.FullLoader)
    device = config['inference_device']
    print(device)

    ## get class names
    testset = MyInferDatasetWrapper(batch_size=1, num_workers=1, test_path=image)

    ## model load
    # model topology
    model = _get_model(model_name)
    model = model(**config['model'])

    model.load_state_dict(checkpoint['net'])
    model = model.to(device)
    model.eval()

    ## test loader
    test_loader = testset.get_test_loaders()

    target_layer = model.features[-2][1].conv2
    print(target_layer)

    wrapped_model = GradCAM(model, target_layer)

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess = T.Compose([T.ToTensor(),
                            normalize
                            ])
    ## test loader
    test_loader = testset.get_test_loaders()

    # myclass = [int(i) for i in range(config['model']['classes'])]
    myclass = ['Normal', 'AMD', 'DR', 'Glaucoma']
    # with torch.no_grad():
    for data in test_loader:
        images, labels = data
        tensor = images.to(device)
        # calculate outputs by running images through the network

        cam, idx, prob, prob_softmax = wrapped_model(tensor)

        print(prob_softmax.cpu())  # 4개의 class에 대한 확률값

        heatmap = visualize(images, cam)
        heatmap = heatmap.squeeze(0).permute(1, 2, 0)

        print(f'Original : {myclass[labels]}\n Predicted: {myclass[idx]}')

        '''
        heatmap = Image.fromarray(heatmap)
        heatmap.save('./heatmap_cam.png')

        '''
        state = {'label': myclass[labels],
                 'pred': myclass[idx],
                 'prob': prob,  # max 확률 값과 해당 label 값 출력
                 'heatmap_tensor': heatmap}
        # torch.save(state, './heatmap_result.pt')
        print('=' * 10)
        print(state)

    return prob_softmax.cpu(), state


if __name__ == "__main__":
    base_model = 'resnet'
    timestamp = '2022-10-28_22:21:43'  # 모델 directory nm
    image = './new_data/DR_IRB20210592_2_039449.1.2.dcm_DR_57.0Y_M_22073628_1300060335.dcm'
    main(base_model, timestamp, image)