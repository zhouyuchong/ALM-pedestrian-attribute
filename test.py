'''
Author: zhouyuchong
Date: 2024-03-07 16:32:52
Description: 
LastEditors: zhouyuchong
LastEditTime: 2024-03-08 11:18:14
'''
import argparse
import os
import shutil
import time
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import model as models
import onnx
from onnxsim import simplify


import cv2

from utils.datasets import *

parser = argparse.ArgumentParser(description='Pedestrian Attribute Framework')
parser.add_argument('--approach', default='inception_iccv', type=str, help='(default=%(default)s)')
parser.add_argument('--img', default="human.png", type=str, required=True, help='path to image')
parser.add_argument('--weight', type=str, default=None, required=True, help='.weight config')
parser.add_argument('--type', type=str, default=None, required=True, help='.dataset type')
parser.add_argument('--conf', type=bool, default=False, help='display confidence or not')
# TODO SPN is not support in TensorRT
parser.add_argument('--onnx', type=bool, default=False, help='export onnx')


def main():
    args = parser.parse_args()
    print('=' * 100)
    print('Arguments = ')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)

    device = torch.device("cuda")
    # create model
    model = models.__dict__[args.approach](pretrained=False, num_classes=attr_nums[args.type])
    model.to(device)

    checkpoint = torch.load(args.weight)
    # single dgpu
    new_state_dict = {'.'.join(k.split('.')[1:]): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(new_state_dict)
    model.eval()    

    # preprocess
    ori_img = cv2.imread(args.img)
    res_img = cv2.resize(ori_img, (256, 128), interpolation = cv2.INTER_LINEAR) 
    img = res_img.reshape(1, 256, 128, 3)
    img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    img = img.to(device).float() / 255.0

    # 导出onnx模型
    if args.onnx:
        dynamic_axes = {'input' : {0 : 'batch_size'},
                'output' : {0 : 'batch_size'}}
        torch.onnx.export(model,                     # model being run
                          img,                       # model input (or a tuple for multiple inputs)
                          "./FastestDet.onnx",       # where to save the model (can be a file or file-like object)
                          export_params=True,        # store the trained parameter weights inside the model file
                          opset_version=11,          # the ONNX version to export the model to
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes=dynamic_axes)  # whether to execute constant folding for optimization
        # onnx-sim
        onnx_model = onnx.load("./FastestDet.onnx")  # load onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        print("onnx sim sucess...")
        onnx.save(model_simp, "./FastestDet.onnx")   

    # 模型推理
    start = time.perf_counter()
    output = model(img)
    end = time.perf_counter()
    usage = (end - start) * 1000.
    print("forward time:%fms"%usage)

    if type(output) == type(()) or type(output) == type([]):
        output = torch.max(torch.max(torch.max(output[0],output[1]),output[2]),output[3])

    if args.conf:
        output = output.detach().cpu().numpy()
        label = np.where(output > 0.5, 1, 0)
        conf = output[0]
        for index in range(label[0].size):
            if label[0][index] == 1:
                print(description[args.type][index], conf[index])
    else:
        output = torch.sigmoid(output.data).cpu().numpy()
        output = np.where(output > 0.5, 1, 0)
        for index in range(output[0].size):
            if output[0][index] == 1:
                print(description[args.type][index])


if __name__=="__main__":
    main()

