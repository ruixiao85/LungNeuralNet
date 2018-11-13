import math
import os
import cv2
import datetime
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from keras.engine.saving import model_from_json,load_model

from a_config import Config
from b2_net_multi import BaseNetM
from image_set import PatchSet
from keras.applications import resnet50, densenet, vgg16, vgg19, inception_v3, inception_resnet_v2, mobilenet, mobilenetv2, nasnet, xception
from osio import mkdir_ifexist,to_excel_sheet
from postprocess import g_kern_rect,draw_text,smooth_brighten
from mrcnn import utils


# VGG # kera-applications

def keras_vgg_backbone(input_image,architecture):
    allowed_backbones={
        'vgg16':([1,2,3,4,5],vgg16.VGG16),
        'vgg19':([1,2,3,4,5],vgg19.VGG19),
    }
    blocks,creator=allowed_backbones[architecture]
    model=creator(input_tensor=input_image,include_top=False,pooling=None)
    return [model.get_layer(name='block{}_pool'.format(idx)).output for idx in blocks]

def v16(input_image):
    return keras_vgg_backbone(input_image,'vgg16') # 64x2 128x2 256x3 512x3 512x3
def v19(input_image):
    return keras_vgg_backbone(input_image,'vgg19') # 64x2 128x2 256x4 512x4 512x4


# Resnet Graph #

def res50(input_image):
    model=resnet50.ResNet50(input_tensor=input_image,include_top=False)
    return [model.get_layer(name=n).output for n in ['activation_1','activation_10','activation_22','activation_40','activation_49']]


# DenseNet # kera-applications # @Fizyr # https://arxiv.org/pdf/1608.06993v3.pdf

def keras_densenet_backbone(input_image,architecture):
    allowed_backbones={
        'densenet121':([6,12,24,16],densenet.DenseNet121),
        'densenet169':([6,12,32,32],densenet.DenseNet169),
        'densenet201':([6,12,48,32],densenet.DenseNet201),
    }
    blocks,creator=allowed_backbones[architecture]
    model=creator(input_tensor=input_image,include_top=False)
    return [model.get_layer(name='conv1/relu').output]+\
           [model.get_layer(name='conv{}_block{}_concat'.format(idx+2,block_num)).output for idx,block_num in enumerate(blocks)]

def densenet121(input_image):
    return keras_densenet_backbone(input_image,'densenet121')
def densenet169(input_image):
    return keras_densenet_backbone(input_image,'densenet169')
def densenet201(input_image):
    return keras_densenet_backbone(input_image,'densenet201')

