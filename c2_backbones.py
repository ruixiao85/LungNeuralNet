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
from keras.applications import resnet50, densenet, vgg16, vgg19
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
    model=creator(input_tensor=input_image,include_top=False,pooling=None,)
    return [model.get_layer(name='block{}_pool'.format(idx)).output for idx in blocks]

def vgg_16(input_image):
    return keras_vgg_backbone(input_image,'vgg16') # 64x2 128x2 256x3 512x3 512x3
def vgg_19(input_image):
    return keras_vgg_backbone(input_image,'vgg19') # 64x2 128x2 256x4 512x4 512x4

class MRCNN_Vgg16(BaseNetM):
    def __init__(self,**kwargs):
        super(MRCNN_Vgg16,self).__init__(backbone=vgg_16,**kwargs)

class MRCNN_Vgg19(BaseNetM):
    def __init__(self,**kwargs):
        super(MRCNN_Vgg19,self).__init__(backbone=vgg_19,**kwargs)


# Resnet Graph #

def resnet_50(input_image):
    model=resnet50.ResNet50(input_tensor=input_image,include_top=False)
    return [model.get_layer(name=name).output for name in ['max_pooling2d_1','activation_10','activation_22','activation_40','activation_49']]

class MRCNN_Res50(BaseNetM):
    def __init__(self,**kwargs):
        super(MRCNN_Res50,self).__init__(backbone=resnet_50,**kwargs)



# DenseNet # kera-applications # @Fizyr # https://arxiv.org/pdf/1608.06993v3.pdf

def keras_densenet_backbone(input_image,architecture):
    allowed_backbones={
        'densenet121':([6,12,24,16],densenet.DenseNet121),
        'densenet169':([6,12,32,32],densenet.DenseNet169),
        'densenet201':([6,12,48,32],densenet.DenseNet201),
    }
    blocks,creator=allowed_backbones[architecture]
    model=creator(input_tensor=input_image,include_top=False)
    return [model.get_layer(name='pool1')]+\
           [model.get_layer(name='conv{}_block{}_concat'.format(idx+2,block_num)).output for idx,block_num in enumerate(blocks)]

def densenet_121(input_image):
    return keras_densenet_backbone(input_image,'densenet121')
def densenet_169(input_image):
    return keras_densenet_backbone(input_image,'densenet169')
def densenet_201(input_image):
    return keras_densenet_backbone(input_image,'densenet201')

class MRCNN_Dense121(BaseNetM):
    def __init__(self,**kwargs):
        super(MRCNN_Dense121,self).__init__(backbone=densenet_121,**kwargs)

class MRCNN_Dense169(BaseNetM):
    def __init__(self,**kwargs):
        super(MRCNN_Dense169,self).__init__(backbone=densenet_169,**kwargs)
