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
from module import uu
from osio import mkdir_ifexist,to_excel_sheet
from postprocess import g_kern_rect,draw_text,smooth_brighten
from mrcnn import utils

# VGG 224x224 # kera-applications
## (/1) b1_c2 , (/2) b1_mp, b2_c2, (/4) b2_mp, b3_c3, (/8) b3_mp, b4_c3, (/16) b4_mp, b5_c3, (/32) b5_mp
##       c1,          p2,     c2,         p3,    c3,         p4,    c4,         p5,     c5,         p6  ## conv,pool
def keras_vgg_backbone(input_image,architecture,**kwargs):
    from c1_vgg import NetU_Vgg
    creator,convs=NetU_Vgg.config[architecture]
    model=creator(input_tensor=input_image,include_top=False,pooling=None,**kwargs)
    from module import dca,ca3,ca33
    from module_complex import c3t2mp, c3t3mp
    return [model.get_layer(name='block{}_pool'.format(bl+1)).output for bl,cv in enumerate(convs)]
    # return [model.get_layer(name='block{}_pool'.format(bl+2)).output for bl,cv in enumerate(convs[1:])]+\
    #        [dca(model.get_layer(name='block{}_pool'.format(convs[-1]+2)).output,2,'dca%d'%(convs[-1]+2),6,512,'relu')]
    # return [model.get_layer(name='block{}_conv{}'.format(bl+2,cv)).output for bl,cv in enumerate(convs[1:])]+\
    #     [ca33(model.get_layer(name='block{}_pool'.format(convs[-1]+2)).output,'f_ca%d'%(convs[-1]+1),6,512,'relu')]

def v16(input_image,**kwargs):
    return keras_vgg_backbone(input_image,'vgg16',**kwargs) # 64x2 128x2 256x3 512x3 512x3
def v19(input_image,**kwargs):
    return keras_vgg_backbone(input_image,'vgg19',**kwargs) # 64x2 128x2 256x4 512x4 512x4

# Resnet Graph 224x224 #
def res50(input_image,**kwargs):
    # from model import resnet_graph
    # return resnet_graph(input_image, "resnet50" ,stage5=True,train_bn=True)
    # return resnet_graph(input_image, "resnet101" ,stage5=True,train_bn=True)
    from c1_resnet import NetU_ResNet
    creator,numbers=NetU_ResNet.config['resnet50']
    model=creator(input_tensor=input_image,include_top=False,pooling=None,**kwargs)
    return [model.get_layer(name='activation_%d'%n).output for n in numbers]

# DenseNet # kera-applications # @Fizyr # https://arxiv.org/pdf/1608.06993v3.pdf
def keras_densenet_backbone(input_image,architecture,**kwargs):
    from c1_dense import NetU_Dense
    creator,blocks=NetU_Dense.config[architecture]
    model=creator(input_tensor=input_image,include_top=False,**kwargs)
    print(model.summary())
    return [model.get_layer(name='conv1/relu').output]+[model.get_layer(name='conv{}_block{}_concat'.format(idx+2,block_num)).output for idx,block_num in enumerate(blocks)]
    # return [model.get_layer(name='conv1/relu').output]+[model.get_layer(name='pool{}_relu'.format(layer)).output for layer in range(2,5)]+[model.get_layer(name='relu').output]

def densenet121(input_image,**kwargs):
    return keras_densenet_backbone(input_image,'densenet121',**kwargs)
def densenet169(input_image,**kwargs):
    return keras_densenet_backbone(input_image,'densenet169',**kwargs)
def densenet201(input_image,**kwargs):
    return keras_densenet_backbone(input_image,'densenet201',**kwargs)

# MobileNet #
def mobile(input_image,**kwargs):
    from keras.applications import mobilenet
    model=mobilenet.MobileNet(input_tensor=input_image,include_top=False,**kwargs)
    return [model.get_layer(name=n).output for n in ['conv_pw_1_relu','conv_pw_3_relu','conv_pw_5_relu','conv_pw_11_relu','conv_pw_13_relu']]

## backbones not extracted correctly  ##
# print(model.summary())
# for layer in model.layers:
#     if layer.__class__.__name__=='Model':
#         print("Model: ",layer.name)
#     elif hasattr(layer,'strides') and layer.strides==(2,2):  # catch down-sample layers
#         print(layer.input,layer.output,sep='->',end='\n')

def mobile2(input_image,**kwargs):
    from keras.applications import mobilenetv2
    model=mobilenetv2.MobileNetV2(input_tensor=input_image,include_top=False,**kwargs)
    return [KL.ZeroPadding2D(((1,0),(1,0)))(model.get_layer(name=n).output) for n in ['Conv1_pad','block_1_pad','block_3_pad','block_6_pad','block_13_pad']]

# Inception Xception # 299x299 hard to match dimension #
def incept3(input_image,**kwargs):
    from keras.applications import inception_v3
    model=inception_v3.InceptionV3(input_tensor=input_image,include_top=False,**kwargs)
    return [KL.ZeroPadding2D(p)(model.get_layer(name=n).output) for p,n in [(((1,0),(1,0)),'activation_3'),(((1,0),(1,0)),'activation_5'),(((1,0),(1,0)),
                                                                               'mixed2'),(((1,1),(1,1)),'mixed7'),(((0,1),(0,1)),'mixed10')]]
def incepres2(input_image,**kwargs):
    from keras.applications import inception_resnet_v2
    model=inception_resnet_v2.InceptionResNetV2(input_tensor=input_image,include_top=False,**kwargs)
    print(model.summary())
    return [KL.ZeroPadding2D(p)(model.get_layer(name=n).output) for p,n in [(((1,1),(1,1)),'conv2d_1'),(((1,2),(1,2)),'max_pooling2d_1'),(((1,2),(2,1)),
                                                                              'max_pooling2d_2'),(((1,1),(1,1)),'max_pooling2d_3'),(((1,1),(1,1)),'max_pooling2d_4')]]

def xcept(input_image,**kwargs):
    from keras.applications import xception
    model=xception.Xception(input_tensor=input_image,include_top=False,**kwargs)
    print(model.summary())
    return model

# NASNet #
def nasmobile(input_image,**kwargs):
    from keras.applications import nasnet
    model=nasnet.NASNetMobile(input_tensor=input_image,include_top=False,**kwargs)
    return model

def naslarge(input_image,**kwargs):
    from keras.applications import nasnet
    model=nasnet.NASNetLarge(input_tensor=input_image,include_top=False,**kwargs)
    return model


