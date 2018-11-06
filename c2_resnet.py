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
from osio import mkdir_ifexist,to_excel_sheet
from postprocess import g_kern_rect,draw_text,smooth_brighten
from mrcnn import utils
from backbone import resnet_50, resnet_101, resnet_152

class MRCNN_ResNet_50(BaseNetM):
    def __init__(self,**kwargs):
        super(MRCNN_ResNet_50,self).__init__(convolution_backbone=resnet_50,**kwargs)

class MRCNN_ResNet_101(BaseNetM):
    def __init__(self,**kwargs):
        super(MRCNN_ResNet_101,self).__init__(convolution_backbone=resnet_101,**kwargs)

class MRCNN_ResNet_152(BaseNetM):
    def __init__(self,**kwargs):
        super(MRCNN_ResNet_152,self).__init__(convolution_backbone=resnet_152,**kwargs)

