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
from c0_backbones import v16,v19,res50,densenet121,densenet169,densenet201
from image_set import PatchSet
from osio import mkdir_ifexist,to_excel_sheet
from postprocess import g_kern_rect,draw_text,smooth_brighten
from mrcnn import utils


class MRCNN_Vgg16(BaseNetM):
    def __init__(self,**kwargs):
        super(MRCNN_Vgg16,self).__init__(backbone=v16,**kwargs)

class MRCNN_Vgg19(BaseNetM):
    def __init__(self,**kwargs):
        super(MRCNN_Vgg19,self).__init__(backbone=v19,**kwargs)


class MRCNN_Res50(BaseNetM):
    def __init__(self,**kwargs):
        super(MRCNN_Res50,self).__init__(backbone=res50,**kwargs)

class MRCNN_Dense121(BaseNetM):
    def __init__(self,**kwargs):
        super(MRCNN_Dense121,self).__init__(backbone=densenet121,**kwargs)

class MRCNN_Dense169(BaseNetM):
    def __init__(self,**kwargs):
        super(MRCNN_Dense169,self).__init__(backbone=densenet169, **kwargs)
