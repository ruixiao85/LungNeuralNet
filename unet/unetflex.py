from __future__ import print_function

import traceback

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, Concatenate, merge, \
    BatchNormalization, Dropout, Activation, Conv2DTranspose, Add
from keras import backend as K

from model_config import ModelConfig

K.set_image_data_format('channels_last')
concat_axis = 3
init='he_normal'

def get_crop_shape(target, refer):
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value  # width, the 3rd dimension
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value  # height, the 2nd dimension
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)
    return (ch1, ch2), (cw1, cw2)

def conv3(in_layer, name=None, fs=None, act=None):
    return Conv2D(fs, (3,3), activation=act, padding='same', kernel_initializer=init, name=name+'_0')(in_layer)

def conv3n(in_layer, name=None, fs=None, act=None):
    return Activation(activation=act, name=name+'_0norm')(BatchNormalization()(Conv2D(fs, (3,3), padding='same', kernel_initializer=init)(in_layer)))

def conv33(in_layer, name=None, fs=None, act=None):
    x = Conv2D(fs, (3, 3), activation=act, padding='same', kernel_initializer=init, name=name + '_0')(in_layer)
    x = Conv2D(fs, (3, 3), activation=act, padding='same', kernel_initializer=init, name=name + '_1')(x)
    return x

def conv3n3n(in_layer, name=None, fs=None, act=None):
    x=Activation(activation=act, name=name+'_0norm')(BatchNormalization()(Conv2D(fs, (3,3),  padding='same', kernel_initializer=init)(in_layer)))
    x=Activation(activation=act, name=name+'_1norm')(BatchNormalization()(Conv2D(fs, (3,3),  padding='same', kernel_initializer=init)(x)))
    return x

def convres131(in_layer, name=None, fs=None, act=None): # [64, 64, 256] or 2X, 3X,...
    strides=(1,1) # same dimension
    x = Activation(activation=act,name=name + '_0n')(BatchNormalization()(Conv2D(fs, (1, 1), strides=strides)(in_layer)))
    x = Activation(activation=act,name=name + '_1n')(BatchNormalization()(Conv2D(fs, (3, 3))(x)))
    x = Activation(activation=act,name=name + '_2nm')(
        Add()([ BatchNormalization()(Conv2D(fs*4, (1, 1))(x)), in_layer])  # direct add
    )
    return x

def d2convres131(in_layer, name=None, fs=None, act=None): # [64, 64, 256] or 2X, 3X,...
    strides=(2,2) # half size
    x = Activation(activation=act,name=name + '_0n')(BatchNormalization()(Conv2D(fs, (1, 1), strides=strides)(in_layer)))
    # x = Activation(activation=act,name=name + '_0n')(BatchNormalization()(Conv2D(fs, (3, 3), strides=strides)(in_layer)))
    x = Activation(activation=act,name=name + '_1n')(BatchNormalization()(Conv2D(fs, (3, 3))(x)))
    x = Activation(activation=act,name=name + '_2nm')(
        Add()([ BatchNormalization()(Conv2D(fs*4, (1, 1))(x)),
                BatchNormalization()(Conv2D(fs*4, (1, 1), strides=strides)(in_layer))])  # shortcut with conv
    )
    return x

def d2conv3(in_layer, name=None, fs=None, act=None):
    return Conv2D(fs, (3,3), strides=(2, 2), activation=act, padding='same', kernel_initializer=init, name=name+'_0')(in_layer)

def d2conv3n(in_layer, name=None, fs=None, act=None):
    return Activation(activation=act, name=name+'_0norm')(BatchNormalization()(Conv2D(fs, (3,3), strides=(2, 2), padding='same', kernel_initializer=init)(in_layer)))

def d2maxpool2(in_layer, name=None, fs=None, act=None):
    return MaxPooling2D((2, 2), strides=(2, 2), name=name)(in_layer)


def d3conv3(in_layer, name=None, fs=None, act=None):
    return Conv2D(fs, (3,3), strides=(3, 3), activation=act, padding='same', kernel_initializer=init, name=name+'_0')(in_layer)

def d3conv3n(in_layer, name=None, fs=None, act=None):
    return Activation(activation=act, name=name+'_0norm')(BatchNormalization()(Conv2D(fs, (3,3), strides=(3, 3), padding='same', kernel_initializer=init)(in_layer)))

def d3maxpool3(in_layer, name=None, fs=None, act=None):
    return MaxPooling2D((3, 3), strides=(3, 3), name=name)(in_layer)


def u2convres131(input_skip, input_up, name=None, fs=None, act=None): # [64, 64, 256] or 2X, 3X,...
    strides=(2,2) # half size
    # x = Activation(activation=act,name=name + '_0n')(BatchNormalization()(Conv2D(fs, (1, 1), strides=strides)(in_layer)))
    x = Activation(activation=act,name=name + '_0n')(BatchNormalization()(Conv2DTranspose(fs, (3, 3), strides=strides)(input_up)))
    x = Activation(activation=act,name=name + '_1n')(BatchNormalization()(Conv2D(fs, (3, 3))(x)))
    x = Activation(activation=act,name=name + '_2nm')(
        Add()([ BatchNormalization()(Conv2D(fs*4, (1, 1))(x)),
                # BatchNormalization()(Conv2DTranspose(fs*4, (1, 1), strides=strides)(in_layer))])  # shortcut with conv
                BatchNormalization()(Conv2DTranspose(fs*4, (3, 3), strides=strides)(input_up))])  # shortcut with conv
    )
    return concatenate([input_skip, x],name=name+'_3c',axis=concat_axis)

def u2mergeup2(input_skip, input_up, name=None, fs=None, act=None):
    x=UpSampling2D(size=(2, 2))(input_up)
    return concatenate([input_skip,x],name=name, axis=concat_axis)

def u2mergetrans3(input_skip, input_up, name=None, fs=None, act=None):
    x=Conv2DTranspose(fs,(3, 3),activation=act,kernel_initializer=init, strides=(2, 2),padding='same')(input_up)
    return concatenate([input_skip, x],name=name, axis=concat_axis)

def u2mergetrans3n(input_skip, input_up, name=None, fs=None, act=None):
    x=Activation(activation=act)(BatchNormalization()(Conv2DTranspose(fs, (3, 3), kernel_initializer=init, strides=(2, 2), padding='same')(input_up)))
    return concatenate([input_skip, x],name=name, axis=concat_axis)

def u3mergeup3(input_skip, input_up, name=None, fs=None, act=None):
    x=UpSampling2D(size=(3, 3))(input_up)
    return concatenate([input_skip,x],name=name, axis=concat_axis)

def u3mergetrans3(input_skip, input_up, name=None, fs=None, act=None):
    x=Conv2DTranspose(fs,(3, 3),activation=act,kernel_initializer=init, strides=(3, 3),padding='same')(input_up)
    return concatenate([input_skip, x],name=name, axis=concat_axis)

def u3mergetrans3n(input_skip, input_up, name=None, fs=None, act=None):
    x=Activation(activation=act)(BatchNormalization()(Conv2DTranspose(fs, (3, 3), kernel_initializer=init, strides=(3, 3), padding='same')(input_up)))
    return concatenate([input_skip, x],name=name, axis=concat_axis)



def unet1(cfg):
    fs = cfg.model_filter
    # img_input = Input((None, None, dim_in))  # r,c,3
    locals()['pool0']=Input((cfg.row_in, cfg.col_in, cfg.dep_in))  # r,c,3

    for i in range(len(fs)):
        locals()['conv'+str(i)]=cfg.model_downconv(locals()['pool'+str(i)], 'conv' + str(i), fs[i], cfg.model_act)
        if i < len(fs)-1:
            locals()['pool' + str(i+1)] =cfg.model_downsamp(locals()['conv'+str(i)], 'pool' + str(i+1), fs[i], cfg.model_act)

    for i in range(len(fs)-2,-1,-1):
        locals()['upsamp'+str(i)]=cfg.model_upsamp(locals()['conv' + str(i)],
               locals()['conv' + str(i + 1)] if i==len(fs)-2 else locals()['decon' + str(i + 1)],
               'upsamp'+str(i), fs[i], cfg.model_act)
        locals()['decon'+str(i)] = cfg.model_upconv(locals()['upsamp'+str(i)], 'decon' + str(i), fs[i], cfg.model_act)

    locals()['out0'] = Conv2D(cfg.dep_out, (1, 1), activation=cfg.model_out, padding='same', name='out0')(locals()['decon0'])
    return Model(locals()['pool0'], locals()['out0'])


def unet2(cfg:ModelConfig):
    fs = cfg.model_filter
    # img_input = Input((None, None, dim_in))  # r,c,3
    locals()['pool0']=Input((cfg.row_in, cfg.col_in, cfg.dep_in))  # r,c,3
    long=len(cfg.model_filter)
    short=int(long/2)

    for lyr, div in [(short,1),(long,2)]:
        for i in range(lyr):
            locals()[str(lyr)+'conv'+str(i)]=cfg.model_downconv(locals()[(str(lyr) if i!=0 else '')+'pool'+str(i)], str(lyr)+'conv' + str(i), int(fs[i]/div), cfg.model_act)
            if i < lyr-1:
                locals()[str(lyr)+'pool' + str(i+1)] =cfg.model_downsamp(locals()[str(lyr)+'conv'+str(i)], str(lyr)+'pool' + str(i+1), fs[i], cfg.model_act)

        for i in range(lyr-2,-1,-1):
            locals()[str(lyr)+'upsamp'+str(i)]=cfg.model_upsamp(locals()[str(lyr)+str(len(fs)) + 'conv' + str(i)],
               locals()[str(lyr)+str(len(fs)) + 'conv' + str(i + 1)] if i==len(fs)-2 else locals()[str(lyr)+ 'decon' + str(i + 1)],
               str(lyr)+'upsamp'+str(i),fs[i],cfg.model_act)
            locals()[str(lyr)+'decon'+str(i)] = cfg.model_upconv(locals()[str(lyr)+'upsamp'+str(i)], str(lyr) + 'decon' + str(i),  int(fs[i]/div), cfg.model_act)


    locals()['out0'] = Conv2D(cfg.dep_out, (1, 1), activation=cfg.model_out, padding='same', name='out0')\
        (concatenate([locals()[str(short)+'decon0'],locals()[str(long)+'decon0']],axis=concat_axis))
    return Model(locals()['pool0'], locals()['out0'])
