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
    return Conv2D(fs, (3,3), activation=act, padding='same', kernel_initializer=init, name=name)(in_layer)
def conv31(in_layer, name=None, fs=None, act=None):
    x=Conv2D(fs*2, (3,3), activation=act, padding='same', kernel_initializer=init, name=name+'_1')(in_layer)
    x=Conv2D(fs, (1,1), activation=act, padding='same', kernel_initializer=init, name=name)(x)
    return x

def conv3n(in_layer, name=None, fs=None, act=None):
    return Activation(activation=act, name=name+'_0norm')(BatchNormalization()(Conv2D(fs, (3,3), padding='same', kernel_initializer=init)(in_layer)))

def conv33(in_layer, name=None, fs=None, act=None):
    x = Conv2D(fs, (3, 3), activation=act, padding='same', kernel_initializer=init, name=name + '_1')(in_layer)
    x = Conv2D(fs, (3, 3), activation=act, padding='same', kernel_initializer=init, name=name)(x)
    return x
def conv331(in_layer, name=None, fs=None, act=None):
    x = Conv2D(fs*2, (3, 3), activation=act, padding='same', kernel_initializer=init, name=name + '_2')(in_layer)
    x = Conv2D(fs*2, (3, 3), activation=act, padding='same', kernel_initializer=init, name=name+'_1')(x)
    x = Conv2D(fs, (1, 1), activation=act, padding='same', kernel_initializer=init, name=name)(x)
    return x

def conv3n3n(in_layer, name=None, fs=None, act=None):
    x=Activation(activation=act, name=name+'_1norm')(BatchNormalization()(Conv2D(fs, (3,3),  padding='same', kernel_initializer=init)(in_layer)))
    x=Activation(activation=act, name=name)(BatchNormalization()(Conv2D(fs, (3,3),  padding='same', kernel_initializer=init)(x)))
    return x

#https://arxiv.org/pdf/1709.00201.pdf
def conv32deepres(in_layer, name=None, fs=None, act=None):
    x = Conv2D(fs, (3, 3), activation=act, padding='same', kernel_initializer=init, name=name + '_2')(in_layer)
    x = Conv2D(fs/2, (2, 2), activation=act, padding='same', kernel_initializer=init, name=name + '_1')(x)
    x = concatenate([in_layer, x],name=name)
    return x
def conv33deepres(in_layer, name=None, fs=None, act=None):
    x = Conv2D(fs, (3, 3), activation=act, padding='same', kernel_initializer=init, name=name + '_2')(in_layer)
    x = Conv2D(fs/2, (3, 3), activation=act, padding='same', kernel_initializer=init, name=name + '_1')(x)
    x = concatenate([in_layer, x],name=name)
    return x


def dxconv(in_layer, rate, name=None, fs=None, act=None):
    kern=rate if rate%2==1 else rate+1
    return Conv2D(fs, (kern,kern), strides=(rate, rate), activation=act, padding='same', kernel_initializer=init, name=name)(in_layer)

def dxconvn(in_layer, rate, name=None, fs=None, act=None):
    kern = rate if rate % 2 == 1 else rate + 1
    return Activation(activation=act, name=name)(BatchNormalization()(Conv2D(fs, (kern,kern), strides=(rate,rate), padding='same', kernel_initializer=init)(in_layer)))

def dxmaxpool(in_layer, rate, name=None, fs=None, act=None):
    return MaxPooling2D((rate,rate), strides=(rate, rate), name=name)(in_layer)

def uxmergeup(input_skip, input_up, rate, name=None, fs=None, act=None):
    x=UpSampling2D(size=(rate,rate))(input_up)
    return concatenate([input_skip,x],name=name, axis=concat_axis)

def uxmergetrans(input_skip, input_up, rate, name=None, fs=None, act=None):
    kern = rate if rate % 2 == 1 else rate + 1
    x=Conv2DTranspose(fs,(kern,kern), strides=(rate,rate),activation=act,kernel_initializer=init,padding='same')(input_up)
    return concatenate([input_skip, x],name=name, axis=concat_axis)

def uxmergetransn(input_skip, input_up, rate, name=None, fs=None, act=None):
    kern = rate if rate % 2 == 1 else rate + 1
    x=Activation(activation=act)(BatchNormalization()(Conv2DTranspose(fs, (kern, kern), strides=(rate, rate), kernel_initializer=init, padding='same')(input_up)))
    return concatenate([input_skip, x],name=name, axis=concat_axis)


def conv131res(in_layer, name, filters, act, convs=None, strides=None):
    filters=filters if isinstance(filters,list) else [int(filters/2),int(filters/2),filters]  if isinstance(filters,int) else [32,32,64]
    convs=convs or [(1,1),(3,3),(1,1)]; strides=strides or (1,1)
    x = Conv2D(filters[0], convs[0], strides=strides, padding='same',activation=act, name=name + '_2n')(in_layer)
    x = Conv2D(filters[1], convs[1], padding='same',activation=act, name=name + '_1n')(x)
    x = Activation(activation=act, name=name)(
        Add()([Conv2D(filters[2], convs[2], padding='same')(x),
               Conv2D(filters[2], convs[2], strides=strides, padding='same')(in_layer)])  # shortcut with conv
    )
    return x
def conv131nres(in_layer, name, filters, act, convs=None, strides=None):
    filters=filters if isinstance(filters,list) else [int(filters/2),int(filters/2),filters]  if isinstance(filters,int) else [32,32,64]
    convs=convs or [(1,1),(3,3),(1,1)]; strides=strides or (1,1)
    x = Activation(activation=act, name=name + '_2n')(BatchNormalization()(Conv2D(filters[0], convs[0], strides=strides, padding='same')(in_layer)))
    x = Activation(activation=act, name=name + '_1n')(BatchNormalization()(Conv2D(filters[1], convs[1], padding='same')(x)))
    x = Activation(activation=act, name=name)(
        Add()([BatchNormalization()(Conv2D(filters[2], convs[2], padding='same')(x)),
               BatchNormalization()(Conv2D(filters[2], convs[2], strides=strides, padding='same')(in_layer))])  # shortcut with conv
    )
    return x
def conv131resx2(in_layer, name, filters, act):
    x, n=in_layer,2
    for i in range(n-1,-1,-1):
        x=conv131res(x, name+str(i)[:i], filters, act)
    return x
def conv131nresx2(in_layer, name, filters, act):
    x, n=in_layer,2
    for i in range(n-1,-1,-1):
        x=conv131nres(x, name+str(i)[:i], filters, act)
    return x
def dxconv131res(in_layer, rate, name=None, fs=None, act=None): # [64, 64, 256] or 2X, 3X,...
    return conv131res(in_layer, name, fs, act, strides=(rate,rate))
def dxconv131nres(in_layer, rate, name=None, fs=None, act=None): # [64, 64, 256] or 2X, 3X,...
    return conv131nres(in_layer, name, fs, act, strides=(rate,rate))

def trans131res(input_skip, input_up, name, filters, act, convs=None, strides=None):
    filters=filters if isinstance(filters,list) else [int(filters/2),int(filters/2),filters]  if isinstance(filters,int) else [32,32,64]
    convs=convs or [(1,1),(3,3),(1,1)]; strides=strides or (1,1)
    x = Conv2DTranspose(filters[0], convs[0], strides=strides, padding='same',activation=act, name=name + '_2n')(input_up)
    x = Conv2D(filters[1], convs[1], padding='same',activation=act, name=name + '_1n')(x)
    x = Activation(activation=act, name=name)(
        Add()([Conv2D(filters[2], convs[2], padding='same')(x), input_skip])
    )
    return x
def trans131nres(input_skip, input_up, name, filters, act, convs=None, strides=None):
    filters=filters if isinstance(filters,list) else [int(filters/2),int(filters/2),filters]  if isinstance(filters,int) else [32,32,64]
    convs=convs or [(1,1),(3,3),(1,1)]; strides=strides or (1,1)
    x = Activation(activation=act, name=name + '_2n')(BatchNormalization()(Conv2DTranspose(filters[0], convs[0], strides=strides, padding='same')(input_up)))
    x = Activation(activation=act, name=name + '_1n')(BatchNormalization()(Conv2D(filters[1], convs[1], padding='same')(x)))
    x = Activation(activation=act, name=name)(
        Add()([BatchNormalization()(Conv2D(filters[2], convs[2], padding='same')(x)), input_skip])
    )
    return x
def uxtrans131res(input_skip, input_up, rate, name=None, fs=None, act=None): # [64, 64, 256] or 2X, 3X,...
    return trans131res(input_skip, input_up, name, fs, act, strides=(rate,rate))
def uxtrans131nres(input_skip, input_up, rate, name=None, fs=None, act=None): # [64, 64, 256] or 2X, 3X,...
    return trans131nres(input_skip, input_up, name, fs, act, strides=(rate,rate))

def unet1s(cfg:ModelConfig):
    fs = cfg.model_filter
    ps = cfg.model_poolsize
    # img_input = Input((None, None, dim_in))  # r,c,3
    locals()['pool0']=Input((cfg.row_in, cfg.col_in, cfg.dep_in))  # r,c,3

    for i in range(len(fs)):
        locals()['conv'+str(i)]=cfg.model_downconv(locals()['pool'+str(i)], 'conv' + str(i), fs[i], cfg.model_act)
        if i < len(fs)-1:
            locals()['pool' + str(i+1)] =cfg.model_downsamp(locals()['conv'+str(i)],  ps[i], 'pool' + str(i+1), fs[i], cfg.model_act)

    for i in range(len(fs)-2,-1,-1):
        locals()['upsamp'+str(i)]=cfg.model_upsamp(locals()['conv' + str(i)],
               locals()['conv' + str(i + 1)] if i==len(fs)-2 else locals()['decon' + str(i + 1)],
               ps[i], 'upsamp'+str(i), fs[i], cfg.model_act)
        locals()['decon'+str(i)] = cfg.model_upconv(locals()['upsamp'+str(i)], 'decon' + str(i), fs[i], cfg.model_act)

    locals()['out0'] = Conv2D(cfg.dep_out, (1, 1), activation=cfg.model_out, padding='same', name='out0')(locals()['decon0'])
    return Model(locals()['pool0'], locals()['out0'])


def unet1d(cfg:ModelConfig):
    fs = cfg.model_filter
    ps = cfg.model_poolsize
    # img_input = Input((None, None, dim_in))  # r,c,3
    locals()['pool0']=Input((cfg.row_in, cfg.col_in, cfg.dep_in))  # r,c,3

    for i in range(len(fs)):
        locals()['conv'+str(i)]=cfg.model_downconv(locals()['pool'+str(i)], 'conv' + str(i), fs[i], cfg.model_act)
        if i < len(fs)-1:
            locals()['pool' + str(i+1)] =cfg.model_downsamp(locals()['conv'+str(i)], ps[i], 'pool' + str(i+1), fs[i], cfg.model_act)

    for i in range(len(fs)-2,-1,-1):
        locals()['upsamp'+str(i)]=cfg.model_upsamp(locals()['conv' + str(i)],
               concatenate([locals()['pool'+str(i+1)], locals()['conv' + str(i + 1)]]) if i==len(fs)-2 else\
               concatenate([locals()['pool' + str(i + 1)], locals()['decon' + str(i + 1)]]),
                ps[i], 'upsamp'+str(i), fs[i], cfg.model_act)
        locals()['decon'+str(i)] = cfg.model_upconv(locals()['upsamp'+str(i)], 'decon' + str(i), fs[i], cfg.model_act)

    locals()['out0'] = Conv2D(cfg.dep_out, (1, 1), activation=cfg.model_out, padding='same', name='out0')(locals()['decon0'])
    return Model(locals()['pool0'], locals()['out0'])

def unet2s(cfg:ModelConfig):
    fs = cfg.model_filter
    ps = cfg.model_poolsize
    # img_input = Input((None, None, dim_in))  # r,c,3
    locals()['pool0']=Input((cfg.row_in, cfg.col_in, cfg.dep_in))  # r,c,3
    long=len(cfg.model_filter)
    short=int(long/2)

    for lyr, div in [(short,1),(long,2)]:
        for i in range(lyr):
            locals()[str(lyr)+'conv'+str(i)]=cfg.model_downconv(locals()[(str(lyr) if i!=0 else '')+'pool'+str(i)], str(lyr)+'conv' + str(i), int(fs[i]/div), cfg.model_act)
            if i < lyr-1:
                locals()[str(lyr)+'pool' + str(i+1)] =cfg.model_downsamp(locals()[str(lyr)+'conv'+str(i)], ps[i], str(lyr)+'pool' + str(i+1), fs[i], cfg.model_act)

        for i in range(lyr-2,-1,-1):
            locals()[str(lyr)+'upsamp'+str(i)]=cfg.model_upsamp(locals()[str(lyr)+str(len(fs)) + 'conv' + str(i)],
               locals()[str(lyr)+str(len(fs)) + 'conv' + str(i + 1)] if i==len(fs)-2 else locals()[str(lyr)+ 'decon' + str(i + 1)],
                ps[i], str(lyr)+'upsamp'+str(i),fs[i],cfg.model_act)

            locals()[str(lyr)+'decon'+str(i)] = cfg.model_upconv(locals()[str(lyr)+'upsamp'+str(i)], str(lyr) + 'decon' + str(i),  int(fs[i]/div), cfg.model_act)


    locals()['out0'] = Conv2D(cfg.dep_out, (1, 1), activation=cfg.model_out, padding='same', name='out0')\
        (concatenate([locals()[str(short)+'decon0'],locals()[str(long)+'decon0']],axis=concat_axis))
    return Model(locals()['pool0'], locals()['out0'])
