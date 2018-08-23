from __future__ import print_function

import traceback

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, Concatenate, merge, \
    BatchNormalization, Dropout
from keras import backend as K

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


def unet_pool_up_dual_1f1(cfg):
    if cfg.model_filter is None:
        # cfg.model_filter = [64, 96, 128, 192]
        cfg.model_filter = [64, 96, 128, 192, 256]
        # cfg.model_filter = [96, 128, 192, 256, 384]
        # cfg.model_filter = [64, 96, 128, 192, 256, 384]
        # cfg.model_filter = [64, 96, 128, 192, 256, 384, 512]
    if cfg.model_kernel is None or len(cfg.model_kernel) != 1:
        cfg.model_kernel=[3]
    fs = cfg.model_filter
    ks = cfg.model_kernel # 0-conv, 1-resample
    model_act, model_out=cfg.model_act, cfg.model_out
    dim_in, dim_out=cfg.dep_in, cfg.dep_out
    # img_input = Input((None, None, dim_in))  # r,c,3
    locals()['pool0']=Input((cfg.row_in, cfg.col_in, dim_in))  # r,c,3

    for i in range(len(fs)):
        locals()['conv'+str(i)]=Conv2D(fs[i], (ks[0], ks[0]), activation=model_act, padding='same', kernel_initializer=init, name='conv' + str(i))(locals()['pool'+str(i)])
        if i < len(fs)-1:
            locals()['pool' + str(i+1)] = MaxPooling2D((2, 2), strides=(2, 2), name='pool' + str(i+1))(locals()['conv'+str(i)])

    for i in range(len(fs)-2,-1,-1):
        locals()['upsamp'+str(i)]=concatenate([locals()['conv'+str(i)], UpSampling2D(size=(2,2))(locals()['conv'+str(i+1)])],axis=concat_axis)\
            if i==len(fs)-2 else concatenate([locals()['conv'+str(i)], UpSampling2D(size=(2,2))(locals()['decon'+str(i+1)])], axis=concat_axis)
        locals()['decon'+str(i)] = Conv2D(fs[i], (ks[0], ks[0]), activation=model_act, kernel_initializer=init, padding='same', name='decon'+str(i))(locals()['upsamp'+str(i)])
    locals()['out0'] = Conv2D(dim_out, (1, 1), activation=model_out, padding='same',name='out0')(locals()['decon0'])
    return Model(locals()['pool0'], locals()['out0']),  traceback.extract_stack(None, 2)[1].name  + "_" + str(cfg)

def unet_pool_up_dual_2f1(cfg):
    if cfg.model_filter is None:
        # cfg.model_filter = [64, 96, 128, 192]
        # cfg.model_filter = [64, 96, 128, 192, 256]
        # cfg.model_filter = [96, 128, 192, 256, 384]
        # cfg.model_filter = [64, 96, 128, 192, 256, 384]
        cfg.model_filter = [64, 96, 128, 192, 256, 320, 384, 460]
    if cfg.model_kernel is None or len(cfg.model_kernel) != 2:
        cfg.model_kernel=[3,3]
    fs = cfg.model_filter
    ks = cfg.model_kernel # 0-conv, 1-resample
    model_act, model_out=cfg.model_act, cfg.model_out
    dim_in, dim_out=cfg.dep_in, cfg.dep_out
    # img_input = Input((None, None, dim_in))  # r,c,3
    locals()['pool0']=Input((cfg.row_in, cfg.col_in, dim_in))  # r,c,3
    long=len(cfg.model_filter)
    short=int(long/2)

    for lyr, div in [(short,1),(long,1)]:
        for i in range(lyr):
            locals()[str(lyr)+'conv'+str(i)]=Conv2D(int(fs[i]/div), (ks[1], ks[1]), activation=model_act, padding='same', kernel_initializer=init, name=str(lyr)+'conv' + str(i))\
                    (Conv2D(int(fs[i]/div), (ks[0], ks[0]), activation=model_act, padding='same', kernel_initializer=init)(locals()[(str(lyr) if i!=0 else '')+'pool'+str(i)]))
            if i < lyr-1:
                locals()[str(lyr)+'pool' + str(i+1)] = MaxPooling2D((2, 2), strides=(2, 2), name=str(lyr)+'pool' + str(i+1))(locals()[str(lyr)+'conv'+str(i)])

        for i in range(lyr-2,-1,-1):
            locals()[str(lyr)+'upsamp'+str(i)]=concatenate([locals()[str(lyr)+'conv'+str(i)], UpSampling2D(size=(2,2))(locals()[str(lyr)+'conv'+str(i+1)])],axis=concat_axis)\
                if i==lyr-2 else concatenate([locals()[str(lyr)+'conv'+str(i)], UpSampling2D(size=(2,2))(locals()[str(lyr)+'decon'+str(i+1)])], axis=concat_axis)
            locals()[str(lyr)+'decon'+str(i)] = Conv2D(int(fs[i]/div), (ks[0], ks[0]), activation=model_act, kernel_initializer=init, padding='same', name=str(lyr)+'decon'+str(i))(locals()[str(lyr)+'upsamp'+str(i)])


    locals()['out0'] = Conv2D(dim_out, (1, 1), activation=model_out, padding='same', name='out0')\
        (Conv2D(dim_out, (1, 1), activation=model_out, padding='same')(concatenate([locals()[str(short)+'decon0'],locals()[str(long)+'decon0']],axis=concat_axis)))
    return Model(locals()['pool0'], locals()['out0']),  traceback.extract_stack(None, 2)[1].name  + "_" + str(cfg)

def unet_pool_up_dual_2f2(cfg):
    if cfg.model_filter is None:
        # cfg.model_filter = [64, 96, 128, 192]
        cfg.model_filter = [64, 96, 128, 192, 256]
        # cfg.model_filter = [96, 128, 192, 256, 384]
        # cfg.model_filter = [64, 96, 128, 192, 256, 384]
        # cfg.model_filter = [64, 96, 128, 192, 256, 384, 512]
    if cfg.model_kernel is None or len(cfg.model_kernel) != 2:
        cfg.model_kernel=[3,3]
    fs = cfg.model_filter
    ks = cfg.model_kernel # 0-conv, 1-resample
    model_act, model_out=cfg.model_act, cfg.model_out
    dim_in, dim_out=cfg.dep_in, cfg.dep_out
    # img_input = Input((None, None, dim_in))  # r,c,3
    locals()['pool0']=Input((cfg.row_in, cfg.col_in, dim_in))  # r,c,3

    for i in range(len(fs)):
        locals()['conv'+str(i)]=Conv2D(fs[i], (ks[1], ks[1]), activation=model_act, padding='same', kernel_initializer=init, name='conv' + str(i))\
                (Conv2D(fs[i], (ks[0], ks[0]), activation=model_act, padding='same', kernel_initializer=init)(locals()['pool'+str(i)]))
        if i < len(fs)-1:
            locals()['pool' + str(i+1)] = MaxPooling2D((2, 2), strides=(2, 2), name='pool' + str(i+1))(locals()['conv'+str(i)])

    for i in range(len(fs)-2,-1,-1):
        locals()['upsamp'+str(i)]=concatenate([locals()['conv'+str(i)], UpSampling2D(size=(2,2))(locals()['conv'+str(i+1)])],axis=concat_axis)\
            if i==len(fs)-2 else concatenate([locals()['conv'+str(i)], UpSampling2D(size=(2,2))(locals()['decon'+str(i+1)])], axis=concat_axis)
        locals()['decon'+str(i)] = Conv2D(fs[i], (ks[1], ks[1]), activation=model_act, kernel_initializer=init, padding='same', name='decon'+str(i))\
            (Conv2D(fs[i], (ks[0], ks[0]), activation=model_act, kernel_initializer=init, padding='same')(locals()['upsamp'+str(i)]))
    locals()['out0'] = Conv2D(dim_out, (1, 1), activation=model_out, padding='same',name='out0')(locals()['decon0'])
    return Model(locals()['pool0'], locals()['out0']),  traceback.extract_stack(None, 2)[1].name  + "_" + str(cfg)
