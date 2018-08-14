import traceback

from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose, Cropping2D, ZeroPadding2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input
from keras import backend as K

K.set_image_data_format("channels_last")
#K.set_image_dim_ordering("th")
concat_axis = 3
init='he_normal'

def unet_conv_trans_1f1(cfg):
    if cfg.filter_size is None:
        # cfg.filter_size = [64, 96, 128, 192]
        cfg.filter_size = [64, 96, 128, 192, 256]
        # cfg.filter_size = [96, 128, 192, 256, 384]
        # cfg.filter_size = [64, 96, 128, 192, 256, 384]
        # cfg.filter_size = [64, 96, 128, 192, 256, 384, 512]
    if cfg.kernel_size is None or len(cfg.kernel_size) != 1:
        cfg.kernel_size=[3]
    fs = cfg.filter_size
    ks = cfg.kernel_size # 0-conv, 1-resample
    act_fun, out_fun=cfg.act_fun, cfg.out_fun
    dim_in, dim_out=cfg.dep_in, cfg.dep_out
    # img_input = Input((None, None, dim_in))  # r,c,3
    locals()['pool0']=Input((cfg.row_in, cfg.col_in, dim_in))  # r,c,3

    for i in range(len(fs)-1):
        locals()['pool'+str(i+1)]=Conv2D(fs[i], (ks[0], ks[0]), strides=(2,2),activation=act_fun, padding='same', kernel_initializer=init, name='pool'+str(i+1))(locals()['pool'+str(i)])

    for i in range(len(fs)-2,-1,-1):
        locals()['upsamp'+str(i)]=concatenate([locals()['pool'+str(i)], Conv2DTranspose(fs[i], (3, 3),
            activation=act_fun, kernel_initializer=init, strides=(2, 2),padding='same',name='upsamp'+str(i))(locals()['pool'+str(i+1)])],
          axis=concat_axis) if i==len(fs)-2 else concatenate([locals()['pool'+str(i)], Conv2DTranspose(fs[i], (3, 3),
           activation=act_fun, kernel_initializer=init, strides=(2, 2), padding='same',name='upsamp'+str(i))(locals()['decon'+str(i+1)])], axis=concat_axis)
        locals()['decon'+str(i)] = Conv2D(fs[i], (ks[0], ks[0]), activation=act_fun, kernel_initializer=init, padding='same', name='decon'+str(i))(locals()['upsamp'+str(i)])
    locals()['out0'] = Conv2D(dim_out, (1, 1), activation=out_fun, padding='same',name='out0')(locals()['decon0'])
    return Model(locals()['pool0'], locals()['out0']), traceback.extract_stack(None, 2)[1].name + "_" + str(cfg)

def unet_conv_trans_2f1(cfg):
    if cfg.filter_size is None:
        # cfg.filter_size = [64, 96, 128, 192]
        cfg.filter_size = [64, 96, 128, 192, 256]
        # cfg.filter_size = [96, 128, 192, 256, 384]
        # cfg.filter_size = [64, 96, 128, 192, 256, 384]
        # cfg.filter_size = [64, 96, 128, 192, 256, 384, 512]
    if cfg.kernel_size is None or len(cfg.kernel_size) != 2:
        cfg.kernel_size=[3,3]
    fs = cfg.filter_size
    ks = cfg.kernel_size # 0-conv, 1-resample
    act_fun, out_fun=cfg.act_fun, cfg.out_fun
    dim_in, dim_out=cfg.dep_in, cfg.dep_out
    # img_input = Input((None, None, dim_in))  # r,c,3
    locals()['pool0']=Input((cfg.row_in, cfg.col_in, dim_in))  # r,c,3

    for i in range(len(fs)):
        locals()['conv'+str(i)]=Conv2D(fs[i], (ks[0], ks[0]), activation=act_fun, padding='same', kernel_initializer=init, name='conv'+str(i))(locals()['pool'+str(i)])
        if i < len(fs)-1:
            locals()['pool' + str(i+1)] = Conv2D(fs[i], (ks[1], ks[1]), activation=act_fun, strides=(2, 2), padding='same', kernel_initializer=init, name='pool' + str(i+1))(locals()['conv'+str(i)])

    for i in range(len(fs)-2,-1,-1):
        locals()['upsamp'+str(i)]=concatenate([locals()['conv'+str(i)], Conv2DTranspose(fs[i], (3, 3),
            activation=act_fun, kernel_initializer=init, strides=(2, 2),padding='same',name='upsamp'+str(i))(locals()['conv'+str(i+1)])],
          axis=concat_axis) if i==len(fs)-2 else concatenate([locals()['conv'+str(i)], Conv2DTranspose(fs[i], (3, 3),
           activation=act_fun, kernel_initializer=init, strides=(2, 2), padding='same',name='upsamp'+str(i))(locals()['decon'+str(i+1)])], axis=concat_axis)
        locals()['decon'+str(i)] = Conv2D(fs[i], (ks[0], ks[0]), activation=act_fun, kernel_initializer=init, padding='same', name='decon'+str(i))(locals()['upsamp'+str(i)])
    locals()['out0'] = Conv2D(dim_out, (1, 1), activation=out_fun, padding='same',name='out0')(locals()['decon0'])
    return Model(locals()['pool0'], locals()['out0']), traceback.extract_stack(None, 2)[1].name  + "_" + str(cfg)


def unet_conv_trans_2f2(cfg):
    if cfg.filter_size is None:
        # cfg.filter_size = [64, 96, 128, 192]
        cfg.filter_size = [64, 96, 128, 192, 256]
        # cfg.filter_size = [96, 128, 192, 256, 384]
        # cfg.filter_size = [64, 96, 128, 192, 256, 384]
        # cfg.filter_size = [64, 96, 128, 192, 256, 384, 512]
    if cfg.kernel_size is None or len(cfg.kernel_size) != 2:
        cfg.kernel_size=[3,3]
    fs = cfg.filter_size
    ks = cfg.kernel_size # 0-conv, 1-resample
    act_fun, out_fun=cfg.act_fun, cfg.out_fun
    dim_in, dim_out=cfg.dep_in, cfg.dep_out
    # img_input = Input((None, None, dim_in))  # r,c,3
    locals()['pool0']=Input((cfg.row_in, cfg.col_in, dim_in))  # r,c,3

    for i in range(len(fs)):
        locals()['conv'+str(i)]=Conv2D(fs[i], (ks[0], ks[0]), activation=act_fun, padding='same', kernel_initializer=init, name='conv'+str(i))(locals()['pool'+str(i)])
        if i < len(fs)-1:
            locals()['pool' + str(i+1)] = Conv2D(fs[i], (ks[1], ks[1]), activation=act_fun, strides=(2, 2), padding='same', kernel_initializer=init, name='pool' + str(i+1))(locals()['conv'+str(i)])

    for i in range(len(fs)-2,-1,-1):
        locals()['upsamp'+str(i)]=concatenate([locals()['conv'+str(i)], Conv2DTranspose(fs[i], (3, 3),
            activation=act_fun, kernel_initializer=init, strides=(2, 2),padding='same',name='upsamp'+str(i))(locals()['conv'+str(i+1)])],
          axis=concat_axis) if i==len(fs)-2 else concatenate([locals()['conv'+str(i)], Conv2DTranspose(fs[i], (3, 3),
           activation=act_fun, kernel_initializer=init, strides=(2, 2), padding='same',name='upsamp'+str(i))(locals()['decon'+str(i+1)])], axis=concat_axis)
        locals()['decon'+str(i)] = Conv2D(fs[i], (ks[1], ks[1]), activation=act_fun, kernel_initializer=init, padding='same', name='decon'+str(i))\
                                (Conv2D(fs[i], (ks[0], ks[0]), activation=act_fun, kernel_initializer=init, padding='same')(locals()['upsamp'+str(i)]))
    locals()['out0'] = Conv2D(dim_out, (1, 1), activation=out_fun, padding='same',name='out0')(locals()['decon0'])
    return Model(locals()['pool0'], locals()['out0']), traceback.extract_stack(None, 2)[1].name  + "_" + str(cfg)
