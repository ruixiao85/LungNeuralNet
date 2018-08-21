import traceback

from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose, Cropping2D, ZeroPadding2D
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input
from keras import backend as K

K.set_image_data_format("channels_last")
#K.set_image_dim_ordering("th")
concat_axis = 3
init='he_normal'

def unet_pool_trans_1f1(cfg):
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
    act_fun, out_fun=cfg.model_act, cfg.model_out
    dim_in, dim_out=cfg.dep_in, cfg.dep_out
    # img_input = Input((None, None, dim_in))  # r,c,3
    locals()['pool0']=Input((cfg.row_in, cfg.col_in, dim_in))  # r,c,3

    for i in range(len(fs)):
        locals()['conv'+str(i)]=Conv2D(fs[i], (ks[0], ks[0]), activation=act_fun, padding='same', kernel_initializer=init, name='conv' + str(i))(locals()['pool'+str(i)])
        if i < len(fs)-1:
            locals()['pool' + str(i+1)] = MaxPooling2D((2, 2), strides=(2, 2), name='pool' + str(i+1))(locals()['conv'+str(i)])

    for i in range(len(fs)-2,-1,-1):
        locals()['upsamp'+str(i)]=concatenate([locals()['conv'+str(i)], Conv2DTranspose(fs[i], (3, 3),
            activation=act_fun, kernel_initializer=init, strides=(2, 2),padding='same',name='upsamp'+str(i))(locals()['conv'+str(i+1)])],
          axis=concat_axis) if i==len(fs)-2 else concatenate([locals()['conv'+str(i)], Conv2DTranspose(fs[i], (3, 3),
           activation=act_fun, kernel_initializer=init, strides=(2, 2), padding='same',name='upsamp'+str(i))(locals()['decon'+str(i+1)])], axis=concat_axis)
        locals()['decon'+str(i)] = Conv2D(fs[i], (ks[0], ks[0]), activation=act_fun, kernel_initializer=init, padding='same', name='decon'+str(i))(locals()['upsamp'+str(i)])
    locals()['out0'] = Conv2D(dim_out, (1, 1), activation=out_fun, padding='same',name='out0')(locals()['decon0'])
    return Model(locals()['pool0'], locals()['out0']), traceback.extract_stack(None, 2)[1].name  + "_" + str(cfg)

def unet_pool_trans_2f1(cfg):
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
    act_fun, out_fun=cfg.model_act, cfg.model_out
    dim_in, dim_out=cfg.dep_in, cfg.dep_out
    # img_input = Input((None, None, dim_in))  # r,c,3
    locals()['pool0']=Input((cfg.row_in, cfg.col_in, dim_in))  # r,c,3

    for i in range(len(fs)):
        locals()['conv'+str(i)]=Conv2D(fs[i], (ks[1], ks[1]), activation=act_fun, padding='same', kernel_initializer=init, name='conv' + str(i))\
                (Conv2D(fs[i], (ks[0], ks[0]), activation=act_fun, padding='same', kernel_initializer=init)(locals()['pool'+str(i)]))
        if i < len(fs)-1:
            locals()['pool' + str(i+1)] = MaxPooling2D((2, 2), strides=(2, 2), name='pool' + str(i+1))(locals()['conv'+str(i)])

    for i in range(len(fs)-2,-1,-1):
        locals()['upsamp'+str(i)]=concatenate([locals()['conv'+str(i)], Conv2DTranspose(fs[i], (3, 3),
            activation=act_fun, kernel_initializer=init, strides=(2, 2),padding='same',name='upsamp'+str(i))(locals()['conv'+str(i+1)])],
          axis=concat_axis) if i==len(fs)-2 else concatenate([locals()['conv'+str(i)], Conv2DTranspose(fs[i], (3, 3),
           activation=act_fun, kernel_initializer=init, strides=(2, 2), padding='same',name='upsamp'+str(i))(locals()['decon'+str(i+1)])], axis=concat_axis)
        locals()['decon'+str(i)] = Conv2D(fs[i], (ks[0], ks[0]), activation=act_fun, kernel_initializer=init, padding='same', name='decon'+str(i))(locals()['upsamp'+str(i)])
    locals()['out0'] = Conv2D(dim_out, (1, 1), activation=out_fun, padding='same',name='out0')(locals()['decon0'])
    return Model(locals()['pool0'], locals()['out0']), traceback.extract_stack(None, 2)[1].name  + "_" + str(cfg)

def unet_pool_trans_2f2(cfg):
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
    act_fun, out_fun=cfg.model_act, cfg.model_out
    dim_in, dim_out=cfg.dep_in, cfg.dep_out
    # img_input = Input((None, None, dim_in))  # r,c,3
    locals()['pool0']=Input((cfg.row_in, cfg.col_in, dim_in))  # r,c,3

    for i in range(len(fs)):
        locals()['conv'+str(i)]=Conv2D(fs[i], (ks[1], ks[1]), activation=act_fun, padding='same', kernel_initializer=init, name='conv' + str(i))\
                (Conv2D(fs[i], (ks[0], ks[0]), activation=act_fun, padding='same', kernel_initializer=init)(locals()['pool'+str(i)]))
        if i < len(fs)-1:
            locals()['pool' + str(i+1)] = MaxPooling2D((2, 2), strides=(2, 2), name='pool' + str(i+1))(locals()['conv'+str(i)])

    for i in range(len(fs)-2,-1,-1):
        locals()['upsamp'+str(i)]=concatenate([locals()['conv'+str(i)], Conv2DTranspose(fs[i], (3, 3),
            activation=act_fun, kernel_initializer=init, strides=(2, 2),padding='same',name='upsamp'+str(i))(locals()['conv'+str(i+1)])],
          axis=concat_axis) if i==len(fs)-2 else concatenate([locals()['conv'+str(i)], Conv2DTranspose(fs[i], (3, 3),
           activation=act_fun, kernel_initializer=init, strides=(2, 2), padding='same',name='upsamp'+str(i))(locals()['decon'+str(i+1)])], axis=concat_axis)
        locals()['decon'+str(i)] = Conv2D(fs[i], (ks[1], ks[1]), activation=act_fun, kernel_initializer=init, padding='same', name='decon'+str(i))\
            (Conv2D(fs[i], (ks[0], ks[0]), activation=act_fun, kernel_initializer=init, padding='same')(locals()['upsamp'+str(i)]))
    locals()['out0'] = Conv2D(dim_out, (1, 1), activation=out_fun, padding='same',name='out0')(locals()['decon0'])
    return Model(locals()['pool0'], locals()['out0']), traceback.extract_stack(None, 2)[1].name  + "_" + str(cfg)
