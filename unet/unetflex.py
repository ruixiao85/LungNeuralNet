from __future__ import print_function


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, Concatenate, \
    BatchNormalization, Dropout, Activation, Conv2DTranspose, Add, AveragePooling2D, ZeroPadding1D, Multiply, Dot
from keras import backend as K

from model_config import ModelConfig

K.set_image_data_format('channels_last')
concat_axis = 3
init='he_normal'

# cv: convolution
# tr: transpose
# dp: dropout
# bn: batch normalization
# ac: activation
# mp: maxpooling
# us: upsampling

def cv(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1):
    return Conv2D(fs, (size,size), strides=(stride,stride), padding='same', kernel_initializer=init, name=name)(in_layer)
def cvac(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1):
    return Conv2D(fs, (size,size), strides=(stride,stride), activation=act, padding='same', kernel_initializer=init, name=name)(in_layer)
def cvacdp(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1):
    x=cvac(in_layer,name+'_cv',fs,act,size,stride)
    return Dropout(0.2,name=name)(x)
def cvbn(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1):
    x=Conv2D(fs, (size,size), strides=(stride,stride), padding='same', kernel_initializer=init, name=name+'_cv')(in_layer)
    return BatchNormalization(name)(x)
def cvbnac(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1):
    x=Conv2D(fs, (size,size), strides=(stride,stride), padding='same', kernel_initializer=init, name=name+'_cv')(in_layer)
    x=BatchNormalization(name=name+'_bn')(x)
    return Activation(activation=act, name=name)(x)
def bnaccv(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1):
    x=BatchNormalization(name=name+'_bn')(in_layer)
    x=Activation(activation=act, name=name+'_ac')(x)
    return Conv2D(fs, (size,size), strides=(stride,stride), padding='same', kernel_initializer=init, name=name)(x)

def tr(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1):
    return Conv2DTranspose(fs, (size, size), strides=(stride, stride),kernel_initializer=init, padding='same', name=name)(in_layer)
def trac(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1):
    return Conv2DTranspose(fs, (size, size), strides=(stride, stride),activation=act, kernel_initializer=init, padding='same', name=name)(in_layer)
def trbnac(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1):
    x=Conv2DTranspose(fs, (size, size), strides=(stride, stride), kernel_initializer=init, padding='same', name=name+'_tr')(in_layer)
    x=BatchNormalization(name=name+'_bn')(x)
    return Activation(activation=act, name=name)(x)
def bnactr(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1):
    x=BatchNormalization(name=name+'_bn')(in_layer)
    x= Activation(activation=act, name=name+'_ac')(x)
    return Conv2DTranspose(fs, (size, size), strides=(stride, stride), kernel_initializer=init, padding='same', name=name)(x)

def mp(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=2):
    return MaxPooling2D((size, size), strides=(stride, stride), name=name, padding='same')(in_layer)
def us(in_layer, name=None, idx=None, fs=None, act=None, size=None, stride=2):
    return UpSampling2D((stride, stride), name=name)(in_layer)


def ca3(in_layer, name=None, idx=None, fs=None, act=None):
    x=cvac(in_layer, name, idx, fs, act, size=3)
    return x
def ca31(in_layer, name=None, idx=None, fs=None, act=None):
    x=cvac(in_layer, name+'_1', idx, fs, act, size=3)
    x=cvac(x, name, idx, fs, act, size=1)
    return x
def ca33(in_layer, name=None, idx=None, fs=None, act=None):
    x=cvac(in_layer, name+'_1', idx, fs, act, size=3)
    x=cvac(x, name, idx, fs, act, size=3)
    return x
def cba33(in_layer, name=None, idx=None, fs=None, act=None):
    x=cvbnac(in_layer, name+'_1', idx, fs, act, size=3)
    x=cvbnac(x, name, idx, fs, act, size=3)
    return x
def ca33d(in_layer, name=None, idx=None, fs=None, act=None):
    x=cvac(in_layer, name+'_1', idx, fs, act, size=3)
    x=cvacdp(x, name, idx, fs, act, size=3)
    return x
def ca3d3(in_layer, name=None, idx=None, fs=None, act=None):
    x=cvacdp(in_layer, name+'_1', idx, fs, act, size=3)
    x=cvac(x, name, idx, fs, act, size=3)
    return x
def ca333(in_layer, name=None, idx=None, fs=None, act=None):
    x=cvac(in_layer, name+'_2', idx, fs, act, size=3)
    x=cvac(x, name+'_1', idx, fs, act, size=3)
    x=cvac(x, name, idx, fs, act, size=3)
    return x
def ca331(in_layer, name=None, idx=None, fs=None, act=None):
    x=cvac(in_layer, name+'_2', idx, fs, act, size=3)
    x=cvac(x, name+'_1', idx, fs, act, size=3)
    x=cvac(x, name, idx, fs, act, size=1)
    return x

def step2kern(step):
    # return step # same
    return step if step%2==1 else step+1 # odd no less than step

def dca(in_layer, rate, name=None, idx=None, fs=None, act=None):
    return cvac(in_layer, name, idx, fs, act, size=step2kern(rate), stride=rate)

def dcba(in_layer, rate, name=None, idx=None, fs=None, act=None):
    return cvbnac(in_layer, name, idx, fs, act, size=step2kern(rate), stride=rate)

def dmp(in_layer, rate, name=None, idx=None, fs=None, act=None):
    return mp(in_layer, name, idx, fs, act, size=step2kern(rate), stride=rate)

def uuc(input_skip, input_up, rate, name=None, idx=None, fs=None, act=None):
    x=us(input_up, name+'_us', idx, fs, act, size=step2kern(rate), stride=rate)
    return Concatenate(name=name, axis=concat_axis)([input_skip,x])
def utc(input_skip, input_up, rate, name=None, idx=None, fs=None, act=None):
    x=tr(input_up, name+'_tr', idx, fs, act, size=step2kern(rate), stride=rate)
    return Concatenate(name=name, axis=concat_axis)([input_skip,x])
def utac(input_skip, input_up, rate, name=None, idx=None, fs=None, act=None):
    x=trac(input_up, name+'_trac', idx, fs, act, size=step2kern(rate), stride=rate)
    return Concatenate(name=name, axis=concat_axis)([input_skip,x])
def utbac(input_skip, input_up, rate, name=None, idx=None, fs=None, act=None):
    x=trbnac(input_up, name+'_tn', idx, fs, act, size=step2kern(rate), stride=rate)
    return Concatenate(name=name, axis=concat_axis)([input_skip,x])


# du: Deep U-Net with residual https://arxiv.org/pdf/1709.00201.pdf 640x640
# first layer pre-conv 64, 3x3. filter number stay the same across stages
def du32(in_layer, name=None, idx=None, fs=None, act=None): # downward: f64k3, f32k2+res per stage
    x=cvac(in_layer,name+'_2',idx,fs,act,size=3)
    x=cv(x,name+'_1',idx,int(fs/2),act,size=2)
    return Activation(activation=act, name=name)(
        Concatenate(name=name+'_c')([in_layer, x])
    )
def du33(in_layer, name=None, idx=None, fs=None, act=None): # upward: f64k3, f32k3+res per stage
    x=cvac(in_layer,name+'_2',idx,fs,act,size=3)
    x=cv(x,name+'_1',idx,int(fs/2),act,size=3)
    return Activation(activation=act, name=name)(
        Concatenate(name=name+'_c')([in_layer, x])
    )

# rn: ResNet https://arxiv.org/pdf/1512.03385.pdf 224x224
# first layer f64, k7x7, s2x2 -> maxpool k3x3, s2x2
# (64,3x3->64,3x3) repeat 2~6 or (64,1x1->64,3x3->256,1x1) repeat 3~8 times, double filters
def rn33n(in_layer, name, idx, filters, act):
    x = cvbnac(in_layer, name+'_2', idx, filters, act, size=3)
    x = cvbn(x, name+'_1', idx, filters, act, size=3)
    y = cvbn(in_layer, name+'_s', idx, filters, act, size=3)
    return Activation(activation=act, name=name)(
        Add(name=name+'_a')([x,y])  # shortcut with conv
    )
def rn33(in_layer, name, idx, filters, act):
    x=cvac(in_layer, name+'_2', idx, filters, act, size=3)
    x=cv(x, name+'_1', idx, filters, act, size=3)
    y=cv(in_layer, name+'_s', idx, filters, act, size=3)
    return Activation(activation=act, name=name)(
        Add(name=name+'_a')([x, y])  # shortcut with conv
    )
def rn33r(in_layer, name, idx, filters, act):
    x, rep=in_layer,2+idx
    for i in range(rep-1,-1,-1):
        x=rn33(x, name+str(i)[:i], idx, filters, act)
    return x
def rn33nr(in_layer, name, idx, filters, act):
    x, rep=in_layer,2+idx
    for i in range(rep-1,-1,-1):
        x=rn33n(x, name+str(i)[:i], idx, filters, act)
    return x

def rn131n(in_layer, name, idx, filters, act):
    filters=filters if isinstance(filters,list) else [int(filters/4),int(filters/4),filters]  if isinstance(filters,int) else [32,32,64]
    x = cvbnac(in_layer, name+'_3', idx, filters[0], act, size=1)
    x = cvbnac(x, name+'_2', idx, filters[1], act, size=3)
    x = cvbn(x, name+'_1', idx, filters[2], act, size=1)
    y = cvbn(in_layer, name+'_s', idx, filters[2], act, size=1)
    return Activation(activation=act, name=name)(
        Add(name=name+'_a')([x,y])  # shortcut with conv
    )
def rn131(in_layer, name, idx, filters, act):
    filters=filters if isinstance(filters, list) else [int(filters/4), int(filters/4), filters] if isinstance(filters, int) else [32,32,64]
    x=cvac(in_layer, name+'_3', idx, filters[0], act, size=1)
    x=cvac(x, name+'_2', idx, filters[1], act, size=3)
    x=cv(x, name+'_1', idx, filters[2], act, size=1)
    y=cv(in_layer, name+'_s', idx, filters[2], act, size=1)
    return Activation(activation=act, name=name)(
        Add(name=name+'_a')([x, y])  # shortcut with conv
    )
def rn131r(in_layer, name, idx, filters, act):
    x, rep=in_layer,3+idx
    for i in range(rep-1,-1,-1):
        x=rn131(x, name+str(i)[:i], idx, filters, act)
    return x
def rn131nr(in_layer, name, idx, filters, act):
    x, rep=in_layer,3+idx
    for i in range(rep-1,-1,-1):
        x=rn131n(x, name+str(i)[:i], idx, filters, act)
    return x

# dn: Dense Net https://arxiv.org/pdf/1608.06993.pdf 224x224
# first layer f64, k7x7, s2x2 -> maxpool k3x3, s2x2
# (12,1x1->12,3x3) repeat 6, 12, 24,... increase repetition/density per block. nb-ac-conv suggested.
def dn13(in_layer, name, idx, filters, act):
    x=cvac(in_layer, name+'_2', idx, filters[0], act, size=1)
    x=cv(x, name+'_1', idx, filters[1], act, size=3)
    y=cv(in_layer, name+'_s', idx, filters[2], act, size=1)
    return Activation(activation=act, name=name)(
        Add(name=name+'_a')([x, y])  # shortcut with conv
    )
def dn13n(in_layer, name, idx, filters, act):
    x=cvbnac(in_layer, name+'_2', idx, filters[0], act, size=1)
    x=cvbnac(x, name+'_1', idx, filters[1], act, size=3)
    y=cvbn(in_layer, name+'_s', idx, filters[2], act, size=1)
    return Activation(activation=act, name=name)(
        Add(name=name+'_a')([x, y])  # shortcut with conv
    )
def dn13r(in_layer, name, idx, filters, act):
    x, rep=in_layer, 6**(idx+1)
    for i in range(rep-1, -1, -1):
        x=dn13(x, name+str(i)[:i], idx, filters, act)
    return x
def dn13nr(in_layer, name, idx, filters, act):
    x, rep=in_layer, 6**(idx+1)
    for i in range(rep-1, -1, -1):
        x=dn13n(x, name+str(i)[:i], idx, filters, act)
    return x
def dn13r1(in_layer, name, idx, filters, act):
    x, rep=in_layer, 6**(idx+1)
    for i in range(rep-1, -1, -1):
        x=dn13(x, name+str(i)[:i+1], idx, filters, act)
    return cvac(x, name, filters, act, size=1)
def dn13nr1(in_layer, name, idx, filters, act):
    x, rep=in_layer, 6**(idx+1)
    for i in range(rep-1, -1, -1):
        x=dn13n(x, name+str(i)[:i+1], idx, filters, act)
    return cvbnac(x, name, filters, act, size=1)


def c7m3d4(in_layer, name, idx, filters, act): #pre-process conv+maxpool size down 4x
    x=cvac(in_layer,name+'_c7d2',idx=0,fs=64,act=act,size=7,stride=2)
    x=mp(x,name+'_m3d2',idx=0,fs=None,act=None,size=3,stride=2)
    return x


def unet1s(cfg: ModelConfig):
    fs=cfg.model_filter
    ps=cfg.model_pool
    locals()['pool0']=Input((cfg.row_in, cfg.col_in, cfg.dep_in))

    for i in range(len(fs)):
        locals()['conv'+str(i)]=cfg.model_downconv(locals()['pool'+str(i)], 'conv'+str(i), fs[i], i, cfg.model_act)
        if i<len(fs)-1:
            locals()['pool'+str(i+1)]=cfg.model_downsamp(locals()['conv'+str(i)], ps[i], 'pool'+str(i+1), i, fs[i], cfg.model_act)

    for i in range(len(fs)-2, -1, -1):
        locals()['upsamp'+str(i)]=cfg.model_upsamp(locals()['conv'+str(i)],
                                                   locals()['conv'+str(i+1)] if i==len(fs)-2 else locals()['decon'+str(i+1)],
                                                   ps[i], 'upsamp'+str(i), i, fs[i], cfg.model_act)
        locals()['decon'+str(i)]=cfg.model_upconv(locals()['upsamp'+str(i)], 'decon'+str(i), i, fs[i], cfg.model_act)

    locals()['out0']=Conv2D(cfg.dep_out, (1, 1), activation=cfg.model_out, padding='same', name='out0')(locals()['decon0'])
    return Model(locals()['pool0'], locals()['out0'])


def unet1d(cfg: ModelConfig):
    fs=cfg.model_filter
    ps=cfg.model_pool
    locals()['pool0']=Input((cfg.row_in, cfg.col_in, cfg.dep_in))
    for i in range(len(fs)):
        locals()['conv'+str(i)]=cfg.model_downconv(locals()['pool'+str(i)], 'conv'+str(i), i, fs[i], cfg.model_act)
        if i<len(fs)-1:
            locals()['pool'+str(i+1)]=cfg.model_downsamp(locals()['conv'+str(i)], ps[i], 'pool'+str(i+1), i, fs[i], cfg.model_act)

    for i in range(len(fs)-2, -1, -1):
        locals()['secbind'+str(i+1)]=Concatenate(name='secbind'+str(i+1))([locals()['pool'+str(i+1)], locals()['conv'+str(i+1)]]) if i==len(fs)-2 else\
                                    Concatenate(name='secbind'+str(i+1))([locals()['pool'+str(i+1)], locals()['decon'+str(i+1)]])
        locals()['upsamp'+str(i)]=cfg.model_upsamp(locals()['conv'+str(i)], locals()['secbind'+str(i+1)],
                                                   ps[i], 'upsamp'+str(i), i, fs[i], cfg.model_act)
        locals()['decon'+str(i)]=cfg.model_upconv(locals()['upsamp'+str(i)], 'decon'+str(i), i, fs[i], cfg.model_act)

    locals()['out0']=Conv2D(cfg.dep_out, (1, 1), activation=cfg.model_out, padding='same', name='out0')(locals()['decon0'])
    return Model(locals()['pool0'], locals()['out0'])


def unet2s(cfg: ModelConfig):
    fs=cfg.model_filter
    ps=cfg.model_pool
    locals()['pool0']=Input((cfg.row_in, cfg.col_in, cfg.dep_in))
    long=len(cfg.model_filter)
    short=int(long/2)

    for lyr, div in [(short, 1), (long, 2)]:
        for i in range(lyr):
            locals()[str(lyr)+'conv'+str(i)]=cfg.model_downconv(locals()[(str(lyr) if i!=0 else '')+'pool'+str(i)],
                                                                str(lyr)+'conv'+str(i), i, int(fs[i]/div), cfg.model_act)
            if i<lyr-1:
                locals()[str(lyr)+'pool'+str(i+1)]=cfg.model_downsamp(locals()[str(lyr)+'conv'+str(i)], ps[i], str(lyr)+'pool'+str(i+1), i, fs[i], cfg.model_act)

        for i in range(lyr-2, -1, -1):
            locals()[str(lyr)+'upsamp'+str(i)]=cfg.model_upsamp(locals()[str(lyr)+str(len(fs))+'conv'+str(i)],
                                                    locals()[str(lyr)+str(len(fs))+'conv'+str(i+1)] if i==len(fs)-2 else locals()[str(lyr)+'decon'+str(i+1)],
                                                    ps[i], str(lyr)+'upsamp'+str(i), i, fs[i], cfg.model_act)

            locals()[str(lyr)+'decon'+str(i)]=cfg.model_upconv(locals()[str(lyr)+'upsamp'+str(i)], str(lyr)+'decon'+str(i), i, int(fs[i]/div), cfg.model_act)
    locals()['twobranch']=Concatenate(name='twobranch')([locals()[str(short)+'decon0'], locals()[str(long)+'decon0']], axis=concat_axis)
    locals()['out0']=Conv2D(cfg.dep_out, (1, 1), activation=cfg.model_out, padding='same', name='out0')(locals()['twobranch'])
    return Model(locals()['pool0'], locals()['out0'])
