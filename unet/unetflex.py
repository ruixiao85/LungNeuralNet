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

def cv(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1, dilation=1):
    return Conv2D(fs, (size,size), strides=(stride,stride), dilation_rate=(dilation,dilation), padding='same', kernel_initializer=init, name=name)(in_layer)
def cvac(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1, dilation=1):
    return Conv2D(fs, (size,size), strides=(stride,stride), dilation_rate=(dilation,dilation), activation=act, padding='same', kernel_initializer=init, name=name)(in_layer)
def cvacdp(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1, dilation=1):
    x=cvac(in_layer,name+'_cv',fs,act,size,stride,dilation)
    return Dropout(0.2,name=name)(x)
def cvbn(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1, dilation=1):
    x=Conv2D(fs, (size,size), strides=(stride,stride), dilation_rate=(dilation,dilation), padding='same', kernel_initializer=init, name=name+'_cv')(in_layer)
    return BatchNormalization(name=name)(x)
def cvbnac(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1, dilation=1):
    x=Conv2D(fs, (size,size), strides=(stride,stride), dilation_rate=(dilation,dilation), padding='same', kernel_initializer=init, name=name+'_cv')(in_layer)
    x=BatchNormalization(name=name+'_bn')(x)
    return Activation(activation=act, name=name)(x)
def bnaccv(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1, dilation=1):
    x=BatchNormalization(name=name+'_bn')(in_layer)
    x=Activation(activation=act, name=name+'_ac')(x)
    return Conv2D(fs, (size,size), strides=(stride,stride), dilation_rate=(dilation,dilation), padding='same', kernel_initializer=init, name=name)(x)

def tr(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1, dilation=1):
    return Conv2DTranspose(fs, (size, size), strides=(stride, stride), dilation_rate=(dilation,dilation), kernel_initializer=init, padding='same', name=name)(in_layer)
def trac(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1, dilation=1):
    return Conv2DTranspose(fs, (size, size), strides=(stride, stride), dilation_rate=(dilation,dilation),activation=act, kernel_initializer=init, padding='same', name=name)(in_layer)
def trbnac(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1, dilation=1):
    x=Conv2DTranspose(fs, (size, size), strides=(stride, stride), dilation_rate=(dilation,dilation), kernel_initializer=init, padding='same', name=name+'_tr')(in_layer)
    x=BatchNormalization(name=name+'_bn')(x)
    return Activation(activation=act, name=name)(x)
def bnactr(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1, dilation=1):
    x=BatchNormalization(name=name+'_bn')(in_layer)
    x= Activation(activation=act, name=name+'_ac')(x)
    return Conv2DTranspose(fs, (size, size), strides=(stride, stride), dilation_rate=(dilation,dilation), kernel_initializer=init, padding='same', name=name)(x)

def mp(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=2):
    return MaxPooling2D((size, size), strides=(stride, stride), name=name, padding='same')(in_layer)
def ap(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=2):
    return AveragePooling2D((size, size), strides=(stride, stride), name=name, padding='same')(in_layer)
def us(in_layer, name=None, idx=None, fs=None, act=None, size=None, stride=2):
    return UpSampling2D((stride, stride), name=name)(in_layer)


def ca1(in_layer, name=None, idx=None, fs=None, act=None):
    x=cvac(in_layer, name, idx, fs*3, act, size=1) # if concat 2 times
    return x
def ca2(in_layer, name=None, idx=None, fs=None, act=None):
    x=cvac(in_layer, name, idx, fs, act, size=2)
    return x
def ca3(in_layer, name=None, idx=None, fs=None, act=None):
    x=cvac(in_layer, name, idx, fs, act, size=3)
    return x
def ca3h(in_layer, name=None, idx=None, fs=None, act=None):
    x=cvac(in_layer, name, idx, int(fs/2), act, size=3)
    return x
def ca13(in_layer, name=None, idx=None, fs=None, act=None):
    x=cvac(in_layer, name+'_1', idx, fs, act, size=1)
    x=cvac(x, name, idx, fs, act, size=3)
    return x
def ca31(in_layer, name=None, idx=None, fs=None, act=None):
    x=cvac(in_layer, name+'_1', idx, fs, act, size=3)
    x=cvac(x, name, idx, fs, act, size=1)
    return x
def ca33(in_layer, name=None, idx=None, fs=None, act=None):
    x=cvac(in_layer, name+'_1', idx, fs, act, size=3)
    x=cvac(x, name, idx, fs, act, size=3)
    return x
def cba3(in_layer, name=None, idx=None, fs=None, act=None):
    x=cvbnac(in_layer, name, idx, fs, act, size=3)
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

def uu(in_layer, rate, name=None, idx=None, fs=None, act=None):
    return us(in_layer, name, idx, fs, act, size=step2kern(rate), stride=rate)
def ut(in_layer, rate, name=None, idx=None, fs=None, act=None):
    return tr(in_layer, name, idx, fs, act, size=step2kern(rate), stride=rate)
def uta(in_layer, rate, name=None, idx=None, fs=None, act=None):
    return trac(in_layer, name, idx, fs, act, size=step2kern(rate), stride=rate)
def utba(in_layer, rate, name=None, idx=None, fs=None, act=None):
    return trbnac(in_layer, name, idx, fs, act, size=step2kern(rate), stride=rate)

def s(in_layer, other_layer, name=None, idx=None, fs=None, act=None, stride=None): # direct
    return in_layer
def c(in_layer, other_layer, name=None, idx=None, fs=None, act=None, stride=None): # concat
    if other_layer is None:
        return in_layer
    elif stride is None:
        return Concatenate(name=name)([in_layer, other_layer])
    else:
        return Concatenate(name=name)([in_layer, mp(other_layer, name+'_rs', idx, fs, act, step2kern(stride), stride)])


# du: Deep U-Net with residual https://arxiv.org/pdf/1709.00201.pdf 640x640
# first layer pre-conv 64, 3x3. filter number stay the same across stages
def du32(in_layer, name=None, idx=None, fs=None, act=None): # downward: f64k3, f32k2+res per stage
    x=cvac(in_layer,name+'_2',idx,fs,act,size=3)
    x=cv(x,name+'_1',idx,int(fs/2),act,size=2)
    return Activation(activation=act, name=name)(
        Concatenate(name=name+'_c')([x, in_layer])
    )
def cdu33(in_layer, other_layer, name=None, idx=None, fs=None, act=None, stride=None): # upward join: f64k3, f32k3+res per stage
    x=Concatenate(name=name+'_j')([in_layer,other_layer])
    x=cvac(x, name+'_2', idx, fs, act, size=3)
    x=cv(x, name+'_1', idx, int(fs/2), act, size=3)
    return Activation(activation=act, name=name)(
        Concatenate(name=name+'_c')([x, in_layer])
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
    x, rep=in_layer,1+idx
    for i in range(rep-1,-1,-1):
        x=rn33(x, name+str(i)[:i], idx, filters, act)
    return x
def rn33rh(in_layer, name, idx, filters, act):
    x, rep=in_layer,1+idx
    for i in range(rep-1,-1,-1):
        x=rn33(x, name+str(i)[:i], idx, int(filters/2), act)
    return x
def rn33nr(in_layer, name, idx, filters, act):
    x, rep=in_layer,2+idx
    for i in range(rep-1,-1,-1):
        x=rn33n(x, name+str(i)[:i], idx, filters, act)
    return x
def rn33nrh(in_layer, name, idx, filters, act):
    x, rep=in_layer,2+idx
    for i in range(rep-1,-1,-1):
        x=rn33n(x, name+str(i)[:i], idx, int(filters/2), act)
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
    x=cvac(in_layer, name+'_2', idx, filters, act, size=1)
    x=cv(x, name+'_1', idx, filters, act, size=3)
    y=cv(in_layer, name+'_s', idx, filters, act, size=1)
    return Activation(activation=act, name=name)(
        Add(name=name+'_a')([x, y])  # shortcut with conv
    )
def dn13n(in_layer, name, idx, filters, act):
    x=cvbnac(in_layer, name+'_2', idx, filters, act, size=1)
    x=cvbnac(x, name+'_1', idx, filters, act, size=3)
    y=cvbn(in_layer, name+'_s', idx, filters, act, size=1)
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


def unet(cfg: ModelConfig):
    fs=cfg.model_filter
    ps=cfg.model_pool
    locals()['in0']=Input((cfg.row_in, cfg.col_in, cfg.dep_in))
    locals()['pre0']=cfg.model_preproc(locals()['in0'],'pre0',0,fs[0],cfg.model_act)
    for i in range(len(fs)-1):
        prev_layer=locals()['pre%d'%i] if i==0 else locals()['dproc%d'%i]
        locals()['dconv%d'%i]=cfg.model_downconv(prev_layer, 'dconv%d'%i, i, fs[i], cfg.model_act)
        locals()['djoin%d'%i]=cfg.model_downjoin(locals()['dconv%d'%i], prev_layer, 'djoin%d'%i, i, fs[i], cfg.model_act)
        locals()['dsamp%d'%(i+1)]=cfg.model_downsamp(locals()['djoin%d'%i], ps[i], 'dsamp%d'%(i+1), i, fs[i], cfg.model_act)
        locals()['dmerge%d'%(i+1)]=cfg.model_downmerge(locals()['dsamp%d'%(i+1)], prev_layer, 'dmerge%d'%(i+1), i+1, fs[i+1], cfg.model_act, stride=ps[i])
        locals()['dproc%d'%(i+1)]=cfg.model_downproc(locals()['dmerge%d'%(i+1)], 'dproc%d'%(i+1), i+1, fs[i+1], cfg.model_act)

    for i in range(len(fs)-2, -1, -1):
        prev_layer=locals()['dproc%d'%(i+1)] if i==len(fs)-2 else locals()['uproc%d'%(i+1)]
        locals()['uconv%d'%(i+1)]=cfg.model_upconv(prev_layer, 'uconv%d'%(i+1), i, fs[i+1], cfg.model_act)
        locals()['ujoin%d'%(i+1)]=cfg.model_upjoin(locals()['uconv%d'%(i+1)], locals()['dmerge%d'%(i+1)], 'ujoin%d'%(i+1), i, fs[i+1], cfg.model_act)
        locals()['usamp%d'%i]=cfg.model_upsamp(locals()['ujoin%d'%(i+1)], ps[i], 'usamp%d'%i, i, fs[i+1], cfg.model_act)
        locals()['umerge%d'%i]=cfg.model_upmerge(locals()['usamp%d'%i], locals()['djoin%d'%i], 'umerge%d'%i, i, fs[i], cfg.model_act)
        locals()['uproc%d'%i]=cfg.model_upproc(locals()['umerge%d'%i], 'uproc%d'%i, i, fs[i], cfg.model_act)

    locals()['post0']=cfg.model_postproc(locals()['uproc0'],'post0',0,fs[0],cfg.model_act)
    locals()['out0']=Conv2D(cfg.dep_out, (1, 1), activation=cfg.model_out, padding='same', name='out0')(locals()['post0'])
    return Model(locals()['in0'], locals()['out0'])
