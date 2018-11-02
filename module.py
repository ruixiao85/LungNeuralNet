from keras.layers import Concatenate, Activation, BatchNormalization, Conv2DTranspose, MaxPooling2D, AveragePooling2D, UpSampling2D, Conv2D, Dropout, Add
from keras import backend as K

K.set_image_data_format('channels_last')
# concat_axis = 3
init='he_normal'


# cv: convolution (default same padding)
# tr: transpose
# dp: dropout
# bn: batch normalization
# ac: activation
# mp: maxpooling
# us: upsampling
# ct: concatenate
# ad: add

def cv(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1, dilation=1):
    return Conv2D(fs, (size,size), strides=(stride,stride), dilation_rate=(dilation,dilation), padding='same', kernel_initializer=init, name=name)(in_layer)
def cvac(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1, dilation=1):
    x=cv(in_layer,name+'_cv',idx,fs,act,size,stride,dilation)
    return Activation(activation=act,name=name)(x)
def ac(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1, dilation=1):
    return Activation(activation=act,name=name)(in_layer)
def accv(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1, dilation=1):
    x=Activation(activation=act,name=name+'_ac')(in_layer)
    return cv(x,name,idx,fs,act,size,stride,dilation)
def cvacdp(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1, dilation=1):
    x=cvac(in_layer,name+'_cv',fs,act,size,stride,dilation)
    return Dropout(0.2,name=name)(x)
def cvbn(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1, dilation=1):
    x=cv(in_layer, name+'_cv', idx, fs, act, size, stride, dilation)
    return BatchNormalization(name=name)(x)
def cvbnac(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1, dilation=1):
    x=cv(in_layer, name+'_cv', idx, fs, act, size, stride, dilation)
    x=BatchNormalization(name=name+'_bn')(x)
    return Activation(activation=act, name=name)(x)
def bnaccv(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=1, dilation=1):
    x=BatchNormalization(name=name+'_bn')(in_layer)
    x=Activation(activation=act, name=name+'_ac')(x)
    return cv(x, name, idx, fs, act, size, stride, dilation)

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
    x=Activation(activation=act, name=name+'_ac')(x)
    return Conv2DTranspose(fs, (size, size), strides=(stride, stride), dilation_rate=(dilation,dilation), kernel_initializer=init, padding='same', name=name)(x)

def mp(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=2):
    return MaxPooling2D((size, size), strides=(stride, stride), name=name, padding='same')(in_layer)
def ap(in_layer, name=None, idx=None, fs=None, act=None, size=3, stride=2):
    return AveragePooling2D((size, size), strides=(stride, stride), name=name, padding='same')(in_layer)
def us(in_layer, name=None, idx=None, fs=None, act=None, size=None, stride=2):
    return UpSampling2D((stride, stride), name=name)(in_layer)

def step2kern(step):
    # return step # same
    return step if step%2==1 else step+1 # odd no less than step

def dca(in_layer, rate, name=None, idx=None, fs=None, act=None, size=None):
    return cvac(in_layer, name, idx, fs, act, size=size or step2kern(rate), stride=rate)

def dcba(in_layer, rate, name=None, idx=None, fs=None, act=None, size=None):
    return cvbnac(in_layer, name, idx, fs, act, size=size or step2kern(rate), stride=rate)

def dmp(in_layer, rate, name=None, idx=None, fs=None, act=None, size=None):
    return mp(in_layer, name, idx, fs, act, size=size or step2kern(rate), stride=rate)

def uu(in_layer, rate, name=None, idx=None, fs=None, act=None, size=None):
    return us(in_layer, name, idx, fs, act, size=size or step2kern(rate), stride=rate)
def ut(in_layer, rate, name=None, idx=None, fs=None, act=None, size=None):
    return tr(in_layer, name, idx, fs, act, size=size or step2kern(rate), stride=rate)
def uta(in_layer, rate, name=None, idx=None, fs=None, act=None, size=None):
    return trac(in_layer, name, idx, fs, act, size=size or step2kern(rate), stride=rate)
def utba(in_layer, rate, name=None, idx=None, fs=None, act=None, size=None):
    return trbnac(in_layer, name, idx, fs, act, size=size or step2kern(rate), stride=rate)

def sk(in_layer, other_layer, name=None, idx=None, fs=None, act=None, stride=None): # direct
    return in_layer
def ct(in_layer, other_layer, name=None, idx=None, fs=None, act=None, stride=None): # concat
    return in_layer if other_layer is None else Concatenate(name=name)([in_layer, other_layer])
def ad(in_layer, other_layer, name=None, idx=None, fs=None, act=None, stride=None): # add
    return in_layer if other_layer is None else Add(name=name)([in_layer, other_layer])

def ctac(in_layer, other_layer, name=None, idx=None, fs=None, act=None, stride=None): # concat, activation
    x=ct(in_layer, other_layer, name+'_ct', idx, fs, act, stride)
    return Activation(activation=act, name=name)(x)
def adac(in_layer, other_layer, name=None, idx=None, fs=None, act=None, stride=None): # add, activation
    x=ad(in_layer, other_layer, name+'_ad', idx, fs, act, stride)
    return Activation(activation=act, name=name)(x)

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
def cadh(in_layer, name=None, idx=None, fs=None, act=None):
    size=in_layer.get_shape()[1].value
    x=cvac(in_layer, name+'_dilate', idx, int(fs/3), act, size=3, dilation=int(size/2))
    x=Concatenate(name=name)([cvac(in_layer, name+'_preconv', idx, fs, act, size=3), x]) #
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
def cb3(in_layer, name=None, idx=None, fs=None, act=None):
    x=cvbn(in_layer, name, idx, fs, act, size=3)
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

