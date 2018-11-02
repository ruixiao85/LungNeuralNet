
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

# Resnet # https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
# rn: ResNet https://arxiv.org/pdf/1512.03385.pdf 224x224
# first layer f64, k7x7, s2x2 -> maxpool k3x3, s2x2
# (64,3x3->64,3x3) repeat 2~6 or (64,1x1->64,3x3->256,1x1) repeat 3~8 times, double filters
from module import cv, cvac, cvbn, cvbnac, adac, mp
def c7m3d4(in_layer, name, idx, filters, act, batch_norm=False): #pre-process conv+maxpool size down 4x
    ca_f=cvbnac if batch_norm else cvac
    x=ca_f(in_layer,name+'_c7d2',idx=idx,fs=filters,act=act,size=7,stride=2)
    x=mp(x,name+'_m3d2',idx=idx,fs=None,act=None,size=3,stride=2)
    return x
def rn131c(in_layer, name, idx, filters, act, batch_norm=False, stride=2, dilate=1):
    ca_f,c_f=(cvbnac,cvbn) if batch_norm else (cvac,cv)
    x=ca_f(in_layer, name+'_3', idx, filters[0], act, size=1, stride=stride)
    x=ca_f(x, name+'_2', idx, filters[1], act, size=3, dilation=dilate)
    x=c_f(x, name+'_1', idx, filters[2], act, size=1)
    y=c_f(in_layer, name+'_s', idx, filters[2], act, size=1, stride=stride)
    return adac(x,y,name,idx,filters,act)
def rn131i(in_layer, name, idx, filters, act, batch_norm=False, stride=1, dilate=1):
    ca_f,c_f=(cvbnac,cvbn) if batch_norm else (cvac,cv)
    x=ca_f(in_layer, name+'_3', idx, filters[0], act, size=1, stride=stride)
    x=ca_f(x, name+'_2', idx, filters[1], act, size=3, dilation=dilate)
    x=c_f(x, name+'_1', idx, filters[2], act, size=1, stride=stride)
    return adac(x,in_layer,name,idx,filters,act)
def rn131r(in_layer, name, repeat, filters, act, batch_norm=False,initial_stride=2):
    filters=filters if isinstance(filters, list) else [filters, filters, filters*4] if isinstance(filters, int) else [64,64,256]
    x=in_layer
    for r in range(repeat):
        if r==0:
            x=rn131c(x,name+'_c',r,filters,act,batch_norm,initial_stride)
        else:
            x=rn131i(x,'%s_i%d'%(name,r),r,filters,act,batch_norm)
    return x

def resnet(input_image, repeats, filters, act='relu', batch_norm=False):
    c1=x=c7m3d4(input_image,'resnet_s1',0,filters[0],act,batch_norm) # downsample twice
    c2=x=rn131r(x,'resnet_s2',repeats[1],filters[1],act,batch_norm,initial_stride=1) # override stride 2->1, same dim as prev maxpool
    c3=x=rn131r(x,'resnet_s3',repeats[2],filters[2],act,batch_norm) # downconv once initially
    c4=x=rn131r(x,'resnet_s4',repeats[3],filters[3],act,batch_norm) # downconv once initially
    c5=rn131r(x,'resnet_s5',repeats[4],filters[4],act,batch_norm) # downconv once initially
    return [c1, c2, c3, c4, c5]
def resnet_50(input_image):
    return resnet(input_image, repeats=[1,3,4,6,3], filters=[64,64,128,256,512])
def resnet_101(input_image):
    return resnet(input_image, repeats=[1,3,4,23,3], filters=[64,64,128,256,512])
def resnet_152(input_image):
    return resnet(input_image, repeats=[1,3,8,36,3], filters=[64,64,128,256,512])


# dn: Dense Net https://arxiv.org/pdf/1608.06993.pdf 224x224
# first layer f64, k7x7, s2x2 -> maxpool k3x3, s2x2
# (12,1x1->12,3x3) repeat 6, 12, 24,... increase repetition/density per block. nb-ac-conv suggested.
def dn13(in_layer, name, idx, filters, act):
    x=cvac(in_layer, name+'_2', idx, filters, act, size=1)
    x=cv(x, name+'_1', idx, filters, act, size=3)
    y=cv(in_layer, name+'_s', idx, filters, act, size=1)
    return adac(x,y,name,idx,filters,act)
def dn13n(in_layer, name, idx, filters, act):
    x=cvbnac(in_layer, name+'_2', idx, filters, act, size=1)
    x=cvbnac(x, name+'_1', idx, filters, act, size=3)
    y=cvbn(in_layer, name+'_s', idx, filters, act, size=1)
    return adac(x,y,name,idx,filters,act)
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
