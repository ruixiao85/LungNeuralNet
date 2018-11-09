
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = KL.BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)
    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = KL.BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)
    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = KL.BatchNormalization(name=bn_name_base + '1')(shortcut, training=train_bn)
    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x

def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = KL.BatchNormalization(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]



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
    c1=x=c7m3d4(input_image,'res1',0,filters[0],act,batch_norm) # downsample twice
    c2=x=rn131r(x,'res2',repeats[1],filters[1],act,batch_norm,initial_stride=1) # override stride 2->1, same dim as prev maxpool
    c3=x=rn131r(x,'res3',repeats[2],filters[2],act,batch_norm) # downconv once initially
    c4=x=rn131r(x,'res4',repeats[3],filters[3],act,batch_norm) # downconv once initially
    c5=rn131r(x,'res5',repeats[4],filters[4],act,batch_norm) # downconv once initially
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
