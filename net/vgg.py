import traceback

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, merge
from keras import backend as K
from net.basenet import Net
from net.module import cvac, ca3, ca33, cb3, cba3, dmp, uu, ct, sk

K.set_image_data_format("channels_last")
#K.set_image_dim_ordering("th")

#VGG 16 weighted layers (224 x 224 x 3 RGB)
# conv3x3-64 conv3x3-64 maxpool
# conv3x3-128 conv3x3-128 maxpool
# conv3x3-256 conv3x3-256 conv3x3/1x1-256 maxpool
# conv3x3-512 conv3x3-512 conv3x3/1x1-512 maxpool
# conv3x3-512 conv3x3-512 conv3x3/1x1-512 maxpool
# FC-4096 -> FC-4096 -> FC-1000 -> softmax
class VggSegNet(Net):
    # also base class for U-shaped networks
    def __init__(self, dim_in=None, dim_out=None, filters=None, poolings=None, preproc=None, downconv=None, downjoin=None, downsamp=None, downmerge=None,
                 downproc=None,
                 upconv=None, upjoin=None, upsamp=None, upmerge=None, upproc=None, postproc=None, **kwargs
                 ):
        super(VggSegNet,self).__init__(dim_in=dim_in or (768, 768, 3), dim_out=dim_out or (768, 768, 1), **kwargs)
        self.fs=filters or [64, 128, 256, 384, 384, 512, 512]
        self.ps=poolings or [2]*len(self.fs)
        self.preproc=preproc or cba3
        self.downconv=downconv or sk
        self.downjoin=downjoin or sk
        self.downsamp=downsamp or dmp
        self.downmerge=downmerge or sk
        self.downproc=downproc or cba3
        self.upconv=upconv or cb3
        self.upjoin=upjoin or sk
        self.upsamp=upsamp or uu
        self.upmerge=upmerge or sk
        self.upproc=upproc or cb3
        self.postproc=postproc or sk
        self.transferlayer=0

        locals()['in0']=Input((self.row_in, self.col_in, self.dep_in))
        locals()['pre0']=self.preproc(locals()['in0'], 'pre0', 0, self.fs[0], self.act)
        vgg16_base=VGG16(input_tensor=locals()['pre0'], include_top=False, weights=None)
        # for layer in vgg16_base.layers: layer.trainable = True

        locals()['djoin0']=locals()['dconv0']=vgg16_base.get_layer("block1_conv2").output
        locals()['dproc1']=locals()['dmerge1']=locals()['dsamp1']=vgg16_base.get_layer("block1_pool").output; self.transferlayer+=1
        locals()['djoin1']=locals()['dconv1']=vgg16_base.get_layer("block2_conv2").output
        locals()['dproc2']=locals()['dmerge2']=locals()['dsamp2']=vgg16_base.get_layer("block2_pool").output; self.transferlayer+=1
        locals()['djoin2']=locals()['dconv2']=vgg16_base.get_layer("block3_conv3").output
        locals()['dproc3']=locals()['dmerge3']=locals()['dsamp3']=vgg16_base.get_layer("block3_pool").output; self.transferlayer+=1
        # locals()['djoin3']=locals()['dconv3']=vgg16_base.get_layer("block4_conv3").output
        # locals()['dproc4']=locals()['dmerge4']=locals()['dsamp4']=vgg16_base.get_layer("block4_pool").output; self.transferlayer+=1
        # locals()['djoin4']=locals()['dconv4']=vgg16_base.get_layer("block5_conv3").output
        # locals()['dproc5']=locals()['dmerge5']=locals()['dsamp5']=vgg16_base.get_layer("block5_pool").output; self.transferlayer+=1

        for i in range(self.transferlayer, len(self.fs)-1):
            prev_layer=locals()['dsamp%d'%i] if i==self.transferlayer else locals()['dproc%d'%i]
            locals()['dconv%d'%i]=self.downconv(prev_layer, 'dconv%d'%i, i, self.fs[i], self.act)
            locals()['djoin%d'%i]=self.downjoin(locals()['dconv%d'%i], prev_layer, 'djoin%d'%i, i, self.fs[i], self.act)
            locals()['dsamp%d'%(i+1)]=self.downsamp(locals()['djoin%d'%i], self.ps[i], 'dsamp%d'%(i+1), i, self.fs[i], self.act)
            locals()['dmerge%d'%(i+1)]=self.downmerge(locals()['dsamp%d'%(i+1)], prev_layer, 'dmerge%d'%(i+1), i+1, self.fs[i+1], self.act, stride=self.ps[i])
            locals()['dproc%d'%(i+1)]=self.downproc(locals()['dmerge%d'%(i+1)], 'dproc%d'%(i+1), i+1, self.fs[i+1], self.act)

        for i in range(len(self.fs)-2, -1, -1):
            prev_layer=locals()['dproc%d'%(i+1)] if i==len(self.fs)-2 else locals()['uproc%d'%(i+1)]
            locals()['uconv%d'%(i+1)]=self.upconv(prev_layer, 'uconv%d'%(i+1), i, self.fs[i+1], self.act)
            locals()['ujoin%d'%(i+1)]=self.upjoin(locals()['uconv%d'%(i+1)], locals()['dmerge%d'%(i+1)], 'ujoin%d'%(i+1), i, self.fs[i+1], self.act)
            locals()['usamp%d'%i]=self.upsamp(locals()['ujoin%d'%(i+1)], self.ps[i], 'usamp%d'%i, i, self.fs[i+1], self.act)
            locals()['umerge%d'%i]=self.upmerge(locals()['usamp%d'%i], locals()['djoin%d'%i], 'umerge%d'%i, i, self.fs[i], self.act)
            locals()['uproc%d'%i]=self.upproc(locals()['umerge%d'%i], 'uproc%d'%i, i, self.fs[i], self.act)

        locals()['post0']=self.postproc(locals()['uproc0'], 'post0', 0, self.fs[0], self.act)
        locals()['out0']=cvac(locals()['post0'], 'out0', 0, self.dep_out, self.out, size=1)
        self.net=Model(locals()['in0'], locals()['out0'])

    def __str__(self):
        return '_'.join([
            type(self).__name__+str(self.transferlayer),
            "%dF%d-%dP%d-%d"%(
            len(self.fs), self.fs[0], self.fs[-1], self.ps[0], self.ps[-1]),
            # "%df%d-%dp%s" % (len(self.fs), self.fs[0], self.fs[-1], ''.join(self.pssize)),
            self.cap_lim_join(10, self.preproc.__name__, self.downconv.__name__,
                              self.downjoin.__name__, self.downsamp.__name__,
                              self.downmerge.__name__, self.downproc.__name__),
            self.cap_lim_join(10, self.upconv.__name__, self.upjoin.__name__,
                              self.upsamp.__name__, self.upmerge.__name__, self.upproc.__name__,
                              self.postproc.__name__),
            self.cap_lim_join(7, self.act, self.out,
                              (self.loss if isinstance(self.loss, str) else self.loss.__name__).
                              replace('_', '').replace('loss', ''))
            +str(self.dep_out)])