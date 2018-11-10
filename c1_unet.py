from __future__ import print_function

from keras.models import Model
from keras.layers import Input
from b1_net_pair import BaseNetU
from module import cvac, ca3, ca33, cb3, cba3, dmp, uu, ct, sk

class UNet(BaseNetU):
    # also base class for U-shaped networks
    def __init__(self, dim_in=None, dim_out=None, filters=None, poolings=None, preproc=None, downconv=None,downjoin=None,downsamp=None,downmerge=None,downproc=None,
                 upconv=None, upjoin=None, upsamp=None, upmerge=None, upproc=None, postproc=None, **kwargs
        ):
        super(UNet,self).__init__(dim_in=dim_in or (768, 768, 3), dim_out=dim_out or (768, 768, 1), **kwargs)
        # UNET valid padding 572,570,568->284,282,280->140,138,136->68,66,64->32,30,28->56,54,52->104,102,100->200,198,196->392,390,388 388/572=67.8322% center
        # UNET same padding 576->288->144->72->36->72->144->288->576 take central 68% =392
        self.fs=filters or [64, 128, 256, 512, 1024]
        self.ps=poolings or [2]*len(self.fs)
        self.preproc=preproc or ca3
        self.downconv=downconv or ca3
        self.downjoin=downjoin or sk
        self.downsamp=downsamp or dmp
        self.downmerge=downmerge or sk
        self.downproc=downproc or ca3
        self.upconv=upconv or sk
        self.upjoin=upjoin or sk  # 2nd no skip
        self.upsamp=upsamp or uu
        self.upmerge=upmerge or ct  # 1st skip
        self.upproc=upproc or ca33
        self.postproc=postproc or sk

        locals()['in0']=Input((self.row_in, self.col_in, self.dep_in))
        locals()['pre0']=self.preproc(locals()['in0'],'pre0',0,self.fs[0],self.act)
        for i in range(len(self.fs)-1):
            prev_layer=locals()['pre%d'%i] if i==0 else locals()['dproc%d'%i]
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
    
        locals()['post0']=self.postproc(locals()['uproc0'],'post0',0,self.fs[0],self.act)
        locals()['out0']=cvac(locals()['post0'], 'out0', 0, self.dep_out, self.out, size=1)
        self.net=Model(locals()['in0'], locals()['out0'])
        self.compile_net()

    def __str__(self):
        return '_'.join([
            type(self).__name__,
            "%dF%d-%dP%d-%d"%(
            len(self.fs), self.fs[0], self.fs[-1], self.ps[0], self.ps[-1]),
            # "%df%d-%dp%s" % (len(self.fs), self.fs[0], self.fs[-1], ''.join(self.pssize)),
            # self.cap_lim_join(10, self.preproc.__name__, self.downconv.__name__,
            #                   self.downjoin.__name__, self.downsamp.__name__,
            #                   self.downmerge.__name__, self.downproc.__name__),
            # self.cap_lim_join(10, self.upconv.__name__, self.upjoin.__name__,
            #                   self.upsamp.__name__, self.upmerge.__name__, self.upproc.__name__,
            #                   self.postproc.__name__),
            self.cap_lim_join(4, self.feed, self.act, self.out,
                              (self.loss if isinstance(self.loss, str) else self.loss.__name__).
                              replace('_', '').replace('loss', ''))
            +str(self.dep_out)])

class UNetS(UNet):
    # default 1296x1296 with 2 skip connections, small memory consumption with 3x3 convolution only once for output
    def __init__(self, dim_in=None, dim_out=None, filters=None, poolings=None, **kwargs):
        super(UNetS,self).__init__(dim_in=dim_in or (1296,1296,3), dim_out=dim_out or (1296,1296,1),
                         filters=filters or [64, 96, 128, 196, 256, 256, 256, 256, 256], poolings=poolings or [2, 2, 2, 2, 3, 3, 3, 3, 3], **kwargs)

class UNet2(UNet):
    # default 768x768 with 2 skip connections, standard 3x3 conv twice per block
    def __init__(self, dim_in=None, dim_out=None, filters=None, poolings=None, **kwargs):
        super(UNet2,self).__init__(dim_in=dim_in or (768,768,3), dim_out=dim_out or (768,768,1),
                         filters=filters or [64, 128, 256, 512, 1024], poolings=poolings or [2, 2, 2, 2, 2],
                         preproc=ca3,downconv=ca3,downjoin=sk,downsamp=dmp,downmerge=sk,downproc=ca3,
                        upconv=sk,upjoin=ct,upsamp=uu,upmerge=ct,upproc=ca33,postproc=sk, **kwargs)

class UNet2m(UNet):
    # default 768x768 with 2 skip connections, lower case small detail. medium memory consumption with 3x3 convolution twice for output
    def __init__(self, dim_in=None, dim_out=None, filters=None, poolings=None, **kwargs):
        super(UNet2m,self).__init__(dim_in=dim_in or (768,768,3), dim_out=dim_out or (768,768,1),
                         filters=filters or [96, 192, 256, 384, 512], poolings=poolings or [2, 2, 2, 2, 2],
                         preproc=ca3, downconv=ca3, downjoin=sk, downsamp=dmp, downmerge=sk, downproc=ca3,
                         upconv=sk, upjoin=ct, upsamp=uu, upmerge=ct, upproc=ca33, postproc=sk, **kwargs)

class UNet2S(UNet):
    # default 1296x1296 with 2 skip connections, UPPER case global context. small memory consumption with 3x3 convolution only once for output
    def __init__(self, dim_in=None, dim_out=None, filters=None, poolings=None, **kwargs):
        super(UNet2S,self).__init__(dim_in=dim_in or (1296,1296,3), dim_out=dim_out or (1296,1296,1),
                         filters=filters or [48, 64, 96, 128, 196, 196, 196, 196, 256], poolings=poolings or [2, 2, 2, 2, 3, 3, 3, 3, 3],
                         preproc=ca3,downconv=ca3,downjoin=sk,downsamp=dmp,downmerge=sk,downproc=ca3,
                        upconv=sk,upjoin=ct,upsamp=uu,upmerge=ct,upproc=ca3,postproc=sk, **kwargs)

class UNet2M(UNet):
    # default 1296x1296 with 2 skip connections, UPPER case global context. medium memory consumption with 3x3 convolution twice for output
    def __init__(self, dim_in=None, dim_out=None, filters=None, poolings=None, **kwargs):
        super(UNet2M,self).__init__(dim_in=dim_in or (1296, 1296, 3), dim_out=dim_out or (1296, 1296, 1),
                         filters=filters or [64, 96, 128, 196, 256, 256, 256, 256, 256], poolings=poolings or [2, 2, 2, 2, 3, 3, 3, 3, 3],
                         preproc=ca3, downconv=ca3, downjoin=sk, downsamp=dmp, downmerge=sk, downproc=ca3,
                         upconv=sk, upjoin=ct, upsamp=uu, upmerge=ct, upproc=ca3, postproc=sk, **kwargs)
class UNet2L(UNet):
    # default 1296x1296 with 2 skip connections, UPPER case global context. large memory consumption with 3x3 convolution twice for output, once more for postproc
    def __init__(self, dim_in=None, dim_out=None, filters=None, poolings=None, **kwargs):
        super(UNet2L,self).__init__(dim_in=dim_in or (1296, 1296, 3), dim_out=dim_out or (1296, 1296, 1),
                         filters=filters or [64, 96, 128, 196, 256, 256, 256, 256, 256], poolings=poolings or [2, 2, 2, 2, 3, 3, 3, 3, 3],
                         preproc=ca3, downconv=ca3, downjoin=sk, downsamp=dmp, downmerge=sk, downproc=ca3,
                         upconv=sk, upjoin=ct, upsamp=uu, upmerge=ct, upproc=ca33, postproc=sk, **kwargs)

class SegNet(UNet):
    # SegNet zero padding downwards: conv->batchnorm->activation downsample: maxpool upwards: conv->batchnorm (no act) upsamp: upsampling activation on output layer
    # #U-shape 64,128(/2),256(/4),512(/8),512(/8),256(/4),128(/2),64,
    def __init__(self, dim_in=None, dim_out=None, filters=None, poolings=None, **kwargs):
        super(SegNet,self).__init__(dim_in=dim_in or (768, 768, 3), dim_out=dim_out or (768,768, 1),
                         filters=filters or [64, 128, 256, 512], poolings=poolings or [2, 2, 2, 2],
                         preproc=cba3, downconv=sk, downjoin=sk, downsamp=dmp, downmerge=sk, downproc=cba3,
                         upconv=cb3, upjoin=sk, upsamp=uu, upmerge=sk, upproc=cb3, postproc=sk, **kwargs)
class SegNetS(UNet):
    # SegNet 1296x1296
    def __init__(self, dim_in=None, dim_out=None, filters=None, poolings=None, **kwargs):
        super(SegNetS,self).__init__(dim_in=dim_in or (1296, 1296, 3), dim_out=dim_out or (1296, 1296, 1),
                         filters=filters or [64, 96, 128, 196, 256, 256, 256, 256, 256], poolings=poolings or [2, 2, 2, 2, 3, 3, 3, 3, 3],
                         preproc=cba3, downconv=sk, downjoin=sk, downsamp=dmp, downmerge=sk, downproc=cba3,
                         upconv=cb3, upjoin=sk, upsamp=uu, upmerge=sk, upproc=cb3, postproc=sk, **kwargs)


class ResN131(UNet):
    def __init__(self, dim_in=None, dim_out=None, filters=None, poolings=None, **kwargs):
        super(ResN131,self).__init__(dim_in=dim_in or (768, 768, 3), dim_out=dim_out or (768,768, 1),
                         filters=filters or [64, 128, 256, 512], poolings=poolings or [2, 2, 2, 2],
                         preproc=rn131r, downconv=sk, downjoin=sk, downsamp=dmp, downmerge=sk, downproc=rn131r,
                         upconv=rn131r, upjoin=sk, upsamp=uu, upmerge=sk, upproc=rn131r, postproc=sk, **kwargs)
class ResBN131(UNet):
    def __init__(self, dim_in=None, dim_out=None, filters=None, poolings=None, **kwargs):
        super(ResBN131,self).__init__(dim_in=dim_in or (768, 768, 3), dim_out=dim_out or (768,768, 1),
                         filters=filters or [64, 128, 256, 512], poolings=poolings or [2, 2, 2, 2],
                         preproc=rn131nr, downconv=sk, downjoin=sk, downsamp=dmp, downmerge=sk, downproc=rn131nr,
                         upconv=rn131nr, upjoin=sk, upsamp=uu, upmerge=sk, upproc=rn131nr, postproc=sk, **kwargs)

class ResN131S(UNet): # ResNet 1296x1296
    def __init__(self, dim_in=None, dim_out=None, filters=None, poolings=None, **kwargs):
        super(ResN131S,self).__init__(dim_in=dim_in or (1296, 1296, 3), dim_out=dim_out or (1296, 1296, 1),
                         filters=filters or [64, 96, 128, 196, 256, 256, 256, 256, 256], poolings=poolings or [2, 2, 2, 2, 3, 3, 3, 3, 3],
                         preproc=rn131r, downconv=sk, downjoin=sk, downsamp=dmp, downmerge=sk, downproc=rn131r,
                         upconv=rn131r, upjoin=sk, upsamp=uu, upmerge=sk, upproc=rn131r, postproc=sk, **kwargs)
class ResBN131S(UNet): # ResNet 1296x1296
    def __init__(self, dim_in=None, dim_out=None, filters=None, poolings=None, **kwargs):
        super(ResBN131S,self).__init__(dim_in=dim_in or (1296, 1296, 3), dim_out=dim_out or (1296, 1296, 1),
                         filters=filters or [64, 96, 128, 196, 256, 256, 256, 256, 256], poolings=poolings or [2, 2, 2, 2, 3, 3, 3, 3, 3],
                         preproc=rn131nr, downconv=sk, downjoin=sk, downsamp=dmp, downmerge=sk, downproc=rn131nr,
                         upconv=rn131nr, upjoin=sk, upsamp=uu, upmerge=sk, upproc=rn131nr, postproc=sk, **kwargs)
