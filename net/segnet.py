from __future__ import print_function

from keras.engine import Layer, InputSpec
from keras.models import Model
from keras.layers import Input
from keras import backend as K
import tensorflow as tf
from net.basenet import Net
from net.module import cvac

class SegNet(Net):
    def __init__(self, filters=None, poolings=None,
                 preproc=None, downconv=None,downjoin=None,downsamp=None,downmerge=None,downproc=None,
                 upconv=None, upjoin=None, upsamp=None, upmerge=None, upproc=None, postproc=None, **kwargs
        ):
        super().__init__(**kwargs)
        # SegNet zero padding downwards: conv->batchnorm->activation downsample: maxpool upwards: conv->batchnorm (no act) upsamp: upsampling activation on output layer
        # #U-shape 64,128(/2),256(/4),512(/8),512(/8),256(/4),128(/2),64,
        from net.module import ca3, ca33, cba3, cb3, dmp, uu, ct, sk
        self.fs=filters or [64, 128, 256, 512]
        self.ps=poolings or [2]*len(self.fs)
        self.preproc=preproc or cba3
        self.downconv=downconv or sk
        self.downjoin=downjoin or sk
        self.downsamp=downsamp or dmp
        self.downmerge=downmerge or sk
        self.downproc=downproc or cba3
        self.upconv=upconv or cb3
        self.upjoin=upjoin or sk # 2nd no skip
        self.upsamp=upsamp or uu
        self.upmerge=upmerge or sk # 1st no skip
        self.upproc=upproc or cb3
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

    def __str__(self):
        return '_'.join([
            type(self).__name__,
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

