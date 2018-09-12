from keras.engine import Layer, InputSpec
from keras.models import Model
from keras.layers import Input
from keras import backend as K
import tensorflow as tf
from net.basenet import Net
from net.module import cvac, ac, cv, ca3, ca33, cb3, cba3, dmp, uu, ct, sk, accv, ad


def resize(inputs, name, scale):
    if scale==1.0: return inputs
    size=tf.round(tf.multiply(tf.to_float(tf.shape(inputs)[1:3]),tf.constant(scale)))
    return tf.image.resize_bilinear(inputs, name=name, size=size)
        # K.resize_images(inputs, name=name, size=[tf.shape(inputs)[1]*scale, tf.shape(inputs)[2]*scale])

def ResidualConvUnit(in_layer, name, idx, filters, act, stride=1, dilate=1):
    x=accv(in_layer, name+'_2', idx, filters, act, size=1, stride=stride, dilation=dilate)
    x=accv(x, name+'_1', idx, filters, act, size=3, stride=stride, dilation=dilate)
    return ad(x, in_layer, name, idx, filters)

def ChainedResidualPooling(inputs,name,idx,fs,act):
    y=ac(inputs,name+'_ac',idx,fs,act,size=1)
    x=dmp(y,1,name+'_mp',idx,fs,act,5)
    x=cv(x,name+'_cv',idx,fs,act,size=3)
    return ad(x,y,name,idx,fs,act)

def MultiResolutionFusion(hl_inputs,hl_rate,name,idx,fs,act):
    hl_inputs[0]=resize(cv(hl_inputs[0],'%s_hl%d'%(name,0),idx,fs,act,size=3),'%s_hl%d_resize'%(name,0),hl_rate[0])
    for i in range(len(hl_inputs)):
        hl_inputs[0]=ad(hl_inputs[0],
            resize(cv(hl_inputs[i],'%s_hl%d'%(name,i),idx,fs,act,size=3),'%s_hl%d_resize'%(name,i),hl_rate[i]),name+'_ad'+str(i),idx,fs,act)
    return hl_inputs[0]


class Refine(Net):
    # 10X 4X 2X 1X 0.4X 0.2X 0.1X
    def __init__(self, dim_in=None, dim_out=None, filters=None, resamples=None, steps=None, merges=None, **kwargs
                 ):
        super().__init__(dim_in=dim_in or (768, 768, 3), dim_out=dim_out or (768, 768, 1), **kwargs)
        self.fs=filters or [64, 96, 128, 192, 256, 384, 512]
        self.rs=resamples or [0.01,0.02,0.04,0.1,0.2,0.4,1.0]
        self.ss=steps or 1 # steps to move up in size after refine
        self.ms=merges or 1 # number of smaller images to merge
        locals()['in0']=Input((self.row_in, self.col_in, self.dep_in))
        for i in range(len(self.fs)):
            locals()['pool%d'%i]=resize(locals()['in0'],'pool%d'%i, self.rs[i])
            print('Shape of %d: %s'%(i,locals()['pool%d'%i].shape))
            # locals()['conv%d'%i]=cv(locals()['pool%d'%i],'conv%d'%i,0,256,self.act,size=1) # default 256
            locals()['conv%d'%i]=cv(locals()['pool%d'%i],'conv%d'%i,0,self.fs[i],self.act,size=1)
        last_idx, last_scale=0, 1.0
        for r in range(0, len(self.fs),self.ss): # refine target. double check the last step
            x=None; last_idx=r; last_scale=self.rs[r]
            for l in range(self.ms):
                i=r-l
                if i>=0:
                    rcu=ResidualConvUnit(locals()['conv%d'%i],'conv%d_r%d-%d'%(i,r,l), i, self.fs[i], self.act)
                    rcu=resize(rcu,'conv%d_r%d-%d_s'%(i,r,l),last_scale/self.rs[i])
                    x=rcu if x is None else ad(x,rcu,'conv%d_a%d-%d'%(i,r,l),i,self.fs[i],self.act)
            x=ChainedResidualPooling(x,'conv%d_crp'%r,r,self.fs[r],self.act)
            locals()['conv%d'%r]=ResidualConvUnit(x,'conv%d_crp_rcu'%r,r,self.fs[r],self.act)
        locals()['out0']=cvac(resize(locals()['conv%d'%last_idx],'last_scale',1.0/last_scale),
                              'cvac_out',last_idx,self.fs[last_idx],self.act,size=1)
        self.net=Model(locals()['in0'], locals()['out0'])


    def __str__(self):
        return '_'.join([
            type(self).__name__,
            "%dF%d-%dR%d-%dS%dM%d"%(
                len(self.fs), self.fs[0], self.fs[-1], self.rs[0], self.rs[-1], self.ss, self.ms),
            self.cap_lim_join(7, self.act, self.out,
                              (self.loss if isinstance(self.loss, str) else self.loss.__name__).
                              replace('_', '').replace('loss', ''))
            +str(self.dep_out)])
