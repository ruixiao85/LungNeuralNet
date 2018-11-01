from keras.engine import Layer, InputSpec
from keras.models import Model
from keras.layers import Input,Lambda
from keras import backend as K
import tensorflow as tf
from net.basenetu import BaseNetU
from net.module import cvac, ac, cv, ca3, ca33, cb3, cba3, dmp, uu, ct, sk, accv, ad
from net.resize_layer import ResizeImages


def resize(inputs, name, target_size):
    row=K.int_shape(inputs)[1]
    if row==target_size: return inputs
    return ResizeImages(output_dim=(target_size,target_size),data_format='channels_last',name=name)(inputs)

def residual_conv_unit(in_layer, name, idx, filters, act, stride=1, dilate=1,rep=1):
    y=x=in_layer
    for r in range(rep-1,-1,-1):
        app=str(r)[: r]
        y=accv(y, name+app+'_2', idx, filters, act, size=3, stride=stride, dilation=dilate)
        y=accv(y, name+app+'_1', idx, filters, act, size=3, stride=stride, dilation=dilate)
        x=ad(x,y,name+app,idx,filters)
    return x

def res_conv_unit(in_layer, name, idx, filters, act, stride=1, dilate=1,rep=1):
    x=in_layer
    for r in range(rep-1,-1,-1):
        app=str(r)[: r]
        y=accv(x, name+app+'_2', idx, filters, act, size=3, stride=stride, dilation=dilate)
        y=accv(y, name+app+'_1', idx, filters, act, size=3, stride=stride, dilation=dilate)
        x=ad(x,y,name+app,idx,filters)
    return x

def chained_residual_pooling(inputs, name, idx, fs, act,rep=1):
    y=x=ac(inputs,name+'_ac',idx,fs,act,size=1)
    for r in range(rep-1,-1,-1):
        app=str(r)[: r]
        y=dmp(y,1,name+app+'_mp',idx,fs,act,5)
        y=cv(y,name+app+'_cv',idx,fs,act,size=3)
        x=ad(x,y,name+app,idx,fs,act)
    return x


class Refine(BaseNetU):
    #  40X  20X  10X  4X  2X  1X .4X .2X .1X
    # 4000,2000,1000,400,200,100, 40, 20, 10

    def __init__(self, dim_in=None, dim_out=None, sizes=None, filters=None, steps=None, merges=None, **kwargs
                 ):
        # super(Refine,self).__init__(dim_in=dim_in or (1000,1000, 3), dim_out=dim_out or (1000,1000, 1), **kwargs)
        # self.sizes=sizes or [10,20,40,100,200,400]
        super(Refine,self).__init__(dim_in=dim_in or (1200,1200, 3), dim_out=dim_out or (1200,1200, 1), **kwargs)
        self.sizes=sizes or [8,20,40,80,200,400]
        # self.sizes=sizes or [40,400]
        self.filters=filters or 128
        self.steps=steps or [1,2,3,4,5] # steps to move up in size after refine
        self.merge=merges or 2 # number of images to merge
        locals()['in0']=Input((self.row_in, self.col_in, self.dep_in))

        last_layer=[None]*len(self.sizes)
        for i in range(len(self.sizes)):
            locals()['pool%d'%i]=resize(locals()['in0'], 'pool%d'%i, self.sizes[i])
            print('Shape of %d: %s'%(i,locals()['pool%d'%i].shape))
            last_layer[i]=locals()['conv%d'%i]=cvac(locals()['pool%d'%i],'conv%d'%i,0,self.filters,self.act,size=1)
        last_idx, target_size=0, self.sizes[-1]
        for r in self.steps: # refine target. double check the last step
            last_idx=r; target_size=self.sizes[r]
            for l in range(self.merge):
                i=r-l
                if i>=0:
                    if 'rcu%d'%i not in locals(): locals()['rcu%d'%i]=residual_conv_unit(last_layer[i], 'rcu%d'%i, i, self.filters, self.act,rep=1)
                    if 'upconv%d'%i not in locals(): locals()['upconv%d'%i]=cvac(locals()['rcu%d'%i],'upconv%d'%i,0,self.filters,self.act,size=3)
                    if 'upsamp%d-%d'%(i,r) not in locals(): locals()['upsamp%d-%d'%(i,r)]=resize(locals()['upconv%d'%i],'upsamp%d-%d'%(i,r),target_size)
                    if 'mrf%d'%r not in locals():
                        locals()['mrf%d'%r]=locals()['upsamp%d-%d'%(i,r)]
                    else:
                        locals()['mrf%d'%r]=ad(locals()['mrf%d'%r],locals()['upsamp%d-%d'%(i,r)],'mrf%d'%i,i,self.filters,self.act)
            locals()['crp%d'%r]=chained_residual_pooling(locals()['mrf%d'%r], 'crp%d'%r, r, self.filters, self.act,rep=1)
            last_layer[r]=locals()['outconv%d'%r]=residual_conv_unit(locals()['crp%d'%r], 'outconv%d'%r, r, self.filters, self.act,rep=1)
        locals()['out_scale']=resize(locals()['outconv%d'%last_idx],'out_scale',self.row_out)
        locals()['out0']=cvac(locals()['out_scale'],'out0',last_idx,self.dep_out,self.out,size=1) # self.fs[last_idx]
        self.net=Model(locals()['in0'], locals()['out_scale'])
        self.compile_net()

    def __str__(self):
        return '_'.join([
            type(self).__name__,
            "%dF%dS%d-%dT%d-%dM%d"%(
                len(self.sizes), self.filters, self.sizes[0], self.sizes[-1], self.steps[0], self.steps[-1], self.merge),
            self.cap_lim_join(4, self.feed, self.act, self.out,
                              (self.loss if isinstance(self.loss, str) else self.loss.__name__).
                              replace('_', '').replace('loss', ''))
            +str(self.dep_out)])
