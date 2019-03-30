from keras.engine.training import Model
from keras.layers import Input
from keras import backend as K
from keras.applications import vgg16,vgg19

from b1_net_pair import BaseNetU
from module import cvac,cb3,cba3,dmp,uu,sk,ca33,ca3,ct

#VGG 16 weighted layers (224 x 224 x 3 RGB)
# conv3x3-64 conv3x3-64 maxpool
# conv3x3-128 conv3x3-128 maxpool
# conv3x3-256 conv3x3-256 conv3x3/1x1-256 maxpool
# conv3x3-512 conv3x3-512 conv3x3/1x1-512 maxpool
# conv3x3-512 conv3x3-512 conv3x3/1x1-512 maxpool
# FC-4096 -> FC-4096 -> FC-1000 -> softmax
class NetU_Vgg(BaseNetU):
    config={
        'vgg16':(vgg16.VGG16,[2,2,3,3,3]),
        'vgg19':(vgg19.VGG19,[2,2,4,4,4]),
    }
    def __init__(self,**kwargs):
        super(NetU_Vgg,self).__init__(**kwargs)
        self.fs=kwargs.get('filters',[64,128,256,512,768,1024])
        self.ps=kwargs.get('poolings', [2]*len(self.fs))
        self.variation=kwargs.get('variation', "vgg16")
        self.preproc=kwargs.get('preproc', sk)
        # self.downconv=kwargs.get('downconv', ca33)
        # self.downmerge=kwargs.get('downmerge', sk) # before downsize, ->1st skip connect
        # self.downsamp=kwargs.get('downsamp', dmp)
        # self.downjoin=kwargs.get('downjoin', sk) # after downsize, ->2nd skip connect
        # self.downproc=kwargs.get('downproc', sk)
        self.upconv=kwargs.get('upconv', ca3)
        self.upjoin=kwargs.get('upjoin', ct) # before upsample, 2nd skip connect->
        self.upsamp=kwargs.get('upsamp', uu)
        self.upmerge=kwargs.get('upmerge', ct) # after upsample, 1st skip connect->
        self.upproc=kwargs.get('upproc', ca3)
        self.postproc=kwargs.get('postproc', ca3)

    def build_net(self,is_train):
        super(NetU_Vgg,self).build_net(is_train)
        locals()['in0']=Input(shape=(self.row_in, self.col_in, self.dep_in))
        locals()['pre0']=self.preproc(locals()['in0'], 'pre0', 0, self.fs[0], self.act)
        creater,convs=self.config[self.variation]
        base_model=creater(input_tensor=locals()['pre0'], include_top=False, weights='imagenet' if self.pre_trained else None)
        # print(base_model.summary())
        for layer in base_model.layers: layer.trainable = True # allow training on pre-trained weights

        for i in range(5):
            locals()['dmerge%d'%i]=locals()['dconv%d'%i]=base_model.get_layer("block%d_conv%d"%(i+1,convs[i])).output
            locals()['dproc%d'%(i+1)]=locals()['djoin%d'%(i+1)]=locals()['dsamp%d'%(i+1)]=base_model.get_layer("block%d_pool"%(i+1)).output

        for i in range(len(self.fs)-2, -1, -1):
            prev_layer=locals()['dproc%d'%(i+1)] if i==len(self.fs)-2 else locals()['uproc%d'%(i+1)]
            locals()['uconv%d'%(i+1)]=self.upconv(prev_layer, 'uconv%d'%(i+1), i, self.fs[i+1], self.act)
            locals()['ujoin%d'%(i+1)]=self.upjoin(locals()['uconv%d'%(i+1)], locals()['djoin%d'%(i+1)], 'ujoin%d'%(i+1), i, self.fs[i+1], self.act)
            locals()['usamp%d'%i]=self.upsamp(locals()['ujoin%d'%(i+1)], self.ps[i], 'usamp%d'%i, i, self.fs[i+1], self.act)
            locals()['umerge%d'%i]=self.upmerge(locals()['usamp%d'%i], locals()['dmerge%d'%i], 'umerge%d'%i, i, self.fs[i], self.act)
            locals()['uproc%d'%i]=self.upproc(locals()['umerge%d'%i], 'uproc%d'%i, i, self.fs[i], self.act)

        locals()['post0']=self.postproc(locals()['uproc0'], 'post0', 0, self.fs[0], self.act)
        locals()['out0']=cvac(locals()['post0'], 'out0', 0, self.dep_out, self.out, size=1)
        self.net=Model(locals()['in0'], locals()['out0'])

    def __str__(self):
        return '_'.join([
            type(self).__name__+''.join(c for c in self.variation if c.isdigit()),
            "%dF%d-%dP%d-%d"%(
            len(self.fs), self.fs[0], self.fs[-1], self.ps[0], self.ps[-1]),
            # "%df%d-%dp%s" % (len(self.fs), self.fs[0], self.fs[-1], ''.join(self.pssize)),
            # self.cap_lim_join(10, self.preproc.__name__, self.downconv.__name__,
            #                   self.downmerge.__name__, self.downsamp.__name__,
            #                   self.downjoin.__name__, self.downproc.__name__),
            self.cap_lim_join(10, self.upconv.__name__, self.upjoin.__name__,
                              self.upsamp.__name__, self.upmerge.__name__, self.upproc.__name__,
                              self.postproc.__name__),
            self.cap_lim_join(4, self.feed, self.act, self.out,
                              (self.loss if isinstance(self.loss, str) else self.loss.__name__).
                              replace('_', '').replace('loss', ''))
            +str(self.dep_out)])