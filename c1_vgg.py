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
    def __init__(self, dim_in=None, dim_out=None,filters=None, poolings=None,variation=None,transfer_layers=None,
            preproc=None, downconv=None, downjoin=None, downsamp=None, downmerge=None, downproc=None,
             upconv=None, upjoin=None, upsamp=None, upmerge=None, upproc=None, postproc=None, **kwargs
         ):
        super(NetU_Vgg,self).__init__(dim_in=dim_in or (768, 768, 3), dim_out=dim_out or (768, 768, 1), **kwargs)
        self.fs=filters or [64, 128, 256, 512, 1024, 1024]
        self.ps=poolings or [2]*len(self.fs)
        self.variation=variation or "vgg16"
        self.translayers=transfer_layers or 5
        self.translayers=min(min(self.translayers,len(self.fs)-1),5) # <=custom and vgg stages
        self.preproc=preproc or sk
        self.downconv=downconv or ca33
        self.downmerge=downmerge or sk # before downsize, ->1st skip connect
        self.downsamp=downsamp or dmp
        self.downjoin=downjoin or sk  # after downsize, ->2nd skip connect
        self.downproc=downproc or sk
        self.upconv=upconv or ca33
        self.upjoin=upjoin or ct  # before upsample, 2nd skip connect->
        self.upsamp=upsamp or uu
        self.upmerge=upmerge or ct  # after upsample, 1st skip connect->
        self.upproc=upproc or sk
        self.postproc=postproc or ca33

    def build_net(self):
        locals()['in0']=Input(shape=(self.row_in, self.col_in, self.dep_in))
        locals()['pre0']=self.preproc(locals()['in0'], 'pre0', 0, self.fs[0], self.act)
        creater,convs=self.config[self.variation]
        base_model=creater(input_tensor=locals()['pre0'], include_top=False) #, weights=None
        for layer in base_model.layers: layer.trainable = True # allow training on pre-trained weights

        for i in range(self.translayers):
            locals()['dmerge%d'%i]=locals()['dconv%d'%i]=base_model.get_layer("block%d_conv%d"%(i+1,convs[i])).output
            locals()['dproc%d'%(i+1)]=locals()['djoin%d'%(i+1)]=locals()['dsamp%d'%(i+1)]=base_model.get_layer("block%d_pool"%(i+1)).output

        for i in range(self.translayers,len(self.fs)-1):
            prev_layer=locals()['pre%d'%i] if i==0 else locals()['dproc%d'%i]
            locals()['dconv%d'%i]=self.downconv(prev_layer, 'dconv%d'%i, i, self.fs[i], self.act)
            locals()['dmerge%d'%i]=self.downmerge(locals()['dconv%d'%i], prev_layer, 'dmerge%d'%i, i, self.fs[i], self.act)
            locals()['dsamp%d'%(i+1)]=self.downsamp(locals()['dmerge%d'%i], self.ps[i], 'dsamp%d'%(i+1), i, self.fs[i], self.act)
            locals()['djoin%d'%(i+1)]=self.downjoin(locals()['dsamp%d'%(i+1)], prev_layer, 'djoin%d'%(i+1), i+1, self.fs[i+1], self.act, stride=self.ps[i])
            locals()['dproc%d'%(i+1)]=self.downproc(locals()['djoin%d'%(i+1)], 'dproc%d'%(i+1), i+1, self.fs[i+1], self.act)

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
            type(self).__name__+"%sT%d"%(''.join(c for c in self.variation if c.isdigit()),self.translayers),
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