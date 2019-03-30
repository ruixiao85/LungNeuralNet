from __future__ import print_function

from keras.models import Model
from keras.layers import Input
from b1_net_pair import BaseNetU
from c0_backbones import xcept, incept3, incepres2, v16, v19, res50, densenet121, densenet169, densenet201, mobile, mobile2, nasmobile, naslarge
from module import cvac, ca3, ca33, cb3, cba3, dmp, uu, ct, sk


class NetUBack(BaseNetU):
    # also base class for U-shaped networks
    def __init__(self, **kwargs):
        super(NetUBack,self).__init__(**kwargs)
        # UNET valid padding 572,570,568->284,282,280->140,138,136->68,66,64->32,30,28->56,54,52->104,102,100->200,198,196->392,390,388 388/572=67.8322% center
        # UNET same padding 576->288->144->72->36->72->144->288->576 take central 68% =392
        from c2_backbones import v16
        self.backbone=kwargs.get('backbone', v16)
        self.freeze_bn=kwargs.get('freeze_bn', False)
        self.fs=kwargs.get('filters', [64, 128, 256, 512, 512, 512])
        self.ps=kwargs.get('poolings', [2]*len(self.fs))
        self.upconv=kwargs.get('upconv', ca3)
        self.upjoin=kwargs.get('upjoin', ct)
        self.upsamp=kwargs.get('upsamp', uu)
        self.upproc=kwargs.get('upproc', sk)
        self.postproc=kwargs.get('postproc', ca3)

    def build_net(self,is_train):
        super(NetUBack,self).build_net(is_train)
        locals()['in0']=Input((self.row_in, self.col_in, self.dep_in))
        # locals()['pre0']=self.preproc(locals()['in0'],'pre0',0,self.fs[0],self.act)
        # for i in range(len(self.fs)-1):
        #     prev_layer=locals()['pre%d'%i] if i==0 else locals()['dproc%d'%i]
        #     locals()['dconv%d'%i]=self.downconv(prev_layer, 'dconv%d'%i, i, self.fs[i], self.act)
        #     locals()['djoin%d'%i]=self.downjoin(locals()['dconv%d'%i], prev_layer, 'djoin%d'%i, i, self.fs[i], self.act)
        #     locals()['dsamp%d'%(i+1)]=self.downsamp(locals()['djoin%d'%i], self.ps[i], 'dsamp%d'%(i+1), i, self.fs[i], self.act)
        #     locals()['dmerge%d'%(i+1)]=self.downmerge(locals()['dsamp%d'%(i+1)], prev_layer, 'dmerge%d'%(i+1), i+1, self.fs[i+1], self.act, stride=self.ps[i])
        #     locals()['dproc%d'%(i+1)]=self.downproc(locals()['dmerge%d'%(i+1)], 'dproc%d'%(i+1), i+1, self.fs[i+1], self.act)
        locals()['join1'],locals()['join2'],locals()['join3'],locals()['join4'],locals()['join5']=self.backbone(locals()['in0'])

        for i in range(len(self.fs)-2, -1, -1):
            prev_layer = locals()['join%d'%(i+1)] if i==len(self.fs)-2 else locals()['uproc%d'%(i+1)]
            locals()['uconv%d'%(i+1)]=self.upconv(prev_layer, 'uconv%d'%(i+1), i, self.fs[i+1], self.act)
            locals()['ujoin%d'%(i+1)]=self.upjoin(locals()['uconv%d'%(i+1)],locals()['join%d'%(i+1)],'ujoin%d'%(i+1),i,self.fs[i+1],self.act)
            locals()['usamp%d'%i]=self.upsamp(locals()['uconv%d'%(i+1)], self.ps[i], 'usamp%d'%i, i, self.fs[i+1], self.act)
            locals()['uproc%d'%i]=self.upproc(locals()['usamp%d'%i], 'uproc%d'%i, i, self.fs[i], self.act)
    
        locals()['post0']=self.postproc(locals()['uproc0'],'post0',0,self.fs[0],self.act)
        locals()['out0']=cvac(locals()['post0'], 'out0', 0, self.dep_out, self.out, size=1)
        self.net=Model(locals()['in0'], locals()['out0'])

        if self.freeze_bn:
            print("Freeze All Batch Normalization Layers",sep=" ")
            for layer in self.net.layers:
                class_name=layer.__class__.__name__
                if class_name=='BatchNormalization':
                    layer.trainable=False
                    print('+',sep="")

    def __str__(self):
        return '_'.join([
            type(self).__name__,
            "%dF%d-%dP%d-%d"%(
            len(self.fs), self.fs[0], self.fs[-1], self.ps[0], self.ps[-1]),
            # "%df%d-%dp%s" % (len(self.fs), self.fs[0], self.fs[-1], ''.join(self.pssize)),
            self.cap_lim_join(10, self.upconv.__name__, self.upjoin.__name__,
                              self.upsamp.__name__, self.upproc.__name__,
                              self.postproc.__name__),
            self.cap_lim_join(4, self.feed, self.act, self.out,
                              (self.loss if isinstance(self.loss, str) else self.loss.__name__).
                              replace('_', '').replace('loss', ''))
            +str(self.dep_out)])
