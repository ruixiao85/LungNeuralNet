from __future__ import print_function

from keras.models import Model
from keras.layers import Input
from b1_net_pair import BaseNetU
from module import cvac,ca3,ca33,cb3,cba3,dmp,uu,ct,sk,db,td,uta,tu


class UDenseNet(BaseNetU):
    # also base class for U-shaped networks, Tiramisu
    def __init__(self,dim_in=None,dim_out=None,growth=None,poolings=None,preproc=None,downconv=None,downjoin=None,downsamp=None,downmerge=None,downproc=None,
                 upconv=None,upjoin=None,upsamp=None,upmerge=None,upproc=None,postproc=None,**kwargs
                 ):
        super(UDenseNet,self).__init__(dim_in=dim_in or (768, 768, 3), dim_out=dim_out or (768, 768, 1), **kwargs)
        # DenseNet 3x3 conv(48), DB(4layer) -TD-> DB(5layer) -TD-> DB(7layer) -TD-> DB(10layer) -TD-> DB(12layer) -TD-> DB(15layer,middle point)
        # DenseNet -TU-> DB(12layer) -TU-> DB(10layer) -TU-> DB(7layer) -TU-> DB(5layer) -TU-> DB(4layer) -> 1x1 conv(classes)
        self.gs=growth or [4,6,8,10,12,14] # Tiramisu
        # self.gs=growth or [2,4,6,8,6]
        self.ps=poolings or [2]*len(self.gs)
        self.preproc=preproc or ca3
        self.downconv=downconv or db
        self.downjoin=downjoin or sk
        self.downsamp=downsamp or td
        self.downmerge=downmerge or sk
        self.downproc=downproc or sk
        self.upconv=upconv or ca3
        self.upjoin=upjoin or sk  # 2nd skip
        self.upsamp=upsamp or uu  # tu
        self.upmerge=upmerge or ct  # 1st skip
        self.upproc=upproc or sk
        self.postproc=postproc or sk

    def build_net(self):
        locals()['in0']=Input((self.row_in, self.col_in, self.dep_in))
        locals()['pre0']=self.preproc(locals()['in0'], 'pre0', 0, 48, self.act) # default 48 filters
        for i in range(len(self.gs)-1):
            prev_layer=locals()['pre%d'%i] if i==0 else locals()['dproc%d'%i]
            locals()['dconv%d'%i]=self.downconv(prev_layer, 'dconv%d'%i,i,self.gs[i],self.act)
            locals()['djoin%d'%i]=self.downjoin(locals()['dconv%d'%i],prev_layer, 'djoin%d'%i,i,self.gs[i],self.act)
            locals()['dsamp%d'%(i+1)]=self.downsamp(locals()['djoin%d'%i],self.ps[i], 'dsamp%d'%(i+1),i,self.gs[i],self.act)
            locals()['dmerge%d'%(i+1)]=self.downmerge(locals()['dsamp%d'%(i+1)],prev_layer, 'dmerge%d'%(i+1),i+1,self.gs[i+1],self.act,stride=self.ps[i])
            locals()['dproc%d'%(i+1)]=self.downproc(locals()['dmerge%d'%(i+1)], 'dproc%d'%(i+1),i+1,self.gs[i+1],self.act)

        for i in range(len(self.gs)-2,-1,-1):
            prev_layer=locals()['dproc%d'%(i+1)] if i==len(self.gs)-2 else locals()['uproc%d'%(i+1)]
            locals()['uconv%d'%(i+1)]=self.upconv(prev_layer, 'uconv%d'%(i+1),i,self.gs[i+1],self.act)
            locals()['ujoin%d'%(i+1)]=self.upjoin(locals()['uconv%d'%(i+1)],locals()['dmerge%d'%(i+1)], 'ujoin%d'%(i+1),i,self.gs[i+1],
                                                  self.act)
            locals()['usamp%d'%i]=self.upsamp(locals()['ujoin%d'%(i+1)],self.ps[i], 'usamp%d'%i,i,self.gs[i+1],self.act)
            locals()['umerge%d'%i]=self.upmerge(locals()['usamp%d'%i],locals()['djoin%d'%i], 'umerge%d'%i,i,self.gs[i],self.act)
            locals()['uproc%d'%i]=self.upproc(locals()['umerge%d'%i], 'uproc%d'%i,i,self.gs[i],self.act)

        locals()['post0']=self.postproc(locals()['uproc0'], 'post0',0,self.gs[0],self.act)
        locals()['out0']=cvac(locals()['post0'], 'out0', 0, self.dep_out, self.out, size=1)
        self.net=Model(locals()['in0'], locals()['out0'])

    def __str__(self):
        return '_'.join([
            type(self).__name__,
            "%dG%d-%dP%d-%d"%(len(self.gs),self.gs[0],self.gs[-1],self.ps[0],self.ps[-1]),
            self.cap_lim_join(10, self.preproc.__name__, self.downconv.__name__,
                              self.downjoin.__name__, self.downsamp.__name__,
                              self.downmerge.__name__, self.downproc.__name__),
            self.cap_lim_join(10, self.upconv.__name__, self.upjoin.__name__,
                              self.upsamp.__name__, self.upmerge.__name__, self.upproc.__name__,
                              self.postproc.__name__),
            self.cap_lim_join(4, self.feed, self.act, self.out,
                              (self.loss if isinstance(self.loss, str) else self.loss.__name__).
                              replace('_', '').replace('loss', ''))
            +str(self.dep_out)])
