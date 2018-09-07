from __future__ import print_function

from keras.engine import Layer, InputSpec
from keras.models import Model
from keras.layers import Input
from keras import backend as K
import tensorflow as tf
from model_config import ModelConfig
from module import cvac


def unet(cfg: ModelConfig):
    fs=cfg.model_filter
    ps=cfg.model_pool
    locals()['in0']=Input((cfg.row_in, cfg.col_in, cfg.dep_in))
    locals()['pre0']=cfg.model_preproc(locals()['in0'],'pre0',0,fs[0],cfg.model_act)
    for i in range(len(fs)-1):
        prev_layer=locals()['pre%d'%i] if i==0 else locals()['dproc%d'%i]
        locals()['dconv%d'%i]=cfg.model_downconv(prev_layer, 'dconv%d'%i, i, fs[i], cfg.model_act)
        locals()['djoin%d'%i]=cfg.model_downjoin(locals()['dconv%d'%i], prev_layer, 'djoin%d'%i, i, fs[i], cfg.model_act)
        locals()['dsamp%d'%(i+1)]=cfg.model_downsamp(locals()['djoin%d'%i], ps[i], 'dsamp%d'%(i+1), i, fs[i], cfg.model_act)
        locals()['dmerge%d'%(i+1)]=cfg.model_downmerge(locals()['dsamp%d'%(i+1)], prev_layer, 'dmerge%d'%(i+1), i+1, fs[i+1], cfg.model_act, stride=ps[i])
        locals()['dproc%d'%(i+1)]=cfg.model_downproc(locals()['dmerge%d'%(i+1)], 'dproc%d'%(i+1), i+1, fs[i+1], cfg.model_act)

    for i in range(len(fs)-2, -1, -1):
        prev_layer=locals()['dproc%d'%(i+1)] if i==len(fs)-2 else locals()['uproc%d'%(i+1)]
        locals()['uconv%d'%(i+1)]=cfg.model_upconv(prev_layer, 'uconv%d'%(i+1), i, fs[i+1], cfg.model_act)
        locals()['ujoin%d'%(i+1)]=cfg.model_upjoin(locals()['uconv%d'%(i+1)], locals()['dmerge%d'%(i+1)], 'ujoin%d'%(i+1), i, fs[i+1], cfg.model_act)
        locals()['usamp%d'%i]=cfg.model_upsamp(locals()['ujoin%d'%(i+1)], ps[i], 'usamp%d'%i, i, fs[i+1], cfg.model_act)
        locals()['umerge%d'%i]=cfg.model_upmerge(locals()['usamp%d'%i], locals()['djoin%d'%i], 'umerge%d'%i, i, fs[i], cfg.model_act)
        locals()['uproc%d'%i]=cfg.model_upproc(locals()['umerge%d'%i], 'uproc%d'%i, i, fs[i], cfg.model_act)

    locals()['post0']=cfg.model_postproc(locals()['uproc0'],'post0',0,fs[0],cfg.model_act)
    locals()['out0']=cvac(locals()['post0'], 'out0', 0, cfg.dep_out, cfg.model_out, size=1)
    return Model(locals()['in0'], locals()['out0'])
