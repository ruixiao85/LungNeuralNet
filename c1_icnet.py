'''
Credits: Aitor Ruano
https://github.com/aitorzip/Keras-ICNet/blob/master/model.py
'''
from keras.layers import Input
from keras.models import Model

from b1_net_pair import BaseNetU


class ICNet(BaseNetU):
    def __init__(self, filters=None, poolings=None,
                 preproc=None, downconv=None, downjoin=None, downsamp=None, downmerge=None, downproc=None,
                 upconv=None, upjoin=None, upsamp=None, upmerge=None, upproc=None, postproc=None, **kwargs
                 ):
        super(ICNet,self).__init__(kwargs)

        from module import ca3, ca33, dmp, uu, ct, sk, cvac
        self.fs=filters or [64, 128, 256, 512, 1024]
        self.ps=poolings or [2]*len(self.fs)
        self.preproc=preproc or ca3
        self.downconv=downconv or ca3
        self.downjoin=downjoin or sk
        self.downsamp=downsamp or dmp
        self.downmerge=downmerge or sk
        self.downproc=downproc or ca3
        self.upconv=upconv or sk
        self.upjoin=upjoin or sk  # 2nd skip
        self.upsamp=upsamp or uu
        self.upmerge=upmerge or ct  # 1st skip
        self.upproc=upproc or ca33
        self.postproc=postproc or sk

        locals()['in0']=Input((self.row_in, self.col_in, self.dep_in))
        locals()['pre0']=self.preproc(locals()['in0'], 'pre0', 0, self.fs[0], self.act)
        for i in range(len(self.fs)-1):
            prev_layer=locals()['pre%d'%i] if i==0 else locals()['dproc%d'%i]
            locals()['dconv%d'%i]=self.downconv(prev_layer, 'dconv%d'%i, i, self.fs[i], self.act)
            locals()['djoin%d'%i]=self.downjoin(locals()['dconv%d'%i], prev_layer, 'djoin%d'%i, i, self.fs[i], self.act)
            locals()['dsamp%d'%(i+1)]=self.downsamp(locals()['djoin%d'%i], self.ps[i], 'dsamp%d'%(i+1), i, self.fs[i],
                                                    self.act)
            locals()['dmerge%d'%(i+1)]=self.downmerge(locals()['dsamp%d'%(i+1)], prev_layer, 'dmerge%d'%(i+1), i+1,
                                                      self.fs[i+1], self.act, stride=self.ps[i])
            locals()['dproc%d'%(i+1)]=self.downproc(locals()['dmerge%d'%(i+1)], 'dproc%d'%(i+1), i+1, self.fs[i+1],
                                                    self.act)

        for i in range(len(self.fs)-2, -1, -1):
            prev_layer=locals()['dproc%d'%(i+1)] if i==len(self.fs)-2 else locals()['uproc%d'%(i+1)]
            locals()['uconv%d'%(i+1)]=self.upconv(prev_layer, 'uconv%d'%(i+1), i, self.fs[i+1], self.act)
            locals()['ujoin%d'%(i+1)]=self.upjoin(locals()['uconv%d'%(i+1)], locals()['dmerge%d'%(i+1)],
                                                  'ujoin%d'%(i+1), i, self.fs[i+1], self.act)
            locals()['usamp%d'%i]=self.upsamp(locals()['ujoin%d'%(i+1)], self.ps[i], 'usamp%d'%i, i, self.fs[i+1],
                                              self.act)
            locals()['umerge%d'%i]=self.upmerge(locals()['usamp%d'%i], locals()['djoin%d'%i], 'umerge%d'%i, i,
                                                self.fs[i], self.act)
            locals()['uproc%d'%i]=self.upproc(locals()['umerge%d'%i], 'uproc%d'%i, i, self.fs[i], self.act)

        locals()['post0']=self.postproc(locals()['uproc0'], 'post0', 0, self.fs[0], self.act)
        locals()['out0']=cvac(locals()['post0'], 'out0', 0, self.dep_out, self.out, size=1)
        self.model=Model(locals()['in0'], locals()['out0'])
        self.compile_net()

    def __str__(self):
        return '_'.join([
            self.__name__,
            "%dF%d-%dP%d-%d"%(
                len(self.fs), self.fs[0], self.fs[-1], self.ps[0], self.ps[-1]),
            # "%df%d-%dp%s" % (len(self.fs), self.fs[0], self.fs[-1], ''.join(self.pssize)),
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


# def icnet(cfg:ModelConfig, train=None, func=None):
#     train=train if train is not None else False
    # func=func if func is not None else cv # no batch norm
    # func=func if func is not None else cvbn # with batch norm
    # fs=cfg.model_filter
    # ps=cfg.model_pool
    # act=cfg.model_act
    # in0=Input((cfg.row_in, cfg.col_in, cfg.dep_in))
    # x=Lambda(lambda  x:(x-127.5)/255.0)(in0) # already normalized
    # x=in0

    # # (1/2)
    # y=Lambda(lambda x:tf.image.resize_bilinear(x, size=(int(x.shape[1])//2, int(x.shape[2])//2)), name='data_sub2')(x)
    # y=rn33(y,None,'conv1_3_3x3',-2,[32,32,64],act,stride=2)
    # y_=dmp(y,2,'pool1_3x3_s2',0)
    # y_=rn131r(y_,'conv2_1/relu',0,[32,32,128],act)
    # z =rn131r(y_,'conv3_1/relu',-2,[64,64,256],act)
    #
    # # (1/4)
    # y_=Lambda(lambda x:tf.image.resize_bilinear(x, size=(int(x.shape[1])//2, int(x.shape[2])//2)), name='conv3_1_sub4')(z)
    # y_=rn131r(y_,'conv3_4/relu',0,[64,64,256],act)
    # y_=rn131r(y_,'conv4_1/relu',-2,[128,128,512],act,dilate=2)
    # y_=rn131r(y_,'conv4_2/relu',-2,[128,128,512],act,dilate=2)
    # y_=rn131r(y_,'conv4_3/relu',-2,[128,128,512],act,dilate=2)
    # y_=rn131r(y_,'conv4_4/relu',-2,[128,128,512],act,dilate=2)
    # y_=rn131r(y_,'conv4_5/relu',-2,[128,128,512],act,dilate=2)
    # y_=rn131r(y_,'conv4_6/relu',-2,[128,128,512],act,dilate=2)
    #
    # y_=rn131r(y_,'conv5_1/relu',-2,[256,256,1024],act,dilate=4)
    # y_=rn131r(y_,'conv5_2/relu',-2,[256,256,1024],act,dilate=4)
    # y =rn131r(y_,'conv5_3/relu',-2,[256,256,1024],act,dilate=4)

    # (1/2)
    # y = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1])//2, int(x.shape[2])//2)), name='data_sub2')(x)
    # y = Conv2D(32, 3, strides=2, padding='same', activation='relu', name='conv1_1_3x3_s2')(y)
    # y = Conv2D(32, 3, padding='same', activation='relu', name='conv1_2_3x3')(y)
    # y = Conv2D(64, 3, padding='same', activation='relu', name='conv1_3_3x3')(y)
    # y_ = MaxPooling2D(pool_size=3, strides=2, name='pool1_3x3_s2')(y)
    # y = Conv2D(128, 1, name='conv2_1_1x1_proj')(y_)
    #
    # y_ = Conv2D(32, 1, activation='relu', name='conv2_1_1x1_reduce')(y_)
    # y_ = ZeroPadding2D(name='padding1')(y_)
    # y_ = Conv2D(32, 3, activation='relu', name='conv2_1_3x3')(y_)
    # y_ = Conv2D(128, 1, name='conv2_1_1x1_increase')(y_)
    # y = Add(name='conv2_1')([y,y_])
    # y_ = Activation('relu', name='conv2_1/relu')(y)
    #
    # y = Conv2D(32, 1, activation='relu', name='conv2_2_1x1_reduce')(y_)
    # y = ZeroPadding2D(name='padding2')(y)
    # y = Conv2D(32, 3, activation='relu', name='conv2_2_3x3')(y)
    # y = Conv2D(128, 1, name='conv2_2_1x1_increase')(y)
    # y = Add(name='conv2_2')([y,y_])
    # y_ = Activation('relu', name='conv2_2/relu')(y)
    #
    # y = Conv2D(32, 1, activation='relu', name='conv2_3_1x1_reduce')(y_)
    # y = ZeroPadding2D(name='padding3')(y)
    # y = Conv2D(32, 3, activation='relu', name='conv2_3_3x3')(y)
    # y = Conv2D(128, 1, name='conv2_3_1x1_increase')(y)
    # y = Add(name='conv2_3')([y,y_])
    # y_ = Activation('relu', name='conv2_3/relu')(y)
    #
    # y = Conv2D(256, 1, strides=2, name='conv3_1_1x1_proj')(y_)
    # y_ = Conv2D(64, 1, strides=2, activation='relu', name='conv3_1_1x1_reduce')(y_)
    # y_ = ZeroPadding2D(name='padding4')(y_)
    # y_ = Conv2D(64, 3, activation='relu', name='conv3_1_3x3')(y_)
    # y_ = Conv2D(256, 1, name='conv3_1_1x1_increase')(y_)
    # y = Add(name='conv3_1')([y,y_])
    # z = Activation('relu', name='conv3_1/relu')(y)
    #
    # # (1/4)
    # y_ = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1])//2, int(x.shape[2])//2)), name='conv3_1_sub4')(z)
    # y = Conv2D(64, 1, activation='relu', name='conv3_2_1x1_reduce')(y_)
    # y = ZeroPadding2D(name='padding5')(y)
    # y = Conv2D(64, 3, activation='relu', name='conv3_2_3x3')(y)
    # y = Conv2D(256, 1, name='conv3_2_1x1_increase')(y)
    # y = Add(name='conv3_2')([y,y_])
    # y_ = Activation('relu', name='conv3_2/relu')(y)
    #
    # y = Conv2D(64, 1, activation='relu', name='conv3_3_1x1_reduce')(y_)
    # y = ZeroPadding2D(name='padding6')(y)
    # y = Conv2D(64, 3, activation='relu', name='conv3_3_3x3')(y)
    # y = Conv2D(256, 1, name='conv3_3_1x1_increase')(y)
    # y = Add(name='conv3_3')([y,y_])
    # y_ = Activation('relu', name='conv3_3/relu')(y)
    #
    # y = Conv2D(64, 1, activation='relu', name='conv3_4_1x1_reduce')(y_)
    # y = ZeroPadding2D(name='padding7')(y)
    # y = Conv2D(64, 3, activation='relu', name='conv3_4_3x3')(y)
    # y = Conv2D(256, 1, name='conv3_4_1x1_increase')(y)
    # y = Add(name='conv3_4')([y,y_])
    # y_ = Activation('relu', name='conv3_4/relu')(y)
    #
    # y = Conv2D(512, 1, name='conv4_1_1x1_proj')(y_)
    # y_ = Conv2D(128, 1, activation='relu', name='conv4_1_1x1_reduce')(y_)
    # y_ = ZeroPadding2D(padding=2, name='padding8')(y_)
    # y_ = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_1_3x3')(y_)
    # y_ = Conv2D(512, 1, name='conv4_1_1x1_increase')(y_)
    # y = Add(name='conv4_1')([y,y_])
    # y_ = Activation('relu', name='conv4_1/relu')(y)
    #
    # y = Conv2D(128, 1, activation='relu', name='conv4_2_1x1_reduce')(y_)
    # y = ZeroPadding2D(padding=2, name='padding9')(y)
    # y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_2_3x3')(y)
    # y = Conv2D(512, 1, name='conv4_2_1x1_increase')(y)
    # y = Add(name='conv4_2')([y,y_])
    # y_ = Activation('relu', name='conv4_2/relu')(y)
    #
    # y = Conv2D(128, 1, activation='relu', name='conv4_3_1x1_reduce')(y_)
    # y = ZeroPadding2D(padding=2, name='padding10')(y)
    # y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_3_3x3')(y)
    # y = Conv2D(512, 1, name='conv4_3_1x1_increase')(y)
    # y = Add(name='conv4_3')([y,y_])
    # y_ = Activation('relu', name='conv4_3/relu')(y)
    #
    # y = Conv2D(128, 1, activation='relu', name='conv4_4_1x1_reduce')(y_)
    # y = ZeroPadding2D(padding=2, name='padding11')(y)
    # y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_4_3x3')(y)
    # y = Conv2D(512, 1, name='conv4_4_1x1_increase')(y)
    # y = Add(name='conv4_4')([y,y_])
    # y_ = Activation('relu', name='conv4_4/relu')(y)
    #
    # y = Conv2D(128, 1, activation='relu', name='conv4_5_1x1_reduce')(y_)
    # y = ZeroPadding2D(padding=2, name='padding12')(y)
    # y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_5_3x3')(y)
    # y = Conv2D(512, 1, name='conv4_5_1x1_increase')(y)
    # y = Add(name='conv4_5')([y,y_])
    # y_ = Activation('relu', name='conv4_5/relu')(y)
    #
    # y = Conv2D(128, 1, activation='relu', name='conv4_6_1x1_reduce')(y_)
    # y = ZeroPadding2D(padding=2, name='padding13')(y)
    # y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_6_3x3')(y)
    # y = Conv2D(512, 1, name='conv4_6_1x1_increase')(y)
    # y = Add(name='conv4_6')([y,y_])
    # y = Activation('relu', name='conv4_6/relu')(y)
    #
    # y_ = Conv2D(1024, 1, name='conv5_1_1x1_proj')(y)
    # y = Conv2D(256, 1, activation='relu', name='conv5_1_1x1_reduce')(y)
    # y = ZeroPadding2D(padding=4, name='padding14')(y)
    # y = Conv2D(256, 3, dilation_rate=4, activation='relu', name='conv5_1_3x3')(y)
    # y = Conv2D(1024, 1, name='conv5_1_1x1_increase')(y)
    # y = Add(name='conv5_1')([y,y_])
    # y_ = Activation('relu', name='conv5_1/relu')(y)
    #
    # y = Conv2D(256, 1, activation='relu', name='conv5_2_1x1_reduce')(y_)
    # y = ZeroPadding2D(padding=4, name='padding15')(y)
    # y = Conv2D(256, 3, dilation_rate=4, activation='relu', name='conv5_2_3x3')(y)
    # y = Conv2D(1024, 1, name='conv5_2_1x1_increase')(y)
    # y = Add(name='conv5_2')([y,y_])
    # y_ = Activation('relu', name='conv5_2/relu')(y)
    #
    # y = Conv2D(256, 1, activation='relu', name='conv5_3_1x1_reduce')(y_)
    # y = ZeroPadding2D(padding=4, name='padding16')(y)
    # y = Conv2D(256, 3, dilation_rate=4, activation='relu', name='conv5_3_3x3')(y)
    # y = Conv2D(1024, 1, name='conv5_3_1x1_increase')(y)
    # y = Add(name='conv5_3')([y,y_])
    # y = Activation('relu', name='conv5_3/relu')(y)
    #
    # h, w=y.shape[1:3].as_list()
    # pool1=AveragePooling2D(pool_size=(h, w), strides=(h, w), name='conv5_3_pool1')(y)
    # pool1=Lambda(lambda x:tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool1_interp')(pool1)
    # pool2=AveragePooling2D(pool_size=(h/2, w/2), strides=(h//2, w//2), name='conv5_3_pool2')(y)
    # pool2=Lambda(lambda x:tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool2_interp')(pool2)
    # pool3=AveragePooling2D(pool_size=(h/3, w/3), strides=(h//3, w//3), name='conv5_3_pool3')(y)
    # pool3=Lambda(lambda x:tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool3_interp')(pool3)
    # pool6=AveragePooling2D(pool_size=(h/4, w/4), strides=(h//4, w//4), name='conv5_3_pool6')(y)
    # pool6=Lambda(lambda x:tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool6_interp')(pool6)
    #
    # y=Add(name='conv5_3_sum')([y, pool1, pool2, pool3, pool6])
    # y=Conv2D(256, 1, activation='relu', name='conv5_4_k1')(y)
    # aux_1=Lambda(lambda x:tf.image.resize_bilinear(x, size=(int(x.shape[1])*2, int(x.shape[2])*2)), name='conv5_4_interp')(y)
    # y=ZeroPadding2D(padding=2, name='padding17')(aux_1)
    # y=Conv2D(128, 3, dilation_rate=2, name='conv_sub4')(y)
    # y_=Conv2D(128, 1, name='conv3_1_sub2_proj')(z)
    # y=Add(name='sub24_sum')([y, y_])
    # y=Activation('relu', name='sub24_sum/relu')(y)
    #
    # aux_2=Lambda(lambda x:tf.image.resize_bilinear(x, size=(int(x.shape[1])*2, int(x.shape[2])*2)), name='sub24_sum_interp')(y)
    # y=ZeroPadding2D(padding=2, name='padding18')(aux_2)
    # y_=Conv2D(128, 3, dilation_rate=2, name='conv_sub2')(y)
    #
    # # (1)
    # y=Conv2D(32, 3, strides=2, padding='same', activation='relu', name='conv1_sub1')(x)
    # y=Conv2D(32, 3, strides=2, padding='same', activation='relu', name='conv2_sub1')(y)
    # y=Conv2D(64, 3, strides=2, padding='same', activation='relu', name='conv3_sub1')(y)
    # y=Conv2D(128, 1, name='conv3_sub1_proj')(y)
    #
    # y=Add(name='sub12_sum')([y, y_])
    # y=Activation('relu', name='sub12_sum/relu')(y)
    # y=Lambda(lambda x:tf.image.resize_bilinear(x, size=(int(x.shape[1])*2, int(x.shape[2])*2)), name='sub12_sum_interp')(y)
    #
    # out0=Conv2D(cfg.dep_out, 1, activation=cfg.model_out, name='conv6_cls')(y)
    #
    # if train:
    #     aux_1=Conv2D(cfg.dep_out, 1, activation=cfg.model_out, name='sub4_out')(aux_1)
    #     aux_2=Conv2D(cfg.dep_out, 1, activation=cfg.model_out, name='sub24_out')(aux_2)
    #     model=Model(inputs=in0, outputs=[out0, aux_2, aux_1])
    # else:
    #     model=Model(inputs=in0, outputs=out0)
    #
    # return model