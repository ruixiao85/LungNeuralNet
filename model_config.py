import colorsys
import random
import numpy as np


def generate_colors(n, shuffle=False):
    hsv = [(i / n, 1.0, 0.85) for i in range(n)]
    colors = [tuple((255*np.array(col)).astype(np.uint8)) for col in map(lambda c: colorsys.hsv_to_rgb(*c), hsv)]
    if shuffle:
        random.shuffle(colors)
    return colors


class ModelConfig:

    def __init__(self, dim_in=None, dim_out=None, num_targets=None, image_format=None, image_resize=None, image_padding=None, mask_color=None,
                 coverage_tr=None, coverage_prd=None, batch_size=None, separate=None,
                 model_name=None, model_filter=None, model_pool=None,
                model_preproc=None, model_downconv=None, model_downjoin=None, model_downsamp=None, model_downmerge=None, model_downproc=None,
                model_upconv=None, model_upjoin=None, model_upsamp=None, model_upmerge=None, model_upproc=None, model_postproc=None,
                 model_act=None, model_out=None, model_loss=None, metrics=None, optimizer=None,
                 call_hardness=None, overlay_color=None, overlay_opacity=None, predict_size=None, predict_all_inclusive=None,
                 train_rep=None, train_epoch=None, train_step=None, train_vali_step=None,
                 train_vali_split=None, train_aug=None, train_continue=None,
                 train_shuffle=None, train_indicator=None
                 ):
        self.row_in, self.col_in, self.dep_in = dim_in or (512,512,3)
        self.row_out, self.col_out, self.dep_out = dim_out or (512,512,1)
        self.num_targets = num_targets or 10 # lead to default overlay_color predict_group
        self.image_format=image_format or "*.jpg"
        self.image_resize= image_resize or 1.0  # default 1.0, reduce size <=1.0
        self.image_padding= image_padding or 1.0  # default 1.0, padding proportionally >=1.0
        self.mask_color=mask_color or "white"  # green/white
        self.separate = separate if separate is not None else True  # True: split into multiple smaller views; False: take one view only
        self.coverage_train = coverage_tr or 2.0
        self.coverage_predict = coverage_prd or 3.0
        from net.unet import unet
        from net.module import ca3, ca33, dmp, uu, ct, sk
        self.model_name =model_name or unet
        self.model_filter = model_filter or [64, 128, 256, 512, 1024]
        self.model_pool =model_pool or [2]*len(self.model_filter)
        self.model_preproc=model_preproc or ca3
        self.model_downconv =model_downconv or ca3
        self.model_downjoin =model_downjoin or sk
        self.model_downsamp =model_downsamp or dmp
        self.model_downmerge =model_downmerge or sk
        self.model_downproc =model_downproc or ca3
        self.model_upconv =model_upconv or sk
        self.model_upjoin =model_upjoin or sk # default not to join here
        self.model_upsamp =model_upsamp or uu
        self.model_upmerge =model_upmerge or ct
        self.model_upproc =model_upproc or ca33
        self.model_postproc=model_postproc or sk
        from metrics import jac,dice,dice67,dice33,acc,acc67,acc33,\
            loss_bce_dice, custom_function_keras
        custom_function_keras() # leakyrelu, swish
        self.model_act = model_act or 'elu'
        self.model_out = model_out or ('sigmoid' if self.dep_out == 1 else 'softmax')
        self.model_loss = model_loss or (loss_bce_dice if self.dep_out == 1 else 'categorical_crossentropy')  # 'binary_crossentropy'
        self.metrics= metrics or ([jac, dice, dice67, dice33] if self.dep_out == 1 else [acc, acc67, acc33])
        from keras.optimizers import SGD,RMSprop,Adam,Nadam
        self.optimizer =optimizer or Adam(1e-5)
        self.call_hardness = call_hardness or 1.0  # 0-smooth 1-hard binary call
        self.overlay_color = overlay_color if isinstance(overlay_color, list) else\
                            generate_colors(overlay_color) if isinstance(overlay_color, int) else\
                            generate_colors(self.num_targets)
        self.overlay_opacity = overlay_opacity or 0.2
        self.predict_size = predict_size or num_targets
        self.predict_all_inclusive = predict_all_inclusive if predict_all_inclusive is not None else False
        self.batch_size = batch_size or 1
        self.train_rep = train_rep or 3  # times to repeat during training
        self.train_epoch = train_epoch or 12  # max epoches during training
        self.train_step = train_step or 120
        self.train_vali_step = train_vali_step or 60
        self.train_vali_split = train_vali_split or 0.33
        self.train_aug = train_aug or 1  # only to training set, not validation or prediction mode, 0-disabled higher the more aug
        self.train_shuffle = train_shuffle if train_shuffle is not None else True  # only to training set, not validation or prediction mode
        self.train_continue = train_continue if train_continue is not None else True  # continue training by loading previous weights
        self.train_indicator = train_indicator or ('val_dice' if self.dep_out == 1 else 'val_acc')  # indicator to maximize

    @staticmethod
    def cap_lim_join(lim,*text):
        test_list=[t.capitalize()[:lim] for t in text]
        return ''.join(test_list)

    def __str__(self):
        return '_'.join([
            self.model_name.__name__.capitalize(),
            "%dF%d-%dP%d-%d" % (len(self.model_filter), self.model_filter[0], self.model_filter[-1], self.model_pool[0], self.model_pool[-1]),
            # "%df%d-%dp%s" % (len(self.model_filter), self.model_filter[0], self.model_filter[-1], ''.join(self.model_poolsize)),
            self.cap_lim_join(10, self.model_preproc.__name__, self.model_downconv.__name__, self.model_downjoin.__name__, self.model_downsamp.__name__, self.model_downmerge.__name__,self.model_downproc.__name__),
            self.cap_lim_join(10, self.model_upconv.__name__, self.model_upjoin.__name__, self.model_upsamp.__name__,self.model_upmerge.__name__, self.model_upproc.__name__, self.model_postproc.__name__),
            self.cap_lim_join(7,self.model_act, self.model_out,
                (self.model_loss if isinstance(self.model_loss, str) else self.model_loss.__name__).replace('_','').replace('loss',''))
            +str(self.dep_out)])

    def __repr__(self):
        return '_'.join([
            self.model_name.__name__.capitalize(),
            "%dF%d-%dP%d-%d" % (len(self.model_filter), self.model_filter[0], self.model_filter[-1], self.model_pool[0], self.model_pool[-1]),
            # "%df%d-%dp%s" % (len(self.model_filter), self.model_filter[0], self.model_filter[-1], ''.join(self.model_poolsize)),
            self.cap_lim_join(10, self.model_preproc.__name__, self.model_downconv.__name__, self.model_downjoin.__name__, self.model_downsamp.__name__, self.model_downmerge.__name__,self.model_downproc.__name__),
            self.cap_lim_join(10, self.model_upconv.__name__, self.model_upjoin.__name__, self.model_upsamp.__name__,self.model_upmerge.__name__, self.model_upproc.__name__, self.model_postproc.__name__),
            self.cap_lim_join(7,self.model_act, self.model_out,
                (self.model_loss if isinstance(self.model_loss, str) else self.model_loss.__name__).replace('_','').replace('loss',''))
            +str(self.dep_out),
            str(self.predict_size)+('A' if self.predict_all_inclusive else 'I')
        ])
