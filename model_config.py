import colorsys
import random
import numpy as np


def generate_colors(n, shuffle=False):
    hsv = [(i / n, 1, 0.5) for i in range(n)] # last number = brightness
    # colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = [tuple((255*np.array(col)).astype(np.uint8)) for col in map(lambda c: colorsys.hsv_to_rgb(*c), hsv)]
    if shuffle:
        random.shuffle(colors)
    return colors


class ModelConfig:

    def __init__(self, dim_in=None, dim_out=None, image_format=None, image_resize=None, image_padding=None, mask_color=None,
                 coverage_tr=None, coverage_prd=None, batch_size=None, separate=None,
                 model_filter=None, model_kernel=None,
                 model_act=None, model_out=None, model_loss=None, metrics=None,
                 overlay_color=None, overlay_opacity=None, call_hardness=None,
                 train_rep=None, train_epoch=None, train_step=None, train_vali_step=None,
                 train_learning_rate=None, train_vali_split=None, train_aug=None, train_continue=None,
                 train_shuffle=None, train_indicator=None
                 ):
        self.row_in, self.col_in, self.dep_in = dim_in or (512,512,3)
        self.row_out, self.col_out, self.dep_out = dim_out or (512,512,1)
        self.image_format=image_format or "*.jpg"
        self.image_resize= image_resize or 1.0  # default 1.0, reduce size <=1.0
        self.image_padding= image_padding or 1.0  # default 1.0, padding proportionally >=1.0
        self.mask_color=mask_color or "green"  # green/white
        self.separate = separate if separate is not None else True  # True: split into multiple smaller views; False: take one view only
        self.coverage_train = coverage_tr or 0.9
        self.coverage_predict = coverage_prd or 1.5
        self.model_filter = model_filter or [96, 128, 192, 256, 384]
        self.model_kernel = model_kernel or [3, 3]
        from metrics import jac,dice,dice_80,dice_60,dice_40,dice_20,acc,acc_80,acc_60,acc_40,acc_20,\
            loss_bce_dice, enable_custom_activation
        # enable_custom_activation() # leakyrelu, swish, twish
        self.model_act = model_act or 'elu'
        self.model_out = model_out or ('sigmoid' if self.dep_out == 1 else 'softmax')
        self.model_loss = model_loss or (loss_bce_dice if self.dep_out == 1 else 'categorical_crossentropy')  # 'binary_crossentropy'
        self.metrics= metrics or ([jac,dice,dice_80,dice_60,dice_40,dice_20] if self.dep_out == 1 else [acc,acc_80,acc_60,acc_40,acc_20])
        self.call_hardness = call_hardness or 1.0  # 0-smooth 1-hard binary call
        self.overlay_color = overlay_color or generate_colors(self.dep_out)
        self.overlay_opacity = overlay_opacity or 0.4
        self.batch_size = batch_size or 1
        self.train_rep = train_rep or 2  # times to repeat during training
        self.train_epoch = train_epoch or 12  # max epoches during training
        self.train_step = train_step or 500
        self.train_vali_step = train_vali_step or 200
        self.train_learning_rate = train_learning_rate or 1e-4
        self.train_vali_split = train_vali_split or 0.33
        self.train_aug = train_aug if train_aug is not None else True  # only to training set, not validation or prediction mode
        self.train_shuffle = train_shuffle if train_shuffle is not None else True  # only to training set, not validation or prediction mode
        self.train_continue = train_continue if train_continue is not None else True  # continue training by loading previous weights
        self.train_indicator = train_indicator or ('val_dice' if self.dep_out == 1 else 'val_acc')  # indicator to maximize

    def __str__(self):
        return '_'.join([
            "%df%d-%d" % (len(self.model_filter), self.model_filter[0], self.model_filter[-1]),
            "%dk%s" % (len(self.model_kernel), ''.join(str(x) for x in self.model_kernel)),
            self.model_act, self.model_out,
            "o%d"% self.dep_out,
            self.model_loss if isinstance(self.model_loss, str) else self.model_loss.__name__,
        ])