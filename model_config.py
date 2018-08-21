import colorsys
import random

def generate_colors(n, shuffle=False):
    hsv = [(i / n, 1, 0.8) for i in range(n)] # last number 0.8 the brightness
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    if shuffle:
        random.shuffle(colors)
    return colors


class ModelConfig:

    def __init__(self, dim_in=None, dim_out=None, image_format=None, resize=None, padding=None, mask_color=None,
                 coverage_tr=None, coverage_prd=None, batch_size=None, separate=None,
                 filter_size=None, kernel_size=None,
                 act_fun=None, out_fun=None, loss_fun=None,
                 overlay_color=None, overlay_opacity=None, call_hardness=None,
                 max_rep=None, max_epoch=None, max_train_step=None, max_vali_step=None,
                 learning_rate=None, valid_split=None, img_aug=None, cont_train=None,
                 shuffle=None
                 ):
        self.row_in, self.col_in, self.dep_in = dim_in or (512,512,3)
        self.row_out, self.col_out, self.dep_out = dim_out or (512,512,1)
        self.image_format=image_format or "*.jpg"
        self.image_resize= resize or 1.0  # default 1.0, reduce size <=1.0
        self.image_padding= padding or 1.0  # default 1.0, padding proportionally >=1.0
        self.mask_color=mask_color or "green"  # green/white
        self.separate = separate or True  # True: split into multiple smaller views; False: take one view only
        self.coverage_train = coverage_tr or 0.9  #
        self.coverage_predict = coverage_prd or 1.4
        self.model_filter = filter_size or [96, 128, 192, 256, 384]
        self.model_kernel = kernel_size or [3, 3]
        self.model_act = act_fun or 'elu'
        self.model_out = out_fun or ('sigmoid' if self.dep_out == 1 else 'softmax')
        self.model_loss = loss_fun or ('binary_crossentropy' if self.dep_out == 1 else 'categorical_crossentropy')
        self.call_hardness = call_hardness or 1.0  # 0-smooth 1-hard binary call
        self.overlay_color = overlay_color or generate_colors(self.dep_out)
        self.overlay_opacity = overlay_opacity or 0.6
        self.batch_size = batch_size or 1
        self.train_rep = max_rep or 3  # times to repeat during training
        self.train_epoch = max_epoch or 12  # max epoches during training
        self.train_step = max_train_step or 500
        self.train_vali_step = max_vali_step or 200
        self.train_learning_rate = learning_rate or 1e-3
        self.train_valid_split = valid_split or 0.33
        self.train_aug = img_aug or True  # only to training set, not validation or prediction mode
        self.train_shuffle = shuffle or True  # only to training set, not validation or prediction mode
        self.train_continue = cont_train or True  # continue training by loading previous weights


    def __str__(self):
        return '_'.join([
            "%d" % int(0.5 * (self.row_in + self.col_in)),
            # "%d_%d"% (self.row, self.col),
            # "%d_%d"% (self.dep_in, self.dep_out),
            # "%.1f" % self.resize,
            # "%d" % self.pad,
             "%df%d-%d_%dk%s" % (len(self.model_filter), self.model_filter[0], self.model_filter[-1], len(self.model_kernel), ''.join(str(x) for x in self.model_kernel)),
             # self.act_fun, self.out_fun, self.loss_fun if isinstance(self.loss_fun, str) else self.loss_fun.__name__,
            # "%d_%.1f_%.1f" %  (self.overlay_color, self.overlay_opacity, self.call_hardness)
             ])