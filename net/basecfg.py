import colorsys
import random
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def generate_colors(n, shuffle=False):
    hsv = [(i / n, 1.0, 0.85) for i in range(n)]
    colors = [tuple((255*np.array(col)).astype(np.uint8)) for col in map(lambda c: colorsys.hsv_to_rgb(*c), hsv)]
    if shuffle:
        random.shuffle(colors)
    return colors

class Config:
    def __init__(self,
                 num_targets=None,image_format=None,image_resize=None,image_padding=None,mask_color=None,
                 coverage_tr=None,coverage_prd=None,batch_size=None,separate=None,out_image=None,
                 call_hardness=None,overlay_color=None,overlay_opacity=None,predict_size=None,predict_proc=None,
                 train_rep=None,train_epoch=None,train_step=None,train_vali_step=None,
                 train_vali_split=None,train_aug=None,train_continue=None,train_shuffle=None):
        self.num_targets=num_targets or 10  # lead to default overlay_color predict_group
        self.image_format=image_format or "*.jpg"
        self.image_resize=image_resize or 1.0  # default 1.0, reduce size <=1.0
        self.image_padding=image_padding or 1.0  # default 1.0, padding proportionally >=1.0
        self.mask_color=mask_color or "white"  # green/white
        self.out_image=out_image if out_image is not None else False # output type: True=image False=mask
        self.separate=separate if separate is not None else True  # True: split into multiple smaller views; False: take one view only
        self.coverage_train=coverage_tr or 2.0
        self.coverage_predict=coverage_prd or 3.0
        self.call_hardness=call_hardness or 1.0  # 0-smooth 1-hard binary call
        self.overlay_color=overlay_color if isinstance(overlay_color, list) else \
            generate_colors(overlay_color) if isinstance(overlay_color, int) else \
                generate_colors(self.num_targets)
        self.overlay_opacity=overlay_opacity or 0.2
        self.predict_size=predict_size or num_targets
        from model import single_call,multi_call,compare_call
        self.predict_proc=predict_proc if predict_proc is not None else single_call
        self.batch_size=batch_size or 1
        self.train_rep=train_rep or 3  # times to repeat during training
        self.train_epoch=train_epoch or 10  # max epoches during training
        self.train_step=train_step or 128
        self.train_vali_step=train_vali_step or 64
        self.train_vali_split=train_vali_split or 0.33
        self.train_aug=train_aug or 2  # only to training set, not validation or prediction mode, 1Flip 2Rotate 3Zoom 4ContrastGray 5BlurSharp 6NoiseDrop
        self.train_shuffle=train_shuffle if train_shuffle is not None else True  # only to training set, not validation or prediction mode
        self.train_continue=train_continue if train_continue is not None else True  # continue training by loading previous weights
