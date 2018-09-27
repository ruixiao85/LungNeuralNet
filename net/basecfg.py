import colorsys
import random
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from process_image import reverse_sigmoid


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
        self.predict_proc=predict_proc if predict_proc is not None else self.single_call
        self.batch_size=batch_size or 1
        self.train_rep=train_rep or 3  # times to repeat during training
        self.train_epoch=train_epoch or 12  # max epoches during training
        self.train_step=train_step or 120
        self.train_vali_step=train_vali_step or 60
        self.train_vali_split=train_vali_split or 0.33
        self.train_aug=train_aug or 1  # only to training set, not validation or prediction mode, 0-disabled higher the more aug
        self.train_shuffle=train_shuffle if train_shuffle is not None else True  # only to training set, not validation or prediction mode
        self.train_continue=train_continue if train_continue is not None else True  # continue training by loading previous weights

    def single_call(self, img, msk):  # sigmoid (r,c,1) blend, np result
        blend=img.copy()
        opa=self.overlay_opacity
        col=self.overlay_color
        for d in range(msk.shape[-1]):
            msk[...,d]=np.rint(msk[...,d])  # sigmoid round to  0/1 # consider range(-1 ~ +1) for multi class voting
            for c in range(3):
                blend[..., c] = np.where(msk[...,d] >= 0.5, blend[..., c] * (1 - opa) + col[d][c] * opa, blend[..., c]) # weighted average
        return blend, np.sum(msk, axis=(0,1), keepdims=False)
        # return blend, np.sum(msk, keepdims=True)
    def multi_call(self, img, msk):  # softmax (r,c,multi_label) blend, np result
        blend=img.copy()
        opa=self.overlay_opacity; col=self.overlay_color
        dim=self.predict_size # do argmax if predict categories covers all possibilities or consider them individually
        msk=np.argmax(msk, axis=-1)
        uni, count=np.unique(msk, return_counts=True)
        map_count=dict(zip(uni,count))
        count_vec=np.zeros(dim)
        for d in range(dim):
            count_vec[d]=map_count.get(d) or 0
            for c in range(3):
                blend[..., c] = np.where(msk == d, blend[..., c] * (1 - opa) + col[d][c] * opa, blend[..., c])
        return blend, count_vec
    def compare_call(self, img, msk):  # compare input and output with same dimension
        diff=np.abs(img-msk)
        # imsave("test.jpg",reverse_sigmoid(msk))
        return msk, np.sum(diff,axis=-1)

    def draw_text(self, img, text_list, width):
        font = "arial.ttf" #times.ttf
        size = round(0.33*(26+0.03*width+width/len(max(text_list, key=len))))
        txt_col = (10, 10, 10)
        origin = Image.fromarray(img.astype(np.uint8),'RGB') # L RGB
        draw = ImageDraw.Draw(origin)
        draw.text((0, 0), '\n'.join(text_list), txt_col, ImageFont.truetype(font, size))
        for i in range(len(text_list)-1):
            sym_col = self.overlay_color[i]
            draw.text((0, 0), ' \n'*(i+1)+' X', sym_col, ImageFont.truetype(font, size))
        return np.array(origin)
