import colorsys
import os
import random
import numpy as np

from osio import find_file_pattern,find_file_pattern_rel


def generate_colors(n, shuffle=False):
    hsv = [(i / n, 0.9, 0.9) for i in range(n)]
    # colors = [tuple((255*np.array(col)).astype(np.uint8)) for col in map(lambda c: colorsys.hsv_to_rgb(*c), hsv)] # rgb
    colors = [tuple((255*np.array(col)[::-1]).astype(np.uint8)) for col in map(lambda c: colorsys.hsv_to_rgb(*c), hsv)] # bgr to match opencv
    if shuffle:
        random.shuffle(colors)
    return colors

class Config:
    def __init__(self,num_targets,target_scale,**kwargs,):
        self.num_targets=num_targets
        self.target_scale=target_scale or 1.0 # pixel scale to target default to 10X=1px/µm (e.g., 40X=4px/µm)
        self.dim_in=kwargs.get('dim_in', (768,768,3))
        self.row_in, self.col_in, self.dep_in=self.dim_in
        self.dim_out=kwargs.get('dim_out', (768,768,1))
        self.row_out, self.col_out, self.dep_out=self.dim_out
        self.dep_out=min(self.dep_out,num_targets)
        self.image_format=kwargs.get('image_format', "*.jpg")
        self.feed=kwargs.get('feed', 'tanh')
        self.act=kwargs.get('act', 'relu')
        self.out=kwargs.get('out', ('sigmoid' if self.dep_out==1 else 'softmax'))
        self.coverage_train=kwargs.get('coverage_train', 3.0)
        self.coverage_predict=kwargs.get('coverage_predict', 2.0)
        self.predict_size=kwargs.get('predict_size', self.num_targets) # output each target invididually or grouped
        self.call_hardness=kwargs.get('call_hardness', 1.0)  # 0-smooth 1-hard binary call
        self.overlay_color=kwargs.get('overlay_color', generate_colors(self.num_targets))
        self.overlay_opacity=kwargs.get('overlay_opacity', [0.2]*self.num_targets)
        self.overlay_textshape_bwif=kwargs.get('overlay_textshape_bwif', (True,True,False,False)) # draw black_legend, white_legend, color_instance_text, fill_shape
        self.save_ind_raw=kwargs.get('save_ind_raw', (True,True))
        self.ntop=kwargs.get('ntop', 1) # numbers of top networks to keep, delete the networks that are less than ideal
        self.batch_size=kwargs.get('batch_size', 1)
        self.pre_trained=kwargs.get('pre_trained', True) # True: load weights pre-trained on imagenet; False: init with random weights
        self.train_epoch=kwargs.get('train_epoch', 50) # max epoches during training
        self.train_step=kwargs.get('train_step', 768)
        self.train_vali_step=kwargs.get('train_vali_step', 256)
        self.train_vali_split=kwargs.get('train_vali_split', 0.33)
        self.train_aug=kwargs.get('train_aug', 2)  # only to training set, not validation or prediction mode, applies to image-mask set and image+patch
        self.train_shuffle=kwargs.get('train_shuffle', True)  # only to training set, not validation or prediction mode
        self.train_continue=kwargs.get('train_continue', False)  # True to continue training by loading previous weights
        self.indicator=kwargs.get('indicator', 'val_acc')
        self.indicator_trend=kwargs.get('indicator_trend', 'max')
        self.indicator_patience=kwargs.get('indicator_patience', 2) # times to continue training even without improvement
        self.save_mode=kwargs.get('save_mode', 'h') # decide which network to save: All/CurrentBest/HistoricalBest/None
        self.sig_digits=kwargs.get('sig_digits', 3) # significant digits for indicator/score
        self._model_cache={}

    @staticmethod
    def get_proper_range(ra,ca,ri,ro,ci,co,tri,tro,tci,tco): # row/col limit of large image, row/col index on large image, row/col index for small image
        # print('%d %d %d:%d,%d,%d %d:%d,%d,%d'%(ra,ca,ri,ro,ci,co,tri,tro,tci,tco),end='')
        if ri<0: tri=-ri; ri=0
        if ci<0: tci=-ci; ci=0
        if ro>ra: tro=tro-(ro-ra); ro=ra
        if co>ca: tco=tco-(co-ca); co=ca
        # print('-> %d %d %d:%d,%d,%d %d:%d,%d,%d'%(ra,ca,ri,ro,ci,co,tri,tro,tci,tco))
        return ri,ro,ci,co,tri,tro,tci,tco

    @staticmethod
    def parse_saved_model(filename):
        parts=filename.split('^')
        return int(parts[1]),float(parts[2]) # epoch, last_best

    def find_best_models(self, pattern, allow_cache=False):
        cwd=os.getcwd()
        print("Scanning for files matching %s in %s"%(pattern,cwd))
        if allow_cache and pattern in self._model_cache:
            return self._model_cache[pattern]
        files=sorted(find_file_pattern_rel(cwd,pattern),
             key=lambda t: (float(t.split('^')[2])*(-1.0 if self.indicator_trend=='max' else 1.0), -1*int(t.split('^')[1]))) # best score then highest epoch
        nfiles=len(files)
        if nfiles>0:
            print('Found %d previous models, keeping the top %d (%s):'%(nfiles,self.ntop,self.indicator_trend))
            for l in range(nfiles):
                if l<self.ntop:
                    print(('* ' if l==0 else '  '),end='')
                    print('%d. %s kept'%(l+1,files[l]))
                else:
                    os.remove(files[l])
                    print('- %d. %s deleted'%(l+1,files[l]))
            if allow_cache:
                self._model_cache[pattern]=files
            return files
        else:
            print('No previus model found, starting fresh')
            return None

    @staticmethod
    def join_targets(tgt_list) :
        # return ','.join(tgt_list)
        # return ','.join(tgt_list[:max(1, int(24 / len(tgt_list)))]) #shorter but >= 1 char, may have error if categories share same leading chars
        maxchar=max(1, int(28 / len(tgt_list))) # clip to fewer leading chars
        # maxchar=9999 # include all
        return ','.join(tgt[:maxchar] for tgt in tgt_list)

