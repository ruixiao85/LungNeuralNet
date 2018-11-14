import colorsys
import random
import numpy as np


def generate_colors(n, shuffle=False):
    hsv = [(i / n, 0.9, 0.7) for i in range(n)]
    colors = [tuple((255*np.array(col)).astype(np.uint8)) for col in map(lambda c: colorsys.hsv_to_rgb(*c), hsv)]
    if shuffle:
        random.shuffle(colors)
    return colors

class Config:
    def __init__(self, dim_in=None, dim_out=None,
                 num_targets=None,image_format=None,image_resize=None,image_padding=None,mask_color=None,
                 feed=None, act=None, out=None,batch_size=None,separate=None,coverage_train=None,coverage_predict=None,out_image=None,
                 call_hardness=None,overlay_color=None,overlay_opacity=None,predict_size=None,
                 train_rep=None,train_epoch=None,train_step=None,train_vali_step=None,
                 train_vali_split=None,train_aug=None,train_continue=None,train_shuffle=None,indicator=None,indicator_trend=None):
        self.dim_in=dim_in or (512,512,3)
        self.row_in, self.col_in, self.dep_in=self.dim_in
        self.dim_out=dim_out or (512,512,1)
        self.row_out, self.col_out, self.dep_out=self.dim_out
        self.num_targets=num_targets or 10  # lead to default overlay_color predict_group
        self.image_format=image_format or "*.jpg"
        self.image_resize=image_resize or 1.0  # default 1.0, reduce size <=1.0
        self.image_padding=image_padding or 1.0  # default 1.0, padding proportionally >=1.0
        self.mask_color=mask_color or "white"  # green/white
        self.feed=feed or 'tanh'
        self.act=act or 'elu'
        self.out=out or ('sigmoid' if self.dep_out==1 else 'softmax')
        self.out_image=out_image if out_image is not None else False # output type: True=image False=mask
        self.separate=separate if separate is not None else True  # True: split into multiple smaller views; False: take one view only
        self.coverage_train=coverage_train or 2.0
        self.coverage_predict=coverage_predict or 3.0
        self.call_hardness=call_hardness or 1.0  # 0-smooth 1-hard binary call
        self.overlay_color=overlay_color if isinstance(overlay_color, list) else \
            generate_colors(overlay_color) if isinstance(overlay_color, int) else \
                generate_colors(self.num_targets)
        self.overlay_opacity=overlay_opacity if isinstance(overlay_color, list) else [0.3]*self.num_targets
        self.predict_size=predict_size or num_targets
        self.batch_size=batch_size or 1
        self.train_rep=train_rep or 2  # times to repeat during training
        self.train_epoch=train_epoch or 20  # max epoches during training
        self.train_step=train_step or 128
        self.train_vali_step=train_vali_step or 64
        self.train_vali_split=train_vali_split or 0.33
        self.train_aug=train_aug or 4  # only to training set, not validation or prediction mode, applies to image-mask set and image+patch
        self.train_shuffle=train_shuffle if train_shuffle is not None else True  # only to training set, not validation or prediction mode
        self.train_continue=train_continue if train_continue is not None else True  # continue training by loading previous weights
        self.indicator=indicator or 'val_acc'
        self.indicator_trend=indicator_trend or 'max'


    def split_train_vali(self,view_coords):
        tr_list,val_list=[],[]  # list view_coords, can be from slices
        tr_image,val_image=set(),set()  # set whole images
        for vc in view_coords:
            if vc.image_name in tr_image:
                tr_list.append(vc)
                tr_image.add(vc.image_name)
            elif vc.image_name in val_image:
                val_list.append(vc)
                val_image.add(vc.image_name)
            else:
                if (len(val_list)+0.05)/(len(tr_list)+0.05)>self.train_vali_split:
                    tr_list.append(vc)
                    tr_image.add(vc.image_name)
                else:
                    val_list.append(vc)
                    val_image.add(vc.image_name)
        print("From %d split into train: %d views %d images; validation %d views %d images"%
              (len(view_coords),len(tr_list),len(tr_image),len(val_list),len(val_image)))
        print("Training Images:"); print(tr_image)
        print("Validation Images:"); print(val_image)
        # tr_list.sort(key=lambda x: str(x), reverse=False)
        # val_list.sort(key=lambda x: str(x), reverse=False)
        return tr_list,val_list

    def get_proper_range(self,ra,ca,ri,ro,ci,co,tri,tro,tci,tco): # row/col limit of large image, row/col index on large image, row/col index for small image
        # print('%d %d %d:%d,%d,%d %d:%d,%d,%d'%(ra,ca,ri,ro,ci,co,tri,tro,tci,tco),end='')
        if ri<0: tri=-ri; ri=0
        if ci<0: tci=-ci; ci=0
        if ro>ra: tro=tro-(ro-ra); ro=ra
        if co>ca: tco=tco-(co-ca); co=ca
        # print('-> %d %d %d:%d,%d,%d %d:%d,%d,%d'%(ra,ca,ri,ro,ci,co,tri,tro,tci,tco))
        return ri,ro,ci,co,tri,tro,tci,tco

    def find_best_models(self, pattern, allow_cache=False):
        import glob,os
        if allow_cache:
            if not hasattr(self,"_model_cache"):
                self._model_cache={}
            if not pattern in self._model_cache:
                self._model_cache[pattern]=sorted(glob.glob(pattern), key=lambda t: t.split('^')[1], reverse=self.indicator_trend=='max')
            return self._model_cache[pattern]
        else: # search
            files=sorted(glob.glob(pattern), key=lambda t: t.split('^')[1], reverse=self.indicator_trend=='max')
            nfiles=len(files)
            if nfiles>0:
                ntop=2
                print('Found %d previous models, keeping the top %d (%s):'%(nfiles,ntop,self.indicator_trend))
                for l in range(nfiles):
                    if l<ntop:
                        print(('* 'if l==0 else '  '),end='')
                        print('%d. %s kept'%(l+1,files[l]))
                    else:
                        os.remove(files[l])
                        print('- %d. %s deleted'%(l+1, files[l]))
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