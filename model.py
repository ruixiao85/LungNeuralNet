import os

from datetime import datetime
import numpy as np
import pandas as pd
from PIL import ImageDraw, Image, ImageFont
from keras.engine.saving import model_from_json
from skimage.io import imsave
from tensorflow.python.keras.models import load_model

from image_gen import MetaInfo, ImagePair, ImageGenerator
from metrics import custom_function_dict
from model_config import ModelConfig
from tensorboard_train_val import TensorBoardTrainVal
from util import mk_dir_if_nonexist, to_excel_sheet

def g_kern(size, sigma):
    from scipy import signal
    gkern1d = signal.gaussian(size, std=sigma).reshape(size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def g_kern_rect(row, col, rel_sig=0.5):
    l=max(row,col)
    mat=g_kern(l, int(rel_sig * l))
    r0, c0=int(0.5*(l-row)),int(0.5*(l-col))
    return mat[r0:r0+row,c0:c0+col]

class MyModel:
    # 'relu6'  # min(max(features, 0), 6)
    # 'crelu'  # Concatenates ReLU (only positive part) with ReLU (only the negative part). Note that this non-linearity doubles the depth of the activations
    # 'elu'  # Exponential Linear Units exp(features)-1, if <0, features
    # 'selu'  # Scaled Exponential Linear Rectifier: scale * alpha * (exp(features) - 1) if < 0, scale * features otherwise.
    # 'softplus'  # log(exp(features)+1)
    # 'softsign' features / (abs(features) + 1)

    # 'mean_squared_error' 'mean_absolute_error'
    # 'binary_crossentropy'
    # 'sparse_categorical_crossentropy' 'categorical_crossentropy'

    #     model_out = 'softmax'   model_loss='categorical_crossentropy'
    #     model_out='sigmoid'    model_loss=[loss_bce_dice] 'binary_crossentropy' "bcedice"

    def __init__(self, cfg:ModelConfig, save):
        self.cfg=cfg
        self.model=cfg.model_name(cfg)
        self.compile_model()
        if save:
            self.save_model()

    def load_model(self):  # load model
        with open(str(self.cfg) + ".json", 'r') as json_file:
            self.model = model_from_json(json_file.read())
        self.compile_model()

    def __str__(self):
        return str(self.cfg)

    def compile_model(self):
        self.model.compile(
            optimizer=self.cfg.optimizer,
            loss=self.cfg.model_loss,
            metrics=self.cfg.metrics)
        self.model.summary()

    def save_model(self):
        model_json = str(self.cfg) + ".json"
        with open(model_json, "w") as json_file:
            json_file.write(self.model.to_json())

    def train(self, cfg, multi:ImagePair):
        for tr, val, dir_out in multi.train_generator():
            export_name = multi.dir_out_ex(dir_out) +'_'+str(self.cfg)
            weight_file = export_name + ".h5"
            if self.cfg.train_continue and os.path.exists(weight_file):
                print("Continue from previous weights")
                self.model.load_weights(weight_file)
                # print("Continue from previous model with weights & optimizer")
                # self.model=load_model(weight_file,custom_objects=custom_function_dict()) # does not work well with custom act, loss func
            print('Fitting neural net...')
            for r in range(self.cfg.train_rep):
                print("Training %d/%d for %s" % (r + 1, self.cfg.train_rep, export_name))
                tr.on_epoch_end()
                val.on_epoch_end()
                from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
                history = self.model.fit_generator(tr, validation_data=val, verbose=1,
                    steps_per_epoch=min(cfg.train_step, len(tr.view_coord)) if isinstance(cfg.train_step , int) else len(tr.view_coord),
                   validation_steps=min(cfg.train_vali_step, len(val.view_coord)) if isinstance(cfg.train_vali_step, int) else len(val.view_coord),
                    epochs=self.cfg.train_epoch, max_queue_size=1, workers=0, use_multiprocessing=False, shuffle=False,
                    callbacks=[
                        ModelCheckpoint(weight_file, monitor=cfg.train_indicator, mode='max', save_weights_only=False, save_best_only=True),
                        EarlyStopping(monitor=cfg.train_indicator, mode='max', patience=1, verbose=1),
                        # ReduceLROnPlateau(monitor=cfg.train_indicator, mode='max', factor=0.1, patience=10, min_delta=1e-5, cooldown=0, min_lr=0, verbose=1),
                        # TensorBoardTrainVal(log_dir=os.path.join("log", export_name), write_graph=True, write_grads=False, write_images=True),
                    ]).history
                if not os.path.exists(export_name + ".txt"):
                    with open(export_name + ".txt", "w") as net_summary:
                        self.model.summary(print_fn=lambda x: net_summary.write(x + '\n'))
                df=pd.DataFrame(history).round(4)
                df['time']=datetime.now().strftime("%Y-%m-%d %H:%M")
                df['repeat']=r+1
                df.to_csv(export_name + ".csv", mode="a", header=(not os.path.exists(export_name + ".csv")))

    def predict(self, multi:ImagePair, pred_dir):
        xls_file="Result_%s_%s.xlsx"%(pred_dir, repr(multi.cfg))
        img_ext=self.cfg.image_format[1:] # *.jpg -> .jpg
        sum_i, sum_g = self.cfg.row_out * self.cfg.col_out, None
        msks, mask_wt, r_i, r_g,  ra, ca= None, None, None, None, None, None
        mrg_in, mrg_out, mrg_out_wt, merge_dir = None, None, None, None
        batch=multi.img_set.view_coord_batch()  # image/1batch -> view_coord
        dir_ex=multi.dir_out_ex()
        dir_cfg_append=str(self.cfg) if dir_ex is None else dir_ex+'_'+str(self.cfg)
        res_ind, res_grp=None, None
        save_ind_image=False
        for dir_out, tgt_list in multi.predict_generator():
            res_i, res_g=None, None
            print('Load model and predict to [%s]...'%dir_out)
            export_name = dir_out+'_'+dir_cfg_append
            target_dir = os.path.join(multi.wd, export_name)
            if save_ind_image or not self.cfg.separate: # skip saving individual images
                mk_dir_if_nonexist(target_dir)
            if self.cfg.separate:
                merge_dir = os.path.join(multi.wd, dir_out+'+'+dir_cfg_append) # group
                mk_dir_if_nonexist(merge_dir)
                mask_wt = g_kern_rect(self.cfg.row_out, self.cfg.col_out)
            for grp, view in batch.items():
                msks=None
                i=0; nt=len(tgt_list)
                while i < nt:
                    o=min(i+self.cfg.dep_out, nt)
                    tgt_sub=tgt_list[i:o]
                    tgt_name=ImagePair.join_targets(tgt_sub)
                    prd=ImageGenerator(multi, False, tgt_sub, view)
                    weight_file=tgt_name+'_'+dir_cfg_append+'.h5'
                    print(weight_file)
                    self.model.load_weights(weight_file) # weights only
                    # self.model=load_model(weight_file,custom_objects=custom_function_dict()) # weight optimizer archtecture
                    msk=self.model.predict_generator(prd, max_queue_size=1, workers=0, use_multiprocessing=False, verbose=1)
                    msks = msk if msks is None else  np.concatenate((msks, msk),axis=-1)
                    i=o
                print('Saving predicted results [%s] to folder [%s]...' % (grp, export_name))
                # r_i=np.zeros((len(multi.img_set.images),len(tgt_list)), dtype=np.uint32)
                if self.cfg.separate:
                    mrg_in = np.zeros((view[0].ori_row, view[0].ori_col, self.cfg.dep_in), dtype=np.float32)
                    mrg_out = np.zeros((view[0].ori_row, view[0].ori_col, len(tgt_list)), dtype=np.float32)
                    mrg_out_wt = np.zeros((view[0].ori_row, view[0].ori_col), dtype=np.float32) + np.finfo(np.float32).eps
                    sum_g = view[0].ori_row * view[0].ori_col
                    # r_g=np.zeros((1,len(tgt_list)), dtype=np.uint32)
                for i, msk in enumerate(msks):
                    # if i>=len(multi.view_coord): break # last batch may have unused entries
                    ind_name = view[i].file_name
                    ind_file = os.path.join(target_dir, ind_name)
                    origin = view[i].get_image(os.path.join(multi.wd, multi.dir_in_ex()), self.cfg)
                    print(ind_name); text_list = [ind_name]
                    blend, r_i=self.mask_call(origin, msk)
                    for d in range(len(tgt_list)):
                        text = "[  %d: %s] %d / %d  %.2f%%" % ( d, tgt_list[d], r_i[d], sum_i, 100. * r_i[d] / sum_i)
                        print(text); text_list.append(text)
                    if save_ind_image or not self.cfg.separate: # skip saving individual images
                        blend = self.draw_text(blend, text_list, self.cfg.row_out)  # RGB:3x8-bit dark text
                        blend.save(ind_file.replace(img_ext, ".jpe"))
                        # imsave(ind_file.replace(img_ext, ".jpe"), blend)
                    res_i =r_i[np.newaxis,...] if res_i is None else np.concatenate((res_i, r_i[np.newaxis,...]))

                    if self.cfg.separate:
                        ri,ro=view[i].row_start, view[i].row_end
                        ci,co=view[i].col_start, view[i].col_end
                        ra,ca=view[i].ori_row,view[i].ori_col
                        tri, tro = 0, self.cfg.row_out
                        tci, tco = 0, self.cfg.col_out
                        if ri<0:
                            tri=-ri; ri=0
                        if ci<0:
                            tci=-ci; ci=0
                        if ro>ra:
                            tro=tro-(ro-ra); ro=ra
                        if co>ca:
                            tco=tco-(co-ca); co=ca
                        mrg_in[ri:ro,ci:co] = origin[tri:tro,tci:tco]
                        for d in range(len(tgt_list)):
                            mrg_out[ri:ro,ci:co,d] += (msk[...,d] * mask_wt)[tri:tro,tci:tco]
                        mrg_out_wt[ri:ro,ci:co] += mask_wt[tri:tro,tci:tco]
                if self.cfg.separate:
                    for d in range(len(tgt_list)):
                        mrg_out[...,d] /= mrg_out_wt
                    print(grp); text_list=[grp]
                    merge_name = view[0].image_name
                    merge_file = os.path.join(merge_dir, merge_name)
                    blend, r_g = self.mask_call(mrg_in, mrg_out)
                    for d in range(len(tgt_list)):
                        text = "[  %d: %s] %d / %d  %.2f%%" % (d, tgt_list[d], r_g[d], sum_g, 100. * r_g[d] / sum_g)
                        print(text); text_list.append(text)
                    # cv2.imwrite(merge_file, mrg_out[..., np.newaxis] * 255.)
                    blend = self.draw_text(blend, text_list, ra)  # RGB:3x8-bit dark text
                    blend.save(merge_file.replace(img_ext, ".jpe"))
                    # imsave(merge_file.replace(img_ext, ".jpe"), blend)
                    res_g=r_g[np.newaxis,...] if res_g is None else np.concatenate((res_g, r_g[np.newaxis,...]))
            res_ind=res_i if res_ind is None else np.hstack((res_ind, res_i))
            res_grp=res_g if res_grp is None else np.hstack((res_grp, res_g))
        df = pd.DataFrame(res_ind, index=multi.img_set.images, columns=multi.targets)
        to_excel_sheet(df, xls_file, multi.origin)  # per slice
        if self.cfg.separate:
            df = pd.DataFrame(res_grp, index=batch.keys(), columns=multi.targets)
            to_excel_sheet(df, xls_file, multi.origin + "_sum")


    def draw_text(self, img, text_list, width):
        font = "arial.ttf" #times.ttf
        size = round(0.33*(26+0.03*width+width/len(max(text_list, key=len))))
        txt_col = (10, 10, 10)
        origin = Image.fromarray(img.astype(np.uint8),'RGB') # L RGB
        draw = ImageDraw.Draw(origin)
        draw.text((0, 0), '\n'.join(text_list), txt_col, ImageFont.truetype(font, size))
        for i in range(len(text_list)-1):
            sym_col = self.cfg.overlay_color[i]
            draw.text((0, 0), ' \n'*(i+1)+' X', sym_col, ImageFont.truetype(font, size))
        return origin

 # def draw_text(self, img, text_list, width):
 #        size = round(0.5*(24+0.03*width))
 #        txt_col = (10, 10, 10)
 #        origin = Image.fromarray(img.astype(np.uint8),'RGB') # L RGB
 #        draw = ImageDraw.Draw(origin)
 #        draw.text((0, 0), '\n'.join(text_list), txt_col, ImageFont.truetype("arial.ttf",size))  # font type size)
 #        for i in range(len(text_list)-1):
 #            sym_col = self.cfg.overlay_color[i]
 #            draw.text((0, round(1.1*size*(i+1))), ' X', sym_col, ImageFont.truetype("arial.ttf", size))  # font type size)
 #        return origin

    def mask_call(self, img, msk):  # blend, np result
        blend=img.copy()
        opa=self.cfg.overlay_opacity
        col=self.cfg.overlay_color
        dim=self.cfg.predict_size if self.cfg.predict_all_inclusive else self.cfg.dep_out # do argmax if predict categories covers all possibilities or consider them individually
        if dim==1: # r x c x 1
            for d in range(msk.shape[-1]):
                msk[...,d]=np.rint(msk[...,d])  # sigmoid round to  0/1 # consider range(-1 ~ +1) for multi class voting
                for c in range(3):
                    blend[..., c] = np.where(msk[...,d] >= 0.5, blend[..., c] * (1 - opa) + col[d][c] * opa, blend[..., c]) # weighted average
            return blend, np.sum(msk, axis=(0,1), keepdims=False)
            # return blend, np.sum(msk, keepdims=True)
        else: # softmax r x c x multi_label
            msk=np.argmax(msk, axis=-1)
            uni, count=np.unique(msk, return_counts=True)
            map_count=dict(zip(uni,count))
            count_vec=np.zeros(dim)
            for d in range(dim):
                count_vec[d]=map_count.get(d) or 0
                for c in range(3):
                    blend[..., c] = np.where(msk == d, blend[..., c] * (1 - opa) + col[d][c] * opa, blend[..., c])
            return blend, count_vec
