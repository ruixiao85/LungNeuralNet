import os

from datetime import datetime
import numpy as np
import pandas as pd
from PIL import ImageDraw, Image, ImageFont
from keras.engine.saving import model_from_json
from skimage.io import imsave
from image_gen import MetaInfo, ImagePairMulti, ImageGeneratorMulti
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
        self.model, self.name=cfg.model_name(cfg)
        self.compile_model()
        if save:
            self.save_model()

    def load_model(self, name:str):  # load model
        self.name=name
        with open(self.name + ".json", 'r') as json_file:
            self.model = model_from_json(json_file.read())
        self.compile_model()

    def __str__(self):
        return self.name

    def compile_model(self):
        from keras.optimizers import Adam, RMSprop, SGD
        self.model.compile(
            optimizer=Adam(self.cfg.train_learning_rate),
            # optimizer=SGD(lr=0.01),
            # optimizer=RMSprop(self.train_learning_rate, decay=1e-6),
            loss=self.cfg.model_loss,
            metrics= self.cfg.metrics)
        self.model.summary()

    def save_model(self):
        model_json = self.name + ".json"
        with open(model_json, "w") as json_file:
            json_file.write(self.model.to_json())

    # def export_name(self, dir, name):
    #     return "%s_%s" % (dir, name)

    def train(self, cfg, multi:ImagePairMulti):
        tr_list, val_list = [], []  # list view_coords, can be from slices
        tr_image, val_image = set(), set()  # set whole images
        for vc in multi.view_coord:
            if vc.image_name in tr_image:
                tr_list.append(vc)
                tr_image.add(vc.image_name)
            elif vc.image_name in val_image:
                val_list.append(vc)
                val_image.add(vc.image_name)
            else:
                if (len(val_list) + 0.05) / (len(tr_list) + 0.05) > self.cfg.train_vali_split:
                    tr_list.append(vc)
                    tr_image.add(vc.image_name)
                else:
                    val_list.append(vc)
                    val_image.add(vc.image_name)
        print("From %d split into train: %d views %d images; validation %d views %d images" %
              (len(multi.view_coord), len(tr_list), len(tr_image), len(val_list), len(val_image)))
        print("Training Images:"); print(tr_image)
        print("Validation Images:"); print(val_image)
        tr=ImageGeneratorMulti(multi, self.cfg.train_aug, tr_list)
        val=ImageGeneratorMulti(multi, False, val_list)

        export_name = "%s_%s_%s" % (multi.dir_out,multi.dir_out_ex, self.name)
        weight_file = export_name + ".h5"
        if self.cfg.train_continue and os.path.exists(weight_file):
            print("Continue from previous weights")
            self.model.load_weights(weight_file)

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
                    ReduceLROnPlateau(monitor=cfg.train_indicator, mode='max', factor=0.1, patience=10, min_delta=1e-5, cooldown=0, min_lr=0, verbose=1),
                    TensorBoardTrainVal(log_dir=os.path.join("log", export_name), write_graph=True, write_grads=False, write_images=True),
                ]).history
            if not os.path.exists(export_name + ".txt"):
                with open(export_name + ".txt", "w") as net_summary:
                    self.model.summary(print_fn=lambda x: net_summary.write(x + '\n'))
            df=pd.DataFrame(history).round(4)
            df['time']=datetime.now().strftime("%Y-%m-%d %H:%M")
            df['repeat']=r+1
            df.to_csv(export_name + ".csv", mode="a", header=(not os.path.exists(export_name + ".csv")))

    def predict(self, multi:ImagePairMulti, xls_file):
        img_ext=self.cfg.image_format[1:]
        mrg_in,mrg_out,mrg_out_wt,merge_dir,mask_wt=None,None,None,None,None
        r_i, r_g, sum_g, res_i, res_g=None,None,None,None,None
        print('Load weights and predicting ...')
        export_name = "%s_%s_%s" % (multi.dir_out, multi.dir_out_ex, self.name)
        weight_file = export_name + ".h5"
        self.model.load_weights(weight_file)  # TODO switch networks for multi-label

        target_dir = os.path.join(multi.wd, export_name)
        if not self.cfg.separate: # skip saving individual images
            mk_dir_if_nonexist(target_dir)
        sum_i = self.cfg.row_out * self.cfg.col_out
        batch=multi.img_set.view_coord_batch()  # image/1batch -> view_coord
        if self.cfg.separate:
            merge_dir = os.path.join(multi.wd, "%s_%s_s_%s" % (multi.dir_out, multi.dir_out_ex, self.name))
            mk_dir_if_nonexist(merge_dir)
            mask_wt = g_kern_rect(self.cfg.row_out, self.cfg.col_out)
        for grp, view in batch.items():
            prd=ImageGeneratorMulti(multi, False, view)
            msks = self.model.predict_generator(prd, max_queue_size=1, workers=0, use_multiprocessing=False, verbose=1)
            print('Saving predicted results [%s] to folder [%s]...' % (grp, export_name))
            # r_i=np.zeros((len(multi.img_set.images),self.cfg.dep_out), dtype=np.uint32)
            if self.cfg.separate:
                mrg_in = np.zeros((view[0].ori_row, view[0].ori_col, self.cfg.dep_in), dtype=np.float32)
                mrg_out = np.zeros((view[0].ori_row, view[0].ori_col, self.cfg.dep_out), dtype=np.float32)
                mrg_out_wt = np.zeros((view[0].ori_row, view[0].ori_col), dtype=np.float32)
                sum_g = view[0].ori_row * view[0].ori_col
                # r_g=np.zeros((1,self.cfg.dep_out), dtype=np.uint32)
            for i, msk in enumerate(msks):
                ind_name = view[i].file_name
                ind_file = os.path.join(target_dir, ind_name)
                origin = view[i].get_image(os.path.join(multi.wd, "%s_%s"%(multi.origin, multi.dir_in_ex)), self.cfg)
                print(ind_name); text_list = [ind_name]
                blend, r_i=self.mask_call(origin, msk)
                for d in range(self.cfg.dep_out):
                    text = "[  %d: %s] %d / %d  %.2f%%" % ( d, multi.targets[0], r_i[d], sum_i, 100. * r_i[d] / sum_i)
                    print(text); text_list.append(text)
                # cv2.imwrite(ind_file, msk[...,np.newaxis] * 255.)
                blend = self.draw_text(blend, text_list)  # RGB:3x8-bit dark text
                if not self.cfg.separate: # skip saving individual images
                    imsave(ind_file.replace(img_ext, ".jpe"), blend)
                res_i = r_i if res_i is None else np.vstack((res_i, r_i))

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
                    for d in range(self.cfg.dep_out):
                        mrg_out[ri:ro,ci:co,d] += (msk[...,d] * mask_wt)[tri:tro,tci:tco]
                    mrg_out_wt[ri:ro,ci:co] += mask_wt[tri:tro,tci:tco]
            if self.cfg.separate:
                for d in range(self.cfg.dep_out):
                    mrg_out[...,d] /= mrg_out_wt
                print(grp); text_list=[grp]
                merge_name = view[0].image_name
                merge_file = os.path.join(merge_dir, merge_name)
                blend, r_g = self.mask_call(mrg_in, mrg_out)
                for d in range(self.cfg.dep_out):
                    text = "[  %d: %s] %d / %d  %.2f%%" % (d, multi.targets[d], r_g[d], sum_g, 100. * r_g[d] / sum_g)
                    print(text); text_list.append(text)
                # cv2.imwrite(merge_file, mrg_out[..., np.newaxis] * 255.)
                blend = self.draw_text(blend, text_list, 2.0)  # RGB:3x8-bit dark text
                imsave(merge_file.replace(img_ext, ".jpe"), blend)
                res_g=r_g if res_g is None else np.vstack((res_g, r_g))
        df = pd.DataFrame(res_i, index=multi.img_set.images, columns=multi.targets)
        to_excel_sheet(df, xls_file, multi.origin)  # per slice
        if self.cfg.separate:
            df = pd.DataFrame(res_g, index=batch.keys(), columns=multi.targets)
            to_excel_sheet(df, xls_file, multi.origin + "_sum")


    def draw_text(self, img, text_list, size_multiple=1.0):
        size=int(0.25*(3.0*25+0.003*(self.cfg.row_out+self.cfg.col_out))*size_multiple) # space=1.15
        op=self.cfg.overlay_opacity
        txt_col = (10, 10, 10, int((1.0+op)*127))
        origin = Image.fromarray(img.astype(np.uint8),'RGB') # L RGB
        draw = ImageDraw.Draw(origin)
        draw.text((0, 0), '\n'.join(text_list), txt_col, ImageFont.truetype("arial.ttf",size))  # font type size)
        for i in range(self.cfg.dep_out):
            sym_col = self.cfg.overlay_color[i]+(int(op*255),)
            draw.text((0, round(size*(i+1))), ' X', sym_col, ImageFont.truetype("arial.ttf", size))  # font type size)
        return origin

    def mask_call(self, img, msk):  # blend, np result
        blend=img.copy()
        opa=self.cfg.overlay_opacity
        col=self.cfg.overlay_color
        dim=self.cfg.dep_out
        if dim==1: # r x c x 1
            msk=np.rint(msk[...,0])  # sigmoid round to  0/1
            for c in range(3):
                blend[..., c] = np.where(msk >= 0.5, blend[..., c] * (1 - opa) + col[0][c] * opa, blend[..., c])
            return blend, np.sum(msk, keepdims=True)
        else: # softmax r x c x multi_label
            msk=np.argmax(msk, axis=-1)
            uni, count=np.unique(msk, return_counts=True)
            map_count=dict(zip(uni,count))
            count_vec=[]
            for d in range(dim):
                count_vec.append(map_count.get(d) or 0)
                for c in range(3):
                    blend[..., c] = np.where(msk == d, blend[..., c] * (1 - opa) + col[d][c] * opa, blend[..., c])
            return blend, count_vec
