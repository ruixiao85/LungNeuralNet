import cv2
import os

from datetime import datetime
import numpy as np
import pandas as pd
from PIL import ImageDraw, Image, ImageFont
from keras import backend as K, metrics
from keras.backend.tensorflow_backend import _to_tensor
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.saving import model_from_json
from keras.layers import Activation
from tensorflow.python.keras.activations import softmax
from tensorflow.python.ops.image_ops_impl import central_crop
from skimage.io import imsave
from scipy import signal
from image_gen import MetaInfo, ImagePairMulti, ImageGeneratorMulti
from model_config import ModelConfig
from process_image import scale_input, scale_input_reverse
from util import mk_dir_if_nonexist, to_excel_sheet

SMOOTH_LOSS = 1e-5
# def depth_softmax(matrix, is_tensor=True): # increase temperature to make the softmax more sure of itself
#     temp = 5.0
#     if is_tensor:
#         exp_matrix = K.exp(matrix * temp)
#         softmax_matrix = exp_matrix / K.sum(exp_matrix, axis=2, keepdims=True)
#     else:
#         exp_matrix = np.exp(matrix * temp)
#         softmax_matrix = exp_matrix / np.sum(exp_matrix, axis=2, keepdims=True)
#     return softmax_matrix
# def depth_softmax(matrix):
#     sigmoid = lambda x: 1 / (1 + K.exp(-x))
#     sigmoided_matrix = sigmoid(matrix)
#     softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)
#     return softmax_matrix

def g_kern(size, sigma):
    gkern1d = signal.gaussian(size, std=sigma).reshape(size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def g_kern_rect(row, col, rel_sig=0.5):
    l=max(row,col)
    mat=g_kern(l, int(rel_sig * l))
    r0, c0=int(0.5*(l-row)),int(0.5*(l-col))
    return mat[r0:r0+row,c0:c0+col]

def jac_d(y_true, y_pred):
    y_true_f, y_pred_f = K.flatten(y_true), K.flatten(y_pred)  # smooth differentiable
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + SMOOTH_LOSS) / (K.sum(y_true_f + y_pred_f) - intersection + SMOOTH_LOSS)  # flatten
    # intersection = K.sum(y_true * y_pred, axis=sum_axis)
    # sum_ = K.sum(y_true + y_pred, axis=sum_axis)
    # return K.mean((intersection + SMOOTH_LOSS) / (sum_ - intersection + SMOOTH_LOSS))  # dimensional

def jac(y_true, y_pred):
    return jac_d(y_true, K.round(K.clip(y_pred, 0, 1)))  # integer call

def dice_d(y_true, y_pred):
    y_true_f, y_pred_f = K.flatten(y_true), K.flatten(y_pred)  # smooth differentiable
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH_LOSS) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH_LOSS)

def dice(y_true, y_pred):
    return dice_d(y_true, K.round(K.clip(y_pred, 0, 1)))  # integer call

def dice_80(y_true, y_pred):
    return dice(central_crop(y_true,0.8), central_crop(y_pred,0.8))
def dice_60(y_true, y_pred):
    return dice(central_crop(y_true,0.6), central_crop(y_pred,0.6))
def dice_40(y_true, y_pred):
    return dice(central_crop(y_true,0.4), central_crop(y_pred,0.4))
def dice_20(y_true, y_pred):
    return dice(central_crop(y_true,0.2), central_crop(y_pred,0.2))

def top5_acc(y_true, y_pred, k=5):  # top_N_categorical_accuracy
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)
def spar_acc(y_true, y_pred):  # sparse_categorical_accuracy
    return K.cast(K.equal(K.max(y_true, axis=-1), K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())
def acc(y_true, y_pred):  # default 'acc'
    return K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)),
                  K.floatx())
def acc_100(y_true, y_pred):
    return acc(y_true, y_pred)
def acc_80(y_true, y_pred):
    return acc(central_crop(y_true,0.8), central_crop(y_pred,0.8))
def acc_60(y_true, y_pred):
    return acc(central_crop(y_true,0.6), central_crop(y_pred,0.6))
def acc_40(y_true, y_pred):
    return acc(central_crop(y_true,0.4), central_crop(y_pred,0.4))
def acc_20(y_true, y_pred):
    return acc(central_crop(y_true,0.2), central_crop(y_pred,0.2))

def loss_bce(y_true, y_pred):  # bootstrapped binary cross entropy
    target_tensor = y_true
    prediction_tensor = y_pred
    _epsilon = _to_tensor(K.epsilon(), prediction_tensor.dtype.base_dtype)
    prediction_tensor = K.tf.clip_by_value(prediction_tensor, _epsilon, 1 - _epsilon)
    prediction_tensor = K.tf.log(prediction_tensor / (1 - prediction_tensor))
    alpha = 0.95
    # bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * K.unet_tf.sigmoid(prediction_tensor)  # soft bootstrap
    bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * K.tf.cast(K.tf.sigmoid(prediction_tensor) > 0.5, K.tf.float32)  # hard bootstrap
    return K.mean(K.tf.nn.sigmoid_cross_entropy_with_logits(labels=bootstrap_target_tensor, logits=prediction_tensor))

def loss_jac(y_true, y_pred):
    return 1. - jac_d(y_true, y_pred)

def loss_dice(y_true, y_pred):
    return 1. - dice_d(y_true, y_pred)

def loss_bce_dice(y_true, y_pred):
    return 0.5 * (loss_bce(y_true, y_pred) + loss_dice(y_true, y_pred))

def loss_jac_dice(y_true, y_pred):
    return loss_jac(y_true, y_pred) + loss_dice(y_true, y_pred)

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

    def __init__(self, func, cfg:ModelConfig, save):
        self.cfg=cfg
        self.model, self.name=func(cfg)
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
            metrics= [acc,acc_80,acc_60,acc_40,acc_20]\
                if self.cfg.model_loss is 'categorical_crossentropy' else [jac, dice, dice_80, dice_60,dice_40, dice_20])
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

        export_name = "%s_%s" % (multi.dir_out, self.name)
        weight_file = export_name + ".h5"
        if self.cfg.train_continue and os.path.exists(weight_file):
            print("Continue from previous weights")
            self.model.load_weights(weight_file)

        indicator, trend = ('val_dice', 'max') if cfg.dep_out==1 else ('val_acc', 'max')
        print('Fitting neural net...')
        for r in range(self.cfg.train_rep):
            print("Training %d/%d for %s" % (r + 1, self.cfg.train_rep, export_name))
            tr.on_epoch_end()
            val.on_epoch_end()
            history = self.model.fit_generator(tr, validation_data=val, verbose=1,
                steps_per_epoch=min(cfg.train_step, len(tr.view_coord)) if isinstance(cfg.train_step , int) else len(tr.view_coord),
               validation_steps=min(cfg.train_vali_step, len(val.view_coord)) if isinstance(cfg.train_vali_step, int) else len(val.view_coord),
                epochs=self.cfg.train_epoch, max_queue_size=1, workers=0, use_multiprocessing=False, shuffle=False,
                callbacks=[
                    ModelCheckpoint(weight_file, monitor=indicator, mode=trend, save_weights_only=False, save_best_only=True),
                    EarlyStopping(monitor=indicator, mode=trend, patience=1, verbose=1),
                    # ReduceLROnPlateau(monitor=indicator, mode=trend, factor=0.1, patience=10, min_delta=1e-5, cooldown=0, min_lr=0, verbose=1),
                    # TensorBoardTrainVal(log_dir=os.path.join("log", export_name), write_graph=True, write_grads=False, write_images=True),
                ]).history
            if not os.path.exists(export_name + ".txt"):
                with open(export_name + ".txt", "w") as net_summary:
                    self.model.summary(print_fn=lambda x: net_summary.write(x + '\n'))
            df=pd.DataFrame(history).round(4)
            df['time']=datetime.now().strftime("%Y-%m-%d %H:%M")
            df['repeat']=r+1
            df.to_csv(export_name + ".csv", mode="a", header=(not os.path.exists(export_name + ".csv")))

    def predict(self, multi:ImagePairMulti, xls_file):
        print('Load weights and predicting ...')
        export_name = "%s_%s" % (multi.dir_out, self.name)
        weight_file = export_name + ".h5"
        self.model.load_weights(weight_file)
        img_ext=self.cfg.image_format[1:]
        mrg_in,mrg_out,mrg_out_wt,merge_dir,mask_wt=None,None,None,None,None
        r_i, r_g, sum_g, res_i, res_g=None,None,None,None,None

        target_dir = os.path.join(multi.wd, export_name)
        mk_dir_if_nonexist(target_dir)
        sum_i = self.cfg.row_out * self.cfg.col_out
        batch=multi.img_set.view_coord_batch()  # image/1batch -> view_coord
        if self.cfg.separate:
            merge_dir = os.path.join(multi.wd, "%s_%s" % (multi.dir_out.split('_')[0], self.name))
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
                origin = view[i].get_image(os.path.join(multi.wd, multi.origin + multi.dir_in_ex), self.cfg)
                print(ind_name); text_list = [ind_name]
                blend, r_i=self.mask_call(origin, msk)
                for d in range(self.cfg.dep_out):
                    text = " [%d: %s] %d / %d  %.0f%%" % ( d, multi.targets[d], r_i[d], sum_i, 100. * r_i[d] / sum_i)
                    print(text); text_list.append(text)
                # cv2.imwrite(ind_file, msk[...,np.newaxis] * 255.)
                blend = self.draw_text(blend, '\n'.join(text_list))  # RGB:3x8-bit dark text
                imsave(ind_file.replace(img_ext, ".jpe"), blend)
                res_i = r_i if res_i is None else np.vstack((res_i, r_i))

                if self.cfg.separate:
                    mrg_in[view[i].row_start:view[i].row_end, view[i].col_start:view[i].col_end] = origin
                    for d in range(self.cfg.dep_out):
                        mrg_out[view[i].row_start:view[i].row_end, view[i].col_start:view[i].col_end, d] += msk[...,d] * mask_wt
                    mrg_out_wt[view[i].row_start:view[i].row_end, view[i].col_start:view[i].col_end] += mask_wt
            if self.cfg.separate:
                for d in range(self.cfg.dep_out):
                    mrg_out[...,d] /= mrg_out_wt
                print(grp); text_list=[grp]
                merge_name = view[0].image_name
                merge_file = os.path.join(merge_dir, merge_name)
                blend, r_g = self.mask_call(mrg_in, mrg_out)
                for d in range(self.cfg.dep_out):
                    text = " [%d: %s] %d / %d  %.0f%%" % (d, multi.targets[d], r_g[d], sum_g, 100. * r_g[d] / sum_g)
                    print(text); text_list.append(text)
                # cv2.imwrite(merge_file, mrg_out[..., np.newaxis] * 255.)
                blend = self.draw_text(blend, '\n'.join(text_list))  # RGB:3x8-bit dark text
                imsave(merge_file.replace(img_ext, ".jpe"), blend)
                res_g=r_g if res_g is None else np.vstack((res_g, r_g))
        df = pd.DataFrame(res_i, index=multi.img_set.images, columns=multi.targets)
        to_excel_sheet(df, xls_file, multi.origin)  # per slice
        if self.cfg.separate:
            df = pd.DataFrame(res_g, index=batch.keys(), columns=multi.targets)
            to_excel_sheet(df, xls_file, multi.origin + "_sum")


    def draw_text(self, img, text, mode='RGB', col=(10, 10, 10)):
        # origin*=255.
        # cv2.putText(origin,text.replace("Pixel","\nPixel"),(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.3,
        #             (255 if oc == 0 else 10, 255 if oc == 1 else 10, 255 if oc == 2 else 10), 1, cv2.LINE_AA, bottomLeftOrigin=False)
        # imwrite(ind_file, origin)
        origin = Image.fromarray(img.astype(np.uint8), mode)  # L RGB
        draw = ImageDraw.Draw(origin)
        draw.text((0, 0), text, col, ImageFont.truetype("arial.ttf", 22))  # font type size)
        return origin

    def mask_call(self, img, msk):  # blend, np result
        blend=img.copy()
        opa=self.cfg.overlay_opacity
        col=self.cfg.overlay_color
        dim=self.cfg.dep_out
        if dim==1: # sigmoid r x c x 1
            d=0
            msk=np.rint(msk[...,d])  # round to  0/1
            for c in range(3):
                blend[..., c] = np.where(msk >= 0.5, blend[..., c] * (1 - opa) + 255 * col[d][c] * opa, blend[..., c])
            return blend, np.sum(msk)
        else: # softmax r x c x multi_label
            msk=np.argmax(msk, axis=-1)
            uni, count=np.unique(msk, return_counts=True)
            map_count=dict(zip(uni,count))
            count_vec=[]
            for d in range(dim):
                count_vec.append(map_count.get(d) or 0)
                for c in range(3):
                    blend[..., c] = np.where(msk == d, blend[..., c] * (1 - opa) + 255 *col[d][c] * opa, blend[..., c])
            return blend, count_vec
