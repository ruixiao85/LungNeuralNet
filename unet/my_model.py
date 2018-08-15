import cv2
import os

from datetime import datetime
import numpy as np
import pandas as pd
from PIL import ImageDraw, Image, ImageFont
from keras import backend as K
from keras.backend.tensorflow_backend import _to_tensor
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.engine.saving import model_from_json
from skimage.io import imsave
from scipy import signal
from image_gen import ImageTrainPair, ImagePredictPair, MetaInfo, ImagePredictGenerator
from model_config import ModelConfig
from process_image import scale_input, scale_input_reverse
from tensorboard_train_val import TensorBoardTrainVal
from util import mk_dir_if_nonexist

SMOOTH_LOSS = 1e-5

def gkern(size,sigma):
    gkern1d = signal.gaussian(size, std=sigma).reshape(size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def g_kern(row, col):
    l=max(row,col)
    mat=gkern(l,int(l/2))
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

from tensorflow.python.ops.image_ops_impl import central_crop
def dice_90(y_true, y_pred):
    return dice(central_crop(y_true,0.9), central_crop(y_pred,0.9))

def dice_80(y_true, y_pred):
    return dice(central_crop(y_true,0.8), central_crop(y_pred,0.8))

def dice_70(y_true, y_pred):
    return dice(central_crop(y_true,0.7), central_crop(y_pred,0.7))

def dice_60(y_true, y_pred):
    return dice(central_crop(y_true,0.6), central_crop(y_pred,0.6))

def dice_50(y_true, y_pred):
    return dice(central_crop(y_true,0.5), central_crop(y_pred,0.5))

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


def blend_mask(origin, image, channel, opacity):
    origin=scale_input(origin)
    for c in range(3):
        if c == channel:
            origin[..., c] = np.tanh(origin[..., c] + opacity * image[..., 0])
        else:
            origin[..., c] = np.tanh(origin[..., c] - opacity * image[..., 0])
    return scale_input_reverse(origin)

def draw_text(image,text,mode='RGB',col=(10,10,10)):
    # origin*=255.
    # cv2.putText(origin,text.replace("Pixel","\nPixel"),(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.3,
    #             (255 if oc == 0 else 10, 255 if oc == 1 else 10, 255 if oc == 2 else 10), 1, cv2.LINE_AA, bottomLeftOrigin=False)
    # imwrite(ind_file, origin)
    origin = Image.fromarray(image.astype(np.uint8), mode) # L RGB
    draw = ImageDraw.Draw(origin)
    draw.text((0, 0), text, col, ImageFont.truetype("arial.ttf", 22))  # font type size)
    return origin


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

    #     out_fun = 'softmax'   loss_fun='categorical_crossentropy'
    #     out_fun='sigmoid'    loss_fun=[loss_bce_dice] 'binary_crossentropy' "bcedice"

    def __init__(self, func, cfg:ModelConfig, save):
        self.continue_train = cfg.continue_train
        self.num_rep = cfg.num_rep
        self.num_epoch = cfg.num_epoch
        self.model, self.name=None, None
        self.model, self.name=func(cfg)
        self.learning_rate=cfg.learning_rate
        self.loss_fun=cfg.loss_fun
        self.compile_model()
        if save:
            self.save_model()
        self.mrg_in, self.mrg_out=None, None  # merge input/output
        self.mrg_in_wt, self.mrg_out_wt=None, None  # weight matrix of merge input/output
        self.separate=cfg.separate
        self.overlay_channel=cfg.overlay_channel
        self.overlay_opacity=cfg.overlay_opacity
        self.call_hardness=cfg.call_hardness
        self.mask_wt = None

    def load_model(self, name:str):  # load model
        self.name=name
        with open(self.name + ".json", 'r') as json_file:
            self.model = model_from_json(json_file.read())
        self.compile_model()

    def __str__(self):
        return self.name

    def compile_model(self):
        from keras.optimizers import Adam
        self.model.compile(optimizer=Adam(self.learning_rate), loss=self.loss_fun, metrics=[jac, dice, dice_90, dice_80, dice_70, dice_60, dice_50])
        self.model.summary()

    def save_model(self):
        model_json = self.name + ".json"
        with open(model_json, "w") as json_file:
            json_file.write(self.model.to_json())

    # def get_export_name(self, pair):
    #     return "%s-%s_%s" % (pair.dir_out, pair.dir_in, self.name)

    def train(self, pair:ImageTrainPair):
        print('Generate iterable data set...')
        tr, val=pair.get_tr_val_generator()

        export_name = "%s_%s" % (tr.dir_out, self.name)
        weight_file = export_name + ".h5"
        if self.continue_train and os.path.exists(weight_file):
            print("Continue from previous weights")
            self.model.load_weights(weight_file)

        # indicator, trend = 'val_loss', 'min'
        indicator, trend = 'val_dice', 'max'
        print('Fitting neural net...')
        for r in range(self.num_rep):
            print("Training %d/%d for %s" % (r + 1, self.num_rep, export_name))
            tr.on_epoch_end()
            val.on_epoch_end()
            history = self.model.fit_generator(tr, validation_data=val,
                # steps_per_epoch=min(100, len(tr.view_coord)), validation_steps=min(30, len(val.view_coord)),
                steps_per_epoch=len(tr.view_coord), validation_steps=len(val.view_coord),
                epochs=self.num_epoch, max_queue_size=1, workers=0, use_multiprocessing=False, shuffle=False,
                callbacks=[
                    ModelCheckpoint(weight_file, monitor=indicator, mode=trend, save_best_only=True),
                    EarlyStopping(monitor=indicator, mode=trend, patience=1, verbose=1),
                    # ReduceLROnPlateau(monitor=indicator, mode=trend, factor=0.1, patience=10, min_delta=1e-5, cooldown=0, min_lr=0, verbose=1),
                    # TensorBoardTrainVal(log_dir=os.path.join("log", export_name), write_graph=True, write_grads=False, write_images=True),
                ]).history
            if not os.path.exists(export_name + ".txt"):
                with open(export_name + ".txt", "w") as net_summary:
                    self.model.summary(print_fn=lambda x: net_summary.write(x + '\n'))
            df=pd.DataFrame(history)
            df['time']=datetime.now().strftime("%Y-%m-%d %H:%M")
            df['repeat']=r+1
            df.to_csv(export_name + ".csv", mode="a", header=(not os.path.exists(export_name + ".csv")))

    def predict(self, pair:ImagePredictPair):
        i_sum=pair.row_out*pair.col_out
        res_i=np.zeros(len(pair.img_set.images),dtype=np.uint32)
        res_g=np.zeros(len(pair.img_set.groups),dtype=np.uint32)
        prd:ImagePredictGenerator = pair.get_prd_generator()

        print('Load weights and predicting ...')
        export_name = "%s_%s" % (prd.dir_out, self.name)
        weight_file = export_name + ".h5"
        self.model.load_weights(weight_file)

        msks = self.model.predict_generator(prd, max_queue_size=1, workers=0,use_multiprocessing=False, verbose=1)
        target_dir = os.path.join(pair.wd, export_name)
        print('Saving predicted results [%s] to files...' % export_name)
        mk_dir_if_nonexist(target_dir)
        merge_dir = os.path.join(pair.wd, "%s_%s" % (prd.dir_out.split('_')[0], self.name))
        if self.separate:
            mk_dir_if_nonexist(merge_dir)
        for i, msk in enumerate(msks):
            ind_name=prd.view_coord[i].file_name
            ind_file = os.path.join(target_dir, ind_name)
            msk, i_val = self.msk_call(msk, self.call_hardness)
            res_i[i]=i_val
            text="%s \nPixels: %.0f / %.0f Percentage: %.0f%%" % (ind_name, i_val, i_sum, 100. * i_val / i_sum)
            print(text)
            cv2.imwrite(ind_file, msk*255.)
            # cv2.imwrite(ind_file, draw_text((msk*255.)[...,0],text.replace("Pixel","\nPixel"),mode='L')) # L:8-bit B&W gray text
            origin=prd.view_coord[i].get_image(os.path.join(pair.wd, pair.dir_in_ex()),pair.separate)
            if self.separate:
                self.merge_images(pair.view_coord, i, origin, msk, merge_dir, res_g, pair.img_set.groups)
            blend=blend_mask(origin,msk,channel=self.overlay_channel,opacity=self.overlay_opacity)
            markup=draw_text(blend,text) # RGB:3x8-bit dark text
            imsave(ind_file.replace(".jpg",".jpe"), markup)
        return res_i, res_g

    @staticmethod
    def msk_call(msk, ch):
        if ch == 1:  # hard sign
            msk = np.rint(msk)
        elif 0 < ch < 1:
            msk = (msk + np.rint(msk) * ch) / (1.0 + ch)  # mixed
        return msk, int(np.sum(msk[..., 0]))

    def merge_images(self, views, idx, img, msk, folder, res, groups):
        view:MetaInfo=views[idx]
        this_file=view.image_name
        last_file=None if idx<1 else views[idx-1].image_name
        next_file=views[idx+1].image_name if idx<len(views)-1 else "LastImage!@#$%^&*()_+"
        mrg_dep, msk_dep=3, 1
        if this_file != last_file:  # create new
            self.mrg_in=np.zeros((view.ori_row,view.ori_col,mrg_dep),dtype=np.float32)
            self.mrg_in_wt=np.zeros((view.ori_row,view.ori_col),dtype=np.float32)
            self.mrg_out=np.zeros((view.ori_row,view.ori_col,msk_dep),dtype=np.float32)
            self.mrg_out_wt=np.zeros((view.ori_row,view.ori_col),dtype=np.float32)
        # insert image
        if self.mask_wt is None or self.mask_wt.shape!=self.mrg_out_wt.shape:
            self.mask_wt=g_kern(view.row_end-view.row_start,view.col_end-view.col_start)
        self.mrg_in[view.row_start:view.row_end, view.col_start:view.col_end,...] += img
        self.mrg_in_wt[view.row_start:view.row_end, view.col_start:view.col_end] += 1.0
        for d in range(msk_dep):
            self.mrg_out[view.row_start:view.row_end, view.col_start:view.col_end, d] += msk[...,d]*self.mask_wt
        self.mrg_out_wt[view.row_start:view.row_end, view.col_start:view.col_end] += self.mask_wt
        if this_file!=next_file:  # export new
            for d in range(mrg_dep):
                self.mrg_in[...,d]/=self.mrg_in_wt
            for d in range(msk_dep):
                self.mrg_out[...,d]/=self.mrg_out_wt
            self.mrg_in_wt, self.mrg_out_wt=None,None
            self.mrg_out, _val=self.msk_call(self.mrg_out, self.call_hardness)
            _sum=view.ori_row*view.ori_col
            text = "%s %dx%d \nPixels: %.0f / %.0f Percentage: %.0f%%" % (view.image_name, view.ori_row, view.ori_col, _val, _sum, 100. * _val / _sum)
            print(text)
            res[groups.index(view.image_name)]=_val
            merge_file=os.path.join(folder, this_file)
            cv2.imwrite(merge_file, self.mrg_out * 255.)
            # cv2.imwrite(merge_file, draw_text((msk*255.)[...,0],text.replace("Pixel","\nPixel"),mode='L')) # L:8-bit B&W gray text
            blend = blend_mask(self.mrg_in, self.mrg_out, channel=self.overlay_channel, opacity=self.overlay_opacity)
            markup = draw_text(blend, text)  # RGB:3x8-bit dark text
            imsave(merge_file.replace(".jpg", ".jpe"), markup)
            self.mrg_in,self.mrg_out=None,None

