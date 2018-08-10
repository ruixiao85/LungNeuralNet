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

from image_gen import ImageTrainPair, ImagePredictPair
from model_config import ModelConfig
from process_image import scale_input
from tensorboard_train_val import TensorBoardTrainVal
from util import mk_dir_if_nonexist

SMOOTH_LOSS = 1e-5
def gkern(l=5, sig=1.):  # creates gaussian kernel with side length l and a sigma of sig
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sig ** 2))
    return kernel / np.sum(kernel)

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

    def load_model(self, name:str):  # load model
        self.name=name
        with open(self.name + ".json", 'r') as json_file:
            self.model = model_from_json(json_file.read())
        self.compile_model()

    def __str__(self):
        return self.name

    def compile_model(self):
        from keras.optimizers import Adam
        self.model.compile(optimizer=Adam(self.learning_rate), loss=self.loss_fun, metrics=[jac, dice])
        self.model.summary()

    def save_model(self):
        model_json = self.name + ".json"
        with open(model_json, "w") as json_file:
            json_file.write(self.model.to_json())

    def get_export_name(self, pair):
        return "%s-%s_%s" % (pair.dir_out, pair.dir_in, self.name)

    def train(self, pair:ImageTrainPair):
        export_name = self.get_export_name(pair)
        weight_file = export_name + ".h5"
        if self.continue_train and os.path.exists(weight_file):
            print("Continue from previous weights")
            self.model.load_weights(weight_file)

        print('Generate iterable data set...')
        tr, val=pair.get_tr_val_generator()
        # indicator, trend = 'val_loss', 'min'
        indicator, trend = 'val_dice', 'max'
        print('Fitting neural net...')
        for r in range(self.num_rep):
            print("Training %d/%d for %s" % (r + 1, self.num_rep, export_name))
            tr.on_epoch_end()
            val.on_epoch_end()
            history = self.model.fit_generator(tr, validation_data=val,
                steps_per_epoch=min(100, len(tr.view_coord)), validation_steps=min(30, len(val.view_coord)),
                epochs=self.num_epoch, max_queue_size=1, workers=1, use_multiprocessing=False, shuffle=False,
                callbacks=[
                    ModelCheckpoint(weight_file, monitor=indicator, mode=trend, save_best_only=True),
                    ReduceLROnPlateau(monitor=indicator, mode=trend, factor=0.1, patience=10, min_delta=1e-5, cooldown=0, min_lr=0, verbose=1),
                    EarlyStopping(monitor=indicator, mode=trend, patience=0, verbose=1),
                    TensorBoardTrainVal(log_dir=os.path.join("log", export_name), write_graph=True, write_grads=False, write_images=True),
                ]).history
            with open(export_name + ".csv", "a") as log:
                log.write("\n" + datetime.now().strftime("%Y-%m-%d %H:%M") + " train history:\n")
            pd.DataFrame(history).to_csv(export_name + ".csv", mode="a")


    def predict(self, pair:ImagePredictPair):
        oc, op, ch = pair.overlay_channel, pair.overlay_opacity, pair.call_hardness
        i_sum=pair.row_out*pair.col_out
        print('Load weights and predicting ...')
        export_name = self.get_export_name(pair)
        self.model.load_weights(export_name + ".h5")
        prd = pair.get_prd_generator()
        # prd.on_epoch_end()
        msk = self.model.predict_generator(prd, verbose=1)

        target_dir = os.path.join(pair.wd, export_name)
        print('Saving predicted results [%s] to files...' % export_name)
        mk_dir_if_nonexist(target_dir)
        for i, image in enumerate(msk):
            ind_name=prd.view_coord[i].file_slice()
            ind_file = os.path.join(target_dir, ind_name)
            if ch==1:  # hard sign
                image = np.rint(image)
            elif 0<ch<1:
                image=(image+np.rint(image)*ch)/(1.0+ch)  # mixed
            i_val=int(np.sum(image[..., 0]))
            text="%s Pixels: %.0f / %.0f Percentage: %.0f%%" % (ind_name, i_val, i_sum, 100. * i_val / i_sum)
            print(text)
            _ind=scale_input(prd.view_coord[i].get_image(os.path.join(pair.wd, pair.dir_in_ex())))
            for c in range(3):
                if c == oc:
                    _ind[..., c] = np.tanh(_ind[..., c] + op * image[..., 0])
                else:
                    _ind[..., c] = np.tanh(_ind[..., c] - op * image[..., 0])
            # _ind*=255.
            # cv2.putText(_ind,text.replace("Pixel","\nPixel"),(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.3,
            #             (255 if oc == 0 else 10, 255 if oc == 1 else 10, 255 if oc == 2 else 10), 1, cv2.LINE_AA, bottomLeftOrigin=False)
            # imwrite(ind_file, _ind)
            _ind = Image.fromarray(((_ind + 1.0) * 127.).astype(np.uint8), 'RGB')
            draw = ImageDraw.Draw(_ind)
            draw.text((0, 0), text.replace("Pixel", "\nPixel"),
                      (10 if oc == 0 else 200, 10 if oc == 1 else 200, 10 if oc == 2 else 200), # contrasting color
                      ImageFont.truetype("arial.ttf", 24))  # font type size)
            imsave(ind_file, _ind)

    def merge_images(self):
        # if whole_file is not None and new_whole_file != whole_file:  # export whole_file
        #     text = "%s \n Pixels: %.0f / %.0f Percentage: %.0f%%" % (whole_file, _val, _sum, 100. * _val / _sum)
        #     print(text)
        #     if w_whole:  # write wholes image
        #         _whole = Image.fromarray(((_whole / _weight + 1.0) * 127.).astype(np.uint8), 'RGB')
        #         draw = ImageDraw.Draw(_whole)
        #         draw.text((0, 0), text,
        #                   (255 if oc == 0 else 10, 255 if oc == 1 else 10, 255 if oc == 2 else 10),
        #                   ImageFont.truetype("arial.ttf", 24))  # font type size)
        #         mk_dir_if_nonexist(os.path.dirname(whole_file))
        #         imsave(whole_file, _whole)
        #     _whole, _weight, _val, _sum = None, None, 0, 0
        # whole_file = os.path.join(target_dir, prd.view_coord[i].file_slice)
        pass