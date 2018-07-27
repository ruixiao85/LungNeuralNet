import numpy as np
from keras.applications.vgg16 import VGG16
from keras.engine.training import Model
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose, Cropping2D, ZeroPadding2D
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, merge
from keras import backend as K
from keras.backend.tensorflow_backend import _to_tensor
from model.metrics import loss_bce_dice, loss_bce, loss_dice, loss_jaccard

K.set_image_data_format("channels_last")
#K.set_image_dim_ordering("th")
concat_axis = 3

def get_unet5_trans(img_rows, img_cols, dim_in, dim_out, act_fun='elu', out_fun='sigmoid', init='he_normal'):
    name="_unet5_trans"
    f1, f2, f3, f4, f5 = 64, 96, 128, 192, 256
    # f1, f2, f3, f4, f5 = 128, 192, 256, 384, 512
    # img_input = Input((None, None, dim_in))  # r,c,3
    img_input=Input((img_rows, img_cols, dim_in))  # r,c,3

    conv1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(img_input)
    conv1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv1)
    pool1 = MaxPooling2D((2,2), strides=(2, 2))(conv1)
    conv2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool1)
    conv2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)
    conv3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool2)
    conv3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)
    conv4 = Conv2D(f4, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool3)
    conv4 = Conv2D(f4, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)
    conv5 = Conv2D(f5, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool4)
    conv5 = Conv2D(f5, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv5)

    up4 = concatenate([conv4, Conv2DTranspose(
        f4, (3, 3), activation=act_fun, kernel_initializer=init, strides=(2, 2),padding='same')(conv5)], axis=concat_axis)
    decon4 = Conv2D(f4, (3, 3), activation=act_fun, kernel_initializer=init, padding='same')(up4)
    up3 = concatenate([conv3, Conv2DTranspose(
        f3, (3, 3), activation=act_fun, kernel_initializer=init, strides=(2, 2), padding='same')(decon4)], axis=concat_axis)
    decon3 = Conv2D(f3, (3, 3), activation=act_fun, kernel_initializer=init, padding='same')(up3)
    up2 = concatenate([conv2, Conv2DTranspose(
        f2, (3, 3), activation=act_fun, kernel_initializer=init, strides=(2, 2), padding='same')(decon3)], axis=concat_axis)
    decon2 = Conv2D(f2, (3, 3), activation=act_fun, kernel_initializer=init, padding='same')(up2)
    up1 = concatenate([conv1, Conv2DTranspose(
        f1, (3, 3), activation=act_fun, kernel_initializer=init, strides=(2, 2), padding='same')(decon2)], axis=concat_axis)
    decon1 = Conv2D(f1, (3, 3), activation=act_fun, kernel_initializer=init, padding='same')(up1)
    # decon1 = BatchNormalization(mode=0, axis=1)(decon1)
    decon1 = Conv2D(dim_out, (1, 1), activation=out_fun)(decon1)
    return Model(img_input, decon1), name


def get_unet7_trans(img_rows, img_cols, dim_in, dim_out, act_fun='elu', out_fun='sigmoid', init='he_normal'):
    name="_unet7_trans"
    f1, f2, f3, f4, f5, f6, f7 = 32, 64, 96, 128, 192, 256, 384
    # f1, f2, f3, f4, f5, f6, f7 = 64, 96, 128, 192, 256, 384, 512
    # img_input = Input((None, None, dim_in))  # r,c,3
    img_input=Input((img_rows, img_cols, dim_in))  # r,c,3

    conv1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(img_input)
    conv1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv1)
    pool1 = MaxPooling2D((2,2), strides=(2, 2))(conv1)
    conv2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool1)
    conv2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)
    conv3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool2)
    conv3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)
    conv4 = Conv2D(f4, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool3)
    conv4 = Conv2D(f4, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)
    conv5 = Conv2D(f5, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool4)
    conv5 = Conv2D(f5, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv5)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5)
    conv6 = Conv2D(f6, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool5)
    conv6 = Conv2D(f6, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv6)
    pool6 = MaxPooling2D((2, 2), strides=(2, 2))(conv6)
    conv7 = Conv2D(f7, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool6)
    conv7 = Conv2D(f7, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv7)

    up6 = concatenate([conv6, Conv2DTranspose(
        f6, (3, 3), activation=act_fun, kernel_initializer=init, strides=(2, 2), padding='same')(conv7)], axis=concat_axis)
    decon6 = Conv2D(f6, (3, 3), activation=act_fun, kernel_initializer=init, padding='same')(up6)
    up5 = concatenate([conv5, Conv2DTranspose(
        f5, (3, 3), activation=act_fun, kernel_initializer=init, strides=(2, 2), padding='same')(decon6)], axis=concat_axis)
    decon5 = Conv2D(f5, (3, 3), activation=act_fun, kernel_initializer=init, padding='same')(up5)
    up4 = concatenate([conv4, Conv2DTranspose(
        f4, (3, 3), activation=act_fun, kernel_initializer=init, strides=(2, 2),padding='same')(decon5)], axis=concat_axis)
    decon4 = Conv2D(f4, (3, 3), activation=act_fun, kernel_initializer=init, padding='same')(up4)
    up3 = concatenate([conv3, Conv2DTranspose(
        f3, (3, 3), activation=act_fun, kernel_initializer=init, strides=(2, 2), padding='same')(decon4)], axis=concat_axis)
    decon3 = Conv2D(f3, (3, 3), activation=act_fun, kernel_initializer=init, padding='same')(up3)
    up2 = concatenate([conv2, Conv2DTranspose(
        f2, (3, 3), activation=act_fun, kernel_initializer=init, strides=(2, 2), padding='same')(decon3)], axis=concat_axis)
    decon2 = Conv2D(f2, (3, 3), activation=act_fun, kernel_initializer=init, padding='same')(up2)
    up1 = concatenate([conv1, Conv2DTranspose(
        f1, (3, 3), activation=act_fun, kernel_initializer=init, strides=(2, 2), padding='same')(decon2)], axis=concat_axis)
    decon1 = Conv2D(f1, (3, 3), activation=act_fun, kernel_initializer=init, padding='same')(up1)
    # decon1 = BatchNormalization(mode=0, axis=1)(decon1)
    decon1 = Conv2D(dim_out, (1, 1), activation=out_fun)(decon1)
    return Model(img_input, decon1), name

def get_unet_trans_compiled(func, img_rows, img_cols, dim_in, dim_out, act_fun='elu', out_fun='sigmoid'):
    model, name = func(img_rows, img_cols, dim_in, dim_out, act_fun, out_fun)
    from keras.optimizers import Adam
    from model.metrics import loss_dice, loss_jaccard, loss_bce,jaccard_coef, jaccard_coef_int, dice_coef_flat, dice_coef_flat_int
    model.compile(optimizer=Adam(1e-5),
                  # loss=[loss_bce],  # 'binary_crossentropy' "bcedice"
                  # loss=[loss_jaccard],  # 'binary_crossentropy' "bcedice"
                  loss=[loss_dice],  # 'binary_crossentropy' "bcedice"
                  # loss=[loss_bce_dice],  # 'binary_crossentropy' "bcedice"
                  metrics=[jaccard_coef_int, dice_coef_flat_int])  #
    return model,name
