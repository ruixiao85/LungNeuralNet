from __future__ import print_function

import os
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D
from keras import backend as KerasBackend

KerasBackend.set_image_data_format('channels_last')
concat_axis = 3


def dice_coef(y_true, y_pred):
    Smooth = 0.05
    y_true_f = KerasBackend.flatten(y_true)
    y_pred_f = KerasBackend.flatten(y_pred)
    intersection = KerasBackend.sum(y_true_f * y_pred_f)
    return (2. * intersection + Smooth) / (KerasBackend.sum(y_true_f) + KerasBackend.sum(y_pred_f) + Smooth)

def dice_coef_loss(y_true, y_pred):
    return - dice_coef(y_true, y_pred)

def get_crop_shape(target, refer):
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value  # width, the 3rd dimension
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value  # height, the 2nd dimension
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)
    return (ch1, ch2), (cw1, cw2)


def get_unet3(img_rows, img_cols, dim_in, dim_out, act_fun='sigmoid'):
    inputs = Input((img_rows, img_cols, dim_in))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up_conv2 = UpSampling2D(size=(2, 2))(conv3)
    ch, cw = get_crop_shape(conv2, up_conv2)
    crop_conv2 = Cropping2D(cropping=(ch, cw))(conv2)
    up2 = concatenate([up_conv2, crop_conv2], axis=concat_axis)
    decon2 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    decon2 = Conv2D(64, (3, 3), activation='relu', padding='same')(decon2)
    up_conv1 = UpSampling2D(size=(2, 2))(decon2)
    ch, cw = get_crop_shape(conv1, up_conv1)
    crop_conv1 = Cropping2D(cropping=(ch, cw))(conv1)
    up1 = concatenate([up_conv1, crop_conv1], axis=concat_axis)
    decon1 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    decon1 = Conv2D(32, (3, 3), activation='relu', padding='same')(decon1)
    ch, cw = get_crop_shape(inputs, decon1)
    decon1 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(decon1)
    outputs = Conv2D(dim_out, (1, 1), activation=act_fun)(decon1)

    return Model(inputs=inputs, outputs=outputs)


def get_unet4(img_rows, img_cols, dim_in, dim_out, act_fun='sigmoid'):
    inputs = Input((img_rows, img_cols, dim_in))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

    up_conv3 = UpSampling2D(size=(2, 2))(conv4)
    ch, cw = get_crop_shape(conv3, up_conv3)
    crop_conv3 = Cropping2D(cropping=(ch, cw))(conv3)
    up3 = concatenate([up_conv3, crop_conv3], axis=concat_axis)
    decon3 = Conv2D(128, (3, 3), activation='relu', padding='same')(up3)
    decon3 = Conv2D(128, (3, 3), activation='relu', padding='same')(decon3)
    up_conv2 = UpSampling2D(size=(2, 2))(decon3)
    ch, cw = get_crop_shape(conv2, up_conv2)
    crop_conv2 = Cropping2D(cropping=(ch, cw))(conv2)
    up2 = concatenate([up_conv2, crop_conv2], axis=concat_axis)
    decon2 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    decon2 = Conv2D(64, (3, 3), activation='relu', padding='same')(decon2)
    up_conv1 = UpSampling2D(size=(2, 2))(decon2)
    ch, cw = get_crop_shape(conv1, up_conv1)
    crop_conv1 = Cropping2D(cropping=(ch, cw))(conv1)
    up1 = concatenate([up_conv1, crop_conv1], axis=concat_axis)
    decon1 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    decon1 = Conv2D(32, (3, 3), activation='relu', padding='same')(decon1)

    ch, cw = get_crop_shape(inputs, decon1)
    decon1 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(decon1)
    outputs = Conv2D(dim_out, (1, 1), activation=act_fun)(decon1)

    return Model(inputs=inputs, outputs=outputs)


def get_unet5(img_rows, img_cols, dim_in, dim_out, act_fun='sigmoid'):
    inputs = Input((img_rows, img_cols, dim_in))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up_conv4 = UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv4)
    crop_conv4 = Cropping2D(cropping=(ch, cw))(conv4)
    up4 = concatenate([up_conv4, crop_conv4], axis=concat_axis)
    decon4 = Conv2D(256, (3, 3), activation='relu', padding='same')(up4)
    decon4 = Conv2D(256, (3, 3), activation='relu', padding='same')(decon4)
    up_conv3 = UpSampling2D(size=(2, 2))(decon4)
    ch, cw = get_crop_shape(conv3, up_conv3)
    crop_conv3 = Cropping2D(cropping=(ch, cw))(conv3)
    up3 = concatenate([up_conv3, crop_conv3], axis=concat_axis)
    decon3 = Conv2D(128, (3, 3), activation='relu', padding='same')(up3)
    decon3 = Conv2D(128, (3, 3), activation='relu', padding='same')(decon3)
    up_conv2 = UpSampling2D(size=(2, 2))(decon3)
    ch, cw = get_crop_shape(conv2, up_conv2)
    crop_conv2 = Cropping2D(cropping=(ch, cw))(conv2)
    up2 = concatenate([up_conv2, crop_conv2], axis=concat_axis)
    decon2 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    decon2 = Conv2D(64, (3, 3), activation='relu', padding='same')(decon2)
    up_conv1 = UpSampling2D(size=(2, 2))(decon2)
    ch, cw = get_crop_shape(conv1, up_conv1)
    crop_conv1 = Cropping2D(cropping=(ch, cw))(conv1)
    up1 = concatenate([up_conv1, crop_conv1], axis=concat_axis)
    decon1 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    decon1 = Conv2D(32, (3, 3), activation='relu', padding='same')(decon1)

    ch, cw = get_crop_shape(inputs, decon1)
    decon1 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(decon1)
    outputs = Conv2D(dim_out, (1, 1), activation=act_fun)(decon1)

    return Model(inputs=inputs, outputs=outputs)


def get_unet6(img_rows, img_cols, dim_in, dim_out, act_fun='sigmoid'):
    inputs = Input((img_rows, img_cols, dim_in))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool5)
    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)

    up_conv5 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv5, up_conv5)
    crop_conv5 = Cropping2D(cropping=(ch, cw))(conv5)
    up5 = concatenate([up_conv5, crop_conv5], axis=concat_axis)
    decon5 = Conv2D(512, (3, 3), activation='relu', padding='same')(up5)
    decon5 = Conv2D(512, (3, 3), activation='relu', padding='same')(decon5)
    up_conv4 = UpSampling2D(size=(2, 2))(decon5)
    ch, cw = get_crop_shape(conv4, up_conv4)
    crop_conv4 = Cropping2D(cropping=(ch, cw))(conv4)
    up4 = concatenate([up_conv4, crop_conv4], axis=concat_axis)
    decon4 = Conv2D(256, (3, 3), activation='relu', padding='same')(up4)
    decon4 = Conv2D(256, (3, 3), activation='relu', padding='same')(decon4)
    up_conv3 = UpSampling2D(size=(2, 2))(decon4)
    ch, cw = get_crop_shape(conv3, up_conv3)
    crop_conv3 = Cropping2D(cropping=(ch, cw))(conv3)
    up3 = concatenate([up_conv3, crop_conv3], axis=concat_axis)
    decon3 = Conv2D(128, (3, 3), activation='relu', padding='same')(up3)
    decon3 = Conv2D(128, (3, 3), activation='relu', padding='same')(decon3)
    up_conv2 = UpSampling2D(size=(2, 2))(decon3)
    ch, cw = get_crop_shape(conv2, up_conv2)
    crop_conv2 = Cropping2D(cropping=(ch, cw))(conv2)
    up2 = concatenate([up_conv2, crop_conv2], axis=concat_axis)
    decon2 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    decon2 = Conv2D(64, (3, 3), activation='relu', padding='same')(decon2)
    up_conv1 = UpSampling2D(size=(2, 2))(decon2)
    ch, cw = get_crop_shape(conv1, up_conv1)
    crop_conv1 = Cropping2D(cropping=(ch, cw))(conv1)
    up1 = concatenate([up_conv1, crop_conv1], axis=concat_axis)
    decon1 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    decon1 = Conv2D(32, (3, 3), activation='relu', padding='same')(decon1)

    ch, cw = get_crop_shape(inputs, decon1)
    decon1 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(decon1)
    outputs = Conv2D(dim_out, (1, 1), activation=act_fun)(decon1)

    return Model(inputs=inputs, outputs=outputs)
