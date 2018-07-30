from __future__ import print_function

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D
from keras import backend as K

K.set_image_data_format('channels_last')
concat_axis = 3

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

def unet_pool_up_4(img_rows, img_cols, dim_in, dim_out, act_fun='elu', out_fun='sigmoid', init='he_normal'):
    name="_unet_pool_up_4"
    f1, f2, f3, f4 = 32, 64, 128, 256
    # f1, f2, f3, f4 = 64, 128, 256, 512
    # img_input = Input((None, None, dim_in))  # r,c,3
    img_input=Input((img_rows, img_cols, dim_in))  # r,c,3

    conv1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(img_input)
    conv1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool1)
    conv2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool2)
    conv3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(f4, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool3)
    conv4 = Conv2D(f4, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv4)

    up_conv3 = UpSampling2D(size=(2, 2))(conv4)
    up3 = concatenate([up_conv3, Cropping2D(cropping=(get_crop_shape(conv3, up_conv3)))(conv3)], axis=concat_axis)
    decon3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(up3)
    decon3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(decon3)
    up_conv2 = UpSampling2D(size=(2, 2))(decon3)
    up2 = concatenate([up_conv2, Cropping2D(cropping=(get_crop_shape(conv2, up_conv2)))(conv2)], axis=concat_axis)
    decon2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(up2)
    decon2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(decon2)
    up_conv1 = UpSampling2D(size=(2, 2))(decon2)
    up1 = concatenate([up_conv1, Cropping2D(cropping=(get_crop_shape(conv1, up_conv1)))(conv1)], axis=concat_axis)
    decon1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(up1)
    decon1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(decon1)

    ch, cw = get_crop_shape(img_input, decon1)
    decon1 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(decon1)
    # decon1 = BatchNormalization(mode=0, axis=concat_axis)(decon1)  # Batch normalization
    outputs = Conv2D(dim_out, (1, 1), activation=out_fun)(decon1)
    return Model(inputs=img_input, outputs=outputs), name

def unet_pool_up_5(img_rows, img_cols, dim_in, dim_out, act_fun='elu', out_fun='sigmoid', init='he_normal'):
    name="_unet_pool_up_5"
    f1, f2, f3, f4, f5 = 64, 96, 128, 192, 256
    # f1, f2, f3, f4, f5 = 128, 192, 256, 384, 512
    # img_input = Input((None, None, dim_in))  # r,c,3
    img_input=Input((img_rows, img_cols, dim_in))  # r,c,3

    conv1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(img_input)
    conv1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool1)
    conv2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool2)
    conv3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(f4, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool3)
    conv4 = Conv2D(f4, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(f5, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool4)
    conv5 = Conv2D(f5, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv5)

    up_conv4 = UpSampling2D(size=(2, 2))(conv5)
    up4 = concatenate([up_conv4, Cropping2D(cropping=(get_crop_shape(conv4, up_conv4)))(conv4)], axis=concat_axis)
    decon4 = Conv2D(f4, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(up4)
    decon4 = Conv2D(f4, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(decon4)
    up_conv3 = UpSampling2D(size=(2, 2))(decon4)
    up3 = concatenate([up_conv3, Cropping2D(cropping=(get_crop_shape(conv3, up_conv3)))(conv3)], axis=concat_axis)
    decon3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(up3)
    decon3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(decon3)
    up_conv2 = UpSampling2D(size=(2, 2))(decon3)
    up2 = concatenate([up_conv2, Cropping2D(cropping=(get_crop_shape(conv2, up_conv2)))(conv2)], axis=concat_axis)
    decon2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(up2)
    decon2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(decon2)
    up_conv1 = UpSampling2D(size=(2, 2))(decon2)
    up1 = concatenate([up_conv1, Cropping2D(cropping=(get_crop_shape(conv1, up_conv1)))(conv1)], axis=concat_axis)
    decon1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(up1)
    decon1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(decon1)

    ch, cw = get_crop_shape(img_input, decon1)
    decon1 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(decon1)
    # decon1 = BatchNormalization(mode=0, axis=concat_axis)(decon1)  # Batch normalization
    outputs = Conv2D(dim_out, (1, 1), activation=out_fun)(decon1)
    return Model(inputs=img_input, outputs=outputs), name

def unet_pool_up_6(img_rows, img_cols, dim_in, dim_out, act_fun='elu', out_fun='sigmoid', init='he_normal'):
    name="_unet_pool_up_6"
    f1, f2, f3, f4, f5, f6 = 64, 96, 128, 192, 256, 384
    # f1, f2, f3, f4, f5, f6 = 96, 128, 192, 256, 384, 512
    # img_input = Input((None, None, dim_in))  # r,c,3
    img_input=Input((img_rows, img_cols, dim_in))  # r,c,3

    conv1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(img_input)
    conv1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool1)
    conv2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool2)
    conv3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(f4, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool3)
    conv4 = Conv2D(f4, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(f5, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool4)
    conv5 = Conv2D(f5, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    conv6 = Conv2D(f6, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool5)
    conv6 = Conv2D(f6, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv6)
    up_conv5 = UpSampling2D(size=(2, 2))(conv6)
    up5 = concatenate([up_conv5, Cropping2D(cropping=(get_crop_shape(conv5, up_conv5)))(conv5)], axis=concat_axis)
    decon5 = Conv2D(f5, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(up5)
    decon5 = Conv2D(f5, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(decon5)
    up_conv4 = UpSampling2D(size=(2, 2))(decon5)
    up4 = concatenate([up_conv4, Cropping2D(cropping=(get_crop_shape(conv4, up_conv4)))(conv4)], axis=concat_axis)
    decon4 = Conv2D(f4, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(up4)
    decon4 = Conv2D(f4, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(decon4)
    up_conv3 = UpSampling2D(size=(2, 2))(decon4)
    up3 = concatenate([up_conv3, Cropping2D(cropping=(get_crop_shape(conv3, up_conv3)))(conv3)], axis=concat_axis)
    decon3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(up3)
    decon3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(decon3)
    up_conv2 = UpSampling2D(size=(2, 2))(decon3)
    up2 = concatenate([up_conv2, Cropping2D(cropping=(get_crop_shape(conv2, up_conv2)))(conv2)], axis=concat_axis)
    decon2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(up2)
    decon2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(decon2)
    up_conv1 = UpSampling2D(size=(2, 2))(decon2)
    up1 = concatenate([up_conv1, Cropping2D(cropping=(get_crop_shape(conv1, up_conv1)))(conv1)], axis=concat_axis)
    decon1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(up1)
    decon1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(decon1)

    ch, cw = get_crop_shape(img_input, decon1)
    decon1 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(decon1)
    # decon1 = BatchNormalization(mode=0, axis=concat_axis)(decon1)  # Batch normalization
    outputs = Conv2D(dim_out, (1, 1), activation=out_fun)(decon1)
    return Model(inputs=img_input, outputs=outputs), name

def unet_pool_up_7(img_rows, img_cols, dim_in, dim_out, act_fun='elu', out_fun='sigmoid', init='he_normal'):
    name="_unet_pool_up_7"
    f1, f2, f3, f4, f5, f6, f7 = 64, 96, 128, 192, 256, 384, 512
    # img_input = Input((None, None, dim_in))  # r,c,3
    img_input=Input((img_rows, img_cols, dim_in))  # r,c,3

    conv1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(img_input)
    conv1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool1)
    conv2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool2)
    conv3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(f4, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool3)
    conv4 = Conv2D(f4, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(f5, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool4)
    conv5 = Conv2D(f5, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    conv6 = Conv2D(f6, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool5)
    conv6 = Conv2D(f6, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)
    conv7 = Conv2D(f7, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(pool6)
    conv7 = Conv2D(f7, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(conv7)

    up_conv6 = UpSampling2D(size=(2, 2))(conv7)
    up6 = concatenate([up_conv6, Cropping2D(cropping=(get_crop_shape(conv6, up_conv6)))(conv6)], axis=concat_axis)
    decon6 = Conv2D(f6, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(up6)
    decon6 = Conv2D(f6, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(decon6)
    up_conv5 = UpSampling2D(size=(2, 2))(decon6)
    up5 = concatenate([up_conv5, Cropping2D(cropping=(get_crop_shape(conv5, up_conv5)))(conv5)], axis=concat_axis)
    decon5 = Conv2D(f5, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(up5)
    decon5 = Conv2D(f5, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(decon5)
    up_conv4 = UpSampling2D(size=(2, 2))(decon5)
    up4 = concatenate([up_conv4, Cropping2D(cropping=(get_crop_shape(conv4, up_conv4)))(conv4)], axis=concat_axis)
    decon4 = Conv2D(f4, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(up4)
    decon4 = Conv2D(f4, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(decon4)
    up_conv3 = UpSampling2D(size=(2, 2))(decon4)
    up3 = concatenate([up_conv3, Cropping2D(cropping=(get_crop_shape(conv3, up_conv3)))(conv3)], axis=concat_axis)
    decon3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(up3)
    decon3 = Conv2D(f3, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(decon3)
    up_conv2 = UpSampling2D(size=(2, 2))(decon3)
    up2 = concatenate([up_conv2, Cropping2D(cropping=(get_crop_shape(conv2, up_conv2)))(conv2)], axis=concat_axis)
    decon2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(up2)
    decon2 = Conv2D(f2, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(decon2)
    up_conv1 = UpSampling2D(size=(2, 2))(decon2)
    up1 = concatenate([up_conv1, Cropping2D(cropping=(get_crop_shape(conv1, up_conv1)))(conv1)], axis=concat_axis)
    decon1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(up1)
    decon1 = Conv2D(f1, (3, 3), activation=act_fun, padding='same', kernel_initializer=init)(decon1)

    ch, cw = get_crop_shape(img_input, decon1)
    decon1 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(decon1)
    # decon1 = BatchNormalization(mode=0, axis=concat_axis)(decon1)  # Batch normalization
    outputs = Conv2D(dim_out, (1, 1), activation=out_fun)(decon1)
    return Model(inputs=img_input, outputs=outputs), name
