import numpy as np
from keras.applications.vgg16 import VGG16
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, merge
from keras import backend as K
from keras.backend.tensorflow_backend import _to_tensor
K.set_image_data_format("channels_last")
#K.set_image_dim_ordering("th")
CACHED_UNET = None
SMOOTH_LOSS = 1e-12

def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + SMOOTH_LOSS) / (sum_ - intersection + SMOOTH_LOSS)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + SMOOTH_LOSS) / (sum_ - intersection + SMOOTH_LOSS)
    return K.mean(jac)

def jacard_coef_flat(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + SMOOTH_LOSS) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + SMOOTH_LOSS)


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bootstrapped_crossentropy(y_true, y_pred, bootstrap_type='hard', alpha=0.95):
    target_tensor = y_true
    prediction_tensor = y_pred
    _epsilon = _to_tensor(K.epsilon(), prediction_tensor.dtype.base_dtype)
    prediction_tensor = K.tf.clip_by_value(prediction_tensor, _epsilon, 1 - _epsilon)
    prediction_tensor = K.tf.log(prediction_tensor / (1 - prediction_tensor))

    if bootstrap_type == 'soft':
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * K.tf.sigmoid(prediction_tensor)
    else:
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * K.tf.cast(
            K.tf.sigmoid(prediction_tensor) > 0.5, K.tf.float32)
    return K.mean(K.tf.nn.sigmoid_cross_entropy_with_logits(
        labels=bootstrap_target_tensor, logits=prediction_tensor))

def dice_coef_loss_bce(y_true, y_pred):
    dice = 0.5
    bce = 0.5
    bootstrapping = 'hard'
    alpha = 1.
    return bootstrapped_crossentropy(y_true, y_pred, bootstrapping, alpha) * bce + dice_coef_loss(y_true, y_pred) * dice



def get_vgg_7conv(ims, nchannels, n_cls):
    input_shape_base = (None, None, 4)
    img_input = Input(input_shape_base)
    vgg16_base = VGG16(input_tensor=img_input, include_top=False, weights=None)
    #for l in vgg16_base.layers:
    #    l.trainable = True

    conv1 = vgg16_base.get_layer("block1_conv2").output
    conv2 = vgg16_base.get_layer("block2_conv2").output
    conv3 = vgg16_base.get_layer("block3_conv3").output
    pool3 = vgg16_base.get_layer("block3_pool").output

    conv4 = Conv2D(384, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block4_conv1")(pool3)
    conv4 = Conv2D(384, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block4_conv2")(conv4)
   # pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4)
    pool4 = MaxPooling2D((2, 2), strides=None, name='block4_pool')(conv4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block5_conv1")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block5_conv2")(conv5)
   # pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5)

    conv6 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block6_conv1")(pool5)
    conv6 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block6_conv2")(conv6)
    #pool6 = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(conv6)
    pool6 = MaxPooling2D((2, 2), strides=(2,2), name='block6_pool')(conv6)

    conv7 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block7_conv1")(pool6)
    conv7 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block7_conv2")(conv7)

    up8 = concatenate([Conv2DTranspose(384, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='valid')(conv7), conv6], axis=3)
    #up8 = merge([Conv2DTranspose(384, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv7), conv6], mode='concat', concat_axis=3)
    conv8 = Conv2D(384, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up8)

    up9 = concatenate([Conv2DTranspose(256, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv8), conv5], axis=3)
    conv9 = Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up9)

    up10 = concatenate([Conv2DTranspose(192, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv9), conv4], axis=3)
    conv10 = Conv2D(192, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up10)

    up11 = concatenate([Conv2DTranspose(128, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv10), conv3], axis=3)
    conv11 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up11)

    up12 = concatenate([Conv2DTranspose(64, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv11), conv2], axis=3)
    conv12 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up12)

    up13 = concatenate([Conv2DTranspose(32, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv12), conv1], axis=3)
    conv13 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up13)

    # #Batch normalization
    #conv13 = BatchNormalization(mode=0, axis=1)(conv13)

    conv13 = Conv2D(n_cls, (1, 1), activation='sigmoid')(conv13)
    #conv13 = Conv2D(1, (1, 1))(conv13)
    #conv13 = Activation("sigmoid")(conv13)
    model = Model(img_input, conv13)

    # Recalculate weights on first layer
    conv1_weights = np.zeros((3, 3, nchannels, 64), dtype="float32")
    vgg = VGG16(include_top=False, input_shape=(ims, ims, 3))
    conv1_weights[:, :, :3, :] = vgg.get_layer("block1_conv1").get_weights()[0][:, :, :, :]
    bias = vgg.get_layer("block1_conv1").get_weights()[1]
    model.get_layer('block1_conv1').set_weights((conv1_weights, bias))
    return model

def get_loss_func(loss_mode):
    loss_func = 'binary_crossentropy'
    if loss_mode == "bcedice":
       loss_func = dice_coef_loss_bce

    print("loss:", loss_func)
    return  loss_func

def get_unet_pretrained(config, loss_mode):
    model = get_vgg_7conv(config.ISZ, config.N_CHANNELS, config.NUM_CLASSES)
    model.compile(optimizer=Adam(), loss=get_loss_func(loss_mode),
                  metrics=[jaccard_coef,  jacard_coef_flat, jaccard_coef_int,  dice_coef, 'accuracy'])
    return model

def get_unet(config, loss_mode):
    global CACHED_UNET
    if CACHED_UNET == None:
        CACHED_UNET = get_unet_pretrained(config, loss_mode)
    return  CACHED_UNET
