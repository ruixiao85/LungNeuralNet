import signal

import cv2
from keras import backend as K, metrics
from keras.engine import Layer, InputSpec
from keras.layers import Activation
from keras.losses import mse,mae
from keras.utils import get_custom_objects
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import central_crop
import numpy as np

from image_gen import ImageGenerator

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

def dice67(y_true, y_pred):
    return dice(central_crop(y_true,0.67), central_crop(y_pred,0.67))
def dice33(y_true, y_pred):
    return dice(central_crop(y_true,0.33), central_crop(y_pred,0.33))

def loss_bce(y_true, y_pred):  # bootstrapped binary cross entropy
    from keras.backend.tensorflow_backend import _to_tensor
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

def focal_loss(y_true, y_pred, gamma=0.5, alpha=.25): # (1-alpha)^gamma x CE.  alpha: balance (0.25) gamma: focus (2.0)
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def flatten_pixel(y_true,y_pred):
    return K.reshape(y_true,shape=[-1,3]), K.reshape(y_pred,shape=[-1,3])
def loss_psse(y_true,y_pred):
    y_true_r, y_pred_r=flatten_pixel(y_true,y_pred)
    return K.sum(K.square(y_pred_r-y_true_r), axis=-1)
def psse(y_true,y_pred):
    return -loss_psse(y_true,y_pred)
def loss_psae(y_true,y_pred):
    y_true_r, y_pred_r=flatten_pixel(y_true,y_pred)
    return K.sum(K.abs(y_pred_r - y_true_r), axis=-1)
def psae(y_true,y_pred):
    return -loss_psae(y_true,y_pred)

def swish(x):
    return x * K.sigmoid(x)

def custom_function_keras():
    get_custom_objects().update({'swish': Activation(swish,name='swish')})


def custom_function_dict():
    return {
        'swish':swish,
        'jac_d':jac_d,
        'jac':jac,
        'dice_d':dice_d,
        'dice':dice,
        'dice33':dice33,
        'dice67':dice67,
        'acc':acc,
        'acc33':acc33,
        'acc67':acc67,
        'loss_bce_dice':loss_bce_dice,
        'loss_bce':loss_bce,
        'loss_dice':loss_dice,
        'ImageGenerator':ImageGenerator,
    }

def top5acc(y_true, y_pred, k=5):  # top_N_categorical_accuracy
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)
def sparacc(y_true, y_pred):  # sparse_categorical_accuracy
    return K.cast(K.equal(K.max(y_true, axis=-1), K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())
def acc(y_true, y_pred):  # default 'acc'
    return K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)),
                  K.floatx())
def acc67(y_true, y_pred):
    return acc(central_crop(y_true,0.67), central_crop(y_pred,0.67))
def acc33(y_true, y_pred):
    return acc(central_crop(y_true,0.33), central_crop(y_pred,0.33))
