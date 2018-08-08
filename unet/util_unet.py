import numpy as np
from keras import backend as K
from keras.backend.tensorflow_backend import _to_tensor
from keras.engine.saving import model_from_json

sum_axis=[0,-1,-2]
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

def build_compile(func, cfg, write=False):
    # 'relu6'  # min(max(features, 0), 6)
    # 'crelu'  # Concatenates ReLU (only positive part) with ReLU (only the negative part). Note that this non-linearity doubles the depth of the activations
    # 'elu'  # Exponential Linear Units exp(features)-1, if <0, features
    # 'selu'  # Scaled Exponential Linear Rectifier: scale * alpha * (exp(features) - 1) if < 0, scale * features otherwise.
    # 'softplus'  # log(exp(features)+1)
    # 'softsign' features / (abs(features) + 1)

    # 'mean_squared_error' 'mean_absolute_error'
    # 'binary_crossentropy'
    # 'sparse_categorical_crossentropy' 'categorical_crossentropy'

    # if dim_out>1:
    #     out_fun = 'softmax'
    #     loss_fun='categorical_crossentropy'
    # else:
    #     out_fun='sigmoid'
    #     loss_fun=[loss_bce_dice]
    #     # loss_fun=[loss_bce],  # 'binary_crossentropy' "bcedice"
    #     # loss_fun=[loss_jaccard],  # 'binary_crossentropy' "bcedice"
    #     # loss_fun=[loss_dice],  # 'binary_crossentropy' "bcedice"
    if cfg.act_fun is None:
        cfg.act_fun='relu'
    if cfg.out_fun is None:
        cfg.out_fun='sigmoid' if cfg.dep_out<=1 else 'softmax'
    if cfg.loss_fun is None:
        cfg.loss_fun='binary_crossentropy' if cfg.dep_out<=1 else 'categorical_crossentropy'
    model,name=func(cfg)
    model=compile_model(cfg, model)
    if write:
        save_model(model, name)
    return model, name

def compile_model(cfg, model):
    from keras.optimizers import Adam
    model.compile(optimizer=Adam(1e-3),
                  loss=cfg.loss_fun,
                  metrics=[jac, dice])
    model.summary()
    return model

def load_model(name):
    model_json = name + ".json"
    with open(model_json, 'r') as json_file:
        model = model_from_json(json_file.read())
    return model

def save_model(model, name):
    model_json = name + ".json"
    with open(model_json, "w") as json_file:
        json_file.write(model.to_json())

def load_compile(name, cfg):
    model = load_model(name)
    return compile_model(cfg, model)

