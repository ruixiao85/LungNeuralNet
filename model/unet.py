from keras import backend as K
from keras.backend.tensorflow_backend import _to_tensor
from model_config import config

sum_axis=[0,-1,-2]
SMOOTH_LOSS = 1e-5

def jaccard_coef_flat(y_true, y_pred):
    y_true_f, y_pred_f = K.flatten(y_true), K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + SMOOTH_LOSS) / (K.sum(y_true_f + y_pred_f) - intersection + SMOOTH_LOSS)

def jaccard_coef_flat_int(y_true, y_pred):
    return jaccard_coef_flat(y_true, K.round(K.clip(y_pred, 0, 1)))

def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=sum_axis)
    sum_ = K.sum(y_true + y_pred, axis=sum_axis)
    jac = (intersection + SMOOTH_LOSS) / (sum_ - intersection + SMOOTH_LOSS)
    return K.mean(jac)

def jaccard_coef_int(y_true, y_pred):
    return jaccard_coef(y_true, K.round(K.clip(y_pred, 0, 1)))

def dice_coef_flat(y_true, y_pred):
    y_true_f, y_pred_f = K.flatten(y_true), K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH_LOSS) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH_LOSS)

def dice_coef_flat_int(y_true, y_pred):
    return dice_coef_flat(y_true, K.round(K.clip(y_pred, 0, 1)))

def loss_bce(y_true, y_pred, bootstrap_type='hard', alpha=0.95):  # bootstrapped binary cross entropy
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
    return K.mean(K.tf.nn.sigmoid_cross_entropy_with_logits(labels=bootstrap_target_tensor, logits=prediction_tensor))

def loss_jaccard(y_true, y_pred):
    return 1. - jaccard_coef(y_true, y_pred)

def loss_dice(y_true, y_pred):
    return 1. - dice_coef_flat(y_true, y_pred)

def loss_bce_dice(y_true, y_pred):
    return 0.5 * (loss_bce(y_true, y_pred) + loss_dice(y_true, y_pred))

def loss_jaccard_dice(y_true, y_pred):
    return loss_jaccard(y_true, y_pred) + loss_dice(y_true, y_pred)

def compile_unet(func, img_rows, img_cols, cfg):
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
    if cfg.loss_fun is None:
        cfg.loss_fun=[loss_bce_dice] if cfg.dep_out<=1 else 'categorical_crossentropy'
    model,name=func(img_rows, img_cols, cfg)
    from keras.optimizers import Adam
    model.compile(optimizer=Adam(1e-5),
                  loss=cfg.loss_fun,
                  metrics=[jaccard_coef_int, dice_coef_flat_int,'accuracy'])
    model.summary()
    return model, name
