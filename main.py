import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import ImageDraw, Image, ImageFont
from skimage.io import imsave, imread
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from process_image import fit, preprocess_train, preprocess_predict
from tensorboard_train_val import TensorBoardTrainVal


def get_recursive_rel_path(_wd, _sf, ext='*.jpg'):
    _path = os.path.join(_wd, _sf)
    from glob import glob
    images = [path for fn in os.walk(_path) for path in glob(os.path.join(fn[0], ext))]
    total = len(images)
    print("Found [%d] file from subfolders [/%s] of [%s]" % (total, _sf, _wd))
    for i in range(total):
        images[i] = os.path.relpath(images[i], _path)
    return images, total


def get_train_data(sub_dir, dir_in, dir_out, _cfg):
    wd = os.path.join(os.getcwd(), sub_dir)
    images, total = get_recursive_rel_path(wd, dir_in)
    tgts, _ = get_recursive_rel_path(wd, dir_out)

    images = list(set(images).intersection(tgts))  # image-target pairs only
    total = len(images)  # update # of files
    print("%d image-mask pairs accepted" % total)
    _img = np.ndarray((total, rows, cols, 3), dtype=np.uint8)
    _tgt = np.ndarray((total, rows, cols, 3), dtype=np.uint8)

    for i, image_name in enumerate(images):
        _img[i] = fit(imread(os.path.join(wd, dir_in, image_name)),rows,cols)
        _tgt[i] = fit(imread(os.path.join(wd, dir_out, image_name)),rows,cols)
        if int(10. * (i + 1) / total) > int(10. * i / total):
            print('Loading %d / %d images [%.0f%%]' % (i + 1, total, 10 * int(10. * (i + 1) / total)))
    return preprocess_train(_img, _tgt, _aug=_cfg.aug, _out=_cfg.dep_out)

def get_predict_data(sub_dir, dir_in):
    wd = os.path.join(os.getcwd(), sub_dir)
    images, total = get_recursive_rel_path(wd, dir_in)
    _img = np.ndarray((total, rows, cols, 3), dtype=np.uint8)

    for i, image_name in enumerate(images):
        _img[i] = fit(imread(os.path.join(wd, dir_in, image_name)),rows,cols)
        if int(10. * (i + 1) / total) > int(10. * i / total):
            print('Loading %d / %d images [%.0f%%]' % (i + 1, total, 10 * int(10. * (i + 1) / total)))
    return preprocess_predict(_img, images)

def train(_target, num_epoch=12, cont_train=True):
    weight_file = _target + ".h5"

    if cont_train and os.path.exists(weight_file):
        print("Continue from previous weights")
        model.load_weights(weight_file)

    print('Creating model and checkpoint...')
    # key_metrics = 'val_loss'
    key_metrics = 'dice_coef_flat_int'
    # model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[TrainValTensorBoard(write_graph=False)])
    print('Fitting model...')
    history=model.fit(img, msk, batch_size=1, epochs=num_epoch, shuffle=True, validation_split=0.25,
          callbacks=[
              ModelCheckpoint(weight_file, monitor=key_metrics, save_best_only=True),
              EarlyStopping(monitor=key_metrics, patience=0, verbose=1, mode='auto'),
              TensorBoardTrainVal(log_dir=os.path.join("log", _target), write_graph=True, write_grads=False, write_images=True),
          ]).history
    with open(_target+".csv", "a") as log:
        log.write("\n"+datetime.now().strftime("%Y-%m-%d %H:%M")+" train history:\n")
    pd.DataFrame(history).to_csv(_target+".csv", mode="a")


def predict(_name, _target, _vec, _cfg):
    print('Load weights and predicting ...')
    model.load_weights(_target + ".h5")
    imgs_mask_test = model.predict(tst, verbose=1, batch_size=1)

    target_dir = os.path.join(args.pred_dir, _target)
    print('Saving predicted results [%s] to files...' % _target)
    image_sum = rows * cols
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    for i, image in enumerate(imgs_mask_test):
        target_file = os.path.join(target_dir, _name[i].replace(".jpg", ".png"))
        target_new_dir = os.path.dirname(target_file)
        if not os.path.exists(target_new_dir):
            os.mkdir(target_new_dir)
        if _cfg.call_hardness==1:  # hard sign
            image = np.rint(image)
        elif 0<_cfg.call_hardness<1:
            image=(image+np.rint(image)*_cfg.call_hardness)/(1.0+_cfg.call_hardness)  # mixed
        _vec[i] = int(np.sum(image[..., 0]))
        print("%s pixel sum: %.1f" % (_name[i], _vec[i]))
        # image = (image[:, :, 0] * 255.).astype(np.uint8)  # pure BW
        # image = ((0.6 * image[:, :, 0] + 0.4 * (tst[i, :, :, 1] + 0.99)) * 127.).astype(np.uint8)  # gray mixed
        mix = tst[i].copy()
        ch=_cfg.overlay_channel
        op=_cfg.overlay_opacity
        for c in range(3):
            if c == ch:
                mix[:, :, c] = np.tanh(mix[:, :, c] + op * image[:, :, 0])
            else:
                mix[:, :, c] = np.tanh(mix[:, :, c] - op * image[:, :, 0])
        mix = Image.fromarray(((mix + 1.) * 127.).astype(np.uint8), 'RGB')
        draw = ImageDraw.Draw(mix)
        draw.text((0, 0), " Pixels: %.0f / %.0f \n Percentage: %.0f%%" % (_vec[i], image_sum, 100. * _vec[i] / image_sum),
                  (255 if ch == 0 else 10, 255 if ch == 1 else 10, 255 if ch == 2 else 10),
                  ImageFont.truetype("arial.ttf", 36))  # font type size)
        imsave(target_file, mix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', '--dir', dest='dir', action='store',
                        default='', help='work directory, empty->current dir')
    parser.add_argument('-t', '--train', dest='train_dir', action='store',
                        default='train', help='train sub-directory')
    parser.add_argument('-p', '--pred', dest='pred_dir', action='store',
                        default='pred', help='predict sub-directory')
    parser.add_argument('-m', '--mode', dest='mode', action='store',
                        default='both', help='mode: train pred both')
    parser.add_argument('-c', '--width', dest='width', type=int,
                        default='1280', help='width/columns')
    parser.add_argument('-r', '--height', dest='height', type=int,
                        default='1024', help='height/rows')
    parser.add_argument('-e', '--ext', dest='ext', action='store',
                        default='jpg', help='extension')
    parser.add_argument('-i', '--input', dest='input', type=str,
                        default='Original', help='input: Original')
    parser.add_argument('-o', '--output', dest='output', type=str,
                        default='Paren,InflamMild,InflamSevere', help='output: targets separated by comma')
    args = parser.parse_args()

    os.chdir(os.getcwd() if (args.dir == '') else args.dir)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '-1'  # force cpu
    rows, cols = args.height, args.width
    targets = args.output.split(',')

    # model_json = "unet5.json"
    # with open(model_json, "w") as json_file:
    #     json_file.write(model.to_json())
    # with open(model_json, 'r') as json_file:
    #     model = model_from_json(json_file.read())
    from unet import compile_unet
    from model.unet_pool_trans import unet_pool_trans_5,unet_pool_trans_6,unet_pool_trans_7
    from model.unet_conv_trans import unet_conv_trans_5, unet_conv_trans_6,unet_conv_trans_7
    from model.unet_pool_up import unet_pool_up_5, unet_pool_up_6, unet_pool_up_7
    from model.unet_pool_up_31 import unet_pool_up_5_dure, unet_pool_up_6_dure, unet_pool_up_7_dure
    from model.unet_vgg import unet_vgg_7conv
    models = [
        # unet_pool_trans_5,
        # unet_pool_trans_7,
        # unet_conv_trans_5,
        # unet_conv_trans_7,
        # unet_pool_up_5,
        # unet_pool_up_7,
        unet_pool_up_5_dure,
        # unet_pool_up_7_dure,
        # unet_vgg_7conv,
    ]
    from model_config import config
    configs = [
        # config(3, 1, 'elu', 'sigmoid'),
        # config(3, 2, 'relu', 'softmax', 'categorical_crossentropy'),
        config(3, 2, 'elu', 'softmax'),
        # config(3, 2, 'tanh', 'softmax', 'categorical_crossentropy'),
        # config(3, 2, 'softsign', 'softmax', 'categorical_crossentropy'),  # trouble
        # config(3, 2, 'selu', 'softmax', 'categorical_crossentropy'),  # trouble
        # config(3, 2, 'softplus', 'softmax', 'categorical_crossentropy'), # trouble
    ]
    mode = args.mode[0].lower()
    if mode != 'p':
        for cfg in configs:
            for mod in models:
                model, nn = compile_unet(mod, rows, cols, cfg)
                print("Network specifications: " + nn.replace("_", " "))
                for target in targets:
                    n_repeats = 3
                    for r in range(n_repeats):  # repeat imgaug and train
                        print("Repeating training %d/%d for %s" % (r + 1, n_repeats, target))
                        img, msk = get_train_data(args.train_dir, args.input, target, cfg)
                        train(target + nn, 20, True)

    if mode != 't':
        tst, name = get_predict_data(args.pred_dir, args.input)
        for cfg in configs:
            for mod in models:
                model, nn = compile_unet(mod, rows, cols, cfg)
                res = np.zeros((len(name), len(targets)), np.float32)
                for x, target in enumerate(targets):
                    predict(name, target + nn, res[:, x], cfg)
                res_df = pd.DataFrame(res, name, targets)
                # res_df.to_csv("result_" + args.pred_dir + "_" + nn + ".csv")
                res_df.to_excel("result_" + args.pred_dir + "_" + nn + ".xlsx")
