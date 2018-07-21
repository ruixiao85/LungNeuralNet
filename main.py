import os
import numpy as np
import pandas as pd
from PIL import ImageDraw, Image, ImageFont
from keras.engine.saving import model_from_json
from skimage.io import imsave, imread
from model import dice_coef, dice_coef_loss, get_unet3, get_unet4, get_unet5, get_unet6
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop
import argparse


def standardize(preimg):
    mean, std = np.mean(preimg), np.std(preimg)
    print("  mean %.2f std %.2f" % (mean, std))
    preimg -= mean
    preimg /= (1.5 * std)
    preimg = np.tanh(preimg)
    return preimg


def preprocess_color(preimg, stand):
    if stand:
        preimg = standardize(preimg)
    return preimg


def preprocess_channel(preimg, ch):
    # tmp = preimg[..., ch].copy()
    # return tmp[..., np.newaxis]
    return preimg[..., ch][..., np.newaxis]


def get_recursive_rel_path(fp, ext='*.jpg'):
    from glob import glob
    images = [path for fn in os.walk(fp) for path in glob(os.path.join(fn[0], ext))]
    for i in range(len(images)):
        images[i] = os.path.relpath(images[i], fp)
    return images


def get_data_pair(sub_dir, dir_in, dir_out, rows, cols, tgt_ch):
    wd = os.path.join(os.getcwd(), sub_dir)
    images = get_recursive_rel_path(os.path.join(wd, dir_in))
    total = len(images)
    print("Found [%d] file from subfolders [/%s] of [%s]" % (total, dir_in, wd))

    if dir_out != '':
        tgts = get_recursive_rel_path(os.path.join(wd, dir_out))
        print("Found [%d] file from subfolders [/%s] of [%s]" % (len(tgts), dir_out, wd))

        images = list(set(images).intersection(tgts))  # image-target pairs only
        total = len(images)  # update # of files
        print("%d image-mask pairs accepted" % total)
        _tgt = np.ndarray((total, rows, cols, 1), dtype=np.float32)
    else:
        _tgt = images
    _img = np.ndarray((total, rows, cols, 3), dtype=np.float32)

    for i, image_name in enumerate(images):
        _img[i] = preprocess_color(imread(os.path.join(wd, dir_in, image_name)) / 255., True)
        if dir_out != '':
            _tgt[i] = preprocess_channel(imread(os.path.join(wd, dir_out, image_name)) / 255., tgt_ch)
        if int(10. * (i + 1) / total) > int(10. * i / total):
            print('Leading %d / %d images [%.0f%%]' % (i + 1, total, 10 * int(10. * (i + 1) / total)))
    return _img, _tgt


def train(_target, num_epoch=12, cont_train=True):
    weight_file = _target + ".h5"

    if cont_train and os.path.exists(weight_file):
        model.load_weights(weight_file)

    print('Creating model and checkpoint...')
    model_checkpoint = ModelCheckpoint(weight_file, monitor='val_loss', save_best_only=True)

    print('Fitting model...')
    model.fit(img, msk, batch_size=1, epochs=num_epoch, shuffle=True, validation_split=0.3,
              callbacks=[model_checkpoint])


def predict(_name, _target, vec, ch):
    print('Load weights and predicting ...')
    model.load_weights(_target + ".h5")
    imgs_mask_test = model.predict(tst, verbose=1, batch_size=1)

    target_dir = os.path.join(args.pred_dir, _target)
    print('Saving predicted results [%s] to files...' % _target)
    image_sum = args.width * args.height
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    for i, image in enumerate(imgs_mask_test):
        target_file = os.path.join(target_dir, _name[i].replace(".jpg", ".png"))
        target_new_dir = os.path.dirname(target_file)
        if not os.path.exists(target_new_dir):
            os.mkdir(target_new_dir)
        vec[i] = float(np.sum(image[:, :, 0]))
        print("%s pixel sum: %.1f" % (_name[i], vec[i]))
        # image = (image[:, :, 0] * 255.).astype(np.uint8)  # pure BW
        # image = ((0.6 * image[:, :, 0] + 0.4 * (tst[i, :, :, 1] + 0.99)) * 127.).astype(np.uint8)  # gray mixed
        mix = tst[i].copy()
        for c in range(3):
            if c != ch:
                mix[:, :, c] = np.tanh(mix[:, :, c] - 0.4 * image[:, :, 0])
        mix = Image.fromarray(((mix + 1.) * 127.).astype(np.uint8), 'RGB')
        draw = ImageDraw.Draw(mix)
        draw.text((0, 0), " Pixels: %.0f / %.0f \n Percentage: %.0f%%" % (vec[i], image_sum, 100. * vec[i] / image_sum),
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
                        default='1392', help='width/columns')
    parser.add_argument('-r', '--height', dest='height', type=int,
                        default='1040', help='height/rows')
    parser.add_argument('-e', '--ext', dest='ext', action='store',
                        default='jpg', help='extension')
    parser.add_argument('-i', '--input', dest='input', type=str,
                        default='Original', help='input: Original')
    parser.add_argument('-o', '--output', dest='output', type=str,
                        default='Paren,InflamMild,InflamSevere', help='output: targets separated by comma')
    args = parser.parse_args()

    os.chdir(os.getcwd() if (args.dir == '') else args.dir)
    targets = args.output.split(',')
    model = get_unet5(args.height, args.width, 3, 1, 'sigmoid')
    # model_json = "unet5.json"
    # with open(model_json, "w") as json_file:
    #     json_file.write(model.to_json())
    # with open(model_json, 'r') as json_file:
    #     model = model_from_json(json_file.read())
    model.compile(optimizer=Adam(lr=1e-6), loss=dice_coef_loss, metrics=[dice_coef])
    # optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),#

    mode = args.mode[0].lower()
    if mode != 'p':
        for target in targets:
            img, msk = get_data_pair(args.train_dir, args.input, target, args.height, args.width, 2)  # Blue
            train(target, 20, True)

    if mode != 't':
        tst, name = get_data_pair(args.pred_dir, args.input, '', args.height, args.width, 1)
        res = np.zeros((len(name), len(targets)), np.float32)
        for x, target in enumerate(targets):
            predict(name, target, res[:, x], 2)
        res_df = pd.DataFrame(res, name, targets)
        # res_df.to_csv("result.csv")
        res_df.to_excel("result.xlsx")
