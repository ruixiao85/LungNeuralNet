import os
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import ImageDraw, Image, ImageFont
from skimage.io import imsave, imread

from keras.callbacks import ModelCheckpoint, EarlyStopping
import argparse


def get_recursive_rel_path(fp, ext='*.jpg'):
    from glob import glob
    images = [path for fn in os.walk(fp) for path in glob(os.path.join(fn[0], ext))]
    for i in range(len(images)):
        images[i] = os.path.relpath(images[i], fp)
    return images


def get_data_pair(sub_dir, dir_in, dir_out, cfg, aug):
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
        _tgt = np.ndarray((total, rows, cols, 3), dtype=np.uint8)
    else:
        _tgt = images
    _img = np.ndarray((total, rows, cols, 3), dtype=np.uint8)

    from process_image import fit, preprocess_train, preprocess_predict
    for i, image_name in enumerate(images):
        _img[i] = fit(imread(os.path.join(wd, dir_in, image_name)),rows,cols)
        if dir_out != '':
            _tgt[i] = fit(imread(os.path.join(wd, dir_out, image_name)),rows,cols)
        if int(10. * (i + 1) / total) > int(10. * i / total):
            print('Loading %d / %d images [%.0f%%]' % (i + 1, total, 10 * int(10. * (i + 1) / total)))
    if dir_out != '':
        return preprocess_train(_img, _tgt, _aug=aug, _out=cfg.dep_out)
    else:
        return preprocess_predict(_img, _tgt)

def train(_target, num_epoch=12, cont_train=True):
    weight_file = _target + ".h5"

    if cont_train and os.path.exists(weight_file):
        print("Continue from previous weights")
        model.load_weights(weight_file)

    print('Creating model and checkpoint...')
    model_checkpoint = ModelCheckpoint(weight_file, monitor='val_loss', save_best_only=True)  # monitor='loss'
    # tb_callback = TensorBoard(log_dir="tensorboard", histogram_freq=0, # batch_size=config.BATCH_SIZE,
    #                           write_graph=True, write_grads=False, write_images=True,
    #                           embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

    print('Fitting model...')
    history_callback=model.fit(img, msk, batch_size=1, epochs=num_epoch, shuffle=True, validation_split=0.3,
              callbacks=[model_checkpoint, early_stop])  # ,tb_callback
    with open(_target+".csv", "a") as log:
        log.write("\n"+datetime.now().strftime("%Y-%m-%d %H:%M")+" train history:\n")
    pd.DataFrame(history_callback.history).to_csv(_target+".csv", mode="a")


def predict(_name, _target, vec, cfg):
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
        vec[i] = (np.sum(image[:, :, 0])).astype(np.float16)
        print("%s pixel sum: %.1f" % (_name[i], vec[i]))
        # image = (image[:, :, 0] * 255.).astype(np.uint8)  # pure BW
        # image = ((0.6 * image[:, :, 0] + 0.4 * (tst[i, :, :, 1] + 0.99)) * 127.).astype(np.uint8)  # gray mixed
        mix = tst[i].copy()
        ch=cfg.overlay_channel
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
    from model.unet_pool_up import unet_pool_up_5
    from model.unet_pool_trans import unet_pool_trans_5
    from model.unet_conv_trans import unet_conv_trans_5
    from model.unet_pool_up_31 import unet_pool_up_5_dure
    models=[
        unet_pool_up_5,
        # unet_pool_up_7,
        unet_pool_trans_5,
        # unet_pool_trans_7,
        unet_conv_trans_5,
        # unet_conv_trans_7,
        unet_pool_up_5_dure,
        # unet_pool_up_7_dure,
    ]
    from model_config import config
    mode = args.mode[0].lower()
    for cfg in [config(3,1,'elu','sigmoid'), config(3,2,'elu','softmax')]:
        if mode != 'p':
            for mod in models:
                    model,nn=compile_unet(mod, rows, cols, cfg)
                    print("Using "+nn)
                    for target in targets:
                        n_repeats=5
                        for r in range(n_repeats):  # repeat imgaug and train
                            print("Repeating training %d/%d for %s" % (r + 1, n_repeats, target))
                            img, msk = get_data_pair(args.train_dir, args.input, target, cfg, aug=True)
                            train(target+nn, 18, True)

        if mode != 't':
            tst, name = get_data_pair(args.pred_dir, args.input, '', cfg, aug=False)
            for mod in models:
                model, nn = compile_unet(mod, rows, cols, cfg)
                res = np.zeros((len(name), len(targets)), np.float16)
                for x, target in enumerate(targets):
                    predict(name, target+nn, res[:, x], cfg)
                res_df = pd.DataFrame(res, name, targets)
                # res_df.to_csv("result_"+args.pred_dir+"_"+nn+".xlsx")
                res_df.to_excel("result_"+args.pred_dir+"_"+nn+".xlsx")
