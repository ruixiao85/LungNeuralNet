import os
import numpy as np
import pandas as pd
from PIL import ImageDraw, Image, ImageFont
from skimage.io import imsave, imread

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import argparse


def standardize(preimg):
    mean, std = np.mean(preimg), np.std(preimg)
    print("  mean %.2f std %.2f" % (mean, std))
    preimg -= mean
    preimg /= (2.5 * std)
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

    r, c = 1040, 1392
    ri = int((r - rows) / 2)
    ci = int((c - cols) / 2)
    for i, image_name in enumerate(images):
        _img[i] = preprocess_color(imread(os.path.join(wd, dir_in, image_name))[ri:ri+rows,ci:ci+cols] / 255., True)
        if dir_out != '':
            _tgt[i] = preprocess_channel(imread(os.path.join(wd, dir_out, image_name))[ri:ri+rows,ci:ci+cols] / 255., tgt_ch)
        if int(10. * (i + 1) / total) > int(10. * i / total):
            print('Loading %d / %d images [%.0f%%]' % (i + 1, total, 10 * int(10. * (i + 1) / total)))
    return _img, _tgt

def get_data_pair_slice(sub_dir, dir_in, dir_out, rows, cols, tgt_ch):
    def slice_2x2(my_matrix):
        upper_half = np.hsplit(np.vsplit(my_matrix, 2)[0], 2)
        lower_half = np.hsplit(np.vsplit(my_matrix, 2)[1], 2)
        return upper_half[0], upper_half[1], lower_half[0], lower_half[1]

    wd = os.path.join(os.getcwd(), sub_dir)
    images = get_recursive_rel_path(os.path.join(wd, dir_in))
    total = len(images)
    print("Found [%d] file from subfolders [/%s] of [%s]" % (total, dir_in, wd))
    multi = 4
    rows = int(rows / 2)
    cols = int(cols / 2)

    if dir_out != '':
        tgts = get_recursive_rel_path(os.path.join(wd, dir_out))
        print("Found [%d] file from subfolders [/%s] of [%s]" % (len(tgts), dir_out, wd))

        images = list(set(images).intersection(tgts))  # image-target pairs only
        total = len(images)  # update # of files
        print("%d image-mask pairs accepted" % total)
        _tgt = np.ndarray((total * multi, rows, cols, 1), dtype=np.float32)
    else:
        _tgt = images
    _img = np.ndarray((total * multi, rows, cols, 3), dtype=np.float32)

    r, c = 1040,1392
    ri = int((r - rows) / 2)
    ci = int((c - cols) / 2)
    for i, image_name in enumerate(images):
        _img[4 * i], _img[4 * i + 1], _img[4 * i + 2], _img[4 * i + 3] = slice_2x2(
            preprocess_color(imread(os.path.join(wd, dir_in, image_name))[ri:ri+rows,ci:ci+cols] / 255., True))
        if dir_out != '':
            _tgt[4 * i], _tgt[4 * i + 1], _tgt[4 * i + 2], _tgt[4 * i + 3] = slice_2x2(
                preprocess_channel(imread(os.path.join(wd, dir_out, image_name))[ri:ri+rows,ci:ci+cols] / 255., tgt_ch))
        if int(10. * (i + 1) / total) > int(10. * i / total):
            print('Loading %d / %d images [%.0f%%]' % (i + 1, total, 10 * int(10. * (i + 1) / total)))
    return _img, _tgt


def train( _target, num_epoch=12, cont_train=True):
    weight_file = _target + ".h5"

    if cont_train and os.path.exists(weight_file):
        model.load_weights(weight_file)

    print('Creating model and checkpoint...')
    model_checkpoint = ModelCheckpoint(weight_file, monitor='val_loss', save_best_only=True)  # monitor='loss'
    # tb_callback = TensorBoard(log_dir="tensorboard", histogram_freq=0, # batch_size=config.BATCH_SIZE,
    #                           write_graph=True, write_grads=False, write_images=True,
    #                           embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')

    print('Fitting model...')
    history_callback=model.fit(img, msk, batch_size=1, epochs=num_epoch, shuffle=True, validation_split=0.3,
              callbacks=[model_checkpoint, early_stop])  # ,tb_callback
    loss_history = history_callback.history["loss"]
    np.savetxt("loss_history"+_target+".txt", np.array(loss_history), delimiter=",")



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
    targets = args.output.split(',')

    # model, nn = get_unet_vgg(args.height, args.width, 3, 1)
    # model=get_unet(args.height, args.width, 3, 1)
    # model = get_unet(int(args.height/2), int(args.width/2), 3, 1)
    # model = get_unet_vgg(args.height, args.width, 3, 1)
    # model_json = "unet5.json"
    # with open(model_json, "w") as json_file:
    #     json_file.write(model.to_json())
    # with open(model_json, 'r') as json_file:
    #     model = model_from_json(json_file.read())
    from model.unet_upsample import get_unet4_up, get_unet5_up, get_unet6_up, get_unet7_up
    from model.unet_transpose import  get_unet5_trans,get_unet6_trans, get_unet7_trans
    from model.unet import get_unet_compiled
    models=[
        # get_unet4_up,
        get_unet5_up,
        get_unet6_up,
        get_unet5_trans,
        get_unet7_trans,
    ]
    mode = args.mode[0].lower()
    if mode != 'p':
        for mod in models:
            model,nn=get_unet_compiled(mod, args.height, args.width, 3, 1)
            for target in targets:
                img, msk = get_data_pair(args.train_dir, args.input, target, args.height, args.width, 2)  # Blue
                # img, msk = get_data_pair_slice(args.train_dir, args.input, target, args.height, args.width, 2)  # Blue
                train(target+nn, 20, True)

    if mode != 't':
        tst, name = get_data_pair(args.pred_dir, args.input, '', args.height, args.width, 1)
        for mod in models:
            model, nn = get_unet_up_compiled(mod, args.height, args.width, 3, 1)
            res = np.zeros((len(name), len(targets)), np.float32)
            for x, target in enumerate(targets):
                predict(name, target+nn, res[:, x], 2)
            res_df = pd.DataFrame(res, name, targets)
            # res_df.to_csv("result"+target+nn+".csv")
            res_df.to_excel("result"+target+nn+".xlsx")
