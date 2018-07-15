import os
import numpy as np
import pandas as pd
from PIL import ImageDraw, Image, ImageFont
from keras.engine.saving import model_from_json
from skimage.io import imsave, imread
from model import dice_coef, dice_coef_loss, get_unet3, get_unet4, get_unet5, get_unet6
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop

ImageOrig, ImagePred = "train", "pred"
image_cols, image_rows, image_sum = 1392, 1040, 1392 * 1040


def standardize(preimg):
    mean, std = np.mean(preimg), np.std(preimg)
    print("  mean %.2f std %.2f" % (mean, std))
    preimg -= mean
    preimg /= (1.5 * std)
    preimg = np.tanh(preimg)
    return preimg


def preprocessColor(preimg, stand):
    if stand:
        preimg = standardize(preimg)
    return preimg


def preprocessChannel(preimg, ch):
    # tmp = preimg[..., ch].copy()
    # return tmp[..., np.newaxis]
    return preimg[..., ch][..., np.newaxis]

def get_recursive_rel_path(fp, ext='*.jpg'):
    from glob import glob
    images = [path for fn in os.walk(fp) for path in glob(os.path.join(fn[0], ext))]
    for i in range(len(images)):
        images[i] = os.path.relpath(images[i], fp)
    return images


def get_data_pair(ori, dir_in, dir_out, rows, cols, tgt_ch):
    wd = os.path.join(os.getcwd(), ori)
    images = get_recursive_rel_path(os.path.join(wd, dir_in))
    total = len(images)
    print("Found [%d] file from subfolders [/%s] of [%s]" % (total, dir_in, wd))

    if dir_out != '':
        targets = get_recursive_rel_path(os.path.join(wd, dir_out))
        print("Found [%d] file from subfolders [/%s] of [%s]" % (len(targets), dir_out, wd))

        images = list(set(images).intersection(targets))  # image-target pairs only
        total = len(images)  # update # of files
        print("%d image-mask pairs accepted" % total)
        tgt = np.ndarray((total, rows, cols, 1), dtype=np.float32)
    else:
        tgt = images
    img = np.ndarray((total, rows, cols, 3), dtype=np.float32)

    for i, image_name in enumerate(images):
        img[i] = preprocessColor(imread(os.path.join(wd, dir_in, image_name)) / 255., True)
        if dir_out != '':
            tgt[i] = preprocessChannel(imread(os.path.join(wd, dir_out, image_name)) / 255., tgt_ch)
        if int(10. * (i + 1) / total) > int(10. * i / total):
            print('Leading %d / %d images [%.0f%%]' % (i + 1, total, 10 * int(10. * (i + 1) / total)))
    return img, tgt


def train(model, img, msk, target, num_epoch=12, cont_train=True):
    weight_file = target + ".h5"

    if cont_train and os.path.exists(weight_file):
        model.load_weights(weight_file)

    print('Creating model and checkpoint...')
    model_checkpoint = ModelCheckpoint(weight_file, monitor='val_loss', save_best_only=True)

    print('Fitting model...')
    model.fit(img, msk, batch_size=1, epochs=num_epoch, shuffle=True, validation_split=0.3, callbacks=[model_checkpoint])


def predict(model, tst, name, target, vec, ch):
    print('Load weights and predicting ...')
    model.load_weights(target + ".h5")
    imgs_mask_test = model.predict(tst, verbose=1, batch_size=1)

    target_dir = os.path.join(ImagePred, target)
    print('Saving predicted results [%s] to files...' % target)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    for i, image in enumerate(imgs_mask_test):
        target_file = os.path.join(target_dir, name[i].replace(".jpg", ".png"))
        target_new_dir = os.path.dirname(target_file)
        if not os.path.exists(target_new_dir):
            os.mkdir(target_new_dir)
        vec[i] = float(np.sum(image[:, :, 0]))
        print("%s pixel sum: %.1f" % (name[i], vec[i]))
        # image = (image[:, :, 0] * 255.).astype(np.uint8)  # pure BW
        # image = ((0.6 * image[:, :, 0] + 0.4 * (tst[i, :, :, 1] + 0.99)) * 127.).astype(np.uint8)  # gray mixed
        mix = tst[i].copy()
        for c in range(3):
            if c != ch:
                mix[:, :, c] = np.tanh(mix[:, :, c] - 0.3 * image[:, :, 0])
        mix = Image.fromarray(((mix + 1.) * 127.).astype(np.uint8), 'RGB')
        draw = ImageDraw.Draw(mix)
        draw.text((0, 0), " Pixels: %.0f / %.0f \n Percentage: %.0f%%" % (vec[i], image_sum, 100. * vec[i] / image_sum),
                  (255 if ch == 0 else 10, 255 if ch == 1 else 10, 255 if ch == 2 else 10),
                  ImageFont.truetype("arial.ttf", 36))  # font type size)
        imsave(target_file, mix)

if __name__ == '__main__':
    os.chdir(os.getcwd() + "//1_single_label_sigmoid")
    Original = "Original"
    Targets = ["Paren", "InflamMild", "InflamSevere"]  #
    model = get_unet5(image_rows, image_cols, 3, 1, 'sigmoid')
    # model_json = "Unet5.json"
    # with open(model_json, "w") as json_file:
    #     json_file.write(model.to_json())
    # with open(model_json, 'r') as json_file:
    #     model = model_from_json(json_file.read())
    model.compile(optimizer=Adam(lr=1e-6), loss=dice_coef_loss, metrics=[dice_coef])
    # optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),#
    for target in Targets:
        img, msk = get_data_pair(ImageOrig, Original, target, image_rows, image_cols, 2)  # Blue
        train(model, img, msk, target, 8, True)

    tst, name = get_data_pair(ImagePred, Original, '', image_rows, image_cols, 1)
    res = np.zeros((len(name), len(Targets)), np.float32)
    for x, target in enumerate(Targets):
            predict(model, tst, name, target, res[:, x], 2)
    res_df = pd.DataFrame(res, name, Targets)
    # res_df.to_csv("result.csv")
    res_df.to_excel("result.xlsx")


    # # multi-label softmax function
    # os.chdir(os.getcwd() + "//2_multi_label_sigmoid")
    # Original = "Original"
    # Targets = ["TriColor"]
    # model = get_unet5(image_rows, image_cols, 3, 1, 'sigmoid')
    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    # for target in Targets:
    #     for x in range(3):
    #         img, msk = get_data_pair(ImageOrig, Original, target, image_rows, image_cols, x)
    #         # imsave("test.jpg", (msk[7, :, :, x] * 255.).astype(np.uint8))
    #         train(model, img, msk, target + "_%s" % x, 5, True)
    #
    # tst, name = get_data_pair(ImagePred, Original, '', image_rows, image_cols, 1)
    # res = np.zeros((len(name), len(Targets)), np.float32)
    # for target in Targets:
    #     for x in range(3):
    #         predict(model, tst, name, target + "_%s" % x, res[:, x], x)
    # res_df = pd.DataFrame(res, name, Targets)
    # res_df.to_csv("test.csv")
