import os
import numpy as np
from skimage.io import imsave, imread
from model import dice_coef, dice_coef_loss, get_unet3, get_unet4, get_unet5, get_unet6
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop

ImageOrig, ImagePred = "train", "pred"
image_cols, image_rows = 1392, 1040
Original = "Original"
Targets = ["Paren", "InflamMild", "InflamSevere"]


def standardize(preimg):
    mean, std = np.mean(preimg), np.std(preimg)
    # print("  mean %.2f std %.2f" % (mean, std))
    preimg -= mean
    preimg /= (4. * std)
    preimg = np.tanh(preimg)
    # mean = np.mean(preimg)
    # preimg -= mean
    return preimg


def preprocess(preimg, gray, stand):
    if (gray):
        if (stand):
            preimg = standardize(preimg)
        # np.savetxt("debugGray.csv", preimg, delimiter=',')
        return preimg[..., np.newaxis]
    else:
        preimg = preimg / 255.
        if (stand):
            preimg = standardize(preimg)
        # np.savetxt("debugColor.csv", preimg[:, :, 1], delimiter=',')
        return preimg


def get_recursive_rel_path(fp, ext='*.jpg'):
    from glob import glob
    images = [y for x in os.walk(fp) for y in glob(os.path.join(x[0], ext))]
    for i in range(len(images)):
        images[i] = os.path.relpath(images[i], fp)
    return images


def get_data_pair(ori, dir_in, dir_out, rows, cols, ori_ch, tgt_ch):
    wd = os.path.join(os.getcwd(), ori)
    images = get_recursive_rel_path(os.path.join(wd, dir_in))
    total = len(images)
    print("Found [%d] file from subfolders [/%s] of [%s]" % (total, dir_in, wd))

    ori_gray = True if (ori_ch == 1) else False
    tgt_gray = True if (tgt_ch == 1) else False
    if dir_out != '':
        targets = get_recursive_rel_path(os.path.join(wd, dir_out))
        print("Found [%d] file from subfolders [/%s] of [%s]" % (len(targets), dir_out, wd))

        images = list(set(images).intersection(targets))  # image-target pairs only
        total = len(images)  # update # of files
        print("%d image-mask pairs accepted" % total)
        tgt = np.ndarray((total, rows, cols, tgt_ch), dtype=np.float32)
    else:
        tgt = images
    img = np.ndarray((total, rows, cols, ori_ch), dtype=np.float32)

    for i, image_name in enumerate(images):
        img[i] = preprocess(imread(os.path.join(wd, dir_in, image_name), as_grey=ori_gray), ori_gray, True)
        if dir_out != '':
            tgt[i] = preprocess(imread(os.path.join(wd, dir_out, image_name), as_grey=tgt_gray), tgt_gray, False)
        if int(10. * (i + 1) / total) > int(10. * i / total):
            print('Done: %d/%d images [%.0f%%]' % (i + 1, total, 10 * int(10. * (i + 1) / total)))
    return img, tgt


def train(model, target, num_epoch=12, cont_train=True):
    img, msk = get_data_pair(ImageOrig, Original, target, image_rows, image_cols, dim_in, dim_out)
    weight_file = target + ".h5"

    if cont_train and os.path.exists(weight_file):
        model.load_weights(weight_file)

    print('Creating model and checkpoint...')
    model_checkpoint = ModelCheckpoint(weight_file, monitor='val_loss', save_best_only=True)

    print('Fitting model...')
    model.fit(img, msk, batch_size=1, epochs=num_epoch, verbose=1, shuffle=True,
              validation_split=0.3, callbacks=[model_checkpoint])


def predict(model, target):
    tst, name = get_data_pair(ImagePred, Original, '', image_rows, image_cols, dim_in, dim_out)

    print('Load weights and predicting ...')
    model.load_weights(target + ".h5")
    imgs_mask_test = model.predict(tst, verbose=1, batch_size=1)

    target_dir = os.path.join(ImagePred, target)
    print('Saving predicted results [%s] to files...' % target)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    for i, image in enumerate(imgs_mask_test):
        print("%s pixel sum: %.1f" % (name[i], np.sum(image[:, :, 0])))
        # image = (image[:, :, 0] * 255.).astype(np.uint8)  # pure BW
        # image = ((0.6 * image[:, :, 0] + 0.4 * (tst[i][:, :, 1] + 0.99)) * 127.).astype(np.uint8)  # gray mixed
        tst[i][:, :, 2] += 0.49 * image[:, :, 0]
        target_file = os.path.join(target_dir, name[i]).replace(".jpg", ".png")
        target_dir = os.path.dirname(target_file)
        if (not os.path.exists(target_dir)):
            os.mkdir(target_dir)
        imsave(target_file,  # image # for gray or BW
               ((tst[i]+0.99) * 127.).astype(np.uint8))


if __name__ == '__main__':
    # single label sigmoid function
    os.chdir(os.getcwd() + "//1_single_label_sigmoid")
    dim_in, dim_out = 3, 1
    model = get_unet5(image_rows, image_cols, dim_in, dim_out, 'sigmoid')
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    # optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),#
    for target in Targets:
        train(model, target, 8, True)
        predict(model, target)

    # # multi-label softmax function
    # os.chdir(os.getcwd() + "//2_multi_label_softmax")
    # dim_in, dim_out = 3, 3
    # model = get_unet5(image_rows, image_cols, dim_in, dim_out, 'softmax')
    # model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])
    # for target in ImageTargets:
    #     train(model, target, True)
    #     predict(model, target)
