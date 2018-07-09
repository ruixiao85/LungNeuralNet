import os
import numpy as np
from skimage.io import imsave, imread
from model import dice_coef, dice_coef_loss, get_unet3, get_unet5, get_unet6
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


def standardize(preimg):
    mean = np.mean(preimg)
    std = np.std(preimg)
    preimg -= mean
    preimg /= (2. * std)
    preimg = np.tanh(preimg)
    return preimg

def preprocess(preimg, gray, stand):
    if (gray):
        if (stand):
            preimg = standardize(preimg)
        return preimg[..., np.newaxis]
    else:
        preimg = preimg / 255.
        if (stand):
            preimg = standardize(preimg)
        return preimg


WorkDir = os.getcwd()
ImageOrig = "ori"
ImageMask = "msk"
ImageTest = "test"
ImagePred = "pred"
image_cols = 696
image_rows = 520
ifGray = False
nChannel = 1 if ifGray else 3
weightFile = 'weights.h5'


def train_and_predict(contTrain=True):
    print("Scanning subfolders [/%s] and [/%s] of [%s]" % (ImageOrig, ImageMask, WorkDir))
    images = os.listdir(os.path.join(WorkDir, ImageOrig))
    masks = os.listdir(os.path.join(WorkDir, ImageMask))
    print("%d images and %d masks found" % (len(images), len(masks)))

    images = list(set(images).intersection(masks))  # image-mask pairs only
    total_train = len(images)
    print("%d image-mask pairs accepted" % len(images))

    img = np.ndarray((total_train, image_rows, image_cols, nChannel), dtype=np.float32)
    msk = np.ndarray((total_train, image_rows, image_cols, 1), dtype=np.float32)

    for i, image_name in enumerate(images):
        img[i] = preprocess(imread(os.path.join(WorkDir, ImageOrig, image_name), as_grey=ifGray), ifGray, True)
        # imsave("testimg.jpg", img[i])
        msk[i] = preprocess(imread(os.path.join(WorkDir, ImageMask, image_name), as_grey=True), True, False)
        # imsave("testmsk.jpg", msk[i])
        print('Done: {0}/{1} images'.format(i, total_train))

    # model = get_unet3(image_rows, image_cols, nChannel)
    model = get_unet5(image_rows, image_cols, nChannel)
    # model = get_unet6(image_rows, image_cols, nChannel)

    if contTrain and os.path.exists(weightFile):
        model.load_weights(weightFile)
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    print('Creating model and checkpoint...')
    model_checkpoint = ModelCheckpoint(weightFile, monitor='val_loss', save_best_only=True)

    print('Fitting model...')
    model.fit(img, msk, batch_size=4, epochs=12, verbose=1, shuffle=True,
              validation_split=0.2, callbacks=[model_checkpoint])

    print('Loading and preprocessing test data...')
    print("Scanning test subfolders [/%s] of [%s]" % (ImageTest, WorkDir))
    testimages = os.listdir(os.path.join(WorkDir, ImageTest))

    tst = np.ndarray((len(testimages), image_rows, image_cols, nChannel), dtype=np.float32)
    total_test = len(tst)

    for i, image_name in enumerate(testimages):
        tst[i] = preprocess(imread(os.path.join(WorkDir, ImageTest, image_name), as_grey=ifGray), ifGray, True)
        print('Done: {0}/{1} images'.format(i, total_test))

    print('Predicting masks on test data...')
    # model.load_weights(weightFile)
    imgs_mask_test = model.predict(tst, verbose=1)

    print('Saving predicted masks to files...')
    if not os.path.exists(ImagePred):
        os.mkdir(ImagePred)
    for i, image in enumerate(imgs_mask_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(ImagePred, testimages[i] + '_pred.png'), image)


if __name__ == '__main__':
    train_and_predict(True)
