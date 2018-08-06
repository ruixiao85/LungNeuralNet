import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

def fit(_img, _r, _c):
    tr, tc, _ = _img.shape
    ri = int((tr - _r) / 2)
    ci = int((tc - _c) / 2)
    # ri = 0 if ri < 0 else ri
    # ci = 0 if ci < 0 else ci
    return _img[np.newaxis,ri:ri+_r,ci:ci+_c,...]

def scale_input(_array):
    return _array.astype(np.float32) / 127.5 - 1.0
    # mean = np.mean(_array)
    # std = np.std(_array)
    # mean_per_channel = _array.mean(axis=(0, 1), keepdims=True)
    # print("  mean %.2f std %.2f" % (mean, std))
    # _array -= ((mean + 0.6) / 2.0)
    # _array /= _array.std(axis=(0, 1), keepdims=True)
    # return _array

def scale_output(_array, _cfg):
    _array = _array.astype(np.float32) / 255.0
    if _cfg.dep_out==1:
        return _array[...,2][...,np.newaxis]  # blue channel to only channel
    else:
        _array[..., 0] = _array[..., 2]
        _array[..., 1] = 1.0 - _array[..., 2]
        return _array[..., 0:2]  # blue first, reverse second

def augment_image_pair(_img, _tgt):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # flip left-right 50% chance
        iaa.Flipud(0.5),  # flip up-down 50% chance
        iaa.Sometimes(0.8, iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},  # scale images to 80-120% of their size, individually per axis
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
            rotate=(-180, 180),  # rotate
            # shear=(-16, 16),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode='wrap'  # use any of scikit-image's warping modes
        )),
        iaa.Sometimes(0.7,
                      iaa.OneOf([
                          iaa.GaussianBlur((0, 2.0)),  # blur sigma
                          iaa.AverageBlur(k=(1, 5)),  # blur image using local means with kernel sizes between 2 and 7
                          iaa.Sharpen((0, 1.0), lightness=(0.8, 1.3))  # sharpen
                      ]),
                      ),
        # iaa.ContrastNormalization((0.75, 1.25), per_channel=0.5),  # improve or worsen the contrast, but will negatively affect target mask
        # iaa.OneOf([
        #     iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
        #     iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
        #     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),  # add gaussian noise to images
        # ])
    ])
    seq_det = seq.to_deterministic()  # get a deterministic stochastic sequence of augmenters. call this for each batch.
    rep=50 / _img.shape[0] # duplicate if less than 50
    if rep>1:
        print("replicate %d times before augmentation" % (rep))
        _img = np.repeat(_img, int(rep), axis=0)
        _tgt = np.repeat(_tgt, int(rep), axis=0)
    # from skimage.io import imsave, imread
    # ni, nt = _img.shape[0], _tgt.shape[0]
    # assert (ni == nt)
    # for i in range(ni):
    #     imsave('train_in.jpg', _img[i])
    #     imsave('train_out.jpg', _tgt[i])
    return seq_det.augment_images(_img), seq_det.augment_images(_tgt)
