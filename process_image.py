import numpy as np
import imgaug as ia
from cv2 import cv2
from imgaug import augmenters as iaa


def read_resize_padding(_file, _resize, _padding):
    if _resize < 1.0:
        img = cv2.resize(cv2.imread(_file), (0, 0), fx=_resize, fy=_resize, interpolation=INTER_AREA)
        # print(" Resize [%.1f] applied "%_resize,end='')
    else:
        img =  cv2.imread(_file)
        # print(" Resize [%.1f] not applied "%_resize,end='')
    if _padding > 1.0:
        row,col,_=img.shape
        row_pad=int(_padding*row-row)
        col_pad=int(_padding*col-col)
        # print(" Padding [%.1f] applied "%_padding,end='')
        return np.pad(img,((row_pad,row_pad),(col_pad,col_pad),(0,0)), 'reflect')
    else:
    #     print(" Padding [%.1f] not applied "%_padding,end='')
        return img


def extract_pad_image(lg_img, r0, r1, c0, c1):
    _row, _col, _ = lg_img.shape
    r0p, r1p, c0p, c1p = 0, 0, 0, 0
    if r0 < 0:
        r0p = -r0
        r0 = 0
    if c0 < 0:
        c0p = -c0
        c0 = 0
    if r1 > _row:
        r1p = r1 - _row
        r1 = _row
    if c1 > _col:
        c1p = c1 - _col
        c1 = _col
    if r0p + r1p + c0p + c1p > 0:
        return np.pad(lg_img[r0:r1, c0:c1, ...], ((r0p, r1p), (c0p, c1p), (0, 0)), 'reflect')
    else:
        return lg_img[r0:r1, c0:c1, ...]

def scale_input(_array):
    return _array.astype(np.float32) / 127.5 - 1.0
    # mean = np.mean(_array)
    # std = np.std(_array)
    # mean_per_channel = _array.mean(axis=(0, 1), keepdims=True)
    # print("  mean %.2f std %.2f" % (mean, std))
    # _array -= ((mean + 0.6) / 2.0)
    # _array /= _array.std(axis=(0, 1), keepdims=True)
    # return _array
def scale_input_reverse(_array):
    return (_array.astype(np.float32) + 1.0) * 127.5

def scale_output(_array, _color):
    _array=_array.astype(np.float32) / 255.0
    code=_color[0].lower()
    if code=='g':  # green
        # cv2.imwrite("testd_2f_-0.3.jpg",np.clip(2.0*(_array[..., 1] - _array[..., 0]-0.3), 0, 1)[..., np.newaxis][0]*255.)
        # cv2.imwrite("testd_2f_-0.4.jpg",np.clip(2.0*(_array[..., 1] - _array[..., 0]-0.4), 0, 1)[..., np.newaxis][0]*255.)
        # cv2.imwrite("testd_2f_-0.5.jpg",np.clip(2.0*(_array[..., 1] - _array[..., 0]-0.5), 0, 1)[..., np.newaxis][0]*255.)
        # cv2.imwrite("testd_2f_-0.6.jpg",np.clip(2.0*(_array[..., 1] - _array[..., 0]-0.6), 0, 1)[..., np.newaxis][0]*255.)
        return np.clip(2.0*(_array[..., 1] - _array[..., 0]-0.4), 0, 1)[..., np.newaxis]
    else:  # default to white/black from blue channel
        return _array[...,2][...,np.newaxis]  # blue channel to only channel

def augment_image_pair(_img, _tgt, _level=1.0):
    seg_both_1 = iaa.Sequential([
        iaa.Fliplr(0.5),  # flip left-right 50% chance
        iaa.Flipud(0.5),  # flip up-down 50% chance
    ])
    seg_both_2 = iaa.Sequential([
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
    ])
    seg_both_3 = iaa.Sequential([
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
    ])
    seq_img_4 = iaa.Sequential([  # only apply to original images not the mask
        iaa.OneOf([
            # iaa.Dropout((0.01, 0.05), per_channel=0.5),  # randomly remove pixels
            # iaa.SaltAndPepper(p=(0.01,0.05)), # same white same black
            # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),  # add gaussian noise to images
            iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5),  # improve or worsen the contrast
            iaa.Grayscale(alpha=(0.0, 1.0))
        ]),
    ])
    # if _level<0:
    #     return _img, _tgt
    # else:
    #     rep=50 / _img.shape[0] # duplicate if less than 50
    #     if rep>1:
    #         print("replicate %d times before augmentation" % rep)
    #         _img = np.repeat(_img, int(rep), axis=0)
    #         _tgt = np.repeat(_tgt, int(rep), axis=0)
    if _level<1:
        return _img, _tgt
    elif 1<=_level<2:  # paired image augmentation 1
        seq_det = seg_both_1.to_deterministic()
        return seq_det.augment_images(_img), seq_det.augment_images(_tgt)
    elif 2<=_level<3:  # paired image augmentation 2
        seq_det = seg_both_2.to_deterministic()
        return seq_det.augment_images(_img), seq_det.augment_images(_tgt)
    elif 3<=_level<4:  # paired image augmentation 3
        seq_det = seg_both_3.to_deterministic()
        return seq_det.augment_images(_img), seq_det.augment_images(_tgt)
    elif 4<=_level:  # paired aug + additional aug for original images
        seq_det = seg_both_3.to_deterministic()
        return seq_img_4.augment_images(seq_det.augment_images(_img)), seq_det.augment_images(_tgt)
