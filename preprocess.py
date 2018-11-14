import numpy as np
import imgaug as ia
from cv2 import cv2
from imgaug import augmenters as iaa
'''
Image format supported by opencv (cv2)
Windows bitmaps - *.bmp, *.dib (always supported)
JPEG files - *.jpeg, *.jpg, *.jpe (see the Notes section)
JPEG 2000 files - *.jp2 (see the Notes section)
Portable Network Graphics - *.png (see the Notes section)
WebP - *.webp (see the Notes section)
Portable image format - *.pbm, *.pgm, *.ppm (always supported)
Sun rasters - *.sr, *.ras (always supported)
TIFF files - *.tiff, *.tif (see the Notes section)
'''

def normalize_meanstd(a, axis=(1,2)):
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    norm = (a - mean) / std
    # norm /= 3.0
    return norm
    # return norm/(1+np.abs(norm)) # softsign
    # return np.tanh(norm) # tanh

def prep_scale(_img,fun=None):
    if fun=='tanh':
        return scale_tanh(_img)
        # return normalize_meanstd(scale_tanh(_img))
    elif fun=='sigmoid':
        return scale_sigmoid(_img)
    else:
        raise("function %s not supported" % fun)

def rev_scale(_img,fun=None):
    if fun=='tanh':
        rev=reverse_tanh(_img)
    elif fun=='sigmoid':
        rev=reverse_sigmoid(_img)
    else:
        raise("function %s not supported" % fun)
    return rev

def scale_sigmoid(_array):
    return _array.astype(np.float32)/255.0  # 0 ~ 1

def reverse_sigmoid(_array):
    return (_array.astype(np.float32)*255.0).astype(np.uint8)  # 0 ~ 1

def scale_tanh(_array):
    return _array.astype(np.float32)/127.5-1.0  # -1 ~ +1
    # return 1.0-_array.astype(np.float32)/127.5  # +1 ~ -1

def reverse_tanh(_array):
    return ((_array.astype(np.float32)+1.0)*127.5).astype(np.uint8)  # -1 ~ +1
    # return ((1.0-_array.astype(np.float32))*127.5).astype(np.uint8)  # +1 ~ -1


def read_resize_padding(_file, _resize, _padding):
    if _resize != 1.0:
        img = cv2.resize(cv2.imread(_file), (0, 0), fx=_resize, fy=_resize, interpolation=cv2.INTER_AREA)
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
        r0p = -r0; r0 = 0
    if c0 < 0:
        c0p = -c0; c0 = 0
    if r1 > _row:
        r1p = r1 - _row; r1 = _row
    if c1 > _col:
        c1p = c1 - _col; c1 = _col
    if r0p + r1p + c0p + c1p > 0:
        return np.pad(lg_img[r0:r1, c0:c1, ...], ((r0p, r1p), (c0p, c1p), (0, 0)), 'reflect')
    else:
        return lg_img[r0:r1, c0:c1, ...]

def skip_image(s_img, mode, if_print=True):
    def print_return(_if_skip, _report):
        if _if_skip:
            print(_report + " skip"); return True
        else:
            print(_report + " accept"); return False
    col = mode[0].lower()
    if col == 'g':  # green mask
        gmin = float(np.min(s_img[..., 0])) + float(np.min(s_img[..., 2]))  # min_R min_B
        if_skip = gmin > 12.0
        return print_return(if_skip, "checking tile for green mask (min_R+B=%.1f)" % gmin) if if_print else if_skip
    else:  # default white/black or rgb
        std = float(np.std(s_img))
        if_skip = std < 12.0
        return print_return(if_skip, "checking tile for contrast (std=%.1f)" % std) if if_print else if_skip


# image-mask pair # pad with reflect and black 0
aug_both_1 = iaa.Sequential([
    iaa.Fliplr(0.5),  # flip left-right 50% chance
    iaa.Flipud(0.5),  # flip up-down 50% chance
])
aug_both_2 = iaa.Sequential([
    iaa.Fliplr(0.5),  # flip left-right 50% chance
    iaa.Flipud(0.5),  # flip up-down 50% chance
    iaa.Sometimes(0.7, iaa.Affine(
        rotate=(-180, 180),  # rotate
        mode='reflect',  # use any of scikit-image's warping modes
        cval=0,  # if mode is constant, use a cval between 0 and 255
    )),
])
aug_both_3 = iaa.Sequential([
    iaa.Fliplr(0.5),  # flip left-right 50% chance
    iaa.Flipud(0.5),  # flip up-down 50% chance
    # iaa.CropAndPad(percent=0.2, pad_mode='constant', pad_cval=255), # pad with constance white 255
    iaa.Sometimes(0.7, iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, individually per axis
        # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
        rotate=(-180, 180),  # rotate
        shear=(-12, 12),  # shear by -16 to +16 degrees
        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
        mode='reflect',  # use any of scikit-image's warping modes
        cval=0  # if mode is constant, use a cval between 0 and 255
    )),
])
aug_img_1 = iaa.Sequential([  # only apply to original images not the mask
    iaa.Sometimes(0.6,
        iaa.OneOf([
          iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5),  # improve or worsen the contrast
          iaa.Grayscale(alpha=(0.0, 1.0))
        ]),
    ),
])
aug_img_2 = iaa.Sequential([  # only apply to original images not the mask
    iaa.Sometimes(0.6,
        iaa.OneOf([
            iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5),  # improve or worsen the contrast
            iaa.Grayscale(alpha=(0.0, 1.0)),
            iaa.Multiply((0.8,1.3)),
        ]),
    ),
    iaa.Sometimes(0.6,
        iaa.OneOf([
            iaa.GaussianBlur((0, 2.0)),  # blur sigma
            iaa.AverageBlur(k=(1, 5)),  # blur image using local means with kernel sizes between 2 and 7
            iaa.Sharpen((0, 1.0), lightness=(0.8, 1.3))  # sharpen
        ]),
    ),
])
aug_img_3 = iaa.Sequential([  # only apply to original images not the mask
    iaa.Sometimes(0.6,
        iaa.OneOf([
          iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5),  # improve or worsen the contrast
          iaa.Grayscale(alpha=(0.0, 1.0))
        ]),
    ),
    iaa.Sometimes(0.6,
        iaa.OneOf([
            iaa.GaussianBlur((0, 2.0)),  # blur sigma
            iaa.AverageBlur(k=(1, 5)),  # blur image using local means with kernel sizes between 2 and 7
            iaa.Sharpen((0, 1.0), lightness=(0.8, 1.3))  # sharpen
        ]),
    ),
    iaa.Sometimes(0.6,
        iaa.OneOf([
          iaa.Dropout((0.01, 0.05), per_channel=0.5),  # randomly remove pixels
          iaa.SaltAndPepper(p=(0.01, 0.05)),  # same white same black
          iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),  # add gaussian noise to images
        ]),
    ),
])
def augment_image_pair(_img, _tgt, _level):
    if _level<1:
        return _img, _tgt
    else:
        if 1<=_level<2:
            aug_det = aug_both_1.to_deterministic() # paired aug
            return aug_det.augment_images(_img), aug_det.augment_images(_tgt)
        elif 2<=_level<3:
            aug_det=aug_both_2.to_deterministic()  # paired aug
            return aug_img_1.augment_images(aug_det.augment_images(_img)), aug_det.augment_images(_tgt)
        elif 3<=_level<4:
            aug_det=aug_both_3.to_deterministic()  # paired aug
            return aug_img_2.augment_images(aug_det.augment_images(_img)), aug_det.augment_images(_tgt)
        else:
            aug_det=aug_both_3.to_deterministic()  # paired aug
            return aug_img_3.augment_images(aug_det.augment_images(_img)), aug_det.augment_images(_tgt)


def augment_image_set(_img,_msks,_level):
    if _level<1:
        return _img,_msks
    else:
        aug_det=aug_pat_1.to_deterministic()  # paired aug
        if 1<=_level<2:
            return aug_det.augment_image(_img),augment_per_channel(aug_det,_msks)
        elif 2<=_level<3:
            return aug_img_1.augment_image(aug_det.augment_image(_img)),augment_per_channel(aug_det,_msks)
        elif 3<=_level<4:
            return aug_img_2.augment_image(aug_det.augment_image(_img)),augment_per_channel(aug_det,_msks)
        else:
            return aug_img_3.augment_image(aug_det.augment_image(_img)),augment_per_channel(aug_det,_msks)

def augment_per_channel(_aug,_msks):
    for i in range(_msks.shape[-1]):
        _msks[...,i]=_aug.augment_image(_msks[...,i])
    return _msks



# Patch # padding with white 255
aug_pat_1 = iaa.Sequential([
    iaa.Fliplr(0.5),  # flip left-right 50% chance
    iaa.Flipud(0.5),  # flip up-down 50% chance
])
aug_pat_2 = iaa.Sequential([
    iaa.Fliplr(0.5),  # flip left-right 50% chance
    iaa.Flipud(0.5),  # flip up-down 50% chance
    iaa.Sometimes(0.7, iaa.Affine(
        rotate=(-180, 180),  # rotate
        mode='reflect',  # use any of scikit-image's warping modes
        cval=255,  # if mode is constant, use a cval between 0 and 255
    )),
])
aug_pat_3 = iaa.Sequential([
    iaa.Fliplr(0.5),  # flip left-right 50% chance
    iaa.Flipud(0.5),  # flip up-down 50% chance
    # iaa.CropAndPad(percent=0.2, pad_mode='constant', pad_cval=255), # pad with constance white 255
    iaa.Sometimes(0.7, iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, individually per axis
        # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
        rotate=(-180, 180),  # rotate
        shear=(-12, 12),  # shear by -16 to +16 degrees
        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
        cval=255,  # if mode is constant, use a cval between 0 and 255
        mode='constant'  # use any of scikit-image's warping modes
    )),
])
def augment_patch(_pat,_level):
    if _level<1:
        return _pat
    else:
        if 1<=_level<2:
            return aug_pat_1.augment_image(_pat)
        elif 2<=_level<3:
            return aug_pat_2.augment_image(_pat)
        else:
            return aug_pat_3.augment_image(_pat)
