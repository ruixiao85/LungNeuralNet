import math

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


def read_image(_file):
    return cv2.imread(_file)

def read_resize(_file, _resize=1.0):
    img=read_image(_file)
    if _resize != 1.0:
        img=cv2.resize(img, (0, 0), fx=_resize, fy=_resize, interpolation=cv2.INTER_AREA)
    return img

def read_resize_pad(_file, _resize=1.0,_padding=1.0):
    img=read_resize(_file, _resize)
    if _padding > 1.0:
        row,col,_=img.shape
        row_pad=int(_padding*row-row)
        col_pad=int(_padding*col-col)
        # print(" Padding [%.1f] applied "%_padding,end='')
        img=np.pad(img,((row_pad,row_pad),(col_pad,col_pad),(0,0)), 'reflect')
    return img

def read_resize_fit(_file, _resize, _row, _col):
    img=read_resize(_file, _resize)
    row,col,_=img.shape
    if row==_row and col==_col:
        return img
    if row<_row or col<_col: # pad needed
        row_pad=max(0,int(math.ceil(_row-row)/2.0))
        col_pad=max(0,int(math.ceil(_col-col)/2.0))
        # print(" Padding [%.1f] applied "%_padding,end='')
        img=np.pad(img,((row_pad,row_pad),(col_pad,col_pad),(0,0)), 'reflect')
    ri=(row-_row)//2; ci=(col-_col)//2
    return img[ri:ri+_row,ci:ci+_col,...]

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
        print("padding %dx%d image by subset [r0 %d r1 %d, c0 %d c1 %d] and padd [r0p %d r1p %d, c0p %d c1p %d]"%(_row,_col,r0,r1,c0,c1,r0p,r1p,c0p,c1p))
        return np.pad(lg_img[r0:r1, c0:c1, ...], ((r0p, r1p), (c0p, c1p), (0, 0)), 'reflect')
    else:
        return lg_img[r0:r1, c0:c1, ...]


## shifting, suitable to apply to image-mask pairs ##
# padblack={'mode':'constant','cval':0} # default black pad
# padkwargs={'mode':'reflect'}
# symkwargs={'mode':'symmetric'}
def aug_shift_1(**kwargs)->list: return [iaa.Fliplr(0.5),iaa.Flipud(0.5)]
def aug_shift_2(**kwargs)->list: return aug_shift_1(**kwargs)+[iaa.Affine(rotate=(-12,12),shear=(-8,8),order=[0,1],**kwargs)]
def aug_shift_3(**kwargs)->list: return aug_shift_1(**kwargs)+[iaa.Affine(rotate=(-25,25),shear=(-12,12),scale={"x":(0.9,1.15),"y":(0.9,1.15)},order=[0,1],**kwargs)]
def aug_shift_4(**kwargs)->list:
    # filtered_kwargs={k:v for k,v in kwargs.items() if k not in ['backend','fit_output']} # PiecewiseAffine does not support these
    return aug_shift_1(**kwargs)+[iaa.Affine(rotate=(-45,45),shear=(-15,15),scale={"x":(0.85,1.2),"y":(0.85,1.2)},order=[0,1],**kwargs),
        iaa.PiecewiseAffine(scale=(0.00,0.03),**kwargs)]

## decorative, not moving/shifting places. suitable for original RGB image ##
def aug_decor_1(**kwargs)->list: return [iaa.Multiply((0.9,1.1))]
def aug_decor_2(**kwargs)->list: return [iaa.OneOf([iaa.Multiply((0.8,1.2)),iaa.ContrastNormalization((0.9,1.1),per_channel=True),iaa.Grayscale(alpha=(0.0,0.25)),iaa.AddToHueAndSaturation((-14,14))])]
def aug_decor_3(**kwargs)->list: return aug_decor_2()+[iaa.OneOf([iaa.GaussianBlur((0,0.6)),iaa.Sharpen((0,0.5),lightness=(0.9,1.2)),iaa.Emboss(alpha=(0,0.4),strength=(0.8,1.2))])]
def aug_decor_4(**kwargs)->list: return aug_decor_2()+[iaa.OneOf([iaa.GaussianBlur((0,1.2)),iaa.Sharpen((0,0.7),lightness=(0.85,1.25)),iaa.Emboss(alpha=(0,0.6),strength=(0.7,1.3))])]
# def aug_decor_4(**kwargs)->list: return aug_decor_3()+[iaa.JpegCompression((0,50))]

def augment_single(_img,_level,_list,_kwargs):
    if _level<1: return _img
    else:
        level_id=min(math.floor(_level),len(_list))-1
        return iaa.Sequential(_list[level_id](**_kwargs)).augment_images(_img) if _img.ndim==4 else iaa.Sequential(_list[level_id](**_kwargs)).augment_image(_img)
def augment_dual(_img,_msk,_level,_list,_kwargs):
    if _level<1: return _img,_msk
    else:
        level_id=min(math.floor(_level),len(_list))-1
        aug_det=iaa.Sequential(_list[level_id](**_kwargs)).to_deterministic()
        return (aug_det.augment_images(_img),aug_det.augment_images(_msk)) if _img.ndim==4 else\
               (aug_det.augment_image(_img),aug_det.augment_image(_msk))


def augment_single_decor(_img,_level,_kwargs=None):
    _kwargs=_kwargs or {'None':'None'}
    return augment_single(_img,_level,_list=[aug_decor_1,aug_decor_2,aug_decor_3,aug_decor_4],_kwargs=_kwargs)
def augment_single_shift(_img,_level,_kwargs=None):
    _kwargs=_kwargs or {'mode':'reflect'}
    return augment_single(_img,_level,_list=[aug_shift_1,aug_shift_2,aug_shift_3,aug_shift_4],_kwargs=_kwargs)
def augment_dual_shift(_img,_msk,_level,_kwargs=None):
    _kwargs=_kwargs or {'mode':'reflect'}
    return augment_dual(_img,_msk,_level,_list=[aug_shift_1,aug_shift_2,aug_shift_3,aug_shift_4],_kwargs=_kwargs)

def augment_image_mask_pair(_img,_msk,_level):
    _img,_msk=augment_dual_shift(_img,_msk,_level,_kwargs={'mode':'reflect'})
    _img=augment_single_decor(_img,_level)
    return _img,_msk

def augment_patch_mask_pair(_img,_msk,_level):
    _msk=255-_msk # inverse to pad 255 to the outer edge
    _img,_msk=augment_dual_shift(_img,_msk,min(1,_level),_kwargs={'mode':'constant','cval':255}) # pad white to RGB patches, and insure fit
    _img=augment_single_decor(_img,min(2,_level)) # milder augmentations on patches
    _img_max=np.max(_img)
    # if _img_max!=255: print(_img_max)
    return 255-_img_max+_img, 255-_msk
