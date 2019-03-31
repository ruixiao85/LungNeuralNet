import math
from functools import lru_cache

import numpy as np
import imgaug as ia
from cv2 import cv2
from imgaug import augmenters as iaa

from c2_mrcnn_matterport import extract_bboxes

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
pad_params=[
    {'mode':'reflect'}, # 0 reflect
    {'mode':'constant','cval':255}, # 1 white
    # {'mode':'constant','cval':0}, # 2 black
]
cache_size=2 # >=size of pad_params
@lru_cache(maxsize=cache_size,typed=False)
def aug_shift_1(pad_type)->list: return [iaa.Fliplr(0.5),iaa.Flipud(0.5)]
@lru_cache(maxsize=cache_size,typed=False)
def aug_shift_2(pad_type)->list: return aug_shift_1(pad_type)+[iaa.Affine(scale={"x":(0.95,1.1),"y":(0.95,1.1)},order=[0,1],**pad_params[pad_type])]
@lru_cache(maxsize=cache_size,typed=False)
def aug_shift_3(pad_type)->list: return aug_shift_2(pad_type)+[iaa.OneOf([iaa.Affine(rotate=(-25,25),**pad_params[pad_type]), iaa.Affine(shear=(-5,5),**pad_params[pad_type])])]
@lru_cache(maxsize=cache_size,typed=False)
def aug_shift_4(pad_type)->list:
    # filtered_kwargs={k:v for k,v in kwargs.items() if k not in ['backend','fit_output']} # PiecewiseAffine does not support these
    return aug_shift_2(pad_type)+[iaa.OneOf([iaa.Affine(rotate=(-45,45),**pad_params[pad_type]), iaa.Affine(shear=(-8,8),**pad_params[pad_type]),
                                             iaa.PiecewiseAffine(scale=(0.00,0.02),**pad_params[pad_type])])]

## decorative, not moving/shifting places. suitable for original RGB image ##
@lru_cache(maxsize=cache_size,typed=False)
def aug_decor_1(pad_type)->list: return [iaa.Multiply((0.9,1.12))]
@lru_cache(maxsize=cache_size,typed=False)
def aug_decor_2(pad_type)->list: return [iaa.OneOf([iaa.Multiply((0.9,1.12)),iaa.ContrastNormalization((0.93,1.1),per_channel=True),iaa.Grayscale(alpha=(0.0,0.14)),iaa.AddToHueAndSaturation((-9,9))])]
@lru_cache(maxsize=cache_size,typed=False)
def aug_decor_3(pad_type)->list: return aug_decor_2(pad_type)+[iaa.OneOf([iaa.GaussianBlur((0,0.3)),iaa.Sharpen((0,0.3),lightness=(0.95,1.1)),iaa.Emboss(alpha=(0,0.2),strength=(0.9,1.1))])]
@lru_cache(maxsize=cache_size,typed=False)
def aug_decor_4(pad_type)->list: return aug_decor_2(pad_type)+[iaa.OneOf([iaa.GaussianBlur((0,0.6)),iaa.Sharpen((0,0.3),lightness=(0.9,1.1)),iaa.Emboss(alpha=(0,0.3),strength=(0.9,1.1))])]
# def aug_decor_4(pad_type)->list: return aug_decor_3()+[iaa.JpegCompression((0,50))]

def augment_single(_img,_level,_list,_pad_type):
    if _level<1: return _img
    else:
        level_idx=min(math.floor(_level),len(_list))-1
        return iaa.Sequential(_list[level_idx](_pad_type)).augment_images(_img) if _img.ndim==4 else iaa.Sequential(_list[level_idx](_pad_type)).augment_image(_img)
def augment_dual(_img,_msk,_level,_list,_pad_type):
    if _level<1: return _img,_msk
    else:
        level_idx=min(math.floor(_level),len(_list))-1
        aug_det=iaa.Sequential(_list[level_idx](_pad_type)).to_deterministic()
        return (aug_det.augment_images(_img),aug_det.augment_images(_msk)) if _img.ndim==4 else\
               (aug_det.augment_image(_img),aug_det.augment_image(_msk))


def augment_single_decor(_img,_level,_pad_type=0):
    return augment_single(_img,_level,_list=[aug_decor_1,aug_decor_2,aug_decor_3,aug_decor_4],_pad_type=_pad_type)
def augment_single_shift(_img,_level,_pad_type=0):
    return augment_single(_img,_level,_list=[aug_shift_1,aug_shift_2,aug_shift_3,aug_shift_4],_pad_type=_pad_type)
def augment_dual_shift(_img,_msk,_level,_pad_type=0):
    return augment_dual(_img,_msk,_level,_list=[aug_shift_1,aug_shift_2,aug_shift_3,aug_shift_4],_pad_type=_pad_type)

def augment_image_mask_pair(_img,_msk,_level):
    _img,_msk=augment_dual_shift(_img,_msk,_level,_pad_type=0) # 0 reflect to positively-mark background
    # _img,_msk=augment_dual_shift(_img,255-_msk,_level,_pad_type=1) # 1 white
    _img=augment_single_decor(_img,_level)
    return _img,_msk # reflect
    # return _img,255-_msk # constant

def augment_patch_mask_pair(_img,_msk,_level):
    # milder simplier approach
    _img,_msk=augment_dual_shift(_img,255-_msk,min(1,_level),_pad_type=1) # 1 white pad RGB patches and inversed mask, and insure fit
    _img=augment_single_decor(_img,min(2,_level)) # milder augmentations on patches

    # normal but pad->crop approach
    # row,col,_=_img.shape
    # length=(row**2+col**2)**0.5
    # rp,cp=(length-row)//2,(length-col)//2
    # _img=np.pad(_img,((rp,length-row-rp),(cp,length-col-cp)),mode='constant',constant_values=255)
    # _msk=np.pad(_msk,((rp,length-row-rp),(cp,length-col-cp)),mode='constant',constant_values=0)
    # _img,_msk=augment_dual_shift(_img,255-_msk,_level,_pad_type=1) # 1 white pad RGB patches and inversed mask, and insure fit
    # _img=augment_single_decor(_img,_level) # milder augmentations on patches
    # y1,x1,y2,x2=extract_bboxes(_msk)[0]
    # _img,_msk=_img[y1:y2,x1:x2,...],_msk[y1:y2,x1:x2,...]

    _img_max=np.max(_img)
    # if _img_max!=255: print(_img_max)
    return 255-_img_max+_img, 255-_msk
