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
        return scale_minus1_plus1(_img)
        # return normalize_meanstd(scale_tanh(_img))
    elif fun in ['sigmoid','softmax']:
        return scale_0_1(_img)
    else:
        raise("function %s not supported" % fun)

def rev_scale(_img,fun=None):
    if fun=='tanh':
        rev=reverse_minus1_plus1(_img)
    elif fun in ['sigmoid','softmax']:
        rev=reverse_0_1(_img)
    else:
        raise("function %s not supported" % fun)
    return rev

def scale_0_1(_array):
    return _array.astype(np.float32)/255.0  # 0 ~ 1

def reverse_0_1(_array):
    return (_array.astype(np.float32)*255.0).astype(np.uint8)  # 0 ~ 1

def scale_minus1_plus1(_array):
    return _array.astype(np.float32)/127.5-1.0  # -1 ~ +1
    # return 1.0-_array.astype(np.float32)/127.5  # +1 ~ -1

def reverse_minus1_plus1(_array):
    return ((_array.astype(np.float32)+1.0)*127.5).astype(np.uint8)  # -1 ~ +1
    # return ((1.0-_array.astype(np.float32))*127.5).astype(np.uint8)  # +1 ~ -1


def read_image(_file):
    return cv2.imread(_file)
    # from PIL import Image
    # return np.asarray(Image.open(_file))

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

def read_mask_default_zeros(_file,_row,_col):
    img=read_image(_file)
    return np.zeros((_row,_col),np.uint8) if img is None else img[...,1]/255

def extract_pad_image(lg_img, _row, _col, r0, r1, c0, c1, pad_value):
    # _row,_col=lg_img.shape[0:2]
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
        # print("padding %dx%d image by subset [r0 %d r1 %d, c0 %d c1 %d] and pad [r0p %d r1p %d, c0p %d c1p %d]"%(_row,_col,r0,r1,c0,c1,r0p,r1p,c0p,c1p))
        # return np.pad(lg_img[r0:r1, c0:c1, ...], ((r0p, r1p), (c0p, c1p), (0, 0)), 'reflect')
        return np.pad(lg_img[r0:r1, c0:c1, ...], ((r0p, r1p), (c0p, c1p), (0, 0)), mode='constant',constant_values=pad_value)
    else:
        return lg_img[r0:r1, c0:c1, ...]


class AugBase:
    def __init__(self,level,pad_params):
        self.level=level # aug_value higher=harsher
        self.pad_params=pad_params # {'mode':'constant','cval':0} # default pad black
        self.aug_shift=self.prep_shift()
        self.aug_decor=self.prep_decor()

    def prep_shift(self):
        if self.level<1: return None
        if self.level<2: return [iaa.Fliplr(0.5),iaa.Flipud(0.5)]
        if self.level<3: return [iaa.Fliplr(0.5),iaa.Flipud(0.5),iaa.SomeOf((0,1),iaa.Affine(scale={"x":(0.95,1.1),"y":(0.95,1.1)},order=[0,1],**self.pad_params))]
        if self.level<4: return [iaa.Fliplr(0.5),iaa.Flipud(0.5),iaa.SomeOf((0,1),iaa.Affine(scale={"x":(0.95,1.1),"y":(0.95,1.1)},**self.pad_params)),
                                iaa.SomeOf((0,2),[iaa.Affine(rotate=(-25,25),**self.pad_params),iaa.Affine(shear=(-5,5),**self.pad_params)])]
        return [iaa.Fliplr(0.5),iaa.Flipud(0.5),iaa.SomeOf((0,1),iaa.Affine(scale={"x":(0.95,1.1),"y":(0.95,1.1)},**self.pad_params)),
                iaa.SomeOf((0,3),[iaa.Affine(rotate=(-45,45),**self.pad_params),iaa.Affine(shear=(-8,8),**self.pad_params),
                iaa.PiecewiseAffine(scale=(0.00,0.02),**self.pad_params)])]

    def prep_decor(self):
        if self.level<1: return None
        if self.level<2: return [iaa.SomeOf((0,1),iaa.Multiply((0.9,1.12)))]
        if self.level<3: return [iaa.SomeOf((0,2),[iaa.Multiply((0.9,1.12)),iaa.ContrastNormalization((0.93,1.1),per_channel=True),iaa.Grayscale(alpha=(0.0,0.14)),iaa.AddToHueAndSaturation((-9,9))])]
        if self.level<4: return [iaa.SomeOf((0,3),[iaa.Multiply((0.9,1.12)),iaa.ContrastNormalization((0.93,1.1),per_channel=True),iaa.Grayscale(alpha=(0.0,0.14)),iaa.AddToHueAndSaturation((-9,9))]),
                                iaa.SomeOf((0,3),[iaa.GaussianBlur((0,0.3)),iaa.Sharpen((0,0.3),lightness=(0.95,1.1)),iaa.Emboss(alpha=(0,0.2),strength=(0.9,1.1))])]
        return [iaa.SomeOf((0,4),[iaa.Multiply((0.9,1.12)),iaa.ContrastNormalization((0.93,1.1),per_channel=True),iaa.Grayscale(alpha=(0.0,0.14)),iaa.AddToHueAndSaturation((-9,9))]),
                iaa.SomeOf((0,3),[iaa.GaussianBlur((0,0.6)),iaa.Sharpen((0,0.3),lightness=(0.9,1.1)),iaa.Emboss(alpha=(0,0.3),strength=(0.9,1.1))])]
        # decor=aug_decor_3()+[iaa.JpegCompression((0,50))]

    def decor1(self,_img):
        if self.aug_decor is None: return _img
        return iaa.Sequential(self.aug_decor).augment_images(_img) if _img.ndim==4 else iaa.Sequential(self.aug_decor).augment_image(_img)

    def shift1(self,_img):
        if self.aug_shift is None: return _img
        return iaa.Sequential(self.aug_shift).augment_images(_img) if _img.ndim==4 else iaa.Sequential(self.aug_shift).augment_image(_img)

    def shift2(self,_img,_msk):
        if self.aug_shift is None: return _img,_msk
        aug_det=iaa.Sequential(self.aug_shift).to_deterministic()
        return (aug_det.augment_images(_img),aug_det.augment_images(_msk)) if _img.ndim==4 else (aug_det.augment_image(_img),aug_det.augment_image(_msk))

    def shift2_decor1(self,_img,_msk):
        _img,_msk=self.shift2(_img,_msk)  # depending on pad_params
        _img=self.decor1(_img)
        return _img,_msk

class AugImageMask(AugBase):
    def __init__(self,level):
        super(AugImageMask,self).__init__(level,pad_params={'mode':'reflect'})  # 0 reflect

class AugPatchMask(AugBase):
    def __init__(self,level):
        super(AugPatchMask,self).__init__(level,pad_params={'mode':'constant','cval':255})  # 1 white

    def pad_shift2_decor1_crop(self,_img,_msk):
        row,col,_=_img.shape
        length=math.ceil((row**2+col**2)**0.5)
        rp,cp=(length-row)//2,(length-col)//2
        _img=np.pad(_img,pad_width=((rp,rp),(cp,cp),(0,0)),mode='constant',constant_values=255)
        _msk=(255-np.pad(_msk,pad_width=((rp,rp),(cp,cp),(0,0)),mode='constant',constant_values=0)).astype(np.bool) # pad zero and inverse
        _img,_msk=self.shift2_decor1(_img,_msk) # pad white 255
        _msk=255-255*(_msk.astype(np.uint8))  # inverse back
        y1,x1,y2,x2=extract_bboxes(_msk)[0]
        _img,_msk=_img[y1:y2,x1:x2,...],_msk[y1:y2,x1:x2,...]
        _img_max=np.max(_img)
        return 255-_img_max+_img,_msk

