import math
import os
import shutil
import threading

import numpy as np
import keras
from cv2.cv2 import imread, resize, imwrite

from process_image import scale_input, scale_output, augment_image_pair
from util import get_recursive_rel_path


class MetaInfo:
    def __init__(self, file, r_index, c_index, ori_row, ori_col, ori_dep, ri, ro, ci, co):
        self.file_name = file
        self.r_index = r_index
        self.c_index = c_index
        self.ori_row = ori_row
        self.ori_col = ori_col
        self.ori_dep = ori_dep
        self.row_start = ri
        self.row_end = ro
        self.col_start = ci
        self.col_end = co

    def file_slice(self):
        if self.r_index is not None:
            return self.file_name.replace(".jpg", "_r%d_c%d.jpg" % (self.r_index,self.c_index))
        return self.file_name

    def __str__(self):
        return str(self.file_name)

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

class ImageGenerator(keras.utils.Sequence):
    def __init__(self, name, cfg, wd, dir_in, dir_out, shuffle, batch_size=2, img_aug=False):
        self.name=name
        self.resize=cfg.resize
        self.row_in, self.col_in, self.dep_in= cfg.row_in, cfg.col_in, cfg.dep_in
        self.row_out, self.col_out, self.dep_out= cfg.row_out, cfg.col_out, cfg.dep_out
        self.full=cfg.full
        self.wd=wd
        self.dir_in, self.dir_out=dir_in, dir_out
        self.shuffle=shuffle
        self.img_aug=img_aug
        self.batch_size=batch_size
        self.indexes=[]
        self.view_coord=[]
        self.skip_convert=os.path.exists(os.path.join(self.wd, self.dir_in_reg()))\
                          and (os.path.exists(os.path.join(self.wd,self.dir_out_reg())) or self.dir_out is None)
        if  self.skip_convert:
            images, _ = get_recursive_rel_path(self.wd, self.dir_in_reg())
            for img in images:
                self.view_coord.append(MetaInfo(img, None, None, self.row_out, self.col_out, self.dep_out, 0, self.row_out, 0, self.col_out))
        # self.lock=threading.Lock()
        # self.dep=3
        # self.lab=1

    def dir_in_reg(self):
        return "%s%s_%s_%dx%d"%(self.dir_in,self.dir_out,self.name,self.row_in,self.col_in)

    def dir_out_reg(self):
        return "%s%s_%s_%dx%d"%(self.dir_out,self.dir_in,self.name,self.row_out,self.col_out)

    def __len__(self):  # Denotes the number of batches per epoch
        return int(np.floor(len(self.view_coord) / self.batch_size))

    def __getitem__(self, index):  # Generate one batch of data
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # print(" getting index %d with %d batch size"%(index,self.batch_size))
        return self.__data_generation([self.view_coord[k] for k in indexes])

    def on_epoch_end(self):  # Updates indexes after each epoch
        self.indexes = np.arange(len(self.view_coord))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def recreate_dir(self):
        down_img = os.path.join(self.wd, self.dir_in_reg())
        down_tgt = os.path.join(self.wd, self.dir_out_reg())
        if os.path.exists(down_img):
            shutil.rmtree(down_img)
        if os.path.exists(down_tgt):
            shutil.rmtree(down_tgt)
        os.mkdir(down_img)
        os.mkdir(down_tgt)

    def register_image(self, image_name):
        down_img=os.path.join(self.wd,self.dir_in_reg())
        down_tgt=os.path.join(self.wd,self.dir_out_reg())
        rd = int(round(0.5 * (self.row_in - self.row_out), 0))
        cd = int(round(0.5 * (self.col_in - self.col_out), 0))
        _img = imread(os.path.join(self.wd, self.dir_in, image_name))
        lg_row, lg_col, lg_dep=_img.shape
        if self.dir_out is not None:  # training mode check file size
            _tgt = imread(os.path.join(self.wd, self.dir_out, image_name))
            tgt_row, tgt_col, _=_tgt.shape
            if lg_row!=tgt_row or lg_col!=tgt_col:
                print("Skipping for size mismatch in %s"%image_name)
                return 0 # skip the file
        if self.full:
            redundancy=1.2 # more
            r_len = max(1,int(math.ceil(redundancy*(lg_row-self.row_out) / self.row_out)))
            c_len = max(1,int(math.ceil(redundancy*(lg_col-self.col_out) / self.col_out)))
        else:
            sparsity=1.0 # less
            r_len = max(1,int(math.floor(sparsity*(lg_row-self.row_out) / self.row_out)))
            c_len = max(1,int(math.floor(sparsity*(lg_col-self.col_out) / self.col_out)))
        print("%s target %d x %d (full %r): original %d x %d x %d ->  row /%d col /%d" %
              (image_name, self.row_out, self.col_out, self.full, lg_row, lg_col, lg_dep, r_len, c_len))
        r_step = float(lg_row - self.row_out) / (r_len - 1) if r_len > 1 else 0
        c_step = float(lg_col - self.col_out) / (c_len - 1) if c_len > 1 else 0
        for r_index in range(r_len):
            for c_index in range(c_len):
                ri = int(round(r_index * r_step))
                ci = int(round(c_index * c_step))
                ro = ri+self.row_out
                co=ci+self.col_out
                s_img = _img[ri:ro, ci:co, ...]
                s_tgt=_tgt[ri:ro,ci:co,...]
                if np.std(s_img) < 20 or np.std(s_tgt) < 5:  # skip low contrast image
                    continue
                entry = MetaInfo(image_name, r_index, c_index, lg_row, lg_col, lg_dep, ri, ro, ci, co)
                self.view_coord.append(entry)
                if rd != 0 or cd != 0:
                    r0, r1 = ri - rd, ro + rd
                    c0, c1 = ci- cd, co + cd
                    r0p, r1p, c0p, c1p = 0, 0, 0, 0
                    if r0 < 0:
                        r0p = -r0
                        r0 = 0
                    if c0 < 0:
                        c0p = -c0
                        c0 = 0
                    if r1 > lg_row:
                        r1p = r1 - lg_row
                        r1 = lg_row
                    if c1 > lg_col:
                        c1p = c1 - lg_col
                        c1 = lg_col
                    s_img= np.pad(_img[r0:r1, c0:c1, ...], ((r0p, r1p), (c0p, c1p), (0, 0)), 'reflect')
                imwrite(os.path.join(down_img,entry.file_slice()),s_img)
                imwrite(os.path.join(down_tgt,entry.file_slice()),s_tgt)
        # print("%s was split into %d views and added"%(image_name,p_index))
        return 1

    def __data_generation(self, view_coords):  # Generates data containing batch_size samples # X : (n_samples, *dim, n_channels)
        _img=np.zeros((self.batch_size, self.row_in, self.col_in, self.dep_in),dtype=np.uint8)
        for i,vc in enumerate(view_coords):
            _img[i,...]=imread(os.path.join(self.wd, self.dir_in_reg(), vc.file_slice()))
        if self.dir_out is not None:
            _tgt = np.zeros((self.batch_size, self.row_out, self.col_out, 3),dtype=np.uint8)
            for i,vc in enumerate(view_coords):
                _tgt[i,...] =imread(os.path.join(self.wd, self.dir_out_reg(), vc.file_slice()))
        else:
            _tgt=None
        if self.dir_out is not None:
            if self.img_aug:
                _img,_tgt=augment_image_pair(_img, _tgt, _level=1)
            _img,_tgt=scale_input(_img),scale_output(_tgt,self.dep_out)
        else:
            _img=scale_input(_img)
        return _img, _tgt

    def __data_generation_old(self, view_coords):  # Generates data containing batch_size samples # X : (n_samples, *dim, n_channels)
        _img=np.zeros((self.batch_size, self.row_in, self.col_in, self.dep_in),dtype=np.uint8)
        rd=round(0.5*(self.row_in-self.row_out),0)
        cd=round(0.5*(self.col_in-self.col_out),0)
        assert(self.row_in>=self.row_out and self.col_in>=self.col_out)
        for i,vc in enumerate(view_coords):
            large=imread(os.path.join(self.wd, self.dir_in, vc.file_name))
            if rd==0 and cd==0:
                _img[i,...]=large[vc.row_start:vc.row_end, vc.col_start:vc.col_end, ...]
            else:
                lg_row,lg_col,lg_dep=large.shape
                r0, r1 = vc.row_start - rd, vc.row_end + rd
                c0, c1 = vc.col_start - cd, vc.col_end + cd
                r0p, r1p, c0p, c1p = 0, 0, 0, 0
                if r0 < 0:
                    r0p = -r0
                    r0 = 0
                if c0 < 0:
                    c0p = -c0
                    c0 = 0
                if r1 > lg_row:
                    r1p = r1 - lg_row
                    r1 = lg_row
                if c1 > lg_col:
                    c1p = c1 - lg_col
                    c1 = lg_col
                _img[i,...] =np.pad(large[r0:r1,c0:c1,...],((r0p,r1p),(c0p,c1p),(0,0)),'reflect')
        if self.dir_out is not None:
            _tgt = np.zeros((self.batch_size, self.row_out, self.col_out, 3),dtype=np.uint8)
            for i,vc in enumerate(view_coords):
                large = imread(os.path.join(self.wd, self.dir_out, vc.file_name))
                _tgt[i,...] =large[vc.row_start:vc.row_end, vc.col_start:vc.col_end,...]
        else:
            _tgt=None
        if self.dir_out is not None:
            if self.img_aug:
                _img,_tgt=augment_image_pair(_img, _tgt, _level=1)
            _img,_tgt=scale_input(_img),scale_output(_tgt,self.dep_out)
        else:
            _img=scale_input(_img)
        return _img, _tgt
        # for i, ID in enumerate(view_coords):
        #     X[i,] = np.load('data/' + ID + '.npy')
        #     y[i] = self.labels[ID]
        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

