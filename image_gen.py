import copy
import math
import os
import random

import numpy as np
import keras
from cv2.cv2 import imread, resize, imwrite

from model_config import ModelConfig
from process_image import scale_input, scale_output, augment_image_pair


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

    @staticmethod
    def parse_coord(file):
        seg = file.split("#")
        return int(seg[1]), int(seg[2]), int(seg[3]), int(seg[4])

    def file_slice(self):
        if self.r_index is not None:
            # return self.file_name.replace(".jpg", "_r%d_c%d.jpg" % (self.r_index,self.c_index))
            return self.file_name.replace(".jpg", "_#%d#%d#%d#%d#.jpg" % (self.row_start,self.row_end,self.col_start,self.col_end))
        return self.file_name

    def get_image(self, path):
        return extract_pad_image(imread(os.path.join(path, self.file_slice())), self.row_start, self.row_end, self.col_start, self.col_end)

    def __str__(self):
        return self.file_slice()

    def __eq__(self, other):
        return self.file_slice()==other.file_slice()
    def __hash__(self):
        return self.file_slice().__hash__()

class ImageSet:
    def __init__(self, cfg:ModelConfig, wd, sf, train):
        self.work_directory=wd
        self.sub_folder=sf
        self.image_format=cfg.image_format
        self.images, self.total=[], None
        self.scan_image()
        self.row, self.col = None, None
        self.coverage=cfg.tr_coverage if train else cfg.prd_coverage
        self.skip_low_contrast=train
        self.view_coord=[]

    def scan_image(self):
        path = os.path.join(self.work_directory, self.sub_folder)
        self.images, self.total = self.find_file_recursive(path, self.image_format)
        for i in range(len(self.images)):
            self.images[i] = os.path.relpath(self.images[i], path)

    @staticmethod
    def find_file_recursive(_path, _ext):
        from glob import glob
        _images = [path for fn in os.walk(_path) for path in glob(os.path.join(fn[0], _ext))]
        _total = len(_images)
        print("Found [%d] file from [%s]" % (_total, _path))
        return _images, _total

    def size_folder_update(self, images, row, col, new_dir):
        if images is not None:
            self.images=images  # update filtered images
        self.row, self.col=row, col
        if self.sub_folder!=new_dir:
            new_path=os.path.join(self.work_directory,new_dir)
            # shutil.rmtree(new_path)  # force delete
            if not os.path.exists(new_path): # change folder and not found
                os.makedirs(new_path)
                self.view_coord=self.split_image_coord(new_path)
            self.sub_folder=new_dir
            self.scan_image()
        self.view_coord=self.single_image_coord()

    def single_image_coord(self):
        view_coord=[]
        for image_name in self.images:
            _img = imread(os.path.join(self.work_directory, self.sub_folder, image_name))
            print(image_name)
            lg_row, lg_col, lg_dep=_img.shape
            ri, ro, ci, co=0, lg_row, 0, lg_col
            if self.row is not None or self.col is not None: # dimension specified
                ratio=0.5 # if self.predict_mode is True else random.random() # add randomness if not prediction/full
                rd=int(ratio*(lg_row-self.row))
                cd=int(ratio*(lg_col-self.col))
                ri+=rd
                ci+=cd
                ro-=lg_row-self.row-rd
                co-=lg_col-self.col-cd
            entry = MetaInfo(image_name, None, None, lg_row, lg_col, lg_dep, ri, ro, ci, co)
            view_coord.append(entry)
        return view_coord

    def split_image_coord(self, ex_dir):
        view_coord=[]
        for image_name in self.images:
            _img = imread(os.path.join(self.work_directory, self.sub_folder, image_name))
            lg_row, lg_col, lg_dep = _img.shape
            r_len = max(1, int(round(self.coverage * (lg_row - self.row) / self.row)))
            c_len = max(1, int(round(self.coverage * (lg_col - self.col) / self.col)))
            print("%s target %d x %d (coverage %.1f): original %d x %d ->  row /%d col /%d" %
                  (image_name, self.row, self.col, self.coverage, lg_row, lg_col, r_len, c_len))
            r0, c0, r_step, c_step = 0, 0, 0, 0  # start position and step default to (0,0)
            if r_len > 1:
                r_step = float(lg_row - self.row) / (r_len - 1)
            else:
                r0 = int(0.5 * (lg_row - self.row))
            if c_len > 1:
                c_step = float(lg_col - self.col) / (c_len - 1)
            else:
                c0 = int(0.5 * (lg_col - self.col))
            for r_index in range(r_len):
                for c_index in range(c_len):
                    ri = r0 + int(round(r_index * r_step))
                    ci = c0 + int(round(c_index * c_step))
                    ro = ri + self.row
                    co = ci + self.col
                    s_img = _img[ri:ro, ci:co, ...]
                    std=float(np.std(s_img))
                    if self.skip_low_contrast and std < 20:  # low coverage implies training mode, skip this tile if low contrast
                        print("skip tile r%d_c%d for low contrast (std=%.1f) for %s"%(r_index,c_index,std,image_name))
                        continue
                    entry = MetaInfo(image_name, r_index, c_index, lg_row, lg_col, lg_dep, ri, ro, ci, co) # temporary
                    imwrite(os.path.join(ex_dir, entry.file_slice()), s_img)
                    entry.file_name = entry.file_slice()
                    entry.r_index, entry.c_index = None, None
                    entry.ori_row, entry.ori_col, entry.ori_dep = self.row, self.col, lg_dep
                    entry.ri, entry.ro, entry.ci, entry.co = 0, self.row, 0, self.col
                    view_coord.append(entry) # updated to target single exported file
        return view_coord


class ImageTrainPair:
    def __init__(self, cfg:ModelConfig, ori_set, tgt_set):
        # self.img_set=ori_set
        self.img_set=copy.deepcopy(ori_set)
        # self.msk_set=tgt_set
        self.msk_set=copy.deepcopy(tgt_set)
        self.wd=self.img_set.work_directory
        self.dir_in=self.img_set.sub_folder
        self.dir_out=self.msk_set.sub_folder
        self.row_in, self.col_in, self.dep_in = cfg.row_in, cfg.col_in, cfg.dep_in
        self.row_out, self.col_out, self.dep_out = cfg.row_out, cfg.col_out, cfg.dep_out
        self.separate=cfg.separate
        self.valid_split=cfg.valid_split
        self.batch_size=cfg.batch_size
        self.img_aug = cfg.img_aug
        self.shuffle = cfg.shuffle

        images =list(set(self.img_set.images).intersection(self.msk_set.images)) # match image file
        self.img_set.size_folder_update(images, self.row_in, self.col_in, self.dir_in_ex())
        self.msk_set.size_folder_update(images, self.row_out, self.col_out, self.dir_out_ex())

        self.view_coord=list(set(self.img_set.view_coord).intersection(self.msk_set.view_coord))

    def get_tr_val_generator(self):
        tr_list, val_list=[], []
        ti,vi=0,0
        for i, vc in enumerate(self.view_coord):
            if 0.1*(i%10)<self.valid_split: # validation
                val_list.append(vc)
                vi+=1
            else: # training
                tr_list.append(vc)
                ti+=1
        print("From %d split into train : validation  %d : %d"%(len(self.view_coord),ti,vi))
        return ImageTrainGenerator(self, tr_list), ImageTrainGenerator(self, val_list)

    def dir_in_ex(self):
        return "%s-%s_%dx%d" % (self.dir_in, self.dir_out, self.row_in, self.col_in) if self.separate else self.dir_in

    def dir_out_ex(self):
        return "%s-%s_%dx%d" % (self.dir_out, self.dir_in, self.row_out, self.col_out) if self.separate else self.dir_out

    @staticmethod
    def filter_match_pair(set1: ImageSet, set2: ImageSet):
        shared_names = set(set1.images).intersection(set2.images)
        size_map = {}  # file:(row,col)
        if not set1.view_coord:
            set1.single_image_coord()
        for mi in set1.view_coord:
            if mi.file_name in shared_names:
                size_map[mi.file_name] = (mi.ori_row, mi.or_col)
        if not set2.view_coord:
            set2.single_image_coord()
        for mi in set2.view_coord:
            if mi.file_name in size_map.keys():
                if size_map[mi.file_name] != (mi.ori_row, mi.or_col):
                    print("size mismatch for %s" % mi.file_name)
                    shared_names.remove(mi.file_name)
        return list(shared_names)


ALL_TARGET='All'
class ImagePredictPair:
    def __init__(self, cfg: ModelConfig, ori_set: ImageSet, tgt=ALL_TARGET):
        self.img_set = ori_set  # reference
        self.wd = ori_set.work_directory
        self.dir_in = ori_set.sub_folder
        self.dir_out = tgt
        self.row_in, self.col_in, self.dep_in = cfg.row_in, cfg.col_in, cfg.dep_in
        self.row_out, self.col_out, self.dep_out = cfg.row_out, cfg.col_out, cfg.dep_out
        self.separate = cfg.separate
        self.batch_size=cfg.batch_size

        self.overlay_channel=cfg.overlay_channel
        self.overlay_opacity=cfg.overlay_opacity
        self.call_hardness=cfg.call_hardness

        self.img_set.size_folder_update(None, self.row_in, self.col_in, self.dir_in_ex())
        self.view_coord = self.img_set.view_coord

    def change_target(self, tgt):
        self.dir_out=tgt

    def get_prd_generator(self):
        return ImagePredictGenerator(self)

    def dir_in_ex(self):
        return "%s-%s_%dx%d" % (self.dir_in, ALL_TARGET, self.row_in, self.col_in) if self.separate else self.dir_in

    def dir_out_ex(self):
        return "%s-%s_%dx%d" % (self.dir_out, self.dir_in, self.row_out, self.col_out) if self.separate else self.dir_out

class ImageTrainGenerator(keras.utils.Sequence):
    def __init__(self, pair:ImageTrainPair, view_coord):
        self.view_coord=view_coord
        self.indexes = np.arange(len(view_coord))
        self.wd=pair.wd
        self.dir_in, self.dir_out=pair.dir_in_ex(), pair.dir_out_ex()
        self.batch_size=pair.batch_size
        self.img_aug=pair.img_aug
        self.shuffle=pair.shuffle
        self.row_in, self.col_in, self.dep_in=pair.row_in, pair.col_in, pair.dep_in
        self.row_out, self.col_out, self.dep_out=pair.row_out, pair.col_out, pair.dep_out

    def __len__(self):  # Denotes the number of batches per epoch
        return int(np.floor(len(self.view_coord) / self.batch_size))

    def __getitem__(self, index):  # Generate one batch of data
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # print(" getting index %d with %d batch size"%(index,self.batch_size))
        _img = np.zeros((self.batch_size, self.row_in, self.col_in, self.dep_in), dtype=np.uint8)
        _tgt = np.zeros((self.batch_size, self.row_out, self.col_out, 3), dtype=np.uint8)
        for i, vc in enumerate([self.view_coord[k] for k in indexes]):
            _img[i, ...] = vc.get_image(os.path.join(self.wd, self.dir_in))
            _tgt[i, ...] = vc.get_image(os.path.join(self.wd, self.dir_out))
        if self.img_aug:
            _img, _tgt = augment_image_pair(_img, _tgt, _level=1)
        return scale_input(_img), scale_output(_tgt, self.dep_out)

    def on_epoch_end(self):  # Updates indexes after each epoch
        self.indexes = np.arange(len(self.view_coord))
        if self.shuffle:
            np.random.shuffle(self.indexes)

class ImagePredictGenerator(keras.utils.Sequence):
    def __init__(self, pair:ImagePredictPair):
        self.view_coord=pair.view_coord
        self.indexes = np.arange(len(pair.view_coord))
        self.wd=pair.wd
        self.dir_in, self.dir_out=pair.dir_in_ex(), pair.dir_out_ex()
        self.batch_size=pair.batch_size
        self.row_in, self.col_in, self.dep_in=pair.row_in, pair.col_in, pair.dep_in
        # self.row_out, self.col_out, self.dep_out=pair.row_out, pair.col_out, pair.dep_out

    def __len__(self):  # Denotes the number of batches per epoch
        return int(np.floor(len(self.view_coord) / self.batch_size))

    def __getitem__(self, index):  # Generate one batch of data
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # print(" getting index %d with %d batch size"%(index,self.batch_size))
        _img = np.zeros((self.batch_size, self.row_in, self.col_in, self.dep_in), dtype=np.uint8)
        for i, vc in enumerate([self.view_coord[k] for k in indexes]):
            _img[i, ...] = vc.get_image(os.path.join(self.wd, self.dir_in))
        return scale_input(_img),None

    def on_epoch_end(self):  # Updates indexes after each epoch
        self.indexes = np.arange(len(self.view_coord))

