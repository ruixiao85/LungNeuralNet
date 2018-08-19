import copy
import math
import os
import random

import numpy as np
import keras
from cv2.cv2 import imread, resize, imwrite, INTER_AREA

from model_config import ModelConfig
from process_image import scale_input, scale_output, augment_image_pair

ALL_TARGET='All'

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


def read_resize_padding(_file, _resize, _padding):
    if _resize < 1.0:
        img = resize(imread(_file), (0, 0), fx=_resize, fy=_resize, interpolation=INTER_AREA)
        # print(" Resize [%.1f] applied "%_resize,end='')
    else:
        img = imread(_file)
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


class MetaInfo:
    def __init__(self, file, image, ori_row, ori_col, ri, ro, ci, co):
        self.file_name = file  # direct file can be a slice
        self.image_name = image # can be name of the whole image, can different from file_name (slice)
        self.ori_row = ori_row
        self.ori_col = ori_col
        self.row_start = ri
        self.row_end = ro
        self.col_start = ci
        self.col_end = co

    @classmethod
    def from_single(cls, file):
        ls = file.split("_#")
        ext = file.split(".")
        ss = file.split("#")
        if len(ls) == 2 and len(ss) == 8:  # slice
            return cls(file, "%s.%s" % (ls[0], ext[len(ext) - 1]), int(ss[1]), int(ss[2]), int(ss[3]), int(ss[4]), int(ss[5]), int(ss[6]))
        return cls(file, file, None, None, None, None, None, None)

    @classmethod
    def from_whole(cls, image_name, lg_row, lg_col, ri, ro, ci, co):
        obj=cls(None,image_name,lg_row,lg_col,ri,ro,ci,co)
        obj.file_name=obj.file_slice()
        return obj

    def update_coord(self, lg_row, lg_col, ri, ro, ci, co):
        self.ori_row, self.ori_col, self.row_start, self.row_end, self.col_start, self.col_end =\
            lg_row,lg_col,ri,ro,ci,co

    # def duplicate_parsed(self):
    #     ls=self.file_name.split("_#")
    #     ss=self.file_name.split("#")
    #     if len(ls)==2 and len(ss)==6:
    #         newcopy=copy.deepcopy(self)
    #         newcopy.file_name=ls[0]
    #         newcopy.row_start,newcopy.row_end,newcopy.col_start,newcopy.col_end=int(ss[1]), int(ss[2]), int(ss[3]), int(ss[4])
    #         return newcopy
    #     else:
    #         print("already or unable to parse %s" % self.file_name)
    #         return self

    def file_slice(self):
        return self.image_name.replace(".jpg", "_#%d#%d#%d#%d#%d#%d#.jpg"
                                          % (self.ori_row, self.ori_col, self.row_start,self.row_end,self.col_start,self.col_end))

    def get_image(self, _path, _separate, _resize, _padding):
        if _separate:
            return read_resize_padding(os.path.join(_path, self.file_name),_resize=1.0,_padding=1.0)
        return extract_pad_image(read_resize_padding(os.path.join(_path, self.image_name),_resize,_padding), self.row_start, self.row_end, self.col_start, self.col_end)

    def __str__(self):
        return self.file_name

    def __eq__(self, other):
        return str(self)==str(other)

    def __hash__(self):
        return str(self).__hash__()


class ImageSet:
    def __init__(self, cfg:ModelConfig, wd, sf, train, filter_type=None):
        self.work_directory=wd
        self.sub_folder=sf
        self.image_format=cfg.image_format
        self.resize=cfg.resize
        self.padding=cfg.padding
        self.images, self.total=[], None
        self.groups = []
        self.scan_image()
        self.row, self.col = None, None
        self.coverage=cfg.tr_coverage if train else cfg.prd_coverage
        self.train=train
        self.filter_type=filter_type
        self.view_coord=[]

    def scan_image(self):
        path = os.path.join(self.work_directory, self.sub_folder)
        self.images, self.total = self.find_file_recursive(path, self.image_format)
        for i in range(len(self.images)):
            self.images[i] = os.path.relpath(self.images[i], path)
            group=self.file_to_whole_image(self.images[i])
            if group not in self.groups:
                self.groups.append(group)

    @staticmethod
    def file_to_whole_image(text):
        half=text.split('_#')
        if len(half)==2:
            dot_sep=text.split('.')
            return "%s.%s"%(half[0],dot_sep[len(dot_sep)-1])
        return text

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
            _img = read_resize_padding(os.path.join(self.work_directory, self.sub_folder, image_name),self.resize,self.padding)
            print(image_name)
            entry=MetaInfo.from_single(image_name)
            if entry.row_start is None:
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
                entry.update_coord(lg_row, lg_col, ri, ro, ci, co)
            view_coord.append(entry)
        return view_coord

    def split_image_coord(self, ex_dir):
        view_coord=[]
        for image_name in self.images:
            _img = read_resize_padding(os.path.join(self.work_directory, self.sub_folder, image_name),self.resize,self.padding)
            lg_row, lg_col, lg_dep = _img.shape
            if self.train:
                r_len = max(1, 1+int(math.floor((lg_row - self.row) / self.row * self.coverage)))
                c_len = max(1, 1+int(math.floor((lg_col - self.col) / self.col * self.coverage)))
            else:
                r_len = max(1, 1+int(math.ceil((lg_row - self.row) / self.row * self.coverage)))
                c_len = max(1, 1+int(math.ceil((lg_col - self.col) / self.col * self.coverage)))
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
                    if self.filter_type is not None: # skip if low contrast or masked information
                        col=self.filter_type[0].lower()
                        if col=='g': # green mask
                            gmin=float(np.min(s_img[...,0]))+float(np.min(s_img[...,2])) # min_R min_B
                            if gmin>15.0:
                                print("skip tile r%d_c%d for no green mask (min_R+B=%.1f) for %s" % (r_index, c_index, gmin, image_name))
                                continue
                        else: # default white/black or rgb
                            std=float(np.std(s_img))
                            if std<15.0:
                                print("skip tile r%d_c%d for low contrast (std=%.1f) for %s" % (r_index, c_index, std, image_name))
                                continue
                    entry = MetaInfo.from_whole(image_name, lg_row, lg_col, ri, ro, ci, co)
                    imwrite(os.path.join(ex_dir, entry.file_name), s_img)
                    # entry.ri, entry.ro, entry.ci, entry.co = 0, self.row, 0, self.col
                    view_coord.append(entry)  # updated to target single exported file
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
        self.resize = cfg.resize
        self.padding = cfg.padding
        self.separate=cfg.separate
        self.valid_split=cfg.valid_split
        self.batch_size=cfg.batch_size
        self.max_train_step=cfg.max_train_step
        self.max_vali_step=cfg.max_vali_step
        self.img_aug = cfg.img_aug
        self.shuffle = cfg.shuffle
        self.mask_color = cfg.mask_color

        images =list(set(self.img_set.images).intersection(self.msk_set.images)) # match image file
        self.img_set.size_folder_update(images, self.row_in, self.col_in, self.dir_in_ex())
        self.msk_set.size_folder_update(images, self.row_out, self.col_out, self.dir_out_ex())

        self.view_coord=list(set(self.img_set.view_coord).intersection(self.msk_set.view_coord))

    def get_tr_val_generator(self):
        tr_list, val_list=[], [] # list view_coords, can be from slices
        tr_image, val_image=set(), set() # set whole images
        for vc in self.view_coord:
            if vc.image_name in tr_image:
                tr_list.append(vc)
                tr_image.add(vc.image_name)
            elif vc.image_name in val_image:
                val_list.append(vc)
                val_image.add(vc.image_name)
            else:
                if (len(val_list)+0.05)/(len(tr_list)+0.05)>self.valid_split:
                    tr_list.append(vc)
                    tr_image.add(vc.image_name)
                else:
                    val_list.append(vc)
                    val_image.add(vc.image_name)
        print("From %d split into train: %d views %d images; validation %d views %d images"%
              (len(self.view_coord),len(tr_list),len(tr_image),len(val_list),len(val_image)))
        print("Training Images:")
        print(tr_image)
        print("Validation Images:")
        print(val_image)
        return ImageTrainGenerator(self, tr_list), ImageTrainGenerator(self, val_list, aug=False)

    def dir_in_ex(self):
        return "%s-%s_%.1f_%dx%d" % (self.dir_in, ALL_TARGET, self.resize, self.row_in, self.col_in) if self.separate else self.dir_in

    def dir_out_ex(self):
        return "%s-%s_%.1f_%dx%d" % (self.dir_out, self.dir_in, self.resize, self.row_out, self.col_out) if self.separate else self.dir_out

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


class ImagePredictPair:
    def __init__(self, cfg: ModelConfig, ori_set: ImageSet, tgt=None):
        self.img_set = ori_set  # reference
        self.wd = ori_set.work_directory
        self.dir_in = ori_set.sub_folder
        self.dir_out = tgt if tgt is not None else ALL_TARGET
        self.resize = cfg.resize
        self.padding = cfg.padding
        self.row_in, self.col_in, self.dep_in = cfg.row_in, cfg.col_in, cfg.dep_in
        self.row_out, self.col_out, self.dep_out = cfg.row_out, cfg.col_out, cfg.dep_out
        self.separate = cfg.separate
        self.batch_size=cfg.batch_size

        self.img_set.size_folder_update(None, self.row_in, self.col_in, self.dir_in_ex())
        self.view_coord = self.img_set.view_coord

    def change_target(self, tgt):
        self.dir_out=tgt

    def get_prd_generator(self):
        return ImagePredictGenerator(self)

    def get_prd_generators(self):
        if self.separate:
            lpg={}
            for vc in self.view_coord:
                existing=lpg[vc.image_name]
                sublist=existing if existing is not None else []
                sublist.append(vc)
                lpg[vc.image_name]=sublist
            return [ImagePredictGenerator(self,l) for l in lpg.values()]
        else:
            return [ImagePredictGenerator(self)]

    def dir_in_ex(self):
        return "%s-%s_%.1f_%dx%d" % (self.dir_in, ALL_TARGET, self.resize, self.row_in, self.col_in) if self.separate else self.dir_in

    def dir_out_ex(self):
        return "%s-%s_%.1f_%dx%d" % (self.dir_out, self.dir_in, self.resize, self.row_out, self.col_out) if self.separate else self.dir_out

class ImageTrainGenerator(keras.utils.Sequence):
    def __init__(self, pair:ImageTrainPair, view_coord, aug=None):
        self.view_coord=view_coord
        self.indexes = np.arange(len(view_coord))
        self.wd=pair.wd
        self.dir_in, self.dir_out=pair.dir_in_ex(), pair.dir_out_ex()
        self.wd_dir_in = os.path.join(self.wd, self.dir_in)
        self.wd_dir_out = os.path.join(self.wd, self.dir_out)
        self.resize=pair.resize
        self.padding=pair.padding
        self.separate=pair.separate
        self.batch_size=pair.batch_size
        self.img_aug=pair.img_aug if aug is None else aug
        self.shuffle=pair.shuffle
        self.mask_color=pair.mask_color
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
            _img[i, ...] = vc.get_image(self.wd_dir_in,self.separate,self.resize,self.padding)
            _tgt[i, ...] = vc.get_image(self.wd_dir_out,self.separate,self.resize,self.padding)
        if self.img_aug:
            _img, _tgt = augment_image_pair(_img, _tgt, _level=random.randint(0, 4))  # integer N: a <= N <= b.
        return scale_input(_img), scale_output(_tgt, self.mask_color)

    def on_epoch_end(self):  # Updates indexes after each epoch
        self.indexes = np.arange(len(self.view_coord))
        if self.shuffle:
            np.random.shuffle(self.indexes)

class ImagePredictGenerator(keras.utils.Sequence):
    def __init__(self, pair:ImagePredictPair, view_coord=None):
        self.wd=pair.wd
        self.dir_in, self.dir_out=pair.dir_in_ex(), pair.dir_out_ex()
        self.wd_dir_in=os.path.join(self.wd, self.dir_in)
        self.resize = pair.resize
        self.padding = pair.padding
        self.batch_size=pair.batch_size
        self.row_in, self.col_in, self.dep_in=pair.row_in, pair.col_in, pair.dep_in
        # self.row_out, self.col_out, self.dep_out=pair.row_out, pair.col_out, pair.dep_out
        self.separate = pair.separate
        self.view_coord=view_coord if view_coord is not None else pair.view_coord
        self.indexes = np.arange(len(pair.view_coord))

    def __len__(self):  # Denotes the number of batches per epoch
        return int(np.floor(len(self.view_coord) / self.batch_size))

    def __getitem__(self, index):  # Generate one batch of data
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # print(" getting index %d with %d batch size"%(index,self.batch_size))
        _img = np.zeros((self.batch_size, self.row_in, self.col_in, self.dep_in), dtype=np.uint8)
        for i, vc in enumerate([self.view_coord[k] for k in indexes]):
            _img[i, ...] = vc.get_image(self.wd_dir_in,self.separate,self.resize,self.padding)
        return scale_input(_img),None

    def on_epoch_end(self):  # Updates indexes after each epoch
        self.indexes = np.arange(len(self.view_coord))

