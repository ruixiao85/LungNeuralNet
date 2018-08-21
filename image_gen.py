import copy
import math
import os
import random

import numpy as np
import keras
from cv2.cv2 import imread, resize, imwrite, INTER_AREA

from model_config import ModelConfig
from process_image import scale_input, scale_output, augment_image_pair, read_resize_padding, extract_pad_image

ALL_TARGET = 'All'
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

    def file_slice(self):
        return self.image_name.replace(".jpg", "_#%d#%d#%d#%d#%d#%d#.jpg"
                 % (self.ori_row, self.ori_col, self.row_start,self.row_end,self.col_start,self.col_end))

    def get_image(self, _path, _separate, _resize, _padding):
        return read_resize_padding(os.path.join(_path, self.file_name),_resize=1.0,_padding=1.0) if _separate else\
            extract_pad_image(read_resize_padding(os.path.join(_path, self.image_name),_resize,_padding), self.row_start, self.row_end, self.col_start, self.col_end)

    def get_mask(self, _path, _separate, _resize, _padding, _mask_col):
        img=self.get_image(_path,_separate,_resize,_padding)
        # imwrite("img.jpg",img)
        code=_mask_col[0].lower()
        if code!='g':  # default to white/black from blue channel
            return img[...,2]  # blue channel to only channel
        else: # green
            img=img.astype(np.int16)
            # imwrite("testd_2f_-0.3.jpg",np.clip(5*(img[..., 1] - img[..., 0]-100), 0, 255)[..., np.newaxis])
            # imwrite("testd_2f_-0.4.jpg",np.clip(5*(img[..., 1] - img[..., 0]-120), 0, 255)[..., np.newaxis])
            # imwrite("testd_2f_-0.5.jpg",np.clip(5*(img[..., 1] - img[..., 0]-140), 0, 255)[..., np.newaxis])
            # imwrite("testd_2f_-0.6.jpg",np.clip(5*(img[..., 1] - img[..., 0]-160), 0, 255)[..., np.newaxis])
            return np.clip(5*(img[..., 1] - img[..., 0]-110), 0, 255).astype(np.uint8)


    def __str__(self):
        return self.file_name

    def __eq__(self, other):
        return str(self)==str(other)

    def __hash__(self):
        return str(self).__hash__()


class ImageSet:
    def __init__(self, cfg:ModelConfig, wd, sf, is_train, is_image):
        self.cfg=cfg
        self.work_directory=wd
        self.sub_folder=sf
        self.images, self.total=[], None
        self.groups = []
        self.scan_image()
        self.is_train=is_train
        self.coverage=cfg.coverage_train if is_train else cfg.coverage_predict
        self.is_image=is_image
        self.row=self.cfg.row_in if is_image else self.cfg.row_out
        self.col=self.cfg.col_in if is_image else self.cfg.col_out
        self.view_coord=[]

    def scan_image(self):
        path = os.path.join(self.work_directory, self.sub_folder)
        self.images, self.total = self.find_file_recursive(path, self.cfg.image_format)
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

    @staticmethod
    def ext_folder(cfg, is_image):
        if cfg.separate:
            return "_%.1f_%dx%d" % (cfg.image_resize, cfg.row_in, cfg.col_in)\
                if is_image else "_%.1f_%dx%d" % (cfg.image_resize, cfg.row_out, cfg.col_out)
        else:
            return None

    def size_folder_update(self):
        ext=self.ext_folder(self.cfg, self.is_image)
        if ext is not None:
            new_dir=self.sub_folder+ext
            new_path=os.path.join(self.work_directory, new_dir)
            # shutil.rmtree(new_path)  # force delete
            if not os.path.exists(new_path): # change folder and not found
                os.makedirs(new_path)
                self.view_coord=self.split_image_coord(new_path)
            self.sub_folder=new_dir
            self.scan_image()
        self.view_coord=self.single_image_coord()
        return self

    def single_image_coord(self):
        view_coord=[]
        for image_name in self.images:
            _img = read_resize_padding(os.path.join(self.work_directory, self.sub_folder, image_name),self.cfg.image_resize,self.cfg.image_padding)
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
            _img = read_resize_padding(os.path.join(self.work_directory, self.sub_folder, image_name),self.cfg.image_resize,self.cfg.image_padding)
            lg_row, lg_col, lg_dep = _img.shape
            if self.is_train:
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
                    s_img = extract_pad_image(_img, ri, ro, ci, co)
                    if self.is_train: # skip if low contrast or masked information
                        # col=self.filter_type[0].lower()
                        # if col=='g': # green mask
                        #     gmin=float(np.min(s_img[...,0]))+float(np.min(s_img[...,2])) # min_R min_B
                        #     if gmin>15.0:
                        #         print("skip tile r%d_c%d for no green mask (min_R+B=%.1f) for %s" % (r_index, c_index, gmin, image_name))
                        #         continue
                        # else: # default white/black or rgb
                        std=float(np.std(s_img))
                        if std<15.0:
                            print("skip tile r%d_c%d for low contrast (std=%.1f) for %s" % (r_index, c_index, std, image_name))
                            continue
                    entry = MetaInfo.from_whole(image_name, lg_row, lg_col, ri, ro, ci, co)
                    imwrite(os.path.join(ex_dir, entry.file_name), s_img)
                    # entry.ri, entry.ro, entry.ci, entry.co = 0, self.row, 0, self.col
                    view_coord.append(entry)  # updated to target single exported file
        return view_coord

class ImagePairMulti:
    def __init__(self, cfg: ModelConfig, wd, origin, targets, is_train):
        self.cfg=cfg
        self.wd = wd
        self.origin, self.dir_in = origin, origin
        self.targets, self.dir_out = None, None
        self.change_target(targets)
        self.img_set=ImageSet(cfg, wd, origin, is_train, is_image=True).size_folder_update()
        self.view_coord=self.img_set.view_coord
        self.is_train = is_train
        if self.is_train:
            views=set(self.view_coord)
            self.msk_set=[]
            for t in targets:
                msk=ImageSet(cfg, wd, t, is_train, is_image=False).size_folder_update()
                self.msk_set.append(msk)
                views=views.intersection(msk.view_coord)
            self.view_coord=list(views)
        else:
            self.msk_set=targets

    def change_target(self, targets):
        self.targets = targets
        self.dir_out = targets[0] if len(targets)==1 else ','.join(targets) # [t[:4] for t in targets])

    @property
    def dir_in_ex(self, _dir=None):
        ori_dir=_dir or self.dir_in
        ext=ImageSet.ext_folder(self.cfg, True)
        return ori_dir if ext is None else ori_dir+ext

    @property
    def dir_out_ex(self, _dir=None):
        ori_dir = _dir or self.dir_out
        ext = ImageSet.ext_folder(self.cfg, False)
        return ori_dir if ext is None else ori_dir + ext

    def get_prd_generator(self):
        return ImageGeneratorMulti(self, aug=False)

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
                if (len(val_list)+0.05)/(len(tr_list)+0.05)>self.cfg.train_valid_split:
                    tr_list.append(vc)
                    tr_image.add(vc.image_name)
                else:
                    val_list.append(vc)
                    val_image.add(vc.image_name)
        print("From %d split into train: %d views %d images; validation %d views %d images"%
              (len(self.view_coord),len(tr_list),len(tr_image),len(val_list),len(val_image)))
        print("Training Images:"); print(tr_image)
        print("Validation Images:"); print(val_image)
        return ImageGeneratorMulti(self, self.cfg.train_aug, tr_list), ImageGeneratorMulti(self, False, val_list)


class ImageGeneratorMulti(keras.utils.Sequence):
    def __init__(self, pair:ImagePairMulti, aug, view_coord=None):
        self.pair=pair
        self.cfg=pair.cfg
        self.img_aug=aug
        self.view_coord=pair.view_coord if view_coord is None else view_coord
        self.indexes = np.arange(len(self.view_coord))
        # self.img_set=pair.img_set
        # self.train = pair.train
        # self.wd=pair.wd
        # self.org_name=pair.dir_in
        # self.tgt_names=pair.tgt_names
        # self.dir_in_ex, self.dir_out_ex= pair.dir_in_ex(), pair.dir_out_ex()
        # self.dir_out=pair.dir_out
        # self.wd_dir_in = os.path.join(self.wd, self.dir_in)
        # self.wd_dir_out = os.path.join(self.wd, self.dir_out)
        # self.resize=pair.resize
        # self.padding=pair.padding
        # self.separate=pair.separate
        # self.batch_size=pair.batch_size
        # self.shuffle=pair.shuffle
        # self.mask_color=pair.mask_color
        # self.row_in, self.col_in, self.dep_in= pair.row_in, pair.col_in, pair.dep_in
        # self.row_out, self.col_out, self.dep_out= pair.row_out, pair.col_out, pair.dep_out

    def __len__(self):  # Denotes the number of batches per epoch
        return int(np.floor(len(self.view_coord) / self.cfg.batch_size))

    def __getitem__(self, index):  # Generate one batch of data
        indexes = self.indexes[index * self.cfg.batch_size:(index + 1) * self.cfg.batch_size]
        # print(" getting index %d with %d batch size"%(index,self.batch_size))
        if self.pair.is_train:
            _img = np.zeros((self.cfg.batch_size, self.cfg.row_in, self.cfg.col_in, self.cfg.dep_in), dtype=np.uint8)
            _tgt = np.zeros((self.cfg.batch_size, self.cfg.row_out, self.cfg.col_out, self.cfg.dep_out), dtype=np.uint8)
            for vi, vc in enumerate([self.view_coord[k] for k in indexes]):
                _img[vi, ...] = vc.get_image(os.path.join(self.pair.wd, self.pair.origin + self.pair.dir_in_ex), self.cfg.separate, self.cfg.image_resize, self.cfg.image_padding)
                for ti,tgt in enumerate(self.pair.targets):
                    _tgt[vi, ...,ti] = vc.get_mask(os.path.join(self.pair.wd, tgt + self.pair.dir_out_ex), self.cfg.separate, self.cfg.image_resize, self.cfg.image_padding, self.cfg.mask_color)
            if self.img_aug:
                _img, _tgt = augment_image_pair(_img, _tgt, _level=random.randint(0, 4))  # integer N: a <= N <= b.
            return scale_input(_img), scale_output(_tgt)
        else:
            _img = np.zeros((self.cfg.batch_size, self.cfg.row_in, self.cfg.col_in, self.cfg.dep_in), dtype=np.uint8)
            for vi, vc in enumerate([self.view_coord[k] for k in indexes]):
                _img[vi, ...] = vc.get_image(os.path.join(self.pair.wd, self.pair.origin+self.pair.dir_in_ex), self.cfg.separate, self.cfg.image_resize, self.cfg.image_padding)
            return scale_input(_img), None

    def on_epoch_end(self):  # Updates indexes after each epoch
        self.indexes = np.arange(len(self.view_coord))
        if self.pair.is_train and self.cfg.train_shuffle:
            np.random.shuffle(self.indexes)
