import os
from cv2 import cv2
import numpy as np

from a_config import Config
from osio import find_file_recursive
from preprocess import read_resize_padding,extract_pad_image

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

    def file_name_insert(self, cfg:Config, text=None):
        ext=cfg.image_format[1:]
        return self.file_name.replace(ext,'') if text is None else self.file_name.replace(ext,text+ext)

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

    def get_image(self, _path, cfg:Config, override=None):
        file=override or self.file_name
        return read_resize_padding(os.path.join(_path, file),_resize=1.0,_padding=1.0) if cfg.separate else\
            extract_pad_image(read_resize_padding(os.path.join(_path, file),cfg.image_resize,cfg.image_padding), self.row_start, self.row_end, self.col_start, self.col_end)

    def get_mask(self, _path, cfg:Config, override=None):
        img=self.get_image(_path, cfg, override)
        # imwrite("test_img.jpg",img)
        code=cfg.mask_color[0].lower()
        if code=='g': # green
            img=img.astype(np.int16)
            # imwrite("test_2f_-0.3.jpg",np.clip(5*(img[..., 1] - img[..., 0]-100), 0, 255)[..., np.newaxis])
            # imwrite("test_2f_-0.4.jpg",np.clip(5*(img[..., 1] - img[..., 0]-120), 0, 255)[..., np.newaxis])
            # imwrite("test_2f_-0.5.jpg",np.clip(5*(img[..., 1] - img[..., 0]-140), 0, 255)[..., np.newaxis])
            # imwrite("test_2f_-0.6.jpg",np.clip(5*(img[..., 1] - img[..., 0]-160), 0, 255)[..., np.newaxis])
            return np.clip(5*(img[..., 1] - img[..., 0]-110), 0, 255).astype(np.uint8)
        else: # default to white/black
            # return img[...,1]  # from green channel
            return np.max(img,axis=-1,keepdims=False)  # max channel
            # return np.clip(np.sum(img,axis=-1,keepdims=False), 0, 255).astype(np.uint8)  # sum channel

    def get_masks(self, _path, cfg:Config):
        import glob
        files=glob.glob(os.path.join(_path,self.file_name.replace(cfg.image_format[1:],cfg.image_format)))
        # print('found %d files matching %s'%(len(files),self.file_name))
        msks,clss=None,[]
        for f in files:
            class_split=f.split('^')
            clss.append(class_split[int(len(class_split)-2)])
            msk=self.get_mask(_path, cfg, f)[...,np.newaxis] # np.uint8 0-255
            msks=msk if msks is None else np.concatenate((msks,msk),axis=-1)
        return msks, np.array(clss,dtype=np.uint8) # 0 ~ 255

    def __str__(self):
        return self.file_name

    def __eq__(self, other):
        return str(self)==str(other)

    def __hash__(self):
        return str(self).__hash__()

class ImageSet:
    def __init__(self,cfg:Config,wd,sf,is_train,is_image):
        self.cfg=cfg
        self.work_directory=wd
        self.sub_folder=sf
        self.images=None
        self.is_train=is_train
        self.coverage=cfg.coverage_train if is_train else cfg.coverage_predict
        self.is_image=is_image
        self.row=self.cfg.row_in if is_image else self.cfg.row_out
        self.col=self.cfg.col_in if is_image else self.cfg.col_out
        self.view_coord=None  # list

    def find_file_recursive_rel(self):
        path=os.path.join(self.work_directory,self.sub_folder)
        self.images=[os.path.relpath(absp,path) for absp in find_file_recursive(path, self.cfg.image_format)]

    def size_folder_update(self):
        self.find_file_recursive_rel()
        if self.cfg.separate:
            new_dir="%s_%s"%(self.sub_folder,self.ext_folder(self.cfg,self.is_image))
            new_path=os.path.join(self.work_directory,new_dir)
            # shutil.rmtree(new_path)  # force delete
            if not os.path.exists(new_path):  # change folder and not found
                os.makedirs(new_path)
                self.split_image_coord(new_path)
            self.sub_folder=new_dir
            self.find_file_recursive_rel()
        self.single_image_coord()
        return self

    def single_image_coord(self):
        self.view_coord=[]
        ratio=0.5  # if self.predict_mode is True else random.random() # add randomness if not prediction/full
        total=len(self.images)
        print('Parsing %d files...'%total)
        for i,image_name in enumerate(self.images):
            _img=read_resize_padding(os.path.join(self.work_directory,self.sub_folder,image_name),self.cfg.image_resize,self.cfg.image_padding)
            pct10=10*(i+1)//total
            if pct10 > 10*i//total:
                print('%.0f%% ... %s'%(pct10*10,image_name))
            entry=MetaInfo.from_single(image_name)
            if entry.row_start is None:
                lg_row,lg_col,lg_dep=_img.shape
                ri,ro,ci,co=0,lg_row,0,lg_col
                if self.row is not None or self.col is not None:  # dimension specified
                    rd=int(ratio*(lg_row-self.row))
                    cd=int(ratio*(lg_col-self.col))
                    ri+=rd; ci+=cd
                    ro-=lg_row-self.row-rd
                    co-=lg_col-self.col-cd
                entry.update_coord(lg_row,lg_col,ri,ro,ci,co)
            self.view_coord.append(entry)

    def split_image_coord(self,ex_dir):
        self.view_coord=[]
        for image_name in self.images:
            _img=read_resize_padding(os.path.join(self.work_directory,self.sub_folder,image_name),self.cfg.image_resize,self.cfg.image_padding)
            lg_row,lg_col,lg_dep=_img.shape
            r_len=max(1,1+int(round((lg_row-self.row)/self.row*self.coverage)))
            c_len=max(1,1+int(round((lg_col-self.col)/self.col*self.coverage)))
            print("%s target %d x %d (coverage %.1f): original %d x %d ->  row /%d col /%d"%
                  (image_name,self.row,self.col,self.coverage,lg_row,lg_col,r_len,c_len))
            r0,c0,r_step,c_step=0,0,0,0  # start position and step default to (0,0)
            if r_len>1:
                r_step=float(lg_row-self.row)/(r_len-1)
            else:
                r0=int(0.5*(lg_row-self.row))
            if c_len>1:
                c_step=float(lg_col-self.col)/(c_len-1)
            else:
                c0=int(0.5*(lg_col-self.col))
            for r_index in range(r_len):
                for c_index in range(c_len):
                    ri=r0+int(round(r_index*r_step))
                    ci=c0+int(round(c_index*c_step))
                    ro=ri+self.row
                    co=ci+self.col
                    s_img=extract_pad_image(_img,ri,ro,ci,co)
                    if self.is_train:  # skip if low contrast or masked information
                        # col=self.filter_type[0].lower()
                        # if col=='g': # green mask
                        #     gmin=float(np.min(s_img[...,0]))+float(np.min(s_img[...,2])) # min_R min_B
                        #     if gmin>15.0:
                        #         print("skip tile r%d_c%d for no green mask (min_R+B=%.1f) for %s" % (r_index, c_index, gmin, image_name))
                        #         continue
                        # else: # default white/black or rgb
                        std=float(np.std(s_img))
                        if not self.is_image and self.cfg.mask_color[0].lower()!='g' and std<3.0:  # none green mask have a lower std requirement
                            print("skip tile r%d_c%d for low contrast (std=%.1f) for %s"%(r_index,c_index,std,image_name))
                            continue
                        elif std<10.0:
                            print("skip tile r%d_c%d for low contrast (std=%.1f) for %s"%(r_index,c_index,std,image_name))
                            continue
                    entry=MetaInfo.from_whole(image_name,lg_row,lg_col,ri,ro,ci,co)
                    cv2.imwrite(os.path.join(ex_dir,entry.file_name),s_img,
                            [int(cv2.IMWRITE_JPEG_QUALITY),100] if self.is_image else [int(cv2.IMWRITE_JPEG_QUALITY),80])
                    # entry.ri, entry.ro, entry.ci, entry.co = 0, self.row, 0, self.col
                    self.view_coord.append(entry)  # updated to target single exported file

    def view_coord_batch(self):
        view_batch={}
        if self.cfg.separate:
            for view in self.view_coord:
                key=self.file_to_whole_image(view.file_name)
                sub_list=view_batch.get(key) or []
                sub_list.append(view)
                view_batch[key]=sub_list
            return view_batch
        else:
            n = self.cfg.train_step
            return { x:self.view_coord[x:x + n] for x in range(0, len(self.view_coord), n)}  # break into sub-lists

    @staticmethod
    def file_to_whole_image(text):
        half=text.split('_#')
        if len(half)==2:
            dot_sep=text.split('.')
            return "%s.%s"%(half[0],dot_sep[len(dot_sep)-1])
        return text

    @staticmethod
    def ext_folder(cfg, is_image):
        return "%.1f_%dx%d" % (cfg.image_resize, cfg.row_in, cfg.col_in)\
            if is_image else "%.1f_%dx%d" % (cfg.image_resize, cfg.row_out, cfg.col_out)

class PatchSet(ImageSet):
    def __init__(self,cfg:Config,wd,sf,is_train,is_image):

        self.patches=None
        super(PatchSet,self).__init__(cfg,wd,sf,is_train,is_image)
        self.size_folder_update()
        self.num_patches=len(self.images)
        # view_coord ori_row=45 ori_col=37 start~end are overrange

        # self.cfg=cfg
        # self.work_directory=wd
        # self.sub_folder=sf
        # self.images=None
        # self.is_train=is_train
        # self.coverage=cfg.coverage_train if is_train else cfg.coverage_predict
        # self.is_image=is_image
        # self.row=self.cfg.row_in if is_image else self.cfg.row_out
        # self.col=self.cfg.col_in if is_image else self.cfg.col_out
        # self.view_coord=None  # list

    def size_folder_update(self):
        print("checking %s"%self.sub_folder)
        self.find_file_recursive_rel()
        # if self.cfg.separate:
        #     new_dir="%s_%s" % (self.sub_folder, self.ext_folder(self.cfg, self.is_image))
        #     new_path=os.path.join(self.work_directory, new_dir)
            # shutil.rmtree(new_path)  # force delete
            # if not os.path.exists(new_path): # change folder and not found
            #     os.makedirs(new_path)
            #     self.split_image_coord(new_path)
            # self.sub_folder=new_dir
            # self.find_file_recursive_rel()
        self.single_image_coord()
        return self

    def single_image_coord(self):
        self.view_coord=[]
        self.patches=[]
        for image_name in self.images:
            _img = read_resize_padding(os.path.join(self.work_directory, self.sub_folder, image_name),
                                       _resize=self.cfg.image_resize,_padding=self.cfg.image_padding)
            minVal,maxVal,minLoc,maxLoc=cv2.minMaxLoc(np.max(_img,axis=-1))
            # print(','.join([str(minVal),str(maxVal),str(minLoc),str(maxLoc)]))
            _img=_img+(255-maxVal)
            # cv2.imwrite('this.jpg',_img)
            self.patches.append(_img)
            entry=MetaInfo.from_single(image_name)
            if entry.row_start is None:
                lg_row, lg_col, lg_dep=_img.shape
                ri, ro, ci, co=int(np.average(_img)), int(np.std(_img)), np.max(_img), np.min(_img)
                entry.update_coord(lg_row, lg_col, ri, ro, ci, co) # store ave, std, max, min of patch
                print('%s ave:%f std:%f max:%f min:%f'%(image_name,ri,ro,ci,co))
            self.view_coord.append(entry)


    def split_image_coord(self, ex_dir):
        self.view_coord=[]
        for image_name in self.images:
            _img = read_resize_padding(os.path.join(self.work_directory, self.sub_folder, image_name),self.cfg.image_resize,self.cfg.image_padding)
            lg_row, lg_col, lg_dep = _img.shape
            r_len = max(1, 1+int(round((lg_row - self.row) / self.row * self.coverage)))
            c_len = max(1, 1+int(round((lg_col - self.col) / self.col * self.coverage)))
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
                        if not self.is_image and self.cfg.mask_color[0].lower()!='g' and std<3.0: # none green mask have a lower std requirement
                            print("skip tile r%d_c%d for low contrast (std=%.1f) for %s" % (r_index, c_index, std, image_name))
                            continue
                        elif std<10.0:
                            print("skip tile r%d_c%d for low contrast (std=%.1f) for %s" % (r_index, c_index, std, image_name))
                            continue
                    entry = MetaInfo.from_whole(image_name, lg_row, lg_col, ri, ro, ci, co)
                    cv2.imwrite(os.path.join(ex_dir, entry.file_name), s_img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100] if self.is_image else [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    # entry.ri, entry.ro, entry.ci, entry.co = 0, self.row, 0, self.col
                    self.view_coord.append(entry)  # updated to target single exported file

