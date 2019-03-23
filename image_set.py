import itertools
import os
from cv2 import cv2
import numpy as np

from a_config import Config
from osio import mkdir_ifexist,find_file_pattern,find_file_pattern_rel,find_folder_prefix,find_file_ext_recursive,find_file_ext_recursive_rel
from preprocess import read_image,read_resize,read_resize_pad,read_resize_fit,extract_pad_image

def parse_float(text):
    try:
        return float(text)
    except ValueError:
        return None

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

    # def file_name_insert(self, cfg:Config, text=None):
    #     ext=cfg.image_format[1:]
    #     return self.file_name.replace(ext,'') if text is None else self.file_name.replace(ext,text+ext)

    # @classmethod
    # def from_single(cls, file):
    #     ls = file.split("_#")
    #     ext = file.split(".")
    #     ss = file.split("#")
    #     if len(ls) == 2 and len(ss) == 8:  # slice
    #         return cls(file, "%s.%s" % (ls[0], ext[len(ext) - 1]), int(ss[1]), int(ss[2]), int(ss[3]), int(ss[4]), int(ss[5]), int(ss[6]), None)
    #     return cls(file, file, None, None, None, None, None, None, None)

     # def update_coord(self, lg_row, lg_col, ri, ro, ci, co):
     #    self.ori_row, self.ori_col, self.row_start, self.row_end, self.col_start, self.col_end =\
     #        lg_row,lg_col,ri,ro,ci,co

       # def get_masks(self, _path, cfg:Config):
    #     import glob
    #     import random
    #     files=glob.glob(os.path.join(_path,self.file_name.replace(cfg.image_format[1:],cfg.image_format)))
    #     random.shuffle(files) # avaible same order of files
    #     print(' found %d files matching %s'%(len(files),self.file_name))
        # msks,clss=None,[]
        # for f in files:
        #     class_split=f.split('^')
        #     clss.append(class_split[int(len(class_split)-2)])
        #     msk=self.get_mask(_path, f)[...,np.newaxis] # np.uint8 0-255
        #     msks=msk if msks is None else np.concatenate((msks,msk),axis=-1)
        # return msks, np.array(clss,dtype=np.uint8) # 0 ~ 255

    def __str__(self):
        return self.file_name

    def __eq__(self, other):
        return str(self)==str(other)

    def __hash__(self):
        return str(self).__hash__()

class ImageSet:
    def __init__(self,cfg:Config,wd,sf,is_train,channels):
        self.work_directory=wd
        self.sub_category=sf
        self.is_train=is_train
        self.channels=channels
        self.image_format=cfg.image_format
        self.target_scale=cfg.target_scale
        self.train_vali_split=cfg.train_vali_split
        self.target_folder=self.label_scale()
        self.raw_folder,self.raw_scale,self.resize_ratio=None,None,None
        self.images=None # list names
        self.image_data=None # dict RGB data
        self.tr_list,self.val_list=None,None

    def label_scale(self,target=None,scale=None):
        return "%s_%.1f"%(target or self.sub_category, scale or self.target_scale)

    def prep_folder(self):
        self.prescreen_folders()
        self.split_tr_val()
        return self

    def prescreen_folders(self):
        initial_folders=find_folder_prefix(self.work_directory,self.sub_category+'_')
        folders=initial_folders.copy()
        for folder in initial_folders:
            print(' ',folder,sep='',end=' ')
            if folder==self.target_folder:
                print('+ input images.')
            else:
                sections=folder.split('_')
                if len(sections)!=2 or parse_float(sections[1]) is None:
                    folders.remove(folder)
                    print('- omitted.')
                else:
                    print('# potential raw images.')
        self.raw_folder=sorted(folders,key=lambda t:float(t.split('_')[1]),reverse=True)[0]  # high-res first
        self.raw_scale=float(self.raw_folder.split('_')[1])
        self.resize_ratio=round(self.target_scale/self.raw_scale,2)
        input_folder=self.target_folder if self.resize_ratio==1 else self.raw_folder
        self.images=find_file_ext_recursive_rel(os.path.join(self.work_directory,input_folder),self.image_format)
        total=len(self.images)
        self.image_data={}
        print("Processing %d images from folder [%s] with resize_ratio of %.1fx ..."%(total,input_folder,self.resize_ratio))
        if self.raw_scale<self.target_scale:
            print("Warning, upsampling from low-res raw images is not recommended!")
        for i,image in enumerate(self.images):
            _img=self.adapt_channel(read_resize(os.path.join(self.work_directory,input_folder,image),self.resize_ratio))
            pct10=10*(i+1)//total
            if pct10>10*i//total:
                print(' %.0f%% ... %s'%(pct10*10,image))
            # lg_row,lg_col,lg_dep=_img.shape
            self.image_data[image]=_img

    def split_tr_val(self):
        self.tr_list,self.val_list=[],[]
        for img in self.images:
            if (len(self.val_list)+0.05)/(len(self.tr_list)+0.05)>self.train_vali_split:
                self.tr_list.append(img)
            else:
                self.val_list.append(img)
        print("[%s] was split into training [%d] and validation [%d] set."%(self.sub_category,len(self.tr_list),len(self.val_list)))

    def adapt_channel(self,img,channels=None):
        return np.max(img,axis=-1,keepdims=True) if (channels or self.channels)==1 else img

    def get_image(self,view):
        return self.image_data[view.image_name][view.row_start:view.row_end,view.col_start:view.col_end,...]

class ViewSet(ImageSet):
    def __init__(self,cfg: Config,wd,sf,is_train,channels,low_std_ex):
        super(ViewSet,self).__init__(cfg,wd,sf,is_train,channels)
        self.coverage=cfg.coverage_train if self.is_train else cfg.coverage_predict
        self.train_step=cfg.train_step
        self.row,self.col=cfg.row_in,cfg.col_in
        self.low_std_ex=low_std_ex
        self.tr_view,self.val_view=None,None  # lists -> views with specified size
        self.tr_view_ex,self.val_view_ex=None,None  # views with low contrast

    def label_scale_res(self,target=None,scale=None,rows=None,cols=None):
        return "%s_%dx%d"%(self.label_scale(target,scale), rows or self.row, cols or self.col)
    def scale_res(self,scale=None,rows=None,cols=None):
        return "%s_%dx%d"%(scale or self.target_scale, rows or self.row, cols or self.col)

    def prep_folder(self):
        self.prescreen_folders()
        self.split_tr_val() # tr/val-split only needed for img_set
        self.tr_view=self.list_to_view(self.tr_list)
        self.val_view=self.list_to_view(self.val_list)
        if self.low_std_ex:
            self.tr_view_ex=self.low_std_exclusion(self.tr_view)
            self.val_view_ex=self.low_std_exclusion(self.val_view)
        return self

    def list_to_view(self,img_list):
        dotext=self.image_format[1:]
        view_list=[]
        for img in img_list:
            _img=self.image_data[img]
            lg_row,lg_col,lg_dep=_img.shape
            r_len=max(1,1+int(round((lg_row-self.row)/self.row*self.coverage)))
            c_len=max(1,1+int(round((lg_col-self.col)/self.col*self.coverage)))
            print(" %s target %d x %d (coverage %.1f): original %d x %d ->  row /%d col /%d"%(img,self.row,self.col,self.coverage,lg_row,lg_col,r_len,c_len))
            r0,c0,r_step,c_step=0,0,0,0  # start position and step default to (0,0)
            r_step=float(lg_row-self.row)/(r_len-1)
            c_step=float(lg_col-self.col)/(c_len-1)
            for r_index in range(r_len):
                for c_index in range(c_len):
                    ri=r0+int(min(lg_row-self.row,round(r_index*r_step)))
                    ci=c0+int(min(lg_col-self.col,round(c_index*c_step)))
                    ro=ri+self.row
                    co=ci+self.col
                    entry=MetaInfo(img.replace(dotext,("_#%d#%d#%d#%d#%d#%d#"+dotext)%(lg_row,lg_col,ri,ro,ci,co))
                        ,img,lg_row,lg_col,ri,ro,ci,co)
                    view_list.append(entry)  # add to either tr or val set
        print("Images were divided into [%d] views"%(len(view_list)))
        return view_list

    def low_std_exclusion(self,view_list):
        ex_list=[]
        for view in view_list:
            img=self.get_image(view)
            stdnp=np.std(img,axis=(0,1))
            if np.max(stdnp)<8:
                ex_list.append(view)
        return ex_list

    def view_coord_batch(self):
        view_batch={}
        view_name=[]
        for view in itertools.chain(self.tr_view,self.val_view):
            sub=view_batch.get(view.image_name, [])
            sub.append(view)
            view_name.append(view.file_name)
            view_batch[view.image_name]=sub
        return view_batch, view_name
