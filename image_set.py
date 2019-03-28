import itertools
import os
from cv2 import cv2
import numpy as np

from a_config import Config
from osio import mkdir_ifexist,find_file_pattern,find_file_pattern_rel,find_folder_prefix,find_file_ext_recursive,find_file_ext_recursive_rel
from postprocess import morph_close,morph_open,gaussian_smooth,fill_contour
from preprocess import read_image,read_resize,read_resize_pad,read_resize_fit,extract_pad_image

def parse_float(text):
    try:
        return float(text)
    except ValueError:
        return None

class MetaInfo:
    def __init__(self, file, image, ori_row, ori_col, ri, ro, ci, co, min=None, max=None, ave=None, std=None):
        self.file_name = file  # direct file can be a slice
        self.image_name = image # can be name of the whole image, can different from file_name (slice)
        self.ori_row = ori_row
        self.ori_col = ori_col
        self.row_start = ri
        self.row_end = ro
        self.col_start = ci
        self.col_end = co

        #PatchSet
        self.min=min
        self.max=max
        self.ave=ave
        self.std=std
        self.data=None # add data later for image-patch pair (RGB images, Binary masks, classes)

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
        self.image_format=cfg.image_format if channels!=4 else '*.png' # jpg<=3 channels; png<=4 channels (alpha)
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
        return np.mean(img,axis=-1,keepdims=True) if (channels or self.channels)==1 else img

    def get_image(self,view):
        if isinstance(view,MetaInfo):
            return self.image_data[view.image_name][view.row_start:view.row_end,view.col_start:view.col_end,0:3]
        return self.image_data[view][:,:,0:3] # can also be a file
    def get_mask(self,view,threshold=50):
        if isinstance(view,MetaInfo):
            msk=self.image_data[view.image_name][view.row_start:view.row_end,view.col_start:view.col_end,3] if self.channels==4\
                else 255-self.image_data[view.image_name][view.row_start:view.row_end,view.col_start:view.col_end,1] if self.channels==3\
                else self.image_data[view.image_name][view.row_start:view.row_end,view.col_start:view.col_end,0] # 4: alpha 3: process further on green
        else:
            msk=self.image_data[view][:,:,3] if self.channels==4\
                else 255-self.image_data[view.image_name][:,:,1] if self.channels==3\
                else self.image_data[view][:,:,0] # 4: alpha 3: process further on green
        # _,bin=cv2.threshold(msk,threshold,255,cv2.THRESH_BINARY)
        _,bin=cv2.threshold(gaussian_smooth(msk,3),threshold,255,cv2.THRESH_BINARY)
        return fill_contour(bin)
        # return morph_close(bin,erode=5,dilate=5)

    # def get_masks(self, _path, cfg:Config):
    #     import glob
    #     import random
    #     files=glob.glob(os.path.join(_path,self.file_name.replace(cfg.image_format[1:],cfg.image_format)))
    #     random.shuffle(files) # reorder patches
    #     print(' found %d files matching %s'%(len(files),self.file_name))
        # msks,clss=None,[]
        # for f in files:
        #     class_split=f.split('^')
        #     clss.append(class_split[int(len(class_split)-2)])
        #     msk=self.get_mask(_path, cfg, f)[...,np.newaxis] # np.uint8 0-255
        #     msks=msk if msks is None else np.concatenate((msks,msk),axis=-1)
        # return msks, np.array(clss,dtype=np.uint8) # 0 ~ 255

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
            r0,r_step=(0,float(lg_row-self.row)/(r_len-1)) if r_len>1 else (int(0.5*(lg_row-self.row)),0)
            c0,c_step=(0,float(lg_col-self.col)/(c_len-1)) if c_len>1 else (int(0.5*(lg_col-self.col)),0)
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

class PatchSet(ImageSet):
    def __init__(self,cfg: Config,wd,sf,is_train,channels):
        super(PatchSet,self).__init__(cfg,wd,sf,is_train,channels)
        self.tr_view,self.val_view=None,None  # lists -> views with specified size
        self.tr_view_ex,self.val_view_ex=None,None  # views with low contrast

    def prep_folder(self):
        self.prescreen_folders()
        self.split_tr_val() # tr/val-split only needed for img_set
        self.tr_view=self.list_to_view(self.tr_list)
        self.val_view=self.list_to_view(self.val_list)
        return self

    def list_to_view(self,img_list):
        dotext=self.image_format[1:]
        view_list=[]
        for img in img_list:
            _img=self.image_data[img]
            lg_row,lg_col,lg_dep=_img.shape
            ri,ro,ci,co=0,lg_row,0,lg_col
            _gray=np.mean(_img,axis=-1,keepdims=True) # stats based on grayscale
            entry=MetaInfo(img.replace(dotext,("_#%d#%d#%d#%d#%d#%d#"+dotext)%(lg_row,lg_col,ri,ro,ci,co))
                ,img,lg_row,lg_col,ri,ro,ci,co,np.min(_gray),np.max(_gray),np.average(_gray),np.std(_gray))
            view_list.append(entry)  # add to either tr or val set
        print("Images were divided into [%d] views"%(len(view_list)))
        return view_list
