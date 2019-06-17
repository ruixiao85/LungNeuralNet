import itertools
import math
import os
from cv2 import cv2
import numpy as np

from a_config import Config,parse_float
from osio import mkdir_ifexist,find_file_pattern,find_file_pattern_rel,find_folder_prefix,find_file_ext_recursive,find_file_ext_recursive_rel
from postprocess import morph_close,morph_open,gaussian_smooth,fill_contour
from preprocess import read_image,read_resize,read_resize_pad,read_resize_fit,extract_pad_image

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

        self.data=None

    def __str__(self):
        return self.file_name

    def __eq__(self, other):
        return str(self)==str(other)

    def __hash__(self):
        return str(self).__hash__()

class ImageSet:
    def __init__(self,cfg:Config,wd,sf,channels):
        self.work_directory=wd
        self.sub_category=sf
        self.channels=channels
        self.image_format=cfg.image_format if channels!=4 else '*.png' # jpg<=3 channels; png<=4 channels (alpha)
        self.image_val_format=cfg.image_val_format if channels!=4 else '*.png' # jpg<=3 channels; png<=4 channels (alpha)
        self.target_scale=cfg.target_scale
        self.train_val_split=cfg.train_val_split
        self.target_folder=self.label_scale()
        self.raw_folder,self.raw_scale,self.resize_ratio=None,None,None
        self.image_data=None # dict RGB data
        self.tr_list,self.val_list=None,None

    def label_scale(self,target=None,scale=None):
        return "%s_%.1f"%(target or self.sub_category, scale or self.target_scale)

    def prep_folder(self):
        self.folder_screen_split()
        return self

    def folder_screen_split(self):
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
        print("Processing images from folder [%s] with resize_ratio of %.1fx ..."%(input_folder,self.resize_ratio))
        if self.raw_scale<self.target_scale:
            print("Warning, upsampling from low-res raw images is not recommended!")
        self.val_list=find_file_ext_recursive_rel(os.path.join(self.work_directory,input_folder),self.image_val_format) # may find explicit val files
        if len(self.val_list)>0:
            self.tr_list=find_file_ext_recursive_rel(os.path.join(self.work_directory,input_folder),self.image_format) # complete the training set
        else:
            print("No [%s] files found, splitting [%s] images with [%.2f] ratio."%(self.image_val_format,self.image_format,self.train_val_split))
            self.tr_list,self.val_list=[],[]
            images=find_file_ext_recursive_rel(os.path.join(self.work_directory,input_folder),self.image_format) # need more splitting work
            for img in images:
                if (len(self.val_list)+0.05)/(len(self.tr_list)+0.05)>self.train_val_split:
                    self.tr_list.append(img)
                else:
                    self.val_list.append(img)
        print("[%s] was split into training [%d] and validation [%d] set."%(self.sub_category,len(self.tr_list),len(self.val_list)))
        print("Loading image files (train/val) to memory...")
        self.image_data={}
        for sel_list in [self.tr_list,self.val_list]:
            for image in sel_list:
                self.image_data[image]=self.adapt_channel(read_resize(os.path.join(self.work_directory,input_folder,image),self.resize_ratio))
                print("  "+image,end='')
            print()

    def adapt_channel(self,img,channels=None):
        return np.mean(img,axis=-1,keepdims=True) if (channels or self.channels)==1 else img


    def get_raw_image(self,view:MetaInfo):
        return self.adapt_channel(read_image(os.path.join(self.work_directory,self.raw_folder,view.image_name)))

    def get_image(self,view:MetaInfo,whole=False,pad_value=255):
        img=self.image_data[view.image_name]
        return img if whole else extract_pad_image(img,view.ori_row,view.ori_col,view.row_start,view.row_end,view.col_start,view.col_end,pad_value)

    def get_mask(self,view:MetaInfo,whole=False,pad_value=None):
        pad_value=pad_value or (255 if self.channels==3 else 0) # pad 255 for patches with channels=3
        view=self.get_image(view,whole,pad_value)
        return view[...,3] if self.channels==4 else 255-view[...,1] if self.channels==3 else view[...,0] # 4: alpha 3: process further on green

class ViewSet(ImageSet):
    def __init__(self,cfg: Config,wd,sf,channels,is_train,low_std_ex):
        super(ViewSet,self).__init__(cfg,wd,sf,channels)
        self.coverage=cfg.coverage_train if is_train else cfg.coverage_predict
        self.list_to_view=self.list_to_view_with_overlap if self.coverage>0 else self.list_to_view_without_overlap
        self.train_step=cfg.train_step
        self.row,self.col=cfg.row_in,cfg.col_in
        self.low_std_ex=low_std_ex
        self.tr_view,self.val_view=None,None  # lists -> views with specified size
        self.tr_view_ex,self.val_view_ex=None,None  # views with low contrast

    def res(self,rows=None,cols=None):
        return "%dx%d"%(rows or self.row, cols or self.col)
    def labelres_scale(self,target=None,rows=None,cols=None,scale=None):
        return "%s%s_%s"%(target or self.sub_category, self.res(rows,cols), scale or self.target_scale)
    def label_scale_res(self,target=None,scale=None,rows=None,cols=None):
        return "%s_%s"%(self.label_scale(target,scale), self.res(rows,cols))
    def scale_res(self,scale=None,rows=None,cols=None):
        return "%s_%s"%(scale or self.target_scale, self.res(rows,cols))
    def scale_allres(self,scale=None):
        return "%s_*"%(scale or self.target_scale)

    def prep_folder(self):
        self.folder_screen_split()
        self.ext_image(self.tr_list); self.ext_image(self.val_list) # in case the original image is even smaller than the size of one view
        self.tr_view,self.val_view=self.list_to_view(self.tr_list),self.list_to_view(self.val_list)
        if self.low_std_ex:
            self.tr_view_ex,self.val_view_ex=self.low_std_exclusion(self.tr_view),self.low_std_exclusion(self.val_view)
        return self

    def ext_image(self,img_list):
        for name in img_list:
            data=self.image_data[name]
            row,col,_=data.shape
            if row<self.row or col<self.col: # pad needed
                row_pad=max(0,int(math.ceil(self.row-row)/2.0))
                col_pad=max(0,int(math.ceil(self.col-col)/2.0))
                print("  pad[%d,%d]@%s"%(row_pad,col_pad,name),end='')
                self.image_data[name]=np.pad(data,((row_pad,row_pad),(col_pad,col_pad),(0,0)), 'reflect')

    def list_to_view_with_overlap(self,img_list):
        dotext=self.image_format[1:]
        view_list=[]
        for img in img_list:
            _img=self.image_data[img]
            lg_row,lg_col,lg_dep=_img.shape
            r_len=max(1,1+int(math.ceil((lg_row-self.row)*self.coverage/self.row)))
            c_len=max(1,1+int(math.ceil((lg_col-self.col)*self.coverage/self.col)))
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

    def list_to_view_without_overlap(self,img_list):
        dotext=self.image_format[1:]
        view_list=[]
        for img in img_list:
            _img=self.image_data[img]
            lg_row,lg_col,lg_dep=_img.shape
            for ri in range(0,lg_row,self.row):
                for ci in range(0,lg_col,self.col):
                    ro=ri+self.row
                    co=ci+self.col
                    entry=MetaInfo(img.replace(dotext,("_#%d#%d#%d#%d#%d#%d#"+dotext)%(lg_row,lg_col,ri,ro,ci,co))
                        ,img,lg_row,lg_col,ri,ro,ci,co)
                    view_list.append(entry)  # add to either tr or val set
        print("Images were divided into [%d] views without overlap"%(len(view_list)))
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
    def __init__(self,cfg: Config,wd,sf,channels):
        super(PatchSet,self).__init__(cfg,wd,sf,channels)
        self.tr_view,self.val_view=None,None  # lists -> views with specified size
        self.tr_view_ex,self.val_view_ex=None,None  # views with low contrast

    def prep_folder(self):
        self.folder_screen_split()
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
            entry=MetaInfo(img.replace(dotext,("_#%d#%d#%d#%d#%d#%d#"+dotext)%(lg_row,lg_col,ri,ro,ci,co))
                ,img,lg_row,lg_col,ri,ro,ci,co)
            view_list.append(entry)  # add to either tr or val set
        print("Images were divided into [%d] views"%(len(view_list)))
        return view_list

    def get_mask(self,view:MetaInfo,threshold=50,**kwargs):
        msk=super(PatchSet,self).get_mask(view,**kwargs)
        _,binary=cv2.threshold(gaussian_smooth(msk,3),threshold,255,cv2.THRESH_BINARY)
        return fill_contour(binary)
