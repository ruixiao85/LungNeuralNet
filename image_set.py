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

    def get_image(self, _path, overridde=None):
        file=overridde if overridde is not None else self.file_name
        return read_image(os.path.join(_path, file))
        # return extract_pad_image(read_resize_padding(os.path.join(_path, file),_resize=1.0,_padding=1.0), self.row_start, self.row_end, self.col_start, self.col_end)

    def get_mask(self, _path, overridde=None):
        img=self.get_image(_path, overridde)
        # imwrite("test_img.jpg",img)
        # return img[...,1]  # from green channel
        return np.max(img,axis=-1,keepdims=False)  # max channel
        # return np.clip(np.sum(img,axis=-1,keepdims=False), 0, 255).astype(np.uint8)  # sum channel

    def get_masks(self, _path, cfg:Config):
        import glob
        import random
        files=glob.glob(os.path.join(_path,self.file_name.replace(cfg.image_format[1:],cfg.image_format)))
        random.shuffle(files) # avaible same order of files
        # print(' found %d files matching %s'%(len(files),self.file_name))
        msks,clss=None,[]
        for f in files:
            class_split=f.split('^')
            clss.append(class_split[int(len(class_split)-2)])
            msk=self.get_mask(_path, f)[...,np.newaxis] # np.uint8 0-255
            msks=msk if msks is None else np.concatenate((msks,msk),axis=-1)
        return msks, np.array(clss,dtype=np.uint8) # 0 ~ 255

    def __str__(self):
        return self.file_name

    def __eq__(self, other):
        return str(self)==str(other)

    def __hash__(self):
        return str(self).__hash__()

class ImageSet:
    def __init__(self,cfg:Config,wd,sf,is_train):
        self.work_directory=wd
        self.sub_category=sf
        self.is_train=is_train
        self.image_format=cfg.image_format
        self.target_scale=cfg.target_scale
        self.images=None
        self.target_folder=None
        self.raw_folder,self.raw_scale,self.resize_ratio=None,None,None

    def prescreen_folders(self):
        initial_folders=find_folder_prefix(self.work_directory,self.sub_category+'_')
        folders=initial_folders.copy()
        for folder in initial_folders:
            print(' ',folder,end=' ')
            if folder==self.target_folder:
                print('+ individual images.')
            else:
                sections=folder.split('_')
                if len(sections)!=2 or parse_float(sections[1]) is None:
                    folders.remove(folder)
                    print('- omitted.')
                else:
                    print('# potential raw images.')
        self.raw_folder=sorted(folders,key=lambda t:float(t.split('_')[1]),reverse=True)[0]  # high-res first
        self.raw_scale=float(self.raw_folder.split('_')[1])
        self.resize_ratio=round(self.target_scale/self.raw_scale,1)

class ViewSet(ImageSet):
    def __init__(self,cfg: Config,wd,sf,is_train):
        super(ViewSet,self).__init__(cfg,wd,sf,is_train)
        self.coverage=cfg.coverage_train if self.is_train else cfg.coverage_predict
        self.train_step=cfg.train_step
        self.row,self.col=cfg.row_in,cfg.col_in
        self.target_folder=self.category_detail()
        self.view_coord=None  # list

    def category_detail(self,cate=None,scale=None,row=None,col=None):
        return "%s_%.1f_%dx%d"%(cate or self.sub_category,scale or self.target_scale,row or self.row,col or self.col)
    def category_append(self,scale=None,row=None,col=None):
        return "%.1f_%dx%d"%(scale or self.target_scale,row or self.row,col or self.col)

    def prep_folder(self):
        self.prescreen_folders()
        if not mkdir_ifexist(os.path.join(self.work_directory,self.target_folder)):
            self.convert_from_folder()
        self.scan_folder()
        return self

    def convert_from_folder(self): # pre-defined size
        print("Converting from folder [%s -> %s] %.1f -> %.1f (%.1fx) ..."%(self.raw_folder,self.target_folder,self.raw_scale,self.target_scale,self.resize_ratio))
        if self.resize_ratio>1.0:
            print("Warning, upsampling from low-res raw images is not recommended!")
        images=find_file_ext_recursive_rel(os.path.join(self.work_directory,self.raw_folder),self.image_format)
        for image in images:
            _img=read_resize(os.path.join(self.work_directory,self.raw_folder,image),self.resize_ratio)
            lg_row,lg_col,lg_dep=_img.shape
            r_len=max(1,1+int(round((lg_row-self.row)/self.row*self.coverage)))
            c_len=max(1,1+int(round((lg_col-self.col)/self.col*self.coverage)))
            print("%s target %d x %d (coverage %.1f): original %d x %d ->  row /%d col /%d"%(image,self.row,self.col,self.coverage,lg_row,lg_col,r_len,c_len))
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
                    entry=MetaInfo.from_whole(image,lg_row,lg_col,ri,ro,ci,co)
                    cv2.imwrite(os.path.join(self.work_directory,self.target_folder,entry.file_name),s_img,[int(cv2.IMWRITE_JPEG_QUALITY),100])

    def scan_folder(self):
        self.images=find_file_ext_recursive_rel(os.path.join(self.work_directory,self.target_folder),self.image_format)
        total=len(self.images)
        ratio=0.5 # crop the middle portion if actual image is larger
        self.view_coord=[]
        print('Parsing %d files...'%total)
        for i,image in enumerate(self.images):
            _img=read_image(os.path.join(self.work_directory,self.target_folder,image))
            pct10=10*(i+1)//total
            if pct10 > 10*i//total:
                print('%.0f%% ... %s'%(pct10*10,image))
            entry=MetaInfo.from_single(image)
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

    def view_coord_batch(self):
        view_batch={}
        whole_id=0
        for view in self.view_coord:
            key,whole=self.file_to_whole_image(view.file_name)
            if whole:
                whole+=1 if len(view_batch.get(str(whole_id), []))>self.train_step else 0
                sub_list=view_batch.get(str(whole_id), [])
                sub_list.append(view)
                view_batch[str(whole_id)]=sub_list
            else:
                sub_list=view_batch.get(key, [])
                sub_list.append(view)
                view_batch[key]=sub_list
        return view_batch

    @staticmethod
    def file_to_whole_image(text):
        half=text.split('_#')
        if len(half)==2:
            dot_sep=text.split('.')
            return "%s.%s"%(half[0],dot_sep[len(dot_sep)-1]),False
        return text,True


class PatchSet(ImageSet):
    def __init__(self,cfg:Config,wd,sf,is_train):
        super(PatchSet,self).__init__(cfg,wd,sf,is_train)
        self.train_vali_split=cfg.train_vali_split
        self.target_folder="%s_%.1f"%(self.sub_category,self.target_scale) # resolution not specified
        self.view_coord=None  # list
        self.tr_list,self.val_list=None,None
        self.image_data=None

    def prep_folder(self):
        self.prescreen_folders()
        if not mkdir_ifexist(os.path.join(self.work_directory,self.target_folder)):
            folders=sorted(find_folder_prefix(self.work_directory,self.sub_category+'_'),key=lambda t:float(t.split('_')[1]),reverse=True)  # high-res first
            self.convert_from_folder(folders[0])
        self.scan_folder()
        self.split_tr_val() # required for patches only
        return self

    def split_tr_val(self):
        self.tr_list,self.val_list=[],[]
        for i in range(len(self.view_coord)):
            if (len(self.val_list)+0.05)/(len(self.tr_list)+0.05)>self.train_vali_split:
                self.tr_list.append(i)
            else:
                self.val_list.append(i)
        print("%s was split into train %d vs validation %d"%(self.sub_category,len(self.tr_list),len(self.val_list)))

    def convert_from_folder(self,raw_folder): # flexible sizes
        raw_scale=float(raw_folder.split('_')[1])
        resize_ratio=round(self.target_scale/raw_scale,1)
        print("Converting from folder [%s -> %s] %.1f -> %.1f (%.1fx) ..."%(raw_folder,self.target_folder,raw_scale,self.target_scale,resize_ratio))
        if raw_scale<self.target_scale:
            print("Warning, upsampling from low-res raw images is not recommended!")
        images=find_file_ext_recursive_rel(os.path.join(self.work_directory,raw_folder),self.image_format)
        for image in images:
            _img=read_resize(os.path.join(self.work_directory,raw_folder,image),resize_ratio)
            cv2.imwrite(os.path.join(self.work_directory,self.target_folder,image),_img,[int(cv2.IMWRITE_JPEG_QUALITY),100])

    def scan_folder(self):
        self.images=find_file_ext_recursive_rel(os.path.join(self.work_directory,self.target_folder),self.image_format)
        total=len(self.images)
        self.view_coord=[]
        self.image_data=[]
        print('Parsing %d patches...'%total)
        for i,image in enumerate(self.images):
            _img=read_image(os.path.join(self.work_directory,self.target_folder,image))
            pct10=10*(i+1)//total
            if pct10 > 10*i//total:
                print('%.0f%% ... %s'%(pct10*10,image))
            lg_row,lg_col,lg_dep=_img.shape
            self.view_coord.append(MetaInfo(image, image, lg_row, lg_col, 0, lg_row, 0, lg_col))
            self.image_data.append(_img)

