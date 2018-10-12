import os
import random
from cv2 import cv2
import numpy as np
import keras
import util
from net.basenet import Net
from process_image import augment_image_pair,read_resize_padding,extract_pad_image,prep_scale,rev_scale

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

    def get_image(self, _path, cfg:Net):
        return read_resize_padding(os.path.join(_path, self.file_name),_resize=1.0,_padding=1.0) if cfg.separate else\
            extract_pad_image(read_resize_padding(os.path.join(_path, self.image_name),cfg.image_resize,cfg.image_padding), self.row_start, self.row_end, self.col_start, self.col_end)

    def get_mask(self, _path, cfg:Net):
        img=self.get_image(_path, cfg)
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


    def __str__(self):
        return self.file_name

    def __eq__(self, other):
        return str(self)==str(other)

    def __hash__(self):
        return str(self).__hash__()

class FolderSet:
    def __init__(self,cfg: Net,wd,sf,is_train,is_image):
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
        self.images=[os.path.relpath(absp,path) for absp in util.find_file_recursive(path, self.cfg.image_format)]

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
        for image_name in self.images:
            _img=read_resize_padding(os.path.join(self.work_directory,self.sub_folder,image_name),self.cfg.image_resize,self.cfg.image_padding)
            print(image_name)
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

class ImageSet(FolderSet):
    def __init__(self,cfg: Net,wd,sf,is_train,is_image):
        super(ImageSet,self).__init__(cfg,wd,sf,is_train,is_image)

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


class ImageMaskPair:
    def __init__(self, cfg:Net, wd, origin, targets, is_train, is_reverse=False):
        self.cfg=cfg
        self.wd = wd
        self.origin = origin
        self.targets = targets if isinstance(targets,list) else [targets]
        # self.dir_out=targets[0] if len(targets)==1 else ','.join([t[:4] for t in targets])
        self.img_set=ImageSet(cfg, wd, origin, is_train, is_image=True).size_folder_update()
        self.msk_set = None
        self.view_coord=self.img_set.view_coord
        self.is_train = is_train
        self.is_reverse = is_reverse

    def train_generator(self):
        i = 0; no=self.cfg.dep_out; nt=len(self.targets)
        while i < nt:
            o=min(i+no, nt)
            views = set(self.img_set.view_coord)
            self.msk_set = []
            tgt_list=[]
            for t in self.targets[i:o]:
                tgt_list.append(t)
                msk = ImageSet(self.cfg, self.wd, t, is_train=True, is_image=False).size_folder_update()
                self.msk_set.append(msk)
                views = views.intersection(msk.view_coord)
            self.view_coord = list(views)
            tr_list, val_list = [], []  # list view_coords, can be from slices
            tr_image, val_image = set(), set()  # set whole images
            for vc in self.view_coord:
                if vc.image_name in tr_image:
                    tr_list.append(vc)
                    tr_image.add(vc.image_name)
                elif vc.image_name in val_image:
                    val_list.append(vc)
                    val_image.add(vc.image_name)
                else:
                    if (len(val_list) + 0.05) / (len(tr_list) + 0.05) > self.cfg.train_vali_split:
                        tr_list.append(vc)
                        tr_image.add(vc.image_name)
                    else:
                        val_list.append(vc)
                        val_image.add(vc.image_name)
            print("From %d split into train: %d views %d images; validation %d views %d images" %
                  (len(self.view_coord), len(tr_list), len(tr_image), len(val_list), len(val_image)))
            print("Training Images:"); print(tr_image)
            print("Validation Images:"); print(val_image)
            yield(ImageGenerator(self, self.cfg.train_aug, tgt_list, tr_list), ImageGenerator(self, 0, tgt_list, val_list),
                  self.dir_in_ex(self.origin) if self.is_reverse else self.dir_out_ex(self.join_targets(tgt_list)) )
            i=o

    def predict_generator(self):
        # yield (ImageGenerator(self, False, self.targets, self.view_coord),self.join_targets(self.targets), self.targets)
        i = 0; nt = len(self.targets)
        ps = self.cfg.predict_size
        while i < nt:
            o = min(i + ps, nt)
            tgt_list=self.targets[i:o]
            yield (self.join_targets(tgt_list), tgt_list)
            i = o

    @staticmethod
    def join_targets(tgt_list) :
        # return ','.join(tgt_list)
        # return ','.join(tgt_list[:max(1, int(24 / len(tgt_list)))]) #shorter but >= 1 char, may have error if categories share same leading chars
        maxchar=max(1, int(28 / len(tgt_list))) # clip to fewer leading chars
        # maxchar=9999 # include all
        return ','.join(tgt[:maxchar] for tgt in tgt_list)

    def dir_in_ex(self,txt=None):
        ext=ImageSet.ext_folder(self.cfg, True)
        txt=txt or self.origin
        return txt+'_'+ext if self.cfg.separate else txt

    def dir_out_ex(self,txt=None):
        ext=ImageSet.ext_folder(self.cfg, False)
        if txt is None:
            return ext if self.cfg.separate else None
        return txt+'_'+ext if self.cfg.separate else txt

class ImageGenerator(keras.utils.Sequence):
    def __init__(self,pair:ImageMaskPair,aug_value,tgt_list,view_coord=None):
        self.pair=pair
        self.cfg=pair.cfg
        self.aug_value=aug_value
        self.target_list=tgt_list
        self.view_coord=pair.view_coord if view_coord is None else view_coord
        self.indexes = np.arange(len(self.view_coord))

    def __len__(self):  # Denotes the number of batches per epoch
        return int(np.ceil(len(self.view_coord) / self.cfg.batch_size))

    def __getitem__(self, index):  # Generate one batch of data
        indexes = self.indexes[index * self.cfg.batch_size:(index + 1) * self.cfg.batch_size]
        # print(" getting index %d with %d batch size"%(index,self.batch_size))
        if self.pair.is_train:
            _img = np.zeros((self.cfg.batch_size, self.cfg.row_in, self.cfg.col_in, self.cfg.dep_in), dtype=np.uint8)
            _tgt = np.zeros((self.cfg.batch_size, self.cfg.row_out, self.cfg.col_out, self.cfg.dep_out), dtype=np.uint8)
            for vi, vc in enumerate([self.view_coord[k] for k in indexes]):
                _img[vi, ...] = vc.get_image(os.path.join(self.pair.wd, self.pair.dir_in_ex()), self.cfg)
                if self.cfg.out_image:
                    # for ti,tgt in enumerate(self.target_list):
                    #     _tgt[vi, ...,ti] =np.average( vc.get_image(os.path.join(self.pair.wd, self.pair.dir_out_ex(tgt)), self.cfg), axis=-1) # average RGB to gray
                    _tgt[vi, ...] =vc.get_image(os.path.join(self.pair.wd, self.pair.dir_out_ex(self.target_list[0])), self.cfg)
                else:
                    for ti,tgt in enumerate(self.target_list):
                        _tgt[vi, ..., ti] = vc.get_mask(os.path.join(self.pair.wd, self.pair.dir_out_ex(tgt)), self.cfg)
            if self.aug_value > 0:
                aug_value=random.randint(0, self.cfg.train_aug) # random number between zero and pre-set value
                # print("  aug: %.2f"%aug_value,end='')
                _img, _tgt = augment_image_pair(_img, _tgt, aug_value)  # integer N: a <= N <= b.
                # imwrite("tr_img.jpg",_img[0])
                # imwrite("tr_tgt.jpg",_tgt[0])
            return prep_scale(_img, self.cfg.feed), prep_scale(_tgt, self.cfg.out)
        else:
            _img = np.zeros((self.cfg.batch_size, self.cfg.row_in, self.cfg.col_in, self.cfg.dep_in), dtype=np.uint8)
            for vi, vc in enumerate([self.view_coord[k] for k in indexes]):
                _img[vi, ...] = vc.get_image(os.path.join(self.pair.wd, self.pair.dir_in_ex()), self.cfg)
                # imwrite("prd_img.jpg",_img[0])
            return prep_scale(_img, self.cfg.feed), None

    def on_epoch_end(self):  # Updates indexes after each epoch
        self.indexes = np.arange(len(self.view_coord))
        if self.pair.is_train and self.cfg.train_shuffle:
            np.random.shuffle(self.indexes)


def smooth_brighten(img):
    blur=np.average(gaussian_smooth(img),axis=-1).astype(np.uint8)
    _,bin=cv2.threshold(blur,20,255, cv2.THRESH_BINARY)
    # bin=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,353,-50)
    # return cv2.morphologyEx(bin, cv2.MORPH_OPEN, (5,5))
    return morph_operation(bin)

def gaussian_smooth(_img,size=11):
    # return cv2.blur(_img,(size,size))
    return cv2.GaussianBlur(_img,(size,size),0)

def morph_operation(_bin,erode=5,dilate=9):
    return cv2.morphologyEx(cv2.morphologyEx(_bin,cv2.MORPH_ERODE,(erode,erode)),cv2.MORPH_DILATE,(dilate,dilate))


class NoiseSet(FolderSet):
    def __init__(self,cfg:Net,wd,sf,is_train,is_image):
        self.bright_diff=-10 # local brightness should be more than noise patch brightness,
        self.min_initialize=0.000005 # min rate of random points and loop through
        self.max_initialize=0.0001 # max rate of random points and loop through
        self.aj_size=4
        self.aj_std=0.2
        self.patches=None
        super(NoiseSet,self).__init__(cfg,wd,sf,is_train,is_image)
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

    def add_noise(self,img, divider=1, remainder=0):
        inserted=0
        lg_row,lg_col,lg_dep=img.shape
        msk=img.copy()
        lg_sum=lg_row*lg_col
        lg_min,lg_max=int(lg_sum*self.min_initialize),int(lg_sum*self.max_initialize)
        rand_num=[(random.random(),random.uniform(0,1),random.uniform(0,1)) for r in range(random.randint(lg_min,lg_max))] # index,row,col
        times=self.num_patches//divider
        for irc in rand_num:
            idx=int(times*irc[0])+remainder # index of patch to apply
            patch=self.view_coord[idx]
            p_row,p_col,p_ave,p_std=patch.ori_row, patch.ori_col, patch.row_start, patch.row_end
            lri=int(lg_row*irc[1])-p_row//2 # large row in/start
            lci=int(lg_col*irc[2])-p_col//2 # large col in/start
            lro,lco=lri+p_row,lci+p_col # large row/col out/end
            pri=0 if lri>=0 else -lri; lri=max(0,lri)
            pci=0 if lci>=0 else -lci; lci=max(0,lci)
            pro=p_row if lro<=lg_row else p_row-lro+lg_row; lro=min(lg_row,lro)
            pco=p_col if lco<=lg_col else p_col-lco+lg_col; lco=min(lg_col,lco)
            # if np.average(img[lri:lro,lci:lco])-p_ave > self.bright_diff and \
            if np.min(img[lri:lro,lci:lco])-p_ave > self.bright_diff and \
                int(np.std(img[lri-p_row*self.aj_size:lro+p_row*self.aj_size,lci-p_col*self.aj_size:lco+p_col*self.aj_size])>self.aj_std*p_std): # target area is brighter, then add patch
                # print("large row(%d) %d-%d col(%d) %d-%d  patch row(%d) %d-%d col(%d) %d-%d"%(lg_row,lri,lro,lg_col,lci,lco,p_row,pri,pro,p_col,pci,pco))
                pat=self.patches[idx][pri:pro,pci:pco]
                if random.random()>0.5: pat=np.fliplr(pat)
                if random.random()>0.5: pat=np.flipud(pat)
                img[lri:lro,lci:lco]=np.minimum(img[lri:lro,lci:lco],pat)
                # imwrite('test_addnoise.jpg',img)
                # lr=(lri+lro)//2
                # lc=(lci+lco)//2
                # msk[lr:lr+1,lc:lc+1,1]=255
                inserted+=1
        print("inserted %d"%inserted)
        return img, smooth_brighten(msk-img)

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
    #
    # def size_folder_update(self):
    #     self.find_file_recursive_rel()
    #     if self.cfg.separate:
    #         new_dir="%s_%s" % (self.sub_folder, self.ext_folder(self.cfg, self.is_image))
    #         new_path=os.path.join(self.work_directory, new_dir)
    #         # shutil.rmtree(new_path)  # force delete
    #         if not os.path.exists(new_path): # change folder and not found
    #             os.makedirs(new_path)
    #             self.split_image_coord(new_path)
    #         self.sub_folder=new_dir
    #         self.find_file_recursive_rel()
    #     self.single_image_coord()
    #     return self

    def single_image_coord(self):
        self.view_coord=[]
        self.patches=[]
        for image_name in self.images:
            _img = read_resize_padding(os.path.join(self.work_directory, self.sub_folder, image_name),
                                       _resize=1.0,_padding=1.0) # TODO 40X-40X resize=1.0
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



class ImageNoisePair(ImageMaskPair):
    def __init__(self,cfg: Net,wd,origin,targets,is_train):
        prev_separate=cfg.separate
        cfg.separate=False  # separate=False to scan whole images
        self.nos_set=[]
        self.msk_set=[]
        super(ImageNoisePair,self).__init__(cfg,wd,origin,targets,is_train)
        i=0 # only 1 element allowed
        tgt_noise=NoiseSet(cfg,wd,targets[i],is_train,True)
        tgt_noise.sub_folder=targets[i]+'+'
        msk_folder=targets[i]+'-'
        ngroups=len(self.img_set.view_coord)
        exist1=util.mk_dir_if_nonexist(os.path.join(tgt_noise.work_directory,tgt_noise.sub_folder))
        exist2=util.mk_dir_if_nonexist(os.path.join(tgt_noise.work_directory,msk_folder))
        if not exist1 or not exist2:
            self.cfg.separate=True # separate=True to read full scale
            for vi,vc in enumerate(self.img_set.view_coord):
                img=vc.get_image(os.path.join(self.img_set.work_directory,self.img_set.sub_folder),self.cfg)
                # cv2.imwrite(os.path.join(tgt_noise.work_directory,tgt_noise.sub_folder,'_'+vc.image_name),img)
                img,msk=tgt_noise.add_noise(img, divider=ngroups, remainder=vi)
                cv2.imwrite(os.path.join(tgt_noise.work_directory,tgt_noise.sub_folder,vc.image_name),img,[int(cv2.IMWRITE_JPEG_QUALITY),100])
                cv2.imwrite(os.path.join(tgt_noise.work_directory,msk_folder,vc.image_name),msk,[int(cv2.IMWRITE_JPEG_QUALITY),100])
            self.nos_set.append(tgt_noise)
        self.cfg.separate=prev_separate # return to original setting

        # self.origin=tgt_noise.sub_folder
        # self.targets=[origin]
        # self.cfg.dep_out=3
        # self.cfg.out='tanh'

        # self.img_set=NoiseSet(cfg, wd, origin, is_train, is_image=True).size_folder_update()


        # self.cfg=cfg
        # self.wd = wd
        # self.origin = origin
        # self.targets = targets if isinstance(targets,list) else [targets]
        # # self.dir_out=targets[0] if len(targets)==1 else ','.join([t[:4] for t in targets])
        # self.img_set=NoiseSet(cfg, wd, origin, is_train, is_image=True).size_folder_update()
        # self.msk_set = None
        # self.view_coord=self.img_set.view_coord
        # self.is_train = is_train

    def train_generator(self):
        i = 0; no=self.cfg.dep_out; nt=len(self.targets)
        while i < nt:
            o=min(i+no, nt)
            views = set(self.img_set.view_coord)
            self.msk_set = []
            tgt_list=[]
            for t in self.targets[i:o]:
                tgt_list.append(t)
                msk = NoiseSet(self.cfg, self.wd, t, is_train=True, is_image=False).size_folder_update()
                self.msk_set.append(msk)
                views = views.intersection(msk.view_coord)
            self.view_coord = list(views)
            tr_list, val_list = [], []  # list view_coords, can be from slices
            tr_image, val_image = set(), set()  # set whole images
            for vc in self.view_coord:
                if vc.image_name in tr_image:
                    tr_list.append(vc)
                    tr_image.add(vc.image_name)
                elif vc.image_name in val_image:
                    val_list.append(vc)
                    val_image.add(vc.image_name)
                else:
                    if (len(val_list) + 0.05) / (len(tr_list) + 0.05) > self.cfg.train_vali_split:
                        tr_list.append(vc)
                        tr_image.add(vc.image_name)
                    else:
                        val_list.append(vc)
                        val_image.add(vc.image_name)
            print("From %d split into train: %d views %d images; validation %d views %d images" %
                  (len(self.view_coord), len(tr_list), len(tr_image), len(val_list), len(val_image)))
            print("Training Images:"); print(tr_image)
            print("Validation Images:"); print(val_image)
            yield(ImageGenerator(self, self.cfg.train_aug, tgt_list, tr_list), ImageGenerator(self, 0, tgt_list, val_list),
                    self.join_targets(tgt_list))
            i=o

    # def predict_generator(self):
    #     # yield (ImageGenerator(self, False, self.targets, self.view_coord),self.join_targets(self.targets), self.targets)
    #     i = 0; nt = len(self.targets)
    #     ps = self.cfg.predict_size
    #     while i < nt:
    #         o = min(i + ps, nt)
    #         tgt_list=self.targets[i:o]
    #         yield (self.join_targets(tgt_list),tgt_list)
    #         i = o
