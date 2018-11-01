import os
import cv2
import datetime
import random
import numpy as np
import pandas as pd

from keras.engine.saving import model_from_json,load_model

from basecfg import Config
from image_set import NoiseSet
from osio import mkdir_ifexist,to_excel_sheet
from util import g_kern_rect,draw_text,smooth_brighten


class BaseNetU(Config):
    # 'relu6'  # min(max(features, 0), 6)
    # 'crelu'  # Concatenates ReLU (only positive part) with ReLU (only the negative part). Note that this non-linearity doubles the depth of the activations
    # 'elu'  # Exponential Linear Units exp(features)-1, if <0, features
    # 'selu'  # Scaled Exponential Linear Rectifier: scale * alpha * (exp(features) - 1) if < 0, scale * features otherwise.
    # 'softplus'  # log(exp(features)+1)
    # 'softsign' features / (abs(features) + 1)

    # 'mean_squared_error' 'mean_absolute_error'
    # 'binary_crossentropy'
    # 'sparse_categorical_crossentropy' 'categorical_crossentropy'

    #     model_out = 'softmax'   model_loss='categorical_crossentropy'
    #     model_out='sigmoid'    model_loss=[loss_bce_dice] 'binary_crossentropy' "bcedice"
    def __init__(self, feed=None, act=None, out=None, loss=None, metrics=None, optimizer=None, indicator=None,
                 filename=None, **kwargs):
        super(BaseNetU,self).__init__(**kwargs)
        from metrics import jac, dice, dice67, dice33, acc, acc67, acc33, loss_bce_dice, custom_function_keras
        custom_function_keras()  # leakyrelu, swish
        self.feed=feed or 'tanh'
        self.act=act or 'elu'
        self.out=out or ('sigmoid' if self.dep_out==1 else 'softmax')
        self.loss=loss or (
            loss_bce_dice if self.dep_out==1 else 'categorical_crossentropy')  # 'binary_crossentropy'
        self.metrics=metrics or ([jac, dice, dice67, dice33] if self.dep_out==1 else [acc, acc67, acc33])
        from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
        self.optimizer=optimizer or Adam(1e-5)
        self.indicator=indicator if indicator is not None else ('val_dice' if self.dep_out==1 else 'val_acc')  # indicator to maximize
        self.net=None # abstract -> instatiate in subclass
        self.filename=filename

    @classmethod
    def from_json(cls, filename):  # load model from json
        my_net=cls(filename=filename)
        with open(filename+".json", 'r') as json_file:
            my_net.net=model_from_json(json_file.read())

    def save_net(self):
        json_net=(self.filename if self.filename is not None else str(self)) + ".json"
        with open(json_net, "w") as json_file:
            json_file.write(self.net.to_json())

    def compile_net(self,save_net=False,print_summary=True):
        self.net.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        print("Model compiled")
        if save_net:
            self.save_net()
            print('Model saved to file')
        if print_summary:
            self.net.summary()

    def __str__(self):
        return str(self.net)
    def __repr__(self):
        return str(self.net)+self.predict_proc.__name__[0:1].upper()

    @staticmethod
    def cap_lim_join(lim,*text):
        test_list=[t.capitalize()[:lim] for t in text]
        return ''.join(test_list)

    def train(self,pair):
        for tr,val,dir_out in pair.train_generator():
            export_name=dir_out+'_'+str(self)
            weight_file=export_name+".h5"
            if self.train_continue and os.path.exists(weight_file):
                # print("Continue from previous weights")
                # self.net.load_weights(weight_file)
                print("Continue from previous model with weights & optimizer")
                self.net=load_model(weight_file,custom_objects=custom_function_dict())  # does not work well with custom act, loss func
            print('Fitting neural net...')
            for r in range(self.train_rep):
                print("Training %d/%d for %s"%(r+1,self.train_rep,export_name))
                tr.on_epoch_end()
                val.on_epoch_end()
                from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
                from tensorboard_train_val import TensorBoardTrainVal
                history=self.net.fit_generator(tr,validation_data=val,verbose=1,
                   steps_per_epoch=min(self.train_step,len(tr.view_coord)) if isinstance(self.train_step,int) else len(tr.view_coord),
                   validation_steps=min(self.train_vali_step,len(val.view_coord)) if isinstance(self.train_vali_step,int) else len(val.view_coord),
                   epochs=self.train_epoch,max_queue_size=1,workers=0,use_multiprocessing=False,shuffle=False,
                   callbacks=[
                       ModelCheckpoint(weight_file,monitor=self.indicator,mode='max',save_weights_only=False,save_best_only=True),
                       # ReduceLROnPlateau(monitor=self.indicator, mode='max', factor=0.5, patience=1, min_delta=1e-8, cooldown=0, min_lr=0, verbose=1),
                       EarlyStopping(monitor=self.indicator,mode='max',patience=1,verbose=1),
                       # TensorBoardTrainVal(log_dir=os.path.join("log", export_name), write_graph=True, write_grads=False, write_images=True),
                   ]).history
                if not os.path.exists(export_name+".txt"):
                    with open(export_name+".txt","w") as net_summary:
                        self.net.summary(print_fn=lambda x:net_summary.write(x+'\n'))
                df=pd.DataFrame(history).round(4)
                df['time']=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                df['repeat']=r+1
                df.to_csv(export_name+".csv",mode="a",header=(not os.path.exists(export_name+".csv")))

    def predict(self,pair,pred_dir):
        xls_file="Result_%s_%s.xlsx"%(pred_dir,repr(self))
        img_ext=self.image_format[1:]  # *.jpg -> .jpg
        sum_i,sum_g=self.row_out*self.col_out,None
        msks,mask_wt,r_i,r_g,ra,ca=None,None,None,None,None,None
        mrg_in,mrg_out,mrg_out_wt,merge_dir=None,None,None,None
        batch=pair.img_set.view_coord_batch()  # image/1batch -> view_coord
        dir_ex=pair.dir_out_ex()
        dir_cfg_append=str(self) if dir_ex is None else dir_ex+'_'+str(self)
        res_ind,res_grp=None,None
        save_ind_image=False
        for dir_out,tgt_list in pair.predict_generator():
            res_i,res_g=None,None
            print('Load model and predict to [%s]...'%dir_out)
            export_name=dir_out+'_'+dir_cfg_append
            target_dir=os.path.join(pair.wd,export_name)
            if save_ind_image or not self.separate:  # skip saving individual images
                mkdir_ifexist(target_dir)
            if self.separate:
                merge_dir=os.path.join(pair.wd,dir_out+'+'+dir_cfg_append)  # group
                mkdir_ifexist(merge_dir)
                mask_wt=g_kern_rect(self.row_out,self.col_out)
            for grp,view in batch.items():
                msks=None
                i=0; nt=len(tgt_list)
                while i<nt:
                    o=min(i+self.dep_out,nt)
                    tgt_sub=tgt_list[i:o]
                    prd,tgt_name=pair.predict_generator_partial(tgt_sub,view)
                    weight_file=tgt_name+'_'+dir_cfg_append+'.h5'
                    print(weight_file)
                    self.net.load_weights(weight_file)  # weights only
                    # self.net=load_model(weight_file,custom_objects=custom_function_dict()) # weight optimizer archtecture
                    msk=self.net.predict_generator(prd,max_queue_size=1,workers=0,use_multiprocessing=False,verbose=1)
                    msks=msk if msks is None else np.concatenate((msks,msk),axis=-1)
                    i=o
                print('Saving predicted results [%s] to folder [%s]...'%(grp,export_name))
                # r_i=np.zeros((len(multi.img_set.images),len(tgt_list)), dtype=np.uint32)
                if self.separate:
                    mrg_in=np.zeros((view[0].ori_row,view[0].ori_col,self.dep_in),dtype=np.float32)
                    mrg_out=np.zeros((view[0].ori_row,view[0].ori_col,len(tgt_list)*self.dep_out),dtype=np.float32)
                    mrg_out_wt=np.zeros((view[0].ori_row,view[0].ori_col),dtype=np.float32)+np.finfo(np.float32).eps
                    sum_g=view[0].ori_row*view[0].ori_col
                    # r_g=np.zeros((1,len(tgt_list)*self.dep_out), dtype=np.uint32)
                for i,msk in enumerate(msks):
                    # if i>=len(multi.view_coord): print("skip %d for overrange"%i); break # last batch may have unused entries
                    ind_name=view[i].file_name
                    ind_file=os.path.join(target_dir,ind_name)
                    origin=view[i].get_image(os.path.join(pair.wd,pair.dir_in_ex()),self.net)
                    print(ind_name); text_list=[ind_name]
                    blend,r_i=self.predict_proc(self.net,origin,msk,ind_file.replace(img_ext,''))
                    for d in range(len(tgt_list)):
                        text="[  %d: %s] #%d $%d / $%d  %.2f%%"%(d,tgt_list[d],r_i[d][1],r_i[d][0],sum_i,100.*r_i[d][0]/sum_i)
                        print(text); text_list.append(text)
                    if save_ind_image or not self.separate:  # skip saving individual images
                        blendtext=draw_text(self.net,blend,text_list,self.row_out)  # RGB:3x8-bit dark text
                        cv2.imwrite(ind_file,blendtext)
                    res_i=r_i[np.newaxis,...] if res_i is None else np.concatenate((res_i,r_i[np.newaxis,...]))

                    if self.separate:
                        ri,ro=view[i].row_start,view[i].row_end
                        ci,co=view[i].col_start,view[i].col_end
                        ra,ca=view[i].ori_row,view[i].ori_col
                        tri,tro=0,self.row_out
                        tci,tco=0,self.col_out
                        if ri<0: tri=-ri; ri=0
                        if ci<0: tci=-ci; ci=0
                        if ro>ra: tro=tro-(ro-ra); ro=ra
                        if co>ca: tco=tco-(co-ca); co=ca
                        mrg_in[ri:ro,ci:co]=origin[tri:tro,tci:tco]
                        for d in range(len(tgt_list)*self.dep_out):
                            mrg_out[ri:ro,ci:co,d]+=(msk[...,d]*mask_wt)[tri:tro,tci:tco]
                        mrg_out_wt[ri:ro,ci:co]+=mask_wt[tri:tro,tci:tco]
                if self.separate:
                    for d in range(len(tgt_list)*self.dep_out):
                        mrg_out[...,d]/=mrg_out_wt
                    print(grp); text_list=[grp]
                    merge_name=view[0].image_name
                    merge_file=os.path.join(merge_dir,merge_name)
                    blend,r_g=self.predict_proc(self.net,mrg_in,mrg_out,merge_file.replace(img_ext,''))
                    for d in range(len(tgt_list)):
                        text="[  %d: %s] #%d $%d / $%d  %.2f%%"%(d,tgt_list[d],r_g[d][1],r_g[d][0],sum_g,100.*r_g[d][0]/sum_g)
                        print(text); text_list.append(text)
                    blendtext=draw_text(self.net,blend,text_list,ra)  # RGB: 3x8-bit dark text
                    cv2.imwrite(merge_file,blendtext)  # [...,np.newaxis]
                    res_g=r_g[np.newaxis,...] if res_g is None else np.concatenate((res_g,r_g[np.newaxis,...]))
            res_ind=res_i if res_ind is None else np.hstack((res_ind,res_i))
            res_grp=res_g if res_grp is None else np.hstack((res_grp,res_g))
        for i,note in [(0,'_area'),(1,'_count')]:
            df=pd.DataFrame(res_ind[...,i],index=pair.img_set.images,columns=pair.targets*pair.cfg.dep_out)
            to_excel_sheet(df,xls_file,pair.origin+note)  # per slice
        if self.separate:
            for i,note in [(0,'_area'),(1,'_count')]:
                df=pd.DataFrame(res_grp[...,i],index=batch.keys(),columns=pair.targets*pair.cfg.dep_out)
                to_excel_sheet(df,xls_file,pair.origin+note+"_sum")

class ImageNoisePair:
    def __init__(self,cfg:Config,wd,origin,targets,is_train):
        premade=mkdir_ifexist(os.path.join(wd, '+'.join([origin]+targets))) # e.g., Original+LYM+MONO+PMN
        print(premade)
        for tgt in targets:
            premade=mkdir_ifexist(os.path.join(wd, tgt+'+')) and premade # force mkdir e.g., MONO+
        print(premade)
        if not premade:
            # super(ImageNoisePair,self).__init__(cfg,wd,origin,targets,is_train) # split original image
            self.bright_diff=-10  # local brightness should be more than noise patch brightness,
            self.aj_size=2
            self.aj_std=0.2
            self.msk_set=[]
            tgt_set=[NoiseSet(cfg, wd, tgt, is_train, is_image=True) for tgt in targets]
            pixels=cfg.row_in*cfg.col_in
            for vi, vc in enumerate(self.img_set.view_coord):
                rand_num=[(random.randint(0,len(targets)-1), random.random(), random.uniform(0, 1), random.uniform(0, 1))
                          for r in range(random.randint(pixels//5000, pixels//2000))]  # index,label/class,row,col
                img=vc.get_image(os.path.join(self.img_set.work_directory, self.img_set.sub_folder), self.cfg)
                lg_row, lg_col, lg_dep=img.shape
                # cv2.imwrite(os.path.join(tgt_noise.work_directory,tgt_noise.sub_folder,'_'+vc.image_name),img)
                inserted=[0]*len(self.targets) # track # of inserts per category
                vcfilenoext=vc.file_name_insert(cfg)
                mkdir_ifexist(os.path.join(self.wd,'+'.join([origin]+targets),vcfilenoext))
                mkdir_ifexist(os.path.join(self.wd,'+'.join([origin]+targets),vcfilenoext,'images'))
                for tgt in targets:
                    mkdir_ifexist(os.path.join(self.wd,'+'.join([origin]+targets),vcfilenoext,tgt))
                for lirc in rand_num:
                    the_tgt=tgt_set[lirc[0]]
                    prev=img.copy()
                    idx=int(the_tgt.num_patches*lirc[1])  # index of patch to apply
                    patch=the_tgt.view_coord[idx]
                    p_row, p_col, p_ave, p_std=patch.ori_row, patch.ori_col, patch.row_start, patch.row_end
                    lri=int(lg_row*lirc[2])-p_row//2  # large row in/start
                    lci=int(lg_col*lirc[3])-p_col//2  # large col in/start
                    lro, lco=lri+p_row, lci+p_col  # large row/col out/end
                    pri=0 if lri>=0 else -lri; lri=max(0, lri)
                    pci=0 if lci>=0 else -lci; lci=max(0, lci)
                    pro=p_row if lro<=lg_row else p_row-lro+lg_row; lro=min(lg_row, lro)
                    pco=p_col if lco<=lg_col else p_col-lco+lg_col; lco=min(lg_col, lco)
                    # if np.average(img[lri:lro,lci:lco])-p_ave > self.bright_diff and \
                    if np.min(img[lri:lro, lci:lco])-p_ave>self.bright_diff and \
                            int(np.std(img[lri-p_row*self.aj_size:lro+p_row*self.aj_size,
                                       lci-p_col*self.aj_size:lco+p_col*self.aj_size])>self.aj_std*p_std):  # target area is brighter, then add patch
                        # print("large row(%d) %d-%d col(%d) %d-%d  patch row(%d) %d-%d col(%d) %d-%d"%(lg_row,lri,lro,lg_col,lci,lco,p_row,pri,pro,p_col,pci,pco))
                        # pat=patch.get_image(os.path.join(self.wd, the_tgt.sub_folder),self.cfg)  # TODO 40X-40X resize=1.0
                        pat=the_tgt.patches[idx]
                        if random.random()>0.5: pat=np.fliplr(pat)
                        if random.random()>0.5: pat=np.flipud(pat)
                        img[lri:lro, lci:lco]=np.minimum(img[lri:lro, lci:lco], pat[pri:pro, pci:pco])
                        # cv2.imwrite(os.path.join(self.wd, the_tgt.sub_folder+'+',vc.file_name_insert(cfg,'_'+str(idx)+('' if lirc[1]>self.cfg.train_vali_split else '^'))),
                        #             smooth_brighten(prev-img))
                        cv2.imwrite(os.path.join(self.wd, '+'.join([origin]+targets), vcfilenoext, the_tgt.sub_folder,
                                                 vc.file_name_insert(cfg,'_'+str(idx))), #+('' if lirc[1]>self.cfg.train_vali_split else '^'))
                                    smooth_brighten(prev-img))
                        # lr=(lri+lro)//2
                        # lc=(lci+lco)//2
                        # msk[lr:lr+1,lc:lc+1,1]=255
                        inserted[lirc[0]]+=1
                print("inserted %s for %s"%(inserted,vc.file_name))
                cv2.imwrite(os.path.join(self.wd, '+'.join([origin]+targets), vcfilenoext, 'images', vc.file_name), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        super(ImageNoisePair, self).__init__(cfg, wd, '+'.join([origin]+targets), [tgt+'+' for tgt in self.targets], is_train)

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
            self.view_coord = list(views) # TODO refactor split method
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

    def predict_generator(self):
        # yield (ImageGenerator(self, False, self.targets, self.view_coord),self.join_targets(self.targets), self.targets)
        i = 0; nt = len(self.targets)
        ps = self.cfg.predict_size
        while i < nt:
            o = min(i + ps, nt)
            tgt_list=self.targets[i:o]
            yield (self.join_targets(tgt_list),tgt_list)
            i = o



class ImageMasksGenerator(keras.utils.Sequence):
    def __init__(self,pair:ImageMasksPair,aug_value,tgt_list,view_coord=None):
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
