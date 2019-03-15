import datetime
import random

import keras
from keras.engine.saving import model_from_json,load_model

from a_config import Config
from image_set import ImageSet
from metrics import custom_function_dict
import os, cv2
import pandas as pd
import numpy as np

from osio import mkdir_ifexist,to_excel_sheet
from preprocess import augment_image_pair,prep_scale,read_image
from postprocess import g_kern_rect,draw_text

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
    def __init__(self,loss=None, metrics=None, learning_rate=None, learning_continue=None, optimizer=None, indicator=None, indicator_trend=None, predict_proc=None,
                 filename=None, **kwargs):
        super(BaseNetU,self).__init__(**kwargs)
        from metrics import jac, dice, dice67, dice33, acc, acc67, acc33, loss_bce_dice, custom_function_keras
        custom_function_keras()  # leakyrelu, swish
        self.loss=loss or (
            loss_bce_dice if self.dep_out==1 else 'categorical_crossentropy')  # 'binary_crossentropy'
        self.metrics=metrics or ([jac, dice] if self.dep_out==1 else [acc]) # dice67,dice33  acc67,acc33
        self.learning_rate=learning_rate or 1e-5
        self.learning_continue=learning_continue or 2e-1
        from keras.optimizers import Adam
        self.optimizer=optimizer or Adam
        self.indicator=indicator if indicator is not None else ('val_dice' if self.dep_out==1 else 'val_acc')
        self.indicator_trend=indicator_trend or 'max'
        from postprocess import single_call,multi_call
        self.predict_proc=predict_proc if predict_proc is not None else single_call
        self.net=None # abstract -> instatiate in subclass
        self.filename=filename

    def load_json(self,filename=None):  # load model from json
        if filename is not None:
            self.filename=filename
        with open(filename+".json", 'r') as json_file:
            self.net=model_from_json(json_file.read())

    def save_net(self):
        json_net=(self.filename if self.filename is not None else str(self)) + ".json"
        with open(json_net, "w") as json_file:
            json_file.write(self.net.to_json())

    def build_net(self):
        pass

    def compile_net(self,save_net=False,print_summary=True):
        self.net.compile(optimizer=self.optimizer(self.learning_rate), loss=self.loss, metrics=self.metrics)
        print("Model compiled")
        if save_net:
            self.save_net()
            print('Model saved to file')
        if print_summary:
            self.net.summary()

    def __str__(self):
        return '_'.join([
            type(self).__name__,
            self.cap_lim_join(4, self.feed, self.act, self.out,
                              (self.loss if isinstance(self.loss, str) else self.loss.__name__).
                              replace('_', '').replace('loss', ''))
            + str(self.dep_out)])
    def __repr__(self):
        return str(self)+self.predict_proc.__name__[0:1].upper()

    @staticmethod
    def cap_lim_join(lim,*text):
        test_list=[t.capitalize()[:lim] for t in text]
        return ''.join(test_list)

    def train(self,pair):
        self.build_net()
        self.compile_net() # recompile to set optimizers,..
        for tr,val,dir_out in pair.train_generator():
            self.filename=dir_out+'_'+str(self)
            weight_file="%s^{%s:.2f}^.h5"%(self.filename,self.indicator) # e.g., {epoch:02d}-{val_acc:.2f}
            print('Fitting neural net...')
            for r in range(self.train_rep):
                best_val,learning_rate=None,self.learning_rate # store last best, init lr and reduce if continuing
                if self.train_continue:
                    last_saves=self.find_best_models(self.filename+'^*^.h5')
                    if isinstance(last_saves,list) and len(last_saves)>0:
                        last_best=last_saves[0]
                        best_val=float(last_best.split('^')[1])
                        print("Continue from previous weights")
                        self.net.load_weights(last_best)
                        # print("Continue from previous model with weights & optimizer")
                        # self.net=load_model(last_best,custom_objects=custom_function_dict())  # does not work well with custom act, loss func
                        learning_rate*=self.learning_continue
                        print('Lowered learning rate (%f -> %f) for the continued training'%(self.learning_rate,learning_rate))
                if not os.path.exists(self.filename+".txt"):
                    with open(self.filename+".txt","w") as net_summary:
                        self.net.summary(print_fn=lambda x:net_summary.write(x+'\n'))
                if not os.path.exists(self.filename+".json"):
                    self.save_net()
                print("Training %d/%d for %s"%(r+1,self.train_rep,self.filename))
                from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,LearningRateScheduler
                from callbacks import TensorBoardTrainVal,ModelCheckpointCustom
                history=self.net.fit_generator(tr,validation_data=val,verbose=1,
                   steps_per_epoch=min(self.train_step,len(tr.view_coord)) if isinstance(self.train_step,int) else len(tr.view_coord),
                   validation_steps=min(self.train_vali_step,len(val.view_coord)) if isinstance(self.train_vali_step,int) else len(val.view_coord),
                   epochs=self.train_epoch,max_queue_size=1,workers=0,use_multiprocessing=False,shuffle=False,
                   callbacks=[
                       ModelCheckpointCustom(weight_file,monitor=self.indicator,mode=self.indicator_trend,
                                             best=best_val,save_weights_only=False,save_best_only=True,verbose=1),
                       EarlyStopping(monitor=self.indicator,mode=self.indicator_trend,patience=1,verbose=1),
                       LearningRateScheduler(lambda x: learning_rate*(0.1**(0.2*x)),verbose=1),
                       # ReduceLROnPlateau(monitor=self.indicator, mode='max', factor=0.5, patience=1, min_delta=1e-8, cooldown=0, min_lr=0, verbose=1),
                       # TensorBoardTrainVal(log_dir=os.path.join("log", self.filename), write_graph=True, write_grads=False, write_images=True),
                   ]).history
                df=pd.DataFrame(history)
                df['time']=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                df['repeat']=r+1
                df.to_csv(self.filename+".csv",mode="a",header=(not os.path.exists(self.filename+".csv")))

    def predict(self,pair,pred_dir):
        self.build_net()
        xls_file="Result_%s_%s.xlsx"%(pred_dir.replace(os.sep,'_'),repr(self))
        sum_i,sum_g=self.row_out*self.col_out,None
        batch=pair.img_set.view_coord_batch()  # image/1batch -> view_coord
        dir_cfg_append=pair.img_set.category_append(None,self.row_out,self.col_out)+'_'+str(self)
        save_ind,save_raw=pair.cfg.save_ind_raw
        res_ind,res_grp=None,None
        for dir_out,tgt_list in pair.predict_generator_note():
            res_i,res_g=None,None
            print('Load model and predict to [%s]...'%dir_out)
            export_name=dir_out+'_'+dir_cfg_append
            target_dir=os.path.join(pair.wd,export_name); mkdir_ifexist(target_dir) # folder to save individual slices
            merge_dir=os.path.join(pair.wd,dir_out+'+'+dir_cfg_append); mkdir_ifexist(merge_dir) # folder to save grouped/whole images
            mask_wt=g_kern_rect(self.row_out,self.col_out)
            for grp,view in batch.items():
                msks=None
                i=0; nt=len(tgt_list)
                while i<nt:
                    o=min(i+self.dep_out,nt)
                    tgt_sub=tgt_list[i:o]
                    prd,tgt_name=pair.predict_generator_partial(tgt_sub,view)
                    weight_file=self.find_best_models(tgt_name+'_'+dir_cfg_append+'^*^.h5',allow_cache=True)[0]
                    # weight_file_overall=self.find_best_models(tgt_name+'_*^.h5',allow_cache=True)[0]
                    # if weight_file_overall!=weight_file: # load best regardless of archtecture
                    #     self.load_json(weight_file_overall)
                    print(weight_file)
                    self.net.load_weights(weight_file)  # weights only
                    # self.net=load_model(weight_file,custom_objects=custom_function_dict()) # weight optimizer archtecture
                    msk=self.net.predict_generator(prd,max_queue_size=1,workers=0,use_multiprocessing=False,verbose=1)
                    msks=msk if msks is None else np.concatenate((msks,msk),axis=-1)
                    i=o
                print('Saving predicted results [%s] to folder [%s]...'%(grp,export_name))
                # r_i=np.zeros((len(multi.img_set.images),len(tgt_list)), dtype=np.uint32)
                mrg_in=np.zeros((view[0].ori_row,view[0].ori_col,self.dep_in),dtype=np.float32)
                mrg_out=np.zeros((view[0].ori_row,view[0].ori_col,len(tgt_list)*self.dep_out),dtype=np.float32)
                mrg_out_wt=np.zeros((view[0].ori_row,view[0].ori_col),dtype=np.float32)+np.finfo(np.float32).eps
                sum_g=view[0].ori_row*view[0].ori_col
                # r_g=np.zeros((1,len(tgt_list)*self.dep_out), dtype=np.uint32)
                for i,msk in enumerate(msks):
                    # if i>=len(multi.view_coord): print("skip %d for overrange"%i); break # last batch may have unused entries
                    ind_name=view[i].file_name
                    ind_file=os.path.join(target_dir,ind_name)
                    origin=view[i].get_image(os.path.join(pair.wd,pair.img_set.target_folder))
                    print(ind_name); text_list=[ind_name]
                    blend,r_i=self.predict_proc(self,origin,msk,file=None) # ind_file.replace(self.image_format[1:],'')
                    for d in range(len(tgt_list)):
                        text="[  %d: %s] #%d $%d / $%d  %.2f%%"%(d,tgt_list[d],r_i[d][1],r_i[d][0],sum_i,100.*r_i[d][0]/sum_i)
                        print(text); text_list.append(text)
                    blendtext=draw_text(self,blend,text_list,self.col_out)  # RGB:3x8-bit dark text
                    if save_ind:
                        cv2.imwrite(ind_file,blendtext)
                    res_i=r_i[np.newaxis,...] if res_i is None else np.concatenate((res_i,r_i[np.newaxis,...]))

                    ri,ro,ci,co,tri,tro,tci,tco=self.get_proper_range(view[i].ori_row,view[i].ori_col,
                            view[i].row_start,view[i].row_end,view[i].col_start,view[i].col_end,  0,self.row_out,0,self.col_out)
                    mrg_in[ri:ro,ci:co]=origin[tri:tro,tci:tco]
                    for d in range(len(tgt_list)):
                        mrg_out[ri:ro,ci:co,d]+=(msk[...,d]*mask_wt)[tri:tro,tci:tco]
                    mrg_out_wt[ri:ro,ci:co]+=mask_wt[tri:tro,tci:tco]
                for d in range(len(tgt_list)):
                    mrg_out[...,d]/=mrg_out_wt
                print(grp); text_list=[grp]
                merge_name=view[0].image_name
                merge_file=os.path.join(merge_dir,merge_name)
                if save_raw and pair.img_set.resize_ratio<1.0: # high-res raw group image
                    mrg_in=read_image(os.path.join(pair.wd,pair.img_set.raw_folder,view[0].image_name))
                    mrg_r, mrg_c, _=mrg_in.shape
                    mrg_out=cv2.resize(mrg_out, (mrg_c,mrg_r))
                    sum_g=mrg_r*mrg_c
                blend,r_g=self.predict_proc(self,mrg_in,mrg_out,file=None) # merge_file.replace(self.image_format[1:],'')
                for d in range(len(tgt_list)):
                    text="[  %d: %s] #%d $%d / $%d  %.2f%%"%(d,tgt_list[d],r_g[d][1],r_g[d][0],sum_g,100.*r_g[d][0]/sum_g)
                    print(text); text_list.append(text)
                blendtext=draw_text(self,blend,text_list,view[0].ori_col)  # RGB: 3x8-bit dark text
                res_g=r_g[np.newaxis,...] if res_g is None else np.concatenate((res_g,r_g[np.newaxis,...]))
                cv2.imwrite(merge_file,blendtext)  # [...,np.newaxis]
            res_ind=res_i if res_ind is None else np.hstack((res_ind,res_i))
            res_grp=res_g if res_grp is None else np.hstack((res_grp,res_g))
        for i,note in [(0,'_area'),(1,'_count')]:
            df=pd.DataFrame(res_ind[...,i],index=pair.img_set.images,columns=pair.targets*pair.cfg.dep_out)
            to_excel_sheet(df,xls_file,pair.origin+note)  # per slice
        for i,note in [(0,'_area'),(1,'_count')]:
            df=pd.DataFrame(res_grp[...,i],index=batch.keys(),columns=pair.targets*pair.cfg.dep_out)
            to_excel_sheet(df,xls_file,pair.origin+note+"_sum")


class ImageMaskPair:
    def __init__(self,cfg:BaseNetU,wd,origin,targets,is_train):
        self.cfg=cfg
        self.wd=wd
        self.origin=origin
        self.targets=targets if isinstance(targets,list) else [targets]
        self.img_set=ImageSet(cfg, wd, origin, is_train, is_image=True).prep_folder()
        self.msk_set=None
        self.is_train=is_train

    def train_generator(self):
        i = 0; no=self.cfg.dep_out; nt=len(self.targets)
        while i < nt:
            o=min(i+no, nt)
            views = set(self.img_set.view_coord)
            self.msk_set = []
            tgt_list=[]
            for t in self.targets[i:o]:
                tgt_list.append(t)
                msk = ImageSet(self.cfg, self.wd, t, is_train=True, is_image=False).prep_folder()
                self.msk_set.append(msk)
                views = views.intersection(msk.view_coord)
            self.img_set.view_coord = sorted(views,key=lambda x:x.file_name)
            tr_list,val_list=self.cfg.split_train_val_vc(self.img_set.view_coord)
            yield(ImageMaskGenerator(self,self.cfg.train_aug,tgt_list,tr_list),ImageMaskGenerator(self,0,tgt_list,val_list),
                    self.img_set.category_detail(self.cfg.join_targets(tgt_list),self.cfg.target_scale,self.cfg.row_out,self.cfg.col_out))
            i=o

    def predict_generator_note(self):
        i = 0; nt = len(self.targets)
        while i < nt:
            o = min(i + self.cfg.predict_size, nt)
            tgt_list=self.targets[i:o]
            yield (self.cfg.join_targets(tgt_list), tgt_list)
            i = o

    def predict_generator_partial(self,subset,view):
        return ImageMaskGenerator(self,0,subset,view),self.cfg.join_targets(subset)


class ImageMaskGenerator(keras.utils.Sequence):
    def __init__(self,pair:ImageMaskPair,aug_value,tgt_list,view_coord):
        self.pair=pair
        self.cfg=pair.cfg
        self.aug_value=aug_value
        self.target_list=tgt_list
        self.view_coord=pair.img_set.view_coord if view_coord is None else view_coord
        self.indexes=None
        self.on_epoch_end()

    def __len__(self):  # Denotes the number of batches per epoch
        return int(np.ceil(len(self.view_coord) / self.cfg.batch_size))

    def __getitem__(self, index):  # Generate one batch of data
        indexes = self.indexes[index * self.cfg.batch_size:(index + 1) * self.cfg.batch_size]
        # print(" getting index %d with %d batch size"%(index,self.batch_size))
        if self.pair.is_train:
            _img = np.zeros((self.cfg.batch_size, self.cfg.row_in, self.cfg.col_in, self.cfg.dep_in), dtype=np.uint8)
            _tgt = np.zeros((self.cfg.batch_size, self.cfg.row_out, self.cfg.col_out, self.cfg.dep_out), dtype=np.uint8)
            for vi, vc in enumerate([self.view_coord[k] for k in indexes]):
                _img[vi, ...] = vc.get_image(os.path.join(self.pair.wd, self.pair.img_set.target_folder))
                for ti,tgt in enumerate(self.target_list):
                    _tgt[vi, ..., ti] = vc.get_mask(os.path.join(self.pair.wd, self.pair.msk_set[ti].target_folder))
            _img, _tgt = augment_image_pair(_img, _tgt, self.aug_value)  # integer N: a <= N <= b.
            # cv2.imwrite("pair_img.jpg",_img[0]); cv2.imwrite("pair_tgt.jpg",_tgt[0])
            return prep_scale(_img, self.cfg.feed), prep_scale(_tgt, self.cfg.out)
        else:
            _img = np.zeros((self.cfg.batch_size, self.cfg.row_in, self.cfg.col_in, self.cfg.dep_in), dtype=np.uint8)
            for vi, vc in enumerate([self.view_coord[k] for k in indexes]):
                _img[vi, ...] = vc.get_image(os.path.join(self.pair.wd, self.pair.img_set.target_folder))
            # cv2.imwrite("pair_img.jpg",_img[0])
            return prep_scale(_img, self.cfg.feed), None

    def on_epoch_end(self):  # Updates indexes after each epoch
        self.indexes=np.arange(len(self.view_coord))
        if self.pair.is_train and self.cfg.train_shuffle:
            np.random.shuffle(self.indexes)
