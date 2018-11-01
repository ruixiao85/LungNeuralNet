import datetime

from keras.engine.saving import model_from_json,load_model

from metrics import custom_function_dict
from net.basecfg import Config
import os, cv2
import pandas as pd
import numpy as np

from osio import mkdir_ifexist,to_excel_sheet
from util import g_kern_rect,draw_text


class Net(Config):
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
    def __init__(self, dim_in=None, dim_out=None, feed=None, act=None, out=None, loss=None, metrics=None, optimizer=None, indicator=None,
                 filename=None, **kwargs):

        super(Net,self).__init__(**kwargs)
        self.row_in, self.col_in, self.dep_in=dim_in or (512,512,3)
        self.row_out, self.col_out, self.dep_out=dim_out or (512,512,1)
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
        self.net.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics)
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
                                                   steps_per_epoch=min(self.train_step,len(tr.view_coord)) if isinstance(self.train_step,int) else len(
                                                       tr.view_coord),
                                                   validation_steps=min(self.train_vali_step,len(val.view_coord)) if isinstance(self.train_vali_step,
                                                                                                                                    int) else len(
                                                       val.view_coord),
                                                   epochs=self.train_epoch,max_queue_size=1,workers=0,use_multiprocessing=False,shuffle=False,
                                                   callbacks=[
                                                       ModelCheckpoint(weight_file,monitor=self.indicator,mode='max',save_weights_only=False,
                                                                       save_best_only=True),
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
                        print(text);
                        text_list.append(text)
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
