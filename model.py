import os

from datetime import datetime
import numpy as np
import pandas as pd
from PIL import ImageDraw, Image, ImageFont
from cv2.cv2 import imwrite,connectedComponents,connectedComponentsWithStats,CV_32S,merge,cvtColor,COLOR_HSV2BGR

from image_gen import ImageMaskPair, ImageGenerator
from net.basenet import Net
from process_image import reverse_sigmoid,scale_sigmoid
from util import mk_dir_if_nonexist, to_excel_sheet

def g_kern(size, sigma):
    from scipy import signal
    gkern1d = signal.gaussian(size, std=sigma).reshape(size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def g_kern_rect(row, col, rel_sig=0.5):
    l=max(row,col)
    mat=g_kern(l, int(rel_sig * l))
    r0, c0=int(0.5*(l-row)),int(0.5*(l-col))
    return mat[r0:r0+row,c0:c0+col]

def blur(a):
    kernel = np.array([[1.0,2.0,1.0], [2.0,4.0,2.0], [1.0,2.0,1.0]])
    kernel = kernel / np.sum(kernel)
    arraylist = []
    for y in range(3):
        temparray = np.copy(a)
        temparray = np.roll(temparray, y - 1, axis=0)
        for x in range(3):
            temparray_X = np.copy(temparray)
            temparray_X = np.roll(temparray_X, x - 1, axis=1)*kernel[y,x]
            arraylist.append(temparray_X)
    arraylist = np.array(arraylist)
    arraylist_sum = np.sum(arraylist, axis=0)
    return arraylist_sum

def single_call(cfg,img,msk,file=None):  # sigmoid (r,c,1) blend, np result
    blend=img.copy()
    res=None
    # TODO apply blur
    msk=np.rint(msk)  # sigmoid round to  0/1 # consider range(-1 ~ +1) for multi class voting
    for d in range(msk.shape[-1]):
        curr=msk[...,d][...,np.newaxis].astype(np.uint8)
        count, labels=connectedComponents(curr,connectivity=8,ltype=CV_32S)
        newres=np.array([np.sum(curr,keepdims=False),count])
        res=newres if res is None else np.concatenate([res,newres],axis=-1)
        for c in range(3):
            blend[...,c]=np.where(msk[...,d]>=0.5,blend[...,c]*(1-cfg.overlay_opacity)+cfg.overlay_color[d][c]*cfg.overlay_opacity,blend[...,c])  # weighted average
        if file is not None:
            # output=connectedComponentsWithStats(msk[...,d],connectivity=8,ltype=CV_32S)
            label_hue=np.uint8(179*labels/np.max(labels)) # Map component labels to hue val
            blank_ch=255*np.ones_like(label_hue)
            labeled_img=merge([label_hue,blank_ch,blank_ch])
            labeled_img=cvtColor(labeled_img,COLOR_HSV2BGR) # cvt to BGR for display
            labeled_img[label_hue==0]=0 # set bg label to black
            imwrite(file+'_%d.png'%d,labeled_img)
    return blend, res
    # return blend, np.sum(msk, keepdims=True)

def multi_call(cfg,img,msk,file=None):  # softmax (r,c,multi_label) blend, np result
    blend=img.copy()
    dim=cfg.predict_size  # do argmax if predict categories covers all possibilities or consider them individually
    msk=np.argmax(msk,axis=-1)
    uni,count=np.unique(msk,return_counts=True)
    map_count=dict(zip(uni,count))
    count_vec=np.zeros(dim)
    for d in range(dim):
        count_vec[d]=map_count.get(d) or 0
        for c in range(3):
            blend[...,c]=np.where(msk==d,blend[...,c]*(1-cfg.overlay_opacity)+cfg.overlay_color[d][c]*cfg.overlay_opacity,blend[...,c])
    return blend,count_vec

def compare_call(cfg,img,msk,file=None):  # compare input and output with same dimension
    diff=np.round(np.abs(msk-scale_sigmoid(img)))
    # b=np.sum(diff,axis=(0,-1),keepdims=False)
    # c=np.sum(diff,axis=-1,keepdims=False)
    # imsave("test.jpg",reverse_sigmoid(msk))
    return reverse_sigmoid(msk), np.sum(diff,axis=(0,1),keepdims=False)

def draw_text(cfg,img,text_list,width):
    font="arial.ttf"  #times.ttf
    size=round(0.33*(26+0.03*width+width/len(max(text_list,key=len))))
    txt_col=(10,10,10)
    origin=Image.fromarray(img.astype(np.uint8),'RGB')  # L RGB
    draw=ImageDraw.Draw(origin)
    draw.text((0,0),'\n'.join(text_list),txt_col,ImageFont.truetype(font,size))
    for i in range(len(text_list)-1):
        sym_col=cfg.overlay_color[i]
        draw.text((0,0),' \n'*(i+1)+' X',sym_col,ImageFont.truetype(font,size))
    return np.array(origin)

class Model:
    def __init__(self, net:Net, save_net=False):
        self.net=net if isinstance(net, Net) else Net.from_json(net)
        self.net.compile_net()
        if save_net:
            self.net.save_net()

    def __str__(self):
        return str(self.net)
    def __repr__(self):
        return str(self.net)+self.net.predict_proc.__name__[0:1].upper()

    def train(self,multi:ImageMaskPair):
        for tr, val, dir_out in multi.train_generator():
            export_name = dir_out +'_'+str(self)
            weight_file = export_name + ".h5"
            if self.net.train_continue and os.path.exists(weight_file):
                print("Continue from previous weights")
                self.net.net.load_weights(weight_file)
                # print("Continue from previous model with weights & optimizer")
                # self.model=load_model(weight_file,custom_objects=custom_function_dict()) # does not work well with custom act, loss func
            print('Fitting neural net...')
            for r in range(self.net.train_rep):
                print("Training %d/%d for %s" % (r + 1, self.net.train_rep, export_name))
                tr.on_epoch_end()
                val.on_epoch_end()
                from keras.callbacks import ModelCheckpoint, EarlyStopping
                history = self.net.net.fit_generator(tr, validation_data=val, verbose=1,
                                                 steps_per_epoch=min(self.net.train_step, len(tr.view_coord)) if isinstance(self.net.train_step , int) else len(tr.view_coord),
                                                 validation_steps=min(self.net.train_vali_step, len(val.view_coord)) if isinstance(self.net.train_vali_step, int) else len(val.view_coord),
                                                 epochs=self.net.train_epoch, max_queue_size=1, workers=0, use_multiprocessing=False, shuffle=False,
                                                 callbacks=[
                        ModelCheckpoint(weight_file, monitor=self.net.indicator, mode='max', save_weights_only=False, save_best_only=True),
                        EarlyStopping(monitor=self.net.indicator, mode='max', patience=1, verbose=1),
                        # ReduceLROnPlateau(monitor=train_indicator, mode='max', factor=0.1, patience=10, min_delta=1e-5, cooldown=0, min_lr=0, verbose=1),
                        # TensorBoardTrainVal(log_dir=os.path.join("log", export_name), write_graph=True, write_grads=False, write_images=True),
                    ]).history
                if not os.path.exists(export_name + ".txt"):
                    with open(export_name + ".txt", "w") as net_summary:
                        self.net.net.summary(print_fn=lambda x: net_summary.write(x+'\n'))
                df=pd.DataFrame(history).round(4)
                df['time']=datetime.now().strftime("%Y-%m-%d %H:%M")
                df['repeat']=r+1
                df.to_csv(export_name + ".csv", mode="a", header=(not os.path.exists(export_name + ".csv")))

    def predict(self,multi:ImageMaskPair,pred_dir):
        xls_file="Result_%s_%s.xlsx"%(pred_dir, repr(self))
        img_ext=self.net.image_format[1:] # *.jpg -> .jpg
        sum_i, sum_g = self.net.row_out * self.net.col_out, None
        msks, mask_wt, r_i, r_g,  ra, ca= None, None, None, None, None, None
        mrg_in, mrg_out, mrg_out_wt, merge_dir = None, None, None, None
        batch=multi.img_set.view_coord_batch()  # image/1batch -> view_coord
        dir_ex=multi.dir_out_ex()
        dir_cfg_append=str(self) if dir_ex is None else dir_ex+'_'+str(self)
        res_ind, res_grp=None, None
        save_ind_image=False
        for dir_out, tgt_list in multi.predict_generator():
            res_i, res_g=None, None
            print('Load model and predict to [%s]...'%dir_out)
            export_name = dir_out+'_'+dir_cfg_append
            target_dir = os.path.join(multi.wd, export_name)
            if save_ind_image or not self.net.separate: # skip saving individual images
                mk_dir_if_nonexist(target_dir)
            if self.net.separate:
                merge_dir = os.path.join(multi.wd, dir_out+'+'+dir_cfg_append) # group
                mk_dir_if_nonexist(merge_dir)
                mask_wt = g_kern_rect(self.net.row_out, self.net.col_out)
            for grp, view in batch.items():
                msks=None
                i=0; nt=len(tgt_list)
                while i < nt:
                    o=min(i+self.net.dep_out, nt)
                    tgt_sub=tgt_list[i:o]
                    tgt_name=ImageMaskPair.join_targets(tgt_sub)
                    prd=ImageGenerator(multi, 0, tgt_sub, view)
                    weight_file=tgt_name+'_'+dir_cfg_append+'.h5'
                    print(weight_file)
                    self.net.net.load_weights(weight_file) # weights only
                    # self.net.net=load_model(weight_file,custom_objects=custom_function_dict()) # weight optimizer archtecture
                    msk=self.net.net.predict_generator(prd, max_queue_size=1, workers=0, use_multiprocessing=False, verbose=1)
                    msks = msk if msks is None else  np.concatenate((msks, msk),axis=-1)
                    i=o
                print('Saving predicted results [%s] to folder [%s]...' % (grp, export_name))
                # r_i=np.zeros((len(multi.img_set.images),len(tgt_list)), dtype=np.uint32)
                if self.net.separate:
                    mrg_in = np.zeros((view[0].ori_row, view[0].ori_col, self.net.dep_in), dtype=np.float32)
                    mrg_out = np.zeros((view[0].ori_row, view[0].ori_col, len(tgt_list)*self.net.dep_out), dtype=np.float32)
                    mrg_out_wt = np.zeros((view[0].ori_row, view[0].ori_col), dtype=np.float32) + np.finfo(np.float32).eps
                    sum_g = view[0].ori_row * view[0].ori_col
                    # r_g=np.zeros((1,len(tgt_list)*self.net.dep_out), dtype=np.uint32)
                for i, msk in enumerate(msks):
                    # if i>=len(multi.view_coord): print("skip %d for overrange"%i); break # last batch may have unused entries
                    ind_name = view[i].file_name
                    ind_file = os.path.join(target_dir, ind_name)
                    origin = view[i].get_image(os.path.join(multi.wd, multi.dir_in_ex()), self.net)
                    print(ind_name); text_list = [ind_name]
                    blend, r_i=self.net.predict_proc(self.net, origin, msk, ind_file.replace(img_ext,''))
                    for d in range(len(tgt_list)):
                        text = "[  %d: %s] %d / %d  %.2f%%" % (d, tgt_list[d], r_i[d], sum_i, 100. * r_i[d] / sum_i)
                        print(text); text_list.append(text)
                    if save_ind_image or not self.net.separate: # skip saving individual images
                        blendtext = draw_text(self.net, blend, text_list, self.net.row_out) # RGB:3x8-bit dark text
                        imwrite(ind_file, blendtext)
                    res_i =r_i[np.newaxis,...] if res_i is None else np.concatenate((res_i, r_i[np.newaxis,...]))

                    if self.net.separate:
                        ri,ro=view[i].row_start, view[i].row_end
                        ci,co=view[i].col_start, view[i].col_end
                        ra,ca=view[i].ori_row,view[i].ori_col
                        tri, tro = 0, self.net.row_out
                        tci, tco = 0, self.net.col_out
                        if ri<0: tri=-ri; ri=0
                        if ci<0: tci=-ci; ci=0
                        if ro>ra: tro=tro-(ro-ra); ro=ra
                        if co>ca: tco=tco-(co-ca); co=ca
                        mrg_in[ri:ro,ci:co] = origin[tri:tro,tci:tco]
                        for d in range(len(tgt_list)*self.net.dep_out):
                            mrg_out[ri:ro,ci:co,d] += (msk[...,d] * mask_wt)[tri:tro,tci:tco]
                        mrg_out_wt[ri:ro,ci:co] += mask_wt[tri:tro,tci:tco]
                if self.net.separate:
                    for d in range(len(tgt_list)*self.net.dep_out):
                        mrg_out[...,d] /= mrg_out_wt
                    print(grp); text_list=[grp]
                    merge_name = view[0].image_name
                    merge_file = os.path.join(merge_dir, merge_name)
                    blend, r_g = self.net.predict_proc(self.net, mrg_in, mrg_out, merge_file.replace(img_ext,''))
                    for d in range(len(tgt_list)):
                        text = "[  %d: %s] %d / %d  %.2f%%" % (d, tgt_list[d], r_g[d], sum_g, 100. * r_g[d] / sum_g)
                        print(text); text_list.append(text)
                    blendtext = draw_text(self.net, blend, text_list, ra) # RGB: 3x8-bit dark text
                    imwrite(merge_file, blendtext) # [...,np.newaxis]
                    res_g=r_g[np.newaxis,...] if res_g is None else np.concatenate((res_g, r_g[np.newaxis,...]))
            res_ind=res_i if res_ind is None else np.hstack((res_ind, res_i))
            res_grp=res_g if res_grp is None else np.hstack((res_grp, res_g))
        for i,note in [(0,'_area'),(1,'_count')]:
            df = pd.DataFrame(res_ind[...,i], index=multi.img_set.images, columns=multi.targets*multi.cfg.dep_out)
            to_excel_sheet(df, xls_file, multi.origin+note)  # per slice
        if self.net.separate:
            for i,note in [(0,'_area'),(1,'_count')]:
                df = pd.DataFrame(res_grp[...,i], index=batch.keys(), columns=multi.targets*multi.cfg.dep_out)
                to_excel_sheet(df, xls_file, multi.origin+note+"_sum")
