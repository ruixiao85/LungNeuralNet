import os

import numpy as np
from PIL import ImageDraw,Image,ImageFont
import cv2
from preprocess import prep_scale,rev_scale
from math import floor,log,sqrt

def fill_contour(_img):
    contour,hier = cv2.findContours(_img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(_img,[cnt],0,255,-1)
    return _img

def gaussian_smooth(_img,size=5,sigma=None):
    # return cv2.blur(_img,(size,size))
    return cv2.GaussianBlur(_img,(size,size),sigma or size)

def morph_close(_bin,erode=3,dilate=5):
    erode_kernel=np.ones((erode,erode),np.uint8)
    dilate_kernel=np.ones((dilate,dilate),np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(_bin,cv2.MORPH_ERODE,erode_kernel),cv2.MORPH_DILATE,dilate_kernel)

def morph_open(_bin,dilate=3,erode=5):
    dilate_kernel=np.ones((dilate,dilate),np.uint8)
    erode_kernel=np.ones((erode,erode),np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(_bin,cv2.MORPH_DILATE,dilate_kernel),cv2.MORPH_ERODE,erode_kernel)

def g_kern(size, sigma):
    from scipy.signal.windows import gaussian
    gkern1d = gaussian(size, std=sigma).reshape(size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def g_kern_rect(row, col, rel_sig=0.5):
    l=max(row,col)
    mat=g_kern(l, int(rel_sig * l))
    r0, c0=int(0.5*(l-row)),int(0.5*(l-col))
    return mat[r0:r0+row,c0:c0+col]

def cal_count_area_aperc(rc1,sum):
    count,labels=cv2.connectedComponents(rc1,connectivity=8,ltype=cv2.CV_32S)
    sum_pos=np.sum(rc1,keepdims=False)
    newres=np.array([count-1,sum_pos,100.0*sum_pos/sum])
    return newres,labels

def single_call(cfg,img,tgts,msk):  # sigmoid (r,c,1) blend, np result
    res,text_list=None,[]; blend=img.copy()
    row,col,_=img.shape; sum_pixel=row*col
    bw_mask=np.zeros((row,col,len(tgts)))
    msk=np.rint(msk)  # sigmoid round to  0/1 # consider range(-1 ~ +1) for multi class voting
    for d in range(len(tgts)):
        curr=msk[...,d][...,np.newaxis].astype(np.uint8)
        newres,labels=cal_count_area_aperc(curr,sum_pixel)
        text_list.append("[  %d: %s] #%d $%d / $%d  %.2f%%"%(d,tgts[d],newres[0],newres[1],sum_pixel,newres[2]))
        res=newres[np.newaxis,...] if res is None else np.concatenate((res,newres[np.newaxis,...]))
        for c in range(3):
            blend[...,c]=np.where(msk[...,d]>=0.5,blend[...,c]*(1-cfg.overlay_opacity[d])+cfg.overlay_color[d][c]*cfg.overlay_opacity[d],blend[...,c])  # weighted average
            bw_mask[...,d]=np.where(msk[...,d]>=0.5,255,bw_mask[...,d])  # b/w mask maximum object masks
    blend=draw_text(cfg,blend,text_list,col)
    return res,blend,bw_mask

def single_brighten(cfg,img,tgts,msk):  # sigmoid (r,c,1) blend, np result
    res,text_list=None,[]; blend=img.copy()
    row,col,_=img.shape; sum_pixel=row*col
    bw_mask=np.zeros((row,col,len(tgts)))
    msk=np.rint(gaussian_smooth(msk))  # sigmoid round to  0/1 # consider range(-1 ~ +1) for multi class voting
    for d in range(msk.shape[-1]):
        curr=msk[...,d][...,np.newaxis].astype(np.uint8)
        newres,labels=cal_count_area_aperc(curr,sum_pixel)
        text_list.append("[  %d: %s] #%d $%d / $%d  %.2f%%"%(d,tgts[d],newres[0],newres[1],sum_pixel,newres[2]))
        res=newres[np.newaxis,...] if res is None else np.concatenate((res,newres[np.newaxis,...]))
        for c in range(3):
            mskrev=rev_scale(morph_close(msk,6,6),'sigmoid')
            # blend[...,c]=np.where(mskrev[...,d]>=blend[...,c],mskrev[...,d],blend[...,c])  # weighted average
            blend[...,c]=np.maximum(mskrev[...,d],blend[...,c])  # brighter area
            bw_mask[...,d]=np.where(msk[...,d]>=0.5,255,bw_mask[...,d])  # b/w mask maximum object masks
    blend=draw_text(cfg,blend,text_list,col)
    return res,blend,bw_mask

def multi_call(cfg,img,tgts,msk):  # softmax (r,c,multi_label) blend, np result
    res,text_list=None,[]; blend=img.copy()
    row,col,_=img.shape; sum_pixel=row*col
    bw_mask=np.zeros((row,col,len(tgts)))
    msk=np.argmax(msk,axis=-1) # do argmax if predict categories covers all possibilities or consider them individually
    # uni,count=np.unique(msk,return_counts=True)
    # map_count=dict(zip(uni,count))
    # count_vec=np.zeros(cfg.predict_size)
    for d in range(cfg.predict_size):
        # curr=msk[..., d][..., np.newaxis].astype(np.uint8)
        # count_vec[d]=map_count.get(d) or 0
        curr=np.where(msk==d,1,0).astype(np.uint8)
        newres,labels=cal_count_area_aperc(curr,sum_pixel)
        text_list.append("[  %d: %s] #%d $%d / $%d  %.2f%%"%(d,tgts[d],newres[0],newres[1],sum_pixel,newres[2]))
        res=newres[np.newaxis,...] if res is None else np.concatenate((res,newres[np.newaxis,...]))
        for c in range(3):
            blend[...,c]=np.where(msk==d,blend[...,c]*(1-cfg.overlay_opacity[d])+cfg.overlay_color[d][c]*cfg.overlay_opacity[d],blend[...,c])
            bw_mask[...,d]=np.where(msk==d,255,bw_mask[...,d])  # b/w mask maximum object masks
    blend=draw_text(cfg,blend,text_list,col)
    return res,blend,bw_mask

def draw_text(cfg,img,text_list,width):
    black,white,_,_ =cfg.overlay_textshape_bwif
    if black or white:
        font="arial.ttf"  #times.ttf
        size=max(10,int(width/50)) # fontsize at least 12
        off=max(1, size//15)  # text offset
        origin=Image.fromarray(img.astype(np.uint8),'RGB')  # L RGB
        draw=ImageDraw.Draw(origin)
        txtblk='\n'.join(text_list)
        if white: draw.text((0,0),txtblk,(210,210,210),ImageFont.truetype(font,size))
        if black: draw.text((off,off),txtblk,(30,30,30),ImageFont.truetype(font,size))
        for i in range(len(text_list)):
            txtcrs=' \n'*i+' X'
            if white: draw.text((0,0),txtcrs,(210,210,210),ImageFont.truetype(font,size))
            if black: draw.text((off,off),txtcrs,cfg.overlay_color[i],ImageFont.truetype(font,size))
        return np.array(origin)
    else:
        return img


def draw_detection(cfg,image,tgts,box,cls,scr,msk,reg=None):
    font="arial.ttf"  #times.ttf
    ori_row,ori_col,_=image.shape
    size=max(10, ori_col//64) # fontsize at least 10
    lwd=max(3, size//12) # line width
    black, white, instance, fill=cfg.overlay_textshape_bwif
    n=len(box)
    if not n: print("\n*** No instances to display *** \n")
    blend=image.copy(); bw_mask=np.zeros_like(blend)
    nreg,sum_pixels,name_reg=1,[ori_row*ori_col],["Total"] # default only consider whole image
    if reg:
        for (rname,mask) in sorted(reg.items()):  # add more specified regions
            nreg,sum_pixels,name_reg=nreg+1,sum_pixels+[np.sum(mask,keepdims=False)],name_reg+[rname] # green channel as mask
    res=np.zeros((nreg,len(tgts),3),dtype=np.float32) # 0region:total/parenchyma/... 1target:LYM/MONO/PMN/...  2param:count/area/area_pct
    for i in range(n):
        t=cls[i]-1
        y1,x1,y2,x2=box[i]
        mask=np.zeros((ori_row,ori_col),dtype=np.uint8)
        ri,ro,ci,co,tri,tro,tci,tco=cfg.get_proper_range(ori_row,ori_col, y1,y2,x1,x2, 0,y2-y1,0,x2-x1)
        patch=gaussian_smooth(cv2.resize(np.where(msk[i]>=0.5,1,0).astype(np.uint8),(x2-x1,y2-y1),interpolation=cv2.INTER_AREA)[tri:tro,tci:tco],5,5)
        mask[ri:ro,ci:co]=patch  # range(0,1) -> (y2-y1,x2-x1))
        for c in range(3):  # mask per channel
            blend[:,:,c]=np.where(mask>0,blend[:,:,c]*(1-cfg.overlay_opacity[t])+cfg.overlay_color[t][c]*cfg.overlay_opacity[t],blend[:,:,c]) if fill \
            else np.where(cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((lwd,lwd),np.uint8))>0, cfg.overlay_color[t][c], blend[:,:,c])
            bw_mask[:,:,c]=np.where(mask>0,255,bw_mask[:,:,c])
        patch_area=np.sum(patch,keepdims=False)
        for r,rname in enumerate(name_reg):
            if r==0: # default Total
                res[r,t,0]+=1; res[r,t,1]+=patch_area
            else: # When regions are specified
                patch_intercept=np.minimum(reg[rname][ri:ro,ci:co],patch)
                intercept_area=np.sum(patch_intercept,keepdims=False)
                if intercept_area/patch_area>0.5:
                    res[r,t,0]+=1; res[r,t,1]+=intercept_area
            # print(','.join([box[i],cls[i],scr[i],np.sum(patch,keepdims=False)]))
    origin=Image.fromarray(blend.astype(np.uint8),'RGB')  # L RGB
    draw=ImageDraw.Draw(origin)
    if instance:
        for i in sel:
            t=cls[i]-1
            y1,x1,y2,x2=box[i]
            # draw.rectangle((x1,y1,x2,y2),fill=None,outline=cfg.overlay_color[d]) # bbox
            draw.text((x1,y1-size//2),'%s %d'%(tgts[t],floor(10.0*scr[i])),cfg.overlay_color[t],ImageFont.truetype(font,size)) # class score
    txtlist=[]
    size=int(size*1.8) # increase size
    offset=max(1,size//12) # position offset for light dark text
    if black or white:
        for r,rname in enumerate(name_reg):
            total_pixels=sum_pixels[r]
            for t in range(len(tgts)):
                if white: draw.text((0,0),'\n'*t+tgts[t],(210,210,210),ImageFont.truetype(font,size))
                if black: draw.text((offset,offset),'\n'*t+tgts[t],cfg.overlay_color[t],ImageFont.truetype(font,size))
                res[r,t,2]=100.0*res[r,t,1]/total_pixels
                txtlist.append('            # %d $ %d  %.1f%% (%s)'%(res[r,t,0],res[r,t,1],res[r,t,2],rname))
        txtblk='\n'.join(txtlist)
        if white: draw.text((0,0),txtblk,(210,210,210),ImageFont.truetype(font,size))
        if black: draw.text((offset,offset),txtblk,(30,30,30),ImageFont.truetype(font,size))
    return res,np.array(origin),bw_mask
