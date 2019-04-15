import os

import numpy as np
from PIL import ImageDraw,Image,ImageFont
import cv2

from a_config import get_proper_range
from preprocess import prep_scale,rev_scale
from math import floor,log,sqrt

dark_text=(40,40,40)
bright_text=(200,200,200)

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

def cal_area_count_perc(rc1,div=1.0,area=None):
    count,labels=cv2.connectedComponents(rc1,connectivity=8,ltype=cv2.CV_32S)
    sum_pos=np.sum(rc1,keepdims=False)/div
    perc=sum_pos/area*100.0 if area else 100.0 # default 100%
    newres=np.array([sum_pos,count-1,perc])
    return newres,labels

def single_call(cfg,img,tgts,msk):  # sigmoid (r,c,1) blend, np result
    row,col,_=img.shape; div=cfg.target_scale**2.0; sum_pixel=row*col/div
    nt=len(tgts)
    bw_mask=np.zeros((row,col,nt),dtype=np.uint8)
    res,text_list=np.array([sum_pixel,1.0,100.0])[np.newaxis,...],["[  0: Total] $%.1f #%d %.1f%%"%(sum_pixel,1.0,100.0)]
    msk=np.rint(msk)  # sigmoid round to  0/1
    for t,tgt in enumerate(tgts):
        curr=msk[...,t][...,np.newaxis].astype(np.uint8)
        newres,labels=cal_area_count_perc(curr,div,sum_pixel)
        text_list.append("[  %d: %s] $%.1f #%d %.1f%%"%(t+1,tgt,newres[0],newres[1],newres[2]))
        res=newres[np.newaxis,...] if res is None else np.concatenate((res,newres[np.newaxis,...]))
        for c in range(3):
            img[...,c]=np.where(msk[...,t]>=0.5,img[...,c]*(1-cfg.overlay_opacity[t])+cfg.overlay_color(t/nt)[c]*cfg.overlay_opacity[t],img[...,c])  # weighted average
            bw_mask[...,t]=np.where(msk[...,t]>=0.5,255,bw_mask[...,t])  # b/w mask maximum object masks
    img=draw_text(cfg,img,text_list,col)
    return res,img,bw_mask

def single_brighten(cfg,img,tgts,msk):  # sigmoid (r,c,1) blend, np result
    row,col,_=img.shape; div=cfg.target_scale**2.0; sum_pixel=row*col/div
    nt=len(tgts)
    bw_mask=np.zeros((row,col,nt))
    res,text_list=np.array([sum_pixel,1.0,100.0])[np.newaxis,...],["[  0: Total] $%.1f #%d %.1f%%"%(sum_pixel,1.0,100.0)]
    msk=np.rint(gaussian_smooth(msk))  # sigmoid round to  0/1
    for t,tgt in enumerate(tgts):
        curr=msk[...,t][...,np.newaxis].astype(np.uint8)
        newres,labels=cal_area_count_perc(curr,div,sum_pixel)
        text_list.append("[  %d: %s] $%.1f #%d %.1f%%"%(t+1,tgt,newres[0],newres[1],newres[2]))
        res=newres[np.newaxis,...] if res is None else np.concatenate((res,newres[np.newaxis,...]))
        for c in range(3):
            mskrev=rev_scale(morph_close(msk,6,6),'sigmoid')
            # img[...,c]=np.where(mskrev[...,d]>=img[...,c],mskrev[...,d],img[...,c])  # weighted average
            img[...,c]=np.maximum(mskrev[...,t],img[...,c])  # brighter area
            bw_mask[...,t]=np.where(msk[...,t]>=0.5,255,bw_mask[...,t])  # b/w mask maximum object masks
    img=draw_text(cfg,img,text_list,col)
    return res,img,bw_mask

def multi_call(cfg,img,tgts,msk):  # softmax (r,c,multi_label) blend, np result
    row,col,_=img.shape; div=cfg.target_scale**2.0; sum_pixel=row*col/div
    nt=len(tgts)
    bw_mask=np.zeros((row,col,nt))
    res,text_list=np.array([sum_pixel,1.0,100.0])[np.newaxis,...],["[  0: Total] $%.1f #%d %.1f%%"%(sum_pixel,1.0,100.0)]
    msk=np.argmax(msk,axis=-1) # do argmax if predict categories covers all possibilities or consider them individually
    # uni,count=np.unique(msk,return_counts=True)
    # map_count=dict(zip(uni,count))
    # count_vec=np.zeros(cfg.predict_size)
    for t in range(cfg.predict_size):
        # curr=msk[..., d][..., np.newaxis].astype(np.uint8)
        # count_vec[d]=map_count.get(d) or 0
        curr=np.where(msk==t,1,0).astype(np.uint8)
        newres,labels=cal_area_count_perc(curr,div,sum_pixel)
        text_list.append("[  %d: %s] $%.1f #%d %.1f%%"%(t+1,tgts[t],newres[0],newres[1],newres[2]))
        res=newres[np.newaxis,...] if res is None else np.concatenate((res,newres[np.newaxis,...]))
        for c in range(3):
            img[...,c]=np.where(msk==t,img[...,c]*(1-cfg.overlay_opacity[t])+cfg.overlay_color(t/nt)[c]*cfg.overlay_opacity[t],img[...,c])
            bw_mask[...,t]=np.where(msk==t,255,bw_mask[...,t])  # b/w mask maximum object masks
    img=draw_text(cfg,img,text_list,col)
    return res,img,bw_mask

def draw_text(cfg,img,text_list,width):
    black,white,_,_ =cfg.overlay_textshape_bwif
    nt=len(text_list)-1
    if black or white:
        font="arial.ttf"  #times.ttf
        size=max(10,int(width/50)) # fontsize at least 12
        off=max(1, size//15)  # text offset
        origin=Image.fromarray(img.astype(np.uint8),'RGB')  # L RGB
        draw=ImageDraw.Draw(origin)
        txtblk='\n'.join(text_list)
        if white: draw.text((0,0),txtblk,bright_text,ImageFont.truetype(font,size))
        if black: draw.text((off,off),txtblk,dark_text,ImageFont.truetype(font,size))
        for i in range(len(text_list)):
            txtcrs=' \n'*i+' X'
            if white: draw.text((0,0),txtcrs,cfg.overlay_bright((i-1)/nt),ImageFont.truetype(font,size))
            if black and i!=0: draw.text((off,off),txtcrs,cfg.overlay_dark((i-1)/nt),ImageFont.truetype(font,size))
        return np.array(origin)
    else:
        return img


def draw_detection(cfg,image,tgts,box,cls,scr,msk,reg=None):
    ori_row,ori_col,_=image.shape
    nt=len(tgts)
    bw_mask=np.zeros_like(image)
    div=cfg.target_scale**2.0
    font="arial.ttf"  #times.ttf
    size=max(10, ori_col//64) # fontsize at least 10
    lwd=max(3, size//12) # line width
    black, white, instance, fill=cfg.overlay_textshape_bwif
    n=len(box)
    if not n: print("\n*** No instances to display *** \n")
    nreg,sum_pixels,name_reg=1,[ori_row*ori_col/div],["WholeArea"] # default only consider whole image
    if reg:
        for (rname,mask) in sorted(reg.items()):  # add more specified regions
            nreg,sum_pixels,name_reg=nreg+1,sum_pixels+[np.sum(mask,keepdims=False)/div],name_reg+[rname] # green channel as mask
    res=np.zeros((nreg,1+nt,4),dtype=np.float32) # 0region:total/parenchyma/... 1target:TotalArea/LYM/MONO/PMN/...  2param:area/count/area_pct/count_density
    for r in range(nreg):
        res[r,0,0]=sum_pixels[r]
        res[r,0,2]=100.0*res[r,0,0]/res[0,0,0]
    chars=3 # for ALL
    for t in tgts:
        chars=max(chars,len(t)) # longest nchars for targets
    for i in range(n):
        t=cls[i]
        y1,x1,y2,x2=box[i]
        mask=np.zeros((ori_row,ori_col),dtype=np.uint8)
        ri,ro,ci,co,tri,tro,tci,tco=get_proper_range(ori_row,ori_col, y1,y2,x1,x2, 0,y2-y1,0,x2-x1)
        patch=gaussian_smooth(cv2.resize(np.where(msk[i]>=0.5,1,0).astype(np.uint8),(x2-x1,y2-y1),interpolation=cv2.INTER_AREA)[tri:tro,tci:tco],5,5)
        mask[ri:ro,ci:co]=patch  # range(0,1) -> (y2-y1,x2-x1))
        color_t=cfg.overlay_color((t-1)/nt)
        for c in range(3):  # mask per channel
            image[:,:,c]=np.where(mask>0,image[:,:,c]*(1-cfg.overlay_opacity[t-1])+color_t[c]*cfg.overlay_opacity[t-1],image[:,:,c]) if fill \
            else np.where(cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((lwd,lwd),np.uint8))>0, color_t[c], image[:,:,c])
            bw_mask[:,:,c]=np.where(mask>0,255,bw_mask[:,:,c])
        patch_area=np.sum(patch,keepdims=False)/div
        for r,rname in enumerate(name_reg):
            if r==0: # default Total
                res[r,t,0]+=patch_area; res[r,t,1]+=1.0
            else: # When regions are specified
                patch_intercept=np.minimum(reg[rname][ri:ro,ci:co],patch)
                intercept_area=np.sum(patch_intercept,keepdims=False)/div
                if intercept_area/patch_area>0.5:
                    res[r,t,0]+=intercept_area; res[r,t,1]+=1.0
            # print(','.join([box[i],cls[i],scr[i],np.sum(patch,keepdims=False)]))
    origin=Image.fromarray(image.astype(np.uint8),'RGB')  # L RGB
    draw=ImageDraw.Draw(origin)
    if instance:
        for i in range(n):
            t=cls[i]
            y1,x1,y2,x2=box[i]
            # draw.rectangle((x1,y1,x2,y2),fill=None,outline=over_color[d]) # bbox
            draw.text((x1,y1-size//2),'%s %d'%(tgts[t],floor(10.0*scr[i])),cfg.overlay_color((t-1)/nt),ImageFont.truetype(font,size)) # class score
    size=int(size*1.8) # increase size
    offset=max(1,size//12) # position offset for light dark text
    if black or white:
        for r,rname in enumerate(name_reg):
            # rc=(r-1)/nreg)
            txt='\n'*(r+1)+rname
            for t,tgt in enumerate(["ALL"]+tgts):
                if t==0:
                    txt+=' [$%.1f %.1f%%] '%(res[r,t,0],res[r,t,2])
                else:
                    sum_reg=res[r,0,0]
                    if sum_reg!=0: # avoid div by zero
                        res[r,t,2]=100.0*res[r,t,0]/sum_reg
                        res[r,t,3]=1000000.0*res[r,t,1]/sum_reg # 1/mm2
                    txt+=' [#%d %.1f] '%(res[r,t,1],res[r,t,3])
                if r==0:
                    if white: draw.text((0,0),'  '*(chars+3)*(t+1)+tgt,cfg.overlay_bright((t-1)/nt) if t!=0 else bright_text,ImageFont.truetype(font,size))
                    if black: draw.text((offset,offset),'  '*(chars+3)*(t+1)+tgt,cfg.overlay_dark((t-1)/nt) if t!=0 else dark_text,ImageFont.truetype(font,size))
            if white: draw.text((0,0),txt,cfg.overlay_bright((r-1)/nreg) if r!=0 else bright_text,ImageFont.truetype(font,size))
            if black: draw.text((offset,offset),txt,cfg.overlay_dark((r-1)/nreg) if r!=0 else dark_text,ImageFont.truetype(font,size))
    return res,np.array(origin),bw_mask
