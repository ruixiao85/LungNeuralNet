import os

import numpy as np
from PIL import ImageDraw,Image,ImageFont
import cv2

from a_config import get_proper_range
from preprocess import prep_scale,rev_scale
from math import floor,log,sqrt

HEX_DARK_TEXT=(40,40,40)
HEX_BRIGHT_TEXT=(200,200,200)
FONT= "arial.ttf" #times.ttf

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

def cal_area_count_perc(rc,div=1.0,area=None):
    count,labels=cv2.connectedComponents(rc,connectivity=8,ltype=cv2.CV_32S)
    sum_pos=np.sum(rc,keepdims=False)/div
    perc=1.0*sum_pos/area if area else 0.0 # default 0%
    newres=np.array([sum_pos,count-1,perc])
    return newres,labels

def single_call(cfg,img,tgts,msk):  # sigmoid (r,c,1) blend, np result
    row,col,_=img.shape; nt=len(tgts); div=cfg.target_scale**2.0; sum_pixel=row*col/div
    bw_mask=np.zeros((row,col,nt),dtype=np.uint8)
    res=np.array([sum_pixel,1.0,1.0])[np.newaxis,...]
    msk=np.rint(msk)  # sigmoid round to  0/1
    for t in range(nt):
        boolarray=(msk[...,t]>=0.5)
        newres,labels=cal_area_count_perc(boolarray.astype(np.uint8),div,sum_pixel)
        res=newres[np.newaxis,...] if res is None else np.concatenate((res,newres[np.newaxis,...]))
        for c in range(3):
            img[...,c]=np.where(boolarray,img[...,c]*(1-cfg.overlay_opacity[t])+cfg.overlay_color(t/nt)[c]*cfg.overlay_opacity[t],img[...,c])
        bw_mask[...,t]=np.where(boolarray,255,bw_mask[...,t])  # b/w mask maximum object masks
    img=draw_text(cfg,img,tgts,res,fsize=max(10,int(col/50))) if cfg.overlay_legend_instance_fill[0] else img
    return res,img,bw_mask

def multi_call(cfg,img,tgts,msk):  # softmax (r,c,multi_label) blend, np result
    row,col,_=img.shape; nt=len(tgts); div=cfg.target_scale**2.0; sum_pixel=row*col/div
    bw_mask=np.zeros((row,col,nt),dtype=np.uint8)
    res=np.array([sum_pixel,1.0,1.0])[np.newaxis,...]
    msk=np.argmax(msk,axis=-1)
    for t in range(cfg.predict_size):
        boolarray=(msk==t)
        newres,labels=cal_area_count_perc(boolarray.astype(np.uint8),div,sum_pixel)
        res=newres[np.newaxis,...] if res is None else np.concatenate((res,newres[np.newaxis,...]))
        for c in range(3):
            img[...,c]=np.where(boolarray,img[...,c]*(1-cfg.overlay_opacity[t])+cfg.overlay_color(t/nt)[c]*cfg.overlay_opacity[t],img[...,c])
        bw_mask[...,t]=np.where(boolarray,255,bw_mask[...,t]) # b/w mask maximum object masks
    img=draw_text(cfg,img,tgts,res,fsize=max(10,int(col/50))) if cfg.overlay_legend_instance_fill[0] else img
    return res,img,bw_mask

def draw_text(cfg,img,tgts,res,sep='\n',fsize=None):
    fsize=fsize or max(12,int(img.shape[1]/48)); off=max(1, fsize//15)  # text offset
    nt=len(tgts)
    fontsize=ImageFont.truetype(FONT,fsize)
    origin=Image.fromarray(img.astype(np.uint8),'RGB')  # L RGB
    draw=ImageDraw.Draw(origin)
    for ri in range(1+nt):
        sym=sep*ri+' X'
        draw.text((0,0),sym,HEX_BRIGHT_TEXT if ri==0 else cfg.overlay_bright((ri-1)/nt),fontsize)
        draw.text((off,off),sym,HEX_DARK_TEXT if ri==0 else cfg.overlay_dark((ri-1)/nt),fontsize)
        txt=sep*ri+"[  {:.0f}: {}] ${:.0f} #{:.0f} {:.1%}".format(ri,cfg.region0 if ri==0 else tgts[ri-1],res[ri,0],res[ri,1],res[ri,2])
        draw.text((0,0),txt,HEX_BRIGHT_TEXT,)
        draw.text((off,off),txt,HEX_DARK_TEXT,fontsize)
    return np.array(origin)

def draw_detection(cfg,img,tgts,box,cls,scr,msk,reg=None):
    legend, instance, fill=cfg.overlay_legend_instance_fill
    row,col,_=img.shape; div=cfg.target_scale**2.0
    fsize=max(12, col//48) # fontsize at least 10
    off=max(1,fsize//15) # offset for bright/dark text0
    lwd=max(3,(6+fsize//12)//3) # line width if outline not fill
    fontsize=ImageFont.truetype(FONT,fsize)
    sum_pixels,nrl=[row*col/div],5 # default only consider whole image
    regs,nreg=[cfg.region0],1
    opacity=0.2 # if overlay color needed
    if reg:
        nreg+=len(reg)
        for ri,(rname,mask) in enumerate(reg.items()):  # add more specified regions
            sum_pixels=sum_pixels+[np.sum(mask,keepdims=False)/div] # green channel as mask
            regs,nrl=regs+[rname],max(nrl,len(rname))
            rc=ri/nreg
            for c in range(3):
                img[...,c]=np.where(mask>0.5,img[...,c]*(1-opacity)+cfg.overlay_bright(rc)[c]*opacity,img[...,c])
    ni,nt=len(box),len(tgts) # number of instances, targets
    tgts_1=[cfg.target0]+tgts
    res=np.zeros((nreg,1+nt,4),dtype=np.float32) # 0region:total/parenchyma/... 1target:TotalArea/LYM/MONO/PMN/...  2param:area/count/area_pct/count_density
    bw_mask=np.zeros((row,col,nt),dtype=np.uint8)
    for r in range(nreg):
        res[r,0,0]=sum_pixels[r]
        res[r,0,2]=res[r,0,0]/res[0,0,0]
    if ni:
        for i in range(ni):
            t=cls[i]
            y1,x1,y2,x2=box[i]
            mask=np.zeros((row,col),dtype=np.uint8)
            ri,ro,ci,co,tri,tro,tci,tco=get_proper_range(row,col, y1,y2,x1,x2, 0,y2-y1,0,x2-x1)
            patch=gaussian_smooth(cv2.resize(np.where(msk[i]>=0.5,1,0).astype(np.uint8),(x2-x1,y2-y1),interpolation=cv2.INTER_AREA)[tri:tro,tci:tco],5,5)
            mask[ri:ro,ci:co]=patch  # range(0,1) -> (y2-y1,x2-x1))
            color_t=cfg.overlay_color((t-1)/nt)
            boolarray=(mask>0)
            bw_mask[:,:,t-1]=np.where(boolarray,255,bw_mask[:,:,t-1])
            boolarray=boolarray if fill else cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((lwd,lwd),np.uint8))>0 # whole mask or outline for the next step
            for c in range(3):
                # img[:,:,c]=np.where(boolarray,img[:,:,c]*(1-cfg.overlay_opacity[t-1])+color_t[c]*cfg.overlay_opacity[t-1],img[:,:,c]) # blend color
                img[:,:,c]=np.where(boolarray,color_t[c],img[:,:,c]) # pure color
            patch_area=np.sum(patch,keepdims=False)/div
            for r,rname in enumerate(regs):
                if r==0: # default Total
                    res[r,t,0]+=patch_area; res[r,t,1]+=1.0
                else: # When regions are specified
                    patch_intercept=np.minimum(reg[rname][ri:ro,ci:co],patch)
                    intercept_area=np.sum(patch_intercept,keepdims=False)/div
                    if intercept_area/patch_area>0.5:
                        res[r,t,0]+=intercept_area; res[r,t,1]+=1.0
                # print(','.join([box[i],cls[i],scr[i],np.sum(patch,keepdims=False)]))
        if legend or instance:
            origin=Image.fromarray(img.astype(np.uint8),'RGB')  # L RGB
            draw=ImageDraw.Draw(origin)
            if legend:
                len_tgt=max([len(t) for t in tgts_1])  # length of the longest target
                for r,rname in enumerate(regs):
                    rc=(r-1)/nreg
                    txt='\n'*(r+1)+'[  {:.0f}: {:<{}}]'.format(r,rname,nrl)
                    for t,tgt in enumerate(tgts_1):
                        if t==0:
                            txt+=' ${:,.0f}  {:.1%} |'.format(res[r,t,0],res[r,t,2])
                            # txt+=' ${:,.1e}  {:>3d}% |'.format(res[r,t,0],int(round(res[r,t,2]*100)))
                        else:
                            sum_reg=res[r,0,0]
                            if sum_reg!=0: # avoid div by zero
                                res[r,t,2]=res[r,t,0]/sum_reg
                                res[r,t,3]=1000000.0*res[r,t,1]/sum_reg # 1/mm2
                            txt+=' #{:.0f}  {:.1f} |'.format(res[r,t,1],res[r,t,3])
                            # txt+=' #{:>4d}  {:.1e} |'.format(int(round(res[r,t,1])),res[r,t,3])
                        if r==0:
                            draw.text((0,0),'  '*(len_tgt+3)*(t+1)+tgt,cfg.overlay_bright((t-1)/nt) if t!=0 else HEX_BRIGHT_TEXT,fontsize)
                            draw.text((off,off),'  '*(len_tgt+3)*(t+1)+tgt,cfg.overlay_dark((t-1)/nt) if t!=0 else HEX_DARK_TEXT,fontsize)
                    draw.text((0,0),txt,HEX_BRIGHT_TEXT,fontsize)
                    draw.text((off,off),txt,HEX_DARK_TEXT,fontsize)
                    draw.text((0,0),'\n'*(r+1)+' X',cfg.overlay_bright(rc) if r!=0 else HEX_BRIGHT_TEXT,fontsize)
                    draw.text((off,off),'\n'*(r+1)+' X',cfg.overlay_dark(rc) if r!=0 else HEX_DARK_TEXT,fontsize)
            if instance:
                for i in range(ni):
                    t=cls[i]
                    y1,x1,y2,x2=box[i]
                    # draw.rectangle((x1,y1,x2,y2),fill=None,outline=over_color[d]) # bbox
                    draw.text((x1,y1-fsize//2),'{} {:.0f}'.format(tgts[t],floor(10.0*scr[i])),cfg.overlay_color((t-1)/nt),fontsize) # class score
            img=np.array(origin)
    else:
        print("\n*** No instances to display *** \n")
        img=draw_text(cfg,img,tgts,regs,res[:,0,:],fsize=max(10,int(col/50))) if legend else img
    print('Instances Legendes [%d] drawn.'%ni)
    return res,img,bw_mask
