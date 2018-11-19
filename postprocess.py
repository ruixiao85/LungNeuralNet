import numpy as np
from PIL import ImageDraw,Image,ImageFont
import cv2
from preprocess import prep_scale,rev_scale
from math import floor,log,sqrt


def smooth_brighten(img):
    blur=np.average(gaussian_smooth(img),axis=-1).astype(np.uint8)
    _,bin=cv2.threshold(blur,20,255, cv2.THRESH_BINARY)
    # bin=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,353,-50)
    # return cv2.morphologyEx(bin, cv2.MORPH_OPEN, (5,5))
    return morph_close(bin)

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

def connect_component_label(d,file,labels):
    label_hue=np.uint8(179*labels/(1e-6+np.max(labels)))  # Map component labels to hue val
    blank_ch=255*np.ones_like(label_hue)
    labeled_img=cv2.merge([label_hue,blank_ch,blank_ch])
    labeled_img=cv2.cvtColor(labeled_img,cv2.COLOR_HSV2BGR)  # cvt to BGR for display
    labeled_img[label_hue==0]=0  # set bg label to black
    cv2.imwrite(file+'_%d.png'%d,labeled_img) # write each channel

def cal_area_count(rc1):
    count,labels=cv2.connectedComponents(rc1,connectivity=8,ltype=cv2.CV_32S)
    newres=np.array([np.sum(rc1,keepdims=False),count-1])
    return newres,labels

def single_call(cfg,img,msk,file=None):  # sigmoid (r,c,1) blend, np result
    res=None; blend=img.copy()
    msk=np.rint(msk)  # sigmoid round to  0/1 # consider range(-1 ~ +1) for multi class voting
    for d in range(msk.shape[-1]):
        curr=msk[...,d][...,np.newaxis].astype(np.uint8)
        newres,labels=cal_area_count(curr)
        res=newres[np.newaxis,...] if res is None else np.concatenate((res,newres[np.newaxis,...]))
        for c in range(3):
            blend[...,c]=np.where(msk[...,d]>=0.5,blend[...,c]*(1-cfg.overlay_opacity[d])+cfg.overlay_color[d][c]*cfg.overlay_opacity[d],blend[...,c])  # weighted average
        if file is not None:
            connect_component_label(d,file,labels)
    return blend, res

def single_brighten(cfg,img,msk,file=None):  # sigmoid (r,c,1) blend, np result
    res=None; blend=img.copy()
    msk=np.rint(gaussian_smooth(msk))  # sigmoid round to  0/1 # consider range(-1 ~ +1) for multi class voting
    for d in range(msk.shape[-1]):
        curr=msk[...,d][...,np.newaxis].astype(np.uint8)
        newres,labels=cal_area_count(curr)
        res=newres[np.newaxis,...] if res is None else np.concatenate((res,newres[np.newaxis,...]))
        for c in range(3):
            mskrev=rev_scale(morph_close(msk,6,6),'sigmoid')
            # blend[...,c]=np.where(mskrev[...,d]>=blend[...,c],mskrev[...,d],blend[...,c])  # weighted average
            blend[...,c]=np.maximum(mskrev[...,d],blend[...,c])  # brighter area
        if file is not None:
            connect_component_label(d,file,labels)
    return blend, res

def multi_call(cfg,img,msk,file=None):  # softmax (r,c,multi_label) blend, np result
    res=None; blend=img.copy()
    msk=np.argmax(msk,axis=-1) # do argmax if predict categories covers all possibilities or consider them individually
    # uni,count=np.unique(msk,return_counts=True)
    # map_count=dict(zip(uni,count))
    # count_vec=np.zeros(cfg.predict_size)
    for d in range(cfg.predict_size):
        # curr=msk[..., d][..., np.newaxis].astype(np.uint8)
        # count_vec[d]=map_count.get(d) or 0
        curr=np.where(msk==d,1,0).astype(np.uint8)
        newres,labels=cal_area_count(curr)
        res=newres[np.newaxis,...] if res is None else np.concatenate((res,newres[np.newaxis,...]))
        for c in range(3):
            blend[...,c]=np.where(msk==d,blend[...,c]*(1-cfg.overlay_opacity[d])+cfg.overlay_color[d][c]*cfg.overlay_opacity[d],blend[...,c])
        if file is not None:
            connect_component_label(d, file, labels)
    return blend, res

def compare_call(cfg,img,msk,file=None):  # compare input and output with same dimension
    res=None
    diff=np.round(np.abs(msk.copy()-prep_scale(img,cfg.out))).astype(np.uint8)
    for d in range(msk.shape[-1]):
        curr=diff[...,d][...,np.newaxis]
        newres,labels=cal_area_count(curr)
        res=newres[np.newaxis,...] if res is None else np.concatenate((res,newres[np.newaxis,...]))
        if file is not None:
            connect_component_label(d,file,labels)
    return rev_scale(msk,cfg.feed), res

def draw_text(cfg,img,text_list,width):
    black, white=cfg.overlay_text_bw
    if black or white:
        font="arial.ttf"  #times.ttf
        size=max(12,int(width/40)) # fontsize at least 12
        off=max(1, size//15)  # text offset
        origin=Image.fromarray(img.astype(np.uint8),'RGB')  # L RGB
        draw=ImageDraw.Draw(origin)
        txtblk='\n'.join(text_list)
        if white: draw.text((0,0),txtblk,(210,210,210),ImageFont.truetype(font,size))
        if black: draw.text((off,off),txtblk,(30,30,30),ImageFont.truetype(font,size))
        for i in range(len(text_list)-1):
            txtcrs=' \n'*(i+1)+' X'
            if white: draw.text((0,0),txtcrs,(210,210,210),ImageFont.truetype(font,size))
            if black: draw.text((off,off),txtcrs,cfg.overlay_color[i],ImageFont.truetype(font,size))
        return np.array(origin)
    else:
        return img


def draw_detection(cfg,image,class_names,box,cls,scr,msk,sel=None):
    if sel is None:
        n=len(box); sel=range(n) # default to all instances
    else:
        n=len(sel) # only use selected indices
    if not n:
        print("\n*** No instances to display *** \n")
    font="arial.ttf"  #times.ttf
    ori_row,ori_col,_=image.shape
    total_pixels=ori_row*ori_col
    size=max(12, int(ori_col/40)) # fontsize at least 12
    lwd=max(2, size//15) # line width
    blend=image.copy()
    res=np.zeros(cfg.num_targets*3,dtype=np.float32) # 1count 1area 1pct  2count 2area 2pct
    for i in sel:
        d=cls[i]-1
        y1,x1,y2,x2=box[i]
        mask=np.zeros((ori_row,ori_col),dtype=np.uint8)
        ri,ro,ci,co,tri,tro,tci,tco=cfg.get_proper_range(ori_row,ori_col, y1,y2,x1,x2, 0,y2-y1,0,x2-x1)
        patch=cv2.resize(np.where(gaussian_smooth(msk[i],5,5)>=0.5,1,0).astype(np.uint8),(x2-x1,y2-y1),interpolation=cv2.INTER_AREA)[tri:tro,tci:tco]
        mask[ri:ro,ci:co]=patch  # range(0,1) -> (y2-y1,x2-x1))
        area=np.sum(patch,keepdims=False)
        for c in range(3):  # mask per channel
            # blend[:,:,c]=np.where(mask>0,blend[:,:,c]*(1-cfg.overlay_opacity[d])+cfg.overlay_color[d][c]*cfg.overlay_opacity[d],blend[:,:,c]) # overlay
            blend[:,:,c]=np.where(cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((lwd,lwd),np.uint8))>0, cfg.overlay_color[d][c], blend[:,:,c]) # opaque outline
        res[d*3]+=1; res[d*3+1]+=area
        # print(','.join([box[i],cls[i],scr[i],np.sum(patch,keepdims=False)]))
    origin=Image.fromarray(blend.astype(np.uint8),'RGB')  # L RGB
    draw=ImageDraw.Draw(origin)
    for i in sel:
        d=cls[i]-1
        y1,x1,y2,x2=box[i]
        # draw.rectangle((x1,y1,x2,y2),fill=None,outline=cfg.overlay_color[d]) # bbox
        draw.text((x1,y1-size//2),'%s %d'%(class_names[d],floor(10.0*scr[i])),cfg.overlay_color[d],ImageFont.truetype(font,size)) # class score
    txtlist=[]
    size=int(size*1.8) # increase size
    offset=max(1,size//12) # position offset for light dark text
    black, white=cfg.overlay_text_bw
    if black or white:
        for i in range(cfg.num_targets):
            if white: draw.text((0,0),'\n'*i+class_names[i],(210,210,210),ImageFont.truetype(font,size))
            if black: draw.text((offset,offset),'\n'*i+class_names[i],cfg.overlay_color[i],ImageFont.truetype(font,size))
            this_pct=100.0*res[3*i+1]/total_pixels
            res[3*i+2]=this_pct
            txtlist.append('            # %d $ %d  %.1f%%'%(res[3*i],res[3*i+1],res[3*i+2]))
        txtblk='\n'.join(txtlist)
        if white: draw.text((0,0),txtblk,(210,210,210),ImageFont.truetype(font,size))
        if black: draw.text((offset,offset),txtblk,(30,30,30),ImageFont.truetype(font,size))
    return np.array(origin),res
