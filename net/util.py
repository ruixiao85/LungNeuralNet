import numpy as np
from PIL import ImageDraw,Image,ImageFont
import cv2
from process_image import prep_scale,rev_scale


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
    cv2.imwrite(file+'_%d.png'%d,labeled_img)

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
            mskrev=rev_scale(morph_operation(msk,6,6),'sigmoid')
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
    font="arial.ttf"  #times.ttf
    size=round(0.33*(26+0.03*width+width/len(max(text_list,key=len))))
    origin=Image.fromarray(img.astype(np.uint8),'RGB')  # L RGB
    draw=ImageDraw.Draw(origin)
    txtblk='\n'.join(text_list)
    draw.text((0,0),txtblk,(225,225,225),ImageFont.truetype(font,size))
    draw.text((4,4),txtblk,(30,30,30),ImageFont.truetype(font,size))
    for i in range(len(text_list)-1):
        txtcrs=' \n'*(i+1)+' X'
        draw.text((0,0),txtcrs,(225,225,225),ImageFont.truetype(font,size))
        draw.text((5,3),txtcrs,cfg.overlay_color[i],ImageFont.truetype(font,size))
    return np.array(origin)
