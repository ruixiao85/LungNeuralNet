import keras
# import keras_retinanet
from keras_maskrcnn import models
from keras_maskrcnn.utils.visualization import draw_mask
from keras_retinanet.utils.visualization import draw_box, draw_caption, draw_annotations
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())


# adjust this to point to your downloaded/trained model
file='resnet50_coco_v0.1.0.h5'
if not os.path.exists(file):
    from urllib import request
    url='https://github.com/fizyr/keras-maskrcnn/releases/download/0.1/'
    request.urlretrieve(url+file, file)

# load retinanet model
model = models.load_model(file, backbone_name='resnet50')
#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# load image
filename='000000008021'
image=read_image_bgr(filename+'.jpg')

# copy to draw on
draw=image.copy()
draw=cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)

# preprocess image for network
image=preprocess_image(image)
# image,scale=resize_image(image)

# process image
start=time.time()
outputs=model.predict_on_batch(np.expand_dims(image,axis=0))
print("processing time: ",time.time()-start)

boxes=outputs[-4][0]
scores=outputs[-3][0]
labels=outputs[-2][0]
masks=outputs[-1][0]

# correct for image scale
# boxes/=scale

# visualize detections
for box,score,label,mask in zip(boxes,scores,labels,masks):
    if score<0.5:
        break

    color=label_color(label)

    b=box.astype(int)
    draw_box(draw,b,color=color)

    mask=mask[:,:,label]
    draw_mask(draw,b,mask,color=label_color(label))

    caption="{} {:.3f}".format(labels_to_names[label],score)
    draw_caption(draw, b, caption)

cv2.imwrite(filename+'_r.jpg',cv2.cvtColor(draw, cv2.COLOR_RGB2BGR))
