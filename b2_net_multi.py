import math
import os
import re

import cv2
import datetime
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from keras.engine.saving import model_from_json,load_model

from a_config import Config
from image_set import ImageSet,ViewSet
from osio import mkdir_ifexist,to_excel_sheet
from postprocess import g_kern_rect,draw_text,smooth_brighten,draw_detection
from mrcnn import utils
from preprocess import prep_scale,augment_image_set,augment_patch,read_image,read_resize,read_resize_pad,read_resize_fit


class BaseNetM(Config):
    def __init__(self,**kwargs):
        super(BaseNetM,self).__init__(**kwargs)
        self.is_train=None # will set later
        from c0_backbones import v16, v19
        self.backbone=kwargs.get('backbone', v16) # default backbone
        self.learning_rate=kwargs.get('learning_rate', 1e-3 if self.backbone in [v16,v19] else 1e-2) # initial learning rate
        self.learning_decay=kwargs.get('learning_decay', 0.3)
        from keras.optimizers import SGD
        self.optimizer=kwargs.get('optimizer', SGD(lr=self.learning_rate, momentum=0.9, clipnorm=5.0))
        self.loss_weight=kwargs.get('loss_weight', { "rpn_class_loss":1., "rpn_bbox_loss":1.,
                "mrcnn_class_loss":1., "mrcnn_bbox_loss":1., "mrcnn_mask_loss":1.}) # Loss weights for more precise optimization.
        self.indicator=kwargs.get('indicator', 'val_loss')
        self.indicator_trend=kwargs.get('indicator_trend', 'min')
        from postprocess import draw_detection
        self.predict_proc=kwargs.get('predict_proc', draw_detection)
        self.trainlayer_regex=kwargs.get('trainlayer_regex', ".*") # all
        # self.trainlayer_regex=kwargs.get('trainlayer_regex', r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)")  # head
        # self.trainlayer_regex=kwargs.get('trainlayer_regex', r"# (res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)") # 3+
        # self.trainlayer_regex=kwargs.get('trainlayer_regex', r"# (res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)") # 4+
        # self.trainlayer_regex=kwargs.get('trainlayer_regex', r"# (res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)") # 5+
        self.num_class=1+self.num_targets # plus background
        self.meta_shape=[1+3+3+4+1+self.num_class] # last number is NUM_CLASS
        self.batch_norm=kwargs.get('batch_norm', True) # images in small batches also benefit from batchnorm
        self.backbone_strides=kwargs.get('backbone_stride', [4,8,16,32,64]) # strides of the FPN Pyramid
        self.pyramid_size=kwargs.get('pyramid_size', 512) # Size of the top-down layers used to build the feature pyramid
        self.fc_layers_size=kwargs.get('fc_layers_size', 1024) # Size of the fully-connected layers in the classification graph
        self.rpn_anchor_scales=kwargs.get('rpn_anchor_scales', (8,16,32,64,128)) # Length of square anchor side in pixels
        # self.rpn_anchor_scales=kwargs.get('rpn_anchor_scales', (16,32,64,128,256)) # Length of square anchor side in pixels
        # self.rpn_anchor_scales=kwargs.get('rpn_anchor_scales', (32,64,128,256,512)) # Length of square anchor side in pixels
        self.rpn_train_anchors_per_image=kwargs.get('rpn_train_anchors_per_image', 512) # How many anchors per image to use for RPN training
        self.rpn_anchor_ratios=kwargs.get('rpn_anchor_ratio', [0.75,1,1.25]) # Ratios of anchors at each cell (width/height) 1=square 0.5=wide
        # self.rpn_anchor_ratios=kwargs.get('rpn_anchor_ratio', [0.5,1,2]) # Ratios of anchors at each cell (width/height) 1=square 0.5=wide
        self.rpn_anchor_stride=kwargs.get('rpn_anchor_stride', 1) # 1=no-skip cell 2=skip-one
        self.rpn_nms_threshold=kwargs.get('rpn_nms_threshold', 0.9) # Non-max suppression threshold to filter RPN proposals. larger=more propsals.
        self.rpn_bbox_stdev=kwargs.get('rpn_bbox_stdev', np.array([0.1,0.1,0.2,0.2])) # Bounding box refinement standard deviation for RPN and final detections.
        self.pre_nms_limit=kwargs.get('pre_nms_limit', 4000) # ROIs kept after tf.nn.top_k and before non-maximum suppression
        self.post_mns_train=kwargs.get('post_mns_train', 2000) # ROIs kept after non-maximum suppression for train
        self.post_nms_predict=kwargs.get('post_nms_predict', 1000) # ROIs kept after non-maximum suppression for predict
        self.pool_size=kwargs.get('pool_size', 7) # Pooled ROIs
        self.mask_pool_size=kwargs.get('mask_pool_size', 14) # Pooled ROIs for mask
        self.mini_mask_shape=kwargs.get('mini_mask_shape', [28,28,None]) # target shape (downsized) of instance masks to reduce memory load.
        self.train_rois_per_image=kwargs.get('train_rois_per_image', 256) # Number of ROIs per image to feed to classifier/mask heads (MRCNN paper 512)
        self.train_roi_positive_ratio=kwargs.get('train_roi_positive_ratio', 0.33) # Percent of positive ROIs used to train classifier/mask heads
        self.max_gt_instance=kwargs.get('max_gt_instance', 200) # Maximum number of ground truth instances to use in one image
        self.detection_max_instances=kwargs.get('detection_max_instances', 400) # Max number of final detections
        self.detection_min_confidence=kwargs.get('detection_min_confidence', 0.7) # Minimum probability to accept a detected instance, skip ROIs if below this threshold
        self.detection_nms_threshold=kwargs.get('detection_nms_threshold', 0.3) # Non-maximum suppression threshold for detection
        self.detection_mask_threshold=kwargs.get('detection_mask_threshold', 0.5) # threshold to determine fore/back-ground
        self.gpu_count=kwargs.get('gpu_count', 1)
        self.images_per_gpu=kwargs.get('image_per_gpu', 1)
        self.filename=kwargs.get('filename', None)
        self.ntop=8 # override parent class to keep more top networks for further MRCNN evaluation
        self.net=None
        self._anchor_cache={}

    def set_trainable(self,node,indent=0):
        # In multi-GPU training, we wrap the model. Get layers of the inner model because they have the weights.
        layers=node.inner_model.layers if hasattr(node,"inner_model") else node.layers
        for layer in layers:
            if layer.__class__.__name__=='Model':
                print("In model: ",layer.name)
                print(self.trainlayer_regex)
                self.set_trainable(layer,indent=indent+4)
                continue
            if not layer.weights:
                continue
            trainable=bool(re.fullmatch(self.trainlayer_regex,layer.name))
            text='+' if trainable else '-'
            class_name=layer.__class__.__name__
            if class_name=='BatchNormalization':
                trainable=self.batch_norm # override for BatchNorm
                text='B' if trainable else 'b'
            elif class_name=='Conv2D':
                # trainable=not (layer.kernel_size==(7,7) and layer.strides==(2,2) and layer.filters==64) # train all conv filters except first conv7x7
                # trainable=True # force train all conv filters
                text='C' if trainable else 'c'
            elif class_name=='TimeDistributed': # set trainable deeper if TimeDistributed
                layer.layer.trainable=trainable
                text='T' if trainable else 't'
            else:
                layer.trainable=trainable
            # print(" "*indent+'%s - %s - trainable %r'%(layer.name,layer.__class__.__name__,trainable)) # verbose
            print(text, end='')

    def build_net(self, is_train):
        self.is_train=is_train
        input_image=KL.Input(shape=[None,None,self.dep_in],name="input_image")
        input_image_meta=KL.Input(shape=self.meta_shape,name="input_image_meta")
        if self.is_train:
            input_gt_class_ids=KL.Input(shape=[None],name="input_gt_class_ids",dtype=tf.int32)  # GT Class IDs (zero padded)
            input_gt_boxes=KL.Input(shape=[None,4],name="input_gt_boxes",dtype=tf.float32)  # GT Boxes in pixels (zero padded)  (y1, x1, y2, x2)
            input_gt_masks=KL.Input(shape=self.mini_mask_shape,name="input_gt_masks",dtype=bool)  # GT Masks
            input_rpn_match=KL.Input(shape=[None,1],name="input_rpn_match",dtype=tf.int32)
            input_rpn_bbox=KL.Input(shape=[None,4],name="input_rpn_bbox",dtype=tf.float32)
            gt_boxes=KL.Lambda(lambda x:norm_boxes_graph(x,K.shape(input_image)[1:3]))(input_gt_boxes)  # Normalize coordinates
            mrcnn_feature_maps,rpn_feature_maps=self.cnn_fpn_feature_maps(input_image)  # same train/predict
            anchors=self.get_anchors_norm()[1]
            anchors=np.broadcast_to(anchors,(self.batch_size,)+anchors.shape)
            anchors=KL.Lambda(lambda x:tf.Variable(anchors),name="anchors")(input_image)
            rpn_bbox,rpn_class,rpn_class_logits,rpn_rois=self.rpn_outputs(anchors,rpn_feature_maps)  # same train/predict
            active_class_ids=KL.Lambda(lambda x:parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)  # Class ID mask to mark available class IDs
            # Generate detection targets Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero padded. Equally, returned rois and targets are zero padded.
            rois,target_class_ids,target_bbox,target_mask=DetectionTargetLayer(self.images_per_gpu,self.train_rois_per_image,self.train_roi_positive_ratio,
                                                                               self.mini_mask_shape,self.rpn_bbox_stdev,name="proposal_targets")(
                [rpn_rois,input_gt_class_ids,gt_boxes,input_gt_masks])
            # Network Heads
            mrcnn_class_logits,mrcnn_class,mrcnn_bbox=fpn_classifier_graph(rois,mrcnn_feature_maps,input_image_meta,self.pool_size,self.num_class,
                                                                           train_bn=self.batch_norm,fc_layers_size=self.fc_layers_size)
            mrcnn_mask=build_fpn_mask_graph(rois,mrcnn_feature_maps,input_image_meta,self.mask_pool_size,self.num_class,train_bn=self.batch_norm)
            output_rois=KL.Lambda(lambda x:x*1,name="output_rois")(rois)
            # Losses
            rpn_class_loss=KL.Lambda(lambda x:rpn_class_loss_graph(*x),name="rpn_class_loss")([input_rpn_match,rpn_class_logits])
            rpn_bbox_loss=KL.Lambda(lambda x:rpn_bbox_loss_graph(self.images_per_gpu,*x),name="rpn_bbox_loss")([input_rpn_bbox,input_rpn_match,rpn_bbox])
            class_loss=KL.Lambda(lambda x:mrcnn_class_loss_graph(*x),name="mrcnn_class_loss")([target_class_ids,mrcnn_class_logits,active_class_ids])
            bbox_loss=KL.Lambda(lambda x:mrcnn_bbox_loss_graph(*x),name="mrcnn_bbox_loss")([target_bbox,target_class_ids,mrcnn_bbox])
            mask_loss=KL.Lambda(lambda x:mrcnn_mask_loss_graph(*x),name="mrcnn_mask_loss")([target_mask,target_class_ids,mrcnn_mask])
            # Model
            model=KM.Model([input_image,input_image_meta,input_rpn_match,input_rpn_bbox,input_gt_class_ids,input_gt_boxes,input_gt_masks],
                           [rpn_class_logits,rpn_class,rpn_bbox,mrcnn_class_logits,mrcnn_class,mrcnn_bbox,mrcnn_mask,
                            rpn_rois,output_rois,rpn_class_loss,rpn_bbox_loss,class_loss,bbox_loss,mask_loss],name='mask_rcnn')
        else:
            input_anchors=KL.Input(shape=[None,4],name="input_anchors")  # Anchors in normalized coordinates
            mrcnn_feature_maps,rpn_feature_maps=self.cnn_fpn_feature_maps(input_image)  # same train/predict
            rpn_bbox,rpn_class,rpn_class_logits,rpn_rois=self.rpn_outputs(input_anchors,rpn_feature_maps)  # same train/predict
            # Network Heads Proposal classifier and BBox regressor heads
            mrcnn_class_logits,mrcnn_class,mrcnn_bbox=fpn_classifier_graph(rpn_rois,mrcnn_feature_maps,input_image_meta,self.pool_size,self.num_class,
                                                                           train_bn=self.batch_norm,fc_layers_size=self.fc_layers_size)
            # Detections [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
            detections=DetectionLayer(self.rpn_bbox_stdev,self.detection_min_confidence,self.detection_max_instances,self.
                                      detection_nms_threshold,self.gpu_count,self.images_per_gpu,name="mrcnn_detection")(
                [rpn_rois,mrcnn_class,mrcnn_bbox,input_image_meta])
            # Create masks for detections
            detection_boxes=KL.Lambda(lambda x:x[...,:4])(detections)
            mrcnn_mask=build_fpn_mask_graph(detection_boxes,mrcnn_feature_maps,input_image_meta,self.mask_pool_size,self.num_class,train_bn=self.batch_norm)
            model=KM.Model([input_image,input_image_meta,input_anchors],
                           [detections,mrcnn_class,mrcnn_bbox,mrcnn_mask,rpn_rois,rpn_class,rpn_bbox],name='mask_rcnn')
        self.net=model

    def get_anchors_norm(self):
        backbone_shapes=np.array([[int(math.ceil(self.row_in/stride)),int(math.ceil(self.col_in/stride))] for stride in self.backbone_strides])
        # Cache anchors and reuse if image shape is the same
        if not self.dim_in in self._anchor_cache:
            anchors=generate_pyramid_anchors(self.rpn_anchor_scales, self.rpn_anchor_ratios, backbone_shapes,
                                        self.backbone_strides, self.rpn_anchor_stride)
            self._anchor_cache[self.dim_in]=[anchors,utils.norm_boxes(anchors.copy(),self.dim_in[:2])]
        return self._anchor_cache[self.dim_in]


    def cnn_fpn_feature_maps(self,input_image):
        c1,c2,c3,c4,c5=self.backbone(input_image, weights='imagenet' if self.pre_trained else None) # Bottom-up Layers (convolutional neural network backbone)

        p5=KL.Conv2D(self.pyramid_size,(1,1),name='fpn_c5p5')(c5) # Top-down Layers (feature pyramid network)
        p4=KL.Add(name="fpn_p4add")([KL.UpSampling2D(size=(2,2),name="fpn_p5upsampled")(p5),KL.Conv2D(self.pyramid_size,(1,1),name='fpn_c4p4')(c4)])
        p3=KL.Add(name="fpn_p3add")([KL.UpSampling2D(size=(2,2),name="fpn_p4upsampled")(p4),KL.Conv2D(self.pyramid_size,(1,1),name='fpn_c3p3')(c3)])
        p2=KL.Add(name="fpn_p2add")([KL.UpSampling2D(size=(2,2),name="fpn_p3upsampled")(p3),KL.Conv2D(self.pyramid_size,(1,1),name='fpn_c2p2')(c2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        p2=KL.Conv2D(self.pyramid_size,(3,3),padding="SAME",name="fpn_p2")(p2)
        p3=KL.Conv2D(self.pyramid_size,(3,3),padding="SAME",name="fpn_p3")(p3)
        p4=KL.Conv2D(self.pyramid_size,(3,3),padding="SAME",name="fpn_p4")(p4)
        p5=KL.Conv2D(self.pyramid_size,(3,3),padding="SAME",name="fpn_p5")(p5)
        # p6 is used for the 5th anchor scale in RPN. Generated by subsampling from p5 with stride of 2.
        p6=KL.MaxPooling2D(pool_size=(1,1),strides=2,name="fpn_p6")(p5)
        rpn_feature_maps=[p2,p3,p4,p5,p6]  # all used in rpn
        mrcnn_feature_maps=[p2,p3,p4,p5]  # p6 not used in the classifier heads.
        return mrcnn_feature_maps,rpn_feature_maps

    def rpn_outputs(self,anchors,rpn_feature_maps):
        feature_map=KL.Input(shape=[None,None,self.pyramid_size],name="input_rpn_feature_map")  # region proposal network model
        shared=KL.Conv2D(512,(3,3),padding='same',activation='relu',strides=self.rpn_anchor_stride,name='rpn_conv_shared')(feature_map)
        # Anchor Score. [batch, height, width, anchors per location * 2].
        x=KL.Conv2D(2*len(self.rpn_anchor_ratios),(1,1),padding='valid',activation='linear',name='rpn_class_raw')(shared)
        # Reshape to [batch, anchors, 2]
        rpn_class_logits=KL.Lambda(lambda t:tf.reshape(t,[tf.shape(t)[0],-1,2]))(x)
        # Softmax on last dimension of BG/FG.
        rpn_probs=KL.Activation("softmax",name="rpn_class_xxx")(rpn_class_logits)
        # Bounding box refinement. [batch, H, W, anchors per location * depth] where depth is [x, y, log(w), log(h)]
        x=KL.Conv2D(len(self.rpn_anchor_ratios)*4,(1,1),padding="valid",activation='linear',name='rpn_bbox_pred')(shared)
        # Reshape to [batch, anchors, 4]
        rpn_bbox=KL.Lambda(lambda t:tf.reshape(t,[tf.shape(t)[0],-1,4]))(x)
        rpn=KM.Model([feature_map],[rpn_class_logits,rpn_probs,rpn_bbox],name="rpn_model")

        layer_outputs=[rpn([p]) for p in rpn_feature_maps]  # Loop through pyramid layers
        outputs=list(zip(*layer_outputs))  # list of lists of level outputs -> list of lists of outputs across levels
        output_names=["rpn_class_logits","rpn_class","rpn_bbox"]
        rpn_class_logits,rpn_class,rpn_bbox=[KL.Concatenate(axis=1,name=n)(list(o)) for o,n in zip(outputs,output_names)]
        # Generate proposals [batch, N, (y1, x1, y2, x2)] in normalized coordinates and zero padded.
        rpn_rois=ProposalLayer(proposal_count=(self.post_mns_train if self.is_train else self.post_nms_predict),rpn_nms_threshold=self.rpn_nms_threshold,
                               rpn_bbox_stdev=self.rpn_bbox_stdev,pre_nms_limit=self.pre_nms_limit,images_per_gpu=self.images_per_gpu,name="ROI")(
            [rpn_class,rpn_bbox,anchors])
        return rpn_bbox,rpn_class,rpn_class_logits,rpn_rois

    @classmethod
    def from_json(cls, filename):  # load model from json
        my_net=cls(filename=filename)
        with open(filename+".json", 'r') as json_file:
            my_net.net=model_from_json(json_file.read())

    def save_net(self):
        json_net=(self.filename if self.filename is not None else str(self)) + ".json"
        with open(json_net, "w") as json_file:
            json_file.write(self.net.to_json())

    def compile_net(self,save_net=False,print_summary=False):
        assert self.is_train, 'only applicable to training mode'
        self.net._losses=[]
        self.net._per_input_losses={}
        for loss in self.loss_weight.keys():
            layer=self.net.get_layer(loss)
            if layer.output not in self.net.losses: # loss
                loss_fun=(tf.reduce_mean(layer.output,keepdims=True)*self.loss_weight.get(loss,1.))
                self.net.add_loss(loss_fun)
        reg_losses=[keras.regularizers.l2(0.001)(w)/tf.cast(tf.size(w),tf.float32) for w in self.net.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name] # L2 Regularization but skip gamma and beta weights of batch normalization layers.
        self.net.add_loss(tf.add_n(reg_losses))
        self.net.compile(optimizer=self.optimizer, loss=[None]*len(self.net.outputs))
        for loss in self.loss_weight.keys():
            layer=self.net.get_layer(loss)
            if loss not in self.net.metrics_names: # metrics
                self.net.metrics_names.append(loss)
                loss_fun=(tf.reduce_mean(layer.output,keepdims=True)*self.loss_weight.get(loss,1.))
                self.net.metrics_tensors.append(loss_fun)
        print("Model compiled.")
        if save_net:
            self.save_net()
            print('Model saved to file.')
        if print_summary:
            self.net.summary()

    def __str__(self):
        return '_'.join([
            type(self).__name__,
            self.cap_lim_join(4, self.feed, self.act, self.out)
            +str(self.num_targets)])
    def __repr__(self):
        return str(self)+self.predict_proc.__name__[0:1].upper()

    @staticmethod
    def cap_lim_join(lim,*text):
        test_list=[t.capitalize()[:lim] for t in text]
        return ''.join(test_list)

    def train(self,pair):
        self.build_net(is_train=True)
        for tr,val,dir_out in pair.train_generator():
            self.set_trainable(self.net)
            self.compile_net() # set optimizers
            self.filename=dir_out+'_'+str(self)
            print("Training for %s"%(self.filename))
            init_epoch,best_value=0,None # store last best
            last_saves=self.find_best_models(self.filename+'^*^.h5')
            if isinstance(last_saves, list) and len(last_saves)>0:
                last_best=last_saves[0]
                init_epoch,best_value=Config.parse_saved_model(last_best)
                if self.train_continue:
                    print("Continue from previous weights.")
                    self.net.load_weights(last_best,by_name=True)
                    # print("Continue from previous model with weights & optimizer")
                    # self.net=load_model(last_best,custom_objects=custom_function_dict())  # does not work well with custom act, loss func
                else:
                    print("Train with some random weights."); init_epoch=0
            if not os.path.exists(self.filename+".txt"):
                with open(self.filename+".txt","w") as net_summary:
                    self.net.summary(print_fn=lambda x:net_summary.write(x+'\n'))
            # if not os.path.exists(self.filename+".json"): self.save_net() # Lambda not saved correctly
            from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,LearningRateScheduler
            from callbacks import TensorBoardTrainVal, ModelCheckpointCustom
            history=self.net.fit_generator(tr,validation_data=val,verbose=1,
               steps_per_epoch=min(self.train_step,len(tr.view_coord)) if isinstance(self.train_step,int) else len(tr.view_coord),
               validation_steps=min(self.train_vali_step,len(val.view_coord)) if isinstance(self.train_vali_step,int) else len(val.view_coord),
               epochs=self.train_epoch,max_queue_size=1,workers=0,use_multiprocessing=False,shuffle=False,initial_epoch=init_epoch,
               callbacks=[
                   ModelCheckpointCustom(self.filename,monitor=self.indicator,mode=self.indicator_trend,hist_best=best_value,
                                 save_weights_only=True,save_mode=self.save_mode,lr_decay=self.learning_decay,sig_digits=self.sig_digits,verbose=1),
                   EarlyStopping(monitor=self.indicator,mode=self.indicator_trend,patience=2,verbose=1),
                   # LearningRateScheduler(lambda x: learning_rate*(self.learning_decay**x),verbose=1),
                   # ReduceLROnPlateau(monitor=self.indicator, mode='min', factor=0.5, patience=1, min_delta=1e-8, cooldown=0, min_lr=0, verbose=1),
                   # TensorBoardTrainVal(log_dir=os.path.join("log", self.filename), write_graph=True, write_grads=False, write_images=True),
               ]).history
            df=pd.DataFrame(history)
            df['time']=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            df.to_csv(self.filename+".csv",mode="a",header=(not os.path.exists(self.filename+".csv")))
            self.find_best_models(self.filename+'^*^.h5')  # remove unnecessary networks
        del self.net

    def eval(self,pair):
        self.build_net(is_train=False)
        for tr,val,dir_out in pair.train_generator():
            self.filename=dir_out+'_'+str(self)
            print('Evaluating neural net...'); val.set_eval()
            weight_files=self.find_best_models(self.filename+'^*^.h5',allow_cache=True)
            for weight_file in weight_files:
                print(weight_file)
                self.net.load_weights(weight_file,by_name=True)  # weights only
                steps_done,steps=0,len(val)
                print("Total steps: ",steps)
                val.on_epoch_end() #initialize
                valiter=iter(val)
                with open(self.filename+".log","a") as log:
                    log.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
                    APs=[]; print("AP:",end=''); log.write(weight_file+',')
                    while steps_done<steps:
                        img,gt=next(valiter)
                        detections,mrcnn_class,mrcnn_bbox,mrcnn_mask,rpn_rois,rpn_class,rpn_bbox=self.net.predict_on_batch(img)
                        for i in range(np.shape(detections)[0]):
                            final_rois,final_class_ids,final_scores,final_masks=parse_detections(detections[i],mrcnn_mask[i],self.dim_in,full_mask=True) # first element
                            AP,precisions,recalls,overlaps=utils.compute_ap(gt[1][i],gt[0][i],gt[2][i],final_rois,final_class_ids,final_scores,
                                np.transpose(final_masks,(1,2,0)))
                            APs.append(AP); print(' %.2f'%AP,end='',flush=True)
                        steps_done+=1
                    mAP=np.mean(APs)
                    print("\nmAP: ",mAP); log.write(str(mAP)+'\n')

    def predict(self,pair,pred_dir):
        self.build_net(is_train=False)
        xls_file="Result_%s_%s.xlsx"%(pred_dir,repr(self))
        batch=pair.img_set.view_coord_batch()  # image/1batch -> view_coord
        dir_cfg_append=pair.img_set.category_append(None,self.row_out,self.col_out)+'_'+str(self)
        save_ind,save_raw=pair.cfg.save_ind_raw
        res_ind,res_grp=None,None
        for dir_out,tgt_list in pair.predict_generator_note():
            res_i,res_g=None,None
            print('Load model and predict to [%s]...'%dir_out)
            export_name=dir_out+'_'+dir_cfg_append
            target_dir=os.path.join(pair.wd,export_name); mkdir_ifexist(target_dir) # dir for invidual images
            merge_dir=os.path.join(pair.wd,dir_out+'+'+dir_cfg_append); mkdir_ifexist(merge_dir) # dir for grouped/whole images
            for grp,view in batch.items():
                grp_box,grp_cls,grp_scr,grp_msk=None,None,None,None
                prd,tgt_name=pair.predict_generator_partial(tgt_list,view)
                # weight_file=tgt_name+'_'+dir_cfg_append+'.h5'
                weight_file=self.find_best_models(tgt_name+'_'+dir_cfg_append+'^*^.h5',allow_cache=True)[0]
                print(weight_file)
                self.net.load_weights(weight_file,by_name=True)  # weights only
                # self.net=load_model(weight_file,custom_objects=custom_function_dict()) # weight optimizer archtecture
                detections,mrcnn_class,mrcnn_bbox,mrcnn_mask,rpn_rois,rpn_class,rpn_bbox=\
                    self.net.predict_generator(prd,max_queue_size=1,workers=0,use_multiprocessing=False,verbose=1)
                mrg_in=np.zeros((view[0].ori_row,view[0].ori_col,self.dep_in),dtype=np.float32)
                for i,(det,msk) in enumerate(zip(detections,mrcnn_mask)): # each view
                    final_rois,final_class_ids,final_scores,final_masks=parse_detections(det,msk,self.dim_in)
                    origin=view[i].get_image(os.path.join(pair.wd,pair.img_set.target_folder))
                    blend, r_i=self.predict_proc(self,origin,pair.targets,final_rois,final_class_ids,final_scores,final_masks)
                    res_i=r_i[np.newaxis,...] if res_i is None else np.concatenate([res_i,r_i[np.newaxis,...]],axis=0)
                    if save_ind:
                        # cv2.imwrite(os.path.join(target_dir,view[i].file_name),origin)
                        cv2.imwrite(os.path.join(target_dir,view[i].file_name),blend)
                    y_d=view[i].row_start; x_d=view[i].col_start
                    overall_rois=final_rois.copy() # otherwise will update individual coordinates
                    overall_rois[:,0]+=y_d; overall_rois[:,2]+=y_d
                    overall_rois[:,1]+=x_d; overall_rois[:,3]+=x_d
                    grp_box=overall_rois if grp_box is None else np.concatenate((grp_box,overall_rois))
                    grp_cls=final_class_ids if grp_cls is None else np.concatenate((grp_cls,final_class_ids))
                    grp_scr=final_scores if grp_scr is None else np.concatenate((grp_scr,final_scores))
                    grp_msk=final_masks if grp_msk is None else np.concatenate((grp_msk,final_masks))
                    ri,ro,ci,co,tri,tro,tci,tco=self.get_proper_range(view[i].ori_row,view[i].ori_col,
                            view[i].row_start,view[i].row_end,view[i].col_start,view[i].col_end,  0,self.row_out,0,self.col_out)
                    mrg_in[ri:ro,ci:co]=origin[tri:tro,tci:tco]
                sel_index=utils.non_max_suppression(grp_box,grp_scr,threshold=self.detection_nms_threshold) if grp_box.shape[0]>0 and self.coverage_predict>1 else None
                if save_raw and pair.img_set.resize_ratio!=1:  # high-res raw group image
                    mrg_in=read_image(os.path.join(pair.wd,pair.img_set.raw_folder,view[0].image_name))
                    grp_box=(grp_box.astype(np.float32)/pair.img_set.resize_ratio).astype(np.int32)
                # cv2.imwrite(os.path.join(merge_dir,view[0].image_name),mrg_in)
                blend,r_g=self.predict_proc(self,mrg_in,pair.targets,grp_box,grp_cls,grp_scr,grp_msk,sel_index)
                res_g=r_g[np.newaxis,...] if res_g is None else np.concatenate((res_g,r_g[np.newaxis,...]))
                cv2.imwrite(os.path.join(merge_dir,view[0].image_name),blend)
            res_ind=res_i if res_ind is None else np.hstack((res_ind,res_i))
            res_grp=res_g if res_grp is None else np.hstack((res_grp,res_g))
        for i,note in [(0,'_count'),(1,'_area'),(2,'_area_pct')]:
            df=pd.DataFrame(res_ind[...,i::3],index=pair.img_set.images,columns=pair.targets)
            to_excel_sheet(df,xls_file,pair.origin+note) # per slice
            df=pd.DataFrame(res_grp[...,i::3],index=batch.keys(),columns=pair.targets)
            to_excel_sheet(df,xls_file,pair.origin+note+"_sum") # per whole image

class ImagePatchPair:
    def __init__(self,cfg:BaseNetM,wd,origin,targets,is_train):
        self.cfg=cfg
        self.wd=wd
        self.origin=origin
        self.targets=targets if isinstance(targets,list) else [targets]
        self.is_train=is_train
        self.img_set=ViewSet(cfg,wd,origin,is_train).prep_folder()
        self.pch_set=None
        self.view_coord=sorted(self.img_set.view_coord, key=lambda x: x.file_name)
        self.tr_list,self.val_list=self.cfg.split_train_val_vc(self.view_coord)

    def blend_image_patch(self,additional_weight,patch_per_pixel,max_instance,bright_diff,max_std,adjacent_size,adjacent_std):
        img_folder=os.path.join(self.wd,self.img_set.category_detail('+'.join([self.origin]+self.targets)))
        img_exist=mkdir_ifexist(img_folder)
        pch_folder=os.path.join(self.wd, self.img_set.category_detail('-'.join([self.origin]+self.targets)))
        pch_exist=mkdir_ifexist(pch_folder) # tgt_set matchs img_set
        print('Image folder exist? %r'%img_exist) # e.g., Original+LYM+MONO+PMN
        print('Patch folder exist? %r' %pch_exist) # e.g., Original_LYM_MONO_PMN
        if img_exist and pch_exist:
            print("Skip making image/patch folders since both exist"); return
        else:
            print("Create new folders and blend images and patches")
        pixels=self.cfg.row_in*self.cfg.col_in
        print('processing %d categories on val #%d vs tr #%d'%(self.cfg.num_targets,len(self.val_list),len(self.tr_list)))
        with open(img_folder+".csv","w") as csv_class: # cow,0 \n   cat,1 \n  bird,2
            for ti,tgt in enumerate(self.targets):
                csv_class.write(tgt+","+str(ti)+"\n")
        with open(pch_folder+".csv","w") as csv_annot, open(pch_folder+"_val.csv","w") as csv_val: # /data/imgs/img_001.jpg,837,346,981,456,cow,/data/masks/img_001_001.png
            for vi, vc in enumerate(self.img_set.view_coord):
                is_validation=vc in self.val_list # fewer in val_list, faster to check
                pool=list(range(0,self.cfg.num_targets)) + additional_weight # equal chance, + weight to some category
                nexample=random.randint(pixels//patch_per_pixel[1], pixels//patch_per_pixel[0])
                labels=random.choices(pool, k=nexample)
                img=vc.get_image(os.path.join(self.img_set.work_directory, self.img_set.target_folder))
                # cv2.imwrite(os.path.join(tgt_noise.work_directory,tgt_noise.sub_folder,'_'+vc.image_name),img)
                inserted=[0]*self.cfg.num_targets # track # of inserts per category
                for li in labels:
                    the_tgt=self.pch_set[li]
                    idx=random.choice(the_tgt.val_list) if is_validation else random.choice(the_tgt.tr_list)
                    rowpos=random.uniform(0,1)
                    colpos=random.uniform(0,1)
                    prev=img.copy()
                    patch=the_tgt.view_coord[idx]
                    p_row, p_col, p_ave, p_std=patch.ori_row, patch.ori_col, patch.row_start, patch.row_end
                    lri=int(self.cfg.row_in*rowpos)-p_row//2  # large row in/start
                    lci=int(self.cfg.col_in*colpos)-p_col//2  # large col in/start
                    lro, lco=lri+p_row, lci+p_col  # large row/col out/end
                    pri=0 if lri>=0 else -lri; lri=max(0, lri)
                    pci=0 if lci>=0 else -lci; lci=max(0, lci)
                    pro=p_row if lro<=self.cfg.row_in else p_row-lro+self.cfg.row_in; lro=min(self.cfg.row_in, lro)
                    pco=p_col if lco<=self.cfg.col_in else p_col-lco+self.cfg.col_in; lco=min(self.cfg.col_in, lco)
                    # if np.average(img[lri:lro,lci:lco])-p_ave > self.bright_diff and \
                    if np.min(img[lri:lro, lci:lco])-p_ave>bright_diff \
                        and np.std(img[lri:lro, lci:lco])<max_std\
                        and int(np.std(img[lri-p_row*adjacent_size:lro+p_row*adjacent_size,lci-p_col*adjacent_size:lco+p_col*adjacent_size])>adjacent_std*p_std
                        ):  # target area is brighter, then add patch
                        # print("large row(%d) %d-%d col(%d) %d-%d  patch row(%d) %d-%d col(%d) %d-%d"%(self.cfg.row_in,lri,lro,self.cfg.col_in,lci,lco,p_row,pri,pro,p_col,pci,pco))
                        pat=the_tgt.image_data[idx]
                        # cv2.imwrite('pre.jpg',pat)
                        pat=augment_patch(pat,0 if is_validation else self.cfg.train_aug)
                        # cv2.imwrite('post.jpg',pat)
                        img[lri:lro, lci:lco]=np.minimum(img[lri:lro, lci:lco], pat[pri:pro, pci:pco].astype(np.uint8))
                        # img[lri:lro, lci:lco]-=pat[pri:pro, pci:pco].astype(np.uint8) # subtract
                        # cv2.imwrite(os.path.join(self.wd, the_tgt.sub_folder+'+',vc.file_name_insert(cfg,'_'+str(idx)+('' if index>self.cfg.train_vali_split else '^'))),
                        #             smooth_brighten(prev-img))
                        cv2.imwrite(os.path.join(pch_folder,vc.file_name_insert(self.cfg,'_%d^%d^'%(idx,li+1))), #+('' if index>self.cfg.train_vali_split else '^'))
                                    smooth_brighten(prev-img))
                        annot='%s,%d,%d,%d,%d,%s,%s\n'%(os.path.join(img_folder, vc.file_name),lci,lri,lco,lro,
                                  self.targets[li],os.path.join(pch_folder,vc.file_name_insert(self.cfg,'_%d^%d^'%(idx,li+1))))
                        csv_val.write(annot) if is_validation else csv_annot.write(annot)
                        # lr=(lri+lro)//2
                        # lc=(lci+lco)//2
                        # msk[lr:lr+1,lc:lc+1,1]=255
                        inserted[li]+=1
                        if inserted[li]>max_instance: break;
                if sum(inserted)>0:
                    print("inserted %s for %s"%(inserted,vc.file_name))
                    cv2.imwrite(os.path.join(img_folder, vc.file_name), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                else:
                    print("unable to insert for %s"%vc.file_name)


    def train_generator(self):
        self.pch_set=[ImageSet(self.cfg,self.wd,tgt,self.is_train).prep_folder() for tgt in self.targets]
        self.blend_image_patch(
            additional_weight=[1,1],  # if you have three categories default=[0,1,2] to draw from, here you can add more weights
            patch_per_pixel=[2000,18000],  # patch per pixel, range to randomly select from, larger number: smaller density
            max_instance=20,  # break out if more than this amount was inserted
            bright_diff=-10,  # original area should be clean, brighter than patch (original_brightness-patch_brightness>diff)
            max_std=40,  # original area should be clean, standard deviation should be low (< max_std)
            adjacent_size=3,  # sample times of size adjacent to the patch
            # adjacent_std=0.2  # std of adjacent area > x of patch std (>0: only add patch near existing object, 0: add regardless)
            adjacent_std=0.0  # std of adjacent area > x of patch std (>0: only add patch near existing object, 0: add regardless)
        )
        yield(ImagePatchGenerator(self, self.cfg.train_aug, self.targets,view_coord=self.tr_list),
              ImagePatchGenerator(self, 0, self.targets,view_coord=self.val_list),
              self.img_set.category_detail(self.cfg.join_targets(self.targets)))

    def predict_generator_note(self):
        yield (self.cfg.join_targets(self.targets),self.targets)

    def predict_generator_partial(self,subset,view):
        return ImagePatchGenerator(self,0,subset,view_coord=view),self.cfg.join_targets(subset)

class ImagePatchGenerator(keras.utils.Sequence):
    def __init__(self, pair:ImagePatchPair, aug_value, tgt_list, view_coord=None):
        self.pair=pair
        self.cfg=pair.cfg
        self.getitemfun,self._active_class_ids,self._anchors=None,None,None
        if self.cfg.is_train: # train
            self.set_train()
        else: # prediction
            self.set_pred()
        self.aug_value=aug_value
        self.target_list=tgt_list
        self.view_coord=pair.view_coord if view_coord is None else view_coord
        self.indexes=None
        self.on_epoch_end()

    def set_train(self):
        self.getitemfun=self.get_train_item
        self._active_class_ids=np.ones([self.cfg.num_class],dtype=np.int32)
        self._anchors=self.cfg.get_anchors_norm()[0]
    def set_eval(self):
        self.getitemfun=self.get_eval_item
        self._active_class_ids=np.zeros([self.cfg.num_class],dtype=np.int32)
        self._anchors=self.cfg.get_anchors_norm()[1]
    def set_pred(self):
        self.getitemfun=self.get_pred_item
        self._active_class_ids=np.zeros([self.cfg.num_class],dtype=np.int32)
        self._anchors=self.cfg.get_anchors_norm()[1]

    def get_train_item(self,indexes):
        _img,_msk,_cls,_bbox=None,None,None,None
        _img_meta,_rpn_match,_rpn_bbox=None,None,None
        # _tgt = np.zeros((self.cfg.batch_size, self.cfg.row_out, self.cfg.col_out, self.cfg.dep_out), dtype=np.uint8)
        for vi,vc in enumerate([self.view_coord[k] for k in indexes]):
            this_img=vc.get_image(os.path.join(self.pair.wd,self.pair.img_set.category_detail('+'.join([self.pair.origin]+self.pair.targets))))
            this_msk,this_cls=vc.get_masks(os.path.join(self.pair.wd,self.pair.img_set.category_detail('-'.join([self.pair.origin]+self.pair.targets))),self.cfg)
            # cv2.imwrite("pre1.jpg",this_img); cv2.imwrite("pre2.jpg",this_msk[...,0:3])
            this_img,this_msk=augment_image_set(this_img,this_msk,_level=self.aug_value)  # integer N: a <= N <= b.
            # cv2.imwrite("post1.jpg",this_img); cv2.imwrite("post2.jpg",this_msk[...,0:3])
            this_bbox=utils.extract_bboxes(this_msk)
            if self.cfg.mini_mask_shape is not None:
                this_msk=utils.minimize_mask(this_bbox,this_msk,tuple(self.cfg.mini_mask_shape[0:2]))
            if this_bbox.shape[0]>self.cfg.max_gt_instance:
                ids=np.random.choice(np.arange(this_bbox.shape[0]),self.cfg.max_gt_instance,replace=False)
                this_cls,this_bbox,this_msk=this_cls[ids],this_bbox[ids],this_msk[:,:,ids]
            this_img_meta=compose_image_meta(indexes[vi],self.cfg.dim_in,self.cfg.dim_in,(0,0,self.cfg.row_in,self.cfg.col_in),1.0,self._active_class_ids)
            this_rpn_match,this_rpn_bbox=build_rpn_targets(self.cfg.dim_in,self._anchors,this_cls,this_bbox,self.cfg.rpn_train_anchors_per_image,
                self.cfg.rpn_bbox_stdev)
            this_img,this_msk=this_img[np.newaxis,...],this_msk[np.newaxis,...]
            this_cls,this_bbox=this_cls[np.newaxis,...],this_bbox[np.newaxis,...]
            this_img_meta=this_img_meta[np.newaxis,...]
            this_rpn_match,this_rpn_bbox=this_rpn_match[np.newaxis,...,np.newaxis],this_rpn_bbox[np.newaxis,...]
            _img=this_img if _img is None else np.concatenate((_img,this_img),axis=0)
            _msk=this_msk if _msk is None else np.concatenate((_msk,this_msk),axis=0)
            _cls=this_cls if _cls is None else np.concatenate((_cls,this_cls),axis=0)
            _bbox=this_bbox if _bbox is None else np.concatenate((_bbox,this_bbox),axis=0)
            _img_meta=this_img_meta if _img_meta is None else np.concatenate((_img_meta,this_img_meta),axis=0)
            _rpn_match=this_rpn_match if _rpn_match is None else np.concatenate((_rpn_match,this_rpn_match),axis=0)
            _rpn_bbox=this_rpn_bbox if _rpn_bbox is None else np.concatenate((_rpn_bbox,this_rpn_bbox),axis=0)
            _img=prep_scale(_img,self.cfg.feed)
        return [_img,_img_meta,_rpn_match,_rpn_bbox,_cls,_bbox,_msk],[]

    def get_pred_item(self,indexes):
        _img,_img_meta,_anc=None,None,None
        for vi,vc in enumerate([self.view_coord[k] for k in indexes]):
            this_img=vc.get_image(os.path.join(self.pair.wd,self.pair.img_set.target_folder))
            this_img_meta=compose_image_meta(indexes[vi],self.cfg.dim_in,self.cfg.dim_in,(0,0,self.cfg.row_in,self.cfg.col_in),1.0,self._active_class_ids)
            this_img=this_img[np.newaxis,...]
            this_img_meta=this_img_meta[np.newaxis,...]
            this_anchors=self._anchors[np.newaxis,...]
            _img=this_img if _img is None else np.concatenate((_img,this_img),axis=0)
            _img_meta=this_img_meta if _img_meta is None else np.concatenate((_img_meta,this_img_meta),axis=0)
            _anc=this_anchors if _anc is None else np.concatenate((_anc,this_anchors),axis=0)
            _img=prep_scale(_img,self.cfg.feed)
        return [_img,_img_meta,_anc], []

    def get_eval_item(self,indexes):
        _img,_img_meta,_anc=None,None,None
        _cls,_bbox,_msk=None,None,None
        for vi,vc in enumerate([self.view_coord[k] for k in indexes]):
            this_img=vc.get_image(os.path.join(self.pair.wd,self.pair.img_set.category_detail('+'.join([self.pair.origin]+self.pair.targets))))
            this_msk,this_cls=vc.get_masks(os.path.join(self.pair.wd,self.pair.img_set.category_detail('-'.join([self.pair.origin]+self.pair.targets))),self.cfg)
            this_bbox=utils.extract_bboxes(this_msk)
            this_img_meta=compose_image_meta(indexes[vi],self.cfg.dim_in,self.cfg.dim_in,(0,0,self.cfg.row_in,self.cfg.col_in),1.0,self._active_class_ids)
            this_img=this_img[np.newaxis,...]
            this_img_meta=this_img_meta[np.newaxis,...]
            this_anchors=self._anchors[np.newaxis,...]
            _img=this_img if _img is None else np.concatenate((_img,this_img),axis=0)
            _img_meta=this_img_meta if _img_meta is None else np.concatenate((_img_meta,this_img_meta),axis=0)
            _anc=this_anchors if _anc is None else np.concatenate((_anc,this_anchors),axis=0)
            _img=prep_scale(_img,self.cfg.feed)
            this_cls=this_cls[np.newaxis,...]
            this_bbox=this_bbox[np.newaxis,...]
            this_msk=this_msk[np.newaxis,...]
            _cls=this_cls if _cls is None else np.concatenate((_cls,this_cls),axis=0)
            _bbox=this_bbox if _bbox is None else np.concatenate((_bbox,this_bbox),axis=0)
            _msk=this_msk if _msk is None else np.concatenate((_msk,this_msk),axis=0)
        return [_img,_img_meta,_anc], [_cls,_bbox,_msk]

    def __len__(self):  # Denotes the number of batches per epoch
        return int(np.ceil(len(self.view_coord) / self.cfg.batch_size))

    def __getitem__(self, index):  # Generate one batch of data
        indexes=self.indexes[index*self.cfg.batch_size:(index+1)*self.cfg.batch_size]
        # print(" getting index %d with %d batch size"%(index,self.batch_size))
        return self.getitemfun(indexes)

    def on_epoch_end(self):  # Updates indexes after each epoch
        self.indexes=np.arange(len(self.view_coord))
        if self.cfg.is_train and self.cfg.train_shuffle:
            np.random.shuffle(self.indexes)

# Anchors #

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()
    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)
    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])
    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes

def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides, anchor_stride):
    # Anchors [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i], feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)

# Parse Detection #
def parse_detections(detections,mrcnn_mask,original_image_shape,full_mask=False,image_shape=None,window=None):
    image_shape=image_shape or original_image_shape
    window=window or (0,0,image_shape[0],image_shape[1])
    zero_ix=np.where(detections[:,4]==0)[0]
    n=zero_ix[0] if zero_ix.shape[0]>0 else detections.shape[0]
    # Extract boxes, class_ids, scores, and class-specific masks
    boxes=detections[:n,:4]
    class_ids=detections[:n,4].astype(np.int32)
    scores=detections[:n,5]

    # Translate normalized coordinates in the resized image to pixel coordinates in the original image before resizing
    window=utils.norm_boxes(window,image_shape[:2])
    wy1,wx1,wy2,wx2=window
    shift=np.array([wy1,wx1,wy1,wx1])
    wh=wy2-wy1  # window height
    ww=wx2-wx1  # window width
    scale=np.array([wh,ww,wh,ww])
    # Convert boxes to normalized coordinates on the window
    boxes=np.divide(boxes-shift,scale)
    # Convert boxes to pixel coordinates on the original image
    boxes=utils.denorm_boxes(boxes,original_image_shape[:2])

    if full_mask:
        # masks=np.zeros((N,image_shape[0],image_shape[1]),dtype=np.bool)
        # for i,cls in enumerate(class_ids):
        #     masks[i]=utils.unmold_mask(mrcnn_mask[i,:,:,cls],boxes[i],image_shape)
        masks=np.zeros((n,image_shape[0],image_shape[1]),dtype=np.bool)
        for i,cls in enumerate(class_ids):
            # Convert neural network mask to full size mask
            masks[i]=utils.unmold_mask(mrcnn_mask[i,:,:,cls],boxes[i],image_shape)
    else:
        masks=mrcnn_mask[np.arange(n),:,:,class_ids] # default mini mask

    # exclude_ix=np.where((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])<=0)[0] # Only filter zero area detectionsm, may happens in early training

    exclude_ix=np.where(
        np.logical_or((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])<=0, # remove zero area
            # np.logical_or( # more than half outside the frame
            #     np.logical_or(boxes[:,2]+boxes[:,0]>2*image_shape[0],boxes[:,2]+boxes[:,0]<0),
            #     np.logical_or(boxes[:,3]+boxes[:,1]>2*image_shape[1],boxes[:,3]+boxes[:,1]<0)
            # )
            # np.logical_or(  # even touching the frame
            #     np.logical_or(boxes[:,2]>=image_shape[0],boxes[:,0]<=0),
            #     np.logical_or(boxes[:,3]>=image_shape[1],boxes[:,1]<=0)
            # )
            np.logical_or((boxes[:,2]-boxes[:,0])/(boxes[:,3]-boxes[:,1])>2.0,(boxes[:,2]-boxes[:,0])/(boxes[:,3]-boxes[:,1])<0.5)
    ))[0] # also filter boxes have more than half outside the window

    if exclude_ix.shape[0]>0:
        boxes=np.delete(boxes,exclude_ix,axis=0)
        class_ids=np.delete(class_ids,exclude_ix,axis=0)
        scores=np.delete(scores,exclude_ix,axis=0)
        masks=np.delete(masks,exclude_ix,axis=0)
        n=class_ids.shape[0]

    return boxes,class_ids,scores,masks


# Proposal Layer #

def apply_box_deltas_graph(boxes, deltas):
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result

def clip_boxes_graph(boxes, window):
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped

class ProposalLayer(KE.Layer):
    def __init__(self,proposal_count,rpn_nms_threshold,rpn_bbox_stdev,pre_nms_limit,images_per_gpu,**kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.rpn_nms_threshold = rpn_nms_threshold
        self.rpn_bbox_stdev = rpn_bbox_stdev
        self.pre_nms_limit=pre_nms_limit
        self.images_per_gpu = images_per_gpu

    def call(self,inputs,**kwargs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.rpn_bbox_stdev, [1, 1, 4])
        anchors = inputs[2]
        # Improve performance by trimming to top anchors by score and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.pre_nms_limit, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y), self.images_per_gpu)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y), self.images_per_gpu)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x), self.images_per_gpu, names=["pre_nms_anchors"])
        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = utils.batch_slice([pre_nms_anchors, deltas], lambda x, y: apply_box_deltas_graph(x, y), self.images_per_gpu, names=["refined_anchors"])
        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes, lambda x: clip_boxes_graph(x, window), self.images_per_gpu, names=["refined_anchors_clipped"])
        # Non-max suppression
        def nms(_boxes,_scores):
            indices = tf.image.non_max_suppression(
                _boxes, _scores, self.proposal_count,
                self.rpn_nms_threshold, name="rpn_non_max_suppression")
            _proposals = tf.gather(_boxes,indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(_proposals)[0], 0)
            _proposals = tf.pad(_proposals, [(0, padding), (0, 0)])
            return _proposals
        proposals = utils.batch_slice([boxes, scores], nms, self.images_per_gpu)
        return proposals

    def compute_output_shape(self, input_shape):
        return None,self.proposal_count,4


# ROIAlign Layer #

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.log(x) / tf.log(2.0)

class PyramidROIAlign(KE.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.
    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]
    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]
    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self,inputs,**kwargs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )


# Detection Target Layer #

def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),[1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals,gt_class_ids,gt_boxes,gt_masks,train_rois_per_image,roi_positive_ratio,mini_mask_shape,bbox_stdev):
    asserts = [tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals], name="roi_assertion"), ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)
    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2, name="trim_gt_masks")
    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude from training. A crowd box is given a negative class ID.
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    crowd_masks = tf.gather(gt_masks, crowd_ix, axis=2)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)
    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)
    # Compute overlaps with crowd boxes [proposals, crowd_boxes]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)
    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]
    # Subsample ROIs. Aim for 33% positive Positive ROIs
    positive_count = int(train_rois_per_image * roi_positive_ratio)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / roi_positive_ratio
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)
    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)
    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= bbox_stdev
    # Assign positive ROIs to GT masks Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)
    # Compute mask targets
    boxes = positive_rois
    if mini_mask_shape: # Transform ROI coordinates from normalized image space to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32),boxes,box_ids,mini_mask_shape[0:2])
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)
    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)
    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(train_rois_per_image - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0,    P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])
    return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KE.Layer):

    def __init__(self,images_per_gpu,train_rois_per_image,train_roi_positive_ratio,mini_mask_shape,rpn_bbox_stdev,**kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.images_per_gpu=images_per_gpu
        self.train_rois_per_image=train_rois_per_image
        self.train_roi_positive_ratio=train_roi_positive_ratio
        self.mini_mask_shape=mini_mask_shape
        self.bbox_stdev=rpn_bbox_stdev

    def call(self,inputs,**kwargs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice([proposals, gt_class_ids, gt_boxes, gt_masks],
                                    lambda w, x, y, z: detection_targets_graph(w,x,y,z, self.train_rois_per_image,self.train_roi_positive_ratio,
                                                               self.mini_mask_shape,self.bbox_stdev),self.images_per_gpu,names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.train_rois_per_image, 4),  # rois
            (None, self.train_rois_per_image),  # class_ids
            (None, self.train_rois_per_image, 4),  # deltas
            (None, self.train_rois_per_image, self.mini_mask_shape[0],
             self.mini_mask_shape[1])  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]


# Detection Layer #

def refine_detections_graph(rois, probs, deltas, window, bbox_stdev, detection_min_confidence, detection_max_instances, detection_nms_threshold):
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(rois, deltas_specific *bbox_stdev)
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if detection_min_confidence:
        conf_keep = tf.where(class_scores >= detection_min_confidence)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=detection_max_instances,
                iou_threshold=detection_nms_threshold)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = detection_max_instances - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([detection_max_instances])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = detection_max_instances
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = detection_max_instances - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


class DetectionLayer(KE.Layer):

    def __init__(self,rpn_bbox_stdev,detection_min_confidence,detection_max_instances,detection_nms_threshold,gpu_count,images_per_gpu,**kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.bbox_stdev=rpn_bbox_stdev
        self.detection_min_confidence=detection_min_confidence
        self.detection_max_instances=detection_max_instances
        self.detection_nms_threshold=detection_nms_threshold
        self.gpu_count=gpu_count
        self.images_per_gpu=images_per_gpu
        self.batch_size=self.gpu_count*self.images_per_gpu

    def call(self,inputs,**kwargs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice([rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.bbox_stdev,self.detection_min_confidence,
                                   self.detection_max_instances,self.detection_nms_threshold),self.batch_size)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return tf.reshape(detections_batch, [self.batch_size, self.detection_max_instances, 6])

    def compute_output_shape(self, input_shape):
        return None,self.detection_max_instances,6


# Feature Pyramid Network Heads #

def fpn_classifier_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True, fc_layers_size=1024):
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size], name="roi_align_classifier")([rois, image_meta] + feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"), name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)), name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="pool_squeeze")(x)
    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes), name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"), name="mrcnn_class")(mrcnn_class_logits)
    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'), name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)
    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True):
    # ROI Pooling Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size], name="roi_align_mask")([rois, image_meta] + feature_maps)
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"), name="mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"), name="mrcnn_mask")(x)
    return x

# Loss Functions #

def smooth_l1_loss(y_true, y_pred):
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    # rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive, -1=negative, 0=neutral anchor.
    # rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    rpn_match = tf.squeeze(rpn_match, -1)
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    indices = tf.where(K.not_equal(rpn_match, 0))
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    loss = K.sparse_categorical_crossentropy(target=anchor_class, output=rpn_class_logits, from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss

def rpn_bbox_loss_graph(image_per_gpu, target_bbox, rpn_match, rpn_bbox):
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts, image_per_gpu)
    loss = smooth_l1_loss(target_bbox, rpn_bbox)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss

def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids):
    target_class_ids = tf.cast(target_class_ids, 'int64') # cast from default float32
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class_ids, logits=pred_class_logits)
    loss = loss * pred_active
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss

def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    target_class_ids = K.reshape(target_class_ids, (-1,)) # Reshape to merge batch and roi dimensions
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)
    loss = K.mean(K.switch(tf.size(target_bbox) > 0, smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox), tf.constant(0.0)))
    return loss

def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2]) # Permute to [N, num_classes, height, width]
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)
    loss = K.mean(K.switch(tf.size(y_true) > 0, K.binary_crossentropy(target=y_true, output=y_pred), tf.constant(0.0)))
    return loss

def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(gt_class_ids.dtype)
    assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(gt_boxes.dtype)
    assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(gt_masks.dtype)
    # It's common to add GT Boxes to ROIs but we don't do that here because
    # according to XinLei Chen's paper, it doesn't help.
    # Trim empty padding in gt_boxes and gt_masks parts
    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, "Image must contain instances."
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]
    gt_masks = gt_masks[:, :, instance_ids]
    # Compute areas of ROIs and ground truth boxes.
    rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * (rpn_rois[:, 3] - rpn_rois[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    # Compute overlaps [rpn_rois, gt_boxes]
    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i]
        overlaps[:, i] = utils.compute_iou(gt, rpn_rois, gt_box_area[i], rpn_roi_area)
    # Assign ROIs to GT boxes
    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
    rpn_roi_iou_max = overlaps[np.arange(overlaps.shape[0]), rpn_roi_iou_argmax]
    # GT box assigned to each ROI
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]
    # Positive ROIs are those with >= threshold 0.5 IoU with a GT box.
    fg_ids = np.where(rpn_roi_iou_max > config.DETECTION_MASK_THRESHOLD)[0]
    # Negative ROIs are those with max IoU <= threshold 0.1-0.5 (hard example mining)
    # bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
    bg_ids = np.where(rpn_roi_iou_max < config.DETECTION_MASK_THRESHOLD)[0]
    # Subsample ROIs. Aim for 33% foreground.
    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        # not enough samples to maintain the desired balance. Reduce requirements and fill in the rest, likely different from the Mask RCNN paper.
        if keep.shape[0] == 0:
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            keep_extra_ids = np.random.choice(
                keep_bg_ids, remaining, replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, "keep doesn't match ROI batch size {}, {}".format(keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)
    # Reset the gt boxes assigned to BG ROIs.
    rpn_roi_gt_boxes[keep_bg_ids, :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0
    # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
    roi_gt_assignment = rpn_roi_iou_argmax[keep]
    # Class-aware bbox deltas. [y, x, log(h), log(w)]
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE,
                       config.NUM_CLASSES, 4), dtype=np.float32)
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(
        rois[pos_ids], roi_gt_boxes[pos_ids, :4])
    # Normalize bbox refinements
    bboxes /= config.BBOX_STD_DEV
    # Generate class-specific target masks
    masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES),
                     dtype=np.float32)
    for i in pos_ids:
        class_id = roi_gt_class_ids[i]
        assert class_id > 0, "class id must be greater than 0"
        gt_id = roi_gt_assignment[i]
        class_mask = gt_masks[:, :, gt_id]
        if config.USE_MINI_MASK:
            # Create a mask placeholder, the size of the image
            placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1
            placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = np.round(utils.resize(class_mask, (gt_h, gt_w))).astype(bool)
            # Place the mini batch in the placeholder
            class_mask = placeholder
        # Pick part of the mask and resize it
        y1, x1, y2, x2 = rois[i].astype(np.int32)
        m = class_mask[y1:y2, x1:x2]
        mask = utils.resize(m, config.MASK_SHAPE)
        masks[i, :, :, class_id] = mask
    return rois, roi_gt_class_ids, bboxes, masks


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, rpn_train_anchors_per_image, rpn_bbox_stdev):
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((rpn_train_anchors_per_image, 4))
    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (rpn_train_anchors_per_image // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (rpn_train_anchors_per_image -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # For positive anchors, compute shift and scale needed to transform them to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        gt = gt_boxes[anchor_iou_argmax[i]]
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        rpn_bbox[ix] /= rpn_bbox_stdev
        ix += 1
    return rpn_match, rpn_bbox

# Data Formatting #
def compose_image_meta(image_id, original_image_shape, image_shape,window, scale, active_class_ids):
    return np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )

def parse_image_meta_graph(meta):
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


# Miscellenous Graph Functions #

def trim_zeros_graph(boxes, name=None):
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool) # 1D boolean mask identifying the rows to keep
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    outputs = [] # Picks different number of values from each row in x depending on the values in counts.
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    h, w = tf.split(tf.cast(shape, tf.float32), 2) # height, width
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)  # pixel coordinates outside -> inside the box


def denorm_boxes_graph(boxes, shape):
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)

