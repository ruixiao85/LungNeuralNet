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

from a_config import Config,get_proper_range
from c2_mrcnn_matterport import norm_boxes_graph,parse_image_meta_graph,DetectionTargetLayer,fpn_classifier_graph,rpn_class_loss_graph,rpn_bbox_loss_graph,\
    mrcnn_class_loss_graph,mrcnn_bbox_loss_graph,mrcnn_mask_loss_graph,ProposalLayer,build_fpn_mask_graph,DetectionLayer,generate_pyramid_anchors,\
    parse_detections,compose_image_meta,build_rpn_targets,norm_boxes,compute_ap,non_max_suppression,extract_bboxes,minimize_mask
from image_set import ImageSet,ViewSet,PatchSet
from osio import mkdir_ifexist,to_excel_sheet,mkdir_dir,mkdirs_dir
from postprocess import g_kern_rect,draw_text,draw_detection,morph_close
from preprocess import prep_scale,read_image,read_resize,AugPatchMask,read_mask_default_zeros


class BaseNetM(Config):
    def __init__(self,**kwargs):
        super(BaseNetM,self).__init__(**kwargs)
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
        self.coverage_predict=-1 # do not allow overlap during prediction/inference
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
        self.rpn_anchor_ratios=kwargs.get('rpn_anchor_ratio', [0.75,1,1.33]) # Ratios of anchors at each cell (width/height) 1=square 0.5=wide
        # self.rpn_anchor_ratios=kwargs.get('rpn_anchor_ratio', [0.5,1,2]) # Ratios of anchors at each cell (width/height) 1=square 0.5=wide
        self.rpn_anchor_stride=kwargs.get('rpn_anchor_stride', 1) # 1=no-skip cell 2=skip-one
        self.rpn_nms_threshold=kwargs.get('rpn_nms_threshold', 0.9) # Non-max suppression threshold to filter RPN proposals. larger=more propsals.
        self.rpn_bbox_stdev=kwargs.get('rpn_bbox_stdev', np.array([0.1,0.1,0.2,0.2])) # Bounding box refinement standard deviation for RPN and final detections.
        self.pre_nms_limit=kwargs.get('pre_nms_limit', 6000) # ROIs kept after tf.nn.top_k and before non-maximum suppression
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
        # self.detection_min_confidence=kwargs.get('detection_min_confidence', 0.0) # Minimum probability to accept a detected instance, skip ROIs if below this threshold, lower if fewer classes
        self.detection_nms_threshold=kwargs.get('detection_nms_threshold', 0.3) # Non-maximum suppression threshold for detection
        self.detection_mask_threshold=kwargs.get('detection_mask_threshold', 0.5) # threshold to determine fore/back-ground
        self.gpu_count=kwargs.get('gpu_count', 1)
        self.images_per_gpu=kwargs.get('image_per_gpu', 1)
        self.filename=kwargs.get('filename', None)
        self.ntop=15 # override parent class to keep more top networks for further MRCNN evaluation
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
            self._anchor_cache[self.dim_in]=[anchors,norm_boxes(anchors.copy(),self.dim_in[:2])]
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
                                           validation_steps=min(self.train_val_step,len(val.view_coord)) if isinstance(self.train_val_step,int) else len(val.view_coord),
                                           epochs=self.train_epoch,max_queue_size=5,workers=1,use_multiprocessing=False,shuffle=False,initial_epoch=init_epoch,
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

    def eval(self,pair):
        self.build_net(is_train=False)
        for tr,val,dir_out in pair.train_generator():
            self.filename=dir_out+'_'+str(self)
            print('Evaluating neural net...')
            weight_files=self.find_best_models(self.filename+'^*^.h5',allow_cache=True)
            for weight_file in weight_files:
                with open(self.filename+".log","a") as log:
                    log.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+','+weight_file+',')
                    self.net.load_weights(weight_file,by_name=True)  # weights only
                    for part in [tr,val]:
                        part.set_eval()
                        steps_done,steps=0,min(48,len(part)) # limit number of evaluation images
                        print("[%s] is_val=%r running %d/%d steps:\nAP:"%(weight_file,part.is_val,steps,len(part)),end='')
                        part.on_epoch_end() #initialize
                        valiter=iter(part); APs=[]
                        while steps_done<steps:
                            img,gt=next(valiter)
                            detections,mrcnn_class,mrcnn_bbox,mrcnn_mask,rpn_rois,rpn_class,rpn_bbox=self.net.predict_on_batch(img)
                            for i in range(np.shape(detections)[0]):
                                final_rois,final_class_ids,final_scores,final_masks=parse_detections(detections[i],mrcnn_mask[i],self.dim_in,full_mask=True) # first element
                                AP,precisions,recalls,overlaps=compute_ap(gt[1][i],gt[0][i],gt[2][i],final_rois,final_class_ids,final_scores,
                                    np.transpose(final_masks,(1,2,0)))
                                APs.append(AP); print(' %.2f'%AP,end='',flush=True)
                            steps_done+=1
                        mAP=np.mean(APs)
                        print("\nmAP: ",mAP); log.write(str(mAP)+',')
                    print(); log.write('\n')

    def predict(self,pair,pred_dir):
        self.build_net(is_train=False)
        xls_file,cfg=os.path.join(pred_dir,"%s_%s_%s.xlsx"%(pair.origin,pred_dir.split(os.path.sep)[-1],repr(self))),str(self)
        params=["Area","Count","AreaPercentage","CountDensity"]
        regions=["Total","ConductingAirway","RespiratoryAirway","ConnectiveTissue","LargeBloodVessel","SmallBloodVessel"]
        # msks=[ViewSet(self,pair.wd,r,is_train=True,channels=1,low_std_ex=True).prep_folder() for r in regions]
        batch,view_name=pair.img_set.view_coord_batch()  # image/1batch -> view_coord
        save_ind,save_raw,save_msk=pair.cfg.save_ind_raw_msk
        save_raw,out_scale=(True,pair.img_set.raw_scale) if (save_raw and pair.img_set.resize_ratio!=1.0) else (False,pair.img_set.target_scale)
        res_ind,res_grp=None,None
        for dir_out,tgt_list in pair.predict_generator_note():
            res_i,res_g,regmap=None,None,None
            print('Load model and predict to [%s]...'%dir_out)
            ind_dir=mkdir_dir(os.path.join(pred_dir,"%s-%s_%.1f_%s"%(pair.origin,dir_out,pair.img_set.target_scale,cfg))) if save_ind else None  # ind view
            grp_dir=mkdir_dir(os.path.join(pred_dir,"%s-%s_%.1f+%s"%(pair.origin,dir_out,out_scale,cfg)))
            mask_dirs=[mkdir_dir(os.path.join(pred_dir,"%s_%s"%(tgt,out_scale))) for tgt in tgt_list] if save_msk else None  # b/w masks
            for grp,view in batch.items():
                grp_box,grp_cls,grp_scr,grp_msk=None,None,None,None
                regmap={rn:read_mask_default_zeros(os.path.join(pred_dir,pair.img_set.label_scale(rn,out_scale),view[0].image_name),self.row_out,self.col_out)
                    for rn in regions if rn!="Total"}
                prd,tgt_name=pair.predict_generator_partial(tgt_list,view)
                weight_file=None
                for pat in ["%s_%s_%s^*^.h5"%(tgt_name,scale_res,cfg) for scale_res in [pair.img_set.scale_res(),pair.img_set.scale_allres()]]:
                    weight_list=self.find_best_models(pat,allow_cache=True)
                    if weight_list:
                        weight_file=weight_list[0]; break
                print(weight_file or "No trained neural network found.")
                self.net.load_weights(weight_file,by_name=True)  # weights only
                # self.net=load_model(weight_file,custom_objects=custom_function_dict()) # weight optimizer archtecture
                detections,mrcnn_class,mrcnn_bbox,mrcnn_mask,rpn_rois,rpn_class,rpn_bbox=\
                    self.net.predict_generator(prd,max_queue_size=5,workers=1,use_multiprocessing=False,verbose=1)
                mrg_in=np.zeros((view[0].ori_row,view[0].ori_col,self.dep_in),dtype=np.float32)
                for i,(det,msk) in enumerate(zip(detections,mrcnn_mask)): # each view
                    final_rois,final_class_ids,final_scores,final_masks=parse_detections(det,msk,self.dim_in)
                    origin=pair.img_set.get_image(view[i])
                    ri,ro,ci,co,tri,tro,tci,tco=get_proper_range(view[i].ori_row,view[i].ori_col,
                            view[i].row_start,view[i].row_end,view[i].col_start,view[i].col_end,  0,self.row_out,0,self.col_out)
                    mrg_in[ri:ro,ci:co]=origin[tri:tro,tci:tco]
                    if save_ind:
                        r_i,blend,_=self.predict_proc(self,origin,tgt_list,final_rois,final_class_ids,final_scores,final_masks)
                        res_i=r_i[np.newaxis,...] if res_i is None else np.concatenate([res_i,r_i[np.newaxis,...]],axis=0)
                        cv2.imwrite(mkdirs_dir(os.path.join(ind_dir,view[i].file_name)),blend)
                    y_d=view[i].row_start; x_d=view[i].col_start
                    final_rois[:,0]+=y_d; final_rois[:,2]+=y_d
                    final_rois[:,1]+=x_d; final_rois[:,3]+=x_d
                    grp_box=final_rois if grp_box is None else np.concatenate((grp_box,final_rois))
                    grp_cls=final_class_ids if grp_cls is None else np.concatenate((grp_cls,final_class_ids))
                    grp_scr=final_scores if grp_scr is None else np.concatenate((grp_scr,final_scores))
                    grp_msk=final_masks if grp_msk is None else np.concatenate((grp_msk,final_masks))
                if save_raw:
                    mrg_in=read_image(os.path.join(pred_dir,pair.img_set.raw_folder,view[0].image_name))
                    grp_box=(grp_box.astype(np.float32)/pair.img_set.resize_ratio).astype(np.int32)
                r_g,blend,bw=self.predict_proc(self,mrg_in,tgt_list,grp_box,grp_cls,grp_scr,grp_msk,reg=regmap)
                res_g=r_g[np.newaxis,...] if res_g is None else np.concatenate((res_g,r_g[np.newaxis,...]))
                cv2.imwrite(mkdirs_dir(os.path.join(grp_dir,view[0].image_name)),blend)
                if save_msk:
                    [cv2.imwrite(mkdirs_dir(os.path.join(md,view[0].image_name)),bw[...,i]) for (i,md) in enumerate(mask_dirs)]
            res_ind=res_i if res_ind is None else np.hstack((res_ind,res_i))
            res_grp=res_g if res_grp is None else np.hstack((res_grp,res_g))
        if save_ind:
            df=pd.DataFrame(res_ind.reshape(len(view_name),-1),index=pd.MultiIndex.from_product([view_name],names=["view_name"]),
                columns=pd.MultiIndex.from_product([pair.targets,params],names=["targets","params"]))
            to_excel_sheet(df,xls_file,pair.origin) # per slice
        df=pd.DataFrame(res_grp.reshape((len(batch)*len(regions),-1)),index=pd.MultiIndex.from_product([batch.keys(),regions],names=["image_name","regions"]),
            columns=pd.MultiIndex.from_product([pair.targets,params],names=["targets","params"]))
        to_excel_sheet(df,xls_file,pair.origin+"_sum") # per whole image

class ImagePatchPair:
    def __init__(self,cfg:BaseNetM,wd,origin,targets,is_train):
        self.cfg=cfg
        self.wd=wd
        self.origin=origin
        self.targets=targets if isinstance(targets,list) else [targets]
        self.is_train=is_train
        self.img_set=ViewSet(cfg,wd,origin,is_train,channels=3,low_std_ex=False).prep_folder()
        self.pch_set=None

    def train_generator(self):
        self.pch_set=[PatchSet(self.cfg,self.wd,tgt,self.is_train,channels=3).prep_folder() for tgt in self.targets]
        yield(ImagePatchGenerator(self,self.targets,view_coord=self.img_set.tr_view,aug_value=self.cfg.train_val_aug[0]),
              ImagePatchGenerator(self,self.targets,view_coord=self.img_set.val_view,aug_value=self.cfg.train_val_aug[1]),
              self.img_set.label_scale_res(self.cfg.join_targets(self.targets)))

    def predict_generator_note(self):
        yield (self.cfg.join_targets(self.targets),self.targets)

    def predict_generator_partial(self,subset,view):
        return ImagePatchGenerator(self,subset,view_coord=view,aug_value=0),self.cfg.join_targets(subset)


class ImagePatchGenerator(keras.utils.Sequence):
    def __init__(self,pair: ImagePatchPair,tgt_list,view_coord,aug_value):
        self.pair=pair
        self.cfg=pair.cfg
        self.getitemfun,self._active_class_ids,self._anchors=None,None,None
        self.aug=AugPatchMask(aug_value)
        self.target_list=tgt_list
        self.view_coord=view_coord
        self.is_val=view_coord[0] in pair.img_set.val_view
        if self.cfg.is_train: # train
            self.set_train()
        else: # prediction
            self.set_pred()
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
            this_img,this_cls,this_msk=self.blend_image_patch(vc,verbose=1) # always regenerate
            # this_img,this_cls,this_msk=vc.data=vc.data or self.blend_image_patch(vc) # reuse previously generated
            # cv2.imwrite("mult_pre_img.jpg",this_img); cv2.imwrite("mult_pre_msk.jpg",this_msk[...,0:3])
            this_img=self.aug.decor1(this_img)  # integer N: a <= N <= b.
            # cv2.imwrite("mult_post_img.jpg",this_img); cv2.imwrite("mult_post_msk.jpg",this_msk[...,0:3])
            this_bbox=extract_bboxes(this_msk)
            if self.cfg.mini_mask_shape is not None:
                this_msk=minimize_mask(this_bbox,this_msk,tuple(self.cfg.mini_mask_shape[0:2]))
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

    def get_eval_item(self,indexes):
        _img,_img_meta,_anc=None,None,None
        _cls,_bbox,_msk=None,None,None
        for vi,vc in enumerate([self.view_coord[k] for k in indexes]):
            # this_img,this_cls,this_msk=self.blend_image_patch(vc)  # always regenerate
            this_img,this_cls,this_msk=vc.data=vc.data or self.blend_image_patch(vc,verbose=0)  # reuse previously generated
            this_bbox=extract_bboxes(this_msk)
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

    def get_pred_item(self,indexes):
        _img,_img_meta,_anc=None,None,None
        for vi,vc in enumerate([self.view_coord[k] for k in indexes]):
            this_img=self.pair.img_set.get_image(vc)
            this_img_meta=compose_image_meta(indexes[vi],self.cfg.dim_in,self.cfg.dim_in,(0,0,self.cfg.row_in,self.cfg.col_in),1.0,self._active_class_ids)
            this_img=this_img[np.newaxis,...]
            this_img_meta=this_img_meta[np.newaxis,...]
            this_anchors=self._anchors[np.newaxis,...]
            _img=this_img if _img is None else np.concatenate((_img,this_img),axis=0)
            _img_meta=this_img_meta if _img_meta is None else np.concatenate((_img_meta,this_img_meta),axis=0)
            _anc=this_anchors if _anc is None else np.concatenate((_anc,this_anchors),axis=0)
            _img=prep_scale(_img,self.cfg.feed)
        return [_img,_img_meta,_anc],[]

    def blend_image_patch(self,view,verbose,**kwargs): # return img,cls,msk for each view, verbose 0:none 1:+ 2:details
        random_weight=kwargs.get('random_weight',6) # random weight for each category will be added
        patch_per_area=kwargs.get('patch_per_area',6000)  # divided by area, larger number -> fewer patches/smaller density
        max_instance=kwargs.get('max_instance',30)  # break out condition: >? patches inserted in any category
        ave_diff=kwargs.get('ave_diff',10)  # original area should be brighter (spot_ave-brightness-patch_ave-brightness>diff), neg-val: accept dim images
        min_diff=kwargs.get('min_diff',0)  # original area should be brighter (spot_min-brightness-patch_min-brightness>diff), neg-val: accept dim images
        std_diff=kwargs.get('std_diff',10)  # original area should be cleaner, lower std (spot_std-patch_std<diff), pos-val: accept contrasty background
        img=np.copy(self.pair.img_set.get_image(view))
        # cv2.imwrite("multi_%s_pre.jpg"%view.file_name,img)
        img=self.aug.shift1(img)
        # cv2.imwrite("multi_%s_shift.jpg"%view.file_name,img)
        area=int(self.cfg.row_in*self.cfg.col_in/self.cfg.target_scale)
        pool=list(range(0,self.cfg.num_targets+1)) # equal chance
        for _ in range(random_weight): pool.append(random.randint(0,self.cfg.num_targets)) # +random weight
        while True:
            inserted=[0]*self.cfg.num_targets  # track # of inserts per category
            nexample=max(3,area//random.randint(patch_per_area//2,patch_per_area*2))
            labels=random.choices([p for p in pool if p!=0],k=nexample)  # 0background 1,2,...foreground
            clss,msks=[],None
            for li in labels:
                the_pch_set=self.pair.pch_set[li-1]
                pch_view=random.choice(the_pch_set.val_view) if self.is_val else random.choice(the_pch_set.tr_view)
                rowpos=random.uniform(0,1)
                colpos=random.uniform(0,1)
                pat_img,pat_msk=the_pch_set.get_image(pch_view),the_pch_set.get_mask(pch_view)[...,np.newaxis]
                # cv2.imwrite(pch_view.image_name+"_pimg_0.jpg",pat_img);cv2.imwrite(pch_view.image_name+"_pmsk_0.jpg",pat_msk)
                pat_img,pat_msk=self.aug.shift2_decor1(pat_img,pat_msk) # only allow minimal augmentation, preverse [H,W,C]
                # cv2.imwrite(pch_view.image_name+"_pimg_%d.jpg"%self.aug_value,pat_img);cv2.imwrite(pch_view.image_name+"_pmsk_%d.jpg"%self.aug_value,pat_msk)
                p_row,p_col,_=pat_img.shape # insure fit may change image size, so get the updated size
                p_gray=np.mean(pat_img,axis=-1,keepdims=True)  # stats based on grayscale
                p_min,p_max,p_ave,p_std=np.min(p_gray),np.max(p_gray),np.average(p_gray),np.std(p_gray)
                lri=int(self.cfg.row_in*rowpos)-p_row//2  # large row in/start
                lci=int(self.cfg.col_in*colpos)-p_col//2  # large col in/start
                lro,lco=lri+p_row,lci+p_col  # large row/col out/end
                pri=0 if lri>=0 else -lri; lri=max(0,lri)
                pci=0 if lci>=0 else -lci; lci=max(0,lci)
                pro=p_row if lro<=self.cfg.row_in else p_row-lro+self.cfg.row_in; lro=min(self.cfg.row_in,lro)
                pco=p_col if lco<=self.cfg.col_in else p_col-lco+self.cfg.col_in; lco=min(self.cfg.col_in,lco)
                s_gray=np.mean(img[lri:lro,lci:lco],axis=-1)
                s_ave,s_min,s_std=np.average(s_gray),np.min(s_gray),np.std(s_gray)
                if s_ave>p_ave+ave_diff and s_min>p_min+min_diff and np.mean(s_std)<p_std+std_diff:
                    # cv2.imwrite("spot_acepted.jpg",img[lri:lro,lci:lco]); cv2.imwrite("patch_acepted.jpg",pat_img)
                    # img[lri:lro,lci:lco]=np.minimum(img[lri:lro,lci:lco],pat_img[pri:pro,pci:pco].astype(np.uint8)) # darken
                    img[lri:lro,lci:lco]=((img[lri:lro,lci:lco]).astype(np.float16)*pat_img[pri:pro,pci:pco]/255.0).astype(np.uint8)
                    clss.append(li) # 1,2,3 becaue zero is reserved for background
                    msk=np.zeros((self.cfg.row_in,self.cfg.col_in,1),dtype=np.uint8) # np.uint8 0-255
                    msk[lri:lro,lci:lco,0]=pat_msk[pri:pro,pci:pco,0]
                    msks=msk if msks is None else np.concatenate((msks,msk),axis=-1)
                    inserted[li-1]+=1
                    if inserted[li-1]>max_instance: break;
                # else: cv2.imwrite("spot_rejected.jpg",img[lri:lro,lci:lco])
            if verbose>1:
                print(" inserted %s for %s"%(inserted,view.file_name),end='')
            elif verbose>0:
                print("+",end='')
            total_inserted=sum(inserted)
            if total_inserted>0:
                # cv2.imwrite("multi_"+view.file_name,img,[int(cv2.IMWRITE_JPEG_QUALITY),100])
                # for i in range(0,6,3):
                #     cv2.imwrite("multi_%s_mask%d_%s.jpg"%(view.file_name,i,clss[i:i+3]),msks[...,i:i+3],[int(cv2.IMWRITE_JPEG_QUALITY),100])
                return img,np.array(clss,dtype=np.uint8),msks


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
