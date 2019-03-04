'''
https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure?rq=1
Yu Yang
'''
import os
import warnings
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard, Callback


class TensorBoardTrainVal(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TensorBoardTrainVal, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')
        # if not os.path.exists(self.val_log_dir):
        #     os.mkdir(self.val_log_dir)

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TensorBoardTrainVal, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TensorBoardTrainVal, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TensorBoardTrainVal, self).on_train_end(logs)
        self.val_writer.close()

# adapted from keras callbacks #
class ModelCheckpointCustom(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1,
                 best=None):
        super(ModelCheckpointCustom, self).__init__()
        self.monitor=monitor
        self.verbose=verbose
        self.filepath=filepath
        self.save_best_only=save_best_only
        self.save_weights_only=save_weights_only
        self.period=period
        self.epochs_since_last_save=0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, fallback to auto mode.'%(mode), RuntimeWarning)
            mode='auto'

        if mode=='min':
            self.monitor_op=np.less
            self.best=np.Inf
        elif mode=='max':
            self.monitor_op=np.greater
            self.best=-np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op=np.greater
                self.best=-np.Inf
            else:
                self.monitor_op=np.less
                self.best=np.Inf
        if best is not None:
            self.best=best
            print("Starting from previous best value of %s %f"%(self.monitor,best))

    def on_epoch_end(self, epoch, logs=None):
        logs=logs or {}
        self.epochs_since_last_save+=1
        if self.epochs_since_last_save>=self.period:
            self.epochs_since_last_save=0
            filepath=self.filepath.format(epoch=epoch+1, **logs)
            if self.save_best_only:
                current=logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, skipping.'%self.monitor, RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose>0:
                            print('\nEpoch %05d: %s improved %0.5f -> %0.5f, saving to [%s]'%(epoch+1,self.monitor,self.best,current,filepath))
                        self.best=current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose>0:
                            print('\nEpoch %05d: %s %0.5f is no better than %0.5f'%(epoch+1, self.monitor, current, self.best))
            else:
                if self.verbose>0:
                    print('\nEpoch %05d: saving model to [%s]'%(epoch+1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

class ModelEvalmAP(Callback):
    def __init__(self,filepath,generator,max_step,iou_threshold=0.5,score_threshold=0.05,max_detections=100,weighted_average=False):
        super(ModelEvalmAP, self).__init__()
        self.filepath=filepath
        self.generator=generator
        self.max_step=max_step

        self.iou_threshold=iou_threshold # overlap threshold
        self.score_threshold=score_threshold # score threshold
        self.max_detections=max_detections # max # of detections to use
        self.weighted_average=weighted_average # compute mAP using the weighted average of precisions among classes.


    def on_epoch_end(self, epoch, logs=None):
        steps_done,steps=0,self.generator.size()
        output_generator=iter(self.generator)
        while steps_done<steps:
            generator_output=next(output_generator)
            if not hasattr(generator_output,'__len__'):
                raise ValueError('Output of generator should be a tuple '
                                 '(x, y, sample_weight) '
                                 'or (x, y). Found: '+str(generator_output))
            if len(generator_output)==2:
                x,y=generator_output
                sample_weight=None
            elif len(generator_output)==3:
                x,y,sample_weight=generator_output
            else:
                raise ValueError('Output of generator should be a tuple '
                                 '(x, y, sample_weight) '
                                 'or (x, y). Found: '+str(generator_output))
            outs=self.model.test_on_batch(x,y,sample_weight=sample_weight)

            if isinstance(x,list):
                batch_size=x[0].shape[0]
            elif isinstance(x,dict):
                batch_size=list(x.values())[0].shape[0]
            else:
                batch_size=x.shape[0]
            if batch_size==0:
                raise ValueError('Received an empty batch. '
                                 'Batches should at least contain one item.')
            all_outs.append(outs)

            steps_done+=1
            batch_sizes.append(batch_size)
            if verbose==1:
                progbar.update(steps_done)



        average_precisions = evaluate(self.generator,self.model,iou_threshold=self.iou_threshold,score_threshold=self.score_threshold,
            max_detections=self.max_detections,save_path=self.save_path)

        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations) in average_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)
        if self.weighted_average:
            mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        else:
            mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)

        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = mean_ap
            summary_value.tag = "mAP"
            self.tensorboard.writer.add_summary(summary, epoch)

        if self.verbose == 1:
            print('mAP: {:.4f}'.format(mean_ap))

        logs['mAP'] = mean_ap

def _compute_ap(recall,precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec=np.concatenate(([0.],recall,[1.]))
    mpre=np.concatenate(([0.],precision,[0.]))

    # compute the precision envelope
    for i in range(mpre.size-1,0,-1):
        mpre[i-1]=np.maximum(mpre[i-1],mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i=np.where(mrec[1:]!=mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap=np.sum((mrec[i+1]-mrec[i])*mpre[i+1])
    return ap

def _get_detections(generator,model,score_threshold=0.05,max_detections=100,save_path=None):
    """ Get the detections from the model using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections=[[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_masks=[[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image=generator.load_image(i)
        image=generator.preprocess_image(raw_image.copy())
        image,scale=generator.resize_image(image)

        # run network
        outputs=model.predict_on_batch(np.expand_dims(image,axis=0))
        boxes=outputs[-4]
        scores=outputs[-3]
        labels=outputs[-2]
        masks=outputs[-1]

        # correct boxes for image scale
        boxes/=scale

        # select indices which have a score above the threshold
        indices=np.where(scores[0,:]>score_threshold)[0]

        # select those scores
        scores=scores[0][indices]

        # find the order with which to sort the scores
        scores_sort=np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes=boxes[0,indices[scores_sort],:]
        image_scores=scores[scores_sort]
        image_labels=labels[0,indices[scores_sort]]
        image_masks=masks[0,indices[scores_sort],:,:,image_labels]
        image_detections=np.concatenate([image_boxes,np.expand_dims(image_scores,axis=1),np.expand_dims(image_labels,axis=1)],axis=1)

        if save_path is not None:
            # draw_annotations(raw_image, generator.load_annotations(i)[0], label_to_name=generator.label_to_name)
            draw_detections(raw_image,image_boxes,image_scores,image_labels,score_threshold=score_threshold,label_to_name=generator.label_to_name)
            draw_masks(raw_image,image_boxes.astype(int),image_masks,labels=image_labels)

            cv2.imwrite(os.path.join(save_path,'{}.png'.format(i)),raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label]=image_detections[image_detections[:,-1]==label,:-1]
            all_masks[i][label]=image_masks[image_detections[:,-1]==label,...]

        print('{}/{}'.format(i+1,generator.size()),end='\r')

    return all_detections,all_masks

def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations=[[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_masks=[[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        # load the annotations
        annotations=generator.load_annotations(i)
        annotations['masks']=np.stack(annotations['masks'],axis=0)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label]=annotations['bboxes'][annotations['labels']==label,:].copy()
            all_masks[i][label]=annotations['masks'][annotations['labels']==label,...,0].copy()

        print('{}/{}'.format(i+1,generator.size()),end='\r')

    return all_annotations,all_masks

def evaluate(generator,model,iou_threshold=0.5,score_threshold=0.05,max_detections=100,binarize_threshold=0.5,save_path=None):
    """ Evaluate a given dataset using a given model.
    # Arguments
        generator          : The generator that represents the dataset to evaluate.
        model              : The model to evaluate.
        iou_threshold      : The threshold used to consider when a detection is positive or negative.
        score_threshold    : The score confidence threshold to use for detections.
        max_detections     : The maximum number of detections to use per image.
        binarize_threshold : Threshold to binarize the masks with.
        save_path          : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections,all_masks=_get_detections(generator,model,score_threshold=score_threshold,max_detections=max_detections,save_path=save_path)
    all_annotations,all_gt_masks=_get_annotations(generator)
    average_precisions={}

    # import pickle
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_masks, open('all_masks.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))
    # pickle.dump(all_gt_masks, open('all_gt_masks.pkl', 'wb'))

    # process detections and annotations
    for label in range(generator.num_classes()):
        false_positives=np.zeros((0,))
        true_positives=np.zeros((0,))
        scores=np.zeros((0,))
        num_annotations=0.0

        for i in range(generator.size()):
            detections=all_detections[i][label]
            masks=all_masks[i][label]
            annotations=all_annotations[i][label]
            gt_masks=all_gt_masks[i][label]
            num_annotations+=annotations.shape[0]
            detected_annotations=[]

            for d,mask in zip(detections,masks):
                box=d[:4].astype(int)
                scores=np.append(scores,d[4])

                if annotations.shape[0]==0:
                    false_positives=np.append(false_positives,1)
                    true_positives=np.append(true_positives,0)
                    continue

                # resize to fit the box
                mask=cv2.resize(mask,(box[2]-box[0],box[3]-box[1]))

                # binarize the mask
                mask=(mask>binarize_threshold).astype(np.uint8)

                # place mask in image frame
                mask_image=np.zeros_like(gt_masks[0])
                mask_image[box[1]:box[3],box[0]:box[2]]=mask
                mask=mask_image

                overlaps=compute_overlap(np.expand_dims(mask,axis=0),gt_masks)
                assigned_annotation=np.argmax(overlaps,axis=1)
                max_overlap=overlaps[0,assigned_annotation]

                if max_overlap>=iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives=np.append(false_positives,0)
                    true_positives=np.append(true_positives,1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives=np.append(false_positives,1)
                    true_positives=np.append(true_positives,0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations==0:
            average_precisions[label]=0,0
            continue

        # sort by score
        indices=np.argsort(-scores)
        false_positives=false_positives[indices]
        true_positives=true_positives[indices]

        # compute false positives and true positives
        false_positives=np.cumsum(false_positives)
        true_positives=np.cumsum(true_positives)

        # compute recall and precision
        recall=true_positives/num_annotations
        precision=true_positives/np.maximum(true_positives+false_positives,np.finfo(np.float64).eps)

        # compute average precision
        average_precision=_compute_ap(recall,precision)
        average_precisions[label]=average_precision,num_annotations

    return average_precisions