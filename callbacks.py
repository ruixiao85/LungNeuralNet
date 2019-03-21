'''
https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure?rq=1
Yu Yang
'''
import os
import warnings
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard, Callback
from keras import backend as K

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
    def __init__(self, filepath, monitor='val_loss', verbose=0, save_mode='historical', save_weights_only=False, mode='auto', period=1, lr_decay=0.5, sig_digits=3, hist_best=None):
        super(ModelCheckpointCustom, self).__init__()
        self.filepath=filepath+"^{epoch:02d}^{%s:.%df}^.h5"%(monitor, sig_digits)
        self.monitor=monitor
        self.verbose=verbose
        self.save_mode=save_mode.lower()[0] # when achieve All, Current best, Historical best, None
        assert(self.save_mode in ['a','c','h','n'])
        self.save_weights_only=save_weights_only
        self.period=period
        self.lr_decay=lr_decay
        self.sig_digits=sig_digits
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
        if hist_best is not None:
            self.historical_best=hist_best
            print("Aiming to surpass the historical best value of %s=%f"%(self.monitor,self.historical_best))
        else:
            self.historical_best=self.best

    def on_epoch_end(self, epoch, logs=None):
        logs=logs or {}
        logs['lr']=cur_lr=K.get_value(self.model.optimizer.lr)
        self.epochs_since_last_save+=1
        if self.epochs_since_last_save>=self.period:
            self.epochs_since_last_save=0
            filepath=self.filepath.format(epoch=epoch+1, **logs)
            current=round(logs.get(self.monitor),self.sig_digits) # round numbers to disregard minor improvements
            if current is None:
                warnings.warn('Can save best model only with %s available, skipping.'%self.monitor, RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.monitor_op(current, self.historical_best): # better than last epoch and the historical best
                        if self.save_mode=='n': # Not saving
                            if self.verbose>0:
                                print('\nEpoch %05d: %s %0.{0}f->%0.{0}f->%0.{0}f historical best, lr=%.1e, not saving to [%s]'.format(self.sig_digits)%
                                      (epoch+1,self.monitor,current,self.best,self.historical_best,cur_lr,filepath))
                        else: # in ['h','c','a'] 'historical' 'current' 'all'  Save
                            if self.verbose>0:
                                print('\nEpoch %05d: %s %0.{0}f->%0.{0}f->%0.{0}f historical best, lr=%.1e, saving to [%s]'.format(self.sig_digits)%
                                      (epoch+1,self.monitor,current,self.best,self.historical_best,cur_lr,filepath))
                            self.save_network(filepath)
                        self.best=self.historical_best=current
                    else: # better than last epoch, no better than the historical best
                        if self.save_mode in ['n','h']: # 'none' 'historical' Not saving
                            if self.verbose>0:
                                print('\nEpoch %05d: %s %0.{0}f->%0.{0}f->%0.{0}f current best, lr=%.1e, not saving to [%s]'.format(self.sig_digits)%(
                                epoch+1,self.monitor,current,self.best,self.historical_best,cur_lr,filepath))
                        else: #  in ['c', 'a']: # 'current' 'all' Save
                            if self.verbose>0:
                                print('\nEpoch %05d: %s %0.{0}f->%0.{0}f->%0.{0}f current best, lr=%.1e, saving to [%s]'.format(self.sig_digits)%(
                                epoch+1,self.monitor,current,self.best,self.historical_best,cur_lr,filepath))
                            self.save_network(filepath)
                        self.best=current
                else:
                    new_lr=self.lr_decay*cur_lr
                    K.set_value(self.model.optimizer.lr,new_lr)
                    if self.save_mode=='a':
                        if self.verbose>0:
                            print('\nEpoch %05d: %s %0.{0}f->%0.{0}f->%0.{0}f less than ideal, lr*%.2f=%.1e, saving to [%s]'.format(self.sig_digits)%(
                            epoch+1,self.monitor,current,self.best,self.historical_best,self.lr_decay,new_lr,filepath))
                        self.save_network(filepath)
                    else:
                        if self.verbose>0:
                            print('\nEpoch %05d: %s %0.{0}f->%0.{0}f->%0.{0}f less than ideal, lr*%.2f=%.1e, not saving to [%s]'.format(self.sig_digits)%
                                  (epoch+1, self.monitor,current,self.best,self.historical_best,self.lr_decay,new_lr,filepath))

    def save_network(self,filepath):
        if self.save_weights_only:
            self.model.save_weights(filepath,overwrite=True)
        else:
            self.model.save(filepath,overwrite=True)
