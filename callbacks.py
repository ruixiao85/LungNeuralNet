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
            self.best=best+0.005 # last best value stored in 0.00 format
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
                            print('\nEpoch %03d: %s improved %0.5f -> %0.5f, saving to [%s]'%(epoch+1,self.monitor,self.best,current,filepath))
                        self.best=current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose>0:
                            print('\nEpoch %03d: %s did not improve from %0.5f'%(epoch+1, self.monitor, self.best))
            else:
                if self.verbose>0:
                    print('\nEpoch %03d: saving model to [%s]'%(epoch+1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
