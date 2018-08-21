import traceback

from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D, Conv2DTranspose
from keras.layers import UpSampling2D, Dropout, BatchNormalization

'''
U-Net: Convolutional Networks for Biomedical Image Segmentation
(https://arxiv.org/abs/1505.04597)
github: pietz
---
img_shape: (height, width, channels)
out_ch: number of output channels
start_ch: number of channels of the first conv
depth: zero indexed depth of the U-structure
inc_rate: rate at which the conv channels will increase
activation: activation function after convolutions
dropout: amount of dropout in the contracting part
batchnorm: adds Batch Normalization if true
maxpool: use strided conv instead of maxpooling if false
upconv: use transposed conv instead of upsamping + conv if false
residual: add residual connections around each conv block if true
'''

def conv_block(m, dim, actfun, batchnorm, dropout, residual):
	n = Conv2D(dim, 3, activation=actfun, padding='same')(m)
	n = BatchNormalization()(n) if batchnorm else n
	n = Dropout(dropout)(n) if dropout else n
	n = Conv2D(dim, 3, activation=actfun, padding='same')(n)
	n = BatchNormalization()(n) if batchnorm else n
	return Concatenate()([m, n]) if residual else n

def level_block(m, filters, index, actfun, maxpool, upconv, batchnorm, dropout, residual):
	if index >= len(filters)-1:
		n = conv_block(m, filters[index], actfun, batchnorm, 0., residual)
		m = MaxPooling2D()(n) if maxpool else Conv2D(filters[index], 3, strides=2, padding='same')(n)
		m = level_block(m, filters, index + 1, actfun, maxpool, upconv, batchnorm, dropout, residual)
		if upconv:
			m = UpSampling2D()(m)
			m = Conv2D(filters[index], 2, activation=actfun, padding='same')(m)
		else:
			m = Conv2DTranspose(filters[index], 3, strides=2, activation=actfun, padding='same')(m)
		n = Concatenate()([n, m])
		m = conv_block(n, filters[index], actfun, batchnorm, 0.,residual)
	else:
		m = conv_block(m, filters[index], actfun, batchnorm, dropout, residual)
	return m

def unet_recursive(cfg):
	if cfg.model_filter is None:
		# cfg.model_filter = [64, 96, 128, 192]
		cfg.model_filter = [64, 96, 128, 192, 256]
		# cfg.model_filter = [96, 128, 192, 256, 384]
		# cfg.model_filter = [64, 96, 128, 192, 256, 384]
		# cfg.model_filter = [64, 96, 128, 192, 256, 384, 512]
	if cfg.model_kernel is None or len(cfg.model_kernel) != 2:
		cfg.model_kernel = [3, 3]

	i = Input(shape=(cfg.row_in, cfg.col_in, cfg.dep_in))
	o = level_block(i, cfg.model_filter, index=0, actfun=cfg.model_act,
					maxpool=True, upconv=True, batchnorm=True, dropout=0.5, residual=False)
	o = Conv2D(cfg.dep_out, (1, 1), activation='sigmoid')(o)
	return Model(inputs=i, outputs=o), traceback.extract_stack(None, 2)[1].name + "_" + str(cfg)
