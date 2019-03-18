# LungNeuralNet

The convolutional neural network architecture was based on U-Net, Convolutional Networks for Biomedical Image Segmentation.
http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

After training on 14 image pairs, the neural network is able to reach 80%~90% accuracy (dice coefficient) in identifying lung parenchymal region and severe inflammation in the lung in the validation set.
The prediction performed on a new image was shown as an example below.

Data credits: Jeanine D'Armiento, Monica Goldklang, Kyle Stearns; Columbia University Medical Center

<dl>
    <dt>Original Image</dt>
    <dl></dl>
</dl>

![alt text](pred/Original/36_KO_FLU_1.jpg?raw=true "original Image")

<dl>
    <dt>Predicted to be lung parenchymal region</dt>
    <dd>sum of pixels: 813216 (56% of the entire image)</dd>
</dl>

![alt text](pred/Parenchyma+0.2_768x768_NetU_Vgg16_6F64-512P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1/36_KO_FLU_1.jpe?raw=true "lung parenchymal region")

<dl>
    <dt>Predicted to be severe inflammation in the lung</dt>
    <dd>sum of pixels: 279179 (19% of the entire image)</dd>
</dl>

![alt text](pred/SevereInflammation+0.2_768x768_NetU_Vgg16_6F64-512P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1/36_KO_FLU_1.jpe?raw=true "severe inflammation in the lung")

<dl>
    <dt>Multi-label overlay</dd>
</dl>

![alt text](pred/Parenchyma,SevereInflamma+0.2_768x768_NetU_Vgg16_6F64-512P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1/36_KO_FLU_1.jpe?raw=true "severe inflammation in the lung")


<dl>
    <dt>Exported results</dt>    
</dl>

|   | Parenchyma  |  SevereInflammation |
|---|---|---|
| 36_KO_FLU_1.jpg | 813216 | 279179 |

# Training
![alt text](log/dice.png?raw=true "dice coefficient during training")
![alt text](log/jac.png?raw=true "jaccard coefficient during training")
```

Using TensorFlow backend.
Network specifications: Unet_8F64-256P2-2_Ca3Ca3SDmpSCa3_SSUuCCa33S_EluSigmoidBcedice1
Found [14] file from [.\LungNeuralNet\2x_field_lung_flu\train\Original]
Found [84] file from [.\LungNeuralNet\2x_field_lung_flu\train\Original_1.0_512x512]
...
From 84 split into train: 66 views 11 images; validation 18 views 3 images
Training Images:
{'7_WT_FLU_2.jpg', '36_KO_FLU_2.jpg', '6_WT_FLU_3.jpg', '40_WT_FLU_1.jpg', '18_KO_FLU_1.jpg', '40_WT_FLU_2.jpg', '12_KO_FLU_1.jpg', '6_WT_FLU_2.jpg', '39_WT_FLU_1.jpg', '18_KO_FLU_2.jpg', '12_KO_FLU_2.jpg'}
Validation Images:
{'6_WT_FLU_1.jpg', '39_WT_FLU_2.jpg', '7_WT_FLU_1.jpg'}
Fitting neural net...

Training 1/5 for Parenchyma_1.0_512x512_Unet_8F64-256P2-2_Ca3Ca3SDmpSCa3_SSUuCCa33S_EluSigmoidBcedice1
Epoch 1/12
2018-09-05 18:33:26.302356: I T:\src\github\tensorflow\tensorflow\core\platform\cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2018-09-05 18:33:26.476315: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 1070 major: 6 minor: 1 memoryClockRate(GHz): 1.7715
pciBusID: 0000:01:00.0
totalMemory: 8.00GiB freeMemory: 6.63GiB
2018-09-05 18:33:26.476607: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1484] Adding visible gpu devices: 0
2018-09-05 18:33:27.145461: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-09-05 18:33:27.145637: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:971]      0 
2018-09-05 18:33:27.145747: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:984] 0:   N 
2018-09-05 18:33:27.145968: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6398 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1070, pci bus id: 0000:01:00.0, compute capability: 6.1)

 1/66 [..............................] - ETA: 6:51 - loss: 1.8724 - jac: 0.1122 - dice: 0.2018 - dice67: 0.2780 - dice33: 3.9622e-05
 2/66 [..............................] - ETA: 3:27 - loss: 1.2738 - jac: 0.2575 - dice: 0.3880 - dice67: 0.4322 - dice33: 0.3255    
 3/66 [>.............................] - ETA: 2:19 - loss: 1.0166 - jac: 0.3665 - dice: 0.5046 - dice67: 0.5481 - dice33: 0.4753
 4/66 [>.............................] - ETA: 1:45 - loss: 0.9187 - jac: 0.3623 - dice: 0.5081 - dice67: 0.5259 - dice33: 0.5178
 5/66 [=>............................] - ETA: 1:24 - loss: 0.8577 - jac: 0.3556 - dice: 0.5054 - dice67: 0.5349 - dice33: 0.5542
 6/66 [=>............................] - ETA: 1:10 - loss: 0.8033 - jac: 0.3780 - dice: 0.5307 - dice67: 0.5617 - dice33: 0.6056
 7/66 [==>...........................] - ETA: 1:01 - loss: 0.7521 - jac: 0.4211 - dice: 0.5706 - dice67: 0.6035 - dice33: 0.6545
 8/66 [==>...........................] - ETA: 53s - loss: 0.7249 - jac: 0.4357 - dice: 0.5867 - dice67: 0.6347 - dice33: 0.6976 
 9/66 [===>..........................] - ETA: 47s - loss: 0.6823 - jac: 0.4733 - dice: 0.6184 - dice67: 0.6652 - dice33: 0.7296
...
60/66 [==========================>...] - ETA: 0s - loss: 0.1491 - jac: 0.8636 - dice: 0.9214 - dice67: 0.9199 - dice33: 0.9225
61/66 [==========================>...] - ETA: 0s - loss: 0.1497 - jac: 0.8635 - dice: 0.9215 - dice67: 0.9193 - dice33: 0.9216
62/66 [===========================>..] - ETA: 0s - loss: 0.1500 - jac: 0.8639 - dice: 0.9218 - dice67: 0.9195 - dice33: 0.9224
63/66 [===========================>..] - ETA: 0s - loss: 0.1490 - jac: 0.8647 - dice: 0.9223 - dice67: 0.9200 - dice33: 0.9223
64/66 [============================>.] - ETA: 0s - loss: 0.1493 - jac: 0.8646 - dice: 0.9223 - dice67: 0.9200 - dice33: 0.9229
65/66 [============================>.] - ETA: 0s - loss: 0.1484 - jac: 0.8655 - dice: 0.9229 - dice67: 0.9207 - dice33: 0.9236
66/66 [==============================] - 11s 168ms/step - loss: 0.1524 - jac: 0.8600 - dice: 0.9191 - dice67: 0.9144 - dice33: 0.9096 - val_loss: 0.1784 - val_jac: 0.8517 - val_dice: 0.9183 - val_dice67: 0.9273 - val_dice33: 0.9216
Epoch 00007: early stopping

Training 2/5 for Parenchyma_1.0_512x512_Unet_8F64-256P2-2_Ca3Ca3SDmpSCa3_SSUuCCa33S_EluSigmoidBcedice1
...
```

# Prediction
```
Found [1] file from [.\LungNeuralNet\2x_field_lung_flu\pred\Original]
Found [12] file from [.\2x_field_lung_flu\pred\Original_1.0_512x512]
36_KO_FLU_1_#1040#1392#0#512#0#512#.jpg
36_KO_FLU_1_#1040#1392#0#512#293#805#.jpg
36_KO_FLU_1_#1040#1392#0#512#587#1099#.jpg
36_KO_FLU_1_#1040#1392#0#512#880#1392#.jpg
36_KO_FLU_1_#1040#1392#264#776#0#512#.jpg
36_KO_FLU_1_#1040#1392#264#776#293#805#.jpg
36_KO_FLU_1_#1040#1392#264#776#587#1099#.jpg
36_KO_FLU_1_#1040#1392#264#776#880#1392#.jpg
36_KO_FLU_1_#1040#1392#528#1040#0#512#.jpg
36_KO_FLU_1_#1040#1392#528#1040#293#805#.jpg
36_KO_FLU_1_#1040#1392#528#1040#587#1099#.jpg
36_KO_FLU_1_#1040#1392#528#1040#880#1392#.jpg
Load model and predict to [Parenchyma]...
Parenchyma_1.0_512x512_Unet_8F64-256P2-2_Ca3Ca3SDmpSCa3_SSUuCCa3Ca3_EluSigmoidBcedice1.h5

 1/12 [=>............................] - ETA: 17s
 3/12 [======>.......................] - ETA: 5s 
 5/12 [===========>..................] - ETA: 2s
 7/12 [================>.............] - ETA: 1s
 9/12 [=====================>........] - ETA: 0s
11/12 [==========================>...] - ETA: 0s
12/12 [==============================] - 2s 174ms/step

Saving predicted results [36_KO_FLU_1.jpg] to folder [Parenchyma_1.0_512x512_Unet_8F64-256P2-2_Ca3Ca3SDmpSCa3_SSUuCCa3Ca3_EluSigmoidBcedice1]...
36_KO_FLU_1_#1040#1392#0#512#0#512#.jpg
[  0: Parenchyma] 103938 / 262144  39.65%
36_KO_FLU_1_#1040#1392#0#512#293#805#.jpg
[  0: Parenchyma] 150757 / 262144  57.51%
36_KO_FLU_1_#1040#1392#0#512#587#1099#.jpg
[  0: Parenchyma] 188507 / 262144  71.91%
36_KO_FLU_1_#1040#1392#0#512#880#1392#.jpg
[  0: Parenchyma] 183025 / 262144  69.82%
36_KO_FLU_1_#1040#1392#264#776#0#512#.jpg
[  0: Parenchyma] 215833 / 262144  82.33%
36_KO_FLU_1_#1040#1392#264#776#293#805#.jpg
[  0: Parenchyma] 214551 / 262144  81.84%
36_KO_FLU_1_#1040#1392#264#776#587#1099#.jpg
[  0: Parenchyma] 177790 / 262144  67.82%
36_KO_FLU_1_#1040#1392#264#776#880#1392#.jpg
[  0: Parenchyma] 174051 / 262144  66.40%
36_KO_FLU_1_#1040#1392#528#1040#0#512#.jpg
[  0: Parenchyma] 124851 / 262144  47.63%
36_KO_FLU_1_#1040#1392#528#1040#293#805#.jpg
[  0: Parenchyma] 161558 / 262144  61.63%
36_KO_FLU_1_#1040#1392#528#1040#587#1099#.jpg
[  0: Parenchyma] 177674 / 262144  67.78%
36_KO_FLU_1_#1040#1392#528#1040#880#1392#.jpg
[  0: Parenchyma] 148933 / 262144  56.81%
36_KO_FLU_1.jpg
[  0: Parenchyma] 813216 / 1447680  56.17%
...
```

# Neural Network Summary
```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 512, 512, 3)  0                                            
__________________________________________________________________________________________________
pre0 (Conv2D)                   (None, 512, 512, 64) 1792        input_1[0][0]                    
__________________________________________________________________________________________________
dconv0 (Conv2D)                 (None, 512, 512, 64) 36928       pre0[0][0]                       
__________________________________________________________________________________________________
dsamp1 (MaxPooling2D)           (None, 256, 256, 64) 0           dconv0[0][0]                     
__________________________________________________________________________________________________
dproc1 (Conv2D)                 (None, 256, 256, 96) 55392       dsamp1[0][0]                     
__________________________________________________________________________________________________
dconv1 (Conv2D)                 (None, 256, 256, 96) 83040       dproc1[0][0]                     
__________________________________________________________________________________________________
dsamp2 (MaxPooling2D)           (None, 128, 128, 96) 0           dconv1[0][0]                     
__________________________________________________________________________________________________
dproc2 (Conv2D)                 (None, 128, 128, 128 110720      dsamp2[0][0]                     
__________________________________________________________________________________________________
dconv2 (Conv2D)                 (None, 128, 128, 128 147584      dproc2[0][0]                     
__________________________________________________________________________________________________
dsamp3 (MaxPooling2D)           (None, 64, 64, 128)  0           dconv2[0][0]                     
__________________________________________________________________________________________________
dproc3 (Conv2D)                 (None, 64, 64, 192)  221376      dsamp3[0][0]                     
__________________________________________________________________________________________________
dconv3 (Conv2D)                 (None, 64, 64, 192)  331968      dproc3[0][0]                     
__________________________________________________________________________________________________
dsamp4 (MaxPooling2D)           (None, 32, 32, 192)  0           dconv3[0][0]                     
__________________________________________________________________________________________________
dproc4 (Conv2D)                 (None, 32, 32, 256)  442624      dsamp4[0][0]                     
__________________________________________________________________________________________________
dconv4 (Conv2D)                 (None, 32, 32, 256)  590080      dproc4[0][0]                     
__________________________________________________________________________________________________
dsamp5 (MaxPooling2D)           (None, 16, 16, 256)  0           dconv4[0][0]                     
__________________________________________________________________________________________________
dproc5 (Conv2D)                 (None, 16, 16, 256)  590080      dsamp5[0][0]                     
__________________________________________________________________________________________________
dconv5 (Conv2D)                 (None, 16, 16, 256)  590080      dproc5[0][0]                     
__________________________________________________________________________________________________
dsamp6 (MaxPooling2D)           (None, 8, 8, 256)    0           dconv5[0][0]                     
__________________________________________________________________________________________________
dproc6 (Conv2D)                 (None, 8, 8, 256)    590080      dsamp6[0][0]                     
__________________________________________________________________________________________________
dconv6 (Conv2D)                 (None, 8, 8, 256)    590080      dproc6[0][0]                     
__________________________________________________________________________________________________
dsamp7 (MaxPooling2D)           (None, 4, 4, 256)    0           dconv6[0][0]                     
__________________________________________________________________________________________________
dproc7 (Conv2D)                 (None, 4, 4, 256)    590080      dsamp7[0][0]                     
__________________________________________________________________________________________________
usamp6 (UpSampling2D)           (None, 8, 8, 256)    0           dproc7[0][0]                     
__________________________________________________________________________________________________
umerge6 (Concatenate)           (None, 8, 8, 512)    0           usamp6[0][0]                     
                                                                 dconv6[0][0]                     
__________________________________________________________________________________________________
uproc6 (Conv2D)                 (None, 8, 8, 256)    1179904     umerge6[0][0]                    
__________________________________________________________________________________________________
usamp5 (UpSampling2D)           (None, 16, 16, 256)  0           uproc6[0][0]                     
__________________________________________________________________________________________________
umerge5 (Concatenate)           (None, 16, 16, 512)  0           usamp5[0][0]                     
                                                                 dconv5[0][0]                     
__________________________________________________________________________________________________
uproc5 (Conv2D)                 (None, 16, 16, 256)  1179904     umerge5[0][0]                    
__________________________________________________________________________________________________
usamp4 (UpSampling2D)           (None, 32, 32, 256)  0           uproc5[0][0]                     
__________________________________________________________________________________________________
umerge4 (Concatenate)           (None, 32, 32, 512)  0           usamp4[0][0]                     
                                                                 dconv4[0][0]                     
__________________________________________________________________________________________________
uproc4 (Conv2D)                 (None, 32, 32, 256)  1179904     umerge4[0][0]                    
__________________________________________________________________________________________________
usamp3 (UpSampling2D)           (None, 64, 64, 256)  0           uproc4[0][0]                     
__________________________________________________________________________________________________
umerge3 (Concatenate)           (None, 64, 64, 448)  0           usamp3[0][0]                     
                                                                 dconv3[0][0]                     
__________________________________________________________________________________________________
uproc3 (Conv2D)                 (None, 64, 64, 192)  774336      umerge3[0][0]                    
__________________________________________________________________________________________________
usamp2 (UpSampling2D)           (None, 128, 128, 192 0           uproc3[0][0]                     
__________________________________________________________________________________________________
umerge2 (Concatenate)           (None, 128, 128, 320 0           usamp2[0][0]                     
                                                                 dconv2[0][0]                     
__________________________________________________________________________________________________
uproc2 (Conv2D)                 (None, 128, 128, 128 368768      umerge2[0][0]                    
__________________________________________________________________________________________________
usamp1 (UpSampling2D)           (None, 256, 256, 128 0           uproc2[0][0]                     
__________________________________________________________________________________________________
umerge1 (Concatenate)           (None, 256, 256, 224 0           usamp1[0][0]                     
                                                                 dconv1[0][0]                     
__________________________________________________________________________________________________
uproc1 (Conv2D)                 (None, 256, 256, 96) 193632      umerge1[0][0]                    
__________________________________________________________________________________________________
usamp0 (UpSampling2D)           (None, 512, 512, 96) 0           uproc1[0][0]                     
__________________________________________________________________________________________________
umerge0 (Concatenate)           (None, 512, 512, 160 0           usamp0[0][0]                     
                                                                 dconv0[0][0]                     
__________________________________________________________________________________________________
uproc0 (Conv2D)                 (None, 512, 512, 64) 92224       umerge0[0][0]                    
__________________________________________________________________________________________________
post0 (Conv2D)                  (None, 512, 512, 64) 36928       uproc0[0][0]                     
__________________________________________________________________________________________________
out0 (Conv2D)                   (None, 512, 512, 1)  65          post0[0][0]                      
==================================================================================================
Total params: 9,977,569
Trainable params: 9,977,569
Non-trainable params: 0
__________________________________________________________________________________________________

```