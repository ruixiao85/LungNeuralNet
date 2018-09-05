# LungNeuralNet

The convolutional neural network architecture was based on U-Net, Convolutional Networks for Biomedical Image Segmentation.
http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

After training on 14 image pairs, the neural network is able to reach 80%~90% accuracy (dice coefficient) in identifying lung parenchymal region and severe inflammation in the lung in the validation set.
The prediction performed on a new image was shown as an example below.

Image credits: Jeanine D'Armiento, Monica Goldklang, Kyle Stearns; Columbia University Medical Center

<dl>
    <dt>Original Image</dt>
    <dl></dl>
</dl>

![alt text](pred/Original/36_KO_FLU_1.jpg?raw=true "original Image")

<dl>
    <dt>Predicted to be lung parenchymal region</dt>
    <dd>sum of pixels: 813216 (56% of the entire image)</dd>
</dl>

![alt text](pred/Parenchyma1.0_512x512_Unet_8F64-256P2-2_Ca3Ca3SDmpSCa3_SSUuCCa3Ca3_EluSigmoidBcedice1/36_KO_FLU_1.jpe?raw=true "lung parenchymal region")

<dl>
    <dt>Predicted to be severe inflammation in the lung</dt>
    <dd>sum of pixels: 279179 (19% of the entire image)</dd>
</dl>

![alt text](pred/SevereInflammation1.0_512x512_Unet_8F64-256P2-2_Ca3Ca3SDmpSCa3_SSUuCCa3Ca3_EluSigmoidBcedice1/36_KO_FLU_1.jpe?raw=true "severe inflammation in the lung")

<dl>
    <dt>Multi-label overlay</dd>
</dl>

![alt text](pred/Parenchyma,SevereInflam1.0_512x512_Unet_8F64-256P2-2_Ca3Ca3SDmpSCa3_SSUuCCa3Ca3_EluSigmoidBcedice1/36_KO_FLU_1.jpe?raw=true "severe inflammation in the lung")


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
Network specifications: unet pool up 2f1 5f32-512 2k33 elu sigmoid o1 loss bce dice
Found [14] file from [.\LungNeuralNet\2x_field_lung_flu\train\Original]
Spliting into training-sized images (512x512)
Found [84] file from [.\LungNeuralNet\2x_field_lung_flu\train\Original_1.0_512x512]
12_KO_FLU_1_#1040#1392#0#512#0#512#.jpg
12_KO_FLU_1_#1040#1392#0#512#440#952#.jpg
...
Found [14] file from [.\LungNeuralNet\2x_field_lung_flu\train\Parenchyma]
Spliting into training-sized images (512x512)
Found [84] file from [.\LungNeuralNet\2x_field_lung_flu\train\Parenchyma_1.0_512x512]
12_KO_FLU_1_#1040#1392#0#512#0#512#.jpg
12_KO_FLU_1_#1040#1392#0#512#440#952#.jpg
...
From 84 split into train: 54 views 9 images; validation 30 views 5 images
Training Images:
{'36_KO_FLU_2.jpg', '40_WT_FLU_2.jpg', '18_KO_FLU_1.jpg', '12_KO_FLU_2.jpg', '7_WT_FLU_2.jpg', '39_WT_FLU_2.jpg', '6_WT_FLU_1.jpg', '7_WT_FLU_1.jpg', '18_KO_FLU_2.jpg'}
Validation Images:
{'12_KO_FLU_1.jpg', '6_WT_FLU_3.jpg', '40_WT_FLU_1.jpg', '6_WT_FLU_2.jpg', '39_WT_FLU_1.jpg'}
Fitting neural net...
Training 1/3 for Parenchyma_1.0_512x512_unet_pool_up_2f1_5f32-512_2k33_elu_sigmoid_o1_loss_bce_dice
2018-08-24 00:57:53.707125: I T:\src\github\tensorflow\tensorflow\core\platform\cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2018-08-24 00:57:53.897094: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1392] Found device 0 with properties: 
name: GeForce GTX 1070 major: 6 minor: 1 memoryClockRate(GHz): 1.7715
pciBusID: 0000:01:00.0
totalMemory: 8.00GiB freeMemory: 6.63GiB
2018-08-24 00:57:53.897398: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1471] Adding visible gpu devices: 0
2018-08-24 00:57:54.518176: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:952] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-24 00:57:54.518355: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:958]      0 
2018-08-24 00:57:54.518469: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:971] 0:   N 
2018-08-24 00:57:54.518671: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1084] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6404 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1070, pci bus id: 0000:01:00.0, compute capability: 6.1)
Epoch 1/12

 1/54 [..............................] - ETA: 2:06 - loss: 0.5322 - jac: 0.5228 - dice: 0.6866 - dice_80: 0.7024 - dice_60: 0.7205 - dice_40: 0.7924 - dice_20: 0.8586
 2/54 [>.............................] - ETA: 1:04 - loss: 0.6190 - jac: 0.3970 - dice: 0.5566 - dice_80: 0.5407 - dice_60: 0.5230 - dice_40: 0.4525 - dice_20: 0.4293
 3/54 [>.............................] - ETA: 43s - loss: 1.3340 - jac: 0.2647 - dice: 0.3711 - dice_80: 0.3605 - dice_60: 0.3487 - dice_40: 0.3017 - dice_20: 0.2862 
 4/54 [=>............................] - ETA: 33s - loss: 1.2216 - jac: 0.1985 - dice: 0.2783 - dice_80: 0.2704 - dice_60: 0.2615 - dice_40: 0.2263 - dice_20: 0.2147
 5/54 [=>............................] - ETA: 27s - loss: 1.0679 - jac: 0.2987 - dice: 0.3873 - dice_80: 0.4017 - dice_60: 0.4043 - dice_40: 0.3727 - dice_20: 0.3488
 6/54 [==>...........................] - ETA: 22s - loss: 0.9768 - jac: 0.3711 - dice: 0.4638 - dice_80: 0.4823 - dice_60: 0.4838 - dice_40: 0.4537 - dice_20: 0.4183
 7/54 [==>...........................] - ETA: 19s - loss: 0.8987 - jac: 0.4344 - dice: 0.5257 - dice_80: 0.5471 - dice_60: 0.5525 - dice_40: 0.5290 - dice_20: 0.4993
 8/54 [===>..........................] - ETA: 17s - loss: 0.8991 - jac: 0.4530 - dice: 0.5521 - dice_80: 0.5746 - dice_60: 0.5835 - dice_40: 0.5600 - dice_20: 0.5413
 9/54 [====>.........................] - ETA: 15s - loss: 0.8742 - jac: 0.4442 - dice: 0.5512 - dice_80: 0.5687 - dice_60: 0.5718 - dice_40: 0.5340 - dice_20: 0.4840
...
50/54 [==========================>...] - ETA: 0s - loss: 0.4709 - jac: 0.6339 - dice: 0.7390 - dice_80: 0.7444 - dice_60: 0.7447 - dice_40: 0.7374 - dice_20: 0.7501
51/54 [===========================>..] - ETA: 0s - loss: 0.4692 - jac: 0.6352 - dice: 0.7407 - dice_80: 0.7459 - dice_60: 0.7454 - dice_40: 0.7382 - dice_20: 0.7522
52/54 [===========================>..] - ETA: 0s - loss: 0.4654 - jac: 0.6391 - dice: 0.7440 - dice_80: 0.7490 - dice_60: 0.7483 - dice_40: 0.7403 - dice_20: 0.7510
53/54 [============================>.] - ETA: 0s - loss: 0.4592 - jac: 0.6452 - dice: 0.7485 - dice_80: 0.7536 - dice_60: 0.7531 - dice_40: 0.7452 - dice_20: 0.7557
54/54 [==============================] - 9s 159ms/step - loss: 0.4556 - jac: 0.6492 - dice: 0.7517 - dice_80: 0.7568 - dice_60: 0.7563 - dice_40: 0.7489 - dice_20: 0.7586 - val_loss: 0.2591 - val_jac: 0.7852 - val_dice: 0.8655 - val_dice_80: 0.8653 - val_dice_60: 0.8438 - val_dice_40: 0.8947 - val_dice_20: 0.9221
Epoch 2/12
...
Epoch 00002: early stopping
...
From 61 split into train: 38 views 8 images; validation 23 views 5 images
Training Images:
{'36_KO_FLU_2.jpg', '12_KO_FLU_1.jpg', '18_KO_FLU_1.jpg', '7_WT_FLU_2.jpg', '40_WT_FLU_1.jpg', '39_WT_FLU_2.jpg', '6_WT_FLU_2.jpg', '18_KO_FLU_2.jpg'}
Validation Images:
{'40_WT_FLU_2.jpg', '12_KO_FLU_2.jpg', '6_WT_FLU_3.jpg', '39_WT_FLU_1.jpg', '7_WT_FLU_1.jpg'}
Fitting neural net...
Training 1/3 for SevereInflammation_1.0_512x512_unet_pool_up_2f1_5f32-512_2k33_elu_sigmoid_o1_loss_bce_dice
Epoch 1/12

 1/38 [..............................] - ETA: 3s - loss: 0.8592 - jac: 0.0356 - dice: 0.0688 - dice_80: 2.8595e-10 - dice_60: 7.4333e-10 - dice_40: 5.3476e-09 - dice_20: 1.0000
 2/38 [>.............................] - ETA: 3s - loss: 1.0868 - jac: 0.0656 - dice: 0.1217 - dice_80: 0.0436 - dice_60: 0.0196 - dice_40: 0.0026 - dice_20: 0.5000            
 3/38 [=>............................] - ETA: 3s - loss: 1.0560 - jac: 0.0934 - dice: 0.1676 - dice_80: 0.1455 - dice_60: 0.1549 - dice_40: 0.1210 - dice_20: 0.3504
...
36/38 [===========================>..] - ETA: 0s - loss: 0.3650 - jac: 0.4094 - dice: 0.5171 - dice_80: 0.5617 - dice_60: 0.5456 - dice_40: 0.5501 - dice_20: 0.6762
37/38 [============================>.] - ETA: 0s - loss: 0.3714 - jac: 0.4009 - dice: 0.5078 - dice_80: 0.5465 - dice_60: 0.5309 - dice_40: 0.5352 - dice_20: 0.6579
38/38 [==============================] - 4s 116ms/step - loss: 0.3738 - jac: 0.3912 - dice: 0.4963 - dice_80: 0.5343 - dice_60: 0.5198 - dice_40: 0.5475 - dice_20: 0.6669 - val_loss: 0.3450 - val_jac: 0.4207 - val_dice: 0.5381 - val_dice_80: 0.4973 - val_dice_60: 0.4244 - val_dice_40: 0.5337 - val_dice_20: 0.7958
Epoch 00005: early stopping

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