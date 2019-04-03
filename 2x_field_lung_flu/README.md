# LungNeuralNet

The convolutional neural network architecture used in this project was inspired by [U-Net](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) and [dual frame U-Net](https://arxiv.org/abs/1708.08333) with added transfer learning from pre-trained models in keras ([keras-applications](https://keras.io/applications/)).

![alt text](../resource/train_unet.jpg?raw=true "train_unet")

After training on **14** image pairs, the neural network is able to reach **>90%** accuracy (dice coefficient) in identifying lung parenchymal region and **>60%** for severe inflammation in the lung in the validation set.
The prediction results on a separate image, including segmentation mask and area stats, was shown below.

Multi-label overlay (blue: parenchyma, red: severe inflammation)

Original Image

![alt text](pred/Original_0.2/36_KO_FLU_1.jpg?raw=true "original Image")

Predicted lung parenchymal region

![alt text](pred/36_KO_FLU_1_paren.jpg?raw=true "lung parenchymal region")

Predicted severe inflammation in the lung

![alt text](pred/36_KO_FLU_1_inflam.jpg?raw=true "severe inflammation in the lung")

Multi-label overlay (blue: parenchyma, red: severe inflammation)

![alt text](pred/36_KO_FLU_1_both.jpg?raw=true "both parenchyma and severe inflammation in the lung")


Exported results

|   | Parenchyma  |  SevereInflammation |
|---|---|---|
| 36_KO_FLU_1.jpg | 836148 | 203466 |

# Training
![alt text](2x_dice_lr.jpg?raw=true "dice coefficient and learning rate during training (left: parenchyma, right: severe inflammation")
```
Using TensorFlow backend.
Network specifications: NetU_Vgg16_6F64-1024P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1
Found [1] folders start with [Original_] from [.\LungNeuralNet\2x_field_lung_flu\train]
 Original_0.2 + input images.
Processing images from folder [Original_0.2] with resize_ratio of 1.0x ...
Found [0] files recursively matching [*.jpeg] from [.\LungNeuralNet\2x_field_lung_flu\train\Original_0.2]
No [*.jpeg] files found, splitting [*.jpg] images with [0.33] ratio.
Found [14] files recursively matching [*.jpg] from [.\LungNeuralNet\2x_field_lung_flu\train\Original_0.2]
[Original] was split into training [10] and validation [4] set.
Loading image files (train/val) to memory...
  12_KO_FLU_1.jpg  18_KO_FLU_1.jpg  18_KO_FLU_2.jpg  36_KO_FLU_2.jpg  39_WT_FLU_2.jpg  40_WT_FLU_1.jpg  40_WT_FLU_2.jpg  6_WT_FLU_2.jpg  6_WT_FLU_3.jpg  7_WT_FLU_1.jpg
  12_KO_FLU_2.jpg  39_WT_FLU_1.jpg  6_WT_FLU_1.jpg  7_WT_FLU_2.jpg
 12_KO_FLU_1.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 18_KO_FLU_1.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 18_KO_FLU_2.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 36_KO_FLU_2.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 39_WT_FLU_2.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 40_WT_FLU_1.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 40_WT_FLU_2.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 6_WT_FLU_2.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 6_WT_FLU_3.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 7_WT_FLU_1.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
Images were divided into [60] views
 12_KO_FLU_2.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 39_WT_FLU_1.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 6_WT_FLU_1.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 7_WT_FLU_2.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
Images were divided into [24] views
Found [1] folders start with [Parenchyma_] from [.\LungNeuralNet\2x_field_lung_flu\train]
 Parenchyma_0.2 + input images.
Processing images from folder [Parenchyma_0.2] with resize_ratio of 1.0x ...
Found [0] files recursively matching [*.jpeg] from [.\LungNeuralNet\2x_field_lung_flu\train\Parenchyma_0.2]
No [*.jpeg] files found, splitting [*.jpg] images with [0.33] ratio.
Found [14] files recursively matching [*.jpg] from [.\LungNeuralNet\2x_field_lung_flu\train\Parenchyma_0.2]
[Parenchyma] was split into training [10] and validation [4] set.
Loading image files (train/val) to memory...
  12_KO_FLU_1.jpg  18_KO_FLU_1.jpg  18_KO_FLU_2.jpg  36_KO_FLU_2.jpg  39_WT_FLU_2.jpg  40_WT_FLU_1.jpg  40_WT_FLU_2.jpg  6_WT_FLU_2.jpg  6_WT_FLU_3.jpg  7_WT_FLU_1.jpg
  12_KO_FLU_2.jpg  39_WT_FLU_1.jpg  6_WT_FLU_1.jpg  7_WT_FLU_2.jpg
 12_KO_FLU_1.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 18_KO_FLU_1.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 18_KO_FLU_2.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 36_KO_FLU_2.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 39_WT_FLU_2.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 40_WT_FLU_1.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 40_WT_FLU_2.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 6_WT_FLU_2.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 6_WT_FLU_3.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 7_WT_FLU_1.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
Images were divided into [60] views
 12_KO_FLU_2.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 39_WT_FLU_1.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 6_WT_FLU_1.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
 7_WT_FLU_2.jpg target 768 x 768 (coverage 3.0): original 1040 x 1392 ->  row /2 col /3
Images were divided into [24] views
After pairing intersections, train/validation views [60 : 24] -> [60 : 24]
After low contrast exclusion [0 : 0], train/validation views [60 : 24] ->  [60 : 24]
Model compiled.
Training for Parenchyma_0.2_768x768_NetU_Vgg16_6F64-1024P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1
Scanning for files matching Parenchyma_0.2_768x768_NetU_Vgg16_6F64-1024P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1^*^.h5 in .\LungNeuralNet\2x_field_lung_flu
Found [1] files matching [Parenchyma_0.2_768x768_NetU_Vgg16_6F64-1024P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1^*^.h5] from [.\LungNeuralNet\2x_field_lung_flu]
Found 1 previous models, keeping the top 1 (max):
* 1. Parenchyma_0.2_768x768_NetU_Vgg16_6F64-1024P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1^06^0.951^.h5 kept
Train with some random weights.
Aiming to surpass the historical best value of val_dice=0.951000
Epoch 1/50
 1/60 [..............................] - ETA: 9:07 - loss: 1.4055 - jac: 0.5704 - dice: 0.7264
 2/60 [>.............................] - ETA: 4:48 - loss: 0.9675 - jac: 0.6340 - dice: 0.7742
 3/60 [>.............................] - ETA: 3:22 - loss: 0.8347 - jac: 0.5722 - dice: 0.7226
 ...
Epoch 00001: val_dice 0.927->-inf->0.951 current best, lr=5.0e-05, not saving to [Parenchyma_0.2_768x768_NetU_Vgg16_6F64-1024P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1^01^0.927^.h5]
...
Epoch 00006: val_dice 0.944->0.944->0.951 less than ideal, lr*0.30=4.5e-06, not saving to [Parenchyma_0.2_768x768_NetU_Vgg16_6F64-1024P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1^06^0.944^.h5]
Epoch 00007: early stopping
```

# Prediction
```
Using TensorFlow backend.
Network specifications: NetU_Vgg16_6F64-1024P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1
Found [2] folders start with [Original_] from [.\LungNeuralNet\2x_field_lung_flu\pred]
 Original_0.2 + input images.
Processing images from folder [Original_0.2] with resize_ratio of 1.0x ...
Found [0] files recursively matching [*.jpeg] from [.\LungNeuralNet\2x_field_lung_flu\pred\Original_0.2]
No [*.jpeg] files found, splitting [*.jpg] images with [0.33] ratio.
Found [1] files recursively matching [*.jpg] from [.\LungNeuralNet\2x_field_lung_flu\pred\Original_0.2]
[Original] was split into training [1] and validation [0] set.
Loading image files (train/val) to memory...
  36_KO_FLU_1.jpg

 36_KO_FLU_1.jpg target 768 x 768 (coverage 2.0): original 1040 x 1392 ->  row /2 col /3
Images were divided into [6] views
Images were divided into [0] views

Load model and predict to [Parenchyma,SevereInflamma]...
Scanning for files matching Parenchyma_0.2_768x768_NetU_Vgg16_6F64-1024P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1^*^.h5 in .\LungNeuralNet\2x_field_lung_flu
Found [1] files matching [Parenchyma_0.2_768x768_NetU_Vgg16_6F64-1024P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1^*^.h5] from [.\LungNeuralNet\2x_field_lung_flu]
Found 1 previous models, keeping the top 1 (max):
* 1. Parenchyma_0.2_768x768_NetU_Vgg16_6F64-1024P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1^06^0.951^.h5 kept
Parenchyma_0.2_768x768_NetU_Vgg16_6F64-1024P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1^06^0.951^.h5

1/6 [====>.........................] - ETA: 20s
2/6 [=========>....................] - ETA: 8s 
3/6 [==============>...............] - ETA: 4s
4/6 [===================>..........] - ETA: 2s
5/6 [========================>.....] - ETA: 0s
6/6 [==============================] - 5s 870ms/step
Scanning for files matching SevereInflammation_0.2_768x768_NetU_Vgg16_6F64-1024P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1^*^.h5 in .\LungNeuralNet\2x_field_lung_flu
Found [1] files matching [SevereInflammation_0.2_768x768_NetU_Vgg16_6F64-1024P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1^*^.h5] from [.\LungNeuralNet\2x_field_lung_flu]
Found 1 previous models, keeping the top 1 (max):
* 1. SevereInflammation_0.2_768x768_NetU_Vgg16_6F64-1024P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1^05^0.923^.h5 kept
SevereInflammation_0.2_768x768_NetU_Vgg16_6F64-1024P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1^05^0.923^.h5

1/6 [====>.........................] - ETA: 1s
2/6 [=========>....................] - ETA: 0s
3/6 [==============>...............] - ETA: 0s
4/6 [===================>..........] - ETA: 0s
5/6 [========================>.....] - ETA: 0s
6/6 [==============================] - 1s 223ms/step
Saving predicted results [36_KO_FLU_1.jpg] to folder [pred\Original-Parenchyma,SevereInflamma_0.2_768x768_NetU_Vgg16_6F64-1024P2-2_Ca3CtUuCtCa3Ca3_TanhReluSigmBced1]...
36_KO_FLU_1_#1040#1392#0#768#0#768#.jpg
[  0: Parenchyma] #126 $350404 / $589824  59.41%
[  1: SevereInflammation] #66 $32634 / $589824  5.53%
36_KO_FLU_1_#1040#1392#0#768#312#1080#.jpg
[  0: Parenchyma] #221 $397610 / $589824  67.41%
[  1: SevereInflammation] #104 $132518 / $589824  22.47%
36_KO_FLU_1_#1040#1392#0#768#624#1392#.jpg
[  0: Parenchyma] #220 $388495 / $589824  65.87%
[  1: SevereInflammation] #100 $121345 / $589824  20.57%
36_KO_FLU_1_#1040#1392#272#1040#0#768#.jpg
[  0: Parenchyma] #121 $377949 / $589824  64.08%
[  1: SevereInflammation] #92 $62182 / $589824  10.54%
36_KO_FLU_1_#1040#1392#272#1040#312#1080#.jpg
[  0: Parenchyma] #223 $442101 / $589824  74.95%
[  1: SevereInflammation] #172 $185313 / $589824  31.42%
36_KO_FLU_1_#1040#1392#272#1040#624#1392#.jpg
[  0: Parenchyma] #242 $410661 / $589824  69.62%
[  1: SevereInflammation] #179 $171898 / $589824  29.14%
36_KO_FLU_1.jpg
[  0: Parenchyma] #370 $836148 / $1447680  57.76%
[  1: SevereInflammation] #210 $203466 / $1447680  14.05%
```
Data credits: Jeanine D'Armiento, Monica Goldklang, Kyle Stearns; Columbia University Medical Center
