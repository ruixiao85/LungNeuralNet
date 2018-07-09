# LungNeuralNet
Tensorflow Keras Lung

![alt text](test/39_WT_FLU_2_04.jpg?raw=true "Original Image")

![alt text](pred/39_WT_FLU_2_04.jpg_pred.png?raw=true "Recognized as inflammation")

Using TensorFlow backend.
Scanning subfolders [/ori] and [/msk] of [.\LungNeuralNet]
53 images and 60 masks found
53 image-mask pairs accepted
Done: 0/53 images
Done: 1/53 images
Done: 2/53 images
Done: 3/53 images
Done: 4/53 images
Done: 5/53 images
Done: 6/53 images
Done: 7/53 images
Done: 8/53 images
Done: 9/53 images
Done: 10/53 images
Done: 11/53 images
Done: 12/53 images
Done: 13/53 images
Done: 14/53 images
Done: 15/53 images
Done: 16/53 images
Done: 17/53 images
Done: 18/53 images
Done: 19/53 images
Done: 20/53 images
Done: 21/53 images
Done: 22/53 images
Done: 23/53 images
Done: 24/53 images
Done: 25/53 images
Done: 26/53 images
Done: 27/53 images
Done: 28/53 images
Done: 29/53 images
Done: 30/53 images
Done: 31/53 images
Done: 32/53 images
Done: 33/53 images
Done: 34/53 images
Done: 35/53 images
Done: 36/53 images
Done: 37/53 images
Done: 38/53 images
Done: 39/53 images
Done: 40/53 images
Done: 41/53 images
Done: 42/53 images
Done: 43/53 images
Done: 44/53 images
Done: 45/53 images
Done: 46/53 images
Done: 47/53 images
Done: 48/53 images
Done: 49/53 images
Done: 50/53 images
Done: 51/53 images
Done: 52/53 images
Creating model and checkpoint...
Fitting model...
Train on 42 samples, validate on 11 samples
Epoch 1/12
 4/42 [=>............................] - ETA: 56s - loss: -0.3279 - dice_coef: 0.3279
 8/42 [====>.........................] - ETA: 27s - loss: -0.3369 - dice_coef: 0.3369
12/42 [=======>......................] - ETA: 17s - loss: -0.3644 - dice_coef: 0.3644
16/42 [==========>...................] - ETA: 11s - loss: -0.3764 - dice_coef: 0.3764
20/42 [=============>................] - ETA: 8s - loss: -0.3673 - dice_coef: 0.3673
24/42 [================>.............] - ETA: 6s - loss: -0.3627 - dice_coef: 0.3627
28/42 [===================>..........] - ETA: 4s - loss: -0.3824 - dice_coef: 0.3824
32/42 [=====================>........] - ETA: 2s - loss: -0.3837 - dice_coef: 0.3837
36/42 [========================>.....] - ETA: 1s - loss: -0.3859 - dice_coef: 0.3859
40/42 [===========================>..] - ETA: 0s - loss: -0.3750 - dice_coef: 0.3750
42/42 [==============================] - 14s 324ms/step - loss: -0.3761 - dice_coef: 0.3761 - val_loss: -0.3885 - val_dice_coef: 0.3885
Epoch 2/12

 4/42 [=>............................] - ETA: 4s - loss: -0.3637 - dice_coef: 0.3637
 8/42 [====>.........................] - ETA: 3s - loss: -0.3593 - dice_coef: 0.3593
12/42 [=======>......................] - ETA: 3s - loss: -0.3687 - dice_coef: 0.3687
16/42 [==========>...................] - ETA: 2s - loss: -0.3739 - dice_coef: 0.3739
20/42 [=============>................] - ETA: 2s - loss: -0.3868 - dice_coef: 0.3868
24/42 [================>.............] - ETA: 2s - loss: -0.3705 - dice_coef: 0.3705
28/42 [===================>..........] - ETA: 1s - loss: -0.3797 - dice_coef: 0.3797
32/42 [=====================>........] - ETA: 1s - loss: -0.3788 - dice_coef: 0.3788
36/42 [========================>.....] - ETA: 0s - loss: -0.3865 - dice_coef: 0.3865
40/42 [===========================>..] - ETA: 0s - loss: -0.3700 - dice_coef: 0.3700
42/42 [==============================] - 5s 122ms/step - loss: -0.3754 - dice_coef: 0.3754 - val_loss: -0.3894 - val_dice_coef: 0.3894
...
Epoch 12/12

 4/42 [=>............................] - ETA: 4s - loss: -0.8374 - dice_coef: 0.8374
 8/42 [====>.........................] - ETA: 3s - loss: -0.7519 - dice_coef: 0.7519
12/42 [=======>......................] - ETA: 3s - loss: -0.7527 - dice_coef: 0.7527
16/42 [==========>...................] - ETA: 2s - loss: -0.7819 - dice_coef: 0.7819
20/42 [=============>................] - ETA: 2s - loss: -0.7735 - dice_coef: 0.7735
24/42 [================>.............] - ETA: 2s - loss: -0.7442 - dice_coef: 0.7442
28/42 [===================>..........] - ETA: 1s - loss: -0.7115 - dice_coef: 0.7115
32/42 [=====================>........] - ETA: 1s - loss: -0.7042 - dice_coef: 0.7042
36/42 [========================>.....] - ETA: 0s - loss: -0.7066 - dice_coef: 0.7066
40/42 [===========================>..] - ETA: 0s - loss: -0.6901 - dice_coef: 0.6901
42/42 [==============================] - 5s 123ms/step - loss: -0.6787 - dice_coef: 0.6787 - val_loss: -0.7105 - val_dice_coef: 0.7105
Loading and preprocessing test data...
Scanning test subfolders [/test] of [D:\Cel files\2018-07.08 Keras Tensorflow 2X IMAGES\LungNeuralNet]
Done: 0/7 images
Done: 1/7 images
Done: 2/7 images
Done: 3/7 images
Done: 4/7 images
Done: 5/7 images
Done: 6/7 images
Predicting masks on test data...

7/7 [==============================] - 2s 264ms/step
Saving predicted masks to files...

Process finished with exit code 0
