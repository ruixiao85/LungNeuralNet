# LungNeuralNet

The convolutional neural network architecture was based on U-Net, Convolutional Networks for Biomedical Image Segmentation.
http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

After training on 14 image pairs, the neural net can identify lung parenchymal region, mild inflammation and severe inflammation in the lung from a new image.

<dl>
  <dt>Definition list</dt>
  <dd>Is something people use sometimes.</dd>

  <dt>Markdown in HTML</dt>
  <dd>Does *not* work **very** well. Use HTML <em>tags</em>.</dd>
</dl>

Original Image
![alt text](1_single_label_sigmoid/pred/Original/36_KO_FLU_1.jpg?raw=true "original Image")

Predicted to be lung parenchymal region
sum of pixels: 852367.1
![alt text](1_single_label_sigmoid/pred/Paren/36_KO_FLU_1.png?raw=true "lung parenchymal region")

Predicted to be mild inflammation in the lung
sum of pixels: 447767.1 (53% of the parechyma)
![alt text](1_single_label_sigmoid/pred/InflamMild/36_KO_FLU_1.png?raw=true "mild inflammation in the lung")

Predicted to be severe inflammation in the lung
sum of pixels: 226655.1 (27% of the parechyma)
![alt text](1_single_label_sigmoid/pred/InflamSevere/36_KO_FLU_1.png?raw=true "severe inflammation in the lung")

# Training
```
Using TensorFlow backend.
Found [14] file from subfolders [/Original] of [.\LungNeuralNet\1_single_label_sigmoid\train]
Found [15] file from subfolders [/InflamMild] of [.\LungNeuralNet\1_single_label_sigmoid\train]
14 image-mask pairs accepted
Done: 2/14 images [10%]
Done: 3/14 images [20%]
Done: 5/14 images [30%]
Done: 6/14 images [40%]
Done: 7/14 images [50%]
Done: 9/14 images [60%]
Done: 10/14 images [70%]
Done: 12/14 images [80%]
Done: 13/14 images [90%]
Done: 14/14 images [100%]
Creating model and checkpoint...
Fitting model...
Train on 9 samples, validate on 5 samples
Epoch 1/8
1/9 [==>...........................] - ETA: 3s - loss: -0.2825 - dice_coef: 0.2825
2/9 [=====>........................] - ETA: 3s - loss: -0.4343 - dice_coef: 0.4343
3/9 [=========>....................] - ETA: 2s - loss: -0.4981 - dice_coef: 0.4981
4/9 [============>.................] - ETA: 2s - loss: -0.5211 - dice_coef: 0.5211
5/9 [===============>..............] - ETA: 1s - loss: -0.5727 - dice_coef: 0.5727
6/9 [===================>..........] - ETA: 1s - loss: -0.6302 - dice_coef: 0.6302
7/9 [======================>.......] - ETA: 0s - loss: -0.6260 - dice_coef: 0.6260
8/9 [=========================>....] - ETA: 0s - loss: -0.6407 - dice_coef: 0.6407
9/9 [==============================] - 5s 564ms/step - loss: -0.6577 - dice_coef: 0.6577 - val_loss: -0.6899 - val_dice_coef: 0.6899
...
```

# Prediction
```
Found [1] file from subfolders [/Original] of [.\LungNeuralNet\1_single_label_sigmoid\pred]
  mean 0.70 std 0.16
Done: 1/1 images [100%]
Load weights and predicting ...

1/1 [==============================] - 0s 147ms/step
Saving predicted results [InflamMild] to files...
36_KO_FLU_1.jpg pixel sum: 1755.9
```