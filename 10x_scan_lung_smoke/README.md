# LungNeuralNet

The convolutional neural network architecture used in this project was inspired by [U-Net](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) and [dual frame U-Net](https://arxiv.org/abs/1708.08333) with added transfer learning from pre-trained models in keras ([keras-applications](https://keras.io/applications/)).

![alt text](../resource/train_unet.jpg?raw=true "train_unet")

After training and validating (3:1) on **16 whole slide scans**, the neural network is able to identify a variety of areas in a normal mouse lung section (equivalent to 10X, cropped from whole slide scan).

Variations of U-Nets were built to perform
 - **single-class segmentation**
    - output: sigmoid
    - loss: dice & binary crossentropy
    - metrics: dice
 - **multi-class segmentation**
    - output: softmax
    - loss: multiclass crossentropy
    - metrics: accuracy

Among them, dual-frame slightly outperform U-Net with single-frame.
Although more time consuming, single-class segmentation combined with argmax achieved a better classification results than those done by one multi-class segmentation model,
especially for the underrepresented categories.

The best results are listed below:
 - **single-class segmentation** (dice)
    - background: 97%
    - conducting airway: 84%
    - connective tissue: 83%
    - large blood vessel: 78%
    - respiratory airway: 97%
    - small blood vessel: 63%
 - **multi-class segmentation** (accuracy)
    - all six categories: 96%
    
These methods are helpful for identifing and quantifing various structures or tissue types in the lung and extensible to developmental abnormality or diseased areas.

Original Image

![alt text](pred/027327_original.jpg?raw=true "original image")

Non-Parenchymal Region Highlighted Image

![alt text](pred/027327_greenmark.jpg?raw=true "greem-marked image")

Six-Color Segmentation Map

![alt text](pred/027327_pred.jpg?raw=true "6 color segmentation Image")

Data credits: Jeanine D'Armiento, Monica Goldklang, Kyle Stearns; Columbia University Medical Center
