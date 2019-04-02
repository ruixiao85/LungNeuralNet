# LungNeuralNet

The convolutional neural network architecture was based on U-Net, Convolutional Networks for Biomedical Image Segmentation.
http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

![alt text](../resource/train_unet.jpg?raw=true "original Image")

After training and validating (3:1) on 16 image pairs, the neural network is able to identify a variety of areas in a normal mouse lung section (equivalent to 10X, cropped from whole slide scan).

UNet was modified for larger inputs and to learn both local details (particularly helpful for small blood vessels) and general spatial context (valuable for learning the background). Each label was trained separately into individual neural network with sigmoid output function and loss function with combination of binary crossentropy and dice loss. This will 1) allow each network to function separately, 2) concentrate the computational power on each category, and 3) avoid the imbalance problem that may occur to softmax output and multi-clas crossentropy loss.  
With one day of training, the accuracy as indicated by dice coefficient has reached to more than 80% for most categories.<br/>

><b>single-class (sigmoid) dice coefficient:</b> <br/>
background: 97% <br/>
conducting airway: 84% <br/>
connective tissue: 83% <br/>
large blood vessel: 78% <br/>
respiratory airway: 97% <br/>
small blood vessel: 63% <br/>

Since these categories should constitute 100% of the image and are mutually exclusive, the combined output can be further processed with softmax to select the category/label with the highest probability.

><b>multi-class (softmax) accuracy:</b><br/> 96%

The method can be helpful to identify and quantify various structures or tissue types in the lung and extensible to developmental abnormality or diseased areas.


<dl>
    <dt>Original Image</dt>
</dl>

![alt text](pred/027327_2017-12-05 13_53_29_RA5_original.jpg?raw=true "original image")

<dl>
    <dt>Non-Parenchymal Region Highlighted Image</dt>
</dd>

![alt text](pred/027327_2017-12-05 13_53_29_RA5_greenmark.jpg?raw=true "greem-marked image")

<dl>
    <dt>Six-Color Segmentation Map</dt>    
</dd>

![alt text](pred/027327_2017-12-05 13_53_29_RA5_pred.jpg?raw=true "6 color segmentation Image")


Data credits: Jeanine D'Armiento, Monica Goldklang, Kyle Stearns; Columbia University Medical Center
