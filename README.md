<h1>LungNeuralNet</h1>
We demonstrated that the CNNs, including U-Net and Mask R-CNN, can be instrumental to provide
 i) efficient evaluation of pathological lung lesions;
 ii) detailed characterization of the normal lung histology;
 and iii) precise detection and classification for BALF cells.
 These advanced methods allow improved efficiency and quantification of lung cytology and histopathology.
 
<h2>Applications of U-Net like architectures</h2>

The convolutional neural network architecture was based on U-Net, Convolutional Networks for Biomedical Image Segmentation.
http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

<img src="resource/train_unet.jpg" alt="original Image">


<h3>Lung Pathology</h3>
After training on 14 image pairs, the neural network is able to reach >90% accuracy (dice coefficient) in identifying lung parenchymal region and >60% for severe inflammation in the lung in the validation set.
The prediction results on a separate image, including segmentation mask and area stats, was shown below.
<dl>
    <dt>Multi-label overlay (blue: parenchyma, red: severe inflammation)</dd>
</dl>

|   | Parenchyma  |  SevereInflammation |
|---|---|---|
| 36_KO_FLU_1.jpg | 836148 | 203466 |


![alt text](2x_field_lung_flu/pred/36_KO_FLU_1_both.jpg?raw=true "severe inflammation in the lung")


<h3>Lung Histology</h3>

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
    <dt>Non-Parenchymal Region Highlighted Image</dt>
</dd>


<img src"10x_scan_lung_smoke/pred/027327_2017-12-05 13_53_29_RA5_greenmark.jpg" alt="greem-marked image"/>

<dl>
    <dt>Six-Color Segmentation Map</dt>    
</dd>


![alt text](./10x_scan_lung_smoke/pred/027327_2017-12-05 13_53_29_RA5_pred.jpg?raw=true "6 color segmentation Image")



<h2>Applications of Mask R-CNN</h2>

>Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick. Mask R-CNN. arXiv:1703.06870. <br/>
https://github.com/matterport/Mask_RCNN

MRCNN, based on matterport's implementation, was incorporated and adapted to split/merge tiles from large image, simulate bronchoalveolar lavage from background & representative cell images, and batch-evaluate mean average precisions.


![alt text](resource/train_mrcnn.jpg?raw=true "scheme")


<h3>Broncho-alveolar Lavage Fluid Cytology</h3>

After training and validating (3:1) on 21 background image with 26 lymphocytes, 95 monocytes, and 22 polymorphonuclear leukocytes, the neural network is able to detect and categorize these cell types in a mouse lung bronchoalveolar lavage fluid (20X objective).

![alt text](resource/mrcnn_simulate.jpg?raw=true "train with simulated images")
  
Within one day of training, the accuracy represented by mean average precision has reached 75% for all categories. The accuracy is highest for the monocyte category.<br/>

![alt text](20x_pizz mmp13 ko liver/pred/20x_balf_cells.jpg?raw=true "cell detection and categorization results")

<table style="width:100%">
  <tr>
    <th>CNN Architecture</th> 
    <th>mAP</th>
    <th>Val_mAP</th>
  </tr>
  <tr>
    <td>DenseNet121</td>
    <td>0.846</td> 
    <td>0.744</td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>0.848</td>
    <td>0.750</td>
  </tr>
  <tr>
    <td>Vgg16</td>
    <td>0.838</td>
    <td>0.763</td>
  </tr>
</table>

Data credits: Jeanine D'Armiento, Monica Goldklang, Kyle Stearns; Columbia University Medical Center
