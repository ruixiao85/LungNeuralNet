# LungNeuralNet

>Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick. Mask R-CNN. arXiv:1703.06870.
https://github.com/matterport/Mask_RCNN

![alt text](../resource/train_mrcnn.jpg?raw=true "scheme")

After training and validating (3:1) on 21 background image with 26 lymphocytes, 95 monocytes, and 22 polymorphonuclear leukocytes, the neural network is able to detect and categorize these cell types in a mouse lung bronchoalveolar lavage fluid (20X objective).

![alt text](../resource/mrcnn_simulate.jpg?raw=true "train with simulated images")

MRCNN, based on matterport's implementation, was incorporated and adapted to split/merge tiles from large image, simulate bronchoalveolar lavage from background & representative cell images, and batch-evaluate mean average precisions.

  
Within one day of training, the accuracy represented by mean average precision has reached 75% for all categories. The accuracy is highest for monocyte category.<br/>

![alt text](pred/20x_balf_cells.jpg?raw=true "cell detection and categorization results")

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
