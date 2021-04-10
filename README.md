# Drowsiness-Detection-Web-app
Prevents sleep deprivation road accidents, by alerting drowsy drivers.
In this project, we have trained a convlutional neural network, to determine whether the eyes are closed or not, further, eye-ptches are extracted from the face image to make all predictions. The dataset used for training process can be accessed from <a href="https://www.kaggle.com/kutaykutlu/drowsiness-detection" target="_blank">here.</a>

## Understanding The Problem Statement
  According to the survey done by 'The Times of Inidia', nearly 40% of the road accidents are caused by the sleep deprivation. Fatigued drivers, long-duty driving are the major causes for the same. To solve this issue, this app primarily aims to predict whether or not the driver is sleeping, if found sleeping, it alerts the driver by making a high frequency sound. This project is to avoid such sleep deprivation accidents!
  
  ## Implementation
1. A Deep Learning Model will be trained to detect whether the driver's eyelids are open or not. This will be achieved by training a Convolutional Neural Network using Tensorflow.<br>
2. A web-cam will take a picture of the driver's face at regular intervals of time and the patch of the driver's eye will be extracted from that picture. This task will be     <tab>achieved by using OpenCV.<br>
3. This patch will be further used for the prediction purpose with the model trained in step 1.<br>
4. Using this prediction, if the driver's eyes are closed a beep sound will be played, to alert the driver.<br>

## Drowsiness Detetction Model Insights
This model is trained with the help of tensorflow and is based upon convolutional neural networks. It takes RGB images with the dimensions (86 * 86 * 3).
### Model Architecure
<table>
  <th>Layer Number</th><th>Layer Type</th><th>Output Shape</th><th>Trainable Parameters</th><th>Activation Funtion</th>
  <tr><td>1</td><td>CONV2D</td><td>(None, 84, 84, 75)</td><td>2,100</td><td>Relu</td></tr>
  <tr><td>2</td><td>MaxPooling2D</td><td>(None, 16, 16, 75)</td><td>0</td><td>None</td></tr>
  <tr><td>3</td><td>Conv2D</td><td>(None, 15, 15, 64)</td><td>19,264</td><td>Relu</td></tr>
  <tr><td>4</td><td>MaxPooling2D</td><td>(None, 7, 7, 64)</td><td>0</td><td>None</td></tr>
  <tr><td>5</td><td>Conv2D</td><td>(None, 5, 5, 128)</td><td>73,856</td><td>Relu</td></tr>
  <tr><td>6</td><td>MaxPooling2D</td><td>(None, 2, 2, 128)</td><td>0</td><td>None</td></tr>
  <tr><td>7</td><td>Flattern</td><td>(None, 512)</td><td>0</td><td>None</td></tr>
  <tr><td>8</td><td>Dense</td><td>(None, 64)</td><td>32,832</td><td>Sigmoid</td></tr>
  <tr><td>9</td><td>Dense</td><td>(None, 2)</td><td>130</td><td>Softmax</td></tr>
</table>

## Eye_patch_Extractor_and_predictor Insights
This model uses OpenCV's "Haar Cascade Classifier" for face detection and after the proposal of region of interest it extracts the eye-patch by the "Centroid Method" developed by us. This extracted features will be then passed to trained model for Drowsiness Detection.
