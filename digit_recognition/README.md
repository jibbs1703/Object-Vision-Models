# Image/ Object Classification and Detection

## Overview
This repository contains projects that use models created from deep learning frameworks in image/object classification and detection.

## Projects
**Digit Recognition** : The images for the digit recognition project was obtained from [Kaggle](https://www.kaggle.com/competitions/digit-recognizer/data),
consisting of gray-scale images of hand-drawn digits, from zero through nine. Each image was 784 pixels (28 x 28 in dimension). A simple convoluted neural
network was developed using the training set and consequently used to predict the digits in the hand-drawn images of the testing set. The neural network 
achieved a 98.76% accuracy in predicting the hand-drawn digits in the images of the testing dataset.

**Pet Classification** : The images for the pet classification project was obtained from [Kaggle](https://www.kaggle.com/datasets/salader/dogs-vs-cats).
The images consisted of colored images two pet types - cats and dogs. The images had varying dimensions and were imported at a target size of 256 X 256.
Two convoluted neural networks were developed using the training set and consequently used to predict what pet images were in the testing set. The first 
neural network was self-contructed while the second was built upon a pre-trained image classification model.
The self-constructed neural network had three convoluted and three pooling layers within the hidden layer and achieved an 82% accuracy in predicting the
pet images of the testing dataset. On the other hand, the neural network built on the pre-trained model achieved a 99% accuracy in predicting the
pet images of the testing dataset. To further test the models, I downloaded 10 random images from [X](https://x.com), and the self-constructed model correctly predicted
7 of the 12 pet images correctly while the model built on the pre-trained model correctly predicted 10 of the 12 pet images correctly.

**Plant Disease Detection** : The images for the pet classification project was obtained from