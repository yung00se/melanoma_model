# TensorFlow Tools Guide

## Introduction
This guide provides a brief overview of tools relevant to the melanoma model from the TensorFlow library. For a more in depth guide please see TensorFlow's website documentation.

## First Iteration Tools 
During the first iteration of the Melanoma Model the team emphasized a rough and simple design much like what you may see in other convolutional neural network tutorials. 

```Python
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
```
In the code snippet above TensorFlow is imported into the program as it will be the main library used for the machine learning model and there are many online resources for it. 

From the TensorFlow library the team imports Sequential. The Sequential model is appropriate because the project is an image processing tool use for identifying skin cancer. Meaning that the linear structure of the CNN style model will be easy to implement using sequential.


