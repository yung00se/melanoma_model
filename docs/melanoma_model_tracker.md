# Melanoma Model Tracker

## Overview
This document provides a brief explanation of how the model will be tracked in terms of data and iteration as well as it will document the team's thought process and changes as the model changes over time.

Tracking model progress is important to show transparency with users and clients across all domains. AI and machine learning models are all tools and therefore must be properly refined to always be a useful tool and to be as accurate as possible. 

## Initial Design Thoughts
The initial idea the team had developed is a python script that properly and accurately tracks changes in the model. This means all important adjustments of the model to keep track of along with a model iteration ID that is unique to a tweaked version of it. 

Using the CSV file reading and writing library the model will be able to be tracked so that the number of convolutional layers, batch size, etc. can all be viewed over each iteration.. 

Using the `model.fit()` function creates a customizable set of data that we can track and feed to the CSV editing script.

We do have alternate options that are built into Tensorflow such as CSVLogger that must be investigated to make sure that these built in tools don't carry too much bloat or that they are able to track and write important information into a CSV file. 

## Model Architecture Summary

The structure the team had opted for was a convolutional neural network (CNN). This is because there has been a lot of empirical studies on how accurate this model structure is. It has been tested in various image processing model experiment such as MNIST database of handwritten digits, being one of the first applications where it succeeded, which shows nearly one-hundred percent accuracy.

The following sections briefly cover the model's structure:

**Convolutional Layers**: These layers simply act as filters for images. They are the layers that distinguish and extract increasingly complex features in the images to pass on to the next layer. 

**Pooling Layer**: This layer is what received the features that were provided by the previous layer. This layer is best known for reducing spatial size of received features and preserving the most important information in them as it helps reduce computation and helps prevent overfitting.

**Dense Layer**: Dense layers are known to interpret the image features in their entirety to finalize the decision between malignant or benign. 

**Activations (ReLU, Sigmoid)**: This layer enables deep learning of complex patterns using Sigmoid to output a 0 or 1 which is ideal for binary outputs such as benign or malignant.

**Regularization (Dropout, BatchNormalize)**: Dropout prevents overfitting through the random deactivation of neurons which fores the network to learn more robust/generalized features, and BatchNormalization maintains it stability and while speeding up training.

Through these layers the team had to do experimental testing and make adjustments where necessary in order to balance the complexity and accuracy of the model. Ensuring that the model's architecture follows organized and standard design.



## Logging Metrics
The final design choice was to create a function that takes extracts information from the `history` and `conf_matrix` to track the following:

- True Negative: Tracks when the model predicts benign cancer in the image, and its correct (image is benign).


- False Negative: tracks when the model predicts benign cancer in the image, and its incorrect (image is malignant).


- True Positive: Tracks when the model predicts malignant cancer in the image, and its correct (image is malignant).


- False Positive: Tracks when the model predicts malignant cancer in the image, and its incorrect (image is benign). 


- Total: Adds up all values for true positives, false positives, true negatives, and false negatives. This simplifies calculations for other metrics.


- Accuracy: Used to measure how often the model's predictions are correct. This is not the end-all-be-all of accuracy. There are still many things that can influence this variable. True accuracy depends on the domain, scope, and details the problem.

- Precision: This focuses on the quality of the model's positive predictions. In this context we would explain it as "out of all the images the model flagged as malignant how many were actually malignant?"


- Recall: Recall is used to measure how often the model identifies positive cases. Specifically, the proportion of true positives out of all actual positive cases. For our context we would describe it as "out of all the actual malignant cases, how many did the model correctly flag as malignant?"


- F1 Score: A metric used to combine precision and recall. This is used to find a more accurate sense of model performance. This is because the false positives and false negatives are crucial since they can have more severe consequences. 

In the next section we will describe why the F1 score matters more in the context of this project.

## Why F1 Score Matters in Medical Diagnosis
In the previous section we briefly discussed the logging metrics and their meanings. F1 score was specifically described to be important due to false positives and false negatives being crucial.

### False Positives
Let's say that a user decides to try our model. They upload their image and test it and get a false positive. Meaning that the image is benign but was predicted to be malignant. 

This may cause unnecessary fear, resource use, and expenses. It cannot necessarily be avoided as machine learning tools are never perfect but for a small project the team cannot allow for this to cause trouble. Hence, why the team plans to make it very clear to the user that this is just a tool and any real concern should result in a consultation with a real medical professional.

### False Negatives 
Now let's say that a user decides to try our mode. They upload their image and test it and get a false negative. Meaning that the image is malignant but was predicted to be benign.

This is detrimental because if someone has genuine concern and the tool tells them they are fine then the user will take this as fact and may not catch the cancer early enough. Therefore, it can have life-altering implications. The team needs to make sure that its clear to the user that the project is not real medical advice and that they should always seek medical professionals for more guidance. 

