# Melanoma Model Tracker

## Overview
This document provides a brief explanation of how the model will be tracked in terms of data and iteration as well as it will document the team's thought process and changes as the model changes over time. 

## Initial Design Thoughts
The initial idea the team had developed is a python script that properly and accurately tracks changes in the model. This means all important outputs to keep track of along with a model iteration ID that is unique to a tweaked version of that model. 

Using the CSV file reading and writing library the model will be able to be tracked so that improvements can be shown overtime. 

Using the `model.fit()` function creates a customizable set of data that we can track and feed to the CSV editing script.

We do have alternate options that are built into Tensorflow such as CSVLogger that must be investigated to make sure that these built in tools don't carry too much bloat or that they don't carry the ability to track and write important information into a CSV file. 


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

