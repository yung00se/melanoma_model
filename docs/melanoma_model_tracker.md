# Melanoma Model Tracker

## Overview
This document provides a brief explanation of how the model will be tracked in terms of data and iteration as well as it will document the team's thought process and changes as the model changes over time. 

## Initial Design Thoughts
The initial idea the team had developed is a python script that properly and accurately tracks changes in the model. This means all important outputs to keep track of along with a model iteration ID that is unique to a tweaked version of that model. 

Using the CSV file reading and writing library the model will be able to be tracked so that improvements can be shown overtime. 

Using the `model.fit()` function creates a customizable set of data that we can track and feed to the CSV editing script.

We do have alternate options that are built into Tensorflow such as CSVLogger that must be investigated to make sure that these built in tools don't carry too much bloat or that they don't carry the ability to track and write important information into a CSV file. 


