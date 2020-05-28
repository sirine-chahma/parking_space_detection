# parking_space_detection
School project realized with Paul Marchenoir

## Description

It usually takes around 7 minutes for someone to find an available parking spot. It represents roughly 30% of urban traffic and is one of the reasons of traffic jams during rush hours. Most of the solutions that have been found so fare can only be used indoor, or on private areas, and they are often costly.

The solution that we found would be to use a drone that could fly above a street, find an empty parking spot and send its coordinates to a user (through a mobile phone for example). Our project consists in only a subset of this solution. In this repository, you will find the code we used to distinguish an empty parking spot from a busy one.

## Usage

Download the data from the following [link](http://cnrpark.it/) and save it in a `data` folder in the root directory.

To run the fully-connected model using Keras :
- Run the `create_training_data_train` script 
- Run the `NN_model_Keras` script

To run the fully-connected model without Keras :
- Run the `create_training_data_train` script 
- Run the `NN_model_Tensorflow` script

To run the CNN :
- Run the `create_training_data_train` script 
- Run the `CNN_model` script

To run our example : 
- Run the fully-connected model using Keras
- Run the `rogner` script to apply a filter on the images
- Run the `notre_test` script