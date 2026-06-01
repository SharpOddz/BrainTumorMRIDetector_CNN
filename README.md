# BrainTumorMRIDetector_CNN
Utilizing a Convolutional Neural Network to detect tumors in brain MRI images. There are several different versions of CNN architecture are used in this repository. Each version's arcitecture are detailed in the "Model Architecture and Hyperparameters" section and are compared in the result section. Version 1 (v1) was built first and used as a baseline for further versions. Once each version has been optimized it was run once on the test set. 

## Requirements
Downloading of the dataset is required to run this model (The dataset can be found below). Required libraries: 

## Model Architecture & Hyperparameters

#### Version 1 (V1)
V1's architecture is a simple CNN struture and is used as the baseline for this project. It takes input image of 224x224x3 and goes through five convolutional layers (each with a 3x3 filter, doubling number of filters, batch normalization, ReLU activations, and 2x2 max pooling to halve spatial resolution). The sixth and final convolutional layer uses 512 1x1 filters before global average pooling that fully connects to the output layer. For preprocessing: image resizing to 224x224 and light data augmentations (small rotations of up to 5 degrees, small zooming in/out (0.1), and random contrast (0.1)). The training/validation loss/accuracy history plot is shown below for V1:

<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/c381f11a-639a-45ad-b466-d3bfcb930df8" />

#### Version 2 (V2) (Not finished tuning as of 5/31/2026)
V2 adds additional convolutional layers while keeping spatial resolution larger. It also adds spatial dropout to the last 3 convolutional layers.

## Results

#### Version 1 (V1)
For version 1 of this model (V1), the training, validation, and test accuracy was 93.67%, 89.29%, and 85.94% respectively. Confusion matrices for each data set is shown below.

<img width="1960" height="590" alt="image" src="https://github.com/user-attachments/assets/54128eef-3aa7-491f-ac63-75dea2cd6329" />
<img width="456" height="129" alt="image" src="https://github.com/user-attachments/assets/98a0281f-7428-423f-819a-b55d8fedc137" />



## Dataset
Dataset for this project can be found at: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
