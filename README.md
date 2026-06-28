# BrainTumorMRIDetector_CNN
Utilizing a Convolutional Neural Network to detect tumors in brain MRI images. There are several different versions of CNN architecture are used in this repository. Each version's arcitecture are detailed in the "Model Architecture and Hyperparameters" section and are compared in the result section. Version 1 (v1) was built first and used as a baseline for further versions. Once each model version has been optimized it was run once on the test set as a final benchmark. For fair comparison, each model resizes the input image to 224x224 (different model versions will do additional preprocessing steps). 

## Requirements
Downloading of the dataset is required to run this model (The dataset can be found below). Required libraries: 

## Model Architecture & Hyperparameters

#### Version 1 (V1)
V1's architecture is a simple CNN struture and is used as the baseline for this project. It takes input image of 224x224x3 and goes through five convolutional layers (each with a 3x3 filter, doubling number of filters, batch normalization, ReLU activations, and 2x2 max pooling to halve spatial resolution). The sixth and final convolutional layer uses 512 1x1 filters before global average pooling that fully connects to the output layer. For preprocessing: image resizing to 224x224 and light data augmentations (small rotations of up to 5 degrees, small zooming in/out (0.1), and random contrast (0.1)). 

<img width="433" height="539" alt="image" src="https://github.com/user-attachments/assets/90706f6a-c46d-4710-b5a4-33f1f4a70ba7" />


The training/validation loss/accuracy history plot for V1 is shown below:

<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/c381f11a-639a-45ad-b466-d3bfcb930df8" />

#### Version 2 (V2) 
There are several speed and performance optimizations made in V2, especially noticeable if using Google Colab to run the model. V2 adds additional convolutional layers while keeping spatial resolution larger. It also adds spatial dropout to the last convolutional layers. The training/val loss/accuracy history plot was lost and is not shown. The model trained for 71 epochs, with the best model restored at epoch 61. 

<img width="189" height="541" alt="image" src="https://github.com/user-attachments/assets/8efff88a-13ef-4dd5-a75a-4f6e4c9a4cea" />


#### Version 3 (V3) 
Similar to V1 but with residual blocks. Regular dropout is used for the fully connected layer but no spatial dropout is used in V3.

<img width="188" height="536" alt="image" src="https://github.com/user-attachments/assets/7da775ae-d064-4cab-b2ad-cbd61431e66b" />

#### Version 4 (V4)
Adds squeeze and excititation attention. This model has SE-ResNet-Style blocks.

## Results

#### Version 1 (V1)
For version 1 of this model (V1), the training, validation, and test accuracy was 93.67%, 89.29%, and 85.94% respectively. Confusion matrices for each data set is shown below.

<img width="1960" height="590" alt="image" src="https://github.com/user-attachments/assets/54128eef-3aa7-491f-ac63-75dea2cd6329" />
<img width="456" height="129" alt="image" src="https://github.com/user-attachments/assets/98a0281f-7428-423f-819a-b55d8fedc137" />

#### Version 2 (V2)
For the V2 model the training, validation, and test accuracies were 99.03%, 96.43%, and 89.25% respectively. Corresponding confusion matrices and test precision/recall/F1 scores are shown below:

<img width="1960" height="590" alt="image" src="https://github.com/user-attachments/assets/43d7e83a-f5a7-4809-80cd-61b87fb4ceec" />
<img width="461" height="136" alt="image" src="https://github.com/user-attachments/assets/ca7dcc56-ae51-4876-b8a5-6893b233f172" />

#### Version 3 (V3)
For the V3 model the training, validation, and test accuracies were 98.59%, 98.04%, and 90.50% respectively. Corresponding confusion matrices and test precision/recall/F1 scores are shown below:

<img width="1960" height="590" alt="image" src="https://github.com/user-attachments/assets/a3a5b2a2-43eb-4096-994d-31021fc678a0" />
<img width="461" height="139" alt="image" src="https://github.com/user-attachments/assets/d5402b7d-7cc1-4283-8e89-dcc1f95689be" />

#### Version 4 (V4)
For the V4 model the training, validation, and test accuracies were 98.67%, 96.61%, and 89.31% respectively. Corresponding confusion matrices and test precision/recall/F1 scores are shown below:

<img width="1960" height="590" alt="image" src="https://github.com/user-attachments/assets/5b9c038b-cee6-4fbb-a295-624d37059a8c" />
<img width="451" height="129" alt="image" src="https://github.com/user-attachments/assets/7c40f977-0c63-4069-8f08-6b9c68f4f248" />




## Dataset
Dataset for this project can be found at: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
