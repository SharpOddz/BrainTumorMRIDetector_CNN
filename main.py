from google.colab import drive
import random
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

#Mount Google Drive. If the dataset is not within your Google Drive this code is irrelevant
drive.mount('/content/drive')

#Random seed setting for reproducibility
random.seed(23)

#Base path for the dataset. Adjust as needed 
base_path = '/content/drive/MyDrive/Brain_Tumor_MRI_Dataset'

#Loading in the dataset (training and testing sets)
def load_dataset(base_dir):
    data = []
    for split in ['Training', 'Testing']:
        split_path = os.path.join(base_dir, split)
        if not os.path.exists(split_path):
            continue
        for label in ['glioma', 'meningioma', 'notumor', 'pituitary']:
            label_path = os.path.join(split_path, label)
            if not os.path.exists(label_path):
                continue
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                data.append({'filepath': image_path, 'label': label, 'split': split})
    return pd.DataFrame(data)

df = load_dataset(base_path)

#Verify that there are 1400 images for each of the four classes for training set, 400 images for each class for testing set
if df.empty:
    print(f"Dataset was not found. Please ensure the path '{base_path}' is correct and contains 'Training' and 'Testing' folders.")
else:
    print("Dataset Summary:")
    display(df.groupby(['split', 'label']).size().unstack(fill_value=0))
    display(df.head())

#Target image size
IMG_SIZE = (224, 224)

#Resizing image and creating train and test data structures
def prepare_data(df, split_name):
    x = []
    y = []
    # Filter for the specific split
    split_df = df[df['split'] == split_name]
    label_map = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
    print(f"Loading {len(split_df)} {split_name} images...")
    for _, row in split_df.iterrows():
        img = cv2.imread(row['filepath'])
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            x.append(img)
            y.append(label_map[row['label']])
            
    return np.array(x), np.array(y)

#Train and test datasets. 
x_train, y_train = prepare_data(df, 'Training')
x_test, y_test = prepare_data(df, 'Testing')

#Splitting the training set into a train/val split. The validation split pct can be adjusted with the val_split parameter
val_split = 0.10
x_train, x_val, y_train, y_val = train_test_split(
    x_train, 
    y_train, 
    test_size=val_split, 
    shuffle=True, 
    random_state=23
)

print(f"Training set: {x_train.shape}, {y_train.shape}")
print(f"Validation set: {x_val.shape}, {y_val.shape}")
print(f"Testing set: {x_test.shape}, {y_test.shape}")
