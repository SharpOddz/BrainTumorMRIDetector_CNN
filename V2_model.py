import os
import shutil
import random
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from google.colab import drive
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

#Mounting Google Drive (not needed if the dataset is local)
drive.mount('/content/drive')

#Setting random seeds for reproducibility
random.seed(23)
np.random.seed(23)
tf.random.set_seed(23)


gdrive_path = '/content/drive/MyDrive/Brain_Tumor_MRI_Dataset'
local_path = '/content/Brain_Tumor_MRI_Dataset'

#Copying dataset to local drive (more important if mounting google drive)
if not os.path.exists(local_path):
    if os.path.exists(gdrive_path):
        print(f"Copying dataset from {gdrive_path} to {local_path}...")
        shutil.copytree(gdrive_path, local_path)
        print("Copy complete.")
    else:
        print(f"Error: Dataset not found at {gdrive_path}")
else:
    print("Dataset already exists on local disk")
  #Loading dataset
def load_dataset(base_dir):
    data = []
    splits = ['Training', 'Testing']
    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
    for split in splits:
        split_path = os.path.join(base_dir, split)
        if not os.path.exists(split_path): continue
        for label in labels:
            label_path = os.path.join(split_path, label)
            if not os.path.exists(label_path): continue
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                data.append({'filepath': image_path, 'label': label, 'split': split})
    return pd.DataFrame(data)

#Utilizing local dataset rather than from the google drive
base_path = local_path
df = load_dataset(base_path)

if df.empty:
    print(f"Dataset was not found at {base_path}")
else:
    print("Local Dataset Summary:")
    display(df.groupby(['split', 'label']).size().unstack(fill_value=0))
    display(df.head())

#Preprocessing
'''
  Resize image to the target input size for the CNN
  Normalize pixel values to be between 0 and 1
  One-hot encode the labels
  Split the dataset into training, validation, and testing sets
  Save the preprocessed data for future use
'''
#Target input image size
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

def process_image_path(path_label_tuple):
    path, label_idx = path_label_tuple
    img = cv2.imread(path)
    if img is not None:
        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        # Normalize to [0, 1]
        img = img.astype('float32') / 255.0
        return img, label_idx
    return None

def prepare_data(df, split_name):
    # Filter for the specific split
    split_df = df[df['split'] == split_name]
    label_map = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
    # Prepare tuples of (path, label_index) for parallel processing
    tasks = list(zip(split_df['filepath'].values, split_df['label'].map(label_map).values))
    print(f"Loading and preprocessing {len(tasks)} {split_name} images in parallel...")
    # Use ThreadPoolExecutor with controlled max_workers to avoid overloading
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_image_path, tasks))
    # Filter out None values and separate images and labels
    valid_results = [r for r in results if r is not None]
    x, y = zip(*valid_results)
    # Use np.stack for efficient array creation
    return np.stack(x), np.array(y)

# Process the datasets
x_train, y_train = prepare_data(df, 'Training')
x_test, y_test = prepare_data(df, 'Testing')

#Splitting the training set into a train/val split. The validation split pct can be adjusted with the val_split parameter
val_split = 0.10
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=val_split,
    shuffle=True,
    random_state=23,
    stratify=y_train
)

print(f"Training set: {x_train.shape}, {y_train.shape}")
print(f"Validation set: {x_val.shape}, {y_val.shape}")
print(f"Testing set: {x_test.shape}, {y_test.shape}")

import tensorflow as tf
from tensorflow.keras import layers
import os

#Model Hyperparameters
learning_rate = 0.00025
epochs = 75
batch_size = 32
dropout_rate = 0.20

#The model will save to google drive or local path (adjust if not using Google Drive)
checkpoint_path = '/content/drive/MyDrive/Brain_Tumor_Models/best_brain_tumor_model.keras'
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

#Callbacks (early stopping, best model saving, learning rate reduction)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    patience=10,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001,
    verbose=1
)
#Random data augmentations (kept small on purpose to create realistic augmentations)
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])
#Model Architecture
'''

'''
model = tf.keras.models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    data_augmentation,
    #Convolutional Layer 1
    layers.Conv2D(16, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),
    #Convolutional Layer 2
    layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),
    #Convolutional Layer 3
    layers.Conv2D(256, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(512, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),
    #Global average pooling -> output
    layers.GlobalAveragePooling2D(),
    layers.Dropout(dropout_rate),
    layers.Dense(4, activation='softmax')
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(
    x_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, lr_reduction, checkpoint],
    verbose=1
)

import matplotlib.pyplot as plt

# Plot training & validation accuracy and loss
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Execute the plot
if 'history' in globals():
    plot_training_history(history)
else:
    print("History object not found. Please run the training cell first.")

#Verification that the best model was saved to the correct path
if os.path.exists(checkpoint_path):
    print(f"The best model was saved to: {checkpoint_path}")
else:
    print("Manual save triggered...")
    model.save(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_confusion_matrix(y_true, y_pred, title, ax):
    cm = confusion_matrix(y_true, y_pred)
    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

# Model Load Check and Training Metrics Summary
if os.path.exists(checkpoint_path):
    print(f"Loading best model from {checkpoint_path}...")
    model = tf.keras.models.load_model(checkpoint_path)

    if 'history' in globals():
        # Find the index of the best validation accuracy
        best_epoch = np.argmax(history.history['val_accuracy'])
        print(f"\n--- Best Model Training Summary (Epoch {best_epoch + 1}) ---")
        print(f"Training Accuracy:   {history.history['accuracy'][best_epoch]:.4f}")
        print(f"Training Loss:       {history.history['loss'][best_epoch]:.4f}")
        print(f"Validation Accuracy: {history.history['val_accuracy'][best_epoch]:.4f}")
        print(f"Validation Loss:     {history.history['val_loss'][best_epoch]:.4f}")
        print("---------------------------------------------------\n")
else:
    print("Best model file not found. Proceeding with current model weights.")

# Explicit test evaluation
print("Evaluating on Test Set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Get predictions
y_pred_train = model.predict(x_train).argmax(axis=1)
y_pred_val = model.predict(x_val).argmax(axis=1)
y_pred_test = model.predict(x_test).argmax(axis=1)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
plot_confusion_matrix(y_train, y_pred_train, 'Training Set Confusion Matrix', axes[0])
plot_confusion_matrix(y_val, y_pred_val, 'Validation Set Confusion Matrix', axes[1])
plot_confusion_matrix(y_test, y_pred_test, 'Testing Set Confusion Matrix', axes[2])
plt.tight_layout()
plt.show()

# Detailed report including Precision, Recall, and F1 Score
print("\nTest Set Detailed Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=['glioma', 'meningioma', 'notumor', 'pituitary']))
