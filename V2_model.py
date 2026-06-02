from google.colab import drive
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

#Mounting Google Drive
drive.mount('/content/drive')

#Downloading and extracting dataset
!cp /content/drive/MyDrive/BrainTumor_Dataset.zip /content/
!unzip -q -o /content/BrainTumor_Dataset.zip

#Mixed precision for performance
mixed_precision.set_global_policy("mixed_float16")

#Model Hyperparameters
random_seed = 42
epochs = 75
batch_size = 32
img_height, img_width = 224, 224
input_shape = (img_height, img_width, 3)
learning_rate = 0.00005
validation_split = 0.1
spatial_dropout_rate = 0.2
dropout_rate = 0.2

#Loading datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    '/content/Training',
    validation_split=validation_split,
    subset="training",
    seed=random_seed,
    shuffle=True,
    batch_size=batch_size,
    image_size=(img_height, img_width)
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    '/content/Training',
    validation_split=validation_split,
    subset="validation",
    seed=random_seed,
    shuffle=True, 
    batch_size=batch_size,
    image_size=(img_height, img_width)
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    '/content/Testing',
    shuffle=False,
    batch_size=batch_size,
    image_size=(img_height, img_width)
)

class_names = train_ds.class_names
print(f"Class names: {class_names}")

#Cache optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Callbacks
local_checkpoint_path = '/content/best_brain_tumor_model.keras'
drive_checkpoint_path = '/content/drive/MyDrive/Brain_Tumor_Models/best_brain_tumor_model.keras'
os.makedirs(os.path.dirname(drive_checkpoint_path), exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(filepath=local_checkpoint_path, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
]

#Model Architecture
#Small random geometric augmentations
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])

model = tf.keras.models.Sequential([
    layers.Input(shape=input_shape),
    layers.Rescaling(1./255),
    data_augmentation,
    #Convolutional Block 1
    layers.Conv2D(32, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    #Convolutional Block 2
    layers.Conv2D(128, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(256, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.SpatialDropout2D(spatial_dropout_rate),
    layers.MaxPooling2D((2, 2)),
    #Convolutional Block 3
    layers.Conv2D(512, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.SpatialDropout2D(spatial_dropout_rate),
    layers.Conv2D(1024, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.SpatialDropout2D(spatial_dropout_rate),
    layers.MaxPooling2D((2, 2)),
    #Dense Layers
    layers.GlobalAveragePooling2D(),
    layers.Dropout(dropout_rate),
    layers.Dense(128, activation='relu'),
    layers.Dropout(dropout_rate),
    layers.Dense(4, activation='softmax', dtype='float32')
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Model training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks
)

#Saving the best model to Google Drive
if os.path.exists(local_checkpoint_path):
    print(f"\nCopying best model to Drive: {drive_checkpoint_path}")
    shutil.copy(local_checkpoint_path, drive_checkpoint_path)
    print("Model copied successfully!")

#Loading the best model version
if os.path.exists(local_checkpoint_path):
    print(f"\nLoading best model from {local_checkpoint_path}...")
    model = tf.keras.models.load_model(local_checkpoint_path)
    if 'history' in globals():
        #Printing of train and val accuracy/loss for the best model run
        best_epoch = np.argmax(history.history['val_accuracy'])
        print(f"\n Best Model Training Summary (Epoch {best_epoch + 1})")
        print(f"Training Accuracy:   {history.history['accuracy'][best_epoch]:.4f}")
        print(f"Training Loss:       {history.history['loss'][best_epoch]:.4f}")
        print(f"Validation Accuracy: {history.history['val_accuracy'][best_epoch]:.4f}")
        print(f"Validation Loss:     {history.history['val_loss'][best_epoch]:.4f}")
        print("---------------------------------------------------\n")
else:
    print("Could not find saved best model, proceed with current model weights")

# Helper to get labels and predictions from a dataset
def get_labels_and_preds(dataset):
    y_true = []
    y_pred = []
    for x, y in dataset:
        y_true.append(y.numpy())
        preds = model(x, training=False)
        y_pred.append(tf.argmax(preds, axis=1).numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)

#Test set evaluation (Only run once hyperparameters are tuned)
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

#Predicitons for each image in the corresponding dataset
y_true_train, y_pred_train = get_labels_and_preds(train_ds)
y_true_val, y_pred_val = get_labels_and_preds(val_ds)
y_true_test, y_pred_test = get_labels_and_preds(test_ds)

#Plot of the confusion matrix
def plot_confusion_matrix(y_true, y_pred, title, ax):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names))
    )
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Oranges',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_title(title)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

#Generating confusion matrices for train, val, and test
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
plot_confusion_matrix(y_true_train, y_pred_train, 'Training Set Confusion Matrix', axes[0])
plot_confusion_matrix(y_true_val, y_pred_val, 'Validation Set Confusion Matrix', axes[1])
plot_confusion_matrix(y_true_test, y_pred_test, 'Testing Set Confusion Matrix', axes[2])
plt.tight_layout()
plt.show()

#Test precision, recall, and F1
print("\nTest Set Detailed Classification Report:")
print(classification_report(
    y_true_test,
    y_pred_test,
    labels=np.arange(len(class_names)),
    target_names=class_names
))
