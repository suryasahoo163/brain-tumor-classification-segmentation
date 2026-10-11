# Import Libraries
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, 
                                     Dropout, BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import HeUniform, GlorotUniform
from sklearn.preprocessing import StandardScaler
import numpy as npa
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import losses
from sklearn.metrics import classification_report

# ================================
# 1. Load and Preprocess Data
# ================================
DATASET_PATH = r"C:\Users\surya_hwzjtax\Desktop\datascience\BrainTumour\MRI Image Classification"

def load_data(dataset_path):
    images = []
    labels = []
    
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                img = cv2.resize(img, (256, 256))  # Resize to 256x256
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)

# Load train and test data
train_images, train_labels = load_data(os.path.join(DATASET_PATH,"Training"))
test_images, test_labels = load_data(os.path.join(DATASET_PATH,"Testing"))

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape to add channel dimension (since grayscale = 1 channel)
train_images = train_images.reshape(-1, 256, 256, 1)
test_images = test_images.reshape(-1, 256, 256, 1)

# Encode labels
encoder = LabelEncoder()
train_labels = encoder.fit_transform(train_labels)
test_labels = encoder.transform(test_labels)

# Convert to categorical for softmax output
num_classes = len(np.unique(train_labels))
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

print(f"Train shape: {train_images.shape}, Test shape: {test_images.shape}")
print(f"Number of classes: {num_classes}")

# ================================
# 2. Create CNN Model
# ================================
def build_cnn(input_shape, num_classes):
    
    model=Sequential()
    model.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu",kernel_initializer="he_uniform",padding="same",input_shape=(256,256,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu",kernel_initializer="he_uniform",padding="same",input_shape=(256,256,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=128,kernel_size=(3,3),activation="relu",kernel_initializer="he_uniform",padding="same",input_shape=(256,256,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units=128,activation="relu",kernel_initializer="he_uniform"))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes,activation="softmax"))
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    return model
    
  
    
    return model

# Build model
model = build_cnn((256, 256, 1), num_classes)
model.summary()



# ================================
# 4. Train the Model
# ================================
history = model.fit(train_images, train_labels, batch_size=16,
                    epochs=20,
                    validation_data=(test_images, test_labels))

# ================================
# 5. Evaluate the Model
# ================================
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# ================================
# 6. Plot Training Curves
# ================================
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.savefig(r'C:\Users\surya_hwzjtax\Desktop\datascience\BrainTumour\Models\acc_loss_graph.png')
plt.show()

# ================================
# 7. Save the Model
# ================================
model.save(r'C:\Users\surya_hwzjtax\Desktop\datascience\BrainTumour\Models\brain_tumor_cnn.h5')
print("Model saved as 'brain_tumor_cnn.h5'")


# ================================
# 8. Load and Predict (Optional)
# ================================
# Load model
# model = tf.keras.models.load_model('brain_tumor_cnn.h5')

# Example prediction (replace with actual test data)
# example_img = test_images[0].reshape(1, 256, 256, 1)
# prediction = model.predict(example_img)
# predicted_class = np.argmax(prediction)
# print(f"Predicted Class: {encoder.inverse_transform([predicted_class])[0]}")
