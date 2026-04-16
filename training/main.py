import os
import tensorflow as tf
import keras
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image

from model import Model

# TensorFlow automatically chooses the best performance settings in terms of parallelism and prefetching
AUTOTUNE = tf.data.AUTOTUNE

# dataset paths
train_image_folder = os.path.join('dataset-split', 'train')
test_image_folder = os.path.join('dataset-split', 'test')
val_image_folder = os.path.join('dataset-split', 'val')



# In the lfw dataset each image is a 250x250 jpg detected and centered using the openCV implementation of Viola-Jones
IMG_HEIGHT = 250
IMG_WIDTH = 250

# Training settings
NUM_CLASSES = 3 
BATCH_SIZE = 16
LR = 0.01 # learning rate, should be less than 1.0 and greater than 10^-6. A traditional default value for the learning rate is 0.1 or 0.01
ACTIVATION_FUNCTION = 'softmax'
EPOCHS = 100

# Load datasets
train_ds = keras.preprocessing.image_dataset_from_directory(
    train_image_folder,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    val_image_folder,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = keras.preprocessing.image_dataset_from_directory(
    test_image_folder,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names


# Create model train and evaluate
face_classifier = Model(lr=LR, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, num_classes=NUM_CLASSES, activation_function=ACTIVATION_FUNCTION)
face_classifier.train(train_ds=train_ds, epochs=EPOCHS, val_ds=val_ds)
face_classifier.evaluate(test_ds=test_ds)


