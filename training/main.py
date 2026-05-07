import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image

from model import Model

# TensorFlow automatically chooses the best performance settings in terms of parallelism and prefetching
AUTOTUNE = tf.data.AUTOTUNE

# dataset paths
train_image_folder = os.path.join('data', 'og_data_split', 'train')
test_image_folder = os.path.join('data','og_data_split', 'test')
val_image_folder = os.path.join('data','og_data_split', 'val')



# We resize frames to 
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Training settings
NUM_CLASSES = 3 
BATCH_SIZE = 16
LR = 0.001 # matches the reference head-training phase before fine-tuning
ACTIVATION_FUNCTION = 'softmax'
EPOCHS = 50

# Load datasets
train_ds = keras.preprocessing.image_dataset_from_directory(
    train_image_folder,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    label_mode='int',
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    val_image_folder,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    label_mode='int',
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = keras.preprocessing.image_dataset_from_directory(
    test_image_folder,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    label_mode='int',
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names


# Create model train and evaluate
face_classifier = Model(lr=LR, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, num_classes=NUM_CLASSES, activation_function=ACTIVATION_FUNCTION)
#face_classifier.train(train_ds=train_ds, epochs=EPOCHS, val_ds=val_ds)
#face_classifier.evaluate_model(test_ds=test_ds)
#face_classifier.export_model_to_tflite(train_ds=train_ds)
face_classifier.evaluate_tflite_model(test_ds=test_ds)

