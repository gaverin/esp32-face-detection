import os
import tensorflow as tf
import keras
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image

from train import test_image_classifier_all_classes

if __name__ == "__main__":

    model_name = 'face_classifier.h5'
    face_classifier = keras.models.load_model(f'models/{model_name}')

    test_image_classifier_all_classes(
        face_classifier,
        'dataset-split/test',
    )




