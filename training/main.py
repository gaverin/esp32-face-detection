import os
import tensorflow as tf
import keras
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image

train_image_folder = os.path.join('dataset-split', 'train')
test_image_folder = os.path.join('dataset-split', 'test')
val_image_folder = os.path.join('dataset-split', 'val')


# In the lfw dataset each image is a 250x250 jpg detected and centered using the openCV implementation of Viola-Jones
img_height, img_width = 250, 250  # size of images

num_classes = 3 

# Training settings
batch_size = 16
lr = 0.1 # learning rate
activation_function = 'softmax'
epochs = 50

# TensorFlow automatically chooses the best performance settings in terms of parallelism and prefetching
AUTOTUNE = tf.data.AUTOTUNE

# Load datasets
train_ds = keras.preprocessing.image_dataset_from_directory(
    train_image_folder,
    image_size=(img_height, img_width),
    label_mode='categorical',
    batch_size=batch_size,
    shuffle=True
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    val_image_folder,
    image_size=(img_height, img_width),
    label_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)

test_ds = keras.preprocessing.image_dataset_from_directory(
    test_image_folder,
    image_size=(img_height, img_width),
    label_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)

class_names = test_ds.class_names


"""
    Pretrained ResNet50 without its original classification head (include_top=False) act as a feature extractor
    All its layers are frozen so their weights are not updated during training
    The GlobalAveragePooling2D layer converts the spatial feature maps into a compact feature vector
    The Dense layer performs the actual classification, producing one output per class
    The full model is then compiled with a loss function and optimizer, while callbacks like ModelCheckpoint and EarlyStopping monitor validation performance to save the best model and stop training early
"""

base_model = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

for layer in base_model.layers:
    layer.trainable = False

global_avg_pooling = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(num_classes, activation=activation_function)(global_avg_pooling)

face_classifier = keras.models.Model(
    inputs=base_model.input,
    outputs=output,
    name='ResNet50'
)

face_classifier.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    metrics=['accuracy']
)

# Training

checkpoint = ModelCheckpoint(
    "models/face_classifier.h5", # .h5 stands for HDF5 (Hierarchical Data Format)
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1
)

# EarlyStopping to find best model with a large number of epochs
earlystop = EarlyStopping(
    monitor='val_loss',
    restore_best_weights=True,
    patience=3,  # number of epochs with no improvement after which training will be stopped
    verbose=1
)

callbacks = [earlystop, checkpoint]

# Training loop
history = face_classifier.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds
)

# Saves the best model thanks to EarlyStopping
face_classifier.save("models/face_classifier.h5")

print("Final training accuracy:", history.history['accuracy'][-1])
print("Final validation accuracy:", history.history['val_accuracy'][-1])


# testing

def test_image_classifier_with_folder(model, path, y_true, img_height=img_height, img_width=img_width, class_names=class_names):
    '''
    Read all images from 'path' using tensorflow.keras.preprocessing.image module, 
    than classifies them using 'model' and compare result with 'y_true'.
    Calculate total accuracy based on 'path' test set.

    Parameters:
        model : Image classifier
        path (str): Path to the folder with images you want to test classifier on 
        y_true : True label of the images in the folder. Must be in 'class_names' list
        img_height (int): The height of the image that the classifier can process 
        img_width (int): The width of the image that the classifier can process
        class_names (array-like): List of class names 

    Returns:
        None
    '''
    num_classes = len(class_names)  # Number of classes
    total = 0  # number of images total
    correct = 0  # number of images classified correctly

    for filename in os.listdir(path):
        # read each image in the folder and classifies it
        test_path = os.path.join(path, filename)
        test_image = image.load_img(test_path, target_size=(img_height, img_width, 3))
        # from image to array, can try type(test_image)
        test_image = image.img_to_array(test_image)
        # shape from (250, 250, 3) to (1, 250, 250, 3)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)

        y_pred = class_names[np.array(result[0]).argmax(axis=0)]  # predicted class
        iscorrect = 'correct' if y_pred == y_true else 'incorrect'
        print('{} - {}'.format(iscorrect, filename))
        for index in range(num_classes):
            print("\t{:6} with probabily of {:.2f}%".format(class_names[index], result[0][index] * 100))

        total += 1
        if y_pred == y_true:
            correct += 1

    print("\nTotal accuracy is {:.2f}% = {}/{} samples classified correctly".format(
        correct/total*100, correct, total))
    


if __name__ == "__main__":


    model_name = 'face_classifier.h5'
    face_classifier = keras.models.load_model(f'models/{model_name}')

    test_image_classifier_with_folder(
        face_classifier,
        'dataset-split/test/tm1',
        y_true='tm1'
    )

    test_image_classifier_with_folder(
        face_classifier,
        'dataset-split/test/tm2',
        y_true='tm2'
    )




