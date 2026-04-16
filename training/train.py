import os
import tensorflow as tf
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from model import Model

train_image_folder = os.path.join('dataset-split', 'train')
test_image_folder = os.path.join('dataset-split', 'test')
val_image_folder = os.path.join('dataset-split', 'val')


# In the lfw dataset each image is a 250x250 jpg detected and centered using the openCV implementation of Viola-Jones
img_height, img_width = 250, 250  # size of images

num_classes = 3 

# Training settings
batch_size = 16
lr = 0.1 # learning rate, should be less than 1.0 and greater than 10^-6. A traditional default value for the learning rate is 0.1 or 0.01
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

# model definition and training

face_classifier = Model(lr=lr, img_height=img_height, img_width=img_width, num_classes=num_classes, activation_function=activation_function)

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
# start training
face_classifier.train(train_ds=train_ds, epochs=epochs, callbacks=callbacks, val_ds=val_ds)


def test_image_classifier_all_classes(model, path, img_height=img_height, img_width=img_width, num_classes=num_classes, class_names=class_names):
    
    if (len(os.listdir(path)) != num_classes):
        raise ValueError(
            f"Number of classes in the dataset does not match expected number"
        )
    """
        I want to calculate the overall model accuracy for all the classes in the dataset
    """
    total = 0 
    correct = 0

    for class_name in os.listdir(path):
        class_dir = os.path.join(path, class_name)
        if os.path.isdir(class_dir):
            true_label = class_name
            for image_name in os.listdir(class_dir):
                # load the image
                image_path = os.path.join(class_dir, image_name)
                test_image = image.load_img(image_path, target_size=(img_height, img_width, 3))
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis=0)
                # predict the label
                result = model.predict(test_image)
                predicted_label = class_names[np.array(result[0]).argmax(axis=0)]
                # update counters
                total += 1
                if true_label == predicted_label:
                    correct += 1

    # calculate and print result
    overall_accuracy = correct/total*100
    print("\Overall accuracy is {:.2f}% = {}/{} samples classified correctly".format(
        overall_accuracy, correct, total))


def test_image_classifier_single_class(model, path, y_true, img_height=img_height, img_width=img_width, class_names=class_names):
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

    test_image_classifier_all_classes(
        face_classifier,
        'dataset-split/test',
    )