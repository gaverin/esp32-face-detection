import shutil
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from utils import print_confusion_matrix, write_files

"""
    Pretrained ResNet50 without its original classification head (include_top=False) act as a feature extractor
    All its layers are frozen so their weights are not updated during training
    The GlobalAveragePooling2D layer converts the spatial feature maps into a compact feature vector
    The Dense layer performs the actual classification, producing one output per class
    The full model is then compiled with a loss function and optimizer, while callbacks like ModelCheckpoint and EarlyStopping monitor validation performance to save the best model and stop training early
"""

MODEL_C_PATH = "../inference/esp32/main/model_data.cc"
MODEL_H_PATH = "../inference/esp32/main/model_data.h"
MODEL_KERAS_PATH = Path("models/face_classifier.keras")
MODEL_TFLITE_PATH = Path("models/face_classifier.tflite")
SAVED_MODEL_DIR = Path("models/face_classifier_saved_model")
MOBILENET_ALPHA = 0.35


class Model:

    def __init__(self, lr, img_height=224, img_width=224, num_classes=3, activation_function="softmax"):

        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.activation_function = activation_function
        self.learning_rate = lr

        self.base_model = keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            alpha=MOBILENET_ALPHA,
            input_shape=(self.img_height, self.img_width, 3),
            name="mobilenetv2_backbone",
        )

        for layer in self.base_model.layers:
            layer.trainable = False

        self.augmentation = keras.Sequential(
            [
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.1),
                keras.layers.RandomZoom(0.1),
                keras.layers.RandomBrightness(0.2),
            ],
            name="training_augmentation",
        )

        inputs = keras.Input(shape=(self.img_height, self.img_width, 3), dtype=tf.float32)
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = self.base_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(self.num_classes)(x)

        self.face_classifier = keras.Model(inputs, outputs, name="MobileNetV2")
        self._compile_model(self.face_classifier, learning_rate=self.learning_rate)

        self.trained_model = None
        self.tflite_model = None

    def _compile_model(self, model: keras.Model, learning_rate: float):
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["accuracy"],
        )

    def _prepare_dataset(self, dataset: tf.data.Dataset, training: bool = False):
        autotune = tf.data.AUTOTUNE

        def map_batch(images, labels):
            images = tf.cast(images, tf.float32)
            if training:
                images = self.augmentation(images, training=True)
            labels = self._labels_to_sparse(labels)
            return images, labels

        prepared_ds = dataset.map(map_batch, num_parallel_calls=autotune).prefetch(autotune)

        if hasattr(dataset, "class_names"):
            prepared_ds.class_names = dataset.class_names

        return prepared_ds

    def _labels_to_sparse(self, labels):
        labels = tf.convert_to_tensor(labels)

        if labels.shape.rank is not None and labels.shape.rank > 1:
            return tf.argmax(labels, axis=-1, output_type=tf.int32)

        if labels.dtype.is_floating:
            labels = tf.cast(tf.round(labels), tf.int32)
        else:
            labels = tf.cast(labels, tf.int32)

        return tf.reshape(labels, [-1])

    def _build_callbacks(self):
        return [
            EarlyStopping(
                monitor="val_loss",
                restore_best_weights=True,
                patience=5,
            ),
            ModelCheckpoint(
                MODEL_KERAS_PATH.as_posix(),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                patience=3,
                factor=0.5,
            ),
        ]

    def _fine_tune(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, epochs: int):
        base = self.face_classifier.get_layer("mobilenetv2_backbone")
        base.trainable = True

        for layer in base.layers[:-30]:
            layer.trainable = False

        self._compile_model(self.face_classifier, learning_rate=1e-5)

        fine_tune_epochs = max(1, epochs // 3)
        self.face_classifier.fit(
            train_ds,
            epochs=fine_tune_epochs,
            validation_data=val_ds,
            callbacks=self._build_callbacks(),
        )

    def _get_class_names(self, dataset: tf.data.Dataset):
        if hasattr(dataset, "class_names"):
            return dataset.class_names
        return [str(index) for index in range(self.num_classes)]

    def train(self, train_ds: tf.data.Dataset, epochs, val_ds: tf.data.Dataset):

        prepared_train_ds = self._prepare_dataset(train_ds, training=True)
        prepared_val_ds = self._prepare_dataset(val_ds, training=False)
        callbacks = self._build_callbacks()

        history = self.face_classifier.fit(
            prepared_train_ds,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=prepared_val_ds,
        )

        self._fine_tune(prepared_train_ds, prepared_val_ds, epochs)

        print("Final training accuracy:", history.history["accuracy"][-1])
        print("Final validation accuracy:", history.history["val_accuracy"][-1])

        if MODEL_KERAS_PATH.exists():
            self.trained_model = keras.models.load_model(MODEL_KERAS_PATH)
        else:
            self.trained_model = self.face_classifier
            self.trained_model.save(MODEL_KERAS_PATH)

    def get_last_trained_model(self):
        if self.trained_model is None:
            return keras.models.load_model(MODEL_KERAS_PATH)
        return self.trained_model

    def evaluate_model(self, test_ds: tf.data.Dataset):
        model = self.get_last_trained_model()
        prepared_test_ds = self._prepare_dataset(test_ds, training=False)
        _, acc = model.evaluate(prepared_test_ds, verbose=0)
        print(f"Tensorflow Model Test Accuracy {acc * 100:.2f}%")

    def export_model_to_tflite(self, train_ds: tf.data.Dataset, enable_quantization: bool = True):
        print("Converting to TensorFlow Lite model...")

        model = self.get_last_trained_model()
        if SAVED_MODEL_DIR.exists():
            shutil.rmtree(SAVED_MODEL_DIR)
        model.export(SAVED_MODEL_DIR.as_posix())
        converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR.as_posix())

        if enable_quantization:
            def representative_dataset():
                for images, _ in train_ds.unbatch().shuffle(1000).batch(1).take(200):
                    images = tf.cast(images, tf.float32)
                    images = keras.applications.mobilenet_v2.preprocess_input(images)
                    yield [images]

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        self.tflite_model = converter.convert()

        if enable_quantization:
            interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            print("Input scale:", input_details[0]["quantization"][0])
            print("Input zero point:", input_details[0]["quantization"][1])
            print("Output scale:", output_details[0]["quantization"][0])
            print("Output zero point:", output_details[0]["quantization"][1])

        print("Exporting TensorFlow Lite model to C source files...")
        MODEL_TFLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_TFLITE_PATH.write_bytes(self.tflite_model)
        write_files(MODEL_TFLITE_PATH, Path(MODEL_C_PATH), Path(MODEL_H_PATH))

    def evaluate_tflite_model(self, test_ds: tf.data.Dataset):
        if self.tflite_model is None:
            interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE_PATH.as_posix())
        else:
            interpreter = tf.lite.Interpreter(model_content=self.tflite_model)

        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_scale, input_zero_point = input_details[0]["quantization"]
        output_scale, output_zero_point = output_details[0]["quantization"]

        print("input shape:", input_details[0]["shape"])
        print("input dtype:", input_details[0]["dtype"])
        print("input quant:", input_details[0]["quantization"])
        print("output dtype:", output_details[0]["dtype"])
        print("output quant:", output_details[0]["quantization"])
        print("class names:", self._get_class_names(test_ds))

        y_true = []
        y_pred = []

        for images, labels in test_ds:
            images = tf.cast(images, tf.float32)
            images = keras.applications.mobilenet_v2.preprocess_input(images).numpy()
            labels = self._labels_to_sparse(labels).numpy()

            if input_scale > 0:
                images_q = np.round(images / input_scale + input_zero_point)
                images_q = np.clip(images_q, -128, 127).astype(np.int8)
            else:
                images_q = images.astype(np.float32)

            for i in range(images_q.shape[0]):
                interpreter.set_tensor(
                    input_details[0]["index"],
                    images_q[i:i + 1],
                )
                interpreter.invoke()

                output = interpreter.get_tensor(output_details[0]["index"])[0]

                if output_scale > 0:
                    output = (output.astype(np.float32) - output_zero_point) * output_scale

                pred = int(np.argmax(output))

                y_pred.append(pred)
                y_true.append(int(labels[i]))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        accuracy = np.mean(y_true == y_pred)

        print(f"\nTFlite Model Test Accuracy: {accuracy * 100:.4f}%\n")

        class_names = self._get_class_names(test_ds)
        print_confusion_matrix(y_true, y_pred, class_names)
