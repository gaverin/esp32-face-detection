import numpy as np
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import write_model_c_file, write_model_h_file, print_confusion_matrix

"""
    Pretrained ResNet50 without its original classification head (include_top=False) act as a feature extractor
    All its layers are frozen so their weights are not updated during training
    The GlobalAveragePooling2D layer converts the spatial feature maps into a compact feature vector
    The Dense layer performs the actual classification, producing one output per class
    The full model is then compiled with a loss function and optimizer, while callbacks like ModelCheckpoint and EarlyStopping monitor validation performance to save the best model and stop training early
"""

MODEL_C_PATH = '../inference/esp32/main/model.c'
MODEL_H_PATH = '../inference/esp32/main/model.h'

class Model:


    def __init__(self, lr, img_height=224, img_width=224, num_classes=3, activation_function='softmax'):

        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.activation_function = activation_function
        self.learning_rate = lr

        self.base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )

        for layer in self.base_model.layers:
            layer.trainable = False

        inputs = keras.Input(shape=(self.img_height, self.img_width, 3), dtype=tf.float32)
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = self.base_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(self.num_classes, activation=self.activation_function)(x)

        self.face_classifier = keras.Model(inputs, outputs, name='MobileNetV2')

        self.face_classifier.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )

        self.trained_model = None
        self.tflite_model = None

    def train(self, train_ds: tf.data.Dataset, epochs, val_ds: tf.data.Dataset):

        checkpoint = ModelCheckpoint(
            "models/face_classifier.keras",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        )
        # EarlyStopping to find best model with a large number of epochs
        earlystop = EarlyStopping(
            monitor='val_loss',
            restore_best_weights=True,
            patience=3,  # number of epochs with no improvement after which training will be stopped
        )
        callbacks = [earlystop, checkpoint]
        
        history = self.face_classifier.fit(
            train_ds,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_ds
        )   
        
        print("Final training accuracy:", history.history['accuracy'][-1])
        print("Final validation accuracy:", history.history['val_accuracy'][-1])

        self.trained_model = keras.models.load_model("models/face_classifier.keras")
    

    def get_last_trained_model(self):
        if self.trained_model == None:
            return keras.models.load_model("models/face_classifier.keras")
        else:
            return self.trained_model

    def evaluate_model(self, test_ds: tf.data.Dataset):
        model = self.get_last_trained_model()
        _ , acc = model.evaluate(test_ds, verbose=0)
        print(f"Tensorflow Model Test Accuracy {acc * 100:.2f}%")
    

    def export_model_to_tflite(self, train_ds: tf.data.Dataset, enable_quantization: bool = True):
        print('Converting to TensorFlow Lite model...')
        
        model = self.get_last_trained_model()        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if enable_quantization:
            # Define the generator for the representative dataset
            def representative_dataset():
                for images, _ in train_ds.unbatch().shuffle(1000).batch(1).take(200):
                    yield [tf.cast(images, tf.float32)]

            # Set up quantization parameters
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            # Enforce full integer quantization
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        self.tflite_model = converter.convert()
        
        # Print quantization scale and zero point
        if enable_quantization:
            # Load model in interpreter
            interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
            interpreter.allocate_tensors()

            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Do print
            print('Input scale:', input_details[0]['quantization'][0])
            print('Input zero point:', input_details[0]['quantization'][1])
            print('Output scale:', output_details[0]['quantization'][0])
            print('Output zero point:', output_details[0]['quantization'][1])

        # Export TensorFlow Lite model to C source files
        print('Exporting TensorFlow Lite model to C source files...')
        defines = {
            'NUM_CLASSES': self.num_classes,
            'IMG_WIDTH': self.img_width,
            'IMG_HEIGHT': self.img_height
        }
        declarations = []
        write_model_h_file(MODEL_H_PATH, defines, declarations)
        write_model_c_file(MODEL_C_PATH, self.tflite_model)

        # Save TensorFlow Lite model
        with open('models/face_classifier.tflite', 'wb') as f:
            f.write(self.tflite_model)

    def evaluate_tflite_model(self, test_ds: tf.data.Dataset):
        # Load interpreter
        if self.tflite_model == None:
            interpreter = tf.lite.Interpreter(model_path="models/face_classifier.tflite")
        else:
            interpreter = tf.lite.Interpreter(model_content=self.tflite_model)

        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_scale, input_zero_point = input_details[0]['quantization']
        output_scale, output_zero_point = output_details[0]['quantization']

        print("input shape:", input_details[0]["shape"])
        print("input dtype:", input_details[0]["dtype"])
        print("input quant:", input_details[0]["quantization"])
        print("output dtype:", output_details[0]["dtype"])
        print("output quant:", output_details[0]["quantization"])
        print("class names:", test_ds.class_names)

        y_true = []
        y_pred = []

        for images, labels in test_ds:
            images = images.numpy().astype(np.float32)
            labels = np.argmax(labels.numpy(), axis=1)

            # Quantize batch
            if input_scale > 0:
                images_q = np.round(images / input_scale + input_zero_point)
                images_q = np.clip(images_q, -128, 127).astype(np.int8)
            else:
                images_q = images.astype(np.float32)
            
            # input_dtype is int8 as defined in the convertion method above
            images_q = np.clip(images_q, -128, 127).astype(np.int8)
            # use when we do not quantize tflite model, Did this to check quantization bug that dropped accuracy of tflite model
            #images_q = images.astype(np.float32)
            
            # Run inference per sample
            for i in range(images_q.shape[0]):
                interpreter.set_tensor(
                    input_details[0]['index'],
                    images_q[i:i+1]
                )
                interpreter.invoke()

                output = interpreter.get_tensor(output_details[0]['index'])[0]

                if output_scale > 0:
                    output = (output.astype(np.float32) - output_zero_point) * output_scale

                pred = np.argmax(output)

                y_pred.append(pred)
                y_true.append(labels[i])

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        accuracy = np.mean(y_true == y_pred)

        print(f"\nTFlite Model Test Accuracy: {accuracy * 100:.4f}%\n")

        class_names = test_ds.class_names
        print_confusion_matrix(y_true, y_pred, class_names)
