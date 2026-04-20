# ESP32 Face Detection

This project trains a small face classifier in TensorFlow/Keras and exports it for use on an ESP32-based camera device.

## What It Includes

- `training/`: dataset preparation, model training, evaluation, and TensorFlow Lite export
- `esp32/main/`: camera code and generated model files for the embedded app
- `training/dataset-split/`: train, validation, and test folders used by the training pipeline

## Model Pipeline

The training code uses a frozen `ResNet50` backbone with a small classification head for 3 classes.  
After training, the model is converted to an `int8` TensorFlow Lite model and exported to:

- `training/models/model.tflite`
- `esp32/main/model.c`
- `esp32/main/model.h`

## Run

From `training/`, run:

```bash
python main.py
```

This loads the dataset, trains or loads the latest saved model, and exports the TFLite model for the ESP32 project.
esp32/main/model.c