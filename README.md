# ESP32 Face Detection

This project trains a small face classifier in TensorFlow/Keras and exports it for use on an ESP32-based camera device.

## What It Includes

- `training/`: dataset preparation, model training, evaluation, and TensorFlow Lite export
- `python/`: reference Python workflow used to generate the embedded model assets
- `inference/esp32/main/`: camera code and generated model files for the embedded app
- `training/data/og_data_split/`: train, validation, and test folders used by the training pipeline

## Model Pipeline

The training code uses a MobileNetV2-based classifier for 3 classes.  
After training, the model is converted to an `int8` TensorFlow Lite model and exported to:

- `training/models/face_classifier.tflite`
- `inference/esp32/main/model_data.cc`
- `inference/esp32/main/model_data.h`

## Usage

1. Change into the `python/` folder and run the Python entry point:

```bash
cd python
python main.py
```

2. Change back out and build the ESP32 firmware:

```bash
cd ../inference/esp32
idf.py build
```

3. Flash the device and open the serial monitor:

```bash
idf.py flash monitor
```

This workflow generates the model assets first, then rebuilds and flashes the ESP32 application with the updated model.
