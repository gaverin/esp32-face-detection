# ESP32 Face Detection

This project trains a small face classifier in TensorFlow/Keras and exports it for use on an ESP32-based camera device.

## What It Includes

- `training/`: dataset preparation, model training, evaluation, and TensorFlow Lite export
- `inference/esp32/main/`: camera code and generated model files for the embedded app
- `training/data/og_data_split/`: train, validation, and test folders used by the training pipeline
You need to upload your own data and use the methods from the `data.py` module to process and split your dataset

## Model Pipeline

The training code uses a MobileNetV2-based classifier for 3 classes.  
After training, the model is converted to an `int8` TensorFlow Lite model and exported to:

- `training/models/face_classifier.tflite`
- `inference/esp32/main/model_data.cc`
- `inference/esp32/main/model_data.h`

## Usage

You can run the full training, build, flash, and monitor pipeline with the helper script from the repository root:

```bash
./run-pipeline.sh
```

You can also run the steps manually (commands for espressif-idf v6, activation might be  different).

1. Change into the `training/` folder and run the training/export pipeline:

```bash
cd training
uv run python main.py
```

2. Change into the ESP32 project and build the firmware:

```bash
cd ../inference/esp32
source ~/.espressif/tools/activate_idf_v6.0.sh
idf.py build
```

3. Flash the device and open the serial monitor:

```bash
idf.py flash monitor
```
