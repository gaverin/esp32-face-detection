idf_component_register(
    SRCS
        "main.cpp"
        "camera_pipeline.cpp"
        "classifier.cpp"
        "model_data.cc"          # generated; replace with real model file
    INCLUDE_DIRS
        "."
    REQUIRES
        espressif__esp32-camera
        espressif__esp-tflite-micro
        nvs_flash     
        mbedtls
)
 
