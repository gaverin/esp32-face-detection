cmake_minimum_required(VERSION 3.16)

# ── ESP-IDF project ───────────────────────────────────────────────────────────
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(esp32s3_cam_classifier)

# ── External components ───────────────────────────────────────────────────────
# esp-camera  – https://github.com/espressif/esp32-camera
# esp-tflite-micro – https://github.com/espressif/esp-tflite-micro
#
# Add to components/ or reference via idf_component.yml
