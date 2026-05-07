#pragma once

#include "esp_err.h"
#include "esp_camera.h"

/**
 * Result from a single inference pass.
 */
struct ClassifierResult {
    float  prob[3];         ///< [0]=member1, [1]=member2, [2]=unknown
    char   label[32];       ///< Winning label string (e.g. "Team Member 1 (87.3%)")
    int    class_idx;       ///< 0, 1, or 2
};

/**
 * Initialise TFLite interpreter and load the model from flash.
 * Call once at startup.
 */
esp_err_t classifier_init(void);

/**
 * Pre-process the camera frame and run MobileNetV2 inference.
 *
 * @param fb     Frame from camera_pipeline_capture() (RGB565, 320×240).
 * @param result Filled with probabilities and winning label.
 * @return ESP_OK on success.
 */
esp_err_t classifier_run(const camera_fb_t *fb, ClassifierResult *result);
