#pragma once

#include "esp_err.h"
#include "esp_camera.h"

/**
 * Initialise the OV2640 camera module on the ESP32-S3-CAM board.
 * Must be called once before camera_pipeline_capture().
 */
esp_err_t camera_pipeline_init(void);

/**
 * Capture a single JPEG frame.
 * Caller MUST call camera_pipeline_return() after use.
 *
 * @return  Non-null pointer on success, nullptr on failure.
 */
camera_fb_t *camera_pipeline_capture(void);

/**
 * Return the frame buffer obtained from camera_pipeline_capture().
 * Always call this, even on error paths, to avoid buffer starvation.
 */
void camera_pipeline_return(camera_fb_t *fb);
