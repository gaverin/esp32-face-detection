#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "driver/gpio.h"
#include "esp_log.h"

// Project includes
#include "model.h"
#include "camera.h"


// Static variables
static float features[IMG_WIDTH * IMG_HEIGHT];
static float prediction[NUM_CLASSES];
static const char *TAG_INF = "Inference";

/*
    Inference pipeline should be something like take:
    
    1. take picture
    2. crop face only and match size of the model
    3. use tflite model for inference
    4. return picture with label 

*/

/**
 * @brief Main setup function.
 */
void setup(void) 
{
    // setup the camera
    if (!camera_init())
    {
        ESP_LOGE(TAG_INF, "Failed to initialize camera!");
        abort();
    }

    //setup the classification model 
    if (!classifier_init())
    {
        ESP_LOGE(TAG_INF, "Failed to initialize classification model!");
        abort();
    }
    
    //setup the face recognition 


}

/**
 * @brief Main loop function.
 */
void loop(void)
{
    /*
        Capture frame_buffers, detect face, crop, predict, send image + prediction
    */
}

extern "C" void app_main(void)
{
    setup();
    while (true)
    {
        loop();
    }
}

