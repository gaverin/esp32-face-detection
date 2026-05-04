#include <cstdio>
#include <cstdint>
#include <cstring>

// ESP includes
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "esp_log.h"

// Project includes
#include "camera.h"
#include "inference.h"
#include "model.h"


// Static constants and variables
//static const char* PREDICTION_PREAMBLE = "\n===PREDICTION===\n";
//static constexpr size_t CHUNK_SIZE = 256;

// Inference
static uint8_t image_buffer[FRAME_W * FRAME_H * FRAME_C];
static float prediction[NUM_CLASSES];
static const char *TAG_INF = "Inference";

void setup()
{
    // Initialize NVS (required by some drivers)
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    // Initialize camera
    if (!camera_init()) {
        ESP_LOGE(TAG_INF, "Failed to initialize camera!");
        abort();
    }

    // Initialize inference
    if (!classifier_init())
    {
        ESP_LOGE(TAG_INF, "Failed to initialize inference!");
        abort();
    }
}

void loop(void)
{
    // Capture frame into tensor
    if (camera_capture_frame(image_buffer)) {
  
        classifier_put_image(image_buffer);

        if (!classifier_predict(prediction))
        {
            ESP_LOGE(TAG_INF, "Failed to invoke interpreter!");
        }
        else
        {
            // Print output
            ESP_LOGI(TAG_INF, "tm1: %.2f, tm2: %.2f, unkn: %.2f", prediction[0], prediction[1], prediction[2]);

        }
    }

    // Wait ~1 second
    vTaskDelay(pdMS_TO_TICKS(1000));
}

// ---------- ESP-IDF entry point ----------

extern "C" void app_main()
{
    setup();
    classifier_init();
    while (true) {
        loop();
    }
}
