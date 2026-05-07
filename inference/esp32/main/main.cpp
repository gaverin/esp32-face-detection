/**
 * ESP32-S3 Camera + MobileNetV2 Face/Person Classifier
 *
 * Pipeline:
 *   Camera Capture → Preprocess → MobileNetV2 TFLite Inference → Serial Output
 *
 * Output format:
 *   Team Member 1: 87.3%
 *   Team Member 2: 5.1%
 *   Unknown:       7.6%
 */

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "nvs_flash.h"

#include "camera_pipeline.h"
#include "classifier.h"
#include "config.h"

static const char *TAG = "MAIN";

extern "C" void app_main(void)
{
    // ── NVS init (required by camera driver) ──────────────────────────────
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_LOGI(TAG, "=== ESP32-S3 Cam Classifier ===");
    ESP_LOGI(TAG, "Model:      MobileNetV2 (TFLite)");
    ESP_LOGI(TAG, "Classes:    %s / %s / Unknown", CLASS_LABEL_0, CLASS_LABEL_1);
    ESP_LOGI(TAG, "Resolution: %dx%d → model input %dx%d",
             CAM_WIDTH, CAM_HEIGHT, MODEL_INPUT_W, MODEL_INPUT_H);

    // ── Initialise camera ─────────────────────────────────────────────────
    ESP_LOGI(TAG, "Initialising camera...");
    ESP_ERROR_CHECK(camera_pipeline_init());

    // ── Initialise TFLite classifier ─────────────────────────────────────
    ESP_LOGI(TAG, "Loading MobileNetV2 model...");
    ESP_ERROR_CHECK(classifier_init());

    // ── Main inference loop ───────────────────────────────────────────────
    ESP_LOGI(TAG, "Starting inference loop (every %d ms)\n", INFERENCE_INTERVAL_MS);

    ClassifierResult result;

    while (true) {
        // 1. Grab a frame
        camera_fb_t *fb = camera_pipeline_capture();
        if (!fb) {
            ESP_LOGW(TAG, "Camera capture failed, retrying...");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        // 2. Run inference
        esp_err_t err = classifier_run(fb, &result);
        camera_pipeline_return(fb);   // always return frame buffer ASAP

        if (err != ESP_OK) {
            ESP_LOGW(TAG, "Inference error: %s", esp_err_to_name(err));
            vTaskDelay(pdMS_TO_TICKS(INFERENCE_INTERVAL_MS));
            continue;
        }

        // 3. Pretty-print results
        printf("\n┌─────────────────────────────────┐\n");
        printf("│  Classification Result          │\n");
        printf("├─────────────────────────────────┤\n");
        printf("│  %-14s  %6.1f %%         │\n", CLASS_LABEL_0, result.prob[0] * 100.0f);
        printf("│  %-14s  %6.1f %%         │\n", CLASS_LABEL_1, result.prob[1] * 100.0f);
        printf("│  %-14s  %6.1f %%         │\n", "Unknown",     result.prob[2] * 100.0f);
        printf("├─────────────────────────────────┤\n");
        printf("│  >> %-28s │\n", result.label);
        printf("└─────────────────────────────────┘\n");

        vTaskDelay(pdMS_TO_TICKS(INFERENCE_INTERVAL_MS));
    }
}
