#include "camera_pipeline.h"
#include "config.h"
#include "esp_log.h"

static const char *TAG = "CAM";

esp_err_t camera_pipeline_init(void)
{
    // ── Camera configuration ──────────────────────────────────────────────
    camera_config_t cfg = {};

    cfg.pin_pwdn    = CAM_PIN_PWDN;
    cfg.pin_reset   = CAM_PIN_RESET;
    cfg.pin_xclk    = CAM_PIN_XCLK;
    cfg.pin_sccb_sda = CAM_PIN_SIOD;
    cfg.pin_sccb_scl = CAM_PIN_SIOC;

    cfg.pin_d7  = CAM_PIN_D7;
    cfg.pin_d6  = CAM_PIN_D6;
    cfg.pin_d5  = CAM_PIN_D5;
    cfg.pin_d4  = CAM_PIN_D4;
    cfg.pin_d3  = CAM_PIN_D3;
    cfg.pin_d2  = CAM_PIN_D2;
    cfg.pin_d1  = CAM_PIN_D1;
    cfg.pin_d0  = CAM_PIN_D0;

    cfg.pin_vsync = CAM_PIN_VSYNC;
    cfg.pin_href  = CAM_PIN_HREF;
    cfg.pin_pclk  = CAM_PIN_PCLK;

    cfg.xclk_freq_hz = 10000000;          
    cfg.ledc_timer   = LEDC_TIMER_0;
    cfg.ledc_channel = LEDC_CHANNEL_0;

    cfg.pixel_format = PIXFORMAT_RGB565;
    cfg.jpeg_quality = 12;

    cfg.frame_size   = FRAMESIZE_QVGA;   // 320×240

    // Double-buffer 
    cfg.fb_count     = 2;
    cfg.fb_location  = CAMERA_FB_IN_PSRAM;  // ESP32-S3 has PSRAM
    cfg.grab_mode    = CAMERA_GRAB_LATEST;  // always get the newest frame

    esp_err_t err = esp_camera_init(&cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_camera_init failed: 0x%x", err);
        return err;
    }

    ESP_LOGI(TAG, "Camera ready (%dx%d RGB565)", CAM_WIDTH, CAM_HEIGHT);
    return ESP_OK;
}

camera_fb_t *camera_pipeline_capture(void)
{
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "esp_camera_fb_get() returned NULL");
    }
    return fb;
}

void camera_pipeline_return(camera_fb_t *fb)
{
    if (fb) {
        esp_camera_fb_return(fb);
    }
}
