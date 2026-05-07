/**
 * classifier.cpp
 *
 * 
 * Post-processing:
 *   Softmax over 3 logits → apply confidence threshold → label
 */

#include "classifier.h"
#include "config.h"
#include "model_data.h"          

#include "esp_log.h"
#include "esp_heap_caps.h"
#include "img_converters.h"

// TensorFlow Lite Micro headers (ESP-IDF component: esp-tflite-micro)
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"

#include <cmath>
#include <cstring>
#include <cstdio>

static const char *TAG = "CLASSIFIER";

// ── TFLite globals ────────────────────────────────────────────────────────────
static const tflite::Model           *s_model       = nullptr;
static tflite::MicroInterpreter      *s_interpreter = nullptr;
static TfLiteTensor                  *s_input        = nullptr;
static TfLiteTensor                  *s_output       = nullptr;

// Tensor arena allocated from PSRAM to save SRAM
static uint8_t *s_tensor_arena = nullptr;

// ── Op resolver – include only ops used by MobileNetV2 ───────────────────────
using Resolver = tflite::MicroMutableOpResolver<15>;
static Resolver *s_resolver = nullptr;

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Direct centre-crop from RGB565 to Int8 MobileNetV2 input.
 */
static void preprocess(const camera_fb_t *fb, int src_w, int src_h,
                       int8_t *dst, int dst_w, int dst_h,
                       float scale, int zero_point)
{
    int crop_x0 = (src_w - dst_w) / 2;
    int crop_y0 = (src_h - dst_h) / 2;

    uint8_t *rgb888 = (uint8_t*)heap_caps_malloc(src_w * src_h * 3, MALLOC_CAP_SPIRAM);
    if (!rgb888) return;
    fmt2rgb888(fb->buf, fb->len, fb->format, rgb888);

    for (int dy = 0; dy < dst_h; dy++) {
        for (int dx = 0; dx < dst_w; dx++) {
            int src_x = crop_x0 + dx;
            int src_y = crop_y0 + dy;
            
            uint8_t r = 0, g = 0, b = 0;
            if (src_x >= 0 && src_x < src_w && src_y >= 0 && src_y < src_h) {
                int idx = (src_y * src_w + src_x) * 3;
                r = rgb888[idx + 0];
                g = rgb888[idx + 1];
                b = rgb888[idx + 2];
            }

            auto quantize = [&](uint8_t c) -> int8_t {
                int q = (int)lroundf(((float)c / scale) + zero_point);
                return (q < -128) ? -128 : (q > 127) ? 127 : q;
            };

            int base = (dy * dst_w + dx) * 3;
            dst[base + 0] = quantize(r);
            dst[base + 1] = quantize(g);
            dst[base + 2] = quantize(b);
        }
    }
    heap_caps_free(rgb888);
}

/**
 * Softmax for model output.
 */
static void softmax(float *arr, int n)
{
    float max_val = arr[0];
    for (int i = 1; i < n; i++) if (arr[i] > max_val) max_val = arr[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        arr[i] = expf(arr[i] - max_val);
        sum += arr[i];
    }
    for (int i = 0; i < n; i++) arr[i] /= sum;
}


esp_err_t classifier_init(void)
{
    s_tensor_arena = (uint8_t *)heap_caps_malloc(TFLITE_ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_tensor_arena) return ESP_ERR_NO_MEM;

    s_model = tflite::GetModel(g_model_data);
    
    s_resolver = new Resolver();
    s_resolver->AddConv2D();
    s_resolver->AddDepthwiseConv2D();
    s_resolver->AddAdd();
    s_resolver->AddRelu6();
    s_resolver->AddReshape();
    s_resolver->AddAveragePool2D();
    s_resolver->AddFullyConnected();
    s_resolver->AddSoftmax();
    s_resolver->AddQuantize();
    s_resolver->AddDequantize();
    s_resolver->AddPad();
    s_resolver->AddMul();
    s_resolver->AddSub();
    s_resolver->AddMean();

    s_interpreter = new tflite::MicroInterpreter(s_model, *s_resolver, s_tensor_arena, TFLITE_ARENA_SIZE);
    if (s_interpreter->AllocateTensors() != kTfLiteOk) return ESP_FAIL;

    s_input  = s_interpreter->input(0);
    s_output = s_interpreter->output(0);
    
    ESP_LOGI(TAG, "Classifier ready");
    return ESP_OK;
}

esp_err_t classifier_run(const camera_fb_t *fb, ClassifierResult *result)
{
    if (!fb || !result || !s_interpreter) return ESP_FAIL;

    // 1. Preprocess: Decode, Crop, Normalise & Quantise directly into model input
    preprocess(fb, CAM_WIDTH, CAM_HEIGHT,
               s_input->data.int8, MODEL_INPUT_W, MODEL_INPUT_H,
               s_input->params.scale, s_input->params.zero_point);

    // 2. Invoke Model
    if (s_interpreter->Invoke() != kTfLiteOk) return ESP_FAIL;

    // 3. De-quantize output to floats
    float logits[3];
    for (int i = 0; i < 3; i++) {
        logits[i] = (s_output->data.int8[i] - s_output->params.zero_point) * s_output->params.scale;
    }
    
    // 4. Softmax into Probabilities
    softmax(logits, 3);
    for (int i = 0; i < 3; i++) result->prob[i] = logits[i];

    // 5. Select Best Class
    int best_idx = 0;
    for (int i = 1; i < 3; i++) {
        if (result->prob[i] > result->prob[best_idx]) best_idx = i;
    }

    // 6. Apply Confidence Threshold
    if (result->prob[best_idx] < CONFIDENCE_THRESHOLD) {
        best_idx = 2; // "Unknown"
    }

    // 7. Format Result
    result->class_idx = best_idx;
    const char *name = (best_idx == 0) ? CLASS_LABEL_0 :
                       (best_idx == 1) ? CLASS_LABEL_1 : "Unknown";
    snprintf(result->label, sizeof(result->label), "%s (%.1f%%)", name, result->prob[best_idx] * 100.f);

    return ESP_OK;
}
