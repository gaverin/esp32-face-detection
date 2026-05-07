/**
 * classifier.cpp
 *
 * Pre-processing pipeline:
 *   RGB565 (320×240) → centre-crop (240×240) → bilinear resize (96×96)
 *   → normalise to [-1, 1] (MobileNetV2 convention) → TFLite inference
 *
 * Post-processing:
 *   Softmax over 3 logits → apply confidence threshold → label
 */

#include "classifier.h"
#include "config.h"
#include "model_data.h"          // auto-generated: const uint8_t g_model_data[]

#include "esp_log.h"
#include "esp_heap_caps.h"

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
 * Extract one RGB pixel from an RGB565 frame buffer (little-endian packed).
 */
static inline void rgb565_pixel(const uint8_t *buf, int x, int y, int w,
                                 uint8_t &r, uint8_t &g, uint8_t &b)
{
    int idx = (y * w + x) * 2;
    uint16_t pixel = (uint16_t)(buf[idx] << 8) | buf[idx + 1];
    r = (pixel >> 11) & 0x1F;   r = (r << 3) | (r >> 2);
    g = (pixel >> 5)  & 0x3F;   g = (g << 2) | (g >> 4);
    b =  pixel        & 0x1F;   b = (b << 3) | (b >> 2);
}

/**
 * Bilinear resize + centre-crop from a source RGB565 buffer to a float
 * destination buffer normalised to [-1, 1].
 *
 * Steps:
 *  1. Compute a square crop region centred on the source frame.
 *  2. Map each output pixel back to the crop region using bilinear sampling.
 *  3. Normalise: out = (pixel / 127.5) - 1.0
 */
static void preprocess(const uint8_t *src_rgb565,
                        int src_w, int src_h,
                        float *dst,
                        int dst_w, int dst_h)
{
    // Centre-crop dimensions
    int crop_size = (src_w < src_h) ? src_w : src_h;
    int crop_x0   = (src_w - crop_size) / 2;
    int crop_y0   = (src_h - crop_size) / 2;

    float scale_x = (float)crop_size / dst_w;
    float scale_y = (float)crop_size / dst_h;

    for (int dy = 0; dy < dst_h; dy++) {
        for (int dx = 0; dx < dst_w; dx++) {

            // Map to float source coordinates within the crop
            float sx = crop_x0 + dx * scale_x;
            float sy = crop_y0 + dy * scale_y;

            int x0 = (int)sx;
            int y0 = (int)sy;
            int x1 = x0 + 1; if (x1 >= src_w) x1 = src_w - 1;
            int y1 = y0 + 1; if (y1 >= src_h) y1 = src_h - 1;

            float wx = sx - x0;
            float wy = sy - y0;

            uint8_t r00, g00, b00, r01, g01, b01;
            uint8_t r10, g10, b10, r11, g11, b11;

            rgb565_pixel(src_rgb565, x0, y0, src_w, r00, g00, b00);
            rgb565_pixel(src_rgb565, x1, y0, src_w, r01, g01, b01);
            rgb565_pixel(src_rgb565, x0, y1, src_w, r10, g10, b10);
            rgb565_pixel(src_rgb565, x1, y1, src_w, r11, g11, b11);

            auto lerp = [](float a, float b, float t) { return a + t * (b - a); };

            float r = lerp(lerp(r00, r01, wx), lerp(r10, r11, wx), wy);
            float g = lerp(lerp(g00, g01, wx), lerp(g10, g11, wx), wy);
            float b = lerp(lerp(b00, b01, wx), lerp(b10, b11, wx), wy);

            // Normalise to [-1, 1]
            int base = (dy * dst_w + dx) * 3;
            dst[base + 0] = (r / 127.5f) - 1.0f;
            dst[base + 1] = (g / 127.5f) - 1.0f;
            dst[base + 2] = (b / 127.5f) - 1.0f;
        }
    }
}

/**
 * Softmax over a float array in-place.
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

// ─────────────────────────────────────────────────────────────────────────────
//  Public API
// ─────────────────────────────────────────────────────────────────────────────

esp_err_t classifier_init(void)
{
    // ── Allocate tensor arena in PSRAM ────────────────────────────────────
    s_tensor_arena = (uint8_t *)heap_caps_malloc(TFLITE_ARENA_SIZE,
                                                  MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_tensor_arena) {
        ESP_LOGE(TAG, "Failed to allocate %u byte tensor arena in PSRAM",
                 TFLITE_ARENA_SIZE);
        return ESP_ERR_NO_MEM;
    }

    // ── Load model ────────────────────────────────────────────────────────
    s_model = tflite::GetModel(g_model_data);
    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema mismatch: got %lu, expected %d",
                 s_model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_FAIL;
    }

    // ── Build op resolver (MobileNetV2 op set) ────────────────────────────
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

    // ── Create interpreter ────────────────────────────────────────────────
    s_interpreter = new tflite::MicroInterpreter(
        s_model, *s_resolver, s_tensor_arena, TFLITE_ARENA_SIZE);

    if (s_interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return ESP_FAIL;
    }

    s_input  = s_interpreter->input(0);
    s_output = s_interpreter->output(0);

    // ── Sanity checks ─────────────────────────────────────────────────────
    ESP_LOGI(TAG, "Input  tensor: [%d, %d, %d, %d] type=%d",
             s_input->dims->data[0], s_input->dims->data[1],
             s_input->dims->data[2], s_input->dims->data[3],
             s_input->type);
    ESP_LOGI(TAG, "Output tensor: [%d, %d] type=%d",
             s_output->dims->data[0], s_output->dims->data[1],
             s_output->type);
    ESP_LOGI(TAG, "Arena used: %zu / %u bytes",
             s_interpreter->arena_used_bytes(), TFLITE_ARENA_SIZE);

    if (s_output->dims->data[1] < 3) {
        ESP_LOGE(TAG, "Model must have ≥3 output classes (got %d)",
                 s_output->dims->data[1]);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Classifier ready");
    return ESP_OK;
}

esp_err_t classifier_run(const camera_fb_t *fb, ClassifierResult *result)
{
    if (!fb || !result) return ESP_ERR_INVALID_ARG;
    if (!s_interpreter)  return ESP_ERR_INVALID_STATE;

    // ── Pre-process: resize + normalise into input tensor ─────────────────
    if (s_input->type == kTfLiteFloat32) {
        preprocess(fb->buf, CAM_WIDTH, CAM_HEIGHT,
                   s_input->data.f,
                   MODEL_INPUT_W, MODEL_INPUT_H);
    } else if (s_input->type == kTfLiteInt8) {
        // Quantised model: compute float first into PSRAM buffer, then quantise.
        // Using heap_caps_malloc avoids placing 27KB static array in scarce DRAM.
        static float *tmp_buf = nullptr;
        if (!tmp_buf) {
            tmp_buf = (float *)heap_caps_malloc(
                MODEL_INPUT_W * MODEL_INPUT_H * MODEL_INPUT_C * sizeof(float),
                MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            if (!tmp_buf) {
                ESP_LOGE(TAG, "Failed to alloc preprocess buffer in PSRAM");
                return ESP_ERR_NO_MEM;
            }
        }
        preprocess(fb->buf, CAM_WIDTH, CAM_HEIGHT, tmp_buf,
                   MODEL_INPUT_W, MODEL_INPUT_H);
        float scale     = s_input->params.scale;
        int   zero_pt   = s_input->params.zero_point;
        int   n_floats  = MODEL_INPUT_W * MODEL_INPUT_H * MODEL_INPUT_C;
        for (int i = 0; i < n_floats; i++) {
            int q = (int)(tmp_buf[i] / scale) + zero_pt;
            if (q < -128) q = -128;
            if (q >  127) q =  127;
            s_input->data.int8[i] = (int8_t)q;
        }
    } else {
        ESP_LOGE(TAG, "Unsupported input tensor type: %d", s_input->type);
        return ESP_FAIL;
    }

    // ── Invoke ────────────────────────────────────────────────────────────
    if (s_interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() failed");
        return ESP_FAIL;
    }

    // ── Extract raw logits / probabilities ────────────────────────────────
    float logits[3] = {0.f, 0.f, 0.f};

    if (s_output->type == kTfLiteFloat32) {
        for (int i = 0; i < 3; i++) logits[i] = s_output->data.f[i];
    } else if (s_output->type == kTfLiteInt8) {
        float scale   = s_output->params.scale;
        int   zero_pt = s_output->params.zero_point;
        for (int i = 0; i < 3; i++)
            logits[i] = (s_output->data.int8[i] - zero_pt) * scale;
    }

    // ── Softmax (if model outputs raw logits rather than probabilities) ────
    softmax(logits, 3);

    result->prob[0] = logits[0];   // Team Member 1
    result->prob[1] = logits[1];   // Team Member 2
    result->prob[2] = logits[2];   // Unknown (background / third class)

    // ── Determine winning class ───────────────────────────────────────────
    int   best_idx  = 0;
    float best_prob = result->prob[0];
    for (int i = 1; i < 3; i++) {
        if (result->prob[i] > best_prob) {
            best_prob = result->prob[i];
            best_idx  = i;
        }
    }

    // Apply confidence threshold: uncertain → Unknown
    if (best_prob < CONFIDENCE_THRESHOLD) {
        result->class_idx = 2;
        snprintf(result->label, sizeof(result->label),
                 "Unknown (%.1f%%)", result->prob[2] * 100.f);
    } else {
        result->class_idx = best_idx;
        const char *name = (best_idx == 0) ? CLASS_LABEL_0 : CLASS_LABEL_1;
        snprintf(result->label, sizeof(result->label),
                 "%s (%.1f%%)", name, best_prob * 100.f);
    }

    return ESP_OK;
}
