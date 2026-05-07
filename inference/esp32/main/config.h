#pragma once

// ─────────────────────────────────────────────
//  User-configurable labels
// ─────────────────────────────────────────────
#define CLASS_LABEL_0   "Team Member 1"   // TFLite output index 0
#define CLASS_LABEL_1   "Team Member 2"   // TFLite output index 1
// Index 2 is implicitly "Unknown"

// ─────────────────────────────────────────────
//  Inference
// ─────────────────────────────────────────────
#define INFERENCE_INTERVAL_MS   500     // ms between frames
#define CONFIDENCE_THRESHOLD    0.55f   // below this → "Unknown"

// ─────────────────────────────────────────────
//  Model input dimensions (MobileNetV2 default)
// ─────────────────────────────────────────────
#define MODEL_INPUT_W   224
#define MODEL_INPUT_H   224
#define MODEL_INPUT_C   3               // RGB

// ─────────────────────────────────────────────
//  Camera capture resolution
//  Use a larger size so centre-crop has quality
// ─────────────────────────────────────────────
#define CAM_WIDTH    320
#define CAM_HEIGHT   240

// ─────────────────────────────────────────────
//  TFLite tensor arena (bytes)
//  MobileNetV2-224 needs more memory than 96x96
// ─────────────────────────────────────────────
#define TFLITE_ARENA_SIZE   (2048 * 1024)

// ─────────────────────────────────────────────
//  Seeed Studio XIAO ESP32-S3 Sense Pin Map
// ─────────────────────────────────────────────
#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    10
#define CAM_PIN_SIOD    40
#define CAM_PIN_SIOC    39
#define CAM_PIN_D7      48
#define CAM_PIN_D6      11
#define CAM_PIN_D5      12
#define CAM_PIN_D4      14
#define CAM_PIN_D3      16
#define CAM_PIN_D2      18
#define CAM_PIN_D1      17
#define CAM_PIN_D0      15
#define CAM_PIN_VSYNC   38
#define CAM_PIN_HREF    47
#define CAM_PIN_PCLK    13
