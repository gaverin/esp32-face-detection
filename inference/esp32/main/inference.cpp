#include "esp_log.h"
#include "model.h"

// Include TFLM
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

// Tensor arena size, found by trial and error
#define TENSOR_ARENA_SIZE (600 * 1024)

// Static variables
static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
//alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
static TfLiteTensor *input = nullptr;
static TfLiteTensor *output = nullptr;
static const char *TAG_INF = "Inference";


/**
 * @brief Initialize the TFLite Micro interpreter with the model.
 * 
 * @return True if initialization was successful, false otherwise.
 */
bool classifier_init()
{
    // Load TFlite model
    model = tflite::GetModel(model_binary);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        ESP_LOGE(TAG_INF, "Model schema mismatch!");
        return false;
    }
    // Create an interpreter
    static tflite::MicroMutableOpResolver<12> micro_op_resolver;

    // 1. Convolutional Ops    
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    // 2. Element-wise Math Ops (for Inverted Residual skips)
    micro_op_resolver.AddAdd();
    // 3. Activations
    micro_op_resolver.AddRelu6();
    // 4. Pooling / Reduction Ops
    micro_op_resolver.AddMean(); 
    // 5. Dimension & Shaping Ops
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddPad();
    // 6. Classification Head Ops
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    // 7. Quantization Ops (Crucial for INT8 TFLite models)
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();
    // 8. Multiplication
    micro_op_resolver.AddMul();

    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    // Allocate memory for input and output tensors
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        ESP_LOGE(TAG_INF, "Failed to allocate tensors!");
        return false;
    }

    // Get pointers for input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Print input and output tensor types and dimensions
    ESP_LOGI(TAG_INF, "Input tensor type: %s, shape: %d, %d, %d",
             TfLiteTypeGetName(input->type), input->dims->data[0], input->dims->data[1], input->dims->data[2]);
    ESP_LOGI(TAG_INF, "Output tensor type: %s, shape: %d, %d",
             TfLiteTypeGetName(output->type), output->dims->data[0], output->dims->data[1]);
    return true;
}

/*
    Feature matrix size is IMG_WIDTH * IMG_HEIGHT
*/

/**
 *  @brief Quantize the feature matrix from float to int8 and push it into the interpreter.
 *  @param features Pointer to the input feature matrix in float format.
 *  @return Pointer to the output feature matrix in int8 format.
 */
int8_t* classifier_put_image(const uint8_t *image_buffer)
{
    int8_t* input_tensor = input->data.int8;
    float scale = input->params.scale;
    int32_t zero_point = input->params.zero_point;

    for (int i = 0; i < (IMG_WIDTH * IMG_HEIGHT); ++i)
    {
        // 1. Extract RGB565 and expand to 0-255
        uint16_t pixel = (image_buffer[i * 2] << 8) | image_buffer[i * 2 + 1];
        float r = (float)((pixel >> 11) & 0x1F) * (255.0f / 31.0f);
        float g = (float)((pixel >> 5) & 0x3F) * (255.0f / 63.0f);
        float b = (float)(pixel & 0x1F) * (255.0f / 31.0f);

        // 2. Apply MobileNetV2 Preprocessing: Map [0, 255] to [-1, 1]
        // Formula: (pixel / 127.5) - 1.0
        float channels[3] = {
            (r / 127.5f) - 1.0f,
            (g / 127.5f) - 1.0f,
            (b / 127.5f) - 1.0f
        };

        // 3. Quantize the PREPROCESSED float to int8
        for (int c = 0; c < 3; ++c)
        {
            float val_quant = roundf(channels[c] / scale) + zero_point;
            
            // Clip to int8 limits
            if (val_quant > 127.0f) val_quant = 127.0f;
            if (val_quant < -128.0f) val_quant = -128.0f;

            input_tensor[i * 3 + c] = static_cast<int8_t>(val_quant);
        }
    }
    return input_tensor;
}

/**
 * @brief Run inference on the model, obtain the prediction and dequantize to float.
 * @param prediction Pointer to store the prediction result. Expected to be of size NUM_CLASSES.
 * @return True if inference was successful, false otherwise.
 */
bool classifier_predict(float *prediction)
{
    // Run inference
    if (interpreter->Invoke() != kTfLiteOk)
    {
        return false;
    }

    // Dequantize the output from int8 to float
    for (size_t i = 0; i < NUM_CLASSES; ++i)
    {
        prediction[i] = (static_cast<float>(output->data.int8[i]) - output->params.zero_point) * output->params.scale;
    }

    return true;
}