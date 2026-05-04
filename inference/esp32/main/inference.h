bool classifier_init();
int8_t* classifier_put_image(const uint8_t *image_buffer);
bool classifier_predict(float *prediction);