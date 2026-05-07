#!/bin/bash
set -e

cd training
uv run python main.py >/dev/null 2>&1

cd ../inference/esp32
bash -lc 'source ~/.espressif/tools/activate_idf_v6.0.sh && python "$IDF_PATH/tools/idf.py" build && python "$IDF_PATH/tools/idf.py" flash monitor'
