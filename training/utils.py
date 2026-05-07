import os
from pathlib import Path
from sklearn.metrics import confusion_matrix
import numpy as np

def write_model_h_file(path: str, defines: dict, declarations: list[str]):
    # Ensure that the folder exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Write header file
    with open(path, "w") as h_file:
        h_file.write("#ifndef MODEL_H\n")
        h_file.write("#define MODEL_H\n")
        h_file.write("\n")
        for key, value in defines.items():
            h_file.write(f'#define {key} {value}\n')
        h_file.write("\n")
        for declaration in declarations:
            h_file.write(f'{declaration}\n')
        h_file.write("\n")
        h_file.write("extern const unsigned char model_binary[];\n")
        h_file.write("\n")
        h_file.write("#endif\n")


def write_model_c_file(path: str, tflite_model):
    # Ensure that the folder exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Write source file
    with open(path, "w") as c_file:
        c_file.write("const unsigned char model_binary[] = {\n")
        for i, byte in enumerate(tflite_model):
            c_file.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                c_file.write("\n")
        c_file.write("\n};\n")


def write_files(tflite_path: Path, cc_path: Path, h_path: Path) -> None:
    data = tflite_path.read_bytes()
    h_path.write_text(
        "#pragma once\n"
        "#include <stdint.h>\n\n"
        "extern const uint8_t g_model_data[];\n"
        "extern const uint32_t g_model_data_len;\n",
        encoding="utf-8",
    )
    lines = []
    for i in range(0, len(data), 12):
        chunk = data[i:i + 12]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    cc_path.write_text(
        "// Auto-generated. Do not edit by hand.\n"
        "#include \"model_data.h\"\n"
        "#include <stdint.h>\n\n"
        "const uint8_t g_model_data[] = {\n"
        + "\n".join(lines)
        + "\n};\n\n"
        + f"const uint32_t g_model_data_len = {len(data)};\n",
        encoding="utf-8",
    )



def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_labels: list[str]):
    # Count predictions in confusion matrix
    num_classes = len(class_labels)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    # Determine column width
    col_width = max(len(label) for label in class_labels) + 1
    num_digits = len(str(np.max(cm)))
    if (num_digits + 1) > col_width:
        col_width = num_digits + 1
    
    # Print confusion matrix header
    print('Confusion matrix (predicted as columns, actual as rows):')
    print('--------------------------------------------------------')

    # Print class labels for X axis
    print(' ' * col_width, end='')
    for label in class_labels:
        print(f'{label:>{col_width}}', end='')
    print()

    # Print each row of the confusion matrix
    for i in range(num_classes):
        print(f'{class_labels[i]:>{col_width}}', end='')
        for j in range(num_classes):
            print(f'{cm[i, j]:>{col_width}}', end='')
        print()
