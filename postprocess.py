"""
#################lstm postprocess########################
"""
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="atae-lstm postprocess")
parser.add_argument("--label_dir", type=str, required=True, help="label directory")
parser.add_argument("--result_dir", type=str, required=True, help="result directory")
args = parser.parse_args()

if __name__ == '__main__':
    file_names = []
    for root, dirs, files in os.walk(args.result_dir):
        file_names = files

    file_num = len(file_names)
    correct = 0
    for f in file_names:
        label_path = os.path.join(args.label_dir, f)
        result_path = os.path.join(args.result_dir, f)

        label_numpy = np.fromfile(label_path, np.float32).reshape([1, 3])
        polarity_label = np.argmax(label_numpy)
        result_numpy = np.fromfile(result_path, np.float32).reshape([1, 3])
        polarity_result = np.argmax(result_numpy)
        if polarity_result == polarity_label:
            correct += 1

    acc = correct / float(file_num)
    print("\n---accuracy:", acc, "---\n")
