import csv
from python_scripts.create_specifications import vnnlib_template_2
from python_scripts.build_inputs import build_audio_input, build_video_input
from python_scripts.inference_output import inference
import random
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_properties.py <random_seed>")
        sys.exit(1)
        
    try:
        seed = int(sys.argv[1])
    except ValueError:
        print("Error: The random seed must be a numeric integer.")
        sys.exit(1)

    random.seed(seed)

    # create VNN-LIB 2.0 files given the following:
    AUDIO_EPS = 0.05              # size of the input pertubation
    VIDEO_EPS = 0.03              # size of the input pertubation
    VNN_COMP_TIMEOUT = 100  # per-instance verification timeout
    ONNX_MODEL_PATH = "onnx/smart-turn-multimodal-cpu.onnx"
    num_instances = 4 # max 4

    inputs = [
        ("examples/1.wav", "examples/1.MP4"),
        ("examples/2.wav", "examples/2.MP4"),
        ("examples/3.wav", "examples/3.MP4"),
        ("examples/4.wav", "examples/4.MP4")
    ]
    random.shuffle(inputs)

    i = 0
    instance_data = []
    for x1, x2 in inputs[:num_instances]:

        x1_ref = build_audio_input(x1)
        x2_ref = build_video_input(x2)

        y_ref = inference(x1_ref, x2_ref, ONNX_MODEL_PATH)

        lines = vnnlib_template_2(x1_ref, x2_ref, y_ref, AUDIO_EPS, VIDEO_EPS)

        vnnlib_filename = "vnnlib/instance_" + str(i) + ".vnnlib"
        with open(vnnlib_filename, "w") as f:
            f.writelines(line + "\n" for line in lines)

        instance = [ONNX_MODEL_PATH, vnnlib_filename, VNN_COMP_TIMEOUT]
        instance_data.append(instance)

        i += 1

    # save the ONNX/VNN-LIB instance pairs in the required CSV
    with open(f"instances.csv", 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(instance_data)

if __name__ == "__main__":
    main()