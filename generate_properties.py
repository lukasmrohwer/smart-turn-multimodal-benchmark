# create VNN-LIB 2.0 files given the following:
EPS = 0.05              # size of the input pertubation
VNN_COMP_TIMEOUT = 100  # per-instance verification timeout
ONNX_MODEL_PATH = "onnx/smart-turn-multimodal-cpu.onnx"

import csv
from python_scripts.create_specifications import vnnlib_template_2
from python_scripts.build_inputs import build_audio_input, build_video_input
from python_scripts.inference_output import inference

inputs = {
    ("examples/videoplayback.mp3", "examples/videoplayback.mp4")
}

i = 0
instance_data = []
for x1, x2 in inputs:

    x1_ref = build_audio_input(x1)
    x2_ref = build_video_input(x2)

    y_ref = inference(x1_ref, x2_ref, ONNX_MODEL_PATH)

    lines = vnnlib_template_2(x1_ref, x2_ref, y_ref, 0.01)

    vnnlib_filename = "./vnnlib/instance_" + str(i) + ".vnnlib2"
    with open(vnnlib_filename, "w") as f:
        f.writelines(line + "\n" for line in lines)

    instance = [ONNX_MODEL_PATH, vnnlib_filename, VNN_COMP_TIMEOUT]
    instance_data.append(instance)

    i += 1

# save the ONNX/VNN-LIB instance pairs in the required CSV
with open(f"instances.csv", 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(instance_data)