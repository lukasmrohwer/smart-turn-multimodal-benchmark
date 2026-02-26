from python_scripts.create_specifications import vnnlib_template_2
from python_scripts.build_inputs import build_audio_input, build_video_input
from python_scripts.inference_output import inference

VIDEO_FILE = "examples/videoplayback.mp4"
AUDIO_FILE = "examples/videoplayback.mp3"

x1_ref = build_audio_input(AUDIO_FILE)
x2_ref = build_video_input(VIDEO_FILE)

y_ref = inference(x1_ref, x2_ref)

lines = vnnlib_template_2(x1_ref, x2_ref, y_ref, 0.01)

with open("test.txt", "w") as f:
    f.writelines(line + "\n" for line in lines)