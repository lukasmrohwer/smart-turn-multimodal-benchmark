# creates a VNN-LIB 2.0 file (list of text lines) according to a fixed template:
# Arguments:
# - x1_ref: reference audio input
# - x2_ref: reference video input
# - y_ref: reference output
# - audio_eps: radius of the input L-infinity perturbation for audio
# - video_eps: radius of the input L-infinity perturbation for video
def vnnlib_template_2(x1_ref, x2_ref, y_ref, audio_eps, video_eps):

    lines = []

    # intro comment
    lines.append("; Model robustness to exact L-infinity perturbations:")
    lines.append("; a VNN-COMP benchmark with multimodal inputs.")
    lines.append("; Author: Lukas Rohwer")
    lines.append("")

    # tell the verifier to use VNN-LIB 2.0
    lines.append("(vnnlib-version <2.0>)")
    lines.append("")

    # neural network declaration
    lines.append("(declare-network f")
    lines.append("    (declare-input X1 real [1, 80, 800])")
    lines.append("    (declare-input X2 real [1, 3, 32, 112, 112])")
    lines.append("    (declare-output Y real [1, 1])")
    lines.append(")")
    lines.append("")

    # input constraints
    lines.append("; Input Constraints")
    for i in range(80):
        for j in range(800):
            lines.append(f"(assert (and (>= X1[0,{i},{j}] {x1_ref[0,i,j] - audio_eps}) (<= X1[0,{i},{j}] {x1_ref[0,i,j] + audio_eps})))")
    for i in range(3):
        for j in range(32):
            for k in range(112):
                for l in range(112):
                    lines.append(f"(assert (and (>= X2[0,{i},{j},{k},{l}] {x2_ref[0,i,j,k,l] - video_eps}) (<= X2[0,{i},{j},{k},{l}] {x2_ref[0,i,j,k,l] + video_eps})))")
    lines.append("")

    # output constraints
    lines.append("; Output Constraints")
    if y_ref > 0.5:
        lines.append(f"(assert (<= Y[0] 0.5))")
    else:
        lines.append(f"(assert (> Y[0] 0.5))")
    lines.append("")

    return lines