# creates a VNN-LIB 2.0 file (list of text lines) according to a fixed template:
# Arguments:
# - n_in: number of inputs to the classifier
# - n_out: number of output scores from the classifier
# - x_ref: reference input
# - eps_in: radius of the input L-infinity perturbation
# - class_out: predicted class we are checking the robustness of
def vnnlib_template_2(x1_ref, x2_ref, y_ref, eps_in):

    assert eps_in >= 0.0

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
    lines.append("    (declare-input X1 float32 [1, 80, 800])")
    lines.append("    (declare-input X2 float32 [1, 3, 32, 112, 112])")
    lines.append("    (declare-output Y float32 [1, 1])")
    lines.append(")")
    lines.append("")

    # input constraints
    lines.append("; Input Constraints")
    for i in range(80):
        for j in range(800):
            lines.append(f"(assert (and (>= X1[0,{i},{j}] {x1_ref[0,i,j] - eps_in}) (<= X1[0,{i},{j}] {x1_ref[0,i,j] + eps_in})))")
    for i in range(3):
        for j in range(32):
            for k in range(112):
                for l in range(112):
                    lines.append(f"(assert (== X2[0,{i},{j},{k},{l}] {x2_ref[0,i,j,k,l]}))")
    lines.append("")

    # output constraints
    lines.append("; Output Constraints")
    if y_ref > 0.5:
        lines.append(f"(assert (<= Y[0] 0.5))")
    else:
        lines.append(f"(assert (> Y[0] 0.5))")
    lines.append("")

    return lines