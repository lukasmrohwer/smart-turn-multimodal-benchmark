# creates a VNN-LIB 2.0 file (list of text lines) according to a fixed template:
# Arguments:
# - n_in: number of inputs to the classifier
# - n_out: number of output scores from the classifier
# - x_ref: reference input
# - eps_in: radius of the input L2 perturbation
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

    # input constraints
    lines.append("; Input Constraints")
    for i in range(80):
        for j in range(800):
            lines.append(f"(assert (and (>= X1[0,{i},{j}] {x1_ref[0,i,j] - eps_in}) (<= X1[0,{i},{j}] {x1_ref[0,i,j] + eps_in})))")
    lines.append("")

    # output constraints
    lines.append("; Output Constraints")
    lines.append(f"(assert (or (< Y[0] {y_ref - eps_in}) (> Y[0] {y_ref + eps_in})))")
    lines.append("")

    return lines