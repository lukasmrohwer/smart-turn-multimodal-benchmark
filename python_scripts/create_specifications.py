# creates a VNN-LIB 2.0 file (list of text lines) according to a fixed template:
# Arguments:
# - n_in: number of inputs to the classifier
# - n_out: number of output scores from the classifier
# - x_ref: reference input
# - eps_in: radius of the input L2 perturbation
# - class_out: predicted class we are checking the robustness of
def vnnlib_template_2(x1_ref, x2_ref, y_ref, eps_in):

    # assert len(x1_ref) == n_in
    assert eps_in >= 0.0
    # assert class_out >= 0 and class_out < n_out

    lines = []

    # intro comment
    # lines.append("; MNIST robustness to exact L2-ball perturbations:")
    # lines.append("; a toy VNN-COMP benchmark for the AAAI'26 tutorial.")
    # lines.append("; Author: Edoardo Manino")
    # lines.append("")

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

    # output constraints (negated!)
    #(assert (or
    #    (>= Y_0 Y_c)
    #    (>= Y_1 Y_c)
    #        ...
    #    (>= Y_N Y_c)
    #))
    lines.append("; Output Constraints")
    # lines.append("(assert (or")
    # for i in range(n_out):
    #     if i == class_out:
    #         continue
    #     lines.append("  (>= Y_" + str(i) + " Y_" + str(class_out) + ")")
    # lines.append("))")
    # lines.append("")

    return lines