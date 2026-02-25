# creates a VNN-LIB 2.0 file (list of text lines) according to a fixed template:
# Arguments:
# - n_in: number of inputs to the classifier
# - n_out: number of output scores from the classifier
# - x_ref: reference input
# - eps_in: radius of the input L2 perturbation
# - class_out: predicted class we are checking the robustness of
def vnnlib_template_2(n_in, n_out, x_ref, eps_in, class_out):

    assert len(x_ref) == n_in
    assert eps_in >= 0.0
    assert class_out >= 0 and class_out < n_out

    lines = []

    # intro comment
    lines.append("; MNIST robustness to exact L2-ball perturbations:")
    lines.append("; a toy VNN-COMP benchmark for the AAAI'26 tutorial.")
    lines.append("; Author: Edoardo Manino")
    lines.append("")

    # tell the verifier to use VNN-LIB 2.0
    lines.append("(vnnlib-version <2.0>)")
    lines.append("")

    # neural network declaration
    lines.append("(declare-network f")
    lines.append("    (declare-input X real [" + str(n_in) + "])")
    lines.append("    (declare-output Y real [" + str(n_out) + "])")
    lines.append(")")

    # input L2 ball
    # (assert (<= (+(
    # 	* (- X[0] r_0) (- X[0] r_0)
    # 	* (- X[1] r_1) (- X[1] r_1)
    # 	...
    # 	* (- X[783] r_783) (- X[783] r_783)
    # )) eps*eps))
    lines.append("; Input Constraints")
    lines.append("(assert (<= (+(")
    for i in range(n_in):
        diff = "(- X[" + str(i) + "] " + str(x_ref[i]) + ")"
        lines.append("    * " + diff + " " + diff)
    lines.append(")) eps*eps))")
    lines.append("")

    # output constraints (negated!)
    #(assert (or
    #    (>= Y_0 Y_c)
    #    (>= Y_1 Y_c)
    #        ...
    #    (>= Y_N Y_c)
    #))
    lines.append("; Output Constraints")
    lines.append("(assert (or")
    for i in range(n_out):
        if i == class_out:
            continue
        lines.append("  (>= Y_" + str(i) + " Y_" + str(class_out) + ")")
    lines.append("))")
    lines.append("")

    return lines