import onnxruntime as ort

ONNX_MODEL_PATH = "onnx/smart-turn-multimodal-cpu.onnx"

def build_session(onnx_path):
    """Build ONNX inference session with optimized settings."""
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so)

def inference(input_features, pixel_values):
    session = build_session(ONNX_MODEL_PATH)

    outputs = session.run(
        None, {"input_features": input_features, "pixel_values": pixel_values}
    )

    probability = outputs[0][0].item()

    return probability