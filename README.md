# Smart Turn Multimodal Benchmark

## Overview
The **Smart Turn Multimodal Benchmark** is a VNN-LIB 2.0 compliant benchmark designed to evaluate the robustness of a multimodal neural network. The network takes both audio (`.wav`) and video (`.MP4`) inputs and is provided in the ONNX format. This benchmark generates safety properties by applying perturbations to the inputs (`AUDIO_EPS = 0.05` and `VIDEO_EPS = 0.03`) to verify the network's behavior under noisy conditions.

## Prerequisites
- Python 3.x
- Required Python packages listed in `requirements.txt`

You can install the necessary dependencies using pip:
```bash
pip install -r requirements.txt
```

## Usage Instructions
To generate the VNN-LIB properties and the benchmark instances CSV file, run the `generate_properties.py` script with a random seed:

```bash
python generate_properties.py <random_seed>
```
**Example:**
```bash
python generate_properties.py 42
```

### Output
Running the generation script will:
1. Create up to 4 `.vnnlib` specification files in the `vnnlib/` directory based on the examples in the `examples/` folder.
2. Generate an `instances.csv` file in the root directory. This file maps the path of the ONNX model, the path of the generated `.vnnlib` file, and sets the verification timeout (100 seconds) for each instance.

## Directory Structure
```
.
├── README.md               # Documentation for the repository
├── requirements.txt        # Python dependency requirements
├── generate_properties.py  # Main script to generate properties and instances.csv
├── instances.csv           # CSV file containing benchmark instances (generated)
├── examples/               # Sample input files: Audio (.wav) and Video (.MP4)
├── onnx/                   # Directory containing the ONNX model (smart-turn-multimodal-cpu.onnx)
├── python_scripts/         # Helper scripts for building inputs, running inference, and creating specs
└── vnnlib/                 # Output directory for generated .vnnlib files
```
