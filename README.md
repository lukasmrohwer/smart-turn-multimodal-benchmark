# Smart Turn Multimodal Benchmark

This repository contains the **Smart Turn Multimodal Benchmark**, developed to serve as a verification benchmark for **VNN-COMP 2026**.

## Overview

The benchmark evaluates the robustness of a multimodal neural network (processing both audio and video inputs) against exact $L_\infty$ perturbations. It provides a set of verification instances formulated in the VNN-LIB format, challenging verifiers to prove bounds on the output probabilities given bounded input perturbations.

The model under verification is `smart-turn-multimodal-cpu.onnx`, a multimodal network that takes in:
- **Audio Features:** Mel spectrograms extracted via Whisper (shape: `[1, 80, 800]`)
- **Video Features:** Normalized video frames (shape: `[1, 3, 32, 112, 112]`)

## Verification Properties

The verification properties test the model's robustness to input perturbations:
- **Audio Perturbation ($\epsilon$):** 0.05
- **Video Perturbation ($\epsilon$):** 0.03
- **VNN-COMP Timeout:** 100 seconds per instance

Each instance asserts that for any perturbed input within the defined $\epsilon$-balls, the model's output prediction (a probability thresholded at 0.5) remains consistent with the reference output.

## Project Structure

- `generate_properties.py`: The main script used to generate VNN-LIB specifications and the `instances.csv` index file.
- `python_scripts/`: Contains utilities for building inputs and generating VNN-LIB assertions.
- `onnx/`: Contains the ONNX model to be verified.
- `vnnlib/`: The target directory for the generated `.vnnlib` specifications.
- `examples/`: Reference audio (`.wav`) and video (`.MP4`) pairs used to generate the benchmark instances.
- `instances.csv`: The index of verification targets mapping the ONNX model and VNN-LIB spec with the standard timeout, required for VNN-COMP execution.

## Requirements

The codebase requires the following dependencies for instance generation:
- `numpy`
- `onnxruntime>=1.17.0`
- `transformers`
- `librosa`
- `av`
- `Pillow`

## Usage

To generate the verification properties and the `instances.csv` index file, run the `generate_properties.py` script, providing a numeric random seed as an argument:

```bash
python generate_properties.py <random_seed>
```

Example:

```bash
python generate_properties.py 42
```

This will randomly select an example input pair, compute the reference outputs using the ONNX model, generate the `.vnnlib` specifications in the `vnnlib/` directory, and create the `instances.csv` file mapping the ONNX model to the generated specifications.

## Author & Acknowledgments

**Author:** Lukas Rohwer

This project contains code snippets from `susuROBO/Daily`, used under the BSD 2-Clause License. See `LICENSE-susuROBO_Daily.txt` for details.
