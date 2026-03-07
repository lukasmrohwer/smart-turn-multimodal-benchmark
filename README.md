### Smart Turn Multimodal Benchmark

[Smart Turn Multimodal](https://huggingface.co/susuROBO/smart-turn-multimodal) is a multimodal model that takes audio and video inputs to determine whether the subject has finised speaking. This model extends on existing tools using exclusively audio, missing out on important visual cues.

#### Model Architecture

 - ~20M Parameters
 - Audio branch - Whisper Tiny encoder on a 8 second context window.
 - Video branch: Processing last 32 frames of the video.

The video input tensor will take the last 32 frames of the video in 112x112 resolution in 3 colour channels. The audio input tensor is a floating-point Log-Mel Spectrogram 8 seconds long, compressed in 80 frequency bins and 800 time steps. The model outputs a probability between 0 and 1, where a probability greater that 0.5 indicates that the subject has finished speaking.

#### Benchmark Information

This benchmark was designed to test the capabilities of solvers to verify networks with multiple, multimodal inputs as supported in the new VNN-LIB 2.0 standard. The benchmark simply allows an epsilon pertubation to the values of the audio tensor while maintaining the video tensor as a fixed reference, and asserts that there is no assignment such that the classification of the output changes.