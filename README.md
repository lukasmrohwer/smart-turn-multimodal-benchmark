### Benchmark Motivation
During an audio phone call, it can be difficult to tell when to interject because you are missing important visual cues that the other person is finished speaking. This problem is not unique to humans. Traditionally, conversational AI has focused on purely audio conversations with limited accuracy when detecting when the user has finished speaking. For example, a long silence could indicate that the user is now waiting for a response, or the user may simply be pausing to think. It is now more commonplace to speak to devices with both audio and video, meaning that AI can take advantage of important visual cues to detect end-of-turn patterns in real time.

The primary purpose of this benchmark is to test the capabilities of solvers to verify networks with multimodal inputs such as video and audio as supported in the new VNN-LIB 2.0 standard.

### Model Description
susuROBO's Smart Turn Multimodal achieves precisely this using a late fusion approach; processing the audio and video in separate streams and merging them towards the end in a way that the video branch effectively modulates the audio-only prediction. The output is the probability between 0 and 1 that the speaker's turn has ended, where a probability greater that 0.5 indicates that the subject has finished speaking.

The audio branch uses the Whisper Tiny encoder portion from the audio-only Smart Turn v3.2. It converts the audio into a floating-point Log-Mel Spectrogram 8 seconds long, compressed in 80 frequency bins and 800 time steps and uses a Transformer Encoder to produce a 384-dimensional vector that represents the audio.

The video branch extracts the last 32 frames of the video in 112x112 resolution in 3 colour channels and uses a pretrained 3D ResNet-18 to produce a 256-dimensional vector that can distinguish between features such as an open or closed mouth.

The dataset consists of entries of the form $((X_{audio}^{ref}, X_{video}^{ref}), Y)$.  $X_{audio}^{ref}$ and $X_{video}^{ref}$ are input tensors as described above with the shapes $(1, 80, 800)$ and $(1, 3, 32, 112, 112)$ respectively. $Y$ is the output tensor with shape $(1,1)$ describing the probabilty that the subject has finished speaking, scaled from 0 to 1 using a sigmoid activation function.

### Properties Description
As mentioned, the primary purpose of this benchmark is to test the capabilities of solvers to verify networks with multimodal inputs. As such, the only property that the benchmark tests is the adversarial robustness of the network by applying a small epsilon $L_\infty$ norm perturbation to the audio array while keeping the video array constant. 

Then the mathematical property that we seek to verify is of the form:

$$ \forall X_{audio}, X_{video} . \| X_{audio} - X_{audio}^{ref} \|_\infty \le \epsilon  \wedge X_{video} = X_{video}^{ref} \Rightarrow y > 0.5 \Leftrightarrow y^{(ref)} > 0.5 $$

The VNNLIB 2.0 property encodes the negation of the above property. Therefore, given the input constraints, the solver must attempt to satisfy the following output constraint where the output state (end-of-turn or not) is opposite to the reference output, and return any such adversarial input. If the solver returns UNSAT, the network is proven to be robust within the epsilon for that specific reference input.

The benchmark accepts a single numeric argument as the random seed. The generation of the benchmark is randomized by shuffling the order of the datasets.

### References
1. susuROBO, "Smart Turn Multimodal," susuROBO Blog. [Online]. Available: https://susurobo.jp/blog/smart_turn_multimodal.html
2. Daily, "Announcing Smart Turn v3, with CPU inference in just 12ms," Daily Blog, Sep. 11, 2025. [Online]. Available: https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/