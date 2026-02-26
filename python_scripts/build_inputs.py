import numpy as np
import av
import os
from collections import deque

def build_video_input(video_path):
    # Video constants
    VIDEO_FRAMES = 32
    VIDEO_SIZE = 112
    VIDEO_MEAN = np.array([0.43216, 0.394666, 0.37645])
    VIDEO_STD = np.array([0.22803, 0.22145, 0.216989])
    
    def process_video_frames(video_path):
        """
        Decodes, resizes, and normalizes video frames.

        Args:
            video_path: Path to video file, or None for audio-only inference

        Returns:
            numpy array of shape (1, 3, 32, 112, 112)
        """
        # Handle missing video -> zero tensor
        if not video_path or not os.path.exists(video_path):
            return np.zeros(
                (1, 3, VIDEO_FRAMES, VIDEO_SIZE, VIDEO_SIZE), dtype=np.float32
            )

        try:
            # Use context manager to ensure memory is freed immediately
            with av.open(video_path) as container:
                # Use deque to efficiently keep only the last VIDEO_FRAMES frames
                frame_buffer = deque(maxlen=VIDEO_FRAMES)

                # Stream, resize, and store
                for frame in container.decode(video=0):
                    # Resize to 112x112 immediately (memory saving)
                    img = frame.to_image().resize((VIDEO_SIZE, VIDEO_SIZE))
                    img_np = np.array(img, dtype=np.float32) / 255.0
                    frame_buffer.append(img_np)

                frames = list(frame_buffer)

            # Check if we have enough frames
            if not frames:
                return np.zeros(
                    (1, 3, VIDEO_FRAMES, VIDEO_SIZE, VIDEO_SIZE), dtype=np.float32
                )

            # Pad with last frame if fewer than VIDEO_FRAMES
            while len(frames) < VIDEO_FRAMES:
                frames.append(frames[-1])

            # Stack frames: (32, 112, 112, 3)
            video = np.stack(frames)

            # Permute to (3, 32, 112, 112)
            video = video.transpose(3, 0, 1, 2)

            # Normalize with correct broadcasting shape
            mean = VIDEO_MEAN.reshape(3, 1, 1, 1).astype(np.float32)
            std = VIDEO_STD.reshape(3, 1, 1, 1).astype(np.float32)
            video = (video - mean) / std

            # Add batch dimension -> (1, 3, 32, 112, 112)
            return np.expand_dims(video, axis=0).astype(np.float32)

        except Exception as e:
            print(f"Video processing error: {e}")
            return np.zeros(
                (1, 3, VIDEO_FRAMES, VIDEO_SIZE, VIDEO_SIZE), dtype=np.float32
            )
    
    pixel_values = process_video_frames(video_path)  # (1, 3, 32, 112, 112)

    return pixel_values

from transformers import WhisperFeatureExtractor
import librosa

def build_audio_input(audio_path):
    
    SAMPLING_RATE = 16000
    AUDIO_SECONDS = 8

    def truncate_audio_to_last_n_seconds(audio_array, n_seconds=8, sample_rate=16000):
        """Truncate audio to last n seconds or pad with zeros to meet n seconds."""
        max_samples = n_seconds * sample_rate
        if len(audio_array) > max_samples:
            return audio_array[-max_samples:]
        elif len(audio_array) < max_samples:
            # Pad with zeros at the beginning
            padding = max_samples - len(audio_array)
            return np.pad(audio_array, (padding, 0), mode='constant', constant_values=0)
        return audio_array
    
    feature_extractor = WhisperFeatureExtractor(chunk_length=AUDIO_SECONDS)

    audio_array, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
    audio_array = audio_array.astype(np.float32)

    # Truncate to last 8 seconds (keeping the end) or pad to 8 seconds
    audio_array = truncate_audio_to_last_n_seconds(
        audio_array, n_seconds=AUDIO_SECONDS
    )

    # Process audio using Whisper's feature extractor
    inputs = feature_extractor(
        audio_array,
        sampling_rate=SAMPLING_RATE,
        return_tensors="np",
        padding="max_length",
        max_length=AUDIO_SECONDS * SAMPLING_RATE,
        truncation=True,
        do_normalize=True,
    )

    # Extract features and ensure correct shape for ONNX
    input_features = inputs.input_features.squeeze(0).astype(np.float32)
    input_features = np.expand_dims(input_features, axis=0)  # (1, 80, 800)

    return input_features

# print(build_video_input("examples/videoplayback.mp4").shape)
# print(build_audio_input("examples/videoplayback.mp3").shape)

