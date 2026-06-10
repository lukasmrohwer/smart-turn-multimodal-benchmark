import os
import glob
from pathlib import Path
from moviepy import VideoFileClip

def extract_sample(clip, end_time, n, output_dir):
    """Extracts a 32-frame video and 8-second audio sample ending at end_time."""
    folder_path = output_dir / str(n)
    folder_path.mkdir(parents=True, exist_ok=True)
    
    output_mp4 = folder_path / f"{n}.MP4"
    output_wav = folder_path / f"{n}.wav"
    
    fps = clip.fps
    if fps == 0 or fps is None:
        fps = 30 # default fallback
        
    frames_to_keep = 32
    start_time_video = max(0, end_time - (frames_to_keep / fps))
    
    # Trim and resize video
    trimmed_clip = clip.subclipped(start_time_video, end_time)
    resized_clip = trimmed_clip.resized((112, 112))
    
    resized_clip.write_videofile(
        str(output_mp4),
        codec="libx264",
        audio=False,
        logger=None
    )
    
    start_time_audio = max(0, end_time - 8.0)
    if clip.audio is not None:
        audio_clip = clip.audio.subclipped(start_time_audio, end_time)
        audio_clip.write_audiofile(
            str(output_wav),
            logger=None
        )
        audio_clip.close()
    else:
        print(f"Warning: No audio found for sample {n}")
        
    resized_clip.close()
    trimmed_clip.close()

def process_videos():
    input_dir = Path("CasualConversationsA")
    output_dir = Path("examples")
    
    # Recursively find all *02.MP4 files
    mp4_files = sorted(input_dir.rglob("*02.MP4"))
    
    n = 1
    for i, video_path in enumerate(mp4_files, 1):
        print(f"Processing video {i}/{len(mp4_files)}: {video_path}")
        
        try:
            clip = VideoFileClip(str(video_path))
            duration = clip.duration
            
            # Extract from the end of the first half (D/2)
            half_duration = duration / 2.0
            extract_sample(clip, half_duration, n, output_dir)
            n += 1
            
            # Extract from the end of the second half (D)
            extract_sample(clip, duration, n, output_dir)
            n += 1
            
            clip.close()
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

if __name__ == "__main__":
    process_videos()
