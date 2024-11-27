import whisper
import time
import os
import torch
from moviepy.editor import VideoFileClip
from datetime import datetime, timedelta

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Set file paths
current_dir = r"D:\4-Doing\Documents_Max"
video_path = os.path.join(current_dir, "EducatonDirection.mp4")  # Video file path

# Dynamically select Whisper model based on system resources
model_size = "base" if os.cpu_count() <= 4 else "large"
print(f"Loading Whisper model: {model_size} on {device}...")

# Load the model and specify the device
model = whisper.load_model(model_size, device=device)

def extract_audio_segment(video_path, start_time, duration, temp_audio_path):
    """Extract audio from a specific segment of the video"""
    try:
        video = VideoFileClip(video_path)
        segment = video.subclip(start_time, start_time + duration)
        segment.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
        video.close()
        segment.close()
        return True
    except Exception as e:
        print(f"Audio extraction error: {e}")
        return False

def transcribe_segment(video_path, output_dir, start_time, duration, segment_number):
    """Transcribe a specific segment of the video"""
    try:
        print(f"\nProcessing segment {segment_number}: Start={start_time}s, Duration={duration}s")
        
        # Create a temporary audio file
        temp_audio_path = os.path.join(output_dir, f"temp_segment_{segment_number}.wav")
        
        # Extract audio segment
        if not extract_audio_segment(video_path, start_time, duration, temp_audio_path):
            return
        
        # Transcribe the audio
        result = model.transcribe(
            temp_audio_path,
            language="zh",
            verbose=True
        )

        # Save transcription results to a text file
        output_file = os.path.join(output_dir, f"EducatonDirection_segment_{segment_number}.txt")
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(result["text"])

        # Delete the temporary audio file
        os.remove(temp_audio_path)
        print(f"Segment {segment_number} transcription completed! Result saved to: {output_file}")
        
    except Exception as e:
        print(f"An error occurred during transcription of segment {segment_number}: {e}")
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

def process_video_in_segments(video_path, output_dir, segment_duration=1200, start_from=0):
    """
    Process the video in segments
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Output directory for transcriptions
        segment_duration (int): Duration of each segment in seconds
        start_from (int): Start time in seconds to begin processing
    """
    try:
        start_time = time.time()
        
        # Get the total duration of the video
        video = VideoFileClip(video_path)
        total_duration = video.duration
        video.close()

        # Calculate starting segment number based on start_from
        segment_number = (start_from // segment_duration) + 1
        current_start = start_from

        print(f"Starting from {start_from} seconds (segment {segment_number})")
        print(f"Total video duration: {timedelta(seconds=int(total_duration))}")
        print(f"Remaining duration to process: {timedelta(seconds=int(total_duration - start_from))}")

        while current_start < total_duration:
            segment_start_time = time.time()
            duration = min(segment_duration, total_duration - current_start)
            transcribe_segment(video_path, output_dir, current_start, duration, segment_number)
            
            # 计算并显示进度
            segment_time = time.time() - segment_start_time
            progress = (current_start + duration) / total_duration * 100
            remaining_duration = total_duration - (current_start + duration)
            estimated_time = (remaining_duration / duration) * segment_time if duration > 0 else 0
            
            print(f"\nProgress: {progress:.1f}%")
            print(f"Segment processing time: {timedelta(seconds=int(segment_time))}")
            print(f"Estimated remaining time: {timedelta(seconds=int(estimated_time))}")
            
            current_start += segment_duration
            segment_number += 1

        total_time = time.time() - start_time
        print(f"\nTotal processing time: {timedelta(seconds=int(total_time))}")

    except Exception as e:
        print(f"An error occurred during processing of the video: {e}")

def merge_transcriptions(output_dir, base_filename, start_segment=1, end_segment=None):
    """
    合并所有转录文件为一个文档
    
    Args:
        output_dir: 输出目录
        base_filename: 基础文件名（如 'EducatonDirection'）
        start_segment: 起始段号
        end_segment: 结束段号（如果为None，则自动检测）
    """
    try:
        # 确定最后一个段落号
        if end_segment is None:
            segment = start_segment
            while os.path.exists(os.path.join(output_dir, f"{base_filename}_segment_{segment}.txt")):
                segment += 1
            end_segment = segment - 1

        # 合并文件名
        merged_filename = os.path.join(output_dir, f"{base_filename}_merged.txt")
        
        print(f"\nMerging transcriptions from segment {start_segment} to {end_segment}")
        
        with open(merged_filename, 'w', encoding='utf-8') as outfile:
            for segment in range(start_segment, end_segment + 1):
                segment_file = os.path.join(output_dir, f"{base_filename}_segment_{segment}.txt")
                if os.path.exists(segment_file):
                    with open(segment_file, 'r', encoding='utf-8') as infile:
                        content = infile.read().strip()
                        outfile.write(f"\n=== Segment {segment} ===\n\n")
                        outfile.write(content)
                        outfile.write('\n\n')
                else:
                    print(f"Warning: Segment {segment} file not found")
        
        print(f"Merged transcription saved to: {merged_filename}")
        
    except Exception as e:
        print(f"Error merging transcriptions: {e}")

# Set parameters
output_dir = current_dir
segment_duration = 20 * 60  # 20 minutes in seconds
start_from = 0  # 设置开始时间（秒），比如要从30分钟处开始，就设置 start_from = 30 * 60

# Begin processing
print(f"\nBegin processing video: {os.path.basename(video_path)}")
process_video_in_segments(video_path, output_dir, segment_duration, start_from)

# 合并所有转录文件
base_filename = "EducatonDirection"
start_segment = (start_from // segment_duration) + 1
merge_transcriptions(output_dir, base_filename, start_segment)

print("All processing complete!")