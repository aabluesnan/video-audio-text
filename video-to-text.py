#Step 1: Extract Audio from Video
from moviepy.editor import VideoFileClip
import os
import whisper
import time

def extract_audio_from_video(video_path, audio_output_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_output_path)

# Set the directory path
current_dir = r"D:\4-Doing\Documents_Max"
video_path = os.path.join(current_dir, "EducatonDirection.mp4")  
audio_output_path = os.path.join(current_dir, "EducatonDirection.wav")  # Extracted audio file path

extract_audio_from_video(video_path, audio_output_path)

#Step 2: Convert Audio to Text
# Load Whisper model, 'large' model has higher accuracy and supports different languages
model = whisper.load_model("large")

# Set the audio file path
audio_file = os.path.join(current_dir, "EducatonDirection.wav")  # .wav file path

print(f"\nBegin to transcribe audio: {os.path.basename(audio_file)}")
print("Transcribing, please be patient...")

start_time = time.time()

# Use verbose=True to display built-in progress information    
result = model.transcribe(
    audio_file, 
    language="zh",  # Language code, e.g., "en-US", "zh-CN", "fr-FR", "es-ES", etc.
    verbose=True
)

# Save the recognition result to a text file
output_file = os.path.join(current_dir, "EducatonDirection.txt")
with open(output_file, "w", encoding="utf-8") as file:
    file.write(result["text"])

total_time = time.time() - start_time
print(f"\nTranscribing completed!") 
print(f"Total time: {total_time:.1f} seconds")
print(f"Result saved to: {output_file}")
