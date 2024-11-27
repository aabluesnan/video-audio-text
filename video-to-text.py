import whisper
import time
import os
import torch  # Check if a GPU is available

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

# Start transcription
print(f"\nBegin to transcribe video: {os.path.basename(video_path)}")
print("Transcribing, please be patient...")

start_time = time.time()

try:
    # Transcribe directly from the video file
    result = model.transcribe(
        video_path,
        language="zh",  # Specify language (or omit for auto-detection)
        verbose=True
    )

    # Save transcription results to a text file
    output_file = os.path.join(current_dir, "EducatonDirection.txt")
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(result["text"])

    total_time = time.time() - start_time
    print(f"\nTranscribing completed!") 
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Result saved to: {output_file}")

except Exception as e:
    print(f"An error occurred during transcription: {e}")