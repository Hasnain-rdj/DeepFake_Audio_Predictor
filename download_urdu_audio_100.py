from datasets import load_dataset
import os
import soundfile as sf

# Create a directory for audio files
os.makedirs("urdu_audio_100", exist_ok=True)

# Load the dataset in streaming mode (no need for token if you've logged in with huggingface-cli)
ds = load_dataset(
    "CSALT/deepfake_detection_dataset_urdu",
    split="train",
    streaming=True
)

# Download and save the first 100 audio files
for i, sample in enumerate(ds):
    if i >= 100:
        break
    file_path = f"urdu_audio_100/audio_{i}.wav"
    sf.write(file_path, sample["audio"]["array"], sample["audio"]["sampling_rate"])
    print(f"Saved: {file_path}")

print("Downloaded the first 100 Urdu deepfake audio files to the 'urdu_audio_100' folder.")
