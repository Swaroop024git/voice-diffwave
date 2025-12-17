import os
import json
import glob
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

# --------------------------------------------------
# CONFIG (edit these if needed)
# --------------------------------------------------
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
FMIN = 0
FMAX = 8000

SEGMENT_SECONDS = 2.0  # duration of training clips
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_SECONDS)

#RAW_DIR = "data/raw/p225"            # input folder after extracting VCTK
RAW_DIR = r"E:\Workspace\voice-diffwave\data\raw\p225"
PROCESSED_MEL_DIR = "data/processed/mels"
PROCESSED_WAV_DIR = "data/processed/wavs"
INDEX_FILE = "data/processed/index.json"

os.makedirs(PROCESSED_MEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_WAV_DIR, exist_ok=True)


# --------------------------------------------------
# HELPER: Compute Mel Spectrogram
# --------------------------------------------------
def wav_to_mel(wav):
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel


# --------------------------------------------------
# HELPER: Trim silence using librosa effects
# --------------------------------------------------
def trim_silence(wav):
    wav, _ = librosa.effects.trim(wav, top_db=20)
    return wav


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def preprocess():
    items = []  # list of dicts (mel_file, wav_file)

    wav_paths = sorted(glob.glob(os.path.join(RAW_DIR, "*_mic2.flac")))
    print(f"Found {len(wav_paths)} WAV files.")

    index_counter = 0

    for wav_path in tqdm(wav_paths, desc="Processing WAV files"):

        # Load + resample
        wav, sr = sf.read(wav_path)
        if wav.ndim > 1:
            wav = wav[:, 0]  # convert to mono

        if sr != SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)

        # Normalize to [-1, 1]
        wav = wav.astype(np.float32)
        wav = wav / np.max(np.abs(wav) + 1e-9)
  
        # Trim silence
        wav = trim_silence(wav)

        # Skip short files
        if len(wav) < SEGMENT_SAMPLES:
            continue

        # Break into fixed segments
        total_segments = len(wav) // SEGMENT_SAMPLES

        for i in range(total_segments):
            start = i * SEGMENT_SAMPLES
            end = start + SEGMENT_SAMPLES
            segment = wav[start:end]

            # Compute mel
            mel = wav_to_mel(segment)

            # Save files
            mel_file = os.path.join(PROCESSED_MEL_DIR, f"mel_{index_counter:06}.npy")
            wav_file = os.path.join(PROCESSED_WAV_DIR, f"wav_{index_counter:06}.npy")

            np.save(mel_file, mel.astype(np.float32))
            np.save(wav_file, segment.astype(np.float32))

            items.append({
                "mel": mel_file,
                "wav": wav_file
            })

            index_counter += 1

    # Write index file
    with open(INDEX_FILE, "w") as f:
        json.dump(items, f, indent=2)

    print(f"\nDone! Saved:")
    print(f"  {len(items)} mel/wav pairs")
    print(f"  Index file → {INDEX_FILE}")


if __name__ == "__main__":
    preprocess()
