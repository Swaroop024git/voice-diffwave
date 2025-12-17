# Voice-DiffWave

A minimal PyTorch implementation of **DiffWave**, a neural vocoder that generates high-quality audio waveforms from mel spectrograms using diffusion-based generative modeling.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

This project implements a simplified DiffWave vocoder for voice synthesis experiments. DiffWave is a versatile diffusion model for audio synthesis that produces high-fidelity audio through an iterative denoising process.

### Features

- 🎵 **Mel-to-Waveform Synthesis** — Generate raw audio from mel spectrograms
- ⚡ **GPU Accelerated** — Full CUDA support for training and inference
- 🔧 **Configurable Architecture** — Easily adjust model size via YAML config
- 📦 **End-to-End Pipeline** — Preprocessing, training, and inference scripts included

## Project Structure

```
voice-diffwave/
├── configs/
│   └── config.yaml          # Model and training configuration
├── data/
│   ├── raw/
│   │   └── p225/            # Raw audio files (VCTK format)
│   └── processed/
│       ├── mels/            # Preprocessed mel spectrograms
│       ├── wavs/            # Preprocessed audio segments
│       └── index.json       # Dataset index file
├── src/
│   ├── preprocess.py        # Audio preprocessing pipeline
│   ├── dataset.py           # PyTorch dataset class
│   ├── model.py             # DiffWave model architecture
│   ├── train.py             # Training loop
│   └── inference.py         # Audio generation script
├── checkpoints/             # Saved model weights
├── output/                  # Generated audio files
└── README.md
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
cd voice-diffwave

# Install dependencies
pip install torch torchaudio numpy librosa soundfile pyyaml tqdm
```

## Quick Start

### 1. Prepare Your Data

Place your audio files in `data/raw/p225/` (supports `.flac` and `.wav` formats). This project is configured to work with the [VCTK Corpus](https://datashare.ed.ac.uk/handle/10283/3443) format.

### 2. Preprocess Audio

Convert raw audio files into mel spectrograms and segmented waveforms:

```bash
python src/preprocess.py
```

This will:
- Resample audio to 22,050 Hz
- Normalize waveforms to [-1, 1]
- Trim silence
- Segment into 2-second clips
- Compute mel spectrograms
- Save processed data to `data/processed/`

### 3. Train the Model

```bash
python src/train.py
```

Training progress will be logged to console, and checkpoints are saved to `checkpoints/` every 500 steps.

### 4. Generate Audio

```bash
python src/inference.py
```

Generated audio will be saved to `output/generated.wav`.

## Configuration

All parameters are defined in `configs/config.yaml`:

```yaml
train:
  device: cuda              # "cuda" or "cpu"
  batch_size: 2
  learning_rate: 0.001
  epochs: 100
  log_interval: 50
  save_interval: 500

model:
  residual_layers: 30       # Number of residual blocks
  residual_channels: 64     # Hidden dimension
  dilation_cycle: 10        # Dilation pattern cycle length

audio:
  sample_rate: 22050
  n_mels: 80

inference:
  checkpoint: checkpoints/model_6500.pt
  mel: data/processed/mels/mel_000000.npy
  out_dir: output
  out_name: generated.wav
  steps: 50
```

## Audio Processing Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `sample_rate` | 22,050 Hz | Audio sample rate |
| `n_fft` | 1,024 | FFT window size |
| `hop_length` | 256 | Hop length between frames |
| `win_length` | 1,024 | Window length |
| `n_mels` | 80 | Number of mel filterbanks |
| `fmin` | 0 Hz | Minimum frequency |
| `fmax` | 8,000 Hz | Maximum frequency |
| `segment_seconds` | 2.0 s | Training clip duration |

## Model Architecture

The DiffWave model consists of:

- **Input Convolution** — Projects 1-channel audio to hidden dimension
- **Residual Blocks** — 30 layers with dilated convolutions and gated activations
- **Output Convolution** — Projects back to 1-channel audio

Dilations follow a cyclic pattern: `[1, 2, 4, 8, ..., 512, 1, 2, ...]` with cycle length 10.

## Status

⚠️ **Experimental Project** — This is a learning/research implementation. The model architecture is simplified compared to the original DiffWave paper and may not achieve production-quality results.

## References

- [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/abs/2009.09761) (Kong et al., 2020)
- [VCTK Corpus](https://datashare.ed.ac.uk/handle/10283/3443)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

> **Note:** This project was put on hold for a while and has been manually uploaded to GitHub as-is. Some rough edges may exist — bear with it! 🙏

*Built with PyTorch* 🔥
