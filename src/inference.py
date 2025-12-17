import sys
import torch
import numpy as np
import soundfile as sf
import yaml
from pathlib import Path
from model import DiffWave

def load_config(path=None):
    # default: <project_root>/configs/config.yaml
    if path is None:
        project_root = Path(__file__).resolve().parents[1]
        path = project_root / "configs" / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_model(checkpoint_path, config, device):
    model = DiffWave(
        residual_layers=config["model"]["residual_layers"],
        residual_channels=config["model"]["residual_channels"],
        dilation_cycle=config["model"]["dilation_cycle"],
    ).to(device)

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def infer(model, mel, steps=50, device="cpu", audio_scale=None):
    mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)  # (1, n_mels, T)
    # estimate audio length: use hop_length if provided via audio_scale, else use 256
    hop = audio_scale.get("hop_length", 256) if audio_scale else 256
    audio_len = int(mel.shape[-1] * hop)
    audio = torch.randn(1, 1, audio_len, device=device)

    model.eval()
    with torch.no_grad():
        for t in range(steps):
            audio = model(audio, mel)
            print(f"Denoising step {t+1}/{steps}")

    audio = audio.squeeze().cpu().numpy()
    # safe normalize
    peak = np.max(np.abs(audio)) if audio.size else 1.0
    if peak > 0:
        audio = audio / peak
    return audio

def main(argv):
    # optional CLI: first arg = config path
    cfg_path = Path(argv[1]) if len(argv) > 1 else None
    config = load_config(cfg_path)

    project_root = Path(__file__).resolve().parents[1]

    # resolve device safely
    requested = config.get("train", {}).get("device", "cpu")
    device = "cuda" if (requested == "cuda" and torch.cuda.is_available()) else "cpu"
    if requested == "cuda" and device == "cpu":
        print("CUDA requested but not available; using CPU.")

    inf_cfg = config.get("inference", {})
    # checkpoint (resolve relative to project root)
    checkpoint = Path(inf_cfg.get("checkpoint", project_root / "checkpoints" / "model_6500.pt"))
    if not checkpoint.is_absolute():
        checkpoint = project_root / checkpoint

    # mel file (resolve relative to project root)
    mel_file = Path(inf_cfg.get("mel", project_root / "data" / "processed" / "mels" / "mel_000130.npy"))
    if not mel_file.is_absolute():
        mel_file = project_root / mel_file

    out_dir = Path(inf_cfg.get("out_dir", project_root / "output"))
    out_dir = out_dir if out_dir.is_absolute() else project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = inf_cfg.get("out_name", "generated.wav")
    out_path = out_dir / out_name

    # load model
    model = load_model(checkpoint, config, device)

    # load mel
    if not mel_file.exists():
        raise FileNotFoundError(f"Mel file not found: {mel_file}")
    mel = np.load(mel_file)

    steps = int(inf_cfg.get("steps", 50))
    audio = infer(model, mel, steps=steps, device=device, audio_scale=config.get("audio", {}))

    sample_rate = int(config.get("audio", {}).get("sample_rate", 22050))
    sf.write(str(out_path), audio, sample_rate)
    print("Generated:", out_path)

if __name__ == "__main__":
    main(sys.argv)
