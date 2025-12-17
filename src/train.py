import yaml
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Fix imports to match file names and support both run contexts
try:
    from dataset import MelWavDataset
    from model import DiffWave
except ImportError:
    from src.dataset import MelWavDataset
    from src.model import DiffWave

def load_config(path=None):
    # Resolve default to: <project_root>/configs/config.yaml
    if path is None:
        path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    else:
        path = Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_step(model, mel, wav, optimizer, device):
    model.train()

    noise = torch.randn_like(wav).unsqueeze(1)  # (B,1,T)
    wav = wav.unsqueeze(1)

    pred_noise = model(noise, mel)

    loss = F.mse_loss(pred_noise, wav)  # simplified target = waveform

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def main():
    config = load_config()

    project_root = Path(__file__).resolve().parents[1]
    index_path = Path(config["data"]["index_file"])
    if not index_path.is_absolute():
        index_path = project_root / index_path

    device = config["train"]["device"]
    batch_size = config["train"]["batch_size"]
    learning_rate = config["train"]["learning_rate"]
    epochs = config["train"]["epochs"]

    dataset = MelWavDataset(str(index_path))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DiffWave(
        residual_layers = config["model"]["residual_layers"],
        residual_channels = config["model"]["residual_channels"],
        dilation_cycle = config["model"]["dilation_cycle"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoints_dir = Path(__file__).resolve().parents[1] / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    step = 0

    for epoch in range(epochs):
        for mel, wav in dataloader:
            mel = mel.to(device)
            wav = wav.to(device)

            loss = train_step(model, mel, wav, optimizer, device)

            if step % config["train"]["log_interval"] == 0:
                print(f"Step {step}, Loss {loss:.6f}")

            if step % config["train"]["save_interval"] == 0:
                torch.save(model.state_dict(), str(checkpoints_dir / f"model_{step}.pt"))

            step += 1

if __name__ == "__main__":
    main()
