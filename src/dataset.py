import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class MelWavDataset(Dataset):
    def __init__(self, index_file):
        index_path = Path(index_file)
        # Resolve index file relative to project root if needed
        if not index_path.is_absolute():
            project_root = Path(__file__).resolve().parents[1]
            index_path = project_root / index_path

        with open(index_path, "r") as f:
            items = json.load(f)

        self.project_root = Path(__file__).resolve().parents[1]
        self.items = []

        for it in items:
            mel_p = Path(it["mel"])
            wav_p = Path(it["wav"])

            if not mel_p.is_absolute():
                mel_p = self.project_root / mel_p
            if not wav_p.is_absolute():
                wav_p = self.project_root / wav_p

            if not mel_p.exists() or not wav_p.exists():
                print(f"Warning: missing file, skipping: mel={mel_p}, wav={wav_p}")
                continue

            self.items.append({"mel": str(mel_p), "wav": str(wav_p)})

        if len(self.items) == 0:
            raise FileNotFoundError(f"No valid mel/wav pairs found (checked index: {index_path})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        mel = np.load(item["mel"])
        wav = np.load(item["wav"])

        mel_t = torch.from_numpy(mel).float()
        wav_t = torch.from_numpy(wav).float()

        return mel_t, wav_t
