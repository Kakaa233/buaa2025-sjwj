from pathlib import Path
import json
from typing import Tuple, List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class AnomalyDataset(Dataset):
    def __init__(self, root: Path, category: str, split: str, label_json: Path, image_size: int = 256):
        """
        root: path to Image_Anomaly_Detection directory (the inner one)
        category: 'hazelnut' or 'zipper'
        split: 'train' or 'test'
        label_json: path to image_anomaly_labels.json (only used for test)
        """
        self.root = Path(root)
        self.category = category
        self.split = split
        self.label_json = Path(label_json)
        self.image_size = image_size

        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        if split == "train":
            # train has subfolders good/bad
            self.samples = []
            for sub in ["good", "bad"]:
                for p in sorted((self.root / category / split / sub).glob("*.png")):
                    label = 0 if sub == "good" else 1
                    self.samples.append((p, label))
        else:
            # test images are mixed; labels from json
            with open(self.label_json, 'r', encoding='utf-8') as f:
                labels: Dict[str, Dict[str, str]] = json.load(f)
            self.samples = []
            for p in sorted((self.root / category / split).glob("*.png")):
                key = f"{category}/test/{p.name}".replace('\\', '/')
                if key not in labels:
                    raise KeyError(f"Missing label for {key} in {self.label_json}")
                label = 0 if labels[key]["label"] == "good" else 1
                self.samples.append((p, label))

        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label, str(path)
