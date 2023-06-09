from dataclasses import dataclass

from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .dataloader import IAMDataset


@dataclass
class Context:
    model: VisionEncoderDecoderModel
    processor: TrOCRProcessor

    train_dataset: IAMDataset
    train_dataloader: DataLoader

    val_dataset: IAMDataset
    val_dataloader: DataLoader