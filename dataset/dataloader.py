
import csv
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrOCRProcessor

from .config import paths
from .config import constants
from .util import debug_print



#train_df = splitdata(df)[0]
#test_df = splitdata(df)[1]


def load_csv_file(csv_path) -> dict[str, str]:
    assert csv_path.exists(), f"Label csv at {csv_path} does not exist."

    labels: dict[str, str] = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            label = row[2]
            image_name = row[1]
            labels[image_name] = label

    return labels
#print('###################################')
#print(f'labels: {load_csv_file(paths.label_file)}')

#def load_filepaths_and_labels(data_dir: Path = paths.train_dir) -> tuple[list, list]:
#    sample_paths: list[str] = []
#    labels: list[str] = []

#    label_dict = load_csv_file()

#    for file_name in os.listdir(data_dir):
#        path = data_dir / file_name

#        if file_name.endswith(".jpg") or file_name.endswith(".png"):
#            assert file_name in label_dict, f"No label for image '{file_name}'"
#            label = label_dict[file_name]

#            sample_paths.append(path)
#            labels.append(label)

#    debug_print(f"Loaded {len(sample_paths)} samples from {data_dir}")
#    assert len(sample_paths) == len(labels)
#    return sample_paths, labels

def load_filepaths_and_labels(data_dir) -> tuple[list, list]:
    sample_paths: list[str] = []
    labels: list[str] = []

    label_dict = load_csv_file(data_dir)

    with open (data_dir, 'r') as data:
        reader = csv.reader(data, delimiter=',')
        for row in reader: 
            file_name = row[1]
            label = row[2]

            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                assert file_name in label_dict, f"No label for image '{file_name}'"
                label = label_dict[file_name]

                sample_paths.append(file_name)
                labels.append(label)

    debug_print(f"Loaded {len(sample_paths)} samples from {data_dir}")

    assert len(sample_paths) == len(labels)
    return sample_paths, labels

pathss, labels = load_filepaths_and_labels(paths.train_dir)
#print('###################################')
print(f'path: {pathss}')
#print('###################################')
print(f'labels: {labels}')
#print('###################################')

print('data split complete')
#print('----------------------------------------------------')



class IAMDataset(Dataset):
    def __init__(self, data_dir: Path, processor: TrOCRProcessor, max_target_length=128):
        self.image_name_list, self.label_list = load_filepaths_and_labels(data_dir)
        self.processor = processor
        self.max_target_length = max_target_length # max([constants.word_len_padding] + [len(label) for label in self.label_list])

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_name_list[idx]).convert("RGB")
        image_tensor: torch.Tensor = self.processor(image, return_tensors="pt").pixel_values[0]
        
        # add labels (input_ids) by encoding the text
        
        label = self.label_list[idx]
        label_tensor = self.processor.tokenizer(
            label,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=self._max_label_len,
        ).input_ids[0]


        return {"idx" : idx, "input" : image_tensor, "label" : label_tensor}
    
    def get_label(self, idx) -> str:
        assert 0 <= idx < len(self.label_list), f"id {idx} outside of bounds [0, {len(self.label_list)}]"
        return self.label_list[idx]

    def get_path(self, idx) -> str:
        assert 0 <= idx < len(self.label_list), f"id {idx} outside of bounds [0, {len(self.label_list)}]"
        return self.image_name_list[idx]

class MemoryDataset(Dataset):
    def __init__(self, images: list[Image.Image], processor: TrOCRProcessor ):
        self.images = images 
        self.processor = processor
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].convert("RGB")
        image_tensor: torch.Tensor = self.processor(image, return_tensors="pt").pixel_values[0]

        # create fake label
        label_tensor: torch.Tensor = self.processor.tokenizer(
            "",
            return_tensors="pt",
        ).input_ids[0]

        return {"idx": idx, "input": image_tensor, "label": label_tensor}


        