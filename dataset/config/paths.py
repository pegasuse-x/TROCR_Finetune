import os
from pathlib import Path

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
trocr_dir = dir_path.parent.parent  # this file is in TROCR/dataset/config

#print(trocr_dir)

trocr_repo = "microsoft/trocr-base-handwritten"
model_path = trocr_dir / "model"

train_dir = trocr_dir / "train"
train_file = train_dir / "train_data.csv"

val_dir = trocr_dir / "val"
val_file = val_dir / "test_data.csv"

label_dir = trocr_dir / "gt"
label_file = label_dir / "label_data.csv"

gt_path = trocr_dir / "IAM/gt_test.txt"
label_path = trocr_dir / "dataset/data.csv"

image_dir = trocr_dir / "IAM/image/"



#print(gt_path, label_path)


# automatically create all directories
for dir in [train_dir, val_dir, label_dir]:
    dir.mkdir(parents=True, exist_ok=True)