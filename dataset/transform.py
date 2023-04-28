# We use TrOCRProcessor to prepare the data for the model. 
# TrOCRProcessor is actually just a wrapper around a ViTFeatureExtractor 
#(which can be used to resize + normalize images) and a RobertaTokenizer 
#(which can be used to encode and decode text into/from input_ids).

import torch
from torch.utils.data import Dataset
from PIL import Image

from dataloader import *



