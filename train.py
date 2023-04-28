# Here, we initialize the TrOCR model from its pretrained weights. 
# Note that the weights of the language modeling head are already initialized 
# from pre-training, as the model was already trained to generate text during its pre-training stage. 
# Refer to the paper for details. (TROcr, 2021)

from transformers import VisionEncoderDecoderModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
model.to(device)



