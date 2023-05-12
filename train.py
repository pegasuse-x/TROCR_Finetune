# Here, we initialize the TrOCR model from its pretrained weights. 
# Note that the weights of the language modeling head are already initialized 
# from pre-training, as the model was already trained to generate text during its pre-training stage. 
# Refer to the paper for details. (TROcr, 2021)

from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from datasets import *
import evaluate
from transformers import AdamW
from tqdm.notebook import tqdm
import torch
#from .dataset.transform import *
from dataset.dataloader import processor, train_dataset, eval_dataset
from torch.utils.data import DataLoader

from multiprocessing import freeze_support

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)
eval_dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=True, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
model.to(device)

print ('model loaded succesfully')
print('----------------------------')
# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

# evaluation based on  Character Error Rate (CER)
cer_metric = evaluate.load("cer")
def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer



print("Dataloaded without error")
print('-------------------------------------------------------------------')


#def main(train_dataloader, eval_dataloader):



optimizer = AdamW(model.parameters(), lr = 5e-5)
for epoch in range(10):  # loop over the dataset multiple times
  # train
  print('training started')
  print('---------------------------------------------------')
  model.train()
  train_loss = 0.0
  for batch in tqdm(train_dataloader):
      # get the inputs
      for k,v in batch.items():
        batch[k] = v.to(device)

      # forward + backward + optimize
      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      train_loss += loss.item()

  print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
    
  # evaluate
  model.eval()
  valid_cer = 0.0
  with torch.no_grad():
    for batch in tqdm(eval_dataloader):
      # run batch generation
      outputs = model.generate(batch["pixel_values"].to(device))
      # compute metrics
      cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
      valid_cer += cer 

  print("Validation CER:", valid_cer / len(eval_dataloader))


#if __name__ == '__main__':
  #main(train_dataloader, eval_dataloader)

freeze_support()

model.save_pretrained("output/")



