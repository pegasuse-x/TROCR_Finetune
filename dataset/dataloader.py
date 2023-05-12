import pandas as pd
from sklearn.model_selection import train_test_split
from transform import IAMDataset
from transformers import TrOCRProcessor
from PIL import Image
from util import imshow
import numpy as np 
import cv2 as cv 

from torch.utils.data import DataLoader

df = pd.read_fwf('../../IAM/gt_test.txt', header=None)
df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
del df[2]
# some file names end with jp instead of jpg, let's fix this
df['file_name'] = df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)


def splitdata(df):
    train_df, test_df = train_test_split(df, test_size=0.2)
    # we reset the indices to start from zero
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return train_df, test_df

train_df = splitdata(df)[0]
test_df = splitdata(df)[1]


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
train_dataset = IAMDataset(root_dir='../../IAM/image/',
                           df=train_df,
                           processor=processor)
eval_dataset = IAMDataset(root_dir='../../IAM/image/',
                           df=test_df,
                           processor=processor)



train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=4)



# check 

#print("Number of training examples:", len(train_dataset))
#print("Number of validation examples:", len(eval_dataset))

#encoding = train_dataset[0]
#for k,v in encoding.items():
#  print(k, v.shape)

#image = Image.open(train_dataset.root_dir + train_df['file_name'][0]).convert("RGB")
#image = np.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
#print(image.shape)
#imshow(image)

#image.show()
#labels = encoding['labels']
#labels[labels == -100] = processor.tokenizer.pad_token_id
#label_str = processor.decode(labels, skip_special_tokens=True)
#print("-----------------")
#print(label_str)