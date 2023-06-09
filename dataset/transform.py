# We use TrOCRProcessor to prepare the data for the model. 
# TrOCRProcessor is actually just a wrapper around a ViTFeatureExtractor 
#(which can be used to resize + normalize images) and a RobertaTokenizer 
#(which can be used to encode and decode text into/from input_ids).


from torch.utils.data import Dataset
from transformers import TrOCRProcessor
import pandas as pd
from sklearn.model_selection import train_test_split
from config import paths, constants

print('started data processing')
print('----------------------------------------------')


def splitdata(path = paths.gt_path ):
    df = pd.read_fwf(paths.gt_path, header=None)
    df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
    del df[2]
    # some file names end with jp instead of jpg, let's fix this
    df['file_name'] = df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)
    df.to_csv(paths.label_dir / "label_data.csv", index=True)
    train_df, test_df = train_test_split(df, test_size=0.2)
   
    # we reset the indices to start from zero
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    change_csv(train_df, test_df)

 

def change_csv(train_df, test_df):
    
    train_data = train_df.to_csv(paths.train_dir / "train_data.csv", index=True)
    test_data = test_df.to_csv(paths.val_dir / "test_data.csv", index=True)
    

#if __name__ == " __main__":  
#    splitdata()

splitdata()







