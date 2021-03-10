# Importing 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import StratifiedKFold


if __name__=="__main__":
    # Importing the data 
    df  = pd.read_csv('../input/healthcare-dataset-stroke-data.csv')
    # Shuffling the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    df = df[df.gender!='Other'].reset_index(drop=True)
    df['kfold'] = -1
    y = df.stroke.values 
    kfold = StratifiedKFold(n_splits=5)
    for f,(t_,v_) in enumerate(kfold.split(X=df,y=y)):
        df.loc[v_,"kfold"]=f 

    df.to_csv('../input/train_fold.csv',index=False)
    print('Done')