import pandas as pd 
import numpy as np 
from sklearn.metrics import roc_auc_score 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.pipeline import Pipeline
from model_dispatcher import models
import joblib
from sklearn.pipeline import _name_estimators 
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import argparse 
from sklearn.compose import make_column_transformer
def run(fold,model):
    df = pd.read_csv('../input/train_fold.csv')
    df['bmi'] = df['bmi'].fillna(np.mean(df['bmi']))
    df_train = df[df.kfold!=fold].reset_index(drop=True)
    df_valid = df[df.kfold==fold].reset_index(drop=True)
    features =  [f for f in df.columns if f not in ('id','stroke','kfold')]
    categorial_features = [f for f in features if df[f].dtype==object]
    numerical_features = [f for f in features if df[f].dtype!=object]
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    preprocess = make_column_transformer((OneHotEncoder(),categorial_features),(StandardScaler(),numerical_features))
    x_train = df_train[features]
    y_train = df_train.stroke 
    x_valid = df_valid[features]
    y_valid = df_valid.stroke
    clf = models[model]
    steps = [('preprocess',preprocess),('over',over),('under',under),('clf',clf)]
    pipe = Pipeline(steps=steps)
    pipe.fit(x_train,y_train)
    y_pred = pipe.predict_proba(x_valid)[:,1]
    auc = roc_auc_score(y_valid,y_pred)
    print("Fold : {} AUC Score: {:.3f}".format(fold,auc))
    joblib.dump(pipe,f'../models/dt_{model}_{fold}.bin')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold',type=int)
    parser.add_argument('--model',type=str)
    args = parser.parse_args()
    run(fold=args.fold,model=args.model)