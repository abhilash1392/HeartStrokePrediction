# Heart Stroke Prediction

Folder Structure

 |-input

 | |-healthcare-dataset-stroke-data.csv - Dataset from Kaggle 

 | |-train_fold.csv -  Output of src/create_folds.py

 |-src

 | |-model_dispatcher.py - Contains a dictionary of differenct models
 
 | |-train.py -  Main training Script
 
 | |-create_folds.py - Script used to create input/train_fold.csv
 
 | |-run.sh - Script used to run all files

 
 |-models
 
 | |-dt_2.bin
 
 | |-dt_0.bin
 
 | |-dt_logreg_2.bin
 
 |-notebooks
 
 | |-eda.ipynb

Output -- sh run.sh 

Fold : 0 AUC Score: 0.806

Fold : 1 AUC Score: 0.892

Fold : 2 AUC Score: 0.832

Fold : 3 AUC Score: 0.791

Fold : 4 AUC Score: 0.859