#!/bin/bash

# There are three modes to run this script:
# 1. Model optimise mode: Run 5 fold CV with limited output to a text file (parameters, F1-macro on the test for each fold, plus mean F1-macro)
# 2. Model evaluation mode: Run 5 fold CV with full output to a directory (loss vs epoch, confusion matrices)
# 3. Final model fit on all the data. Confusion matrices should not be produced in this mode.


## Optimise the baseline model using grid search: replaced by optuna
## Grid search over params
# for lr in "0.001" "0.005" "0.01"; do
#     for dp in 0.1 0.2 0.5; do
#         for wd in 0.01 0.02; do
#             for bs in 64 128; do
#                 for mom in 0.85 0.9 0.99; do
#                     python train_cv.py --lr $lr --dropout $dp --wd $wd --batch_size $bs --momentum $mom --output_file results_cv.txt
#                 done
#             done
#         done
#     done
# done



## Grid search over structure
# for h1 in 500 800 1000; do
#      for h2 in 30 50 60; do
#         for comp in 40 50 6ß00; do
#             python train_cv.py --epochs 50 --lr 0.005 --dropout 0.5 --wd 0.01 --batch_size 128 --momentum 0.85 --hidden1 $h1 --hidden2 $h2 --subnetout $comp --output_file results_cv_struct.txt
#         done
#     done
# done

## Evaluate the baseline model
##python train_cv.py --epochs 50 --lr 0.01 --dropout 0.5 --wd 0.02 --batch_size 100 --momentum 0.9 --evaluate 1 --outdir results_baseline


## Evaluate the current best model
#python train_cv.py --epochs 50 --lr 0.005 --dropout 0.5 --wd 0.01 --batch_size 128 --momentum 0.85 --evaluate 1 --outdir results_opt1
python train_cv.py --epochs 50 --wd 0.01 --hidden1 900 --hidden2 300 --subnetout 60 --momentum 0.75 --batch_size 256 --dropout 0.4 --lr 0.0099  --evaluate 1 --outdir results_opt1

## Fit the final model
##python train_cv.py --epochs 50 --wd 0.01 --hidden1 900 --hidden2 300 --subnetout 60 --momentum 0.75 --batch_size 256 --dropout 0.4 --lr 0.0099  --final 1 --outdir results_final