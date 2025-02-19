# DeepCNA: an explainable deep learning method for cancer diagnosis and cancer-specific patterns of copy number aberrations

## Code developed by Mohamed Ali al-Badri and Chris P Barnes

### Requirements
Libraries:
PyTorch, numpy, seaborne, matplotlib.pyplot, statsmodels, sklearn

The data folders: DATA and DATA_CN 
These can be obtained from the Zenodo repository here and should be placed in the top level directory of the package.
https://zenodo.org/records/14892622

### Usage
#### train
This is where the netwotk is trained and optimsed. Look at run_train_cv.sh for how to run the 5-fold cross validation training.

train_cv.py <br>
There are three modes to run this script:
1. Model optimise mode: Run 5 fold CV with limited output to a text file (parameters, F1-macro on the test for each fold, plus mean F1-macro)
2. Model evaluation mode: Run 5 fold CV with full output to a directory (loss vs epoch, confusion matrices)
3. Final model fit on all the data. Confusion matrices should not be produced in this mode.

optimize_params.py <br>
This runs train_cv.py in mode 1 within optuna to find the best set of hyperparameters

#### analysis
run_inference.py <br>
This is where the Guided Integrated Gradients is run for samples in which the cancer class prediction was correct

plot_sample.py <br>
This script makes the sample level plots in the paper