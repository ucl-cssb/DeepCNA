import pickle
import torch
import sys
import random
import gc

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from tqdm import tqdm


# ------------------------------------
#           Load my own modules
# ------------------------------------

sys.path.append('../train/')

from GIG import *

from DataLoader import CancerDataInd
from Network import MultiInputsNet

df = pd.read_csv("../annotation/metadata.csv")

cohort_df = pd.read_csv("../annotation/primary_site_info.csv") 
code_to_text = dict(zip(df["primary_site_code"], df["primary_site_text"]))
print(code_to_text)

all_data  = CancerDataInd(df["ID_SAMPLE"].values, df["primary_site_code"].values)
all_loader = torch.utils.data.DataLoader(all_data,batch_size=1,shuffle=True)

model = MultiInputsNet(compression_size = 60, hidden_dim1 = 900, hidden_dim2 = 300, pdropout = 0.4, output_dim=13)
model.load_state_dict(torch.load("../train/results_final/model_weights_final.pth",weights_only=True))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model is on {device}")

#  -----------------------------------
# 
#           Inference
# 
# -----------------------------------

# Dictionary to store the results
all_info_dict = {}

# Dictionary to store the samples for each cohort
cohort_sample_dict = {}
for primary_site in cohort_df["primary_site_text"]:
    cohort_sample_dict[primary_site] = []

model.eval()

count = 0
for X, y, index in tqdm(all_loader, desc="Processing Samples", unit="sample"):  # Assuming test_loader returns (input, label, index)
    #count += 1
    #if count > 1000:
    #    break

    # ------------------------------------------------------
    #  addition September 2023, to avoid negative attributions due to 
    #  inference on samples which the model does not classify correctly:
    # run the model on the sample and get the prediction:
    X = X.to(device).float()
    y = y.to(device)
    with torch.no_grad():
        output = model(X)
        y_pred = torch.argmax(output, dim=1)
        y_prob = torch.softmax(output, dim=1)

    # run the guided integrated gradients on the sample if and only if the prediction is correct:
    if y_pred == y:
    # ------------------------------------------------------ 

        # define the baseline as the median of that sample:
        gig_torch_100 = GuidedIG_torch(model=model, device='cuda', target=y)
        baseline_median = torch.zeros_like(X)
        baseline_median[0][0] = torch.median(X[0][0])
        baseline_median[0][1] = torch.median(X[0][1])
        
        gig_pair_torch_100 = gig_torch_100.guided_ig_impl(X.cpu().numpy(), 
                                                        baseline_median.cpu().numpy(), 
                                                        steps=100, 
                                                        fraction=0.1, 
                                                        max_dist=0.02
                                                        )
        # convert primary_site_code to primary_site_text
        primary_site = code_to_text[y.item()]

        ind_dict = {}
        ind_dict['GIG'] = gig_pair_torch_100
        ind_dict['Sample'] = X.cpu().numpy()
        ind_dict['Label'] = y.cpu().numpy()
        ind_dict['Primary Site'] = primary_site
        ind_dict['Prediction'] = y_pred.cpu().numpy()[0]
        ind_dict['Probability'] = y_prob.cpu().numpy()[0]

        #print(ind_dict)
        #print(ind_dict['Prediction'])
        #print(ind_dict['Probability'])
        #print(ind_dict['Primary Site'])

        all_info_dict[index.item()] = ind_dict 
        cohort_sample_dict[primary_site].append(index.item())

# save the dict to a file:
with open('./results_GIG.pkl', 'wb') as f:
    # pickle.dump(GIG_dict, f)
    pickle.dump(all_info_dict, f)

# write the cohort_sample_dict to a file:
with open('./cohort_sample_dict.pkl', 'wb') as f:
    pickle.dump(cohort_sample_dict, f)
