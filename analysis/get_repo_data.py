import os
import pickle
import torch
import pandas as pd
import numpy as np


with open('./results_GIG.pkl', 'rb') as f:
    attributions_gig = pickle.load(f)

with open('./cohort_sample_dict.pkl', 'rb') as f:
    cohort_sample_dict = pickle.load(f)

output_dir = './data_zenodo'
os.makedirs(output_dir, exist_ok=True)

samples = []
diseases = []
probabilities = []

data_attr_chan0 = np.zeros( (len(attributions_gig), 28749))
data_attr_chan1 = np.zeros( (len(attributions_gig), 28749))

data_cn_chan0 = np.zeros( (len(attributions_gig), 28749))
data_cn_chan1 = np.zeros( (len(attributions_gig), 28749))

data_top_bin_chan0 = np.zeros( (len(attributions_gig), 10))
data_top_bin_chan1 = np.zeros( (len(attributions_gig), 10))

# Loop over the attributions_gig dictionary
count = 0
for sample, data in attributions_gig.items():
    
    # metadata
    samples.append(sample)
    probabilities.append(data['Probability'])
    diseases.append(data['Primary Site'])
    
    # all GIG data
    data_attr_chan0[count,:] = data['GIG'][0,0,:].cpu().numpy()
    data_attr_chan1[count,:] = data['GIG'][0,1,:].cpu().numpy()

    # get the binned CNA data
    raw_sample = np.load(f'../DATA_CN/matrix_{sample}.npy')
    tot_raw = np.array(raw_sample[:28749])
    maj_raw = np.array(raw_sample[28749:])
    ch0_raw = tot_raw.copy()
    ch1_raw = tot_raw - maj_raw
    data_cn_chan0[count,:] = ch0_raw[:]
    data_cn_chan1[count,:] = ch1_raw[:]

    # Extract the indices of the top 10 bins in data_attr_chan0[count,:]
    data_top_bin_chan0[count,:] = np.argsort(data_attr_chan0[count, :])[-10:][::-1]
    data_top_bin_chan1[count,:] = np.argsort(data_attr_chan1[count, :])[-10:][::-1]

    count += 1

# write out the data as csv files
np.savetxt(output_dir+'/data_attr_chan0.csv', data_attr_chan0, delimiter=',')
np.savetxt(output_dir+'/data_attr_chan1.csv', data_attr_chan1, delimiter=',')

#data_cn_chan0_clean = np.nan_to_num(data_cn_chan0, nan=-1).astype(int)
#data_cn_chan1_clean = np.nan_to_num(data_cn_chan1, nan=-1).astype(int)
#np.savetxt(output_dir+'/data_cn_chan0.csv', data_cn_chan0_clean, delimiter=',', fmt='%d')
#np.savetxt(output_dir+'/data_cn_chan1.csv', data_cn_chan1_clean, delimiter=',', fmt='%d')

np.savetxt(output_dir+'/data_cn_chan0.csv', data_cn_chan0, delimiter=',', fmt='%.0f')
np.savetxt(output_dir+'/data_cn_chan1.csv', data_cn_chan1, delimiter=',', fmt='%.0f')

np.savetxt(output_dir+'/data_top_bin_chan0.csv', data_top_bin_chan0, delimiter=',', fmt='%d')
np.savetxt(output_dir+'/data_top_bin_chan1.csv', data_top_bin_chan1, delimiter=',', fmt='%d')

# create a dataframe for the sample index
data = {'ID_SAMPLE': samples, 'primary_site_text': diseases}
df = pd.DataFrame(data)
df.to_csv(output_dir+'/data_sample_index.csv', index=False)
