# Script to return & plot metadata associated with genomic-shaped data.

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import cycle
import pickle

# ------------------------------------
#           Load my own modules
# ------------------------------------
from Plotting import *

random.seed(42)

with open('./results_GIG.pkl', 'rb') as f:
    attributions_gig = pickle.load(f)

with open('./cohort_sample_dict.pkl', 'rb') as f:
    cohort_sample_dict = pickle.load(f)
    print(cohort_sample_dict)

df = pd.read_csv("../annotation/metadata.csv")
code_to_text = dict(zip(df["primary_site_code"], df["primary_site_text"]))
text_to_code = dict(zip(df["primary_site_text"], df["primary_site_code"]))


diseases = ['BREAST', 'LUNG', 'COLORECTAL']

for DISEASE in diseases:

    # get the indices of the samples that are of the given disease:
    indices = cohort_sample_dict[DISEASE]
    random_indices = random.sample(indices, 6)

    for i in random_indices: # 4 samples from each disease

        print('Sample number: ', i)
        
        crc_sample_GIG = attributions_gig[i]['GIG']
       

        # get the probability for this given disease from the dictionary key 'Prediction' and choose the number corresponding to the disease:
        print(attributions_gig[i]['Prediction'])
        
        disease_probability = attributions_gig[i]['Probability'][text_to_code[DISEASE]]
        print(f'Probability of {DISEASE}: {disease_probability}')

        #  -----------------------------------
        #        Plotting - one sample
        #  -----------------------------------

        metadata = CNVMetaDatach0ch1(crc_sample_GIG, normalise=True)

        metadata.high_attribute_metadata(return_df=False, 
                                        #  select top 1% of the distribution:
                                        threshold_percentile=99.
                                        )
        
        metadata.plot_manhattan(positive=True, # only plot positive attributions
                                save_string=f'plots-SAMPLE/{DISEASE}_Manhattan_attributions_sample_{i}.pdf',
                                save_fmt='pdf',
                                # title='Guided Integrated Gradients for CRC sample 0',
                                xlabel='Chromosome',
                                ylabel='Attribution',
                                label=False)
        
        metadata.plot_OG_sample_mosaic(save_string=f'plots-SAMPLE/{DISEASE}_OG_sample_{i}.pdf',
                               save_fmt='pdf',
                               sample_index=i,
                               cancer_type=DISEASE,
                               # title=f'{DISEASE}: sample {i}: P({DISEASE}) = {disease_probability:.2f})',
                               )
























# for i in range(1): # 4 samples from each disease
#     for disease in attributions_gig.keys():

#         # randomly pick a number between 0 and 99:
#         sample_number = np.random.randint(0, 100)

#         # Select the one sample or loop over all samples:
#         crc_sample = attributions_gig[disease][sample_number].squeeze() # first sample, channel 0
#         crc_sample_ch0 = crc_sample[0].numpy().copy()
#         crc_sample_ch1 = crc_sample[1].numpy().copy()


#         #  -----------------------------------
#         #        Plotting - one sample
#         #  -----------------------------------

#         metadata = CNVMetaDatach0ch1(crc_sample, normalise=True)

#         metadata.high_attribute_metadata(return_df=False, 
#                                         #  select top 1% of the distribution:
#                                         threshold_percentile=99.
#                                         )
#         metadata.plot_manhattan(positive=True, # only plot positive attributions
#                                 save_string=f'analysis_MAB_v1/sample/{disease}_Manhattan_attributions_sample_{sample_number}',
#                                 save_fmt='pdf',
#                                 # title='Guided Integrated Gradients for CRC sample 0',
#                                 xlabel='Chromosome',
#                                 ylabel='Attribution',
#                                 label=False)
#         # metadata.plot_bubble(THRESHOLD=0.01,
#         #                     save_string=f'analysis_MAB_v1/sample/Bubble_{disease}_attributions_sample_{sample_number}',
#         #                     save_fmt='pdf',
#         #                     )
