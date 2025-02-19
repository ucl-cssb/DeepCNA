import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import cycle

from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import matplotlib

outdir = "./"

class CNVMetaDatach0ch1:
    
    def __init__(self, both_channel_attribution, normalise=True):
        chrom_lengths_arr = np.array([248956422, 242193529, 198295559, 190214555, 181538259,
                    170805979, 159345973, 145138636, 138394717, 133797422,
                    135086622, 133275309, 114364328, 107043718, 101991189,
                    90338345,  83257441,  80373285,  58617616,  64444167,
                    46709983,  50818468])#,  156040895, 57227415])

        self.chrom_lengths_binned = [int(round(i/100000)) for i in chrom_lengths_arr]
        self.chrom_locations = [sum(self.chrom_lengths_binned[:i+1]) for i in range(len(self.chrom_lengths_binned))]
        self.chrom_locations.insert(0,0)

        # if the input (both_channel_attribution) has a three dimensions, squeeze it to two:
        if len(both_channel_attribution.shape) == 3:
            both_channel_attribution = np.squeeze(both_channel_attribution)
        self.ch0_attr = both_channel_attribution[0]
        self.ch1_attr = both_channel_attribution[1]

        if normalise:
            # Normalise so bin attributes sum to one within a sample

            self.ch0_attr = self.ch0_attr.numpy().copy()
            self.ch1_attr = self.ch1_attr.numpy().copy()

            sign_ch0 = np.sign(self.ch0_attr)
            sign_ch1 = np.sign(self.ch1_attr)

            self.ch0_attr = np.abs ( self.ch0_attr ) / np.sum(np.abs(self.ch0_attr))
            self.ch1_attr = np.abs ( self.ch1_attr ) / np.sum(np.abs(self.ch1_attr))

            # make an assertion that the sum is close to 1:
            assert np.isclose(np.sum(self.ch0_attr), 1), 'The sum of the ch0 attributions is not close to 1'
            assert np.isclose(np.sum(self.ch1_attr), 1), 'The sum of the ch1 attributions is not close to 1'

            self.ch0_attr = self.ch0_attr * sign_ch0
            self.ch1_attr = self.ch1_attr * sign_ch1

    def high_attribute_metadata(self, metadata_filepath='../annotation/all_bins_metadata.csv', threshold_percentile=None, return_df=True, return_counts=False, high_attr_bins=None):

        df_metadata = pd.read_csv(metadata_filepath)

        # make a dataframe of crc_sample_ch0 and crc_sample_ch1 attributions, using the index as the bin_number:
        df_ch0 = pd.DataFrame(self.ch0_attr, columns=['attributions_one_channel'])
        df_ch0['bin_number'] = df_ch0.index

        df_ch1 = pd.DataFrame(self.ch1_attr, columns=['attributions_one_channel'])
        df_ch1['bin_number'] = df_ch1.index

        # add the metadata, fusing on the bin_number:
        df_ch0_metadata = pd.merge(df_ch0, df_metadata, on='bin_number')
        # keep a copy of the full metadata for this sample:
        self.df_full = df_ch0_metadata.copy()

        # add the metadata, fusing on the bin_number:
        df_ch1_metadata = pd.merge(df_ch1, df_metadata, on='bin_number')
        

        if threshold_percentile:
            # if a threshold percentile is specified, return the bins with attributions above that threshold:
            threshold = np.percentile(df_ch0_metadata['attributions_one_channel'], threshold_percentile)
            df_ch0_metadata = df_ch0_metadata[df_ch0_metadata['attributions_one_channel'] > threshold].sort_values(by='attributions_one_channel', ascending=False)

            print(f'Number of bins pre-thresholding: {len(df_ch1_metadata)}')
            threshold = np.percentile(df_ch1_metadata['attributions_one_channel'], threshold_percentile)
            df_ch1_metadata = df_ch1_metadata[df_ch1_metadata['attributions_one_channel'] > threshold].sort_values(by='attributions_one_channel', ascending=False)
            print(f'Number of bins post-thresholding: {len(df_ch1_metadata)}')

            # print(f'max attr: { np.max(df_ch0_metadata["attributions_one_channel"]) }')
            # print(f'max attr: { np.max(df_ch1_metadata["attributions_one_channel"]) }')
            #  now min:
            # print(f'min attr: { np.min(df_ch0_metadata["attributions_one_channel"]) }')
            # print(f'min attr: { np.min(df_ch1_metadata["attributions_one_channel"]) }')
            # get a count of the number of high attribute bins:
            count_ch0 = len(df_ch0_metadata)
            count_ch1 = len(df_ch1_metadata)

        elif high_attr_bins:
            # if a list of bins is specified, return the metadata for those bins:
            df_ch0_metadata = df_ch0_metadata[df_ch0_metadata['bin_number'].isin(high_attr_bins)]

            df_ch1_metadata = df_ch1_metadata[df_ch1_metadata['bin_number'].isin(high_attr_bins)]

        self.df_ch0_metadata = df_ch0_metadata
        self.df_ch1_metadata = df_ch1_metadata

        if return_df:
            return self.df_ch0_metadata, self.df_ch1_metadata
        elif return_counts:
            return count_ch0, count_ch1
        
    def plot_manhattan(self, save_string = 'COLORECTAL_ch0_attributions', save_fmt='png', positive = True, label=True, **kwargs):
        sns.set_style('white')
        sns.set_context('talk')
        if positive:
            # only select the values with positive attributions, leave others as 0
            self.df_ch0_metadata['attributions_one_channel'] = self.df_ch0_metadata['attributions_one_channel'].apply(lambda x: x if x > 0 else 0)
        else:
            pass

        # make own palette for 22 chromosomes:
        hex_colours = [matplotlib.colors.rgb2hex(colour) for colour in cm.tab20.colors]
        hex_colours = hex_colours + ['#06C2AC', '#580F41'] 

        chrom_colours = {i: hex_colours[i-1] for i in range(1, 23)}

        chrom_to_colour = lambda x: chrom_colours[x]
        chromosome_colours_list_ch0 = self.df_ch0_metadata['chromosome'].apply(chrom_to_colour).tolist()
        cmap_ch0 = matplotlib.colors.ListedColormap(chromosome_colours_list_ch0)

        chromosome_colours_list_ch1 = self.df_ch1_metadata['chromosome'].apply(chrom_to_colour).tolist()
        cmap_ch1 = matplotlib.colors.ListedColormap(chromosome_colours_list_ch1)

        # plt.figure(figsize=(20, 5))
        # make a 1x2 grid of subplots:
        fig, ax = plt.subplots(2, 1 )#, figsize=(20, 10))
        # plt.scatter(self.df_ch0_metadata['bin_number'], self.df_ch0_metadata['attributions_one_channel'], c=chromosome_colours_list, cmap=cmap)
        ax[0].scatter(self.df_ch0_metadata['bin_number'], self.df_ch0_metadata['attributions_one_channel'], c=chromosome_colours_list_ch0, cmap=cmap_ch0, s=10.5)
        ax[1].scatter(self.df_ch1_metadata['bin_number'], self.df_ch1_metadata['attributions_one_channel'], c=chromosome_colours_list_ch1, cmap=cmap_ch1, s=10.5)

        if label:
            # Initialize a set to keep track of the strings that have been annotated
            annotated_strings = set()
            
            for i, row in self.df_ch0_metadata.iterrows():
                # Check if the point has any individual label (gene, double_elite_enhancer, cytoband)
                label_text = None
                if pd.notna(row['gene']):
                    label_text = row['gene']
                # elif pd.notna(row['double_elite_enhancer']):
                #     label_text = row['double_elite_enhancer']
                elif pd.notna(row['cytoband']):
                    label_text = row['cytoband']
                
                # Annotate the individual data point with its label if it's not a duplicate
                if label_text is not None and label_text not in annotated_strings:
                    # only annotate two labels (highest attribution) per chromosome:
                    # --------------------------------------------------------------    

                    # first get the chromosome number:
                    chrom_num = row['chromosome']
                    # get the two highest attributions for that chromosome:
                    highest_attributions = self.df_ch0_metadata[self.df_ch0_metadata['chromosome'] == chrom_num].sort_values(by='attributions_one_channel', ascending=False).iloc[:2]
                    # check if the current row is one of the two highest attributions:
                    if row['bin_number'] in highest_attributions['bin_number'].tolist():
                        ax[0].annotate(label_text, (row['bin_number'], row['attributions_one_channel']),
                                       fontweight='bold', 
                                       color = 'white',
                                       fontsize=8, 
                                       zorder=10, 
                                    #    ha='center', 
                                       bbox=dict(facecolor='k', edgecolor='k', boxstyle='round,pad=0.1')
                                       )
                        annotated_strings.add(label_text)
                    
                    ## IF YOU WANT ALL LABELS:
                    ## -----------------------
                    # ax[0].annotate(label_text, (row['bin_number'], row['attributions_one_channel']))
                    # annotated_strings.add(label_text)

            for i, row in self.df_ch1_metadata.iterrows():
                # Check if the point has any individual label (gene, double_elite_enhancer, cytoband)
                label_text = None
                if pd.notna(row['gene']):
                    label_text = row['gene']
                # elif pd.notna(row['double_elite_enhancer']):
                #     label_text = row['double_elite_enhancer']
                elif pd.notna(row['cytoband']):
                    label_text = row['cytoband']
                
                # Annotate the individual data point with its label if it's not a duplicate
                if label_text is not None and label_text not in annotated_strings:

                   # only annotate two labels (highest attribution) per chromosome:
                   # --------------------------------------------------------------    
                    # first get the chromosome number:
                    chrom_num = row['chromosome']
                    # get the two highest attributions for that chromosome:
                    highest_attributions = self.df_ch1_metadata[self.df_ch1_metadata['chromosome'] == chrom_num].sort_values(by='attributions_one_channel', ascending=False).iloc[:2]
                    # check if the current row is one of the two highest attributions:
                    if row['bin_number'] in highest_attributions['bin_number'].tolist():
                        ax[1].annotate(label_text, (row['bin_number'], row['attributions_one_channel']),
                                    fontweight='bold', 
                                    color = 'white',
                                    fontsize=8, 
                                    zorder=10, 
                                    # ha='center', 
                                    bbox=dict(facecolor='k', edgecolor='k', boxstyle='round,pad=0.1')
                                    )
                        annotated_strings.add(label_text)

                    ## IF YOU WANT ALL LABELS:
                    ## -----------------------
                    # ax[1].annotate(label_text, (row['bin_number'], row['attributions_one_channel']))
                    # annotated_strings.add(label_text)


        # Calculate the midpoint of bin_number for each chromosome
        chromosome_bins = self.df_full.groupby('chromosome')['bin_number'].mean()
        # Define chromosome labels from 1 to 22
        chromosome_labels = [f'{i}' for i in range(1, 23)]

        # make sure the xlim is 0, length of the genome:
        ax[0].set_xlim(0, self.df_full['bin_number'].max())
        ax[1].set_xlim(0, self.df_full['bin_number'].max())
        # Set the x-axis tick positions and labels
        ax[0].set_xticks(chromosome_bins)
        ax[0].set_xticklabels(chromosome_labels)
        # rotate them by 90 degrees:
        ax[0].tick_params(axis='x', rotation=90)
        # make them smaller:
        ax[0].tick_params(axis='x', labelsize=10)
        # Set the x-axis tick positions and labels
        ax[1].set_xticks(chromosome_bins)
        ax[1].set_xticklabels(chromosome_labels)
        # rotate them by 90 degrees:
        ax[1].tick_params(axis='x', rotation=90)
        # make them smaller:
        ax[1].tick_params(axis='x', labelsize=10)
        # remove the y tick labels:
        ax[0].set_yticklabels([])
        ax[1].set_yticklabels([])


        # ax[0].set_title('ch0', loc='left')
        # ax[1].set_title('ch1', loc='left')
        
        # read the title & axes labels from the kwargs:
        if 'title' in kwargs:
            fig.suptitle(kwargs['title'])

        if 'xlabel' in kwargs:
            # ax[0].set_xlabel(kwargs['xlabel'])
            ax[1].set_xlabel(kwargs['xlabel'])

        if 'ylabel' in kwargs:
            ax[0].set_ylabel(kwargs['ylabel'])
            ax[1].set_ylabel(kwargs['ylabel'])
        
        plt.tight_layout()
        
        if save_string:
            plt.savefig(save_string, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close('all')


    def return_square_sample(self, crc_single_channel):
        
        return_chrom_data = lambda crc_single_channel, chrom_num : crc_single_channel[self.chrom_locations[chrom_num - 1]:self.chrom_locations[chrom_num]]

        X = np.zeros((22, self.chrom_lengths_binned[0]))
        for i in range(22):
            X[i, :self.chrom_lengths_binned[i]] = return_chrom_data(crc_single_channel, i+1)
        return X
    
    
    def plot_bubble(self, THRESHOLD, save_string = None, save_fmt='png', **kwargs):

        sns.set_style('whitegrid')
        # sns.set_context('talk')
        sns.set_context('paper')
        threshold_ch0 = np.percentile(self.df_ch0_metadata['attributions_one_channel'], THRESHOLD)
        threshold_ch1 = np.percentile(self.df_ch1_metadata['attributions_one_channel'], THRESHOLD)

        binned_p_arm_end = [1234, 939, 909, 500, 488, 598, 601, 452, 430, 398, 534, 355, 177, 172, 190, 368, 251, 185,262,281,120,150]
        binned_q_arm_end = [2489, 2421, 1982, 1902, 1815, 1708, 1593, 1451, 1383, 1337, 1350, 1332, 1143, 1070, 1019, 903, 832, 803, 586, 644, 467, 508]
        binned_q_arm_size = [i-j for i,j in zip(binned_q_arm_end, binned_p_arm_end)]


        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,4), sharex=True) # for the paper
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,8), sharex=True) # for the talk

        # make a bubble plot for the total CN channel:
        square_ch0 = self.return_square_sample(self.ch0_attr) # total CN channel
        square_ch0[square_ch0 < 0] = 0
        # square_ch0 = np.abs(square_ch0)

        p_arm_counts = [np.sum(square_ch0[i, 0:binned_p_arm_end[i]] > threshold_ch0) for i in range(len(square_ch0))]
        q_arm_counts = [np.sum(square_ch0[i, binned_p_arm_end[i]:] > threshold_ch0) for i in range(len(square_ch0))]

        p_arm_counts = [(i/j)*300 for i,j in zip(p_arm_counts, binned_p_arm_end)]
        q_arm_counts = [(i/j)*300 for i,j in zip(q_arm_counts, binned_q_arm_size)]


        # get the max of square_ch0 according to the p arm end:
        ##############################################
        p_arm_maxvals = [np.max(square_ch0[i, 0:binned_p_arm_end[i]]) for i in range(len(square_ch0))]
        q_arm_maxvals = [np.max(square_ch0[i, binned_p_arm_end[i]:]) for i in range(len(square_ch0))]
        # normalise the max values to be between 0 and 1 with respect to the max value of p or q arm, depending on which is higher:
        p_arm_maxvals = [i/max(p_arm_maxvals) if max(p_arm_maxvals) > max(q_arm_maxvals) else i/max(q_arm_maxvals) for i in p_arm_maxvals]
        q_arm_maxvals = [i/max(q_arm_maxvals) if max(q_arm_maxvals) > max(p_arm_maxvals) else i/max(p_arm_maxvals) for i in q_arm_maxvals]
        ##############################################

        colors_p = [plt.cm.plasma_r(i) for i in p_arm_maxvals]
        colors_q = [plt.cm.plasma_r(i) for i in q_arm_maxvals]


        ax1.scatter(np.arange(1, len(p_arm_counts)+1), np.zeros(len(p_arm_counts)), s=p_arm_counts, color=colors_p, label='p arm')
        ax1.scatter(np.arange(1, len(p_arm_counts)+1), np.ones(len(q_arm_counts)), s=q_arm_counts, color=colors_q, label='q arm')
        ax1.set_xticks(np.arange(1, len(p_arm_counts)+1))
        ax1.set_xticklabels(np.arange(1, len(p_arm_counts)+1))

        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['p arm', 'q arm'])

        # change the y limits so that the points are not cut off:
        ax1.set_ylim(-0.5, 1.5)

        ax1.set_title('Total CN', loc='left')

        # make a legend and a size of the bubbles legend:
        handles, labels = ax1.get_legend_handles_labels()
        sizes = [i*60 for i in range(1,6)]

        # make a legend of scatter points, with the size of the scatter points as the legend:
        legend1 = ax1.legend([plt.scatter([],[], s=s, color='tab:gray') for s in sizes], sizes, loc='upper center',
                                        title='Density of high attribute bins (%)',
                                        ncol=5,
                                        bbox_to_anchor=(0.5, 1.5),
                                        # remove the border around the legend:
                                        frameon=False,
                                        # bbox_to_anchor=(1.3, 1.07)
                                        )

        # get the text from the legend and change the text to be half the size:
        for i in range(len(sizes)):
            text = legend1.get_texts()[i]
            text.set_text(str(int(sizes[i]/3)))
            # text.set_fontsize(10)


        ax1.add_artist(legend1)

        square_ch1 = self.return_square_sample(self.ch1_attr) # Major allele channel
        square_ch1[square_ch1 < 0] = 0
        # square_ch1 = np.abs(square_ch1)


        # get the max of square_ch1 according to the p arm end:
        ##############################################
        p_arm_maxvals = [np.max(square_ch1[i, 0:binned_p_arm_end[i]]) for i in range(len(square_ch1))]
        q_arm_maxvals = [np.max(square_ch1[i, binned_p_arm_end[i]:]) for i in range(len(square_ch1))]
        # normalise the max values to be between 0 and 1 with respect to the max value of p or q arm, depending on which is higher:
        p_arm_maxvals = [i/max(p_arm_maxvals) if max(p_arm_maxvals) > max(q_arm_maxvals) else i/max(q_arm_maxvals) for i in p_arm_maxvals]
        q_arm_maxvals = [i/max(q_arm_maxvals) if max(q_arm_maxvals) > max(p_arm_maxvals) else i/max(p_arm_maxvals) for i in q_arm_maxvals]
        ##############################################

        colors_p = [plt.cm.plasma_r(i) for i in p_arm_maxvals]
        colors_q = [plt.cm.plasma_r(i) for i in q_arm_maxvals]

        p_arm_counts = [np.sum(square_ch1[i, 0:binned_p_arm_end[i]] > threshold_ch1) for i in range(len(square_ch1))]
        q_arm_counts = [np.sum(square_ch1[i, binned_p_arm_end[i]:] > threshold_ch1) for i in range(len(square_ch1))]
        p_arm_counts = [(i/j)*300 for i,j in zip(p_arm_counts, binned_p_arm_end)]
        q_arm_counts = [(i/j)*300 for i,j in zip(q_arm_counts, binned_q_arm_size)]

        ax2.scatter(np.arange(1, len(p_arm_counts)+1), np.zeros(len(p_arm_counts)), s=p_arm_counts, color=colors_p, label='p arm')
        ax2.scatter(np.arange(1, len(q_arm_counts)+1), np.ones(len(q_arm_counts)), s=q_arm_counts, color=colors_q, label='q arm')

        ax2.set_xticks(np.arange(1, len(p_arm_counts)+1))
        ax2.set_xticklabels(np.arange(1, len(p_arm_counts)+1))
        ax2.set_xlabel('Chromosome')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['p arm', 'q arm'])

        # change the y limits so that the points are not cut off:
        ax2.set_ylim(-0.5, 1.5)

        # place the title at the left side of the plot:
        ax2.set_title('Minor CN', loc='left')

        # add a colorbar to the figure to show the maximum value of the attribute:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma_r, norm=plt.Normalize(vmin=0, vmax=1))

        sm._A = []

        # make one colorbar for both plots:
        cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='vertical', fraction=0.05, pad=0.05)
        cbar.set_label('Normalised maximum attribute value', rotation=270, labelpad=15)

        if save_string is not None:
            plt.savefig(save_string, bbox_inches='tight')

        plt.close('all')

    def plot_arm_scatter(self, THRESHOLD, save_string = None, save_fmt='png', **kwargs):

        sns.set_style('whitegrid')
        # sns.set_context('talk')
        sns.set_context('paper')
        threshold_ch0 = np.percentile(self.df_ch0_metadata['attributions_one_channel'], THRESHOLD)
        threshold_ch1 = np.percentile(self.df_ch1_metadata['attributions_one_channel'], THRESHOLD)

        binned_p_arm_end = [1234, 939, 909, 500, 488, 598, 601, 452, 430, 398, 534, 355, 177, 172, 190, 368, 251, 185,262,281,120,150]
        binned_q_arm_end = [2489, 2421, 1982, 1902, 1815, 1708, 1593, 1451, 1383, 1337, 1350, 1332, 1143, 1070, 1019, 903, 832, 803, 586, 644, 467, 508]
        binned_q_arm_size = [i-j for i,j in zip(binned_q_arm_end, binned_p_arm_end)]


        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,4), sharex=True) # for the paper
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,8), sharex=True) # for the talk

        # make a bubble plot for the total CN channel:
        square_ch0 = self.return_square_sample(self.ch0_attr) # total CN channel
        square_ch0[square_ch0 < 0] = 0
        # square_ch0 = np.abs(square_ch0)

        p_arm_counts = [np.sum(square_ch0[i, 0:binned_p_arm_end[i]] > threshold_ch0) for i in range(len(square_ch0))]
        q_arm_counts = [np.sum(square_ch0[i, binned_p_arm_end[i]:] > threshold_ch0) for i in range(len(square_ch0))]

        p_arm_counts = [(i/j) for i,j in zip(p_arm_counts, binned_p_arm_end)]
        q_arm_counts = [(i/j) for i,j in zip(q_arm_counts, binned_q_arm_size)]


        # get the max of square_ch0 according to the p arm end:
        ##############################################
        p_arm_maxvals = [np.max(square_ch0[i, 0:binned_p_arm_end[i]]) for i in range(len(square_ch0))]
        q_arm_maxvals = [np.max(square_ch0[i, binned_p_arm_end[i]:]) for i in range(len(square_ch0))]
        # normalise the max values to be between 0 and 1 with respect to the max value of p or q arm, depending on which is higher:
        p_arm_maxvals = [i/max(p_arm_maxvals) if max(p_arm_maxvals) > max(q_arm_maxvals) else i/max(q_arm_maxvals) for i in p_arm_maxvals]
        q_arm_maxvals = [i/max(q_arm_maxvals) if max(q_arm_maxvals) > max(p_arm_maxvals) else i/max(p_arm_maxvals) for i in q_arm_maxvals]
        ##############################################

        colors_p = [plt.cm.plasma_r(i) for i in p_arm_maxvals]
        colors_q = [plt.cm.plasma_r(i) for i in q_arm_maxvals]

        # I want to make a scatter plot of the density of high attribute bins vs maxval of the arm:
        # --------------------------------------------------------------------------------------
        # the density is p_arm_counts and q_arm_counts
        # the maxval is p_arm_maxvals and q_arm_maxvals
        # label the points with str(chrom_num) + 'p' or 'q':
        # --------------------------------------------------
        plt.figure(figsize=(10,10))
        # the values are in chronological order, so I can just plot and label them in order:
        for i in range(len(p_arm_counts)):
            plt.scatter(p_arm_maxvals[i], p_arm_counts[i], color=colors_p[i], label=str(i+1)+'p')
            plt.scatter(q_arm_maxvals[i], q_arm_counts[i], color=colors_q[i], label=str(i+1)+'q')
        plt.legend()
        plt.xlabel('Normalised maximum attribute value')
        plt.ylabel('Density of high attribute bins (%)')
        plt.title('Total CN')
        plt.tight_layout()
        if save_string is not None:
            plt.savefig(save_string, bbox_inches='tight')
        plt.close('all')
        
        # --------------------------------------------------------------------------------------




        # ax1.scatter(np.arange(1, len(p_arm_counts)+1), np.zeros(len(p_arm_counts)), s=p_arm_counts, color=colors_p, label='p arm')
        # ax1.scatter(np.arange(1, len(p_arm_counts)+1), np.ones(len(q_arm_counts)), s=q_arm_counts, color=colors_q, label='q arm')
        # ax1.set_xticks(np.arange(1, len(p_arm_counts)+1))
        # ax1.set_xticklabels(np.arange(1, len(p_arm_counts)+1))

        # ax1.set_yticks([0, 1])
        # ax1.set_yticklabels(['p arm', 'q arm'])

        # # change the y limits so that the points are not cut off:
        # ax1.set_ylim(-0.5, 1.5)

        # ax1.set_title('Total CN', loc='left')

        # # make a legend and a size of the bubbles legend:
        # handles, labels = ax1.get_legend_handles_labels()
        # sizes = [i*60 for i in range(1,6)]

        # # make a legend of scatter points, with the size of the scatter points as the legend:
        # legend1 = ax1.legend([plt.scatter([],[], s=s, color='tab:gray') for s in sizes], sizes, loc='upper center',
        #                                 title='Density of high attribute bins (%)',
        #                                 ncol=5,
        #                                 bbox_to_anchor=(0.5, 1.5),
        #                                 # remove the border around the legend:
        #                                 frameon=False,
        #                                 # bbox_to_anchor=(1.3, 1.07)
        #                                 )

        # # get the text from the legend and change the text to be half the size:
        # for i in range(len(sizes)):
        #     text = legend1.get_texts()[i]
        #     text.set_text(str(int(sizes[i]/3)))
        #     # text.set_fontsize(10)


        # ax1.add_artist(legend1)

        # square_ch1 = self.return_square_sample(self.ch1_attr) # Major allele channel
        # square_ch1[square_ch1 < 0] = 0
        # # square_ch1 = np.abs(square_ch1)


        # # get the max of square_ch1 according to the p arm end:
        # ##############################################
        # p_arm_maxvals = [np.max(square_ch1[i, 0:binned_p_arm_end[i]]) for i in range(len(square_ch1))]
        # q_arm_maxvals = [np.max(square_ch1[i, binned_p_arm_end[i]:]) for i in range(len(square_ch1))]
        # # normalise the max values to be between 0 and 1 with respect to the max value of p or q arm, depending on which is higher:
        # p_arm_maxvals = [i/max(p_arm_maxvals) if max(p_arm_maxvals) > max(q_arm_maxvals) else i/max(q_arm_maxvals) for i in p_arm_maxvals]
        # q_arm_maxvals = [i/max(q_arm_maxvals) if max(q_arm_maxvals) > max(p_arm_maxvals) else i/max(p_arm_maxvals) for i in q_arm_maxvals]
        # ##############################################

        # colors_p = [plt.cm.plasma_r(i) for i in p_arm_maxvals]
        # colors_q = [plt.cm.plasma_r(i) for i in q_arm_maxvals]

        # p_arm_counts = [np.sum(square_ch1[i, 0:binned_p_arm_end[i]] > threshold_ch1) for i in range(len(square_ch1))]
        # q_arm_counts = [np.sum(square_ch1[i, binned_p_arm_end[i]:] > threshold_ch1) for i in range(len(square_ch1))]
        # p_arm_counts = [(i/j)*300 for i,j in zip(p_arm_counts, binned_p_arm_end)]
        # q_arm_counts = [(i/j)*300 for i,j in zip(q_arm_counts, binned_q_arm_size)]

        # ax2.scatter(np.arange(1, len(p_arm_counts)+1), np.zeros(len(p_arm_counts)), s=p_arm_counts, color=colors_p, label='p arm')
        # ax2.scatter(np.arange(1, len(q_arm_counts)+1), np.ones(len(q_arm_counts)), s=q_arm_counts, color=colors_q, label='q arm')

        # ax2.set_xticks(np.arange(1, len(p_arm_counts)+1))
        # ax2.set_xticklabels(np.arange(1, len(p_arm_counts)+1))
        # ax2.set_xlabel('Chromosome')
        # ax2.set_yticks([0, 1])
        # ax2.set_yticklabels(['p arm', 'q arm'])

        # # change the y limits so that the points are not cut off:
        # ax2.set_ylim(-0.5, 1.5)

        # # place the title at the left side of the plot:
        # ax2.set_title('Minor CN', loc='left')

        # # add a colorbar to the figure to show the maximum value of the attribute:
        # sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma_r, norm=plt.Normalize(vmin=0, vmax=1))

        # sm._A = []

        # # make one colorbar for both plots:
        # cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='vertical', fraction=0.05, pad=0.05)
        # cbar.set_label('Normalised maximum attribute value', rotation=270, labelpad=15)

        if save_string is not None:
            plt.savefig(save_string, bbox_inches='tight')

        plt.close('all')

    def plot_OG_sample(self, sample_index, cancer_type, save_string = None, save_fmt='png', **kwargs):

        sns.set_style('whitegrid')
        # sns.set_context('talk')
        sns.set_context('paper')

        raw_sample = np.load(f'../GEL_NPY/matrix_{int(sample_index)}.npy')


        raw_squeeze = raw_sample.squeeze() 
        tot_raw = np.array(raw_squeeze[:28749])
        maj_raw = np.array(raw_squeeze[28749:])
        ch0_raw = tot_raw.copy()
        ch1_raw = tot_raw - maj_raw

        # make an empty numpy array of shape (2, 28749):
        raw_sample = np.empty((2, 28749))
        raw_sample[0] = ch0_raw
        raw_sample[1] = ch1_raw

        # keep a log of the copy state (by classification, Diploid, LOH, ... for the table further down)
        def interpret_state (major_al, minor_al):
            tot = major_al + minor_al

            if major_al == 0 and minor_al == 0:
                return 'Deletion'
            elif major_al == 1 and minor_al == 1:
                return 'Diploid'
            elif ((minor_al == 0 or major_al == 0) and tot < 3):
            # elif ((minor_al == 0 or major_al == 0) and tot != 0):
                return 'LOH'
            elif tot >= 3 and tot < 5:
                return 'Duplication'
            elif tot >= 5:
                return 'Amplification'
            else:
                return f'{str(major_al)}/{str(minor_al)}'
        
        # make a list of the copy states for each bin:
        copy_states = np.array([interpret_state(maj_raw[i], ch1_raw[i]) for i in range(len(maj_raw))])

        # make a subplot of two figures, one for ch0 and one for ch1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        # make the dots very faint and to have an edgecolor of black:
        ax1.scatter(range(len(raw_sample[0])), raw_sample[0], s=25, c='gray', alpha=0.2, label='ch0')#, edgecolors='black')
        ax2.scatter(range(len(raw_sample[1])), raw_sample[1], s=25, c='gray', alpha=0.2, label='ch1')#, edgecolors='black')
        # plt.show()

        # use the bin_number column from self.df_ch0_metadata to get the bin numbers for the high attribute bins, colour these in and make s=50:
        # get the bin numbers for the high attribute bins:
        bin_numbers_ch0 = self.df_ch0_metadata['bin_number'].tolist()
        bin_numbers_ch1 = self.df_ch1_metadata['bin_number'].tolist()

        # get the attributions for the high attribute bins:
        attributions_ch0 = self.df_ch0_metadata['attributions_one_channel'].tolist()
        attributions_ch1 = self.df_ch1_metadata['attributions_one_channel'].tolist()

        # make own palette for 22 chromosomes:
        hex_colours = [matplotlib.colors.rgb2hex(colour) for colour in cm.tab20.colors]
        hex_colours = hex_colours + ['#06C2AC', '#580F41']

        chrom_colours = {i: hex_colours[i-1] for i in range(1, 23)}

        # chrom_to_colour = lambda x: chrom_colours[x]

        # get the colours for the high attribute bins:
        colors_ch0 = self.df_ch0_metadata['chromosome'].apply(lambda x: chrom_colours[x]).tolist()
        colors_ch1 = self.df_ch1_metadata['chromosome'].apply(lambda x: chrom_colours[x]).tolist()

        # plot the high attribute bins:
        ax1.scatter(bin_numbers_ch0, raw_sample[0][bin_numbers_ch0], s=80, c=colors_ch0, label='ch0')
        ax2.scatter(bin_numbers_ch1, raw_sample[1][bin_numbers_ch1], s=80, c=colors_ch1, label='ch1')

        chrom_labels = [f'{i}' for i in range(1, 24)]
        chrom_labels[-1] = ''

        # set the xticks to be the chromosome numbers:
        ax1.set_xticks(self.chrom_locations)
        ax1.set_xticklabels(chrom_labels)
        ax2.set_xticks(self.chrom_locations)
        ax2.set_xticklabels(chrom_labels)

        # rotate the xticks by 90 degrees:
        ax1.tick_params(axis='x', rotation=90)
        ax2.tick_params(axis='x', rotation=90)

        # set the x label to be the chromosome number:
        ax1.set_xlabel('Chromosome')
        ax2.set_xlabel('Chromosome')

        # set the title according to kwargs:
        if 'title' in kwargs:
            fig.suptitle(kwargs['title'])
        # set title for ax1 and ax2:
        ax1.set_title('Total CN', loc='left')
        ax2.set_title('Minor CN', loc='left')

        plt.savefig(save_string, bbox_inches='tight')

        # now make another plot of a table of the top 5 bins for each channel, these would look like a normal table with 
        # the bin number, chromosome, cytoband, gene, double_elite_enhancer, and attributions for each channel:
        # ---------------------------------------------------------------------------------------------------------
        # make a dataframe of the top 5 bins for each channel:

        TOP_N = 15

        top5_ch0 = self.df_ch0_metadata.sort_values(by='attributions_one_channel', ascending=False).iloc[:TOP_N]
        top5_ch1 = self.df_ch1_metadata.sort_values(by='attributions_one_channel', ascending=False).iloc[:TOP_N]

        # make a list of the top 5 bins for each channel:
        top5_ch0_bins = top5_ch0['bin_number'].tolist()
        top5_ch1_bins = top5_ch1['bin_number'].tolist()

        # make a list of the top 5 attributions for each channel:
        top5_ch0_attributions = top5_ch0['attributions_one_channel'].tolist()
        top5_ch1_attributions = top5_ch1['attributions_one_channel'].tolist()

        # normalise the attributions to be between 0 and 1, for each channel:
        top5_ch0_attributions = [i/max(top5_ch0_attributions) for i in top5_ch0_attributions]
        top5_ch1_attributions = [i/max(top5_ch1_attributions) for i in top5_ch1_attributions]

        # make a list of the top 5 cytobands for each channel:
        top5_ch0_cytobands = top5_ch0['cytoband'].tolist()
        top5_ch1_cytobands = top5_ch1['cytoband'].tolist()

        # make a list of the top 5 genes for each channel:
        top5_ch0_genes = top5_ch0['gene'].tolist()
        top5_ch1_genes = top5_ch1['gene'].tolist()

        # make a list of the top 5 double_elite_enhancers for each channel:
        top5_ch0_double_elite_enhancers = top5_ch0['double_elite_enhancer'].tolist()
        top5_ch1_double_elite_enhancers = top5_ch1['double_elite_enhancer'].tolist()

        def remove_nan(list):
            for i in range(len(list)):
                if str(list[i]) == 'nan':
                    list[i] = ''
            return list
        
        # remove the nan values from the lists:
        top5_ch0_cytobands = remove_nan(top5_ch0_cytobands)
        top5_ch1_cytobands = remove_nan(top5_ch1_cytobands)
        top5_ch0_genes = remove_nan(top5_ch0_genes)
        top5_ch1_genes = remove_nan(top5_ch1_genes)
        top5_ch0_double_elite_enhancers = remove_nan(top5_ch0_double_elite_enhancers)
        top5_ch1_double_elite_enhancers = remove_nan(top5_ch1_double_elite_enhancers)
        
        # now plot the table:
        # -------------------
        fig1 = plt.figure(figsize=(4.5,10))#figsize=(size_x/mydpi, size_y/mydpi), dpi=mydpi)
        ax1 = fig1.add_subplot(111)

        sns.set_context('paper')
        # colour the columns according to the attribution value for that bin:
        # -------------------------------------------------------------------
        # make a list of colours for the attributions, using the Blues colormap, but map them to the intensity of the attributions:
        cmap = matplotlib.cm.Blues
        # get the colours for the attributions:
        colours_ch0 = cmap(top5_ch0_attributions)
        colours_ch1 = cmap(top5_ch1_attributions)

        # convert the colours to hex:
        colours_ch0 = [matplotlib.colors.rgb2hex(colour) for colour in colours_ch0]
        colours_ch1 = [matplotlib.colors.rgb2hex(colour) for colour in colours_ch1]

        # make the colours into a list of lists, so that it can be used in the table, making it a list of 5 identical lists:
        colours_ch0 = [colours_ch0 for i in range(4)]
        colours_ch1 = [colours_ch1 for i in range(4)]

        # Function to handle text splitting and wrapping
        def wrap_and_split(text):
            parts = text.split(',')
            # strip parts from spaces:
            parts = [part.strip() for part in parts]
            return '\n'.join(parts)
        
        # make a 4x5 dataframe with the attribution values for each bin:
        df_ch0_attributions = pd.DataFrame([top5_ch0_attributions]*4)
        # transpose the dataframe:
        df_ch0_attributions = df_ch0_attributions.T
        # make columns [bin number, cytoband, gene, double_elite_enhancer]:
        df_ch0_attributions.columns = ['Bin number', 'Cytoband', 'Gene(s)', 'Enhancer(s)']
        # make an annotation dataframe from [top5_ch0_bins, top5_ch0_cytobands, top5_ch0_genes, top5_ch0_double_elite_enhancers]:
        df_ch0_annot = pd.DataFrame([top5_ch0_bins, top5_ch0_cytobands, top5_ch0_genes, top5_ch0_double_elite_enhancers])
        # transpose the dataframe:
        df_ch0_annot = df_ch0_annot.T
        df_ch0_annot.columns = ['Bin number', 'Cytoband', 'Gene(s)', 'Enhancer(s)']
        # make a heatmap of the df_ch0_attributions, but annotate the cells according to the df_ch0_annot:

        ax1 = sns.heatmap(df_ch0_attributions, 
                          annot=df_ch0_annot, 
                          fmt='', 
                          cmap='Blues', 
                          linewidths=2,
                          cbar=False,#True, 
                          xticklabels=1, 
                          yticklabels=False, 
                          annot_kws={'size': 8}, 
                        #   cbar_kws={'label': 'Normalised attribution'}
                          )#, 'orientation': 'vertical', 'shrink': 0.5})
        ax1.xaxis.tick_top() # x axis on top
        ax1.xaxis.set_label_position('top')

        # take the text in the annotation and use the wrap and split function to split the text into two lines:
        for text in ax1.texts:
            text.set_text(wrap_and_split(text.get_text()))

        # save the figure:
        plt.savefig(f'{save_string}.ch0_table.pdf', bbox_inches='tight')

        #######################################3
        # now plot the table:
        # -------------------
        # size_x, size_y = 640, 480
        # mydpi = 500
        fig2 = plt.figure(figsize=(4.5,10))#figsize=(size_x/mydpi, size_y/mydpi), dpi=mydpi)
        # fig2= plt.figure()#figsize=(5, 5))
        ax2 = fig2.add_subplot(111)

        sns.set_context('paper')
        
        # make a 4x5 dataframe with the attribution values for each bin:
        df_ch1_attributions = pd.DataFrame([top5_ch1_attributions]*4)
        # transpose the dataframe:
        df_ch1_attributions = df_ch1_attributions.T


        # make columns [bin number, cytoband, gene, double_elite_enhancer]:
        df_ch1_attributions.columns = ['Bin number', 'Cytoband', 'Gene(s)', 'Enhancer(s)']
        # make an annotation dataframe from [top5_ch0_bins, top5_ch0_cytobands, top5_ch0_genes, top5_ch0_double_elite_enhancers]:
        df_ch1_annot = pd.DataFrame([top5_ch1_bins, top5_ch1_cytobands, top5_ch1_genes, top5_ch1_double_elite_enhancers])
        # transpose the dataframe:
        df_ch1_annot = df_ch1_annot.T
        df_ch1_annot.columns = ['Bin number', 'Cytoband', 'Gene(s)', 'Enhancer(s)']

        # make a heatmap of the df_ch0_attributions, but annotate the cells according to the df_ch0_annot:
        ax2 = sns.heatmap(df_ch1_attributions, 
                          annot=df_ch1_annot, 
                          fmt='', 
                          cmap='Blues', 
                          linewidths=2,
                          cbar=False,#True, 
                          xticklabels=1, 
                          yticklabels=False, 
                          annot_kws={'size': 8}, 
                        #   cbar_kws={'label': 'Normalised attribution'}
                          )#, 'orientation': 'vertical', 'shrink': 0.5})
        ax2.xaxis.tick_top() # x axis on top
        ax2.xaxis.set_label_position('top')


        # take the text in the annotation and use the wrap and split function to split the text into two lines:
        for text in ax2.texts:
            text.set_text(wrap_and_split(text.get_text()))
            


        # save the figure:
        plt.savefig(f'{save_string}.ch1_table.pdf', bbox_inches='tight')
   
    def plot_OG_sample_mosaic(self, sample_index, cancer_type, save_string = None, save_fmt='png', **kwargs):

        sns.set_style('whitegrid')
        # sns.set_context('talk')
        sns.set_context('paper')

        raw_sample = np.load(f'../GEL_NPY/matrix_{int(sample_index)}.npy')


        raw_squeeze = raw_sample.squeeze() 
        tot_raw = np.array(raw_squeeze[:28749])
        maj_raw = np.array(raw_squeeze[28749:])
        ch0_raw = tot_raw.copy()
        ch1_raw = tot_raw - maj_raw

        # make an empty numpy array of shape (2, 28749):
        raw_sample = np.empty((2, 28749))
        raw_sample[0] = ch0_raw
        raw_sample[1] = ch1_raw

        # make a subplot of two figures, one for ch0 and one for ch1:
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

        # make a mosaic plot:
        figure_mosaic = """
        AABC
        AABC
        DDBC
        DDBC
        """

        fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(28, 10))

        # keep a log of the copy state (by classification, Diploid, LOH, ... for the table further down)
        def interpret_state (major_al, minor_al):
            tot = major_al + minor_al

            if major_al == 0 and minor_al == 0:
                return 'Deletion'
            elif major_al == 1 and minor_al == 1:
                return 'Diploid'
            elif ((minor_al == 0 or major_al == 0) and tot < 3):
            # elif ((minor_al == 0 or major_al == 0) and tot != 0):
                return 'LOH'
            elif tot >= 3 and tot < 5:
                return 'Duplication'
            elif tot >= 5:
                return 'Amplification'
            else:
                return f'{str(major_al)}/{str(minor_al)}'
        
        # make a list of the copy states for each bin:
        copy_states = np.array([interpret_state(maj_raw[i], ch1_raw[i]) for i in range(len(maj_raw))])

        # make the dots very faint and to have an edgecolor of black:
        axes["A"].scatter(range(len(raw_sample[0])), raw_sample[0], s=25, c='gray', alpha=0.2, label='ch0')#, edgecolors='black')
        axes["D"].scatter(range(len(raw_sample[1])), raw_sample[1], s=25, c='gray', alpha=0.2, label='ch1')#, edgecolors='black')
        # plt.show()

        # use the bin_number column from self.df_ch0_metadata to get the bin numbers for the high attribute bins, colour these in and make s=50:
        # get the bin numbers for the high attribute bins:
        bin_numbers_ch0 = self.df_ch0_metadata['bin_number'].tolist()
        bin_numbers_ch1 = self.df_ch1_metadata['bin_number'].tolist()

        # get the attributions for the high attribute bins:
        attributions_ch0 = self.df_ch0_metadata['attributions_one_channel'].tolist()
        attributions_ch1 = self.df_ch1_metadata['attributions_one_channel'].tolist()

        # make own palette for 22 chromosomes:
        hex_colours = [matplotlib.colors.rgb2hex(colour) for colour in cm.tab20.colors]
        hex_colours = hex_colours + ['#06C2AC', '#580F41']

        chrom_colours = {i: hex_colours[i-1] for i in range(1, 23)}

        # chrom_to_colour = lambda x: chrom_colours[x]

        # get the colours for the high attribute bins:
        colors_ch0 = self.df_ch0_metadata['chromosome'].apply(lambda x: chrom_colours[x]).tolist()
        colors_ch1 = self.df_ch1_metadata['chromosome'].apply(lambda x: chrom_colours[x]).tolist()

        # plot the high attribute bins:
        axes["A"].scatter(bin_numbers_ch0, raw_sample[0][bin_numbers_ch0], s=80, c=colors_ch0, label='ch0')
        axes["D"].scatter(bin_numbers_ch1, raw_sample[1][bin_numbers_ch1], s=80, c=colors_ch1, label='ch1')

        chrom_labels = [f'{i}' for i in range(1, 24)]
        chrom_labels[-1] = ''

        # set the xticks to be the chromosome numbers:
        axes["A"].set_xticks(self.chrom_locations)
        # axes["A"].set_xticklabels(chrom_labels)

        # remove x axis ticks and labels for axes["A"]:
        # axes["A"].set_xticks([])
        axes["A"].set_xticklabels([])


        axes["D"].set_xticks(self.chrom_locations)
        axes["D"].set_xticklabels(chrom_labels)

        # rotate the xticks by 90 degrees:
        # axes["A"].tick_params(axis='x', rotation=90)
        axes["D"].tick_params(axis='x', rotation=90)

        # set the x label to be the chromosome number:
        # axes["A"].set_xlabel('Chromosome')
        axes["D"].set_xlabel('Chromosome')


        # set xlim for axes["A"] and axes["D"]:
        axes["A"].set_xlim(-100, 28749+100)
        axes["D"].set_xlim(-100, 28749+100)

        # label the y axis with "Copy State":
        axes["A"].set_ylabel('Copy State')
        axes["D"].set_ylabel('Copy State')

        # set the title according to kwargs:
        if 'title' in kwargs:
            fig.suptitle(kwargs['title'])
        # set title for ax1 and ax2:
        # axes["A"].set_title('Total CN', loc='left')
        # axes["D"].set_title('Minor CN', loc='left')

        # plt.savefig(f'analysis_MAB_v1/sample/tests/{cancer_type}_OG_sample_{sample_index}.png', bbox_inches='tight')

        # now make another plot of a table of the top 5 bins for each channel, these would look like a normal table with 
        # the bin number, chromosome, cytoband, gene, double_elite_enhancer, and attributions for each channel:
        # ---------------------------------------------------------------------------------------------------------
        # make a dataframe of the top 5 bins for each channel:

        TOP_N = 15

        top5_ch0 = self.df_ch0_metadata.sort_values(by='attributions_one_channel', ascending=False).iloc[:TOP_N]
        top5_ch1 = self.df_ch1_metadata.sort_values(by='attributions_one_channel', ascending=False).iloc[:TOP_N]

        # make a list of the top 5 bins for each channel:
        top5_ch0_bins = top5_ch0['bin_number'].tolist()
        top5_ch1_bins = top5_ch1['bin_number'].tolist()

        # make a list of the top 5 attributions for each channel:
        top5_ch0_attributions = top5_ch0['attributions_one_channel'].tolist()
        top5_ch1_attributions = top5_ch1['attributions_one_channel'].tolist()

        # normalise the attributions to be between 0 and 1, for each channel:
        top5_ch0_attributions = [i/max(top5_ch0_attributions) for i in top5_ch0_attributions]
        top5_ch1_attributions = [i/max(top5_ch1_attributions) for i in top5_ch1_attributions]

        # make a list of the top 5 cytobands for each channel:
        top5_ch0_cytobands = top5_ch0['cytoband'].tolist()
        top5_ch1_cytobands = top5_ch1['cytoband'].tolist()

        # make a list of the top 5 genes for each channel:
        top5_ch0_genes = top5_ch0['gene'].tolist()
        top5_ch1_genes = top5_ch1['gene'].tolist()

        # make a list of the top 5 double_elite_enhancers for each channel:
        top5_ch0_double_elite_enhancers = top5_ch0['double_elite_enhancer'].tolist()
        top5_ch1_double_elite_enhancers = top5_ch1['double_elite_enhancer'].tolist()

        def remove_nan(list):
            for i in range(len(list)):
                if str(list[i]) == 'nan':
                    list[i] = ''
            return list
        
        # remove the nan values from the lists:
        top5_ch0_cytobands = remove_nan(top5_ch0_cytobands)
        top5_ch1_cytobands = remove_nan(top5_ch1_cytobands)
        top5_ch0_genes = remove_nan(top5_ch0_genes)
        top5_ch1_genes = remove_nan(top5_ch1_genes)
        top5_ch0_double_elite_enhancers = remove_nan(top5_ch0_double_elite_enhancers)
        top5_ch1_double_elite_enhancers = remove_nan(top5_ch1_double_elite_enhancers)
        
        # make a function that makes enhancer names shorter, those that start with 'ENS' have many trailing zeros, so turn 000 into (0x3):
        def shorten_enhancer_name(enhancer_name):
            if enhancer_name.startswith('ENS'):
                # remove all '0' from the string:
                enhancer_name = enhancer_name.replace('0', '')
            return enhancer_name
        
        # apply the function to the enhancer names:
        top5_ch0_double_elite_enhancers = [shorten_enhancer_name(enhancer_name) for enhancer_name in top5_ch0_double_elite_enhancers]
        top5_ch1_double_elite_enhancers = [shorten_enhancer_name(enhancer_name) for enhancer_name in top5_ch1_double_elite_enhancers]

        # now plot the table:
        # -------------------
        # fig1 = plt.figure(figsize=(4.5,10))#figsize=(size_x/mydpi, size_y/mydpi), dpi=mydpi)
        # ax1 = fig1.add_subplot(111)

        sns.set_context('paper')
        # colour the columns according to the attribution value for that bin:
        # -------------------------------------------------------------------
        # make a list of colours for the attributions, using the Blues colormap, but map them to the intensity of the attributions:
        cmap = matplotlib.cm.Blues
        # get the colours for the attributions:
        colours_ch0 = cmap(top5_ch0_attributions)
        colours_ch1 = cmap(top5_ch1_attributions)

        # convert the colours to hex:
        colours_ch0 = [matplotlib.colors.rgb2hex(colour) for colour in colours_ch0]
        colours_ch1 = [matplotlib.colors.rgb2hex(colour) for colour in colours_ch1]

        # make the colours into a list of lists, so that it can be used in the table, making it a list of 5 identical lists:
        colours_ch0 = [colours_ch0 for i in range(4)]
        colours_ch1 = [colours_ch1 for i in range(4)]

        # Function to handle text splitting and wrapping
        def wrap_and_split(text):
            parts = text.split(',')
            # strip parts from spaces:
            parts = [part.strip() for part in parts]
            return '\n'.join(parts)
        
        # make a 4x5 dataframe with the attribution values for each bin:
        df_ch0_attributions = pd.DataFrame([top5_ch0_attributions]*4)
        # transpose the dataframe:
        df_ch0_attributions = df_ch0_attributions.T
        # make columns [bin number, cytoband, gene, double_elite_enhancer]:
        # df_ch0_attributions.columns = ['Bin number', 'Cytoband', 'Gene(s)', 'Enhancer(s)']
        df_ch0_attributions.columns = ['Copy state', 'Cytoband', 'Gene(s)', 'Enhancer(s)']

        # make an annotation dataframe from [top5_ch0_bins, top5_ch0_cytobands, top5_ch0_genes, top5_ch0_double_elite_enhancers]:
        # instead of bin number, get the copy state for each bin for top5_ch0_bins:
        top5_ch0_copy_states = [copy_states[i] for i in top5_ch0_bins]
        df_ch0_annot = pd.DataFrame([top5_ch0_copy_states, top5_ch0_cytobands, top5_ch0_genes, top5_ch0_double_elite_enhancers])
        # df_ch0_annot = pd.DataFrame([top5_ch0_bins, top5_ch0_cytobands, top5_ch0_genes, top5_ch0_double_elite_enhancers])

        # transpose the dataframe:
        df_ch0_annot = df_ch0_annot.T
        # df_ch0_annot.columns = ['Bin number', 'Cytoband', 'Gene(s)', 'Enhancer(s)']
        df_ch0_annot.columns = ['Copy state', 'Cytoband', 'Gene(s)', 'Enhancer(s)']

        # make a heatmap of the df_ch0_attributions, but annotate the cells according to the df_ch0_annot:

        sns.heatmap(df_ch0_attributions, 
                          annot=df_ch0_annot, 
                          fmt='', 
                          cmap='Blues', 
                          linewidths=2,
                          cbar=False,#True, 
                          xticklabels=1, 
                          yticklabels=False, 
                          annot_kws={'size': 8}, 
                          ax = axes["B"]
                        #   cbar_kws={'label': 'Normalised attribution'}
                          )#, 'orientation': 'vertical', 'shrink': 0.5})
        axes["B"].xaxis.tick_top() # x axis on top
        axes["B"].xaxis.set_label_position('top')

        # take the text in the annotation and use the wrap and split function to split the text into two lines:
        for text in axes["B"].texts:
            text.set_text(wrap_and_split(text.get_text()))

        # save the figure:
        # plt.savefig(f'analysis_MAB_v1/sample/tests2/{cancer_type}_OG_sample_{sample_index}_ch0_table.pdf', bbox_inches='tight')

        #######################################3
        # now plot the table:
        # -------------------
        # size_x, size_y = 640, 480
        # mydpi = 500
        # fig2 = plt.figure(figsize=(4.5,10))#figsize=(size_x/mydpi, size_y/mydpi), dpi=mydpi)
        # # fig2= plt.figure()#figsize=(5, 5))
        # ax2 = fig2.add_subplot(111)

        sns.set_context('paper')
        
        # make a 4x5 dataframe with the attribution values for each bin:
        df_ch1_attributions = pd.DataFrame([top5_ch1_attributions]*4)
        # transpose the dataframe:
        df_ch1_attributions = df_ch1_attributions.T


        # make columns [bin number, cytoband, gene, double_elite_enhancer]:
        df_ch1_attributions.columns = ['Copy state', 'Cytoband', 'Gene(s)', 'Enhancer(s)']
        # df_ch1_attributions.columns = ['Bin number', 'Cytoband', 'Gene(s)', 'Enhancer(s)']

        # make an annotation dataframe from [top5_ch0_bins, top5_ch0_cytobands, top5_ch0_genes, top5_ch0_double_elite_enhancers]:
        top5_ch1_copy_states = [copy_states[i] for i in top5_ch1_bins]
        # df_ch1_annot = pd.DataFrame([top5_ch1_bins, top5_ch1_cytobands, top5_ch1_genes, top5_ch1_double_elite_enhancers])
        df_ch1_annot = pd.DataFrame([top5_ch1_copy_states, top5_ch1_cytobands, top5_ch1_genes, top5_ch1_double_elite_enhancers])

        # transpose the dataframe:
        df_ch1_annot = df_ch1_annot.T
        df_ch1_annot.columns = ['Copy state', 'Cytoband', 'Gene(s)', 'Enhancer(s)']
        # df_ch1_annot.columns = ['Bin number', 'Cytoband', 'Gene(s)', 'Enhancer(s)']

        # make a heatmap of the df_ch0_attributions, but annotate the cells according to the df_ch0_annot:
        sns.heatmap(df_ch1_attributions, 
                          annot=df_ch1_annot, 
                          fmt='', 
                          cmap='Blues', 
                          linewidths=2,
                          cbar=False,#True, 
                          xticklabels=1, 
                          yticklabels=False, 
                          ax = axes["C"],
                          annot_kws={'size': 7}, 
                        #   cbar_kws={'label': 'Normalised attribution'}
                          )#, 'orientation': 'vertical', 'shrink': 0.5})
        axes["C"].xaxis.tick_top() # x axis on top
        axes["C"].xaxis.set_label_position('top')


        # take the text in the annotation and use the wrap and split function to split the text into two lines:
        for text in axes["C"].texts:
            text.set_text(wrap_and_split(text.get_text()))
            
        plt.tight_layout()

        # save the figure:
        plt.savefig(save_string, bbox_inches='tight')

class CNVCohortAnalysis:
    # this will be where cohort specific analysis will be performed.

    def __init__(self, GIG_dict, disease='LUNG'):
        self.GIG_dict = GIG_dict # dict of all GIG
        self.disease = disease

        self.nsamples = len(self.GIG_dict[disease])

        # set up arrays
        attributions_gig_n = np.zeros( [self.nsamples, self.GIG_dict[disease][0].shape[1], self.GIG_dict[disease][0].shape[2]] )
        attributions_dir = np.zeros_like(attributions_gig_n)
        attributions_rank = np.zeros_like(attributions_gig_n)
        self.attributions_gig_maxnorm = np.zeros_like(attributions_gig_n)
        self.attributions_gig_norm = np.zeros_like(attributions_gig_n)

        # Normalisation
        # process each channel such that the attributes are rankings. Keep a record of the direction of the attribute (+ve or -ve)
        for i in range(self.nsamples):
            samp_array = self.GIG_dict[disease][i].squeeze().numpy()

            self.ch0_attr = samp_array[0].copy()
            self.ch1_attr = samp_array[1].copy()

            sign_ch0 = np.sign(self.ch0_attr)
            sign_ch1 = np.sign(self.ch1_attr)

            self.ch0_attr = np.abs ( self.ch0_attr ) / np.sum(np.abs(self.ch0_attr))
            self.ch1_attr = np.abs ( self.ch1_attr ) / np.sum(np.abs(self.ch1_attr))

            # make an assertion that the sum is close to 1:
            assert np.isclose(np.sum(self.ch0_attr), 1), f'sum(ch0 attributions) is not close to 1: {np.sum(self.ch0_attr)}'
            assert np.isclose(np.sum(self.ch1_attr), 1), f'sum(ch1 attributions) is not close to 1: {np.sum(self.ch1_attr)}'

            self.ch0_attr = self.ch0_attr * sign_ch0
            self.ch1_attr = self.ch1_attr * sign_ch1

            # put back into the array of all samples: 
            self.attributions_gig_norm[i, 0] = self.ch0_attr
            self.attributions_gig_norm[i, 1] = self.ch1_attr

            self.vals_ch0 = self.attributions_gig_norm[:,0,:]
            self.vals_ch1 = self.attributions_gig_norm[:,1,:]


    def thresholding(self, threshold_percentile=99.):
        # Take the attributions that have been normalised:
        # threshold them at the 99th percentile
        # loop over the samples:

        self.thresh_vals_ch0 = np.zeros_like(self.vals_ch0)
        self.thresh_vals_ch1 = np.zeros_like(self.vals_ch1)

        for i in range(self.vals_ch0.shape[0]):
            # get the threshold for that sample:


            threshold_ch0 = np.percentile(self.vals_ch0[i], threshold_percentile)
            threshold_ch1 = np.percentile(self.vals_ch1[i], threshold_percentile)

            # threshold the attributions, set values below the threshold to 0:
            self.thresh_vals_ch0[i] = np.where(self.vals_ch0[i] > threshold_ch0, self.vals_ch0[i], 0)
            self.thresh_vals_ch1[i] = np.where(self.vals_ch1[i] > threshold_ch1, self.vals_ch1[i], 0)

            # binarise the thresholded attributions:
            self.thresh_vals_ch0[i] = np.where(self.thresh_vals_ch0[i] > 0, 1, 0)
            self.thresh_vals_ch1[i] = np.where(self.thresh_vals_ch1[i] > 0, 1, 0)

        # sum them all to get a count of how many times a bin was 1 across all samples:
        self.sum_thresh_vals_ch0 = np.sum(self.thresh_vals_ch0, axis=0)
        self.sum_thresh_vals_ch1 = np.sum(self.thresh_vals_ch1, axis=0)

        # using this sum, only show bins that are present in more than 5% of samples:
        self.sum_thresh_vals_ch0 = np.where(self.sum_thresh_vals_ch0 > (self.nsamples * 0.05), self.sum_thresh_vals_ch0, 0)
        self.sum_thresh_vals_ch1 = np.where(self.sum_thresh_vals_ch1 > (self.nsamples * 0.05), self.sum_thresh_vals_ch1, 0)

        # # from self.sum_thresh_vals_ch0, remove all values that and 1 and less:
        # # i.e. it has to be present in more than one sample:
        # self.sum_thresh_vals_ch0 = np.where(self.sum_thresh_vals_ch0 > 1, self.sum_thresh_vals_ch0, 0)
        # self.sum_thresh_vals_ch1 = np.where(self.sum_thresh_vals_ch1 > 1, self.sum_thresh_vals_ch1, 0)

    def manhattan(self, metadata_filepath='../annotation/all_bins_metadata.csv', save_string=None, save_fmt='png', label=True, **kwargs):
        # The Manhattan plot will plot the sum of self.thresh_vals_ch0, self.thresh_vals_ch1 for each bin, for each sample, as a subplots of the total CN and minor CN channels.

        # read in the metadata:
        df_metadata = pd.read_csv(metadata_filepath)

        sns.set_style('white')
        sns.set_context('paper')

        # make a 1x2 grid of subplots:
        fig, ax = plt.subplots(2, 1)#, figsize=(20, 10))

        # make a dataframe of crc_sample_ch0 and crc_sample_ch1 attributions, using the index as the bin_number:
        df_ch0 = pd.DataFrame(self.sum_thresh_vals_ch0/self.nsamples, columns=['attributions_one_channel'])
        # if you want count instead of fraction:
        # df_ch0 = pd.DataFrame(self.sum_thresh_vals_ch0, columns=['attributions_one_channel'])

        df_ch0['bin_number'] = df_ch0.index

        df_ch1 = pd.DataFrame(self.sum_thresh_vals_ch1/self.nsamples, columns=['attributions_one_channel'])
        # if you want count instead of fraction:
        # df_ch1 = pd.DataFrame(self.sum_thresh_vals_ch1, columns=['attributions_one_channel'])

        df_ch1['bin_number'] = df_ch1.index

        # add the metadata, fusing on the bin_number:
        df_ch0_metadata = pd.merge(df_ch0, df_metadata, on='bin_number')

        # keep a copy of the full metadata for this sample:
        self.df_full = df_ch0_metadata.copy()

        # remove all bins that have no real hits:
        df_ch0_metadata = df_ch0_metadata[df_ch0_metadata['attributions_one_channel'] != 0]
        # -----------------------

        # add the metadata, fusing on the bin_number:
        df_ch1_metadata = pd.merge(df_ch1, df_metadata, on='bin_number')

        # remove all bins that have no real hits:
        df_ch1_metadata = df_ch1_metadata[df_ch1_metadata['attributions_one_channel'] != 0]
        # -----------------------

        # make own palette for 22 chromosomes:
        hex_colours = [matplotlib.colors.rgb2hex(colour) for colour in cm.tab20.colors]
        hex_colours = hex_colours + ['#06C2AC', '#580F41']

        chrom_colours = {i: hex_colours[i-1] for i in range(1, 23)}

        chrom_to_colour = lambda x: chrom_colours[x]
        chromosome_colours_list_ch0 = df_ch0_metadata['chromosome'].apply(chrom_to_colour).tolist()
        cmap_ch0 = matplotlib.colors.ListedColormap(chromosome_colours_list_ch0)

        chromosome_colours_list_ch1 = df_ch1_metadata['chromosome'].apply(chrom_to_colour).tolist()
        cmap_ch1 = matplotlib.colors.ListedColormap(chromosome_colours_list_ch1)

        # plt.figure(figsize=(20, 5))
        # make a 1x2 grid of subplots:
        fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
        # plt.scatter(self.df_ch0_metadata['bin_number'], self.df_ch0_metadata['attributions_one_channel'], c=chromosome_colours_list, cmap=cmap)
        ax[0].scatter(df_ch0_metadata['bin_number'], df_ch0_metadata['attributions_one_channel'], c=chromosome_colours_list_ch0, cmap=cmap_ch0, s=7.5)
        ax[1].scatter(df_ch1_metadata['bin_number'], df_ch1_metadata['attributions_one_channel'], c=chromosome_colours_list_ch1, cmap=cmap_ch1, s=7.5)

        if label:
            # Initialize a set to keep track of the strings that have been annotated
            annotated_strings = set()
            
            for i, row in df_ch0_metadata.iterrows():
                # Check if the point has any individual label (gene, double_elite_enhancer, cytoband)
                label_text = None
                if pd.notna(row['gene']):
                    label_text = row['gene']
                # elif pd.notna(row['double_elite_enhancer']):
                #     label_text = row['double_elite_enhancer']
                # elif pd.notna(row['cytoband']):
                #     label_text = row['cytoband']
                
                # Annotate the individual data point with its label if it's not a duplicate
                if label_text is not None:# and label_text not in annotated_strings:
                #     # only annotate two labels (highest attribution) per chromosome:
                #     # --------------------------------------------------------------    

                #     # first get the chromosome number:
                #     chrom_num = row['chromosome']
                #     # get the two highest attributions for that chromosome:
                #     highest_attributions = df_ch0_metadata[df_ch0_metadata['chromosome'] == chrom_num].sort_values(by='attributions_one_channel', ascending=False).iloc[:2]
                #     # check if the current row is one of the two highest attributions:
                #     if row['bin_number'] in highest_attributions['bin_number'].tolist():
                #         ax[0].annotate(label_text, (row['bin_number'], row['attributions_one_channel']),
                #                        fontweight='bold', 
                #                        color = 'white',
                #                        fontsize=8, 
                #                        zorder=10, 
                #                     #    ha='center', 
                #                        bbox=dict(facecolor='k', edgecolor='k', boxstyle='round,pad=0.1')
                #                        )
                #         annotated_strings.add(label_text)
                    
                    # IF YOU WANT ALL LABELS:
                    # -----------------------
                    ax[0].annotate(label_text, (row['bin_number'], row['attributions_one_channel']))
                    annotated_strings.add(label_text)

            for i, row in df_ch1_metadata.iterrows():
                # Check if the point has any individual label (gene, double_elite_enhancer, cytoband)
                label_text = None
                if pd.notna(row['gene']):
                    label_text = row['gene']
                # elif pd.notna(row['double_elite_enhancer']):
                #     label_text = row['double_elite_enhancer']
                # elif pd.notna(row['cytoband']):
                #     label_text = row['cytoband']
                
                # Annotate the individual data point with its label if it's not a duplicate
                if label_text is not None:# and label_text not in annotated_strings:

                #    # only annotate two labels (highest attribution) per chromosome:
                #    # --------------------------------------------------------------    
                #     # first get the chromosome number:
                #     chrom_num = row['chromosome']
                #     # get the two highest attributions for that chromosome:
                #     highest_attributions = df_ch1_metadata[df_ch1_metadata['chromosome'] == chrom_num].sort_values(by='attributions_one_channel', ascending=False).iloc[:2]
                #     # check if the current row is one of the two highest attributions:
                #     if row['bin_number'] in highest_attributions['bin_number'].tolist():
                #         ax[1].annotate(label_text, (row['bin_number'], row['attributions_one_channel']),
                #                     fontweight='bold', 
                #                     color = 'white',
                #                     fontsize=8, 
                #                     zorder=10, 
                #                     # ha='center', 
                #                     bbox=dict(facecolor='k', edgecolor='k', boxstyle='round,pad=0.1')
                #                     )
                #         annotated_strings.add(label_text)

                    # IF YOU WANT ALL LABELS:
                    # -----------------------
                    ax[1].annotate(label_text, (row['bin_number'], row['attributions_one_channel']))
                    annotated_strings.add(label_text)


        # Calculate the midpoint of bin_number for each chromosome
        chromosome_bins = self.df_full.groupby('chromosome')['bin_number'].mean()
        # Define chromosome labels from 1 to 22
        chromosome_labels = [f'{i}' for i in range(1, 23)]

        # make sure the xlim is 0, length of the genome:
        ax[0].set_xlim(0, self.df_full['bin_number'].max())
        ax[1].set_xlim(0, self.df_full['bin_number'].max())
        # Set the x-axis tick positions and labels
        ax[0].set_xticks(chromosome_bins)
        ax[0].set_xticklabels(chromosome_labels)
        # Set the x-axis tick positions and labels
        ax[1].set_xticks(chromosome_bins)
        ax[1].set_xticklabels(chromosome_labels)
        # # rotate the x tick labels by 90 degrees:
        # ax[0].tick_params(axis='x', rotation=90)
        # ax[1].tick_params(axis='x', rotation=90)

        # ax[0].set_title('ch0', loc='left')
        # ax[1].set_title('ch1', loc='left')

        # make the y-lim equal for both subplots:
        ax[0].set_ylim(0, 1)
        ax[1].set_ylim(0, 1)

        # read the title & axes labels from the kwargs:
        if 'title' in kwargs:
            fig.suptitle(kwargs['title'])

        if 'xlabel' in kwargs:
            # ax[0].set_xlabel(kwargs['xlabel'])
            ax[1].set_xlabel(kwargs['xlabel'])

        if 'ylabel' in kwargs:
            ax[0].set_ylabel(kwargs['ylabel'])
            ax[1].set_ylabel(kwargs['ylabel'])
        
        plt.tight_layout()
        
        if save_string:
            plt.savefig(save_string, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close('all')


    def attributes_csv(self, metadata_filepath='../annotation/all_bins_metadata.csv'):

        # read in the metadata:
        df_metadata = pd.read_csv(metadata_filepath)

        # make a dataframe of crc_sample_ch0 and crc_sample_ch1 attributions, using the index as the bin_number:
        df_ch0 = pd.DataFrame(self.sum_thresh_vals_ch0/self.nsamples, columns=['fraction_cohort_one_channel'])
        # if you want count instead of fraction:
        # df_ch0 = pd.DataFrame(self.sum_thresh_vals_ch0, columns=['attributions_one_channel'])

        df_ch0['bin_number'] = df_ch0.index

        df_ch1 = pd.DataFrame(self.sum_thresh_vals_ch1/self.nsamples, columns=['fraction_cohort_one_channel'])
        # if you want count instead of fraction:
        # df_ch1 = pd.DataFrame(self.sum_thresh_vals_ch1, columns=['attributions_one_channel'])

        df_ch1['bin_number'] = df_ch1.index

        # add the metadata, fusing on the bin_number:
        df_ch0_metadata = pd.merge(df_ch0, df_metadata, on='bin_number')

        # keep a copy of the full metadata for this sample:
        self.df_full = df_ch0_metadata.copy()

        # remove all bins that have no real hits:
        df_ch0_metadata = df_ch0_metadata[df_ch0_metadata['fraction_cohort_one_channel'] != 0]
        # -----------------------

        # add the metadata, fusing on the bin_number:
        df_ch1_metadata = pd.merge(df_ch1, df_metadata, on='bin_number')

        # remove all bins that have no real hits:
        df_ch1_metadata = df_ch1_metadata[df_ch1_metadata['fraction_cohort_one_channel'] != 0]
        # -----------------------



        return df_ch0_metadata, df_ch1_metadata

    def heatmaps(self, save_string=None, save_fmt='png', clustermap=True, return_raw=False):

        sns.set_context('paper')
        nsamples = len(self.GIG_dict[self.disease])

        # # set up arrays
        # attributions_gig_n = np.zeros( [nsamples, self.GIG_dict[disease][0].shape[1], self.GIG_dict[disease][0].shape[2]] )
        # attributions_dir = np.zeros_like(attributions_gig_n)
        # attributions_rank = np.zeros_like(attributions_gig_n)
        # # Normalisation
        # # process each channel such that the attributes are rankings. Keep a record of the direction of the attribute (+ve or -ve)
        for i in range(nsamples):

            self.vals_ch0 = self.attributions_gig_norm[:,0,:]
            self.vals_ch1 = self.attributions_gig_norm[:,1,:]

            # sample-wise max normalisation for each channel:
            self.attributions_gig_maxnorm[i,0,:] = self.vals_ch0[i,:] / np.max(np.abs(self.vals_ch0[i,:]))
            self.attributions_gig_maxnorm[i,1,:] = self.vals_ch1[i,:] / np.max(np.abs(self.vals_ch1[i,:]))

        # just for visualisation, do samplewise max normalisation
        # select ch0:
        vals_ch0_plotting = self.attributions_gig_maxnorm[:,0,:]
        vals_ch1_plotting = self.attributions_gig_maxnorm[:,1,:]

        if return_raw:
            return vals_ch0_plotting, vals_ch1_plotting
        
        elif not clustermap: 
            fig, ax = plt.subplots(2, 1, figsize=(20, 10))
            # plot the heatmaps, ch0 in ax[0], ch1 in ax[1]:
            sns.heatmap(vals_ch0_plotting, cmap='RdBu_r', center=0.0, xticklabels=False, yticklabels=False, ax=ax[0])
            # sns.heatmap(vals_ch0_plotting, cmap='RdBu_r', center=0.0, xticklabels=False, yticklabels=False)
            ax[0].set_xlabel('Features')
            ax[0].set_ylabel('Samples')
            ax[0].set_title('ch0', loc='left')
            
            sns.heatmap(vals_ch1_plotting, cmap='RdBu_r', center=0.0, xticklabels=False, yticklabels=False, ax=ax[1])
            # sns.heatmap(vals_ch1_plotting, cmap='RdBu_r', center=0.0, xticklabels=False, yticklabels=False)

            ax[1].set_xlabel('Features')
            ax[1].set_ylabel('Samples')
            ax[1].set_title('ch1', loc='left')
            # if save_string is not None:
            #     plt.savefig(f'{save_string}_{self.disease}.{save_fmt}', bbox_inches='tight') 
        else:
            # cluster (hierarchical) the samples using sns.clustermap:
            # sns.clustermap(vals_ch0_plotting, cmap='RdBu_r', center=0.0, xticklabels=False, yticklabels=False, figsize=(20, 10), col_cluster=False)
            
            clustergrid = sns.clustermap(vals_ch0_plotting, 
                                         cmap='plasma',#'RdBu_r', 
                                        #  center=0.0, 
                                         yticklabels=False, 
                                         cbar_kws={}, 
                                         col_cluster=False, 
                                         row_cluster=True,
                                        #  standard_scale=1, # 0 is row-wise
                                        #  z_score=0, # 0 is row-wise
                                         )
            ch0_dendogram_order = clustergrid.dendrogram_row.reordered_ind
            # remove the dendrograms:
            # clustergrid.ax_row_dendrogram.set_visible(False)
            # clustergrid.ax_col_dendrogram.set_visible(False)
            clustergrid.ax_heatmap.set_xlabel('Chromosome')
            clustergrid.ax_heatmap.set_ylabel('Samples')
            # set a title:
            clustergrid.ax_heatmap.set_title(f'ch0: hierarchical clustering of GIG attributions for {self.disease.lower()} cohort (n:{nsamples})', loc='left')
            # plt.setp(sns.clustermap(all_gigs_np_copy[:,0,:], cmap='RdBu_r', center=0.0, xticklabels=False, yticklabels=False).ax_heatmap.collections, visible=False)
            # change the size of the heatmap:
            # plt.gcf().set_size_inches(15, 2)
            # remove the colorbar:
            clustergrid.cax.set_visible(False)

            # load the df with annotations:
            df_meta = pd.read_csv('../annotation/all_bins_metadata.csv')

            # Calculate the midpoint of bin_number for each chromosome
            chromosome_bins = df_meta.groupby('chromosome')['bin_number'].mean()
            # Define chromosome labels from 1 to 22
            chromosome_labels = [f'{i}' for i in range(1, 23)]

            # find chromosome ends:
            chromosome_ends = df_meta.groupby('chromosome')['bin_number'].max()

            # make sure the xlim is 0, length of the genome:
            plt.xlim(0, df_meta['bin_number'].max())
            # Set the x-axis tick positions and labels:
            clustergrid.ax_heatmap.set_xticks(chromosome_bins)
            clustergrid.ax_heatmap.set_xticklabels(chromosome_labels)


            # draw a vertical line at the beginning and end of each chromosome:
            for i in range(1, 23):
                clustergrid.ax_heatmap.axvline(x=chromosome_ends[i], color='w', linewidth=0.5)

            # -------------------------------------------------------------------------------------

            if save_string is not None:
                plt.savefig(f'{save_string}.ch0.pdf', bbox_inches='tight', dpi=400)

            # return the values from the clustergrid:
            ch0_heatmap_values = clustergrid.data2d.values




            clustergrid = sns.clustermap(vals_ch1_plotting, 
                                         cmap='plasma',#'RdBu_r', 
                                        #  center=0.0, 
                                         yticklabels=False, 
                                         cbar_kws={}, 
                                         col_cluster=False, 
                                         row_cluster=True,
                                        #  standard_scale=1, # 0 is row-wise
                                        # z_score=0, # 0 is row-wise
                                         )
            ch1_dendogram_order = clustergrid.dendrogram_row.reordered_ind
            # remove the dendrograms:
            # clustergrid.ax_row_dendrogram.set_visible(False)
            # clustergrid.ax_col_dendrogram.set_visible(False)
            clustergrid.ax_heatmap.set_xlabel('Chromosome')
            clustergrid.ax_heatmap.set_ylabel('Samples')
            # set a title:
            clustergrid.ax_heatmap.set_title(f'ch1: hierarchical clustering of GIG attributions for {self.disease.lower()} cohort (n:{nsamples})', loc='left')
            # plt.gcf().set_size_inches(15, 2)
            # remove the colorbar:
            clustergrid.cax.set_visible(False)

            # change the xlim on this plot:
            plt.xlim(0, df_meta['bin_number'].max())
            
            # Set the x-axis tick positions and labels:
            clustergrid.ax_heatmap.set_xticks(chromosome_bins)
            clustergrid.ax_heatmap.set_xticklabels(chromosome_labels)

            # draw a vertical line at the beginning and end of each chromosome:
            for i in range(1, 23):
                clustergrid.ax_heatmap.axvline(x=chromosome_ends[i], color='w', linewidth=0.5)

            if save_string is not None:
                plt.savefig(f'{save_string}.ch1.pdf', bbox_inches='tight', dpi=400)

            # return the values from the clustergrid:
            ch1_heatmap_values = clustergrid.data2d.values


            return ch0_dendogram_order, ch1_dendogram_order, ch0_heatmap_values, ch1_heatmap_values
            plt.close('all')
                # -------------------------------------------------------------------------------------

    def circular_manhattan(self, metadata_filepath='../annotation/all_bins_metadata.csv', save_string=None, save_fmt='png', label=True, **kwargs):
        # Your existing code here

        # read in the metadata:
        df_metadata = pd.read_csv(metadata_filepath)

        sns.set_style('white')
        sns.set_context('paper')

        # make a 1x2 grid of subplots:
        fig, ax = plt.subplots(2, 1)#, figsize=(20, 10))

        # make a dataframe of crc_sample_ch0 and crc_sample_ch1 attributions, using the index as the bin_number:
        df_ch0 = pd.DataFrame(self.sum_thresh_vals_ch0/self.nsamples, columns=['attributions_one_channel'])
        # if you want count instead of fraction:
        # df_ch0 = pd.DataFrame(self.sum_thresh_vals_ch0, columns=['attributions_one_channel'])

        df_ch0['bin_number'] = df_ch0.index

        df_ch1 = pd.DataFrame(self.sum_thresh_vals_ch1/self.nsamples, columns=['attributions_one_channel'])
        # if you want count instead of fraction:
        # df_ch1 = pd.DataFrame(self.sum_thresh_vals_ch1, columns=['attributions_one_channel'])

        df_ch1['bin_number'] = df_ch1.index

        # add the metadata, fusing on the bin_number:
        df_ch0_metadata = pd.merge(df_ch0, df_metadata, on='bin_number')

        # keep a copy of the full metadata for this sample:
        self.df_full = df_ch0_metadata.copy()

        # remove all bins that have no real hits:
        df_ch0_metadata = df_ch0_metadata[df_ch0_metadata['attributions_one_channel'] != 0]
        # -----------------------

        # add the metadata, fusing on the bin_number:
        df_ch1_metadata = pd.merge(df_ch1, df_metadata, on='bin_number')

        # remove all bins that have no real hits:
        df_ch1_metadata = df_ch1_metadata[df_ch1_metadata['attributions_one_channel'] != 0]
        # -----------------------

        # make own palette for 22 chromosomes:
        hex_colours = [matplotlib.colors.rgb2hex(colour) for colour in cm.tab20.colors]
        hex_colours = hex_colours + ['#06C2AC', '#580F41']

        chrom_colours = {i: hex_colours[i-1] for i in range(1, 23)}

        chrom_to_colour = lambda x: chrom_colours[x]
        chromosome_colours_list_ch0 = df_ch0_metadata['chromosome'].apply(chrom_to_colour).tolist()
        cmap_ch0 = matplotlib.colors.ListedColormap(chromosome_colours_list_ch0)

        chromosome_colours_list_ch1 = df_ch1_metadata['chromosome'].apply(chrom_to_colour).tolist()
        cmap_ch1 = matplotlib.colors.ListedColormap(chromosome_colours_list_ch1)


# ----------------------------------------------------------------------------------------------------------------------------------
        # Calculate the midpoint of bin_number for each chromosome
        chromosome_bins = self.df_full.groupby('chromosome')['bin_number'].mean()
        # Define chromosome labels from 1 to 22
        chromosome_labels = [f'{i}' for i in range(1, 23)]

        # Calculate the angle for each bin based on the bin number
        df_ch0_metadata['angle'] = df_ch0_metadata['bin_number'] / df_ch0_metadata['bin_number'].max() * 2 * np.pi
        df_ch1_metadata['angle'] = df_ch1_metadata['bin_number'] / df_ch1_metadata['bin_number'].max() * 2 * np.pi

        # Calculate the radius for each data point based on attributions
        max_radius = 1.0  # You can adjust this value based on your data
        df_ch0_metadata['radius'] = max_radius * df_ch0_metadata['attributions_one_channel']
        df_ch1_metadata['radius'] = max_radius * df_ch1_metadata['attributions_one_channel']

        # Create two polar subplots side by side
        fig, ax = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': 'polar'})

        # Circular Manhattan plot for channel 0
        ax[0].scatter(df_ch0_metadata['angle'], df_ch0_metadata['radius'], c=chromosome_colours_list_ch0, cmap=cmap_ch0, s=7.5, label='Channel 0')
        ax[0].set_rlabel_position(-20)
        # ax[0].set_xticks(np.linspace(0, 2 * np.pi, len(chromosome_labels), endpoint=False))
        # ax[0].set_xticklabels(chromosome_labels)
        ax[0].set_rlim(0, 1)
        ax[0].legend()

        # Circular Manhattan plot for channel 1
        ax[1].scatter(df_ch1_metadata['angle'], df_ch1_metadata['radius'], c=chromosome_colours_list_ch1, cmap=cmap_ch1, s=7.5, label='Channel 1')
        ax[1].set_rlabel_position(-20)
        # ax[1].set_xticks(np.linspace(0, 2 * np.pi, len(chromosome_labels), endpoint=False))
        # ax[1].set_xticklabels(chromosome_labels)
        ax[1].legend()
        ax[1].set_rlim(0, 1)
        # Set labels and legend

        if label:
            for i, row in df_ch0_metadata.iterrows():
                label_text = None
                if pd.notna(row['gene']):
                    label_text = row['gene']
                if label_text is not None:
                    ax[0].annotate(label_text, (row['angle'], row['radius']))

            for i, row in df_ch1_metadata.iterrows():
                label_text = None
                if pd.notna(row['gene']):
                    label_text = row['gene']
                if label_text is not None:
                    ax[1].annotate(label_text, (row['angle'], row['radius']))

        # Read the title & axes labels from the kwargs
        if 'title' in kwargs:
            fig.suptitle(kwargs['title'])

        if 'xlabel' in kwargs:
            ax[0].set_xlabel(kwargs['xlabel'])
            ax[1].set_xlabel(kwargs['xlabel'])

        # if 'ylabel' in kwargs:
            # ax[0].set_ylabel(kwargs['ylabel'])
            # ax[1].set_ylabel(kwargs['ylabel'])

        if save_string:
            plt.savefig(save_string, bbox_inches='tight')
        else:
            plt.show()

        plt.close('all')





        # # Calculate the angle for each bin based on the bin number
        # df_ch0_metadata['angle'] = df_ch0_metadata['bin_number'] / df_ch0_metadata['bin_number'].max() * 2 * np.pi
        # df_ch1_metadata['angle'] = df_ch1_metadata['bin_number'] / df_ch1_metadata['bin_number'].max() * 2 * np.pi

        # # Calculate the radius for each data point based on attributions
        # max_radius = 1.0  # You can adjust this value based on your data
        # df_ch0_metadata['radius'] = max_radius * df_ch0_metadata['attributions_one_channel']
        # df_ch1_metadata['radius'] = max_radius * df_ch1_metadata['attributions_one_channel']

        # # Create a polar subplot
        # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

        # # Scatter plot for channel 0
        # ax.scatter(df_ch0_metadata['angle'], df_ch0_metadata['radius'], c=chromosome_colours_list_ch0, cmap=cmap_ch0, s=7.5, label='Channel 0')

        # # Scatter plot for channel 1
        # ax.scatter(df_ch1_metadata['angle'], df_ch1_metadata['radius'], c=chromosome_colours_list_ch1, cmap=cmap_ch1, s=7.5, label='Channel 1')

        # # Calculate the midpoint of bin_number for each chromosome
        # chromosome_bins = self.df_full.groupby('chromosome')['bin_number'].mean()
        # # Define chromosome labels from 1 to 22
        # chromosome_labels = [f'{i}' for i in range(1, 23)]

        # # Set labels and legend
        # ax.set_rlabel_position(90)
        # ax.set_xticks(np.linspace(0, 2 * np.pi, len(chromosome_labels), endpoint=False))
        # ax.set_xticklabels(chromosome_labels)
        # ax.legend()

        # if label:
        #     for i, row in df_ch0_metadata.iterrows():
        #         label_text = None
        #         if pd.notna(row['gene']):
        #             label_text = row['gene']
        #         if label_text is not None:
        #             ax.annotate(label_text, (row['angle'], row['radius']))

        #     for i, row in df_ch1_metadata.iterrows():
        #         label_text = None
        #         if pd.notna(row['gene']):
        #             label_text = row['gene']
        #         if label_text is not None:
        #             ax.annotate(label_text, (row['angle'], row['radius']))

        # # Read the title & axes labels from the kwargs
        # if 'title' in kwargs:
        #     plt.title(kwargs['title'])

        # if 'xlabel' in kwargs:
        #     ax.set_xlabel(kwargs['xlabel'])

        # if 'ylabel' in kwargs:
        #     ax.set_ylabel(kwargs['ylabel'])

        # if save_string:
        #     plt.savefig(f'{save_string}.{save_fmt}', bbox_inches='tight')
        # else:
        #     plt.show()

        # plt.close('all')

    def circular_manhattan2(self, metadata_filepath='../annotation/all_bins_metadata.csv', save_string=None, save_fmt='png', label=True, **kwargs):

        # Your existing code here

        # read in the metadata:
        df_metadata = pd.read_csv(metadata_filepath)

        sns.set_style('white')
        sns.set_context('paper')

        # make a 1x2 grid of subplots:
        fig, ax = plt.subplots(2, 1)#, figsize=(20, 10))

        # make a dataframe of crc_sample_ch0 and crc_sample_ch1 attributions, using the index as the bin_number:
        df_ch0 = pd.DataFrame(self.sum_thresh_vals_ch0/self.nsamples, columns=['attributions_one_channel'])
        # if you want count instead of fraction:
        # df_ch0 = pd.DataFrame(self.sum_thresh_vals_ch0, columns=['attributions_one_channel'])

        df_ch0['bin_number'] = df_ch0.index

        df_ch1 = pd.DataFrame(self.sum_thresh_vals_ch1/self.nsamples, columns=['attributions_one_channel'])
        # if you want count instead of fraction:
        # df_ch1 = pd.DataFrame(self.sum_thresh_vals_ch1, columns=['attributions_one_channel'])

        df_ch1['bin_number'] = df_ch1.index

        # add the metadata, fusing on the bin_number:
        df_ch0_metadata = pd.merge(df_ch0, df_metadata, on='bin_number')

        # keep a copy of the full metadata for this sample:
        self.df_full = df_ch0_metadata.copy()

        # remove all bins that have no real hits:
        df_ch0_metadata = df_ch0_metadata[df_ch0_metadata['attributions_one_channel'] != 0]
        # -----------------------

        # add the metadata, fusing on the bin_number:
        df_ch1_metadata = pd.merge(df_ch1, df_metadata, on='bin_number')

        # remove all bins that have no real hits:
        df_ch1_metadata = df_ch1_metadata[df_ch1_metadata['attributions_one_channel'] != 0]
        # -----------------------

        # make own palette for 22 chromosomes:
        hex_colours = [matplotlib.colors.rgb2hex(colour) for colour in cm.tab20.colors]
        hex_colours = hex_colours + ['#06C2AC', '#580F41']

        chrom_colours = {i: hex_colours[i-1] for i in range(1, 23)}

        chrom_to_colour = lambda x: chrom_colours[x]
        chromosome_colours_list_ch0 = df_ch0_metadata['chromosome'].apply(chrom_to_colour).tolist()
        cmap_ch0 = matplotlib.colors.ListedColormap(chromosome_colours_list_ch0)

        chromosome_colours_list_ch1 = df_ch1_metadata['chromosome'].apply(chrom_to_colour).tolist()
        cmap_ch1 = matplotlib.colors.ListedColormap(chromosome_colours_list_ch1)


# ----------------------------------------------------------------------------------------------------------------------------------
 
        # Calculate the midpoint of bin_number for each chromosome
        chromosome_bins = self.df_full.groupby('chromosome')['bin_number'].mean()
        
        chromosome_labels = [f'{i}' for i in range(1, 23)]

        chromosome_sizes = np.array([248956422, 242193529, 198295559, 190214555, 181538259,
                    170805979, 159345973, 145138636, 138394717, 133797422,
                    135086622, 133275309, 114364328, 107043718, 101991189,
                    90338345,  83257441,  80373285,  58617616,  64444167,
                    46709983,  50818468])#,  156040895, 57227415])
        
        
        
        
        sns.set_style('white')
        sns.set_context('talk')

        # Create a figure with two polar subplots
        fig, ax = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': 'polar'})

        # Calculate the angle for each bin based on the bin number
        df_ch0_metadata['angle'] = df_ch0_metadata['bin_number'] / self.df_full['bin_number'].max() * 2 * np.pi
        df_ch1_metadata['angle'] = df_ch1_metadata['bin_number'] / self.df_full['bin_number'].max() * 2 * np.pi

        # # Calculate the radius for each data point based on attributions
        # max_radius = 1.0  # You can adjust this value based on your data
        df_ch0_metadata['radius'] = df_ch0_metadata['attributions_one_channel']
        df_ch1_metadata['radius'] = df_ch1_metadata['attributions_one_channel']

        # Calculate cumulative sizes of chromosomes
        cumulative_sizes_ch0 = np.cumsum(chromosome_sizes[:len(self.df_full['chromosome'])]) * 2 * np.pi / chromosome_sizes.sum()
        cumulative_sizes_ch1 = np.cumsum(chromosome_sizes[:len(self.df_full['chromosome'])]) * 2 * np.pi / chromosome_sizes.sum()

        # insert a 0 at the beginning of the cumulative sizes, and remove the last element:
        cumulative_sizes_ch0 = np.insert(cumulative_sizes_ch0, 0, 0)
        cumulative_sizes_ch0 = cumulative_sizes_ch0[:-1]

        cumulative_sizes_ch1 = np.insert(cumulative_sizes_ch1, 0, 0)
        cumulative_sizes_ch1 = cumulative_sizes_ch1[:-1]

        # make rlim such that they're the same, so first check which channel has the largest radius:
        max_radius_ch0 = df_ch0_metadata['radius'].max()
        max_radius_ch1 = df_ch1_metadata['radius'].max()

        if max_radius_ch0 > max_radius_ch1:
            max_radius = max_radius_ch0
        else:
            max_radius = max_radius_ch1
        
        # make max_radius to the nearest 0.2 (so 0.2, 0.4, 0.6, 0.8, 1.0):
        max_radius = np.ceil(max_radius*5)/5


        # Circular Manhattan plot for channel 0
        ax[0].scatter(df_ch0_metadata['angle'], df_ch0_metadata['radius'], c=chromosome_colours_list_ch0, cmap=cmap_ch0, s=7.5, label='Channel 0')
        ax[0].set_rlabel_position(-20)
        # make rlabel gray instead of black:
        ax[0].tick_params(colors='gray')

        ax[0].set_rlim(0, max_radius)
        # ax[0].legend()

        # Circular Manhattan plot for channel 1
        ax[1].scatter(df_ch1_metadata['angle'], df_ch1_metadata['radius'], c=chromosome_colours_list_ch1, cmap=cmap_ch1, s=7.5, label='Channel 1')
        ax[1].set_rlabel_position(-20)
        ax[1].tick_params(colors='gray')

        ax[1].set_rlim(0, max_radius)
        # ax[1].legend()

        # Set x-axis ticks based on cumulative sizes of chromosomes
        ax[0].set_xticks(cumulative_sizes_ch0)

        # Set x-axis tick labels to chromosome labels, and color each chromosome label according to the color of the corresponding chromosome in hex_colour
        mod_chrom_labels = [f'{i}' for i in range(1, 23)]
        ax[1].set_xticks(cumulative_sizes_ch1)
        # ax[1].set_xticklabels(chromosome_labels, fontsize=10)
        ax[0].set_xticklabels(mod_chrom_labels, fontweight='bold')
        ax[1].set_xticklabels(mod_chrom_labels, fontweight='bold')
                              

        # make a different color for each chromosome label, using colours in hex_colours:

        for xtick, color in zip(ax[0].get_xticklabels(), hex_colours):
            xtick.set_color(color)
        for xtick, color in zip(ax[1].get_xticklabels(), hex_colours):
            xtick.set_color(color)


        # Set labels and legend
        if label:
            for i, row in df_ch0_metadata.iterrows():
                label_text = None
                if pd.notna(row['gene']):
                    label_text = row['gene']
                if label_text is not None:
                    ax[0].annotate(label_text, (row['angle'], row['radius']))

            for i, row in df_ch1_metadata.iterrows():
                label_text = None
                if pd.notna(row['gene']):
                    label_text = row['gene']
                if label_text is not None:
                    ax[1].annotate(label_text, (row['angle'], row['radius']))

        # Read the title & axes labels from the kwargs
        if 'title' in kwargs:
            fig.suptitle(kwargs['title'])

        if 'xlabel' in kwargs:
            ax[0].set_xlabel(kwargs['xlabel'])
            ax[1].set_xlabel(kwargs['xlabel'])

        if save_string:
            plt.savefig(save_string, bbox_inches='tight')
        else:
            plt.show()

        plt.close('all')    


    def heatmaps_dead(self, save_string=None, save_fmt='png', clustermap=True):
        pass 
    #     sns.set_context('paper')
    #     nsamples = len(self.GIG_dict[self.disease])

    #     # # set up arrays
    #     # attributions_gig_n = np.zeros( [nsamples, self.GIG_dict[disease][0].shape[1], self.GIG_dict[disease][0].shape[2]] )
    #     # attributions_dir = np.zeros_like(attributions_gig_n)
    #     # attributions_rank = np.zeros_like(attributions_gig_n)
    #     # # Normalisation
    #     # # process each channel such that the attributes are rankings. Keep a record of the direction of the attribute (+ve or -ve)
    #     for i in range(nsamples):
    #     #     samp_array = self.GIG_dict[disease][i].detach().numpy()
        
    #     #     # get the signs of the attribution
    #     #     attributions_dir[i,0,:] = np.sign( samp_array[0,0,:] )
    #     #     attributions_dir[i,1,:] = np.sign( samp_array[0,1,:] )
    
    #     #     # Normalise such that the sample |attribute| is a probability
    #     #     attributions_gig_n[i,0,:] = np.abs( samp_array[0,0,:]) / np.sum(np.abs( samp_array[0,0,:]))
    #     #     attributions_gig_n[i,1,:] = np.abs( samp_array[0,1,:]) / np.sum(np.abs( samp_array[0,1,:]))
    #     #     #print( np.sum(np.abs(attributions_gig[i,1,:]) ))
            
    #     #     # make an assertion that the sum is close to 1:
    #     #     assert np.isclose(np.sum(np.abs( samp_array[0,0,:])), 1), 'The sum of the ch0 attributions is not close to 1'
    #     #     assert np.isclose(np.sum(np.abs( samp_array[0,1,:])), 1), 'The sum of the ch1 attributions is not close to 1'


    #         #########################################################################################
    #         # pre normalisation, I want to remove all negative values:
    #         # ADDED OCTOBER 2023 - for better visualisation of the heatmaps & more pertinent clustering
    #         #########################################################################################
    #         self.vals_ch0 = self.attributions_gig_norm[:,0,:]
    #         self.vals_ch1 = self.attributions_gig_norm[:,1,:]

    #         # # remove all negative values:
    #         # self.vals_ch0 = np.where(self.vals_ch0 > 0, self.vals_ch0, 0)
    #         # self.vals_ch1 = np.where(self.vals_ch1 > 0, self.vals_ch1, 0)
    #         #########################################################################################

    #         # sample-wise max normalisation for each channel:
    #         self.attributions_gig_maxnorm[i,0,:] = self.vals_ch0[i,:] / np.max(np.abs(self.vals_ch0[i,:]))
    #         self.attributions_gig_maxnorm[i,1,:] = self.vals_ch1[i,:] / np.max(np.abs(self.vals_ch1[i,:]))

    #         # ###################################################################################################
    #         # # now the values should be between 0 and 1, so remove small values for the sake of the clustering:
    #         # # THRESHOLD HERE IS 0.1
    #         # # ADDED OCTOBER 2023 
    #         # ###################################################################################################
    #         # self.attributions_gig_maxnorm[i,0,:] = np.where(self.attributions_gig_maxnorm[i,0,:] > 0.3, self.attributions_gig_maxnorm[i,0,:], 0)
    #         # self.attributions_gig_maxnorm[i,1,:] = np.where(self.attributions_gig_maxnorm[i,1,:] > 0.3, self.attributions_gig_maxnorm[i,1,:], 0)
    #     # # Plot the heatmaps
    #     # vals_ch0 = np.multiply(attributions_gig_n[:, 0, :], attributions_dir[:,0,:])
    #     # vals_ch1 = np.multiply(attributions_gig_n[:, 1, :], attributions_dir[:,1,:])

        
    #     fig, ax = plt.subplots(2, 1, figsize=(20, 10))

    #     # just for visualisation, do samplewise max normalisation
    #     # select ch0:
    #     vals_ch0_plotting = self.attributions_gig_maxnorm[:,0,:]
    #     vals_ch1_plotting = self.attributions_gig_maxnorm[:,1,:]


    #     if not clustermap: 
    #         # plot the heatmaps, ch0 in ax[0], ch1 in ax[1]:
    #         sns.heatmap(vals_ch0_plotting, cmap='RdBu_r', center=0.0, xticklabels=False, yticklabels=False, ax=ax[0])
    #         # sns.heatmap(vals_ch0_plotting, cmap='RdBu_r', center=0.0, xticklabels=False, yticklabels=False)
    #         ax[0].set_xlabel('Features')
    #         ax[0].set_ylabel('Samples')
    #         ax[0].set_title('ch0', loc='left')
            
    #         sns.heatmap(vals_ch1_plotting, cmap='RdBu_r', center=0.0, xticklabels=False, yticklabels=False, ax=ax[1])
    #         # sns.heatmap(vals_ch1_plotting, cmap='RdBu_r', center=0.0, xticklabels=False, yticklabels=False)

    #         ax[1].set_xlabel('Features')
    #         ax[1].set_ylabel('Samples')
    #         ax[1].set_title('ch1', loc='left')
    #         if save_string is not None:
    #             plt.savefig(f'{save_string}_{self.disease}.{save_fmt}', bbox_inches='tight') 
    #     else:
    #         # cluster (hierarchical) the samples using sns.clustermap:
    #         # sns.clustermap(vals_ch0_plotting, cmap='RdBu_r', center=0.0, xticklabels=False, yticklabels=False, figsize=(20, 10), col_cluster=False)
            
    #         clustergrid = sns.clustermap(vals_ch0_plotting, 
    #                                      cmap='RdBu_r', 
    #                                     #  center=0.0, 
    #                                      yticklabels=False, 
    #                                      cbar_kws={}, 
    #                                      col_cluster=False, 
    #                                      row_cluster=True,
    #                                     #  standard_scale=1, # 0 is row-wise
    #                                      z_score=0, # 0 is row-wise
    #                                      )
    #         # remove the dendrograms:
    #         # clustergrid.ax_row_dendrogram.set_visible(False)
    #         # clustergrid.ax_col_dendrogram.set_visible(False)
    #         clustergrid.ax_heatmap.set_xlabel('Chromosome')
    #         clustergrid.ax_heatmap.set_ylabel('Samples')
    #         # set a title:
    #         # clustergrid.ax_heatmap.set_title(f'ch0: hierarchical clustering of GIG attributions for {self.disease.lower()} cohort (n:{nsamples})', loc='left')
    #         # plt.setp(sns.clustermap(all_gigs_np_copy[:,0,:], cmap='RdBu_r', center=0.0, xticklabels=False, yticklabels=False).ax_heatmap.collections, visible=False)
    #         # change the size of the heatmap:
    #         # plt.gcf().set_size_inches(15, 2)
    #         # remove the colorbar:
    #         clustergrid.cax.set_visible(False)

    #         # load the df with annotations:
    #         df_meta = pd.read_csv('./annotation/all_bins_metadata.csv')

    #         # Calculate the midpoint of bin_number for each chromosome
    #         chromosome_bins = df_meta.groupby('chromosome')['bin_number'].mean()
    #         # Define chromosome labels from 1 to 22
    #         chromosome_labels = [f'{i}' for i in range(1, 23)]

    #         # find chromosome ends:
    #         chromosome_ends = df_meta.groupby('chromosome')['bin_number'].max()

    #         # make sure the xlim is 0, length of the genome:
    #         plt.xlim(0, df_meta['bin_number'].max())
    #         # Set the x-axis tick positions and labels:
    #         clustergrid.ax_heatmap.set_xticks(chromosome_bins)
    #         clustergrid.ax_heatmap.set_xticklabels(chromosome_labels)


    #         # draw a vertical line at the beginning and end of each chromosome:
    #         for i in range(1, 23):
    #             clustergrid.ax_heatmap.axvline(x=chromosome_ends[i], color='black', linewidth=0.5)

    #         # -------------------------------------------------------------------------------------

    #         if save_string is not None:
    #             plt.savefig(f'{self.disease}_{save_string}_ch0.{save_fmt}', bbox_inches='tight', dpi=400)

    #         clustergrid = sns.clustermap(vals_ch1_plotting, 
    #                                      cmap='RdBu_r', 
    #                                     #  center=0.0, 
    #                                      yticklabels=False, 
    #                                      cbar_kws={}, 
    #                                      col_cluster=False, 
    #                                      row_cluster=True,
    #                                     #  standard_scale=1, # 0 is row-wise
    #                                     z_score=0, # 0 is row-wise
    #                                      )
    #         # remove the dendrograms:
    #         # clustergrid.ax_row_dendrogram.set_visible(False)
    #         # clustergrid.ax_col_dendrogram.set_visible(False)
    #         clustergrid.ax_heatmap.set_xlabel('Chromosome')
    #         clustergrid.ax_heatmap.set_ylabel('Samples')
    #         # set a title:
    #         # clustergrid.ax_heatmap.set_title(f'ch1: hierarchical clustering of GIG attributions for {self.disease.lower()} cohort (n:{nsamples})', loc='left')
    #         # plt.gcf().set_size_inches(15, 2)
    #         # remove the colorbar:
    #         clustergrid.cax.set_visible(False)

    #         # change the xlim on this plot:
    #         plt.xlim(0, df_meta['bin_number'].max())
            
    #         # Set the x-axis tick positions and labels:
    #         clustergrid.ax_heatmap.set_xticks(chromosome_bins)
    #         clustergrid.ax_heatmap.set_xticklabels(chromosome_labels)

    #         # draw a vertical line at the beginning and end of each chromosome:
    #         for i in range(1, 23):
    #             clustergrid.ax_heatmap.axvline(x=chromosome_ends[i], color='black', linewidth=0.5)

    #         if save_string is not None:
    #             plt.savefig(f'{self.disease}_{save_string}_ch1.{save_fmt}', bbox_inches='tight', dpi=400)

    #         plt.close('all')
    #             # -------------------------------------------------------------------------------------

