import torch
from torch import nn, optim

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
import statsmodels.api as sm

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from DataLoader import CancerData
from Functions import make_weights_for_balanced_classes
from Network import MultiInputsNet

class_label_dict = {0:"Lung", 1:"Breast", 2:"Prostate", 3:"Colorectal", 
                        4:"Ovarian", 5:"Endometrial\ncarcinoma", 6:"Bladder", 
                        7:"Renal", 8:"Adult\nglioma", 9:"Malignant\nmelanoma", 
                        10:"Upper\ngastrointestinal", 11:"Oral\noropharngeal", 12:"Haemonic"}


def get_average_cm(outdir="./"):
    
    # unsorted class labels
    #cohort_df = pd.read_csv("../annotation/primary_site_info.csv")  
    #class_labels = [class_label_dict[i] for i in range(13)] 

    cohort_df = pd.read_csv("../annotation/primary_site_info.csv") 
    cohort_df = cohort_df.sort_values(by="number_of_samples", ascending=False)
    sorted_indices = cohort_df["primary_site_code"].tolist()  # Ordered class indices
    class_labels = [class_label_dict[i] for i in sorted_indices] 

    class_labels_with_metrics_y = class_labels
    class_labels_with_metrics_x = class_labels + ["F1 Score"]

    # load the confusion matrices
    cm1 = np.load(outdir+"/confusion-matrix-1.npy")
    cm2 = np.load(outdir+"/confusion-matrix-2.npy")
    cm3 = np.load(outdir+"/confusion-matrix-3.npy")
    cm4 = np.load(outdir+"/confusion-matrix-4.npy")
    cm5 = np.load(outdir+"/confusion-matrix-5.npy")

    # average the confusion matrices
    cm_avg = (cm1 + cm2 + cm3 + cm4 + cm5) / 5

    # Plot confusion matrix with precision & recall
    plt.figure(figsize=(9, 9))  # Square format
    xf = sns.heatmap(cm_avg, annot=True, fmt=".2f", cmap="Blues",
                     xticklabels=class_labels_with_metrics_x, yticklabels=class_labels_with_metrics_y, cbar=False,
                     annot_kws={"size": 8})  # Reduce font size for clarity
    
    xf.set_yticklabels(xf.get_yticklabels(), size = 8)
    xf.set_xticklabels(xf.get_xticklabels(), size = 8)

    plt.xlabel("Predicted Label", fontsize=8)
    plt.ylabel("True Label", fontsize=8)
    plt.savefig(outdir+"/plot-confusion-matrix-mean.pdf", bbox_inches="tight")

def do_F1_analysis(outdir):

    # get F1 information
    df_f1_all = pd.read_csv(outdir+"/f1_scores.csv")
    df_avg = df_f1_all.groupby("Primary Site", as_index=False)["F1 Score"].mean() 

    #print(df_avg)

    # get PGA information
    column_types = {"index_matrix":int, "Disease Type":str, "PGA":float, "Altered bins":int,"label": int}
    df_pga = pd.read_csv("../annotation/PGA_dataframe_filtered.csv", dtype=column_types)
    
    # calculate the average PGA for each primary site
    df_pga = df_pga.groupby("Disease Type", as_index=False)["PGA"].median()
    df_pga = df_pga.rename( columns={"Disease Type":"Primary Site"} )

    #print(df_pga)

    cohort_df = pd.read_csv("../annotation/primary_site_info.csv") 
    cohort_df = cohort_df.sort_values(by="number_of_samples", ascending=False)
    cohort_df['plot_labels'] = class_label_dict
    cohort_df = cohort_df.rename( columns={"primary_site_text":"Primary Site"} )
    # remove the "code" column
    cohort_df = cohort_df.drop(columns=["primary_site_code"])

    # merge the dataframes
    df_merged = pd.merge(df_avg, df_pga, on="Primary Site")
    df_merged = pd.merge(df_merged, cohort_df, on="Primary Site")

    #print(df_merged)

    # create a two panel figure containing a scatter plot of F1 vs sample size and F1 vs PGA
    primary_sites = df_merged["Primary Site"]
    f1_scores = df_merged["F1 Score"]
    sample_sizes = df_merged["number_of_samples"]
    pga_values = df_merged["PGA"]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot: F1 Score vs Sample Size with Linear Fit, colored by Primary Site
    sns.scatterplot(x=sample_sizes, y=f1_scores, hue=df_merged["plot_labels"], ax=axes[0], palette="tab10", edgecolor="k")
    sns.regplot(x=sample_sizes, y=f1_scores, ax=axes[0], ci=95, scatter=False, color="black")
    axes[0].set_xlabel("Number of Samples")
    axes[0].set_ylabel("F1 Score")
    # remove legend
    axes[0].get_legend().remove()

    # Quadratic Model for F1 Score vs PGA
    X = np.column_stack([np.ones(len(pga_values)), pga_values, pga_values**2])
    model = sm.OLS(f1_scores, X).fit()

    # Generate fitted values and confidence interval
    pga_range = np.linspace(min(pga_values), max(pga_values), 100)
    X_pred = np.column_stack([np.ones(len(pga_range)), pga_range, pga_range**2])
    predictions = model.get_prediction(X_pred)
    pred_mean = predictions.predicted_mean
    pred_ci_lower, pred_ci_upper = predictions.conf_int().T

    # Scatter plot: F1 Score vs PGA with Quadratic Fit, colored by Primary Site
    sns.scatterplot(x=pga_values, y=f1_scores, hue=df_merged["plot_labels"], ax=axes[1], palette="tab10", edgecolor="k")
    axes[1].plot(pga_range, pred_mean, color="black")
    axes[1].fill_between(pga_range, pred_ci_lower, pred_ci_upper, color="gray", alpha=0.2)
    axes[1].set_xlabel("Median PGA")
    axes[1].set_ylabel("F1 Score")
    axes[1].legend(title="Primary Site", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Improve layout
    plt.tight_layout()
    plt.savefig(outdir+"/plot-f1-analysis.pdf", bbox_inches="tight")

def evaluate(model, test_loader, label="", outdir="./"):
    
    # sorted class labels
    cohort_df = pd.read_csv("../annotation/primary_site_info.csv") 
    cohort_df = cohort_df.sort_values(by="number_of_samples", ascending=False)
    sorted_indices = cohort_df["primary_site_code"].tolist()  # Ordered class indices
    class_labels = [class_label_dict[i] for i in sorted_indices] 

    model = model.to("cpu")
    model.eval()

    # Prepare to store predictions and labels
    all_preds = []
    all_labels = []

    # Disable gradient calculations for inference
    with torch.no_grad():
        for inputs, labels in test_loader:  # Assuming test_loader returns (input, label)
            outputs = model(inputs.float())  # Get model predictions
            _, preds = torch.max(outputs, 1)  # Get the index of the max logit as class prediction

            all_preds.extend(preds.cpu().numpy())  # Store predictions
            all_labels.extend(labels.cpu().numpy())  # Store true labels

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    f1 = f1_score(all_labels, all_preds, average=None)

    # re-order the confusion matrix
    cm = cm[sorted_indices, :]
    cm = cm[:, sorted_indices]
    f1 = f1[sorted_indices]

    # Append f1 score to the confusion matrix
    cm_display = np.column_stack((cm, f1))
    class_labels_with_metrics_y = class_labels
    class_labels_with_metrics_x = class_labels + ["F1 Score"]

    # write cm_display to npy file
    np.save(outdir+"/confusion-matrix-"+label+".npy", cm_display)

    # Plot confusion matrix with precision & recall
    plt.figure(figsize=(9, 9))  # Square format
    xf = sns.heatmap(cm_display, annot=True, fmt=".2f", cmap="Blues",
                     xticklabels=class_labels_with_metrics_x, yticklabels=class_labels_with_metrics_y, cbar=False,
                     annot_kws={"size": 8})  # Reduce font size for clarity
    
    xf.set_yticklabels(xf.get_yticklabels(), size = 8)
    xf.set_xticklabels(xf.get_xticklabels(), size = 8)

    plt.xlabel("Predicted Label", fontsize=8)
    plt.ylabel("True Label", fontsize=8)
    plt.savefig(outdir+"/plot-confusion-matrix-"+label+".pdf", bbox_inches="tight")
    plt.close()
 
    # create a dataframe with primary_site_text in the first column and corresponding f1 scores in the second column
    labs = cohort_df["primary_site_text"].tolist() 
    df = pd.DataFrame(list(zip(labs, f1)), columns=["Primary Site", "F1 Score"])
    return df

def main():

    #get_average_cm("results_baseline/")
    #get_average_cm("results_opt1/")

    do_F1_analysis("results_opt1/")

if __name__ == "__main__":
    main()