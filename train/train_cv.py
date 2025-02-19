import sys
import os
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.model_selection import StratifiedKFold

from DataLoader import CancerData
from Functions import make_weights_for_balanced_classes
from Network import MultiInputsNet
from Network import train_network_
from Network import train_network_notest_
from Evaluate import evaluate, get_average_cm, do_F1_analysis


# Define your main function
def main():
    
    # read in the hyperparameters on the command line using argparse

    parser = argparse.ArgumentParser(description='Train a neural network for cancer classification.')
    parser.add_argument('--evaluate', type=int, default=0, help='==0, just output F1 scores; ==1 Make the confusion matrices across folds')
    parser.add_argument('--final', type=int, default=0, help='==0, do cross validation; ==1, train on all data')
    parser.add_argument('--outdir', type=str, default="./", help='Output directory to store the results.')
    parser.add_argument('--output_file', type=str, default="results.txt", help='Output file to store the results.')

    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for the model.')
    parser.add_argument('--wd', type=float, default=0.02, help='Weight decay for the optimizer.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum in the optimizer.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training.')
    
    parser.add_argument('--subnetout', type=int, default=50, help='Output dim of the chromosome subnet.')
    parser.add_argument('--hidden1', type=int, default=1000, help='Dimension of fist hidden layer.')
    parser.add_argument('--hidden2', type=int, default=50, help='Dimension of second hidden layer.')

    args = parser.parse_args()
    do_evaluation = args.evaluate
    do_final = args.final

    outdir = args.outdir
    output_file = args.output_file

    epochs = args.epochs
    lr = args.lr
    momentum = args.momentum
    dropout = args.dropout
    wd = args.wd
    batch_size = args.batch_size

    hidden1 = args.hidden1
    hidden2 = args.hidden2
    subnetout = args.subnetout

    if(do_final == 1 and do_evaluation == 1):
        print("Cannot do final training and evaluation at the same time")
        sys.exit(1)

    if(do_evaluation == 1 or do_final == 1):    
        # make output directory if it does not exist
        os.makedirs(outdir, exist_ok=True)
        

    num_folds = 5  # Number of folds for CV

    # Load the data
    df = pd.read_csv("../annotation/metadata.csv")

    if(do_final == 1):
        # Prepare datasets
        all_data = CancerData(df["ID_SAMPLE"].values,
                                df["primary_site_code"].values,
                                train=True)

        # Compute class weights for balanced sampling
        train_weights = make_weights_for_balanced_classes(df["ID_SAMPLE"].values, df["primary_site_code"].values)
        train_weights = torch.DoubleTensor(train_weights)
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

        # DataLoaders
        all_loader = DataLoader(all_data, batch_size=batch_size, sampler=train_sampler)

        # Initialize model
        model = MultiInputsNet(compression_size = subnetout, hidden_dim1 = hidden1, hidden_dim2 = hidden2, pdropout = dropout, output_dim=13)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        model = model.to(device)

        # Train the model
        model, train_losses = train_network_notest_(
            all_loader,
            epochs=epochs,
            lr=lr,
            momentum=momentum,
            model=model,
            device=device,
            wd=wd
        )

        # Save the model for this fold
        model_filename = f"{outdir}/model_weights_final.pth"
        torch.save(model.state_dict(), model_filename)

    else:
        # proceed with cross validation

        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

        # Create lists to store results
        all_train_losses = []
        all_test_losses = []
        all_f1_scores = []

        # create a dataframe to store the f1 values
        if(do_evaluation):
            df_f1_all = pd.DataFrame(columns=["Primary Site", "F1 Score", "Fold"])
        
        # Perform 5-Fold Cross Validation
        for fold, (train_idx, test_idx) in enumerate(skf.split(df, df["primary_site_code"])):
            print(f"Fold {fold + 1}/{num_folds}")

            # Create train and test splits
            train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

            # Prepare datasets
            training_data = CancerData(train_df["ID_SAMPLE"].values,
                                    train_df["primary_site_code"].values,
                                    train=True)

            test_data = CancerData(test_df["ID_SAMPLE"].values,
                                test_df["primary_site_code"].values,
                                train=False)

            # Compute class weights for balanced sampling
            train_weights = make_weights_for_balanced_classes(train_df["ID_SAMPLE"].values, train_df["primary_site_code"].values)
            train_weights = torch.DoubleTensor(train_weights)
            train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

            # DataLoaders
            train_loader = DataLoader(training_data, batch_size=batch_size, sampler=train_sampler)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

            # Initialize model
            model = MultiInputsNet(compression_size = subnetout, hidden_dim1 = hidden1, hidden_dim2 = hidden2, pdropout = dropout, output_dim=13)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print("Using device:", device)
            model = model.to(device)

            # Train the model
            model, train_losses, test_losses, f1 = train_network_(
                train_loader,
                test_loader,
                epochs=epochs,
                lr=lr,
                momentum=momentum,
                model=model,
                device=device,
                wd=wd
            )

            # Store losses
            all_train_losses.append(train_losses)
            all_test_losses.append(test_losses)
            all_f1_scores.append(f1)

            if(do_evaluation):
                # Save the model for this fold
                model_filename = f"{outdir}/model_weights_fold_{fold + 1}.pth"
                torch.save(model.state_dict(), model_filename)
                print(f"Model weights for Fold {fold + 1} saved to {model_filename}")

                # Plot training and test losses
                test_losses_cpu = [i.cpu() for i in test_losses]
                plt.plot(train_losses, label="Training loss")
                plt.plot(test_losses_cpu, label="Test loss")
                plt.legend(frameon=False)
                plt.savefig(f"{outdir}/plot-loss-fold-{fold + 1}.pdf")
                plt.close()

                # do confusion matrices
                d_f1 = evaluate(model, test_loader, str(fold+1), outdir)
                
                d_f1["Fold"] = fold + 1
                df_f1_all = pd.concat([df_f1_all, d_f1])
                
                #print(df_f1_all)
                df_f1_all.to_csv(f"{outdir}/f1_scores.csv", index=False)


        print("Cross-validation completed!")

        if(do_evaluation):
            print("mean f1 score:", sum(all_f1_scores)/num_folds)
            get_average_cm(outdir)

            do_F1_analysis(outdir)

        else:
            # print out the hyper parameters and the five f1 scores
            with open(output_file, "a") as file:
                print(f"{subnetout} {hidden1} {hidden2} {epochs} {lr} {dropout} {wd} {batch_size} {momentum} {' '.join(f'{x:.2f}' for x in all_f1_scores)} {sum(all_f1_scores)/num_folds:.2f}", file=file)


if __name__ == "__main__":
    main()
