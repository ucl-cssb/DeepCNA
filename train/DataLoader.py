import numpy as np
import torch


class CancerData(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels, train=True):
        "Initialization"
        self.labels = labels
        self.list_IDs = list_IDs
        self.train = train

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.list_IDs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Load data and get label
        #if self.train:
        #    X = np.load(f"../GEL_DATA/train/matrix_{self.list_IDs[index]}.npy", allow_pickle=True)
        #else:
        #    X = np.load(f"../GEL_DATA/test/matrix_{self.list_IDs[index]}.npy", allow_pickle=True)

        X = np.load(f"../DATA/matrix_{self.list_IDs[index]}.npy", allow_pickle=True)

        # How do you want to deal with nans?
        X[0] = np.nan_to_num(X[0], nan=np.nanmedian(X[0]))
        X[1] = np.nan_to_num(X[1], nan=np.nanmedian(X[1]))

        X = torch.from_numpy(X)
        y = int(self.labels[index])

        return X, y

class CancerDataInd(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels, train=True):
        "Initialization"
        self.labels = labels
        self.list_IDs = list_IDs
        self.train = train

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.list_IDs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Load data and get label
        #if self.train:
        #    X = np.load(f"../GEL_DATA/train/matrix_{self.list_IDs[index]}.npy", allow_pickle=True)
        #else:
        #    X = np.load(f"../GEL_DATA/test/matrix_{self.list_IDs[index]}.npy", allow_pickle=True)

        X = np.load(f"../DATA/matrix_{self.list_IDs[index]}.npy", allow_pickle=True)

        # How do you want to deal with nans?
        X[0] = np.nan_to_num(X[0], nan=np.nanmedian(X[0]))
        X[1] = np.nan_to_num(X[1], nan=np.nanmedian(X[1]))

        X = torch.from_numpy(X)
        y = int(self.labels[index])

        return X, y, self.list_IDs[index]
