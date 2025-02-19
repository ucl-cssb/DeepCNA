import numpy as np
import torch
from torch import nn
from torch import optim

from sklearn.metrics import f1_score


class MultiInputsNet(nn.Module):
    def __init__(self, compression_size = 50, hidden_dim1 = 1000, hidden_dim2 = 50, pdropout = 0.5, output_dim=13):
        """
        Initialize the MultiInputsNet with the given parameters and layers.

        Args:
        - train_labels (list or array): Training labels to determine the output dimension.
        """
        super(MultiInputsNet, self).__init__()

        # Constants
        #compression_size = 50
        #hidden_dim = [1000, 50]
        
        number_heads = 22
        number_channels = 2

        chrom_lengths_arr = np.array([
            248956422, 242193529, 198295559, 190214555, 181538259,
            170805979, 159345973, 145138636, 138394717, 133797422,
            135086622, 133275309, 114364328, 107043718, 101991189,
            90338345,  83257441,  80373285,  58617616,  64444167,
            46709983,  50818468
        ])

        chrom_lengths_binned = [int(round(i / 100000)) for i in chrom_lengths_arr]
        chrom_locations = [sum(chrom_lengths_binned[:i + 1]) for i in range(len(chrom_lengths_binned))]
        chrom_locations.insert(0, 0)
        self.chrom_locations = chrom_locations

        input_dim = number_heads * compression_size * number_channels

        # Create head layers using nn.ModuleList
        self.heads = nn.ModuleList([
            nn.Linear(chrom_lengths_binned[i], compression_size)
            for i in range(number_heads)
        ])

        # Concatenation layer
        self.conc = nn.Linear(input_dim, input_dim)

        # MLP main network
        self.seq = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.Dropout(p=pdropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Dropout(p=pdropout),
            nn.LeakyReLU(),
            nn.Dropout(p=pdropout),
            nn.Linear(hidden_dim2, output_dim),
            #nn.LogSoftmax(dim=1)
        )

    def forward(self, input_tensor):
        """
        Define the forward pass of the MultiInputsNet.

        Args:
        - input_tensor (Tensor): input_tensor tensor with shape (batch_size, channels, features).

        Returns:
        - Tensor: Output tensor after passing through the network.
        """
        flat_func = lambda c: c.view(c.size(0), -1)

        # Process the different inputs (chromosomes) to each head
        compressed_chroms = []
        for i in range(22):
            chrom_input1 = input_tensor[:, 0, self.chrom_locations[i]:self.chrom_locations[i + 1]]
            chrom_input2 = input_tensor[:, 1, self.chrom_locations[i]:self.chrom_locations[i + 1]]
            compressed_chroms.append(flat_func(self.heads[i](chrom_input1)))
            compressed_chroms.append(flat_func(self.heads[i](chrom_input2)))

        # Concatenate the compressed chromosomes
        combined = torch.cat(compressed_chroms, dim=1)

        # Pass through concatenation layer
        concatenated_compressed = self.conc(combined)

        # Pass through the main MLP
        return self.seq(concatenated_compressed)


def train_network_(train_loader, validation_loader, model, device, epochs = 10, lr = 1e-4, momentum=0.9, wd = 5e-1 ):
    """Pytorch training and test loops for a supervised neural network, as defined by the input model, training and validation loaders.

    Args:
        train_loader (Dataloader): Pytorch dataloader for training data.
        validation_loader (Dataloader): Pytorch dataloader for validation data.
        model (nn.Module): Pytorch model to be trained. Defaults to None.
        epochs (int, optional): Number of epochs to train the model for. Defaults to 10.
        lr (float, optional): The learning rate for the optimizer. Defaults to 1e-4.
        wd (float, optional): Weight decay. Defaults to 5e-1.

    Returns:
        best_model (nn.Module): The trained model.
        best_model_weights (dict): The weights of the trained model.
        train_losses (list): The training loss for each epoch.
        test_losses (list): The test loss for each epoch.
    """

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

    #best_test_loss = 1e10

    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for samples_batch, labels_batch in train_loader:

            samples = samples_batch.to(device)
            labels = labels_batch.to(device)

            optimizer.zero_grad()
            log_ps = model(samples.float())

            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        test_loss = 0
        accuracy = 0

        # Turn off gradients for validation
        with torch.no_grad():
            model.eval()
            for samples_batch, labels_batch in validation_loader:
                samples = samples_batch.to(device)
                labels = labels_batch.to(device)
                
                out = model(samples.float())
                test_loss += criterion(out, labels)
                top_p, top_class = out.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                
                #accuracy += torch.mean(equals.type(torch.FloatTensor))
                # calculate the F1-macro score
                accuracy += f1_score(labels.cpu().numpy(), top_class.cpu().numpy(), average='macro')


        model.train()

        train_losses.append(running_loss/len(train_loader))
        test_losses.append(test_loss/len(validation_loader))

        test_accuracy = accuracy/len(validation_loader)

        print(f"Epoch: {e+1}/{epochs} | Training loss: {train_losses[-1]:.3f} | Test loss: {test_losses[-1]:.3f} | Test F1-macro: {(test_accuracy)*100:.3f}%")

    return model, train_losses, test_losses, test_accuracy


def train_network_notest_(train_loader, model, device, epochs = 10, lr = 1e-4, momentum=0.9, wd = 5e-1 ):
    """Pytorch training and test loops for a supervised neural network, as defined by the input model, training and validation loaders.

    Args:
        train_loader (Dataloader): Pytorch dataloader for training data.
        validation_loader (Dataloader): Pytorch dataloader for validation data.
        model (nn.Module): Pytorch model to be trained. Defaults to None.
        epochs (int, optional): Number of epochs to train the model for. Defaults to 10.
        lr (float, optional): The learning rate for the optimizer. Defaults to 1e-4.
        wd (float, optional): Weight decay. Defaults to 5e-1.

    Returns:
        best_model (nn.Module): The trained model.
        best_model_weights (dict): The weights of the trained model.
        train_losses (list): The training loss for each epoch.
        test_losses (list): The test loss for each epoch.
    """

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

    #best_test_loss = 1e10

    train_losses = []
    for e in range(epochs):
        running_loss = 0
        for samples_batch, labels_batch in train_loader:

            samples = samples_batch.to(device)
            labels = labels_batch.to(device)

            optimizer.zero_grad()
            log_ps = model(samples.float())

            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        test_loss = 0
        accuracy = 0

        train_losses.append(running_loss/len(train_loader))

        print(f"Epoch: {e+1}/{epochs} | Training loss: {train_losses[-1]:.3f}")

    return model, train_losses