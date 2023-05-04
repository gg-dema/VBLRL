import numpy as np
import torch
import torchbnn


'''
this class should just implement the base bnn and backprop step for it
consider the dynamics module for a complete dynamics model (that contains a set of bnn) 
'''

class BNN:
    def __init__(self):
        pass

    def init_weight_from(self, world_model):
        pass

    def vectorize_params(self) -> np.array:
        pass

    def load_params(self, params: np.array):
        pass

    def sample_params(self):
        # should return the entire set of W, b for a net
        pass


# ---------------------------------------
# SAMPLE ON IRIS BAYESIAN
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class IrisDataset(Dataset):

    def __init__(self, src_file, num_rows=None):
        temp_x = np.loadtxt(src_file, max_rows=num_rows, usecols=range(0, 4))
        temp_y = np.loadtxt(src_file, max_rows=num_rows, usecols=4)

        self.x_data = torch.tensor(temp_x, dtype=torch.float32)
        self.y_data = torch.tensor(temp_y, dtype=torch.int64)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        preds = self.x_data[idx]
        spcs = self.y_data[idx]

        sample = {'predictors': preds, 'species': spcs}
        return sample

def accuracy(model, dataset):
    # assume model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    n_correct = 0
    n_wrong = 0

    for (_, batch) in enumerate(dataloader):
        X = batch['predictors']
        Y = batch['species']
        with torch.no_grad():
            logit = model(X)
        logit = torch.argmax(logit)

        if logit == Y:
            n_correct += 1
        else:
            n_wrong += 1

        acc = (n_correct)/(n_correct + n_wrong)
        return acc

def accuracy_quick(model, dataset):
    n = len(dataset)
    X = dataset[0:n]['predictos']
    Y = torch.flatten(dataset[0:n]['species'])

    with torch.no_grad():
        logit = model(X)
    logit = torch.argmax(logit, dim=1)  # collapse cols
    num_correct = torch.sum(Y == logit)
    acc = (num_correct * 1.0 / len(dataset))
    return acc.item()


def train():

    train_file = ...
    train_ds = IrisDataset(train_file, num_rows=120)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    net = BayesianNet()
    epochs = 100
    ep_log_interval = 10

    cross_entropy_loss = nn.CrossEntropyLoss()
    kl_divergence_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)


    for epoch in range(epochs):
        epoch_loss = 0
        for (batch_idx, batch) in enumerate(train_loader):
            X = batch['predictors']
            Y = batch['species']
            optimizer.zero_grad()
            logit = net(X)

            cross_loss = cross_entropy_loss(logit, Y)
            kl_loss = kl_divergence_loss(net)

            tot_loss = cross_loss + (0.001 * kl_loss)

class BayesianNet(nn.Module):
    def __init__(self):
        super(BayesianNet, self).__init__()
        self.hidden1 = bnn.BayesLinear(prior_mu=0,
                                       prior_sigma=0.1,
                                       in_features=4,
                                       out_features=100)
        self.out = bnn.BayesLinear(prior_mu=0,
                                   prior_sigma=0.1,
                                   in_features=100,
                                   out_features=3)
    def forward(self, x):
        z = torch.relu(self.hidden1(x))
        z = self.out(z)   # avoid softmax: CrossEntropyLoss()
        return z

