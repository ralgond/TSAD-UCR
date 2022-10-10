import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import time

class DeepSVDDTrainer:
    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6):
        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        # Deep SVDD parameters
        self.R = torch.tensor(R, device='cuda')  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device='cuda') if c is not None else None
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        self.lr = lr
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
    
    def train(self, train_loader: DataLoader, net: nn.Module):
        # Set device for network
        net = net.to('cuda')

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            print('Center c initialized.')

        net.train()
        for epoch in range(self.n_epochs):

            # scheduler.step()
            # if epoch in self.lr_milestones:
            #     logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs = data.unsqueeze(1)
                inputs = inputs.to('cuda')

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device="cuda")

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

    def test(self, test_loader:DataLoader, net:nn.Module):
        ret = []
        net.to("cuda")
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs = data.unsqueeze(1)
                inputs = inputs.to("cuda")
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist
                ret.extend(scores.cpu())
        return ret

    def init_center_c(self, train_loader: DataLoader, net: nn.Module, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device="cuda")

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs = data.unsqueeze(1)
                inputs = inputs.to('cuda')
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)