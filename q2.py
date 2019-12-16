import numpy as np
import torch

from helpers import train, plot_stats, initialize_weights


def main():
    # N is batch size; D_in is input dimension; H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 16, 4, 1
    x = torch.from_numpy(np.load('datasets/random/random_imgs.npy').reshape((N, D_in))).float()
    y = torch.from_numpy(np.load('datasets/random/random_labs.npy').reshape((N, D_out))).float()

    # ========== QUESTION 1 ==========
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H, D_out),
        torch.nn.Sigmoid()
    )
    initialize_weights(model)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    losses, accuracies = train(model, loss_fn, x, y)
    plot_stats(losses, accuracies, 'Q2.1 - Sigmoid with L2')

    # ========== QUESTION 2 ==========
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H, D_out),
        torch.nn.Sigmoid()
    )
    initialize_weights(model)
    loss_fn = torch.nn.BCELoss(reduction='sum')
    losses, accuracies = train(model, loss_fn, x, y)
    plot_stats(losses, accuracies, 'Q2.2 - Sigmoid with CE')

    # ========== QUESTION 3 ==========
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        torch.nn.Sigmoid()
    )
    initialize_weights(model)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    losses, accuracies = train(model, loss_fn, x, y)
    plot_stats(losses, accuracies, 'Q2.3 - ReLU with L2')

    # ========== QUESTION 4 ==========
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        torch.nn.Sigmoid()
    )
    initialize_weights(model)
    loss_fn = torch.nn.BCELoss(reduction='sum')
    losses, accuracies = train(model, loss_fn, x, y, 0.05)
    plot_stats(losses, accuracies, 'Q2.4 - ReLU with CE')


main()
