import numpy as np
import torch
import matplotlib.pyplot as plt


def train(model, loss_fn, x, y):
    losses, accuracies = [], []
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for t in range(10000):
        y_pred = model(x)

        # Save loss and accurate
        loss = loss_fn(y_pred, y)
        losses.append(loss)
        correct = (y.eq((y_pred + 0.5).long())).sum().item()
        accuracies.append(correct / len(x))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()

        if accuracies[-1] == 1:
            break

    return losses, accuracies


def plot_stats(losses, accuracies, title):
    plt.plot(losses)
    plt.title(title + ': Loss vs Training Iteration')
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(accuracies)
    plt.title(title + ': Accuracy vs Training Iteration')
    plt.xlabel('Training Iteration')
    plt.ylabel('Accuracy')
    plt.show()


def main():
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 16, 4, 1

    x = torch.from_numpy(np.load('datasets/random/random_imgs.npy').reshape((N, D_in))).float()
    y = torch.from_numpy(np.load('datasets/random/random_labs.npy').reshape((N, D_out))).float()

    # ========== QUESTION 1 ==========
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.Linear(H, D_out),
        torch.nn.Sigmoid()
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')
    losses, accuracies = train(model, loss_fn, x, y)
    plot_stats(losses, accuracies, 'Question 1')

    # # ========== QUESTION 2 ==========
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(D_in, H),
    #     torch.nn.Linear(H, D_out),
    #     torch.nn.Sigmoid()
    # )
    # loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    # losses, accuracies = train(model, loss_fn, x, y)
    # plot_stats(losses, accuracies, 'Question 2')

# ========== QUESTION 3 ==========
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        torch.nn.Sigmoid()
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')
    losses, accuracies = train(model, loss_fn, x, y)
    plot_stats(losses, accuracies, 'Question 3')


main()
