import torch
import matplotlib.pyplot as plt
from tqdm import trange


def train(model, loss_fn, x, y, lr=0.1):
    losses, accuracies = [], []
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in trange(10000):
        y_pred = model(x)

        # Save loss and accuracy
        loss = loss_fn(y_pred, y)
        losses.append(loss.item())
        correct = (y.eq((y_pred + 0.5).long())).sum().item()
        accuracies.append(correct / len(x))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()

        if accuracies[-1] == 1:
            break

    return losses, accuracies


def train2(model, loss_fn1, loss_fn2, lr1, lr2, x, y1, y2):
    losses1, accuracies1, losses2, accuracies2 = [], [], [], []
    optimizer = torch.optim.SGD(model.parameters(), lr=min(lr1, lr2))

    for _ in trange(10000):
        y_pred1, y_pred2 = model(x)

        # Save loss and accuracy
        loss1 = loss_fn1(y_pred1, y1)
        losses1.append(loss1.item())
        correct1 = (y1.eq((y_pred1 + 0.5).long())).sum().item()
        accuracies1.append(correct1 / len(x))

        loss2 = loss_fn2(y_pred2, y2)
        losses2.append(loss2.item())
        guess2 = (y_pred2 + 0.5).long()
        correct2 = (y2.eq(guess2)).sum().item()
        accuracies2.append(correct2 / len(x))

        optimizer.zero_grad()
        loss = (loss1 * lr1 + loss2 * lr2) / (lr1 + lr2)
        loss.backward()
        optimizer.step()

        if accuracies1[-1] == 1 and accuracies2[-1] == 1:
            break

    return losses1, accuracies1, losses2, accuracies2


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

    print(str(len(losses)) + ' iterations for ' + title)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0, std=0.1)
            torch.nn.init.constant_(m.bias, 0.1)
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.normal_(m.weight, mean=0, std=0.1)
