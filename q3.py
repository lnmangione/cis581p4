import numpy as np
import torch
import torch.nn.functional as F
from helpers import train, plot_stats, train2, initialize_weights


class LineModel(torch.nn.Module):
    def __init__(self):
        super(LineModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 7)
        self.conv2 = torch.nn.Conv2d(16, 8, 7)
        self.fc = torch.nn.Linear(8 * 4 * 4, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(64, -1)
        return torch.sigmoid(self.fc(x))


class DetModel(torch.nn.Module):
    def __init__(self):
        super(DetModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 7)
        self.conv2 = torch.nn.Conv2d(16, 8, 7)
        self.fc1 = torch.nn.Linear(8 * 4 * 4, 1)
        self.fc2 = torch.nn.Linear(8 * 4 * 4, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(64, -1)
        return torch.sigmoid(self.fc1(x)), self.fc2(x)  # Return label, width


def main():
    N, im_size = 64, 16
    x_line = torch.from_numpy(np.load('datasets/line/line_imgs.npy')).float().reshape(N, 1, im_size, im_size)
    y_line = torch.from_numpy(np.load('datasets/line/line_labs.npy')).float().reshape(N, 1)
    x_det = torch.from_numpy(np.load('datasets/detection/detection_imgs.npy')).float().reshape(N, 1, im_size, im_size)
    y_det = torch.from_numpy(np.load('datasets/detection/detection_labs.npy')).float().reshape(N, 1)
    y_detwid = torch.from_numpy(np.load('datasets/detection/detection_width.npy')).float().reshape(N, 1)

    # ========== QUESTION 1 ==========
    line_model = LineModel()
    initialize_weights(line_model)
    loss_fn = torch.nn.BCELoss(reduction='mean')
    losses, accuracies = train(line_model, loss_fn, x_line, y_line)
    plot_stats(losses, accuracies, 'Q3.1')

    # ========== QUESTION 2 ==========
    det_model = DetModel()
    initialize_weights(det_model)
    loss_fn1 = torch.nn.BCELoss(reduction='sum')
    loss_fn2 = torch.nn.MSELoss(reduction='sum')
    losses1, accuracies1, losses2, accuracies2 = train2(det_model, loss_fn1, loss_fn2, 0.01, 0.001, x_det, y_det, y_detwid)
    plot_stats(losses1, accuracies1, 'Q3.2 - Cross-Entropy')
    plot_stats(losses2, accuracies2, 'Q3.2 - L2')


main()
