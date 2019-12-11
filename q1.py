import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_wbz(Wi, Bi, Zi, title, z_label):
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_title(title)
    ax.set_xlabel('Weight')
    ax.set_ylabel('Bias')
    ax.set_zlabel(z_label)

    surf = ax.plot_surface(Wi, Bi, Zi, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


W = np.arange(-2, 2, 0.1)
B = np.arange(-2, 2, 0.1)
W, B = np.meshgrid(W, B)

# ---- Question 1 -----
Z1 = 1 / (np.ones(W.shape) + np.exp(-B - W))
plot_wbz(W, B, Z1, 'Sigmoid Function', 'Output')

# ---- Question 2 -----
Z2 = (0.5 - Z1)**2
plot_wbz(W, B, Z2, 'L2 Loss', 'Loss')

# ---- Question 3 -----
Z3 = np.exp(W + B)*(np.exp(W + B)-1) / (np.exp(W + B) + 1)**3
plot_wbz(W, B, Z3, 'Gradient of L2 Loss', 'Gradient')

# ---- Question 4 -----
Z4 = -1 * (0.5 * np.log(Z1) + 0.5 * np.log(1 - Z1))
plot_wbz(W, B, Z4, 'Cross-Entropy Loss', 'Loss')

# ---- Question 5 -----
# dL/dw = x(a - y) = 1(Z1 - 0.5)
Z5 = Z1 - 0.5
plot_wbz(W, B, Z5, 'Gradient of Cross-Entropy Loss', 'Gradient')
