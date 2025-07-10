import numpy as np
import itertools
import matplotlib.pyplot as plt


def imshow_text(data, labels=None, ax=None, color_high="w", color_low="k", color_threshold=50, skip_zeros=False):
    """Text labels for individual cells of an imshow plot

    Args:
        data ([type]): Color values
        labels ([type], optional): Text labels. Defaults to None.
        ax ([type], optional): axis. Defaults to plt.gca().
        color_high (str, optional): [description]. Defaults to 'w'.
        color_low (str, optional): [description]. Defaults to 'k'.
        color_threshold (int, optional): [description]. Defaults to 50.
        skip_zeros (bool, optional): [description]. Defaults to False.
    """
    if ax is None:
        ax = plt.gca()

    if labels is None:
        labels = data

    for x, y in itertools.product(range(data.shape[0]), range(data.shape[1])):
        if skip_zeros and labels[y, x] == 0:
            continue
        ax.text(
            x, y, f"{labels[y, x]:1.0f}", ha="center", va="center", c=color_high if data[y, x] > color_threshold else color_low
        )


def plot_confmat(C, cmap="Greys", ax=None):
    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)

    plt.imshow(C * 100, cmap=cmap)
    plt.clim(0, 100)
    plt.colorbar(shrink=0.5, ticks=[0, 50, 100])
    plt.xticks(np.arange(C.shape[1]))
    plt.yticks(np.arange(C.shape[0]))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.axis("square")
    imshow_text((C * 100).astype(int))
