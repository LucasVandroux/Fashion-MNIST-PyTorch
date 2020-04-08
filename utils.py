import os
import shutil

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, only_trainable_parameters: bool = False,) -> int:
    """ Count the numbers of parameters in a model

    Args:
        model (nn.Module): model to count the parameters from.
        only_trainable_parameters (bool: False): only count the trainable parameters.

    Returns:
        num_parameters (int): number of parameters in the model.
    """
    if only_trainable_parameters:
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num_parameters = sum(p.numel() for p in model.parameters())

    return num_parameters


def save_checkpoint(
    current_epoch: int,
    num_iteration: int,
    best_accuracy: float,
    model_state_dict: dict,
    optimizer_state_dict: dict,
    is_best: bool,
    experiment_path: str,
    checkpoint_filename: str = "checkpoint.pth.tar",
    best_filename: str = "model_best.pth.tar",
):
    """ Save the checkpoint and the best model to the disk

    Args:
        current_epoch (int): current epoch of the training.
        num_iteration (int): number of iterations since the beginning of the training.
        best_accuracy (float): last best accuracy obtained during the training.
        model_state_dict (dict): dictionary containing information about the model's state.
        optimizer_state_dict (dict): dictionary containing information about the optimizer's state.
        is_best (bool): boolean to save the current model as the new best model.
        experiment_path (str): path to the directory where to save the checkpoints and the best model.
        checkpoint_filename (str: "checkpoint.pth.tar"): filename to give to the checkpoint.
        best_filename (str: "model_best.pth.tar"):  filename to give to the best model's checkpoint.

    """
    print(
        f'Saving checkpoint{f" and new best model (best accuracy: {100 * best_accuracy:05.2f})" if is_best else f""}...'
    )
    checkpoint_filepath = os.path.join(experiment_path, checkpoint_filename)
    torch.save(
        {
            "epoch": current_epoch,
            "num_iteration": num_iteration,
            "best_accuracy": best_accuracy,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
        },
        checkpoint_filepath,
    )
    if is_best:
        shutil.copyfile(
            checkpoint_filepath, os.path.join(experiment_path, best_filename),
        )


class MetricTracker:
    """Computes and stores the average and current value of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all the tracked parameters """
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value: float, num: int = 1):
        """ Update the tracked parameters

        Args:
            value (float): new value to update the tracker with
            num (int: 1): number of elements used to compute the value
        """
        self.value = value
        self.sum += value
        self.count += num
        self.average = self.sum / self.count


class ConfusionMatrix:
    """Store, update and plot a confusion matrix."""

    def __init__(self, classes: dict):
        """ Create and initialize a confusion matrix

        Args:
            classes (dict): dictionary containing all the classes (e.g. {"0": "label_0", "1": "label_1",...})
        """
        self.classes = classes
        self.num_classes = len(self.classes)
        self.labels_classes = range(self.num_classes)
        self.list_classes = list(self.classes.values())

        self.cm = np.zeros([len(classes), len(classes)], dtype=np.int)

    def update_confusion_matrix(self, targets: torch.Tensor, predictions: torch.Tensor):
        """ Update the confusion matrix

        Args:
            targets (torch.Tensor): tensor on the cpu containing the target classes
            predictions(torch.Tensor): tensor on the cpu containing the predicted classes
        """
        # use sklearn to update the confusion matrix
        self.cm += confusion_matrix(targets, predictions, labels=self.labels_classes)

    def plot_confusion_matrix(
        self,
        normalize: bool = True,
        title: str = None,
        cmap: matplotlib.colors.Colormap = plt.cm.Blues,
    ) -> matplotlib.figure.Figure:
        """
        This function plots the confusion matrix.

        Args:
            normalize (bool: True): boolean to control the normalization of the confusion matrix.
            title (str: ""): title for the figure
            cmap (matplotlib.colors.Colormap: plt.cm.Blues): color map, defaults to 'Blues'

        Returns:
            matplotlib.figure.Figure: the ready-to-show/save figure
        """
        if not title:
            title = f"Normalized Confusion Matrix" if normalize else f"Confusion Matrix"

        if normalize:
            self.cm = self.cm.astype("float") / np.maximum(
                self.cm.sum(axis=1, keepdims=True), 1
            )

        # Create figure with size determined by number of classes.
        fig, ax = plt.subplots(
            figsize=[0.4 * self.num_classes + 4, 0.4 * self.num_classes + 2]
        )
        im = ax.imshow(self.cm, interpolation="nearest", cmap=cmap)
        ax.figure.colorbar(im, ax=ax)

        # Show all ticks and label them with the respective list entries.
        # Add a tick at the start and end in order to not cut off the figure.
        ax.set(
            xticks=np.arange(-1, self.cm.shape[1] + 1),
            yticks=np.arange(-1, self.cm.shape[0] + 1),
            xticklabels=[""] + self.list_classes,
            yticklabels=[""] + self.list_classes,
            title=title,
            ylabel="True label",
            xlabel="Predicted label",
        )

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = ".2f" if normalize else "d"
        thresh = self.cm.max() / 2.0
        for i in range(self.cm.shape[0]):
            for j in range(self.cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(self.cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if self.cm[i, j] > thresh else "black",
                )
        fig.tight_layout()

        return fig
