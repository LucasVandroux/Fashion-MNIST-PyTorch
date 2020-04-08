import argparse
import datetime
import json
import os
import random
import shutil
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms, datasets, models

import numpy as np
from prefetch_generator import BackgroundGenerator

from models.SimpleCNNModel import SimpleCNNModel
import utils

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["SimpleCNNModel", "ResNet18"],
    help="Name of the model to train.",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="",
    help="Name of the experiement, it will be used to name the directory to store all the data to track and restart the experiment.",
)
parser.add_argument(
    "--percentage_validation_set",
    type=int,
    default=10,
    choices=range(0, 101),
    metavar="[0-100]",
    help="Percentage of data from the training set that should be used as a validation set.",
)
parser.add_argument(
    "--patience",
    type=int,
    default=-1,
    help="Number of epochs without improvements on the validation set's accuracy before stopping the training. Use -1 to desactivate early stopping.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    choices=range(1, 1025),
    metavar="[1-1024]",
    help="Batch size.",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=30,
    choices=range(1, 1001),
    metavar="[1-1000]",
    help="Maximum number of epochs.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=1,
    choices=range(1, 65),
    metavar="[1-64]",
    help="Number of workers.",
)
parser.add_argument("--manual_seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--learning_rate", required=True, type=float, help="Initial learning rate."
)
parser.add_argument("--momentum", default=0, type=float, help="Momentum.")
parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay.")
parser.add_argument("--nesterov", help="Enable Nesterov momentum.", action="store_true")
parser.add_argument(
    "--path_classes",
    default=os.path.join("models", "classes.json"),
    type=str,
    help="Path to the json containing the classes of the Fashion MNIST.",
)
parser.add_argument(
    "--random_crop",
    help="Add random cropping at the beggining of the transformations list.",
    action="store_true",
)
parser.add_argument(
    "--random_erasing",
    help="Add random erasing at the end of the transformations list.",
    action="store_true",
)
parser.add_argument(
    "--convert_to_RGB",
    help="Repeat the image over 3 channels to convert it to RGB.",
    action="store_true",
)
parser.add_argument(
    "--pretrained_weights",
    help="Use pretrained weights if possible with the model.",
    action="store_true",
)

args = parser.parse_args()


def main():
    # --- SETUP ---
    torch.backends.cudnn.benchmark = True

    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed_all(args.manual_seed)

    # Create experiement name if it doesn't exist
    if not args.experiment_name:
        current_datetime = datetime.datetime.now()
        timestamp = current_datetime.strftime("%y%m%d-%H%M%S")
        args.experiment_name = f"{timestamp}_{args.model}"

    # Create the folder to store the results of all the experiments
    args.experiment_path = os.path.join("experiments", args.experiment_name)
    if not os.path.isdir(args.experiment_path):
        os.makedirs(args.experiment_path)

    # Import the classes
    with open(args.path_classes) as json_file:
        classes = json.load(json_file)

    # --- DATA ---
    # Generate the transformations
    train_list_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]

    test_list_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]

    # Add random cropping to the list of transformations
    if args.random_crop:
        train_list_transforms.insert(0, transforms.RandomCrop(28, padding=4))

    # Add random erasing to the list of transformations
    if args.random_erasing:
        train_list_transforms.append(
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value="random",
                inplace=False,
            )
        )

    if args.convert_to_RGB:
        convert_to_RGB = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        train_list_transforms.append(convert_to_RGB)
        test_list_transforms.append(convert_to_RGB)

    # Train Data
    train_transform = transforms.Compose(train_list_transforms)

    train_dataset = datasets.FashionMNIST(
        root="data", train=True, transform=train_transform, download=True
    )

    # Define the size of the training set and the validation set
    train_set_length = int(
        len(train_dataset) * (100 - args.percentage_validation_set) / 100
    )
    val_set_length = int(len(train_dataset) - train_set_length)
    train_set, val_set = torch.utils.data.random_split(
        train_dataset, (train_set_length, val_set_length)
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )

    # Test Data
    test_transform = transforms.Compose(test_list_transforms)

    test_set = datasets.FashionMNIST(
        root="./data", train=False, transform=test_transform, download=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # --- MODEL ---
    if args.model == "SimpleCNNModel":
        model = SimpleCNNModel()

    elif args.model == "ResNet18":
        model = models.resnet18(pretrained=args.pretrained_weights)
        model.fc = nn.Linear(model.fc.in_features, len(classes))

    num_trainable_parameters = utils.count_parameters(
        model, only_trainable_parameters=True
    )

    # Load the model on the GPU if available
    if use_cuda:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )

    # Print information from the parameters and model before training
    print(
        f"Loaded {args.model} with {num_trainable_parameters} trainable parameters (GPU: {use_cuda})."
    )
    print(args)

    # --- MODEL TRAINING & TESTING ---
    start_num_iteration = 0
    start_epoch = 0
    best_accuracy = 0.0
    epochs_without_improvement = 0
    purge_step = None

    # Restore the last checkpoint if available
    checkpoint_filepath = os.path.join(args.experiment_path, "checkpoint.pth.tar")
    if os.path.exists(checkpoint_filepath):
        print(f"Restoring last checkpoint from {checkpoint_filepath}...")
        checkpoint = torch.load(checkpoint_filepath)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        start_num_iteration = checkpoint["num_iteration"] + 1
        best_accuracy = checkpoint["best_accuracy"]
        purge_step = start_num_iteration
        print(
            f"Last checkpoint restored. Starting at epoch {start_epoch + 1} with best accuracy at {100 * best_accuracy:05.3f}."
        )

    # Create the tensorboard summary writers for training and validation steps
    train_writer = SummaryWriter(
        os.path.join(args.experiment_path, "train"), purge_step=purge_step
    )
    valid_writer = SummaryWriter(
        os.path.join(args.experiment_path, "valid"), purge_step=purge_step
    )
    test_writer = SummaryWriter(
        os.path.join(args.experiment_path, "test"), purge_step=purge_step
    )

    # main training loop
    num_iteration = start_num_iteration
    for epoch in range(start_epoch, args.num_epochs):

        # --- TRAIN ---

        num_iteration, _, _, _, _ = train(
            model=model,
            classes=classes,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            num_iteration=num_iteration,
            use_cuda=use_cuda,
            tensorboard_writer=train_writer,
        )

        # --- VALID ---

        is_best = False

        _, valid_accuracy_top1, _, _ = test(
            model=model,
            classes=classes,
            data_loader=val_loader,
            criterion=criterion,
            epoch=epoch,
            num_iteration=num_iteration,
            use_cuda=use_cuda,
            tensorboard_writer=valid_writer,
            name_step="Valid",
        )

        # Save the best model
        if valid_accuracy_top1 > best_accuracy:
            is_best = True
            best_accuracy = valid_accuracy_top1
            # Re-initialize epochs_without_improvement
            epochs_without_improvement = 0

        # Early stopping
        elif (args.patience >= 0) and (epochs_without_improvement >= args.patience):
            print(
                f"No improvement for the last {epochs_without_improvement} epochs, stopping the training (best accuracy: {100 * best_accuracy:05.2f})."
            )
            break

        else:
            epochs_without_improvement += 1

        utils.save_checkpoint(
            current_epoch=epoch,
            num_iteration=num_iteration,
            best_accuracy=best_accuracy,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            is_best=is_best,
            experiment_path=args.experiment_path,
        )

        # increment num_iteration after evaluation for the next epoch of training
        num_iteration += 1

    # --- TEST ---

    # Restore the best model to test it
    best_model_filepath = os.path.join(args.experiment_path, "model_best.pth.tar")
    if os.path.exists(best_model_filepath):
        print(f"Loading best model from {best_model_filepath}...")
        checkpoint = torch.load(best_model_filepath)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_accuracy = checkpoint["best_accuracy"]
        epoch = checkpoint["epoch"]
        num_iteration = checkpoint["num_iteration"]

    _, test_accuracy_top1, _, _ = test(
        model=model,
        classes=classes,
        data_loader=test_loader,
        criterion=criterion,
        epoch=epoch,
        num_iteration=num_iteration,
        use_cuda=use_cuda,
        tensorboard_writer=test_writer,
        name_step="Test",
    )

    # Print final accuracy of the best model on the test set
    print(
        f"Best {args.model} model has an accuracy of {100 * test_accuracy_top1:05.2f} on the Fashion MNIST test set."
    )


def train(
    model: nn.Module,
    classes: dict,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: nn.Module,
    epoch: int,
    num_iteration: int,
    use_cuda: bool,
    tensorboard_writer: torch.utils.tensorboard.SummaryWriter,
):
    """ Train a given model

    Args:
        model (nn.Module): model to train.
        classes (dict): dictionnary containing the classes and their indice.
        data_loader (torch.utils.data.DataLoader): data loader with the data to train the model on.
        criterion (nn.Module): loss function.
        optimizer (nn.Module): optimizer function.
        epoch (int): epoch of training.
        num_iteration (int): number of iterations since the beginning of the training.
        use_cuda (bool): boolean to decide if cuda should be used.
        tensorboard_writer (torch.utils.tensorboard.SummaryWriter): writer to write the metrics in tensorboard.

    Returns:
        num_iteration (int): number of iterations since the beginning of the training (increased during the training).
        loss (float): final loss
        accuracy_top1 (float): final accuracy top1
        accuracy_top5 (float): final accuracy top5
        confidence_mean (float): mean confidence
    """
    # Switch the model to train mode
    model.train()

    # Initialize the trackers for the loss and the accuracy
    loss_tracker = utils.MetricTracker()
    accuracy_top1_tracker = utils.MetricTracker()
    accuracy_top5_tracker = utils.MetricTracker()
    confidence_tracker = utils.MetricTracker()

    # Initialize confusing matrix
    confusion_matrix_tracker = utils.ConfusionMatrix(classes)

    # create BackgroundGenerator and wrap it in tqdm progress bar
    progress_bar = tqdm(
        BackgroundGenerator(data_loader, max_prefetch=32), total=len(data_loader)
    )

    for i, data in enumerate(progress_bar):
        inputs, targets = data

        # Save the inputs to the disk
        # img_grid = torchvision.utils.make_grid(inputs)
        # torchvision.utils.save_image(img_grid,"inputs.jpg")

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        confidence, prediction = outputs.topk(dim=1, k=5)

        # Backward pass and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss, accuracy and confidence
        loss_tracker.update(loss.item())
        accuracy_top1_tracker.update(
            (prediction[:, 0] == targets).sum().item(), targets.numel()
        )
        accuracy_top5_tracker.update(
            (prediction[:, :5] == targets[:, None]).sum().item(), targets.numel()
        )
        confidence_tracker.update(confidence[:, 0].sum().item(), targets.numel())

        # Update the confusion matrix
        confusion_matrix_tracker.update_confusion_matrix(targets.cpu(), prediction[:, 0].cpu())

        # Add the new values to the tensorboard summary writer
        tensorboard_writer.add_scalar("loss", loss_tracker.average, num_iteration)
        tensorboard_writer.add_scalar(
            "accuracy_top1", accuracy_top1_tracker.average, num_iteration
        )
        tensorboard_writer.add_scalar(
            "accuracy_top5", accuracy_top5_tracker.average, num_iteration
        )
        # tensorboard_writer.add_scalar(
        #     "confidence_mean", confidence_tracker.average, num_iteration
        # )

        # Update the progress_bar information
        progress_bar.set_description(f"Epoch {epoch + 1}/{args.num_epochs} Train")
        progress_bar.set_postfix(
            loss=f"{loss_tracker.average:05.5f}",
            accuracy_top1=f"{100 * accuracy_top1_tracker.average:05.2f}",
            accuracy_top5=f"{100 * accuracy_top5_tracker.average:05.2f}",
        )

        # Increment num_iteration on all iterations except the last,
        # so that the evaluation is logged to the correct iteration
        if i < len(data_loader) - 1:
            num_iteration += 1

    # Add the normalized confusion matrix to tensorboard and flush it
    tensorboard_writer.add_figure(
        "confusion_matrix", confusion_matrix_tracker.plot_confusion_matrix(normalize=True), num_iteration
    )
    tensorboard_writer.flush()

    return (
        num_iteration,
        loss_tracker.average,
        accuracy_top1_tracker.average,
        accuracy_top5_tracker.average,
        confidence_tracker.average,
    )


def test(
    model: nn.Module,
    classes: dict,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    epoch: int,
    num_iteration: int,
    use_cuda: bool,
    tensorboard_writer: torch.utils.tensorboard.SummaryWriter,
    name_step: str,
):
    """ Test a given model

    Args:
        model (nn.Module): model to test.
        classes (dict): dictionnary containing the classes and their indice.
        data_loader (torch.utils.data.DataLoader): data loader with the data to test the model on.
        criterion (nn.Module): loss function.
        epoch (int): epoch of training corresponding to the model.
        num_iteration (int): number of iterations since the beginning of the training corresponding to the model.
        use_cuda (bool): boolean to decide if cuda should be used.
        tensorboard_writer (torch.utils.tensorboard.SummaryWriter): writer to write the metrics in tensorboard.
        name_step (str): name of the step to write it in the description of the progress_bar

    Returns:
        loss (float): final loss
        accuracy_top1 (float): final accuracy top1
        accuracy_top5 (float): final accuracy top5
        confidence_mean (float): mean confidence
    """
    # Switch the model to eval mode
    model.eval()

    # Initialize the trackers for the loss and the accuracy
    loss_tracker = utils.MetricTracker()
    accuracy_top1_tracker = utils.MetricTracker()
    accuracy_top5_tracker = utils.MetricTracker()
    confidence_tracker = utils.MetricTracker()

    # Initialize confusing matrix
    confusion_matrix_tracker = utils.ConfusionMatrix(classes)

    # create BackgroundGenerator and wrap it in tqdm progress bar
    progress_bar = tqdm(
        BackgroundGenerator(data_loader, max_prefetch=32), total=len(data_loader)
    )

    for data in progress_bar:
        inputs, targets = data

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        confidence, prediction = outputs.topk(dim=1, k=5)

        # Track loss, accuracy and confidence
        loss_tracker.update(loss.item())
        accuracy_top1_tracker.update(
            (prediction[:, 0] == targets).sum().item(), targets.numel()
        )
        accuracy_top5_tracker.update(
            (prediction[:, :5] == targets[:, None]).sum().item(), targets.numel()
        )
        confidence_tracker.update(confidence[:, 0].sum().item(), targets.numel())

        # Update the confusion matrix
        confusion_matrix_tracker.update_confusion_matrix(targets.cpu(), prediction[:, 0].cpu())

        # Update the progress_bar information
        progress_bar.set_description(f"Epoch {epoch + 1}/{args.num_epochs} {name_step}")
        progress_bar.set_postfix(
            loss=f"{loss_tracker.average:05.5f}",
            accuracy_top1=f"{100 * accuracy_top1_tracker.average:05.2f}",
            accuracy_top5=f"{100 * accuracy_top5_tracker.average:05.2f}",
        )

    # Add the new values to the tensorboard summary writer
    tensorboard_writer.add_scalar("loss", loss_tracker.average, num_iteration)
    tensorboard_writer.add_scalar(
        "accuracy_top1", accuracy_top1_tracker.average, num_iteration
    )
    tensorboard_writer.add_scalar(
        "accuracy_top5", accuracy_top5_tracker.average, num_iteration
    )
    # tensorboard_writer.add_scalar(
    #     "confidence_mean", confidence_tracker.average, num_iteration
    # )

    tensorboard_writer.add_figure(
        "confusion_matrix", confusion_matrix_tracker.plot_confusion_matrix(normalize=True), num_iteration
    )
    tensorboard_writer.flush()

    return (
        loss_tracker.average,
        accuracy_top1_tracker.average,
        accuracy_top5_tracker.average,
        confidence_tracker.average,
    )


if __name__ == "__main__":
    main()
