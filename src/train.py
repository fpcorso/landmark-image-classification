import mlflow
import torch
from tqdm import tqdm

from src.helpers import get_device


def train_one_epoch(train_dataloader, model, optimizer, loss):
    """
    Performs one train_one_epoch epoch
    """

    device = get_device()
    model.to(device)

    model.train()

    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        data, target = data.to(device), target.to(device)

        # Forward pass.
        optimizer.zero_grad()
        output = model(data)

        # Backpropagation.
        loss_value = loss(output, target)
        loss_value.backward()
        optimizer.step()

        # update average training loss
        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )

    return train_loss


def valid_one_epoch(valid_dataloader, model, loss):
    """
    Validate at the end of one epoch
    """

    with torch.no_grad():
        model.train()

        device = get_device()
        model.to(device)

        valid_loss = 0.0
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            data, target = data.to(device), target.to(device)

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # 2. calculate the loss
            loss_value = loss(output, target)

            # Calculate average validation loss
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )

    return valid_loss


def optimize(
    data_loaders,
    model,
    optimizer,
    loss,
    n_epochs,
    save_path,
):
    valid_loss_min = None
    logs = {}

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, threshold=0.01, patience=7, factor=0.5
    )

    # Early stopping parameters
    best_val_loss = float("inf")
    threshold = 0.001
    below_threshold = 0
    patience = 30

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(data_loaders["train"], model, optimizer, loss)

        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)

        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )

        # If the validation loss decreases by more than 1%, save the model
        if valid_loss_min is None or (
            (valid_loss_min - valid_loss) / valid_loss_min > 0.01
        ):
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")

            # Save the weights to save_path
            torch.save(model.state_dict(), save_path)

            valid_loss_min = valid_loss

        # Update learning rate, i.e., make a step in the learning rate scheduler
        scheduler.step(valid_loss)

        # Log our metrics to our current run in mlflow.
        mlflow.log_metrics(
            {
                "epoch_train_loss": train_loss,
                "epoch_valid_loss": valid_loss,
                "epoch_lr": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

        # Determine if this should stop early.
        validation_diff = best_val_loss - valid_loss
        if best_val_loss > valid_loss:
            best_val_loss = valid_loss
        if validation_diff < threshold:
            below_threshold += 1
            if below_threshold > patience:
                print(
                    "Validation loss has gone below threshold more than allowed times. Stopping early."
                )
                break


def one_epoch_test(test_dataloader, model, loss):
    # monitor test loss and accuracy
    test_loss = 0.0
    correct = 0.0
    total = 0.0

    # set the module to evaluation mode
    with torch.no_grad():
        # set the model to evaluation mode
        model.eval()

        device = get_device()
        model.to(device)

        for batch_idx, (data, target) in tqdm(
            enumerate(test_dataloader),
            desc="Testing",
            total=len(test_dataloader),
            leave=True,
            ncols=80,
        ):
            data, target = data.to(device), target.to(device)

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            logits = model(data)
            # 2. calculate the loss
            loss_value = loss(logits, target)

            # update average test loss
            test_loss = test_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss)
            )

            # convert logits to predicted class
            # HINT: the predicted class is the index of the max of the logits
            pred = logits.max(1)[1]

            # compare predictions to true label
            correct += torch.sum(
                torch.squeeze(pred.eq(target.data.view_as(pred))).cpu()
            )
            total += data.size(0)

    print("Test Loss: {:.6f}\n".format(test_loss))

    accuracy = 100.0 * correct / total

    print("\nTest Accuracy: %2d%% (%2d/%2d)" % (accuracy, correct, total))

    return test_loss, accuracy
