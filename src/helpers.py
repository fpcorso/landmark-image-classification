import torch


def get_device() -> torch.device:
    """Gets the device to use for training."""
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
