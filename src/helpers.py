import os
import torch


def get_data_mean_and_std() -> tuple:
    """
    Returns the mean and standard deviation of the dataset.
    """
    cache_file = "mean_and_std.pt"
    if os.path.exists(cache_file):
        d = torch.load(cache_file)
        return d["mean"], d["std"]

    print('Missing cache file. Please run `python manage.py setup` first')
