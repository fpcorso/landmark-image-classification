from io import BytesIO
import math
import os
from pathlib import Path
import torch
import torch.utils.data
from torchvision import datasets, transforms
import multiprocessing
import urllib.request
from zipfile import ZipFile

import matplotlib.pyplot as plt


def get_raw_data_location() -> Path:
    return Path('data', 'raw')


def get_landmark_data_location() -> Path:
    return Path('data', 'raw', 'landmark_images')


def get_raw_data_stats_cache_path() -> Path:
    return Path('data', 'raw', 'mean_and_std.pt')


def download_and_extract_data() -> None:
    """
    Download and extract the data if it is not already there.
    """
    if not os.path.exists(get_landmark_data_location()):
        print("Downloading dataset...")
        url = "https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip"
        with urllib.request.urlopen(url) as resp:
            with ZipFile(BytesIO(resp.read())) as fp:
                fp.extractall(get_raw_data_location())

        print("Download complete.")


def get_data_loaders(
    batch_size: int = 32,
    valid_size: float = 0.2,
    num_workers: int = -1,
    limit: int = -1,
) -> dict:
    """
    Create and returns the train_one_epoch, validation and test data loaders.

    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset to use for validation. For example 0.2
                       means that 20% of the dataset will be used for validation
    :param num_workers: number of workers to use in the data loaders. Use -1 to mean
                        "use all my cores"
    :param limit: maximum number of data points to consider
    :return a dictionary with 3 keys: 'train_one_epoch', 'valid' and 'test' containing respectively the
            train_one_epoch, validation and test data loaders
    """

    if num_workers == -1:
        # Use all cores.
        num_workers = multiprocessing.cpu_count()

    data_loaders = {"train": None, "valid": None, "test": None}
    base_path = get_landmark_data_location()
    mean, std = get_data_mean_and_std()

    # Get our transforms.
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }

    # Create train and validation datasets.
    train_data = datasets.ImageFolder(
        str(base_path / "train"),
        transform=data_transforms["train"],
    )
    valid_data = datasets.ImageFolder(
        str(base_path / "train"),
        transform=data_transforms["valid"],
    )

    # obtain training indices that will be used for validation
    n_tot = len(train_data)
    indices = torch.randperm(n_tot)

    # If requested, limit the number of data points to consider
    if limit > 0:
        indices = indices[:limit]
        n_tot = limit

    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    # prepare data loaders
    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
    )

    # Now create the test data loader
    test_data = datasets.ImageFolder(
        str(base_path / "test"),
        transform=data_transforms["test"],
    )

    if limit > 0:
        indices = torch.arange(limit)
        test_sampler = torch.utils.data.SubsetRandomSampler(indices)
    else:
        test_sampler = None

    data_loaders["test"] = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        shuffle=False,
    )

    return data_loaders


def get_data_mean_and_std() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the mean and standard deviation of the dataset.
    """
    cache_file = get_raw_data_stats_cache_path()
    if not os.path.exists(cache_file):
        cache_data_mean_and_std()

    d = torch.load(cache_file)
    return d["mean"], d["std"]


def cache_data_mean_and_std() -> None:
    """
    Cache the mean and standard deviation of the dataset.
    """
    cache_file = get_raw_data_stats_cache_path()
    data_folder = str(get_landmark_data_location())
    if not os.path.exists(cache_file):
        print("Calculating mean and standard deviation...")
        ds = datasets.ImageFolder(
            data_folder, transform=transforms.Compose([transforms.ToTensor()])
        )
        dl = torch.utils.data.DataLoader(
            ds, batch_size=1, num_workers=multiprocessing.cpu_count()
        )

        mean = 0.0
        for images, _ in dl:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
        mean = mean / len(dl.dataset)

        var = 0.0
        npix = 0
        for images, _ in dl:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
            npix += images.nelement()

        std = torch.sqrt(var / (npix / 3))

        # Cache results so we don't need to redo the computation
        torch.save({"mean": mean, "std": std}, cache_file)


def visualize_one_batch(data_loaders, max_n: int = 5):
    """
    Visualize one batch of data.

    :param data_loaders: dictionary containing data loaders
    :param max_n: maximum number of images to show
    :return: None
    """

    # Get a sample of images and labels.
    images, labels = next(iter(data_loaders["train"]))

    # Undo the normalization (for visualization purposes).
    mean, std = get_data_mean_and_std()
    inv_trans = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    images = inv_trans(images)

    # Get class names from the train data loader.
    class_names = [a.split(".")[1] for a in data_loaders["train"].dataset.classes]

    # Convert from BGR (the format used by pytorch) to
    # RGB (the format expected by matplotlib)
    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        ax.set_title(class_names[labels[idx].item()])
