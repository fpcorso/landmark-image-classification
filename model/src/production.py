import os

import numpy as np
import torch
import torch.nn.functional as functional
import torchvision.transforms as transforms
from torch import nn
from torchvision import datasets

from .data import get_landmark_data_location


class Predictor(nn.Module):
    def __init__(self, model, class_names, mean, std):
        super().__init__()

        self.model = model.eval()
        self.class_names = class_names

        # We use nn.Sequential and not nn.Compose because the former
        # is compatible with torch.script, while the latter isn't
        self.transforms = nn.Sequential(
            transforms.Resize(
                [
                    256,
                ]
            ),  # We use single int value inside a list due to torchscript type restrictions
            transforms.CenterCrop(224),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean.tolist(), std.tolist()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            x = self.model(x)
            x = functional.softmax(x, dim=1)

            return x


def predictor_test(model_reloaded):
    """
    Test the predictor. Since the predictor does not operate on the same tensors
    as the non-wrapped model, we need a specific test function (can't use one_epoch_test)
    """

    folder = get_landmark_data_location()
    test_data = datasets.ImageFolder(
        os.path.join(folder, "test"), transform=transforms.ToTensor()
    )

    pred = []
    truth = []
    for x in test_data:
        softmax = model_reloaded(x[0].unsqueeze(dim=0))

        idx = softmax.squeeze().argmax()

        pred.append(int(x[1]))
        truth.append(int(idx))

    pred = np.array(pred)
    truth = np.array(truth)

    print(f"Accuracy: {(pred==truth).sum() / pred.shape[0]}")

    return truth, pred
