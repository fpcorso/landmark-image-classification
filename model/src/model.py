import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 5120),
            nn.BatchNorm1d(5120),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(5120, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        self.model = nn.Sequential(self.features, nn.Flatten(), self.classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_transfer_learning_model(model_name="resnet18", n_classes=50):
    """Creates a model using an existing model as the foundation and
    replacing the linear layer to meet the needs for this model."""

    # Get the requested architecture
    if hasattr(models, model_name):
        model_transfer = getattr(models, model_name)(weights="DEFAULT")

    else:
        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])

        raise ValueError(
            f"Model {model_name} is not known. List of available models: "
            f"https://pytorch.org/vision/{torchvision_major_minor}/models.html"
        )

    # Freeze all parameters in the model
    for p in model_transfer.parameters():
        p.freeze = True

    # Add the linear layer at the end with the appropriate number of classes
    num_ftrs = model_transfer.fc.in_features
    model_transfer.fc = nn.Sequential(nn.Linear(num_ftrs, n_classes))

    return model_transfer
