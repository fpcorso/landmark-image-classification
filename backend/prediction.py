import io
from PIL import Image
import torch
import torchvision.transforms as transforms


model = torch.jit.load("../model/checkpoints/transfer_exported.pt")
model.eval()


def predict(image_bytes: bytes) -> str:
    """Predict the landmark class of an image using a trained model."""

    # Convert image to tensor.
    image = Image.open(io.BytesIO(image_bytes))
    tensor = transforms.ToTensor()(image).unsqueeze_(0)

    # Make prediction.
    softmax = model(tensor)

    # Get index of predicted class.
    _, y_hat = softmax.max(1)
    predicted_idx = int(y_hat.item())

    return model.class_names[predicted_idx].split('.')[1]
