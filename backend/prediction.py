import io
from PIL import Image
import torch
import torchvision.transforms as transforms


model = torch.jit.load("transfer_exported.pt")
model.eval()


def predict(image_bytes):
    tensor = transforms.ToTensor()(image_bytes).unsqueeze_(0)

    softmax = model(tensor).data.cpu().numpy().squeeze()

    # Get the indexes of the classes ordered by softmax
    # (larger first)
    idxs = np.argsort(softmax)[::-1]

    # Loop over the classes with the largest softmax
    for i in range(5):
        # Get softmax value
        p = softmax[idxs[i]]

        # Get class name
        landmark_name = model.class_names[idxs[i]]

        labels[i].value = f"{landmark_name} (prob: {p:.2f})"
