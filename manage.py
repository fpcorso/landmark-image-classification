from io import BytesIO
import os
import torch
import torch.utils.data
import typer
import urllib.request
from zipfile import ZipFile
from torchvision import datasets, transforms
import multiprocessing

app = typer.Typer()


@app.command()
def setup():
    typer.echo("Initializing database...")

    data_folder = "landmark_images"

    if not os.path.exists(data_folder):
        typer.echo("Downloading dataset...")
        url = "https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip"
        with urllib.request.urlopen(url) as resp:
            with ZipFile(BytesIO(resp.read())) as fp:
                fp.extractall(".")

        typer.echo("Download complete.")

    cache_file = "mean_and_std.pt"

    if not os.path.exists(cache_file):
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


if __name__ == "__main__":
    app()
