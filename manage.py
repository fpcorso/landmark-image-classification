import typer
from src.data import cache_data_mean_and_std, download_and_extract_data

app = typer.Typer()


@app.command()
def setup():
    typer.echo("Initializing project...")

    download_and_extract_data()
    cache_data_mean_and_std()

    typer.echo("Initialization complete")


@app.command()
def hello():
    typer.echo("Hello!")


if __name__ == "__main__":
    app()
