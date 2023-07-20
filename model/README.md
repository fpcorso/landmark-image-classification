# Landmark Image Classification Model

## Getting Started

### Install Dependencies

This project uses pip and requirements.txt for managing dependencies. To get started, run `pip install -r requirements.txt` to install all dependencies.

### Setting Up The Project

This project uses `typer` to power some command line scripts which are located in `manage.py`. 

To get started, you can use the setup command using `python manage.py setup`. This will download the dataset and cache some needed values, such as ones for normalization.

## Training The Models

To train either the from scratch CNN model or the transfer learning model, you can use the relevant Jupyter notebook. The notebooks train the model and export the final model to be used.