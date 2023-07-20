# Landmark Image Classification

This is a quick proof-of-concept for building a model for image classification and then deploying it to be used in a web app using Fast API for backend and Vue for frontend.

The model is built using Pytorch and I use MLflow to track experiments and log metrics.

The model and training data was originally part of a project I completed during the Udacity Deep Learning Nanodegree. The training data was provided by Udacity and is a subset of the Google Landmark Recognition dataset.

## How the project is set up

The three components each have their own folder with their own readme within the project.

### Model

All the code for training, building, and deploying the model is within the `model` folder. I built one model from scratch using a CNN architecture and then a second one using transfer learning built on the ResNet50 model.

### Backend

The backed is built in Python using Fast API and can be found in the `backend` folder. The backend is responsible for serving the model and making predictions via the /predict endpoint.

### Frontend

The frontend is built in Vue and can be found in the `frontend` folder. The frontend is responsible for displaying the web app and making requests to the backend.
