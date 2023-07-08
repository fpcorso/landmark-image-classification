from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prediction import predict

app = FastAPI(docs_url=None, redoc_url=None)

origins = ["http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    with open("../model/data/raw/landmark_images/test/07.Stonehenge/7cadd9ffc1d24563.jpg", 'rb') as f:
        img = f.read()
    return {"Hello": predict(img)}
