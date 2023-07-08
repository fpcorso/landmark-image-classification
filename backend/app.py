from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prediction import predict

app = FastAPI(docs_url=None, redoc_url=None)

origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict_landmark(image: UploadFile):
    img = await image.read()
    return {"landmark": predict(img)}
