from fastapi import FastAPI, UploadFile, File
from typing import Annotated
import pickle


app = FastAPI()

with open("./yolo.pkl", "rb") as file:
    # Load data from the pickle file
    model = pickle.load(file)


@app.get("/")
def hello():
    return "hello from fastapi"


@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}


@app.post("/upload/image/")
async def upload_image(file: UploadFile = File(...)):
    result = model(file.filename)
    return result
