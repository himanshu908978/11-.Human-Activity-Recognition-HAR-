from fastapi import FastAPI,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
from model import inference
import os

CLASSES = [
    "walk",
    "talk",
    "stand",
    "sit",
    "smile",
    "eat",
    "drink",
    "laugh"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_headers = ["*"],
    allow_methods = ["*"]
)

@app.post("/Activity-recogniser")
async def recogniser(data:UploadFile = File(...)):
    file_location = f"temp_{data.filename}"

    with open(file_location,"wb") as buffer:
        buffer.write(await data.read())

    pred_class,conf = inference(file_location)
    conf = round(conf,4)
    os.remove(file_location)
    return{
        "pred_label":CLASSES[pred_class],
        "conf":conf*100
    }