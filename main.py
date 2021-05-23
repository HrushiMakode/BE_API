from model.predict import get_caption
import shutil
from fastapi import FastAPI, UploadFile, File

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def read_root():
    return f'Hello Word'


def save_file(file):
    with open('Images/image.jpg', "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


@app.post('/predict')
async def predict_caption(file: UploadFile = File(...)):
    save_file(file)
    caption = get_caption('Images/image.jpg')
    print(caption)
    return caption
