from fastapi import FastAPI, Request, Form, File, UploadFile, Body
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pydantic import BaseModel
from make_prediction import make_prediction
from typing import List
import json
import pandas as pd
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

@app.get("/")
async def root(request: Request, message='Выкладывай свои семплы, дружок)'):
    return templates.TemplateResponse('index.html',
                                      {"request": request,
                                       "message": message})


@app.post("/predict_item")
def predict_item(item: Item) -> float:

    values = [v for k, v in dict(item).items()]
    columns = [k for k, v in dict(item).items()]
    df_sample = pd.DataFrame(data=values).T
    df_sample.columns = columns

    y_sample_pred = make_prediction(df_sample)

    return y_sample_pred[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:

    y_sample_pred = make_prediction(items)

    return y_sample_pred

# для одного объекта
@app.post("/upload_one_object")
def upload_single(
        name: str = Form()
    ):
    print(name)
    name = json.loads(name)
    y_pred = predict_item(name)

    return {'price': y_pred}

# для нескольких объектов
@app.post("/upload_object")
async def upload(request: Request,
                 name: str = Form(...),
                 item_file: UploadFile = File(...)):
    file_name = '_'.join(name.split()) + '.csv'
    save_path = f'static/objects/{file_name}'

    sample = pd.read_csv(item_file.file)

    sample['price'] = predict_items(sample)

    sample.to_csv(save_path, index=False)

    return FileResponse(save_path, filename=file_name)
