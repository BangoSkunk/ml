import numpy as np
from pydantic import BaseModel
from typing import List
import pickle
import re
from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
from fastapi.responses import StreamingResponse
from io import StringIO, BytesIO

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: int


class Items(BaseModel):
    objects: List[Item]


# DATA PREPROCESSING
def extract_engine(sample):
    if (type(sample) != str and np.isnan(sample)) or type(sample) == int or type(sample) == float:
        return sample
    return float(sample.replace(' CC', ''))

def extract_maxpower(sample):
    if (type(sample) != str and np.isnan(sample)) or type(sample) == int or type(sample) == float:
        return sample
    sample = sample.replace(' bhp', '')
    if len(sample) > 0:
        return float(sample)
    else:
        return np.nan

def extract_mileage(sample):
    if (type(sample) != str and np.isnan(sample)) or type(sample) == int or type(sample) == float:
        return sample
    mileage = float(sample.split(' ')[0])
    if 'kg' in sample:
        mileage *= 0.755
    return mileage

def extract_rpm(sample):
    if (type(sample) != str and np.isnan(sample)) or type(sample) == int or type(sample) == float:
        return sample
    if 'rpm' not in sample.lower():
        return np.nan
    rpm = ' '.join(sample.split(' ')[1:]).split('-')[-1]
    rpm = re.sub("[^\d\.\,]", "", rpm)
    rpm = rpm.replace(',', '')
    return float(rpm)

def extract_torque(sample):
    if (type(sample) != str and np.isnan(sample)) or type(sample) == int or type(sample) == float:
        return sample
    torque = sample.split(' ')[0]
    torque = re.sub("[^\d\.\,]", "", torque)
    torque = float(torque)
    if 'kgm' in sample.lower():
        torque *= 10
    return torque


def fill_nans(item: Item,
              fill_dict: dict):
    for feature, value in fill_dict.items():
        if not hasattr(item, feature):
            continue
        current_feature = getattr(item, feature)
        if current_feature is None or current_feature == '':
            setattr(item, feature, value)
    return item

#OBJECTS PREPARATION
with open('objects_dict.pkl', 'rb') as f:
    objects_dict = pickle.load(f)

model = objects_dict['model']
train_median_dict = objects_dict['train_median_dict']
ohe = objects_dict['ohe']
cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
num_cols =['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']



# BASIC PREDICTION FUNC
def predict_price(item: Item) -> float:

    item = fill_nans(item, train_median_dict)

    engine = extract_engine(item.engine)
    maxpower = extract_maxpower(item.max_power)
    mileage = extract_mileage(item.mileage)
    rpm = extract_rpm(item.torque)
    torque = extract_torque(item.torque)
    # sorry for this
    if type(rpm) != int or type(rpm) != float:
        rpm = train_median_dict['max_torque_rpm']
        torque = train_median_dict['torque']

    cat_cols = np.array([[item.fuel, item.seller_type, item.transmission, item.owner, item.seats]], dtype=object)
    cat_cols = ohe.transform(cat_cols)
    num_cols = np.array([[item.year, item.km_driven, mileage, engine, maxpower, torque, rpm]])
    feature_vector = np.concatenate((num_cols, cat_cols), axis=-1)
    prediction = model.predict(feature_vector)
    return round(prediction[0], 2)


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    predicted_price = predict_price(item)
    return predicted_price


# @app.post("/predict_items")
# def predict_items(items: List[Item]) -> List[float]:
#     predicted_prices = list(map(predict_price, items))
#     return predicted_prices
def process_csv(file_content: str) -> pd.DataFrame:
    try:
        file_like_object = StringIO(file_content)
        df = pd.read_csv(file_like_object)
        return df
    except Exception as e:
        # Handle exceptions appropriately
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):
    try:
        content = await file.read()
        file_content = content.decode('utf-8')
        df = process_csv(file_content)

        df.engine = df.engine.apply(extract_engine)
        df.max_power = df.max_power.apply(extract_maxpower)
        df.mileage = df.mileage.apply(extract_mileage)
        df['torque'], df['max_torque_rpm'] = df.torque.apply(extract_torque), df.torque.apply(extract_rpm)
        df.fillna(value=train_median_dict)

        df_cat_cols = ohe.transform(df[cat_cols].to_numpy())
        df_num_cols = df[num_cols].to_numpy()
        df_cat_full = np.concatenate((df_num_cols, df_cat_cols), axis=-1)
        prediction = model.predict(df_cat_full)
        df['selling_price'] = prediction
        csv_content = df.to_csv(index=False)

        csv_bytes = BytesIO(csv_content.encode('utf-8'))
        return StreamingResponse(content=csv_bytes, media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=prediction_data.csv"})
    except Exception as e:
        # Handle exceptions appropriately
        raise HTTPException(status_code=500, detail=str(e))