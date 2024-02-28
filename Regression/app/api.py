from fastapi import FastAPI  #To create the api
from pydantic import BaseModel #To provide a structure for the responses
import dill  #To read the pickle file/gradient booster model on the server
import pandas as pd 
from utils import Preprocessor #Our own created function

#Create an api
app=FastAPI()

#load the model
with open('gb.pkl','rb') as f:
    model=dill.load(f)

#type checking class using pydantic
class ScoringItem(BaseModel):
    TransactionDate: str
    HouseAge: float
    DistanceToStation: float
    NumberOfPubs:float
    PostCode: str

@app.post('/')
async def scoring_item(item:ScoringItem):
    df=pd.DataFrame([item.dict().values()],columns=item.dict().keys())
    yhat=model.predict(df)
    return {"Prediction":int(yhat)}

