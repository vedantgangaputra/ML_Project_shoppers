# This file defines FastAPI app with prediction, health check and frontend endpoints

from pathlib import Path
from typing import Union
from fastapi import FastAPI, HTTPException, Request  # fastapi framework
from fastapi.staticfiles import StaticFiles  # serve static files
from fastapi.templating import Jinja2Templates  # serve html templates
from fastapi.responses import HTMLResponse  # html response
from pydantic import BaseModel  # request body validation
import sys  # system path manipulation
import os  # access env variables
from dotenv import load_dotenv  # load env variables

load_dotenv()  # load .env file

base_dir = Path(__file__).resolve().parent
root_dir = base_dir.parent.parent
sys.path.append(str(root_dir / "ml"))

from predict import predict  # import predict function

app = FastAPI(
    title="Online Shoppers Intention API",
    description="Predicts if a shopper will purchase or not",
    version="1.0.0"
)

# mount static files directory
app.mount("/static", StaticFiles(directory=str(base_dir)), name="static")

# setup jinja2 templates
templates = Jinja2Templates(directory=str(base_dir / "templets"))

# define input schema
class ShopperInput(BaseModel):
    Administrative: int
    Administrative_Duration: float
    Informational: int
    Informational_Duration: float
    ProductRelated: int
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float
    Month: Union[str, int]
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: Union[str, int]
    Weekend: int

# define output schema
class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    message: str

@app.get("/", response_class=HTMLResponse)  # serve frontend
def frontend(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})

@app.get("/health")  # health check endpoint
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionOutput)  # prediction endpoint
def predict_endpoint(data: ShopperInput):
    try:
        input_dict = data.model_dump()  # convert to dict
        result = predict(input_dict)    # run prediction
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))