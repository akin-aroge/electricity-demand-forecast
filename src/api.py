from fastapi import FastAPI
from pydantic import BaseModel

from src.inference import main as inf_main


app = FastAPI()

class InputData(BaseModel):
  days: float
  model_name: str

@app.get("/")
async def root():
    return {"message": "Electricity Predictor"}

@app.get("/predict")
async def predict(data: InputData):
  # Access data from the request body
  days = data.days
  model_name = data.model_name

  prediction = inf_main.main(model_name=model_name, n_days=days)
  return {"prediction": prediction}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
