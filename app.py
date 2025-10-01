from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
from typing import Optional


# Load your trained model
model = joblib.load("best_svm_model.pkl") 

# Define input data structure
class InputData(BaseModel):
    Time_spent_Alone: float = Field(4, ge=0, description= "Enter positive value")
    Stage_fear: str = Field("No" ,pattern=r"^(?i)(yes|no)$", description= "Enter 'Yes' or 'No'")  # Case insensitive Yes/No
    Social_event_attendance: float = Field(5 , ge=0, description= "Enter positive integer")
    Going_outside: float = Field(7, ge=0, description= "Enter positive integer")
    Drained_after_socializing: Optional[str] = Field(None, description= "Optional")  # Case insensitive Yes/No
    Friends_circle_size: float = Field(6, ge=0, description= "Enter positive integer")
    Post_frequency: float = Field(6, ge=0 , description= "Enter positive value")

app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    return {"message": "Personality Prediction API is running"}


@app.post("/predict")
def predict(data: InputData):
    # Convert Yes/No to 1/0
    stage_fear = 1 if data.Stage_fear == "Yes" else 0
    drained = None
    if data.Drained_after_socializing is not None:
        drained = 1 if data.Drained_after_socializing == "Yes" else 0
    
    # Only pass 6 features (ignoring 'drained' for prediction)
    features = [[
        data.Time_spent_Alone,
        stage_fear,
        data.Social_event_attendance,
        data.Going_outside,
        data.Friends_circle_size,
        data.Post_frequency
    ]]
    
    # Make prediction
    pred = model.predict(features)[0]

    if hasattr(pred, "item"):
        pred = pred.item()
    
    # Convert to label
    label = "Extrovert" if pred == 1 else "Introvert"
    
    return {
        "prediction": label,
        "note": "Drained_after_socializing feature accepted but not used in prediction"
    }