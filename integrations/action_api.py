from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .db_mock import update_record

app = FastAPI(title="Voice AI Action API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UpdateRecordRequest(BaseModel):
    record_id: str
    value: str


class WeatherRequest(BaseModel):
    location: str


@app.post("/api/actions/update_record")
def update_record_action(request: UpdateRecordRequest):
    return update_record(request.record_id, request.value)


@app.post("/api/actions/get_weather")
def get_weather(request: WeatherRequest):
    return {
        "ok": True,
        "location": request.location,
        "weather": "Mock weather data",
        "temperature": "28°C"
    }


@app.get("/")
def root():
    return {"message": "Action API is running"}