from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from pydantic import BaseModel
import csv
import os

app = FastAPI()

# Разрешаем запросы со всех источников
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загружаем модели
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
fake_news_model = pipeline("text-classification", model="mariagrandury/roberta-base-fakenews-liar")  # ✅ публичная модель
hate_speech_model = pipeline("text-classification", model="cardiffnlp/twitter-roberta-hate")

class PostData(BaseModel):
    text: str
    username: str
    timestamp: float

LOG_FILE = "analysis_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["username", "text", "timestamp", "toxicity", "fake_news", "hate_speech"])

@app.post("/analyze")
async def analyze(data: PostData):
    tox = toxicity_model(data.text)[0]
    fake = fake_news_model(data.text)[0]
    hate = hate_speech_model(data.text)[0]

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([data.username, data.text, data.timestamp, tox["label"], fake["label"], hate["label"]])

    return {
        "toxicity": tox,
        "fake_news": fake,
        "hate_speech": hate,
    }

@app.get("/")
async def root():
    return {"message": "API is running."}
