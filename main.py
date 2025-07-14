from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from pydantic import BaseModel
import os
import time
import csv
from typing import List

app = FastAPI()

# Разрешаем CORS для всех источников (можно ограничить при необходимости)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка моделей (все открытые, токен не нужен)
print("Загрузка моделей...")
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
fake_news_model = pipeline("text-classification", model="mariagrandury/roberta-base-fakenews-liar")
hate_speech_model = pipeline("text-classification", model="cardiffnlp/twitter-roberta-hate")
print("Готово.")

# Структура данных
class PostData(BaseModel):
    text: str
    username: str
    timestamp: float

# CSV лог-файл
CSV_LOG = "analysis_log.csv"
if not os.path.exists(CSV_LOG):
    with open(CSV_LOG, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["username", "text", "timestamp", "toxicity", "fake_news", "hate_speech"])

def analyze_text_with_models(text: str):
    tox = toxicity_model(text)[0]
    hate = hate_speech_model(text)[0]

    # Обработка fake-news
    fake_raw = fake_news_model(text)[0]
    fake = {"label": fake_raw["label"].lower(), "score": fake_raw["score"]}

    # Преобразуем топ-к результатов toxicity и hate
    tox_label = tox["label"].lower()
    tox_score = tox["score"]
    hate_label = hate["label"].lower()
    hate_score = hate["score"]

    return {
        "toxicity": {"label": tox_label, "score": tox_score},
        "fake_news": fake,
        "hate_speech": {"label": hate_label, "score": hate_score}
    }

@app.post("/analyze")
async def analyze_post(data: PostData):
    result = analyze_text_with_models(data.text)

    # Сохраняем в CSV
    with open(CSV_LOG, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            data.username,
            data.text,
            data.timestamp,
            f"{result['toxicity']['label']} ({result['toxicity']['score']:.2f})",
            f"{result['fake_news']['label']} ({result['fake_news']['score']:.2f})",
            f"{result['hate_speech']['label']} ({result['hate_speech']['score']:.2f})"
        ])

    return result

@app.get("/")
async def root():
    return {"message": "Добро пожаловать в API анализа цифрового поведения!"}
