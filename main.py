from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загружаем модели один раз при старте
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
fake_news_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
hate_speech_classifier = pipeline(
    "text-classification",
    model="Hate-speech-CNERG/bert-base-uncased-hatexplain",
    top_k=None
)
@app.get("/")
def root():
    return {"message": "Цифровой след API работает"}

@app.post("/analyze/")
async def analyze_text(text: str = Form(...)):
    toxicity = toxicity_classifier(text)
    fake_news = fake_news_classifier(text, candidate_labels=["fake", "real", "satire", "political", "conspiracy"])
    hate_speech = hate_speech_classifier(text)

    return {
        "toxicity": toxicity,
        "fake_news": fake_news,
        "hate_speech": hate_speech
    }
