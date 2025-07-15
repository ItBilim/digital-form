from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, MarianMTModel, MarianTokenizer
from typing import List
from pydantic import BaseModel
from langdetect import detect
from sklearn.metrics import classification_report

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NLP модели ---
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
fake_news_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
hate_speech_classifier = pipeline(
    "text-classification",
    model="Hate-speech-CNERG/bert-base-uncased-hatexplain",
    top_k=None
)

# --- Переводчик (русский -> английский) ---
translator_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
translator_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

def translate_ru_to_en(text: str) -> str:
    batch = translator_tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", padding=True)
    gen = translator_model.generate(**batch)
    return translator_tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

# --- API ---

@app.get("/")
def root():
    return {"message": "Цифровой след API работает"}

@app.post("/analyze/")
async def analyze_text(text: str = Form(...)):
    if detect(text) == "ru":
        text = translate_ru_to_en(text)

    toxicity = toxicity_classifier(text)
    fake_news = fake_news_classifier(text, candidate_labels=["fake", "real", "satire", "political", "conspiracy"])
    hate_speech = hate_speech_classifier(text)

    return {
        "toxicity": toxicity,
        "fake_news": fake_news,
        "hate_speech": hate_speech
    }

# --- Модель оценки метрик ---
class EvalSample(BaseModel):
    text: str
    true_label: str  # toxic / non-toxic

@app.post("/evaluate/")
async def evaluate(samples: List[EvalSample]):
    y_true = []
    y_pred = []

    for sample in samples:
        text = sample.text
        if detect(text) == "ru":
            text = translate_ru_to_en(text)

        result = toxicity_classifier(text)
        # Универсально достаём label
        if isinstance(result, list) and isinstance(result[0], dict):
            predicted_label = result[0]['label']
        elif isinstance(result, list) and isinstance(result[0], list):
            predicted_label = result[0][0]['label']
        else:
            predicted_label = "unknown"

        y_true.append(sample.true_label.lower())
        y_pred.append(predicted_label.lower())

    report = classification_report(y_true, y_pred, output_dict=True)
    return report

