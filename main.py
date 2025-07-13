from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
import uuid
from transformers import pipeline
from datetime import datetime
import os
import csv

token = os.getenv("HUGGINGFACE_TOKEN")
assert token is not None, "❌ HUGGINGFACE_TOKEN не установлен!"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

conn = sqlite3.connect("social_behavior.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS posts (
    id TEXT PRIMARY KEY,
    text TEXT,
    toxicity TEXT,
    fake_label TEXT,
    fake_score REAL,
    hate_label TEXT,
    hate_score REAL,
    created_at TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS interactions (
    id TEXT PRIMARY KEY,
    post_id TEXT,
    action TEXT,
    timestamp TEXT
)
""")
conn.commit()

class PostIn(BaseModel):
    text: str
    toxicity: Optional[dict]
    fake_news: Optional[dict]
    hate_speech: Optional[dict]

class PostOut(BaseModel):
    text: str
    toxicity: dict
    fake_news: dict
    hate_speech: dict

class InteractionIn(BaseModel):
    post_id: str
    action: str

print("Загрузка моделей...")
toxicity_model = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    top_k=None,
    use_auth_token=token
)
fake_news_model = pipeline(
    "text-classification",
    model="microsoft/deberta-v3-small-mnli"
)
hate_speech_model = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-hate",
    use_auth_token=token
)
print("Модели загружены ✅")

def analyze_text_with_models(text: str):
    tox_results = toxicity_model(text)[0]
    toxicity = {r["label"].lower(): r["score"] for r in tox_results}

    fake_raw = fake_news_model(text, truncation=True)[0]
    fake_label = fake_raw["label"]
    fake_score = fake_raw["score"]

    if fake_label == "LABEL_0":
        fake = {"label": "fake", "score": fake_score}
    elif fake_label == "LABEL_2":
        fake = {"label": "real", "score": fake_score}
    else:
        fake = {"label": "neutral", "score": fake_score}

    hate_res = hate_speech_model(text)[0]
    hate = {"label": hate_res["label"], "score": hate_res["score"]}

    return {
        "toxicity": toxicity,
        "fake_news": fake,
        "hate_speech": hate
    }

def export_csv():
    with sqlite3.connect("social_behavior.db") as conn:
        c = conn.cursor()
        c.execute("SELECT id, text, toxicity, fake_label, fake_score, hate_label, hate_score, created_at FROM posts")
        with open("posts_export.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "text", "toxicity", "fake_label", "fake_score", "hate_label", "hate_score", "created_at"])
            writer.writerows(c.fetchall())

        c.execute("SELECT id, post_id, action, timestamp FROM interactions")
        with open("interactions_export.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "post_id", "action", "timestamp"])
            writer.writerows(c.fetchall())

export_csv()

@app.post("/api/analyze")
async def analyze_text(item: PostIn):
    result = analyze_text_with_models(item.text)
    return result

@app.post("/api/save_post")
async def save_post(post: PostIn):
    pid = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    cursor.execute("""
        INSERT INTO posts (id, text, toxicity, fake_label, fake_score, hate_label, hate_score, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        pid,
        post.text,
        str(post.toxicity),
        post.fake_news["label"],
        post.fake_news["score"],
        post.hate_speech["label"],
        post.hate_speech["score"],
        created_at
    ])
    conn.commit()
    export_csv()
    return {"status": "saved", "id": pid}

@app.get("/api/posts", response_model=List[PostOut])
async def get_posts():
    cursor.execute("SELECT text, toxicity, fake_label, fake_score, hate_label, hate_score FROM posts ORDER BY rowid DESC")
    rows = cursor.fetchall()
    posts = []
    for r in rows:
        posts.append({
            "text": r[0],
            "toxicity": eval(r[1]),
            "fake_news": {"label": r[2], "score": r[3]},
            "hate_speech": {"label": r[4], "score": r[5]}
        })
    return posts

@app.post("/api/interact")
async def record_interaction(interaction: InteractionIn):
    iid = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    cursor.execute("""
        INSERT INTO interactions (id, post_id, action, timestamp)
        VALUES (?, ?, ?, ?)
    """, [iid, interaction.post_id, interaction.action, timestamp])
    conn.commit()
    export_csv()
    return {"status": "logged", "interaction_id": iid}

@app.get("/api/export/posts")
def download_posts():
    return FileResponse("posts_export.csv", media_type="text/csv", filename="posts_export.csv")

@app.get("/api/export/interactions")
def download_interactions():
    return FileResponse("interactions_export.csv", media_type="text/csv", filename="interactions_export.csv")

@app.get("/")
async def root():
    return {"message": "Digital Form API is running!"}
