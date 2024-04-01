# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .similarity import TextSimilarityModel

app = FastAPI()

class Texts(BaseModel):
    text1: str
    text2: str

model = TextSimilarityModel()

@app.post("/similarity/")
async def get_similarity(texts: Texts):
    try:
        similarity_score = model.compute_similarity(texts.text1, texts.text2)
        return {"similarity score": similarity_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))