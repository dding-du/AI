from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# rag_search_txt.py가 프로젝트 루트에 있으므로 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from rag_search_txt import run_rag

app = FastAPI()

class Query(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "RAG Search API running"}

@app.post("/search")
def search(body: Query):
    result = run_rag(body.query)
    return {"result": result}
