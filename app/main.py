
import os
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .config import settings

app = FastAPI(title="Intelligent Document Q&A", version="1.0")
_qa = None


def get_qa():
    
    global _qa
    if _qa is None:
        from .pipeline import DocQA
        _qa = DocQA()
    return _qa


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    use_rerank: bool = True
    hybrid: bool = False
    retrieve_k: int = Field(20, ge=1, le=100)
    final_k: int = Field(4, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok", "vector_backend": settings.vector_backend}


@app.post("/ingest")
def ingest(files: List[UploadFile] = File(...)):
    qa = get_qa()
    upload_dir = os.path.join(settings.data_dir, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    results = []
    for f in files:
        name = os.path.basename(f.filename or "")
        if not name.lower().endswith(".pdf"):
            raise HTTPException(400, f"Only PDF files are supported: {name!r}")
        path = os.path.join(upload_dir, name)
        with open(path, "wb") as out:
            out.write(f.file.read())
        results.append(qa.ingest_pdf(path, doc_id=name))
    return {"ingested": results}


@app.post("/query")
def query(req: QueryRequest):
    qa = get_qa()
    return qa.query(req.question, use_rerank=req.use_rerank, hybrid=req.hybrid,
                    retrieve_k=req.retrieve_k, final_k=req.final_k)
