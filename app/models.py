"""
Pydantic models untuk Portfolio Chatbot API
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model untuk endpoint /chat"""
    message: str = Field(
        ...,
        description="Pertanyaan atau pesan dari user",
        min_length=1,
        max_length=1000
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Apa saja skill programming yang kamu miliki?"
            }
        }


class SourceDocument(BaseModel):
    """Model untuk dokumen sumber yang digunakan dalam response"""
    content: str = Field(..., description="Isi konten dokumen")
    metadata: Optional[dict] = Field(default=None, description="Metadata dokumen")


class ChatResponse(BaseModel):
    """Response model untuk endpoint /chat"""
    response: str = Field(..., description="Jawaban dari chatbot")
    sources: List[SourceDocument] = Field(
        default=[],
        description="Dokumen sumber yang digunakan untuk menjawab"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Saya memiliki skill dalam Python, JavaScript, dan FastAPI.",
                "sources": [
                    {
                        "content": "Skills: Python, JavaScript, FastAPI...",
                        "metadata": {"source": "portfolio.txt"}
                    }
                ]
            }
        }


class HealthResponse(BaseModel):
    """Response model untuk endpoint /health"""
    status: str = Field(..., description="Status kesehatan aplikasi")
    ollama_status: str = Field(..., description="Status koneksi Ollama")
    vector_store_status: str = Field(..., description="Status vector store")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "ollama_status": "connected",
                "vector_store_status": "ready"
            }
        }


class ReloadResponse(BaseModel):
    """Response model untuk endpoint /reload-data"""
    status: str = Field(..., description="Status reload")
    message: str = Field(..., description="Pesan detail")
    documents_loaded: int = Field(..., description="Jumlah dokumen yang di-load")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Data portfolio berhasil di-reload",
                "documents_loaded": 10
            }
        }


class ErrorResponse(BaseModel):
    """Response model untuk error"""
    error: str = Field(..., description="Pesan error")
    detail: Optional[str] = Field(default=None, description="Detail error")
