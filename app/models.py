from typing import List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' atau 'assistant'")
    content: str = Field(..., description="Isi pesan")


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        description="Pertanyaan atau pesan dari user",
        min_length=1,
        max_length=1000
    )
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Riwayat percakapan sebelumnya"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Apa saja skill programming yang kamu miliki?",
                "history": [
                    {"role": "user", "content": "Halo"},
                    {"role": "assistant", "content": "Halo! Ada yang bisa saya bantu?"}
                ]
            }
        }


class ChatResponse(BaseModel):
    response: str = Field(..., description="Jawaban dari chatbot")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Saya memiliki skill dalam Python, JavaScript, dan FastAPI."
            }
        }


class HealthResponse(BaseModel):
    status: str = Field(..., description="Status kesehatan aplikasi")
    groq_status: str = Field(..., description="Status koneksi Groq")
    vector_store_status: str = Field(..., description="Status vector store")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "groq_status": "connected",
                "vector_store_status": "ready"
            }
        }


class ReloadResponse(BaseModel):
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
    error: str = Field(..., description="Pesan error")
    detail: Optional[str] = Field(default=None, description="Detail error")
