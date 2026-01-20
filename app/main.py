"""
Portfolio Chatbot - FastAPI Application
RAG-based chatbot menggunakan Groq API dan ChromaDB
"""
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    ChatRequest,
    ChatResponse,
    SourceDocument,
    HealthResponse,
    ReloadResponse,
    ErrorResponse
)
from rag import VectorStoreManager, LLMManager

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Konfigurasi paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PORTFOLIO_FILE = DATA_DIR / "portfolio.txt"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))

# Global managers
vector_store_manager: VectorStoreManager = None
llm_manager: LLMManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager untuk startup dan shutdown.
    """
    global vector_store_manager, llm_manager

    logger.info("Starting Portfolio Chatbot...")

    # Inisialisasi Vector Store Manager
    vector_store_manager = VectorStoreManager(persist_directory=CHROMA_PERSIST_DIR)

    # Coba load vector store yang sudah ada, atau buat baru
    existing_store = vector_store_manager.load_existing_vector_store()

    if existing_store is None:
        logger.info("No existing vector store found. Loading portfolio data...")
        if PORTFOLIO_FILE.exists():
            try:
                vector_store_manager.reload_data(str(PORTFOLIO_FILE))
                logger.info("Portfolio data loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load portfolio data: {str(e)}")
        else:
            logger.warning(f"Portfolio file not found: {PORTFOLIO_FILE}")
    else:
        logger.info("Using existing vector store")

    # Inisialisasi LLM Manager
    llm_manager = LLMManager(vector_store_manager)

    try:
        llm_manager.initialize_llm()
        if vector_store_manager.is_ready():
            llm_manager.create_qa_chain()
        logger.info("LLM Manager initialized successfully")
    except Exception as e:
        logger.warning(f"LLM initialization warning: {str(e)}. Chat may not work until Groq is available.")

    logger.info("Portfolio Chatbot started successfully!")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Portfolio Chatbot...")


# Inisialisasi FastAPI app
app = FastAPI(
    title="Portfolio Chatbot API",
    description="RAG-based chatbot untuk menjawab pertanyaan tentang portfolio menggunakan Groq API dan ChromaDB",
    version="1.0.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)

# CORS middleware untuk frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dalam production, ganti dengan domain spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat dengan Portfolio Bot",
    description="Kirim pertanyaan dan dapatkan jawaban berdasarkan informasi portfolio"
)
async def chat(request: ChatRequest):
    """
    Endpoint untuk chat dengan portfolio bot.

    - **message**: Pertanyaan atau pesan dari user

    Returns response beserta source documents yang digunakan.
    """
    global llm_manager, vector_store_manager

    if llm_manager is None:
        raise HTTPException(
            status_code=503,
            detail="LLM Manager belum di-inisialisasi"
        )

    if not vector_store_manager.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Vector store belum siap. Silakan reload data terlebih dahulu."
        )

    try:
        logger.info(f"Received chat request: {request.message[:50]}...")

        # Proses pertanyaan
        response, source_docs = llm_manager.chat(request.message)

        # Format source documents
        sources = [
            SourceDocument(
                content=doc.page_content[:500],  # Limit content length
                metadata=doc.metadata
            )
            for doc in source_docs
        ]

        return ChatResponse(response=response, sources=sources)

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error memproses pertanyaan: {str(e)}"
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Cek status kesehatan aplikasi dan dependensi"
)
async def health_check():
    """
    Endpoint untuk health check (lightweight).

    Mengecek status:
    - Aplikasi
    - Vector Store
    - Groq (hanya cek apakah manager sudah di-inisialisasi, tanpa test connection)
    """
    global llm_manager, vector_store_manager

    # Check Groq status (lightweight - tanpa test connection yang blocking)
    groq_status = "initialized" if llm_manager is not None and llm_manager.llm is not None else "not_initialized"

    # Check vector store status
    vector_store_status = "not_initialized"
    if vector_store_manager is not None:
        if vector_store_manager.is_ready():
            vector_store_status = "ready"
        else:
            vector_store_status = "empty"

    # Determine overall status
    if groq_status == "initialized" and vector_store_status == "ready":
        status = "healthy"
    elif groq_status == "not_initialized" or vector_store_status in ["not_initialized", "empty"]:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        groq_status=groq_status,
        vector_store_status=vector_store_status
    )


@app.post(
    "/reload-data",
    response_model=ReloadResponse,
    summary="Reload Portfolio Data",
    description="Reload data portfolio dari file dan rebuild vector store"
)
async def reload_data():
    """
    Endpoint untuk reload data portfolio.

    Akan membaca ulang file portfolio.txt dan membangun ulang vector store.
    Berguna setelah update data portfolio.
    """
    global vector_store_manager, llm_manager

    if vector_store_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Vector Store Manager belum di-inisialisasi"
        )

    if not PORTFOLIO_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File portfolio tidak ditemukan: {PORTFOLIO_FILE}"
        )

    try:
        logger.info("Reloading portfolio data...")

        # Reload data
        docs_count = vector_store_manager.reload_data(str(PORTFOLIO_FILE))

        # Refresh LLM chain
        if llm_manager is not None:
            llm_manager.refresh_chain()

        logger.info(f"Portfolio data reloaded successfully. Documents: {docs_count}")

        return ReloadResponse(
            status="success",
            message="Data portfolio berhasil di-reload",
            documents_loaded=docs_count
        )

    except Exception as e:
        logger.error(f"Reload error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reload data: {str(e)}"
        )


@app.get("/", summary="Root", description="Welcome endpoint")
async def root():
    """Welcome endpoint."""
    return {
        "message": "Selamat datang di Portfolio Chatbot API!",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=9999,
        reload=True
    )
