# Portfolio Chatbot

RAG-based chatbot untuk menjawab pertanyaan tentang portfolio menggunakan Groq API dan ChromaDB.

## Tech Stack

- **Backend**: Python dengan FastAPI
- **LLM**: Groq API (Llama 4 / Llama 3.3 dengan auto-fallback)
- **RAG**: LangChain + ChromaDB
- **Embeddings**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **Deployment**: Docker Compose

## Project Structure

```
├── app/
│   ├── main.py
│   ├── models.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── llm.py
│   └── data/
│       └── portfolio.txt
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Deployment

### Prerequisites

- Docker dan Docker Compose
- Groq API Key (gratis di https://console.groq.com)

### Setup

```bash
# Clone repository
cd portfolio-chatbot

# Copy dan edit environment
cp .env.example .env
nano .env  # Masukkan GROQ_API_KEY

# Edit portfolio data
nano app/data/portfolio.txt

# Build dan start
docker compose build
docker compose up -d
```

### Verify

```bash
# Health check
curl http://localhost:9999/health

# Test chat
curl -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Apa saja skill programming yang kamu miliki?"}'
```

## API Endpoints

### POST /chat

```json
// Request
{"message": "Apa pengalaman kerja kamu?"}

// Response
{
  "response": "Saya memiliki pengalaman sebagai...",
  "sources": [{"content": "...", "metadata": {"source": "portfolio.txt"}}]
}
```

### GET /health

```json
{
  "status": "healthy",
  "groq_status": "initialized",
  "vector_store_status": "ready"
}
```

### POST /reload-data

```json
{
  "status": "success",
  "message": "Data portfolio berhasil di-reload",
  "documents_loaded": 15
}
```

## Commands

```bash
docker compose up -d          # Start
docker compose down           # Stop
docker compose logs -f        # View logs
docker compose restart        # Restart
docker compose build --no-cache && docker compose up -d  # Rebuild
```

## Update Portfolio Data

1. Edit `app/data/portfolio.txt`
2. Reload: `curl -X POST http://localhost:9999/reload-data`

## API Docs

Swagger UI: http://localhost:9999/docs
