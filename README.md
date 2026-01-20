# Portfolio Chatbot

RAG-based chatbot untuk menjawab pertanyaan tentang portfolio menggunakan Ollama (Mistral) dan ChromaDB.

## Tech Stack

- **Backend**: Python dengan FastAPI
- **LLM**: Ollama (model Mistral) - self-hosted
- **RAG**: LangChain + ChromaDB
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Deployment**: Docker Compose

## Project Structure

```
├── app/
│   ├── main.py           # FastAPI application
│   ├── models.py         # Pydantic models
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py # HuggingFace embeddings
│   │   ├── vector_store.py # ChromaDB vector store
│   │   └── llm.py        # Ollama LLM integration
│   └── data/
│       └── portfolio.txt # Portfolio data
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Deployment Instructions

### Prerequisites

- Docker dan Docker Compose terinstall
- Minimal 8GB RAM (untuk Ollama + Mistral model)
- 10GB+ disk space

### Step 1: Clone dan Setup

```bash
# Clone repository (atau copy files)
cd portfolio-chatbot

# Edit portfolio.txt dengan data Anda
nano app/data/portfolio.txt
```

### Step 2: Build Docker Images

```bash
# Build images
docker-compose build
```

### Step 3: Start Services

```bash
# Start semua services
docker-compose up -d
```

### Step 4: Pull Mistral Model

Setelah services running, pull model Mistral ke Ollama:

```bash
# Masuk ke container Ollama
docker exec -it portfolio-ollama ollama pull mistral

# Atau langsung dari host
docker exec portfolio-ollama ollama pull mistral
```

Tunggu hingga download selesai (sekitar 4GB).

### Step 5: Verify Installation

```bash
# Check health status
curl http://localhost:9999/health

# Test chat
curl -X POST http://localhost:9999/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Apa saja skill programming yang kamu miliki?"}'
```

## API Endpoints

### POST /chat
Chat dengan portfolio bot.

**Request:**
```json
{
  "message": "Apa pengalaman kerja kamu?"
}
```

**Response:**
```json
{
  "response": "Saya memiliki pengalaman sebagai Senior Full Stack Developer di PT Tech Inovasi Indonesia sejak Januari 2022...",
  "sources": [
    {
      "content": "=== PENGALAMAN KERJA ===...",
      "metadata": {"source": "portfolio.txt"}
    }
  ]
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "ollama_status": "connected",
  "vector_store_status": "ready"
}
```

### POST /reload-data
Reload portfolio data.

**Response:**
```json
{
  "status": "success",
  "message": "Data portfolio berhasil di-reload",
  "documents_loaded": 15
}
```

## Swagger Documentation

Akses dokumentasi API di: `http://localhost:9999/docs`

## Commands Cheat Sheet

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# View chatbot logs only
docker-compose logs -f chatbot

# Restart chatbot setelah update code
docker-compose restart chatbot

# Rebuild setelah update requirements
docker-compose build --no-cache chatbot
docker-compose up -d

# Pull model Mistral
docker exec portfolio-ollama ollama pull mistral

# List models di Ollama
docker exec portfolio-ollama ollama list

# Hapus semua data dan mulai fresh
docker-compose down -v
docker-compose up -d
```

## Update Portfolio Data

1. Edit file `app/data/portfolio.txt`
2. Call endpoint reload:
   ```bash
   curl -X POST http://localhost:9999/reload-data
   ```

## Troubleshooting

### Ollama tidak connected
```bash
# Check Ollama health
curl http://localhost:11434/api/tags

# Restart Ollama
docker-compose restart ollama
```

### Vector store empty
```bash
# Reload data
curl -X POST http://localhost:9999/reload-data
```

### Out of memory
Edit `docker-compose.yml` dan sesuaikan memory limits.

## Offline Usage

Setelah setup selesai, chatbot dapat berjalan fully offline:
- Embedding model sudah di-download saat build Docker image
- Mistral model sudah tersimpan di volume Ollama
- Tidak ada API call ke external services

## License

MIT License
