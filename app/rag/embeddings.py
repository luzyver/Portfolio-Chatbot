"""
Embeddings module menggunakan sentence-transformers
Model: all-MiniLM-L6-v2 dari HuggingFace
"""
import logging
from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Nama model embedding yang digunakan
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Mendapatkan instance HuggingFaceEmbeddings dengan model all-MiniLM-L6-v2.
    Menggunakan lru_cache untuk singleton pattern - model hanya di-load sekali.

    Returns:
        HuggingFaceEmbeddings: Instance embeddings model
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("Embedding model berhasil di-load")
        return embeddings
    except Exception as e:
        logger.error(f"Gagal load embedding model: {str(e)}")
        raise


def test_embeddings():
    """
    Test function untuk memastikan embeddings berfungsi dengan baik.
    """
    embeddings = get_embeddings()
    test_text = "Ini adalah test embedding"
    result = embeddings.embed_query(test_text)
    logger.info(f"Test embedding berhasil. Dimensi: {len(result)}")
    return len(result)
