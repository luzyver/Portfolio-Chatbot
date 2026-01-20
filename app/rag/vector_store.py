"""
Vector Store module menggunakan ChromaDB
Mengelola penyimpanan dan retrieval dokumen portfolio
"""
import os
import logging
from typing import List, Optional
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from .embeddings import get_embeddings

logger = logging.getLogger(__name__)

# Konfigurasi
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "portfolio_collection"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVER_K = 3


class VectorStoreManager:
    """
    Manager untuk mengelola ChromaDB vector store.
    Menangani loading dokumen, splitting, dan retrieval.
    """

    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR):
        """
        Inisialisasi VectorStoreManager.

        Args:
            persist_directory: Direktori untuk menyimpan ChromaDB
        """
        self.persist_directory = persist_directory
        self.embeddings = get_embeddings()
        self.vector_store: Optional[Chroma] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        # Pastikan direktori persist ada
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"VectorStoreManager initialized. Persist dir: {self.persist_directory}")

    def load_documents_from_file(self, file_path: str) -> List[Document]:
        """
        Load dan split dokumen dari file teks.

        Args:
            file_path: Path ke file portfolio.txt

        Returns:
            List[Document]: Daftar dokumen yang sudah di-split
        """
        logger.info(f"Loading documents from: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Buat dokumen dengan metadata
        doc = Document(
            page_content=content,
            metadata={"source": file_path, "type": "portfolio"}
        )

        # Split dokumen
        chunks = self.text_splitter.split_documents([doc])
        logger.info(f"Dokumen di-split menjadi {len(chunks)} chunks")

        return chunks

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """
        Membuat vector store baru dari dokumen.

        Args:
            documents: List dokumen yang akan di-embed

        Returns:
            Chroma: Instance ChromaDB vector store
        """
        logger.info(f"Creating vector store dengan {len(documents)} dokumen")

        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=COLLECTION_NAME
        )

        logger.info("Vector store berhasil dibuat")
        return self.vector_store

    def load_existing_vector_store(self) -> Optional[Chroma]:
        """
        Load vector store yang sudah ada dari disk.

        Returns:
            Chroma atau None jika tidak ada
        """
        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=COLLECTION_NAME
            )

            # Cek apakah ada dokumen
            count = self.vector_store._collection.count()
            if count > 0:
                logger.info(f"Loaded existing vector store dengan {count} dokumen")
                return self.vector_store
            else:
                logger.info("Vector store kosong")
                return None
        except Exception as e:
            logger.warning(f"Tidak dapat load vector store yang ada: {str(e)}")
            return None

    def get_retriever(self, k: int = RETRIEVER_K):
        """
        Mendapatkan retriever untuk mencari dokumen relevan.

        Args:
            k: Jumlah dokumen terdekat yang diambil

        Returns:
            Retriever instance
        """
        if self.vector_store is None:
            raise ValueError("Vector store belum di-inisialisasi. Panggil create_vector_store atau load_existing_vector_store terlebih dahulu.")

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    def similarity_search(self, query: str, k: int = RETRIEVER_K) -> List[Document]:
        """
        Melakukan pencarian similarity untuk query.

        Args:
            query: Query pencarian
            k: Jumlah hasil yang diinginkan

        Returns:
            List[Document]: Dokumen yang relevan
        """
        if self.vector_store is None:
            raise ValueError("Vector store belum di-inisialisasi")

        return self.vector_store.similarity_search(query, k=k)

    def reload_data(self, file_path: str) -> int:
        """
        Reload data portfolio - hapus data lama dan load yang baru.

        Args:
            file_path: Path ke file portfolio.txt

        Returns:
            int: Jumlah dokumen yang di-load
        """
        logger.info("Reloading portfolio data...")

        # Hapus collection lama jika ada
        if self.vector_store is not None:
            try:
                self.vector_store.delete_collection()
                logger.info("Collection lama dihapus")
            except Exception as e:
                logger.warning(f"Gagal hapus collection lama: {str(e)}")

        # Load dokumen baru
        documents = self.load_documents_from_file(file_path)

        # Buat vector store baru
        self.create_vector_store(documents)

        return len(documents)

    def is_ready(self) -> bool:
        """
        Cek apakah vector store siap digunakan.

        Returns:
            bool: True jika siap
        """
        if self.vector_store is None:
            return False

        try:
            count = self.vector_store._collection.count()
            return count > 0
        except Exception:
            return False
