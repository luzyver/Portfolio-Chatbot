"""
LLM module untuk integrasi dengan Groq API
Menggunakan model Groq untuk generate response
"""
import os
import logging
from typing import List, Tuple, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document

from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# Konfigurasi dari environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "whisper-large-v3-turbo")

# Custom prompt template dalam Bahasa Indonesia
PROMPT_TEMPLATE = """Kamu adalah asisten AI untuk portfolio chatbot. Tugasmu adalah menjawab pertanyaan berdasarkan informasi portfolio yang diberikan.

ATURAN PENTING:
1. Jawab HANYA berdasarkan informasi dari konteks portfolio yang diberikan di bawah
2. Jika informasi yang ditanyakan TIDAK ADA dalam konteks, katakan dengan jujur: "Maaf, informasi tersebut tidak tersedia dalam portfolio"
3. JANGAN mengarang atau membuat informasi yang tidak ada dalam konteks
4. Jawab dalam Bahasa Indonesia dengan sopan dan profesional
5. Berikan jawaban yang ringkas namun informatif
6. Gunakan bahasa yang natural dan bervariasi; hindari frasa yang monoton
7. JANGAN menyalin ulang konteks atau pertanyaan; berikan jawaban final saja

Konteks Portfolio:
{context}

Pertanyaan: {question}

Jawaban:"""


class LLMManager:
    """
    Manager untuk mengelola LLM (Groq) dan RAG chain.
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        model: str = GROQ_MODEL,
        api_key: Optional[str] = GROQ_API_KEY
    ):
        """
        Inisialisasi LLMManager.

        Args:
            vector_store_manager: Instance VectorStoreManager untuk retrieval
            model: Nama model yang digunakan (default: whisper-large-v3-turbo)
            api_key: Groq API key
        """
        self.vector_store_manager = vector_store_manager
        self.model = model
        self.api_key = api_key
        self.llm: Optional[ChatGroq] = None
        self.qa_chain = None

        logger.info(f"LLMManager initialized. Groq Model: {model}")

    def initialize_llm(self) -> ChatGroq:
        """
        Inisialisasi koneksi ke Groq LLM.

        Returns:
            ChatGroq: Instance LLM
        """
        if not self.api_key:
            raise ValueError("GROQ_API_KEY belum di-set")

        logger.info(f"Initializing Groq LLM: {self.model}")

        self.llm = ChatGroq(
            api_key=self.api_key,
            model=self.model,
            temperature=0.3,
            max_tokens=256,
        )

        logger.info("Groq LLM berhasil di-inisialisasi")
        return self.llm

    def create_qa_chain(self) -> RetrievalQA:
        """
        Membuat QA chain dengan retriever dan custom prompt.

        Returns:
            RetrievalQA: Chain untuk question answering
        """
        if self.llm is None:
            self.initialize_llm()

        if not self.vector_store_manager.is_ready():
            raise ValueError("Vector store belum siap. Load data terlebih dahulu.")

        # Buat prompt template
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        # Buat retriever
        retriever = self.vector_store_manager.get_retriever()

        # Buat QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        logger.info("QA Chain berhasil dibuat")
        return self.qa_chain

    def chat(self, question: str) -> Tuple[str, List[Document]]:
        """
        Melakukan chat/tanya jawab dengan RAG.

        Args:
            question: Pertanyaan dari user

        Returns:
            Tuple[str, List[Document]]: (response, source_documents)
        """
        if self.qa_chain is None:
            self.create_qa_chain()

        cleaned_question = question.strip()
        logger.info(f"Processing question: {cleaned_question[:50]}...")

        # Fast-path untuk sapaan singkat agar tidak dianggap "kurang informasi".
        lower_question = cleaned_question.lower()
        greetings = {
            "halo", "hai", "hi", "hello", "pagi", "siang", "sore", "malam"
        }
        if lower_question in greetings or len(lower_question.split()) <= 1:
            return (
                "Halo! Kamu bisa tanya hal spesifik tentang portfolio, misalnya pengalaman kerja, proyek, atau kontak.",
                []
            )

        try:
            # Heuristik sederhana untuk pertanyaan kontak agar tidak gagal di model kecil.
            contact_keywords = {
                "kontak", "contact", "email", "linkedin", "github", "website", "web",
                "nama", "lokasi", "alamat"
            }
            if any(keyword in lower_question for keyword in contact_keywords):
                retriever = self.vector_store_manager.get_retriever()
                if hasattr(retriever, "invoke"):
                    docs = retriever.invoke(cleaned_question)
                else:
                    docs = retriever.get_relevant_documents(cleaned_question)
                contact_lines = []
                wanted_prefixes = (
                    "Nama:", "Jabatan:", "Lokasi:", "Email:",
                    "LinkedIn:", "GitHub:", "Website:"
                )
                for doc in docs:
                    for line in doc.page_content.splitlines():
                        line = line.strip()
                        if line.startswith(wanted_prefixes):
                            contact_lines.append(line)
                if contact_lines:
                    response = "\n".join(contact_lines)
                    logger.info("Contact info answered via heuristic extraction.")
                    return response, docs

            result = self.qa_chain.invoke({"query": cleaned_question})

            response = result.get("result", "Maaf, terjadi kesalahan dalam memproses pertanyaan.")
            source_docs = result.get("source_documents", [])

            # Variasi respons agar tidak monoton saat info tidak tersedia.
            not_available_phrases = [
                "Maaf, informasi itu belum tersedia di portfolio.",
                "Belum ada informasi tersebut di portfolio yang saya miliki.",
                "Sepertinya informasi itu tidak tercantum di portfolio saat ini.",
            ]
            default_not_available = "Maaf, informasi tersebut tidak tersedia dalam portfolio"
            if default_not_available.lower() in response.lower():
                index_seed = sum(ord(ch) for ch in cleaned_question) % len(not_available_phrases)
                response = not_available_phrases[index_seed]

            logger.info(f"Response generated. Sources: {len(source_docs)}")

            return response, source_docs

        except Exception as e:
            logger.error(f"Error during chat: {str(e)}")
            raise

    def test_connection(self) -> bool:
        """
        Test koneksi ke Groq service.

        Returns:
            bool: True jika koneksi berhasil
        """
        try:
            if self.llm is None:
                self.initialize_llm()

            # Simple test invoke
            test_response = self.llm.invoke("Halo, apa kabar?")
            preview = test_response.content if hasattr(test_response, "content") else str(test_response)
            logger.info(f"Groq connection test successful: {preview[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Groq connection test failed: {str(e)}")
            return False

    def refresh_chain(self):
        """
        Refresh QA chain setelah data di-reload.
        """
        self.qa_chain = None
        self.create_qa_chain()
        logger.info("QA Chain refreshed")
