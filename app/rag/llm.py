import os
import logging
from typing import List, Tuple, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document

from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

FALLBACK_MODELS = [
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "moonshotai/kimi-k2-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "groq/compound",
    "groq/compound-mini",
    "llama-3.1-8b-instant",
    "allam-2-7b",
]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", FALLBACK_MODELS[0])

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
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        model: str = GROQ_MODEL,
        api_key: Optional[str] = GROQ_API_KEY
    ):
        self.vector_store_manager = vector_store_manager
        self.model = model
        self.api_key = api_key
        self.llm: Optional[ChatGroq] = None
        self.qa_chain = None

        logger.info(f"LLMManager initialized. Groq Model: {model}")

    def initialize_llm(self) -> ChatGroq:
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
        if self.llm is None:
            self.initialize_llm()

        if not self.vector_store_manager.is_ready():
            raise ValueError("Vector store belum siap. Load data terlebih dahulu.")

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        retriever = self.vector_store_manager.get_retriever()

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        logger.info("QA Chain berhasil dibuat")
        return self.qa_chain

    def switch_model(self, new_model: str) -> None:
        logger.info(f"Switching model from {self.model} to {new_model}")
        self.model = new_model
        self.llm = ChatGroq(
            api_key=self.api_key,
            model=self.model,
            temperature=0.3,
            max_tokens=256,
        )
        self.qa_chain = None
        self.create_qa_chain()
        logger.info(f"Successfully switched to model: {new_model}")

    def chat(self, question: str) -> Tuple[str, List[Document]]:
        if self.qa_chain is None:
            self.create_qa_chain()

        cleaned_question = question.strip()
        logger.info(f"Processing question: {cleaned_question[:50]}...")

        models_to_try = [self.model] + [m for m in FALLBACK_MODELS if m != self.model]
        last_error = None

        for model in models_to_try:
            try:
                if model != self.model:
                    self.switch_model(model)

                result = self.qa_chain.invoke({"query": cleaned_question})

                response = result.get("result", "Maaf, terjadi kesalahan dalam memproses pertanyaan.")
                source_docs = result.get("source_documents", [])

                logger.info(f"Response generated using model: {self.model}. Sources: {len(source_docs)}")
                return response, source_docs

            except Exception as e:
                error_str = str(e)
                last_error = e

                if "429" in error_str or "rate_limit" in error_str.lower() or "Resource has been exhausted" in error_str:
                    logger.warning(f"Rate limit hit for model {model}. Trying next model...")
                    continue
                elif "does not support" in error_str or "not found" in error_str.lower():
                    logger.warning(f"Model {model} not available. Trying next model...")
                    continue
                else:
                    logger.error(f"Error during chat: {error_str}")
                    raise

        logger.error(f"All models failed. Last error: {str(last_error)}")
        raise last_error

    def test_connection(self) -> bool:
        try:
            if self.llm is None:
                self.initialize_llm()

            test_response = self.llm.invoke("Halo, apa kabar?")
            preview = test_response.content if hasattr(test_response, "content") else str(test_response)
            logger.info(f"Groq connection test successful: {preview[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Groq connection test failed: {str(e)}")
            return False

    def refresh_chain(self):
        self.qa_chain = None
        self.create_qa_chain()
        logger.info("QA Chain refreshed")
