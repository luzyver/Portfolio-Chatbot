import os
import json
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from .embeddings import get_embeddings

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
COLLECTION_NAME = "portfolio_collection"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K"))


class VectorStoreManager:
    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR):
        self.persist_directory = persist_directory
        self.embeddings = get_embeddings()
        self.vector_store: Optional[Chroma] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"VectorStoreManager initialized. Persist dir: {self.persist_directory}")

    def _format_personal_info(self, data: Dict[str, Any]) -> str:
        info = data.get("personal_info", {})
        return f"""Informasi Pribadi:
Nama: {info.get('name', 'N/A')}
Jabatan: {info.get('title', 'N/A')}
Lokasi: {info.get('location', 'N/A')}
Email: {info.get('email', 'N/A')}
LinkedIn: {info.get('linkedin', 'N/A')}
GitHub: {info.get('github', 'N/A')}
Website: {info.get('website', 'N/A')}"""

    def _format_skills(self, category: str, skills: List[Dict]) -> str:
        lines = [f"Skills - {category}:"]
        for skill in skills:
            name = skill.get('name', '')
            level = skill.get('level', '')
            years = skill.get('years', '')
            desc = skill.get('description', '')
            
            if level and years:
                lines.append(f"- {name}: {level}, {years} tahun pengalaman")
            elif desc:
                lines.append(f"- {name}: {desc}")
            else:
                lines.append(f"- {name}")
        return "\n".join(lines)

    def _format_experience(self, exp: Dict[str, Any]) -> str:
        period = exp.get('period', {})
        start = period.get('start', '')
        end = 'Sekarang' if period.get('current') else period.get('end', '')
        
        lines = [
            f"Pengalaman Kerja - {exp.get('position', '')} di {exp.get('company', '')}",
            f"Periode: {start} - {end}",
            "Tanggung Jawab:"
        ]
        for resp in exp.get('responsibilities', []):
            lines.append(f"- {resp}")
        lines.append(f"Teknologi: {', '.join(exp.get('technologies', []))}")
        return "\n".join(lines)

    def _format_project(self, project: Dict[str, Any]) -> str:
        lines = [
            f"Proyek: {project.get('name', '')} ({project.get('type', '')})",
            f"Deskripsi: {project.get('description', '')}",
            f"Teknologi: {', '.join(project.get('technologies', []))}",
            "Fitur Utama:"
        ]
        for feature in project.get('features', []):
            lines.append(f"- {feature}")
        if project.get('link'):
            lines.append(f"Link: {project.get('link')}")
        return "\n".join(lines)

    def _format_education(self, edu: Dict[str, Any]) -> str:
        period = edu.get('period', {})
        lines = [
            f"Pendidikan - {edu.get('institution', '')}",
            f"Gelar: {edu.get('degree', '')}",
            f"Jurusan: {edu.get('major', '')}",
            f"Periode: {period.get('start', '')} - {period.get('end', '')}",
            f"IPK: {edu.get('gpa', '')} / {edu.get('gpa_scale', '')}",
            "Kegiatan:"
        ]
        for activity in edu.get('activities', []):
            lines.append(f"- {activity}")
        return "\n".join(lines)

    def _format_certifications(self, certs: List[Dict]) -> str:
        lines = ["Sertifikasi:"]
        for cert in certs:
            lines.append(f"- {cert.get('name', '')} ({cert.get('year', '')}) - {cert.get('issuer', '')}")
        return "\n".join(lines)

    def _format_languages(self, languages: List[Dict]) -> str:
        lines = ["Kemampuan Bahasa:"]
        for lang in languages:
            proficiency = lang.get('proficiency', '')
            score = lang.get('score', '')
            level = lang.get('level', '')
            extra = f" - {score}" if score else (f" - {level}" if level else "")
            lines.append(f"- {lang.get('language', '')}: {proficiency}{extra}")
        return "\n".join(lines)

    def load_documents_from_json(self, file_path: str) -> List[Document]:
        logger.info(f"Loading JSON documents from: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []

        if "personal_info" in data:
            documents.append(Document(
                page_content=self._format_personal_info(data),
                metadata={"source": file_path, "type": "personal_info", "section": "personal_info"}
            ))

        if "summary" in data:
            documents.append(Document(
                page_content=f"Ringkasan Profesional:\n{data['summary']}",
                metadata={"source": file_path, "type": "summary", "section": "summary"}
            ))

        if "skills" in data:
            skills = data["skills"]
            skill_categories = {
                "programming_languages": "Programming Languages",
                "backend_frameworks": "Backend Frameworks",
                "frontend_frameworks": "Frontend Frameworks",
                "databases": "Database & Storage",
                "devops": "DevOps & Cloud",
                "ai_ml": "AI/ML Tools"
            }
            for key, label in skill_categories.items():
                if key in skills:
                    documents.append(Document(
                        page_content=self._format_skills(label, skills[key]),
                        metadata={"source": file_path, "type": "skills", "section": key}
                    ))

        if "work_experience" in data:
            for i, exp in enumerate(data["work_experience"]):
                documents.append(Document(
                    page_content=self._format_experience(exp),
                    metadata={
                        "source": file_path,
                        "type": "work_experience",
                        "section": f"experience_{i}",
                        "company": exp.get("company", ""),
                        "position": exp.get("position", "")
                    }
                ))

        if "projects" in data:
            for i, project in enumerate(data["projects"]):
                documents.append(Document(
                    page_content=self._format_project(project),
                    metadata={
                        "source": file_path,
                        "type": "project",
                        "section": f"project_{i}",
                        "project_name": project.get("name", ""),
                        "project_type": project.get("type", "")
                    }
                ))

        if "education" in data:
            for i, edu in enumerate(data["education"]):
                documents.append(Document(
                    page_content=self._format_education(edu),
                    metadata={
                        "source": file_path,
                        "type": "education",
                        "section": f"education_{i}",
                        "institution": edu.get("institution", "")
                    }
                ))

        if "certifications" in data:
            documents.append(Document(
                page_content=self._format_certifications(data["certifications"]),
                metadata={"source": file_path, "type": "certifications", "section": "certifications"}
            ))

        if "languages" in data:
            documents.append(Document(
                page_content=self._format_languages(data["languages"]),
                metadata={"source": file_path, "type": "languages", "section": "languages"}
            ))

        if "interests" in data:
            interests_text = "Hobi & Minat:\n" + "\n".join(f"- {i}" for i in data["interests"])
            documents.append(Document(
                page_content=interests_text,
                metadata={"source": file_path, "type": "interests", "section": "interests"}
            ))

        if "availability" in data:
            avail = data["availability"]
            avail_text = "Terbuka untuk:\n" + "\n".join(f"- {i}" for i in avail.get("open_for", []))
            documents.append(Document(
                page_content=avail_text,
                metadata={"source": file_path, "type": "availability", "section": "availability"}
            ))

        final_documents = []
        for doc in documents:
            if len(doc.page_content) > CHUNK_SIZE:
                chunks = self.text_splitter.split_documents([doc])
                final_documents.extend(chunks)
            else:
                final_documents.append(doc)

        logger.info(f"Loaded {len(final_documents)} dokumen dari JSON")
        return final_documents

    def load_documents_from_file(self, file_path: str) -> List[Document]:
        logger.info(f"Loading documents from: {file_path}")

        if file_path.endswith('.json'):
            return self.load_documents_from_json(file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        doc = Document(
            page_content=content,
            metadata={"source": file_path, "type": "portfolio"}
        )

        chunks = self.text_splitter.split_documents([doc])
        logger.info(f"Dokumen di-split menjadi {len(chunks)} chunks")

        return chunks

    def create_vector_store(self, documents: List[Document]) -> Chroma:
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
        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=COLLECTION_NAME
            )

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
        if self.vector_store is None:
            raise ValueError("Vector store belum di-inisialisasi. Panggil create_vector_store atau load_existing_vector_store terlebih dahulu.")

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    def similarity_search(self, query: str, k: int = RETRIEVER_K) -> List[Document]:
        if self.vector_store is None:
            raise ValueError("Vector store belum di-inisialisasi")

        return self.vector_store.similarity_search(query, k=k)

    def reload_data(self, file_path: str) -> int:
        logger.info("Reloading portfolio data...")

        if self.vector_store is not None:
            try:
                self.vector_store.delete_collection()
                logger.info("Collection lama dihapus")
            except Exception as e:
                logger.warning(f"Gagal hapus collection lama: {str(e)}")

        documents = self.load_documents_from_file(file_path)

        self.create_vector_store(documents)

        return len(documents)

    def is_ready(self) -> bool:
        if self.vector_store is None:
            return False

        try:
            count = self.vector_store._collection.count()
            return count > 0
        except Exception:
            return False
