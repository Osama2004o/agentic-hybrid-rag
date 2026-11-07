import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.loaders import document_loader_branch


class ChromaLangChainRetriever(BaseRetriever):
    """LangChain BaseRetriever wrapper over ChromaDB for LCEL pipeline integration."""

    pipeline: Any
    k: int = 4
    where_filter: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        return self.pipeline.retrieve_as_documents(
            query=query, k=self.k, where=self.where_filter
        )


class KnowledgeBasePipeline:
    """Manages external knowledge ingestion, embedding, persistent storage, and retrieval in ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "./database/chroma_db",
        collection_name: str = "knowledge_base",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        self.persist_directory = os.path.abspath(persist_directory)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize SentenceTransformer embedding function (using local cache)
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )

        # Initialize Persistent ChromaDB Client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        except ValueError as e:
            if "Embedding function conflict" in str(e):
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_fn,
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                raise e

        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures all metadata values are primitive types supported by ChromaDB."""
        sanitized = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            elif v is None:
                sanitized[k] = ""
            elif isinstance(v, (list, dict)):
                sanitized[k] = json.dumps(v)
            else:
                sanitized[k] = str(v)
        return sanitized

    def _generate_chunk_id(self, source: str, chunk_index: int, content: str) -> str:
        """Generates a deterministic unique chunk ID based on source and content snippet."""
        unique_key = f"{source}_{chunk_index}_{content[:60]}"
        return hashlib.sha256(unique_key.encode("utf-8")).hexdigest()

    def _detect_source_type(self, source: str) -> str:
        """Determines the semantic source type from the path or URI."""
        source_lower = source.lower()
        if source_lower.startswith("http://") or source_lower.startswith("https://"):
            if "wikipedia.org" in source_lower:
                return "wikipedia"
            return "web"
        if source_lower.startswith("wiki:"):
            return "wikipedia"

        _, ext = os.path.splitext(source_lower)
        type_mapping = {
            ".pdf": "pdf",
            ".docx": "docx",
            ".doc": "docx",
            ".txt": "txt",
            ".ppt": "presentation",
            ".pptx": "presentation",
            ".wav": "audio_transcript",
            ".py": "python_code",
            ".js": "javascript_code",
            ".ts": "typescript_code",
            ".java": "java_code",
            ".cpp": "cpp_code",
            ".c": "c_code",
        }
        return type_mapping.get(ext, ext.removeprefix(".") or "text")

    def ingest(self, input_source: str) -> Dict[str, Any]:
        """Loads, chunks, embeds, and stores an external knowledge source into ChromaDB."""
        # 1. Load raw documents using the unified document loader branch
        raw_docs = document_loader_branch.invoke(input_source)
        if not raw_docs:
            return {
                "status": "warning",
                "source": input_source,
                "chunks_ingested": 0,
                "message": "No documents extracted from source.",
            }

        # 2. Chunk documents
        chunked_docs = self.text_splitter.split_documents(raw_docs)
        if not chunked_docs:
            return {
                "status": "warning",
                "source": input_source,
                "chunks_ingested": 0,
                "message": "Source yielded no content chunks.",
            }

        source_type = self._detect_source_type(input_source)

        # 3. Prepare Chroma batch payloads
        ids = []
        documents = []
        metadatas = []

        for idx, chunk in enumerate(chunked_docs):
            chunk_content = chunk.page_content.strip()
            if not chunk_content:
                continue

            chunk_id = self._generate_chunk_id(input_source, idx, chunk_content)
            meta = dict(chunk.metadata)
            meta["source"] = input_source
            meta["source_type"] = source_type
            meta["chunk_index"] = idx

            sanitized_meta = self._sanitize_metadata(meta)

            ids.append(chunk_id)
            documents.append(chunk_content)
            metadatas.append(sanitized_meta)

        if not ids:
            return {
                "status": "warning",
                "source": input_source,
                "chunks_ingested": 0,
                "message": "All chunks were empty.",
            }

        # 4. Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        return {
            "status": "success",
            "source": input_source,
            "source_type": source_type,
            "chunks_ingested": len(ids),
        }

    def ingest_batch(self, sources: List[str]) -> List[Dict[str, Any]]:
        """Batch ingestion helper for multiple knowledge sources."""
        results = []
        for src in sources:
            try:
                res = self.ingest(src)
                results.append(res)
            except Exception as e:
                results.append(
                    {
                        "status": "error",
                        "source": src,
                        "chunks_ingested": 0,
                        "error": str(e),
                    }
                )
        return results

    def retrieve(
        self, query: str, k: int = 4, where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Queries the vector store and returns matching chunks with metadata and scores."""
        query = query.strip()
        if not query:
            return []

        query_kwargs = {
            "query_texts": [query],
            "n_results": k,
        }
        if where:
            query_kwargs["where"] = where

        res = self.collection.query(**query_kwargs)

        formatted_results = []
        if res and res.get("documents") and res["documents"][0]:
            docs = res["documents"][0]
            metas = res["metadatas"][0] if res.get("metadatas") else [{}] * len(docs)
            distances = res["distances"][0] if res.get("distances") else [0.0] * len(docs)
            ids = res["ids"][0] if res.get("ids") else [""] * len(docs)

            for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
                # For cosine distance, similarity = 1 - distance
                similarity = 1.0 - dist if dist is not None else 1.0
                formatted_results.append(
                    {
                        "id": doc_id,
                        "content": doc,
                        "metadata": meta,
                        "distance": dist,
                        "similarity_score": round(similarity, 4),
                    }
                )

        return formatted_results

    def retrieve_as_documents(
        self, query: str, k: int = 4, where: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieves results converted into standard LangChain Document objects."""
        raw_results = self.retrieve(query=query, k=k, where=where)
        docs = []
        for r in raw_results:
            meta = dict(r["metadata"])
            meta["score"] = r["similarity_score"]
            meta["chunk_id"] = r["id"]
            docs.append(Document(page_content=r["content"], metadata=meta))
        return docs

    def as_retriever(
        self, k: int = 4, where: Optional[Dict[str, Any]] = None
    ) -> ChromaLangChainRetriever:
        """Returns a LangChain-compatible retriever for LCEL chain integration."""
        return ChromaLangChainRetriever(pipeline=self, k=k, where_filter=where)

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about the current persistent knowledge base."""
        total_chunks = self.collection.count()
        sources = self.list_sources()
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "total_chunks": total_chunks,
            "unique_sources_count": len(sources),
            "sources": sources,
        }

    def list_sources(self) -> List[str]:
        """Lists all distinct source identifiers stored in the knowledge base."""
        data = self.collection.get(include=["metadatas"])
        metas = data.get("metadatas", [])
        sources = set()
        for m in metas:
            if m and "source" in m:
                sources.add(m["source"])
        return sorted(list(sources))

    def delete_source(self, source: str) -> Dict[str, Any]:
        """Deletes all chunks associated with a specific knowledge source."""
        self.collection.delete(where={"source": source})
        return {
            "status": "success",
            "deleted_source": source,
            "remaining_chunks": self.collection.count(),
        }

    def clear(self) -> Dict[str, Any]:
        """Empties the knowledge base collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        return {"status": "success", "message": "Knowledge base cleared."}
