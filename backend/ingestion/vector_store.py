"""
ChromaDB persistent vector store manager for the UpToSkills RAG ingestion pipeline.

Provides vector storage, deduplication, atomic document updates, deletion,
and query access for downstream query-processing and retrieval teams.
Persists all embeddings locally to survive application restarts.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings

from backend.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from backend.ingestion.embeddings import EmbeddingService

logger = logging.getLogger("rag.vector_store")


class VectorStoreManager:
    """
    Manages persistent vector indexing and retrieval using ChromaDB.
    """

    def __init__(
        self,
        persist_dir: Optional[Union[str, Path]] = None,
        collection_name: Optional[str] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.persist_dir = Path(persist_dir or CHROMA_PERSIST_DIR).resolve()
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._embedding_service = embedding_service
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection: Optional[chromadb.Collection] = None
        self._init_collection()

    def _get_client(self) -> chromadb.ClientAPI:
        """Initializes and returns the persistent ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
        return self._client

    def _init_collection(self) -> None:
        """Initializes or retrieves the persistent collection."""
        client = self._get_client()
        # Uses cosine distance metadata for normalized embeddings
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' initialized at '%s' (current count=%d).",
            self.collection_name,
            self.persist_dir,
            self._collection.count(),
        )

    @property
    def collection(self) -> chromadb.Collection:
        """Returns the active ChromaDB collection."""
        if self._collection is None:
            self._init_collection()
        return self._collection

    @property
    def embedding_service(self) -> EmbeddingService:
        """Lazy-loads the embedding service if not provided at initialization."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    @staticmethod
    def generate_chunk_id(document_id: str, chunk_index: int) -> str:
        """
        Creates a deterministic unique ID for a chunk.
        Format: '{document_id}_chunk_{chunk_index}'
        """
        return f"{document_id}_chunk_{chunk_index}"

    @staticmethod
    def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
        """
        Ensures all metadata values conform to ChromaDB's supported primitive types
        (str, int, float, bool). Replaces None with empty string or 0.
        """
        sanitized: Dict[str, Union[str, int, float, bool]] = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            elif isinstance(v, (list, dict)):
                # Convert complex types to JSON string representation
                sanitized[k] = json.dumps(v)
            else:
                sanitized[k] = str(v)
        return sanitized

    def add_chunks(
        self,
        document_id: Union[str, List[Dict[str, Any]]],
        chunks: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> int:
        """
        Inserts new chunk records and embeddings for a document.
        Supports both add_chunks(doc_id, chunks) and add_chunks(chunks).

        Args:
            document_id: ID of the parent document, or list of chunk dicts.
            chunks: List of processed chunk dictionaries.
            embeddings: Optional precomputed list of vectors. If None,
                        the embedding service calculates them.

        Returns:
            int: Number of chunks added.
        """
        if isinstance(document_id, list):
            target_chunks = document_id
            target_doc_id = target_chunks[0].get("document_id", "default_doc") if target_chunks else "default_doc"
        else:
            target_doc_id = document_id
            target_chunks = chunks or []

        if not target_chunks:
            return 0

        # Generate vectors if not supplied
        if embeddings is None or len(embeddings) != len(target_chunks):
            embeddings = self.embedding_service.embed_chunks(target_chunks)

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Union[str, int, float, bool]]] = []

        for index, chunk in enumerate(target_chunks):
            chunk_text = chunk.get("text", "")
            chunk_idx = chunk.get("chunk_index", index)
            chunk_doc_id = chunk.get("document_id") or target_doc_id
            chunk_id = chunk.get("chunk_id") or self.generate_chunk_id(chunk_doc_id, chunk_idx)

            raw_meta = {
                "document_id": chunk_doc_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_idx,
                "total_chunks": chunk.get("total_chunks", len(target_chunks)),
                "filename": chunk.get("filename", ""),
                "file_type": chunk.get("file_type", ""),
                "category": chunk.get("category", "documents"),
                "page_number": chunk.get("page_number") or 0,
                "estimated_tokens": chunk.get("estimated_tokens", 0),
            }
            # Preserve extra metadata from chunk dict
            for extra_k, extra_v in chunk.items():
                if extra_k not in raw_meta and extra_k not in ("text", "embedding"):
                    raw_meta[extra_k] = extra_v

            ids.append(chunk_id)
            documents.append(chunk_text)
            metadatas.append(self.sanitize_metadata(raw_meta))

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "Added/upserted %d vector chunks for document '%s' in ChromaDB.",
            len(ids),
            target_doc_id,
        )
        return len(ids)

    def delete_document(self, document_id: str) -> int:
        """
        Deletes all chunks associated with a document ID from ChromaDB.

        Args:
            document_id: ID of document whose vectors should be removed.

        Returns:
            int: Count of deleted vectors.
        """
        existing = self.collection.get(
            where={"document_id": document_id},
        )
        existing_ids = existing.get("ids", [])
        if not existing_ids:
            return 0

        self.collection.delete(ids=existing_ids)
        logger.info(
            "Deleted %d vector chunks for document '%s' from ChromaDB.",
            len(existing_ids),
            document_id,
        )
        return len(existing_ids)

    def update_document(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ) -> int:
        """
        Updates a document's vector representation atomically:
        removes any stale chunks and stores the newly processed chunks,
        preventing duplicate vectors on re-upload or re-processing.

        Returns:
            int: Number of new chunks added.
        """
        self.delete_document(document_id)
        return self.add_chunks(document_id, chunks, embeddings)

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all stored chunks and metadata for a specific document.

        Returns:
            List[Dict[str, Any]]: List of chunk items sorted by chunk_index.
        """
        results = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )

        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        chunks = []
        for c_id, text, meta in zip(ids, docs, metas):
            item = dict(meta or {})
            item["chunk_id"] = c_id
            item["text"] = text
            chunks.append(item)

        # Sort chunks by chunk_index
        chunks.sort(key=lambda c: int(c.get("chunk_index", 0)))
        return chunks

    def count_vectors(self) -> int:
        """Returns the total number of chunks/vectors across the entire collection."""
        return self.collection.count()

    def count_by_document(self, document_id: str) -> int:
        """Returns the count of vectors stored for a specific document."""
        res = self.collection.get(where={"document_id": document_id})
        return len(res.get("ids", []))

    def list_indexed_document_ids(self) -> List[str]:
        """Returns a list of distinct document IDs present in the vector store."""
        all_data = self.collection.get(include=["metadatas"])
        metas = all_data.get("metadatas", [])
        doc_ids = set()
        for m in metas:
            if m and "document_id" in m:
                doc_ids.add(str(m["document_id"]))
        return sorted(list(doc_ids))

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search against stored chunks using a query vector.
        Primary interface for the query-processing and retrieval teams.

        Args:
            query_embedding: Vector representation of the user query.
            top_k: Number of nearest chunks to retrieve.
            where: Optional ChromaDB metadata filter (e.g. {"category": "documents"}).

        Returns:
            List[Dict[str, Any]]: Ranked list of matching chunks with similarity scores.
        """
        if self.count_vectors() == 0:
            return []

        query_args: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.count_vectors()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_args["where"] = where

        res = self.collection.query(**query_args)

        hits: List[Dict[str, Any]] = []
        ids_list = res.get("ids", [[]])[0]
        docs_list = res.get("documents", [[]])[0]
        metas_list = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]

        for chunk_id, text, meta, dist in zip(ids_list, docs_list, metas_list, distances):
            # ChromaDB cosine distance: distance = 1 - cosine_similarity
            similarity = round(max(0.0, 1.0 - float(dist)), 4)
            hits.append({
                "chunk_id": chunk_id,
                "document_id": (meta or {}).get("document_id", ""),
                "text": text,
                "similarity": similarity,
                "score": similarity,
                "distance": round(float(dist), 4),
                "metadata": meta,
            })

        return hits

    def query_by_text(
        self,
        query_text: str = "",
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convenience method: encodes the query string using the embedding service
        and executes similarity search.
        """
        effective_query = query_text or query or ""
        query_vector = self.embedding_service.embed_text(effective_query)
        return self.similarity_search(query_vector, top_k=top_k, where=where)

    def reset_collection(self) -> None:
        """Completely purges the ChromaDB collection. Intended for testing."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._init_collection()
