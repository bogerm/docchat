from __future__ import annotations

from typing import Sequence, Optional, Any
from loguru import logger
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

from config.settings import settings
from config.models import get_embeddings

try:
    from langchain.retrievers import EnsembleRetriever  # type: ignore
except Exception:
    from langchain_classic.retrievers import EnsembleRetriever  # type: ignore


class RetrieverBuilder:
    def __init__(self):
        emb = get_embeddings()
        self.embeddings = emb.get_embeddings()
        logger.info(f"Initialized embeddings: {emb.get_model_name()}")

    def build_hybrid_retriever(
        self,
        docs: Sequence[Any],
        *,
        k_bm25: Optional[int] = None,
        k_vector: Optional[int] = None,
        collection_suffix: Optional[str] = None,
        persist: bool = False,
    ):
        if not docs:
            raise ValueError("Cannot build retriever: docs is empty")

        k_bm25 = k_bm25 or settings.VECTOR_SEARCH_K
        k_vector = k_vector or settings.VECTOR_SEARCH_K

        # If you ever set persist=True, suffix becomes mandatory to avoid mixing.
        if persist and not collection_suffix:
            raise ValueError("persist=True requires collection_suffix to avoid mixing collections")

        # Collection name: base + optional suffix
        collection_name = settings.CHROMA_COLLECTION_NAME
        if collection_suffix:
            collection_name = f"{collection_name}-{collection_suffix}"

        persist_dir = settings.CHROMA_DB_PATH if persist else None

        logger.info(
            f"Building hybrid retriever: docs={len(docs)}, "
            f"collection={collection_name}, persist={persist}, persist_dir={persist_dir}"
        )

        vector_store = Chroma.from_documents(
            documents=list(docs),
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )

        bm25 = BM25Retriever.from_documents(list(docs))
        bm25.k = k_bm25

        vector_retriever = vector_store.as_retriever(search_kwargs={"k": k_vector})

        return EnsembleRetriever(
            retrievers=[bm25, vector_retriever],
            weights=list(settings.HYBRID_RETRIEVER_WEIGHTS),
        )
