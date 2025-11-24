
"""Centralized configuration management for workflow parametrization.

This module avoids importing heavy optional dependencies (like
sentence_transformers) at import time so other parts of the application
can import `Config` even when those packages are not installed. Use
`Config.get_embedding_model()` to obtain a SentenceTransformer instance
when needed; this will raise an informative error if the package is missing.
"""

from typing import Optional

from langchain_ollama import ChatOllama


class Config:
    """Centralized configuration management for workflow parametrization"""

    # =======================
    #   Model settings
    # =======================
    PLANNER_MODEL = "granite3.3:8b"
    RESEARCHER_MODEL = "llama3.1:8b"  # llama seems to work better in tool calling
    CURATOR_MODEL = "granite3.3:8b"
    ANALYZER_MODEL = "granite3.3:8b"

    RESEARCHER_MODEL_TEMPERATURE = 0.7

    # name of the embedding model to use (do not import at module level)
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

    # internal cache for a lazy-loaded SentenceTransformer instance
    _embedding_model: Optional[object] = None

    @classmethod
    def get_embedding_model(cls):
        """Return a SentenceTransformer instance, loading it lazily.

        Raises:
            ImportError: if `sentence_transformers` is not installed.
        """
        if cls._embedding_model is not None:
            return cls._embedding_model

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:  # ImportError or other failures
            raise ImportError(
                "`sentence_transformers` is required for embeddings. "
                "Install it (e.g. pip install -r requirements.txt) or set up "
                "an alternative embedding provider. Original error: %s" % e
            )

        cls._embedding_model = SentenceTransformer(cls.EMBEDDING_MODEL_NAME)
        return cls._embedding_model

    # ==============================
    #   LLM settings - agent ready
    # ==============================
    granite2b = ChatOllama(model="granite3.3:2b", temperature=0.7)
    granite8b = ChatOllama(model="granite3.3:8b", temperature=0.7)
    llama = ChatOllama(model="llama3.1:8b", temperature=0.7)

    # convenience alias that will attempt to load the model when accessed
    @property
    def embedding_model(self):
        return self.get_embedding_model()

    # =======================
    #   Workflow settings
    # =======================
    MAX_RETRIES = 3
    MAX_PAPERS_PER_HYPOTHESIS = 2
    MAX_SEARCH_RESULTS = 5  # Max results per tool, per search key


# Backwards-compatible module-level constants
# Some parts of the code import these names from the module directly
MAX_RETRIES = Config.MAX_RETRIES
MAX_PAPERS_PER_HYPOTHESIS = Config.MAX_PAPERS_PER_HYPOTHESIS
MAX_SEARCH_RESULTS = Config.MAX_SEARCH_RESULTS



