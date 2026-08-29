from abc import ABC, abstractmethod
from typing import List

class BaseEmbedder(ABC):
    """
    Abstract base interface defining the contract for all embedding implementations.
    Any concrete embedder (HuggingFace, OpenAI, etc.) must implement these methods.
    """

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string for retrieval."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document strings for indexing."""
        pass

    @abstractmethod
    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously embed a single query string."""
        pass

    @abstractmethod
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed a list of document strings."""
        pass