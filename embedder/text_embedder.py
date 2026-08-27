from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings

from base_embedder import BaseEmbedder
from embedding_config import EmbeddingConfig


class TextEmbedder(BaseEmbedder):
    """
    A generic text embedding service configured via an EmbeddingConfig object.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config

        if not self.config.model_name:
            raise ValueError("Model name must be provided in config or via 'MODEL_NAME' environment variable.")

        model_kwargs = {"device": self.config.device}
        model_kwargs.update(self.config.extra_model_kwargs)

        encode_kwargs = {
            "normalize_embeddings": self.config.normalize_embeddings,
            "batch_size": self.config.batch_size
        }
        encode_kwargs.update(self.config.extra_encode_kwargs)

        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.config.model_name,
            cache_folder=self.config.cache_folder,
            show_progress=self.config.show_progress,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        formatted_text = self._format_text(text, self.config.query_instruction)
        return self._embeddings.embed_query(formatted_text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document strings."""
        formatted_texts = [
            self._format_text(text, self.config.document_instruction) 
            for text in texts
        ]
        return self._embeddings.embed_documents(formatted_texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously embed a single query string."""
        formatted_text = self._format_text(text, self.config.query_instruction)
        return await self._embeddings.aembed_query(formatted_text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed a list of document strings."""
        formatted_texts = [
            self._format_text(text, self.config.document_instruction) 
            for text in texts
        ]
        return await self._embeddings.aembed_documents(formatted_texts)

    def _format_text(self, text: str, instruction: Optional[str]) -> str:
        """Prepends the instruction if it exists."""
        if instruction:
            return f"{instruction.rstrip()} {text}"
        return text

    def get_embedding_dimension(self) -> int:
        return len(self.embed_query("dimension probe"))