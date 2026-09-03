from functools import cached_property
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from .embedding_config import EmbeddingConfig


class TextEmbedder(Embeddings):
    """
    A generic text embedding service configured via an EmbeddingConfig object.
    Inherits from langchain_core.embeddings.Embeddings so it plugs directly
    into LangChain vector stores.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config

        if not self.config.model_name:
            raise ValueError("Model name must be provided in config or via 'MODEL_NAME' environment variable.")

        model_kwargs = {"device": self.config.device}
        model_kwargs.update(self.config.extra_model_kwargs)

        encode_kwargs = {
            "normalize_embeddings": self.config.normalize_embeddings,
            "batch_size": self.config.batch_size,
        }
        encode_kwargs.update(self.config.extra_encode_kwargs)

        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.config.model_name,
            cache_folder=self.config.cache_folder,
            show_progress=self.config.show_progress,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    def embed_query(self, text: str) -> List[float]:
        formatted_text = self._format_text(text, self.config.query_instruction)
        return self._embeddings.embed_query(formatted_text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        formatted_texts = [
            self._format_text(text, self.config.document_instruction)
            for text in texts
        ]
        return self._embeddings.embed_documents(formatted_texts)

    def _format_text(self, text: str, instruction: Optional[str]) -> str:
        if instruction:
            return f"{instruction.rstrip()} {text}"
        return text

    @cached_property
    def embedding_dimension(self) -> int:
        return len(self.embed_query("dimension probe"))
