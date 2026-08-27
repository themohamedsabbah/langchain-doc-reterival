import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass(frozen= True)
class EmbeddingConfig:
    model_name: Optional[str] = field(default_factory=lambda: os.environ.get("MODEL_NAME"))
    device: str = field(default_factory=lambda: os.environ.get("DEVICE", "cpu"))
    normalize_embeddings: bool = True

    batch_size: int = 32
    cache_folder: Optional[str] = None
    show_progress: bool = False

    extra_model_kwargs: Dict[str, Any] = field(default_factory=dict)
    extra_encode_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Example for BGE: "Represent this sentence for searching relevant passages: "
    query_instruction: Optional[str] = None
    # Usually empty for most models, but E5 requires "passage: "
    document_instruction: Optional[str] = None

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
