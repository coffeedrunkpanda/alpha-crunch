from enum import Enum

class VectorDBProvider(str, Enum):
    CHROMA = "chroma"
    CHROMA_MODAL = "chroma-modal"
    PINECONE = "pinecone"