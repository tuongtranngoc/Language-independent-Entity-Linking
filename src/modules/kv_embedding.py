from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from sentence_transformers import SentenceTransformer
import torch

class KVEmbedding:
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
    
    def embedding(self, sentences):
        embeddings = self.model.encode(sentences)
        return abs(embeddings.mean())
