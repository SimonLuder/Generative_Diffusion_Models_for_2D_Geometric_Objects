import torch
from torch import nn


# https://github.com/coderpiaobozhe/classifier-free-diffusion-guidance-Pytorch/blob/master/embedding.py
class ConditionalClassEmbedding(nn.Module):
    """This class represents a conditional class embedding layer. 

    Args:
        input_dim (int): Number of classes.
        embedding_dim (int): The size of each embedding vector.
        out_dim (int): The size of the embedding vector after passing the subsequent linear layers.

    Returns:
        torch.Tensor: embedding vectors for the classes
    """
    def __init__(self, input_dim:int, embedding_dim:int, out_dim:int):
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim),
            nn.Linear(embedding_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        emb = self.condEmbedding(x)
        return emb