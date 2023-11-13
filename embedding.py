import torch
from torch import nn
import clip

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
    def __init__(self, num_embeddings:int, embedding_dim:int, out_dim:int):
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim),
            nn.Linear(embedding_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        emb = self.condEmbedding(x)
        return emb
    
    


class CLIPTextEmbedding(nn.Module):
    
    def __init__(self, out_dim:int, clip_model_name:str="ViT-B/32", device:str="cpu"):
        super().__init__()
        
        self.device = device
        self.clip_encoder, _ = clip.load(name=clip_model_name, device=device)
        self.embedding_dim = self.get_embedding_dim()
        self.fully_connected = nn.Sequential(
            nn.Linear(self.embedding_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # we only want to train the fully connected layers without clip
        with torch.no_grad():
            x = self.clip_encoder.encode_text(x)
        emb = self.fully_connected(x)
        return emb
    
    def get_embedding_dim(self):

        # Create a dummy input tensor
        dummy_input = clip.tokenize("This is a test string").to(self.device)
        
        # Pass the dummy input through the clip_encoder to get the embedding size
        with torch.no_grad():
            embedding_dim = self.clip_encoder.encode_text(dummy_input).size(1)
            return embedding_dim
