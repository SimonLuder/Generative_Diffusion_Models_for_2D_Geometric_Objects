import torch
from torch import nn
import clip

# https://github.com/coderpiaobozhe/classifier-free-diffusion-guidance-Pytorch/blob/master/embedding.py
class ConditionalClassEmbedding(nn.Module):
    """This class represents a conditional class embedding. 

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
    
    
class TabularEmbedding(nn.Module):
    
    def __init__(self, input_dim:int, out_dim:int):
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        emb = self.condEmbedding(x)
        return emb
    

class CLIPTextEmbedding(nn.Module):
    """
    A text embedding model using the CLIP model.

    This class implements a text embedding model using the CLIP model. 
    The network consists of a pre-trained CLIP model and a fully connected layer block. 
    The output of the CLIP model is passed through the fully connected layers to generate the final embedding.

    Attributes:
    device (str): The device on which the model is running.
    clip_encoder (clip.Model): The pre-trained CLIP model.
    embedding_dim (int): The dimension of the output from the CLIP model.
    fully_connected (nn.Sequential): The fully connected layer block for generating the final embedding.

    Methods:
    forward(x: torch.Tensor) -> torch.Tensor: Defines the computation performed at every call.
    get_embedding_dim() -> int: Returns the embedding dimension of the CLIP model.
    """

    def __init__(self, out_dim:int, model_name:str="ViT-B/32", device:str="cpu"):
        super().__init__()
        
        self.device = device
        self.clip_encoder, _ = clip.load(name=model_name, device=device)
        self.embedding_dim = self.get_embedding_dim()
        self.fully_connected = nn.Sequential(
            nn.Linear(self.embedding_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # we only want to train the fully connected layers without clip
        with torch.no_grad():
            x = self.clip_encoder.encode_text(x).type(torch.float32)
        emb = self.fully_connected(x)
        return emb
    
    def get_embedding_dim(self):
        """
        Returns the embedding dimension of the CLIP model.

        Returns:
        int: The embedding dimension.
        """

        # Create a dummy input tensor
        dummy_input = clip.tokenize("This is a test string").to(self.device)
        
        # Pass the dummy input through the clip_encoder to get the embedding size
        with torch.no_grad():
            embedding_dim = self.clip_encoder.encode_text(dummy_input).size(1)
            return embedding_dim
        
        
class CLIPImageEmbedding(nn.Module):
    """
    A Convolutional Neural Network (CNN) based image embedding model using CLIP.

    This class implements a CNN for generating embeddings from images using the CLIP model. 
    The network consists of a pre-trained CLIP model and a fully connected layer block. 
    The output of the CLIP model is passed through the fully connected layers to generate the final embedding.

    Attributes:
    device (str): The device on which the model is running.
    clip_encoder (clip.Model): The pre-trained CLIP model.
    embedding_dim (int): The dimension of the output from the CLIP model.
    fully_connected (nn.Sequential): The fully connected layer block for generating the final embedding.

    Methods:
    forward(x: torch.Tensor) -> torch.Tensor: Defines the computation performed at every call.
    get_embedding_dim() -> int: Returns the embedding dimension of the CLIP model.
    """
    
    def __init__(self, out_dim:int, model_name:str="ViT-B/32", device:str="cpu"):
        super().__init__()
        
        self.device = device
        self.clip_encoder, _ = clip.load(name=model_name, device=device)
        self.embedding_dim = self.get_embedding_dim()
        self.fully_connected = nn.Sequential(
            nn.Linear(self.embedding_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # we only want to train the fully connected layers without clip
        with torch.no_grad():
            x = self.clip_encoder.encode_image(x).type(torch.float32)
        emb = self.fully_connected(x)
        return emb
    
    def get_embedding_dim(self):
        """
        Returns the embedding dimension of the CLIP model.

        Returns:
        int: The embedding dimension.
        """
        # Create a dummy input tensor
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Pass the dummy input through the clip_encoder to get the embedding size
        with torch.no_grad():
            embedding_dim = self.clip_encoder.encode_image(dummy_input).size(1)
            return embedding_dim
        

class CNNImageEmbedding(nn.Module):

    """
    A Convolutional Neural Network (CNN) based image embedding model.

    This class implements a CNN for generating embeddings from images. 
    The network consists of three convolutional layers, each followed by a ReLU activation function, and an adaptive average pooling layer. 
    The output of the convolutional layers is flattened and passed through a fully connected layer to generate the final embedding.

    Attributes:
    conv (nn.Sequential): The sequential container of convolutional layers.
    fc (nn.Linear): The fully connected layer for generating the final embedding.

    Methods:
    forward(x: torch.Tensor) -> torch.Tensor: Defines the computation performed at every call.
    """
    
    def __init__(self, input_channel:int, out_dim:int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # this will make the output (N, 128, 1, 1)
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        emb = self.fc(x)
        return emb

