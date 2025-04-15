############################################################################################
# Reference: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
############################################################################################


import torch
import torch.nn as nn
from open_clip import create_model_from_pretrained, get_tokenizer
from typing import List, Union, Optional, Tuple


class CLIPEncoder(nn.Module):
    """
    Wrapper class for CLIP text encoder to generate embeddings for diffusion model.
    """
    def __init__(
        self,
        model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        max_length: int = 77,
        embedding_dim: Optional[int] = None,
        use_projection: bool = True
    ):
        super().__init__()
        self.model, _ = create_model_from_pretrained(f'hf-hub:{model_name}')
        self.tokenizer = get_tokenizer(f'hf-hub:{model_name}')
        self.max_length = max_length

        for param in self.model.parameters():
            param.requires_grad = False

        # Get embedding dimension if not provided
        if embedding_dim is None:
            with torch.no_grad():
                dummy_text = ["test"]
                dummy_tokens = self.tokenizer(dummy_text, context_length=self.max_length)
                dummy_embedding = self.model.encode_text(dummy_tokens)
                embedding_dim = dummy_embedding.shape[-1]
        
        self.embedding_dim = embedding_dim
        
        # Convert embedding dims to what diffusion is needed.
        self.use_projection = use_projection
        if use_projection:
            self.diffusion_dim = 512
            self.projection = nn.Sequential(
                nn.Linear(embedding_dim, self.diffusion_dim),
                nn.SiLU(),
                nn.Linear(self.diffusion_dim, self.diffusion_dim)
            )

    def forward(self, text: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text into embeddings."""
        if isinstance(text, str):
            text = [text]
        
        tokenized = self.tokenizer(
            text,
            context_length=self.max_length
        )

        if isinstance(tokenized, torch.Tensor):
            tokenized = tokenized.to(self.device)

        with torch.no_grad():
            raw_embeddings = self.model.encode_text(tokenized)
            raw_embeddings = raw_embeddings.unsqueeze(1)  
        
        if hasattr(self.model, 'logit_scale'):
            raw_embeddings = raw_embeddings / raw_embeddings.norm(dim=-1, keepdim=True)
        
        if self.use_projection:
            projected_embeddings = self.projection(raw_embeddings)
            return raw_embeddings, projected_embeddings
        else:
            return raw_embeddings, raw_embeddings
        
    def encode_batch(self, captions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(captions)

    def to(self, device):
        """Move model to a specified device."""
        self.model.to(device)
        if self.use_projection:
            self.projection.to(device)
        return super().to(device)
    
    @property
    def device(self):
        """Get the model's device."""
        return next(self.model.parameters()).device
    
