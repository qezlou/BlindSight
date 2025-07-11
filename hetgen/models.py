
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
# --- Spectrum Normalizer ---
class SpectrumNormalizer:
    """
    Normalizes 1D spectra with multiple features such as flux, sky spectrum, and noise.

    Methods:
        fit(x): Computes per-feature mean and std from input tensor.
        transform(x): Applies normalization using stored stats.
        inverse_transform(x): Reverts normalization to original scale.

    Args:
        eps (float): Small constant to avoid division by zero.
    """
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.means = None
        self.stds = None

    def fit(self, x):
        """
        Compute mean and std for each feature (channel).

        Args:
            x (Tensor): Input of shape [batch_size, seq_len, num_features]
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        flat = x.view(-1, x.size(-1))  # collapse batch and sequence
        self.means = flat.mean(dim=0)
        self.stds = flat.std(dim=0)

    def transform(self, x):
        """
        Normalize input using stored mean and std.

        Args:
            x (Tensor): Input of shape [batch_size, seq_len, num_features]

        Returns:
            Tensor: Normalized input, same shape.
        """
        if self.means is None or self.stds is None:
            raise RuntimeError("Call fit() before transform().")
        x = (x - self.means) / (self.stds + self.eps)
        return x.float()

    def inverse_transform(self, x):
        """
        Revert normalization.

        Args:
            x (Tensor): Normalized tensor.

        Returns:
            Tensor: Original-scale tensor.
        """
        return x * (self.stds + self.eps) + self.means

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    """
    Applies sinusoidal positional encoding to represent the position of each spectral bin.

    Args:
        d_model (int): Dimension of embedding.
        max_len (int): Maximum length of the sequence.

    Inputs:
        x (Tensor): Shape (batch_size, seq_len, d_model)

    Returns:
        Tensor: Same shape as input with position information added.
    """
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        # Create positional encodings for each spectral bin (wavelength position) using sinusoids of varying frequencies.
        # This encodes the relative position of each point in the 1D spectrum (important for transformer models).
        pe = torch.zeros(max_len, d_model)  # [seq_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Store the positional encodings as a non-trainable buffer so they move with the model across devices.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add position encoding to input embeddings: [batch, seq_len, emb_dim].
        # Each spectral bin now contains both the original information and its relative position.
        return x + self.pe[:x.size(1)]

# --- VAE Encoder ---
class TransformerVAEEncoder(nn.Module):
    """
    Transformer-based encoder that embeds 1D spectral input and encodes it into a Gaussian latent distribution.

    Args:
        input_dim (int): Input feature dimension.
            For 1D spectra, it is 1 (flux per wavelength).
            Other contextual information can be:
            - Noise estimate
            - Sky spectrum
            - Mask or quality flag
            - Fiber/IFU position index.
        emb_dim (int): Embedding dimension, Default = 96
            emb_dim > input_dim provides more expressive power.
            Try 128 if not enough.
        nhead (int): Number of attention heads in transformer, Default = 8
            Depends on emb_dim, make sure emb_dim % nhead == 0.
            If you have multiple overlapping features, more heads 
            help distinguish them. Maybe keep between 2-8.
        num_layers (int): Number of transformer layers, Default = 4
            Deeper layers can capture long-range dependencies along
            the spectral dimension.
                - Simple spectra 1-2, complex spectra 3-6.
        latent_dim (int): Dimension of the latent space, Default = 32
            Larger number makes outlier detection more powerful.
            Also try 48.
            You should scale latent_dim with:
	            - Model size (emb_dim, num_layers, etc.)
	            - Training dataset size and diversity


    Inputs:
        x (Tensor): Shape (batch_size, seq_len, input_dim)
        The input is a 1D spectrum, where each point represents flux at a specific wavelength.
    Returns:
        Tuple[Tensor, Tensor]: Mean and log-variance tensors of shape (batch_size, latent_dim)
    """
    def __init__(self, input_dim, emb_dim=96, nhead=8, num_layers=4, latent_dim=32):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_mu = nn.Linear(emb_dim, latent_dim)
        self.to_logvar = nn.Linear(emb_dim, latent_dim)

    def forward(self, x):
        # Project the 1D spectral input (e.g., flux per wavelength) into a higher dimensional embedding.
        #print(f'TransformerVAE x.shape {x.shape} before embeding', flush=True)
        x = self.embedding(x)  # [B, L, emb_dim]
        # Add sinusoidal position encoding to inject wavelength position information.
        x = self.pos_encoder(x)
        # Permute dimensions to match transformer's expected input shape [seq_len, batch, features].
        x = x.permute(1, 0, 2)  # [L, B, emb_dim] for transformer
        # Run through transformer encoder to model contextual dependencies between spectral bins.
        encoded = self.transformer_encoder(x)  # [L, B, emb_dim]
        # Use global average pooling to summarize the sequence into a fixed-size latent vector.
        encoded = encoded.mean(dim=0)  # Global average pooling
        mu = self.to_mu(encoded)
        logvar = self.to_logvar(encoded)
        return mu, logvar

# --- Reparameterization ---
def reparameterize(mu, logvar):
    """
    Samples a latent vector from a Gaussian using the reparameterization trick.

    Args:
        mu (Tensor): Mean tensor of shape (batch_size, latent_dim)
        logvar (Tensor): Log-variance tensor of shape (batch_size, latent_dim)

    Returns:
        Tensor: Sampled latent vector of shape (batch_size, latent_dim)
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# --- VAE Decoder ---
class TransformerVAEDecoder(nn.Module):
    """
    Decoder that reconstructs 1D spectra from latent vectors using a GRU.

    Args:
        latent_dim (int): Dimension of the latent space.
        output_dim (int): Output feature dimension.
        emb_dim (int): Embedding dimension.
        seq_len (int): Length of the output sequence.

    Inputs:
        z (Tensor): Latent vector of shape (batch_size, latent_dim)

    Returns:
        Tensor: Reconstructed spectrum of shape (batch_size, seq_len, output_dim)
    """
    def __init__(self, latent_dim, output_dim, emb_dim, seq_len):
        super().__init__()
        self.latent_to_emb = nn.Linear(latent_dim, emb_dim)
        self.decoder = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.output_layer = nn.Linear(emb_dim, output_dim)
        self.seq_len = seq_len

    def forward(self, z):
        # Map latent vector back to the embedding space.
        z = self.latent_to_emb(z)  # [B, emb_dim]
        # Repeat across the expected sequence length to match shape for GRU decoder.
        z = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # [B, seq_len, emb_dim]
        # Decode to reconstruct the spectrum from the latent representation.
        out, _ = self.decoder(z)
        return self.output_layer(out)  # [B, seq_len, output_dim]

# --- Full VAE Model ---
class TransformerVAE(nn.Module):
    """
    Full Transformer-based VAE model for encoding and decoding 1D spectral data.

        Args:

        input_dim (int): Input feature dimension, default = 3
            For 1D spectra, it is 1 (flux per wavelength).
            Other contextual information can be 
            (default includes the first three):
            - Noise estimate
            - Sky spectrum
            - Mask or quality flag
            - Fiber/IFU position index.

        seq_len (int): Spectral sequence length, Default = 1036
            Length of the input spectrum (number of spectral bins).
            For HETDEX, this is typically 1036 with 2 Angstrom resolution.

        emb_dim (int): Embedding dimension, Default = 96
            emb_dim > input_dim provides more expressive power.
            Try 128 if not enough.
            
        nhead (int): Number of attention heads in transformer, Default = 8
            Depends on emb_dim, make sure emb_dim % nhead == 0.
            If you have multiple overlapping features, more heads 
            help distinguish them.

        num_layers (int): Number of transformer layers, Default = 4
            Deeper layers can capture long-range dependencies along
            the spectral dimension.
                - Simple spectra 1-2, complex spectra 3-6.

        latent_dim (int): Dimension of the latent space, Default = 32
            Larger number makes outlier detection more powerful.
            Also try 48.
            You should scale latent_dim with:
	            - Model size (emb_dim, num_layers, etc.)
	            - Training dataset size and diversity

    
    Inputs:
        x (Tensor): Input spectrum of shape (batch_size, seq_len, input_dim)

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Reconstructed spectrum, mu, logvar
    """
    def __init__(self, input_dim=3, seq_len=1036, emb_dim=96, nhead=4, num_layers=4, latent_dim=32):
        super().__init__()
        self.encoder = TransformerVAEEncoder(input_dim, emb_dim, nhead, num_layers, latent_dim)
        self.decoder = TransformerVAEDecoder(latent_dim, input_dim, emb_dim, seq_len)

    def forward(self, x):
        # Full VAE pipeline: encode input spectrum, sample from latent distribution, reconstruct spectrum.
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # from [B, L] → [B, L, 1]
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


def vae_loss(x_hat, x, mu, logvar):
    """
    Computes total VAE loss composed of reconstruction error and KL divergence.

    Args:
        x_hat (Tensor): Reconstructed spectrum of shape (batch_size, seq_len, output_dim)
        x (Tensor): Ground truth spectrum of shape (batch_size, seq_len, output_dim)
        mu (Tensor): Mean of latent Gaussian.
        logvar (Tensor): Log-variance of latent Gaussian.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Total loss, reconstruction loss, KL divergence
    """
    if x.dim() == 2:
        x = x.unsqueeze(-1) # [B, L] -> [B, L, 1]
    # VAE loss = reconstruction loss + KL divergence (latent regularization).
    # This encourages the latent space to follow a standard normal distribution and ensures reconstruction quality.
    recon_loss = F.mse_loss(x_hat, x, reduction='mean')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_div, recon_loss, kl_div
