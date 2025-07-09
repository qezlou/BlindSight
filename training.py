import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [seq_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

# --- VAE Encoder ---
class TransformerVAEEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, nhead, num_layers, latent_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_mu = nn.Linear(emb_dim, latent_dim)
        self.to_logvar = nn.Linear(emb_dim, latent_dim)

    def forward(self, x):
        x = self.embedding(x)  # [B, L, emb_dim]
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # [L, B, emb_dim] for transformer
        encoded = self.transformer_encoder(x)  # [L, B, emb_dim]
        encoded = encoded.mean(dim=0)  # Global average pooling
        mu = self.to_mu(encoded)
        logvar = self.to_logvar(encoded)
        return mu, logvar

# --- Reparameterization ---
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# --- VAE Decoder ---
class TransformerVAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, emb_dim, seq_len):
        super().__init__()
        self.latent_to_emb = nn.Linear(latent_dim, emb_dim)
        self.decoder = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.output_layer = nn.Linear(emb_dim, output_dim)
        self.seq_len = seq_len

    def forward(self, z):
        z = self.latent_to_emb(z)  # [B, emb_dim]
        z = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # [B, seq_len, emb_dim]
        out, _ = self.decoder(z)
        return self.output_layer(out)  # [B, seq_len, output_dim]

# --- Full VAE Model ---
class TransformerVAE(nn.Module):
    def __init__(self, input_dim=1, seq_len=128, emb_dim=64, latent_dim=16, nhead=4, num_layers=2):
        super().__init__()
        self.encoder = TransformerVAEEncoder(input_dim, emb_dim, nhead, num_layers, latent_dim)
        self.decoder = TransformerVAEDecoder(latent_dim, input_dim, emb_dim, seq_len)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar



def vae_loss(x_hat, x, mu, logvar):
    recon_loss = F.mse_loss(x_hat, x, reduction='mean')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_div, recon_loss, kl_div


# Training Setup:
"""
model = TransformerVAE(input_dim=1, seq_len=128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

x_batch = ...  # [B, L, 1] noisy spectrum
target_batch = ...  # [B, L, 1] clean Gaussian

x_hat, mu, logvar = model(x_batch)
loss, recon, kl = vae_loss(x_hat, target_batch, mu, logvar)
loss.backward()
optimizer.step()
"""

# Next steps
"""
Next Ideas
	•	Add classifier head after residuals to tag false positives.
	•	Visualize x vs x̂ to analyze contamination patterns.
	•	Use attention maps to understand what the model focuses on — especially useful for scientific interpretation.
"""

