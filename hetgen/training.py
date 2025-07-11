import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional

from .models import TransformerVAE, vae_loss, SpectrumNormalizer
from .data import load_training_data

class SpectralDataset(Dataset):
    """
    Dataset class for loading and preprocessing spectral data.
    
    This class handles spectral data loading and preprocessing for training VAE models.
    It supports both self-supervised (autoencoder) and supervised training modes.
    
    Args:
        spectra (np.ndarray): Input spectral data array of shape (N, seq_len, input_dim).
            - N: Number of spectra in the dataset
            - seq_len: Length of each spectrum (e.g., 1036 for HETDEX)
            - input_dim: Number of features per spectral bin (e.g., flux, noise, sky)
        targets (np.ndarray, optional): Target spectral data for supervised training.
            If None, uses input spectra as targets (autoencoder mode).
            Must have same shape as spectra if provided. Default: None
        normalizer (SpectrumNormalizer, optional): Pre-fitted normalizer instance
            for standardizing spectral data. If provided, normalization is applied
            to both input and target spectra. Default: None
    
    Attributes:
        spectra (torch.Tensor): Normalized input spectra as PyTorch tensors
        targets (torch.Tensor): Normalized target spectra as PyTorch tensors
        normalizer (SpectrumNormalizer): Normalizer instance used for preprocessing
    
    Example:
        >>> spectra = np.random.randn(1000, 1036, 3)  # 1000 spectra
        >>> normalizer = SpectrumNormalizer()
        >>> normalizer.fit(spectra)
        >>> dataset = SpectralDataset(spectra, normalizer=normalizer)
        >>> data_loader = DataLoader(dataset, batch_size=32)
    """
    def __init__(self, spectra: np.ndarray, targets: Optional[np.ndarray] = None, 
                 normalizer: Optional[SpectrumNormalizer] = None):
        self.spectra = torch.Tensor(spectra.astype(np.float32))
        self.targets = torch.Tensor(targets.astype(np.float32)) if targets is not None else self.spectra
        self.normalizer = normalizer
        
        if self.normalizer:
            self.spectra = self.normalizer.transform(self.spectra)
            self.targets = self.normalizer.transform(self.targets)
    
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        return self.spectra[idx], self.targets[idx]


class SpectralVAETrainer:
    """
    Comprehensive trainer class for the Transformer VAE model on spectral data.
    
    This class provides a complete training pipeline with support for training,
    validation, checkpointing, visualization, and logging. It handles the entire
    training lifecycle from initialization to model evaluation.
    
    Args:
        model (TransformerVAE): The VAE model instance to train. Should be
            initialized with appropriate architecture parameters.
        train_loader (DataLoader): PyTorch DataLoader for training data.
            Should yield batches of (input_spectra, target_spectra) tuples.
        val_loader (DataLoader, optional): PyTorch DataLoader for validation data.
            If None, validation is skipped during training. Default: None
        device (str): Device to run training on. Options: 'cuda', 'cpu', or
            specific GPU like 'cuda:0'. Default: 'cuda'
        learning_rate (float): Initial learning rate for the Adam optimizer.
            Typical values: 1e-3 to 1e-4. Default: 1e-3
        kl_weight (float): Weight multiplier for the KL divergence term in VAE loss.
            Controls the balance between reconstruction and regularization.
            - 1.0: Standard VAE loss
            - < 1.0: Emphasizes reconstruction (β-VAE with β < 1)
            - > 1.0: Emphasizes regularization (β-VAE with β > 1)
            Default: 1.0
        save_dir (str): Directory path for saving checkpoints, logs, and plots.
            Will be created if it doesn't exist. Default: './checkpoints'
    
    Attributes:
        model (TransformerVAE): The model being trained
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (str): Training device
        kl_weight (float): KL divergence weight
        optimizer (torch.optim.Adam): Adam optimizer instance
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): Learning rate scheduler
        save_dir (pathlib.Path): Directory for saving outputs
        train_losses (List[Dict]): History of training losses per epoch
        val_losses (List[Dict]): History of validation losses per epoch
        best_val_loss (float): Best validation loss achieved so far
        logger (logging.Logger): Logger instance for training progress
    
    Example:
        >>> model = TransformerVAE(input_dim=3, seq_len=1036)
        >>> trainer = SpectralVAETrainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     learning_rate=1e-3,
        ...     kl_weight=0.5,  # β-VAE with β=0.5
        ...     save_dir='./my_experiment'
        ... )
        >>> trainer.fit(num_epochs=100)
    """
    
    def __init__(self, 
                 model: TransformerVAE,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 device: str = 'cuda',
                 learning_rate: float = 1e-3,
                 kl_weight: float = 1.0,
                 save_dir: str = './checkpoints'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.kl_weight = kl_weight
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Tracking
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Setup logging
        self.setup_logging()
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        self.logger.info(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU found'}")
    
    def setup_logging(self):
        """
        Setup logging configuration for training progress and debugging.
        
        Configures logging to write to both a file and console with timestamps
        and appropriate formatting. Creates a training.log file in the save_dir.
        
        The logging configuration includes:
        - INFO level logging for general training progress
        - Timestamped entries for tracking training duration
        - Dual output to both file and console
        - Proper formatting for readability
        
        Note:
            This method is called automatically during trainer initialization.
            The log file will be created at {save_dir}/training.log
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def weighted_vae_loss(self, x_hat: torch.Tensor, x: torch.Tensor, 
                         mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss with configurable KL divergence weighting (β-VAE).
        
        Calculates the total VAE loss as a weighted sum of reconstruction loss
        and KL divergence regularization. The KL weight allows for β-VAE training
        where β controls the balance between reconstruction quality and latent
        space regularization.
        
        Args:
            x_hat (torch.Tensor): Reconstructed spectra from the decoder.
                Shape: (batch_size, seq_len, output_dim)
            x (torch.Tensor): Target/ground truth spectra.
                Shape: (batch_size, seq_len, output_dim)
            mu (torch.Tensor): Mean of the latent Gaussian distribution.
                Shape: (batch_size, latent_dim)
            logvar (torch.Tensor): Log-variance of the latent Gaussian distribution.
                Shape: (batch_size, latent_dim)
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - total_loss: Weighted sum of reconstruction and KL losses
                - recon_loss: Mean squared error between x_hat and x
                - kl_div: KL divergence between latent distribution and N(0,I)
        
        Note:
            - Reconstruction loss uses MSE, suitable for continuous spectral data
            - KL divergence is normalized by batch size
            - Total loss = recon_loss + kl_weight * kl_div
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1) # [B, L] -> [B, L, 1]
        recon_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        total_loss = recon_loss + self.kl_weight * kl_div
        return total_loss, recon_loss, kl_div
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Execute one complete training epoch over the training dataset.
        
        Performs forward and backward passes for all batches in the training
        loader, updates model parameters, and tracks training metrics. Includes
        gradient clipping for training stability.
        
        Returns:
            Dict[str, float]: Dictionary containing averaged losses for the epoch:
                - 'total': Total VAE loss (reconstruction + weighted KL)
                - 'recon': Reconstruction loss (MSE)
                - 'kl': KL divergence loss
        
        Training Process:
            1. Set model to training mode
            2. For each batch:
               - Forward pass through the model
               - Compute weighted VAE loss
               - Backward pass and gradient computation
               - Gradient clipping (max_norm=1.0)
               - Parameter update
            3. Average losses across all batches
        
        Note:
            - Uses gradient clipping to prevent exploding gradients
            - Tracks progress with tqdm progress bar
            - All losses are accumulated and averaged over the epoch
        """
        self.model.train()
        epoch_losses = {'total': 0, 'recon': 0, 'kl': 0}
        num_batches = 0
        
        for batch_idx, (data, targets) in enumerate(tqdm(self.train_loader, desc="Training")):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            x_hat, mu, logvar = self.model(data)
            total_loss, recon_loss, kl_div = self.weighted_vae_loss(x_hat, targets, mu, logvar)
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['kl'] += kl_div.item()
            num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model on the validation dataset.
        
        Evaluates the model performance on validation data without updating
        model parameters. Used for monitoring overfitting and learning rate
        scheduling.
        
        Returns:
            Dict[str, float]: Dictionary containing averaged validation losses:
                - 'total': Total VAE loss (reconstruction + weighted KL)
                - 'recon': Reconstruction loss (MSE)
                - 'kl': KL divergence loss
            Returns empty dict if no validation loader is provided.
        
        Process:
            1. Set model to evaluation mode
            2. Disable gradient computation for efficiency
            3. Forward pass on all validation batches
            4. Compute and accumulate losses
            5. Return averaged losses
        
        Note:
            - No gradient computation or parameter updates
            - Uses torch.no_grad() for memory efficiency
            - Tracks progress with tqdm progress bar
        """
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        val_losses = {'total': 0, 'recon': 0, 'kl': 0}
        num_batches = 0
        
        with torch.no_grad():
            for data, targets in tqdm(self.val_loader, desc="Validation"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                x_hat, mu, logvar = self.model(data)
                total_loss, recon_loss, kl_div = self.weighted_vae_loss(x_hat, targets, mu, logvar)
                
                val_losses['total'] += total_loss.item()
                val_losses['recon'] += recon_loss.item()
                val_losses['kl'] += kl_div.item()
                num_batches += 1
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
            
        return val_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint to disk for resuming training or inference.
        
        Saves the complete training state including model weights, optimizer state,
        scheduler state, and training history. Optionally saves the best model
        separately for easy access.
        
        Args:
            epoch (int): Current epoch number for checkpoint naming and tracking.
            is_best (bool): Whether this checkpoint represents the best model
                so far based on validation loss. If True, saves an additional
                copy as 'best_model.pth'. Default: False
        
        Saved checkpoint contains:
            - Model state dictionary (weights and biases)
            - Optimizer state (for proper resuming)
            - Scheduler state (learning rate schedule)
            - Training and validation loss history
            - Current best validation loss
            - KL weight parameter
            - Epoch number
        
        Files created:
            - checkpoint_epoch_{epoch}.pth: Regular checkpoint
            - best_model.pth: Best model (if is_best=True)
        
        Note:
            Checkpoints are saved in the save_dir specified during initialization.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'kl_weight': self.kl_weight
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model and training state from a saved checkpoint.
        
        Restores the complete training state from a checkpoint file, allowing
        for seamless resumption of training from any saved epoch.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file (.pth format).
                Can be either a regular checkpoint or the best model checkpoint.
        
        Returns:
            int: The epoch number from which training can be resumed.
        
        Restored state includes:
            - Model weights and architecture parameters
            - Optimizer state (momentum, learning rate, etc.)
            - Learning rate scheduler state
            - Complete training and validation loss history
            - Best validation loss achieved so far
            - KL weight parameter
        
        Note:
            - The model architecture must match the saved checkpoint
            - Checkpoint is loaded to the device specified during trainer init
            - Training can be resumed from the returned epoch + 1
        
        Example:
            >>> trainer = SpectralVAETrainer(...)
            >>> start_epoch = trainer.load_checkpoint('./checkpoints/best_model.pth')
            >>> trainer.fit(num_epochs=100, start_from_epoch=start_epoch+1)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        return checkpoint['epoch']
    
    def plot_losses(self):
        """
        Generate and save comprehensive loss visualization plots.
        
        Creates a three-panel figure showing the training progress across epochs:
        1. Total VAE loss (reconstruction + weighted KL)
        2. Reconstruction loss (MSE)
        3. KL divergence loss
        
        Each panel shows both training and validation curves (if validation
        data is available) with proper legends and grid lines for easy reading.
        
        Features:
            - Automatic legend generation for train/validation curves
            - Grid lines for better readability
            - High-resolution PNG output (300 DPI)
            - Tight layout for professional appearance
            - Saves to save_dir/training_losses.png
        
        Note:
            - Only plots if training losses are available
            - Validation curves are included if validation was performed
            - Plot is both saved to disk and displayed (plt.show())
            - Useful for monitoring training progress and diagnosing issues
        """
        if not self.train_losses:
            return
            
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Total loss
        axes[0].plot(epochs, [loss['total'] for loss in self.train_losses], 'b-', label='Train')
        if self.val_losses:
            axes[0].plot(epochs, [loss['total'] for loss in self.val_losses], 'r-', label='Validation')
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Reconstruction loss
        axes[1].plot(epochs, [loss['recon'] for loss in self.train_losses], 'b-', label='Train')
        if self.val_losses:
            axes[1].plot(epochs, [loss['recon'] for loss in self.val_losses], 'r-', label='Validation')
        axes[1].set_title('Reconstruction Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        # KL divergence
        axes[2].plot(epochs, [loss['kl'] for loss in self.train_losses], 'b-', label='Train')
        if self.val_losses:
            axes[2].plot(epochs, [loss['kl'] for loss in self.val_losses], 'r-', label='Validation')
        axes[2].set_title('KL Divergence')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('KL Div')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_losses.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_reconstructions(self, num_samples: int = 5):
        """
        Visualize model reconstruction quality on validation samples.
        
        Creates a visual comparison between original spectra, target spectra,
        and model reconstructions to assess the quality of the learned
        representations and reconstruction capability.
        
        Args:
            num_samples (int): Number of sample spectra to visualize.
                Limited by the batch size of the validation loader.
                Default: 5
        
        Generated visualization includes:
            - Original input spectra (left column)
            - Target spectra (middle column) 
            - Model reconstructions (right column)
            - Grid layout with num_samples rows and 3 columns
            - Individual titles for each spectrum sample
        
        Features:
            - Uses validation data for unbiased evaluation
            - Model is set to evaluation mode during visualization
            - No gradient computation for efficiency
            - High-resolution PNG output saved to save_dir
            - Both file output and display (plt.show())
        
        Note:
            - Requires validation loader to be provided during initialization
            - Only plots the first channel of multi-dimensional spectra
            - Useful for qualitative assessment of model performance
            - Saved as save_dir/reconstructions.png
        """
        if self.val_loader is None:
            self.logger.warning("No validation loader available for visualization")
            return
            
        self.model.eval()
        
        with torch.no_grad():
            data, targets = next(iter(self.val_loader))
            data, targets = data.to(self.device), targets.to(self.device)
            
            x_hat, mu, logvar = self.model(data)
            
            # Move to CPU for plotting
            data = data.cpu().numpy()
            targets = targets.cpu().numpy()
            x_hat = x_hat.cpu().numpy()
            
            fig, axes = plt.subplots(num_samples, 3, figsize=(15, 3*num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(min(num_samples, len(data))):
                # Handle both 2D and 3D data (input_dim=1 vs input_dim>1)
                if data.ndim == 3:
                    axes[i, 0].plot(data[i, :, 0])
                    axes[i, 1].plot(targets[i, :, 0])
                    axes[i, 2].plot(x_hat[i, :, 0])
                else:
                    axes[i, 0].plot(data[i, :])
                    axes[i, 1].plot(targets[i, :])
                    axes[i, 2].plot(x_hat[i, :])

                axes[i, 0].set_title(f'Original Spectrum {i+1}')
                axes[i, 0].grid(True)

                axes[i, 1].set_title(f'Target Spectrum {i+1}')
                axes[i, 1].grid(True)

                axes[i, 2].set_title(f'Reconstructed Spectrum {i+1}')
                axes[i, 2].grid(True)
            
            plt.tight_layout()
            plt.savefig(self.save_dir / 'reconstructions.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def fit(self, num_epochs: int, save_every: int = 10, validate_every: int = 1):
        """
        Execute the main training loop for the specified number of epochs.
        
        This is the primary method for training the VAE model. It orchestrates
        the entire training process including training epochs, validation,
        checkpointing, learning rate scheduling, and progress logging.
        
        Args:
            num_epochs (int): Total number of training epochs to run.
                Each epoch processes the entire training dataset once.
            save_every (int): Frequency of checkpoint saving in epochs.
                For example, save_every=10 saves a checkpoint every 10 epochs.
                Does not affect best model saving. Default: 10
            validate_every (int): Frequency of validation in epochs.
                For example, validate_every=2 runs validation every 2 epochs.
                More frequent validation provides better monitoring but slows training.
                Default: 1 (validate every epoch)
        
        Training workflow per epoch:
            1. Log epoch information and model parameter count
            2. Execute training epoch with gradient updates
            3. Run validation (if scheduled)
            4. Update learning rate based on validation loss
            5. Save best model if validation loss improved
            6. Log training progress and losses
            7. Save regular checkpoint (if scheduled)
            8. Generate final visualizations after training completes
        
        Features:
            - Automatic best model tracking and saving
            - Learning rate reduction on validation plateau
            - Comprehensive logging of training progress
            - Periodic checkpointing for fault tolerance
            - Final loss plots and reconstruction visualizations
        
        Note:
            - Training can be interrupted and resumed using checkpoints
            - Validation loss is used for learning rate scheduling
            - Best model is saved automatically when validation improves
            - Final visualizations are generated at the end of training
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        for epoch in range(1, num_epochs + 1):
            self.logger.info(f"\nEpoch {epoch}/{num_epochs}")
            
            # Training
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)
            
            # Validation
            val_losses = {}
            if epoch % validate_every == 0:
                val_losses = self.validate()
                if val_losses:
                    self.val_losses.append(val_losses)
                    
                    # Learning rate scheduling
                    self.scheduler.step(val_losses['total'])
                    
                    # Check for best model
                    if val_losses['total'] < self.best_val_loss:
                        self.best_val_loss = val_losses['total']
                        self.save_checkpoint(epoch, is_best=True)
            
            # Logging
            log_msg = f"Train Loss: {train_losses['total']:.6f} (Recon: {train_losses['recon']:.6f}, KL: {train_losses['kl']:.6f})"
            if val_losses:
                log_msg += f" | Val Loss: {val_losses['total']:.6f} (Recon: {val_losses['recon']:.6f}, KL: {val_losses['kl']:.6f})"
            self.logger.info(log_msg)
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch)
        
        self.logger.info("Training completed!")
        
        # Final visualizations
        self.plot_losses()
        self.visualize_reconstructions()


def create_trainer_from_config(config_path: str) -> SpectralVAETrainer:
    """
    Create a SpectralVAETrainer instance from a JSON configuration file.
    
    This factory function provides a convenient way to set up training
    experiments using configuration files, promoting reproducibility and
    easy parameter management.
    
    Args:
        config_path (str): Path to the JSON configuration file containing
            model and training parameters. The file should contain nested
            dictionaries with keys 'model' and 'training'.
    
    Returns:
        SpectralVAETrainer: Configured trainer instance ready for training.
            Note: Data loaders need to be set separately as they depend on
            specific data loading implementations.
    
    Expected configuration file structure:
        {
            "model": {
                "input_dim": 3,
                "seq_len": 1036,
                "emb_dim": 96,
                "nhead": 4,
                "num_layers": 4,
                "latent_dim": 32
            },
            "training": {
                "device": "cuda",
                "learning_rate": 1e-3,
                "kl_weight": 1.0,
                "save_dir": "./checkpoints"
            }
        }
    
    Note:
        - The returned trainer has None for train_loader and val_loader
        - You must set these manually based on your data loading implementation
        - This function is useful for experiment management and reproducibility
    
    Example:
        >>> trainer = create_trainer_from_config('config.json')
        >>> trainer.train_loader = my_train_loader
        >>> trainer.val_loader = my_val_loader
        >>> trainer.fit(num_epochs=100)
    """
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    train_data, val_data = load_training_data( output_file=config['data']['output_file'],
                                              train_split=config['data']['train_split'])
    print(f'train_data.shape {train_data.shape} | val_data.shape {val_data.shape}', flush=True) 
    normalizer = SpectrumNormalizer()
    normalizer.fit(train_data)

    train_dataset = SpectralDataset(train_data, normalizer=normalizer)
    val_dataset = SpectralDataset(val_data, normalizer=normalizer)

    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False)

    model = TransformerVAE(**config['model'])

    trainer = SpectralVAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        **config['training']
    )
    return trainer


# Example usage and configuration
def create_example_config():
    """
    Create an example configuration file for training setup.

    Generates a comprehensive example configuration file with typical
    parameters for HETDEX spectral data training. This serves as a
    template for setting up new training experiments.

    Returns:
        dict: Configuration dictionary with model, training, and data parameters.
            The same dictionary is also saved as 'training_config.json'.

    Configuration includes:
        - Model architecture parameters (dimensions, layers, etc.)
        - Training hyperparameters (learning rate, device, etc.)
        - Data loading parameters (batch size, splits, etc.)

    Generated file structure:
        {
            "model": {
                "input_dim": 3,        # Flux, noise, sky features
                "seq_len": 1036,       # HETDEX spectral length
                "emb_dim": 96,         # Embedding dimension
                "nhead": 4,            # Attention heads
                "num_layers": 4,       # Transformer layers
                "latent_dim": 32       # Latent space dimension
            },
            "training": {
                "device": "cuda",      # GPU training
                "learning_rate": 1e-3, # Adam learning rate
                "kl_weight": 1.0,      # Standard VAE (β=1)
                "save_dir": "./checkpoints"
            },
            "data": {
                "batch_size": 32,      # Training batch size
                "num_workers": 4,      # Data loading workers
                "train_split": 0.8,    # Training data fraction
                "output_file": "hetdex_spectra.h5py"  # File to use for load_training_data
            }
        }

    Note:
        - Creates 'training_config.json' in the current directory
        - Parameters are optimized for HETDEX spectral data
        - Modify values as needed for your specific dataset
    """
    config = {
        "model": {
            "input_dim": 3,
            "seq_len": 1036,
            "emb_dim": 96,
            "nhead": 4,
            "num_layers": 4,
            "latent_dim": 32
        },
        "training": {
            "device": "cuda",
            "learning_rate": 1e-3,
            "kl_weight": 1.0,
            "save_dir": "../checkpoints"
        },
        "data": {
            "batch_size": 32,
            "num_workers": 4,
            "train_split": 0.8,
            "output_file": "../data/mock_spectra.h5"  # File to use for load_training_data
        }
    }
    
    with open('training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    return config
