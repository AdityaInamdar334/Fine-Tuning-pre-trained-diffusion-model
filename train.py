import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt

from unet import UNet
from ddpm import DDPMScheduler

def train():
    # Hyperparameters
    batch_size = 128
    learning_rate = 1e-4
    epochs = 10
    num_timesteps = 1000
    save_dir = "samples"
    checkpoint_dir = "checkpoints"
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 1. Device configuration (use MPS on Apple Silicon if available)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 2. Data loading & Preprocessing
    # Diffusion models work best when pixels are normalized to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("Loading Fashion-MNIST dataset...")
    dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 3. Model & Scheduler initialization
    model = UNet(in_channels=1, out_channels=1).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=num_timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        
        for step, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            batch_size_current = images.shape[0]
            
            # Sample random timesteps t for each image in the batch
            timesteps = torch.randint(0, num_timesteps, (batch_size_current,), device=device).long()
            
            # Sample standard Gaussian noise
            noise = torch.randn_like(images)
            
            # Add noise to the clean images (Forward process)
            noisy_images = scheduler.add_noise(images, noise, timesteps)
            
            # Predict the noise using the U-Net (Backward process prediction)
            predicted_noise = model(noisy_images, timesteps)
            
            # Calculate MSE loss between actual and predicted noise
            loss = criterion(predicted_noise, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} finished. Average Loss: {avg_loss:.6f}")
        
        # Save checkpoints and generate visual progression samples
        torch.save(model.state_dict(), f"{checkpoint_dir}/ddpm_epoch_{epoch}.pth")
        generate_samples(model, scheduler, epoch, device, save_dir)

@torch.no_grad()
def generate_samples(model, scheduler, epoch, device, save_dir):
    """
    Generate samples using the DDPM reverse denoising process.
    """
    model.eval()
    print(f"Generating samples for epoch {epoch}...")
    
    # Start with pure random noise: Shape [16, 1, 28, 28]
    n_samples = 16
    samples = torch.randn(n_samples, 1, 28, 28, device=device)
    
    # Iterative reverse diffusion process
    for t in tqdm(reversed(range(scheduler.num_train_timesteps)), total=scheduler.num_train_timesteps, desc="Sampling"):
        # Create a tensor of the current timestep for the whole batch
        timesteps = torch.full((n_samples,), t, device=device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = model(samples, timesteps)
        
        # Remove noise / step backward
        samples = scheduler.step(predicted_noise, t, samples)
        
    # Scale back to [0, 1] range for visualization/saving
    samples = (samples + 1.0) / 2.0
    samples = torch.clamp(samples, 0.0, 1.0)
    
    # Save as image grid
    grid = make_grid(samples, nrow=4)
    save_path = f"{save_dir}/epoch_{epoch}.png"
    save_image(grid, save_path)
    print(f"Saved samples grid to {save_path}")

if __name__ == "__main__":
    train()
