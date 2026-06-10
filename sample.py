import os
import argparse
import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from unet import UNet
from ddpm import DDPMScheduler

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Generate samples from trained DDPM model.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ddpm_epoch_10.pth", help="Path to the model checkpoint")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--output", type=str, default="generated_samples.png", help="Path to output image")
    args = parser.parse_args()

    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load Model
    model = UNet(in_channels=1, out_channels=1).to(device)
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found. Please run training first.")
        return
        
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Initialize Scheduler
    scheduler = DDPMScheduler(num_train_timesteps=args.num_timesteps, device=device)

    # Generate noise
    print(f"Generating {args.num_samples} samples...")
    samples = torch.randn(args.num_samples, 1, 28, 28, device=device)

    # Sampling Loop
    for t in tqdm(reversed(range(args.num_timesteps)), total=args.num_timesteps, desc="Sampling"):
        timesteps = torch.full((args.num_samples,), t, device=device, dtype=torch.long)
        predicted_noise = model(samples, timesteps)
        samples = scheduler.step(predicted_noise, t, samples)

    # Normalize samples to [0, 1]
    samples = (samples + 1.0) / 2.0
    samples = torch.clamp(samples, 0.0, 1.0)

    # Save
    grid = make_grid(samples, nrow=int(args.num_samples ** 0.5))
    save_image(grid, args.output)
    print(f"Saved generated samples to {args.output}")

if __name__ == "__main__":
    main()
