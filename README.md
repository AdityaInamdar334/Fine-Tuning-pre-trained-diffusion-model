# Denoising Diffusion Probabilistic Models (DDPM) on Fashion-MNIST

This repository contains a clean, from-scratch PyTorch implementation of a **Denoising Diffusion Probabilistic Model (DDPM)**. It is designed to train and fine-tune on the **Fashion-MNIST** dataset, demonstrating the forward and reverse diffusion processes.

## 🚀 Key Features

* **Custom U-Net Architecture**: Built with skip connections, sinusoidal time-step embeddings, and Group Normalization (`unet.py`).
* **DDPM Denoising Scheduler**: Custom variance schedule ($\beta_t$) implementation mapping the forward process (noising) and backward process (denoising step-by-step) (`ddpm.py`).
* **Hardware Acceleration**: Automatic support for Apple Silicon GPU (`mps`), NVIDIA CUDA (`cuda`), and CPU.
* **Epoch-wise Visual Tracking**: Generates grid samples after each epoch to visualize training progress.

---

## 📁 Repository Structure

* **`ddpm.py`**: Implementation of `DDPMScheduler` which controls adding noise to image samples and executing backward denoising steps.
* **`unet.py`**: U-Net model with a sinusoidal timestep embedding module, downsampling blocks, bottleneck, and upsampling blocks (with skip connections).
* **`train.py`**: Script to set up data loaders, initialize the optimizer and DDPM components, run training loops, and save checkpoint parameters alongside sample generation.
* **`sample.py`**: Standalone command-line tool to load a saved checkpoint and generate arbitrary samples.

---

## 🛠️ Installation & Setup

1. **Clone or Navigate to the Directory**:
   ```bash
   cd "Fine Tuning pre-trained-diffusion-model"
   ```

2. **Set up Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install torch torchvision tqdm matplotlib
   ```

---

## 🏋️ Training the Model

Start the training process using the following command:

```bash
python train.py
```

* **Dataset**: Automatically downloads the Fashion-MNIST dataset into a `./data` folder.
* **Epochs**: Default is set to `10` epochs.
* **Checkpoints**: Saved to `./checkpoints/ddpm_epoch_*.pth` at the end of each epoch.
* **Progress Samples**: Generated grids are saved to `./samples/epoch_*.png` to verify the learning progression.

---

## 🎨 Generating Samples

You can generate samples using a trained checkpoint (e.g., from epoch 10) with the standalone script:

```bash
python sample.py --checkpoint checkpoints/ddpm_epoch_10.pth --num_samples 16 --output generated_samples.png
```

### Options:
* `--checkpoint`: Path to the trained `.pth` model checkpoint (default: `checkpoints/ddpm_epoch_10.pth`).
* `--num_samples`: Total number of samples to generate in the output grid (default: `16`).
* `--num_timesteps`: Total number of backward diffusion steps (default: `1000`).
* `--output`: Filepath to save the generated grid image (default: `generated_samples.png`).
