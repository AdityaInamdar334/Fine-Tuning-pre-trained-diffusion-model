# Denoising Diffusion Probabilistic Models (DDPM) on Fashion-MNIST

This repository contains a clean, from-scratch PyTorch implementation of a **Denoising Diffusion Probabilistic Model (DDPM)**. It is designed to train and fine-tune on the **Fashion-MNIST** dataset, demonstrating the forward and reverse diffusion processes.

## 🚀 Key Features

* **Custom U-Net Architecture**: Built with skip connections, sinusoidal time-step embeddings, and Group Normalization (`unet.py`).
* **DDPM Denoising Scheduler**: Custom variance schedule ($\beta_t$) implementation mapping the forward process (noising) and backward process (denoising step-by-step) (`ddpm.py`).
* **Hardware Acceleration**: Automatic support for Apple Silicon GPU (`mps`), NVIDIA CUDA (`cuda`), and CPU.
* **Epoch-wise Visual Tracking**: Generates grid samples after each epoch to visualize training progress.

---

## 🏗️ Architectural Improvements

Compared to standard vanilla U-Net implementations, this model incorporates key improvements crucial for stable diffusion training:

1. **Sinusoidal Timestep Embeddings**:
   * Timesteps $t$ are projected into high-dimensional sinusoidal embeddings. This allows a single network to learn the shared denoising function across all diffusion timesteps ($t \in [0, 999]$).
2. **Group Normalization (GroupNorm) over BatchNorm**:
   * Standard Batch Normalization is highly dependent on batch size and can be unstable during diffusion training. Replacing it with **GroupNorm (8 groups)** ensures normalization per sample/group, stabilizing training statistics.
3. **Conditioned Double Conv Blocks**:
   * Every convolutional block projects the time embedding to match the channel dimension and adds it directly to the intermediate spatial feature maps (`h = h + time_projected`). This ensures the model dynamically conditions its output based on the noise level of the current step.
4. **Dynamic Rescaling & Padding in Decoder**:
   * Added dynamic padding calculation during decoding upsampling to prevent dimensions mismatches when dealing with arbitrary feature map resolutions.

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

## 📊 Training Progression Results

Here is the quality progression of generated samples across different training epochs:

| Epoch 1 (Initial noise/coarse outlines) | Epoch 5 (Recognizable article shapes) | Epoch 10 (Clear clothing articles) |
|:---:|:---:|:---:|
| ![Epoch 1](samples/epoch_1.png) | ![Epoch 5](samples/epoch_5.png) | ![Epoch 10](samples/epoch_10.png) |

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
