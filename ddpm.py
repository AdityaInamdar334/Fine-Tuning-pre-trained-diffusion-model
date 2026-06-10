import torch

class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_train_timesteps = num_train_timesteps
        self.device = device
        
        # 1. Define linear variance schedule beta_t
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, device=device)
        
        # 2. Compute alphas and cumulative alphas (alpha_bar)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute values needed for the forward process (adding noise): q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Precompute values needed for the backward process (sampling): p(x_{t-1} | x_t)
        # We need alphas_cumprod_prev to calculate the posterior variance
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        
        # Posterior variance: \tilde{\beta}_t = \beta_t * (1 - \bar{\alpha}_{t-1}) / (1 - \bar{\alpha}_t)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, original_samples, noise, timesteps):
        """
        Forward process: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t)*x_0, (1 - alpha_bar_t)*I)
        """
        # Ensure timesteps shape is compatible for broadcasting: [batch_size, 1, 1, 1]
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        noisy_samples = sqrt_alpha_bar * original_samples + sqrt_one_minus_alpha_bar * noise
        return noisy_samples

    def step(self, model_output, timestep, sample):
        """
        Backward process step: Sample x_{t-1} from p(x_{t-1} | x_t) using predicted noise.
        """
        t = timestep
        
        # Retrieve scheduler parameters for the current timestep
        beta = self.betas[t].view(-1, 1, 1, 1)
        alpha = self.alphas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Predict the mean of x_{t-1}
        # Equation: x_{t-1} mean = 1/sqrt(alpha) * (x_t - beta/sqrt(1 - alpha_bar) * predicted_noise)
        pred_prev_sample = (1.0 / torch.sqrt(alpha)) * (sample - (beta / sqrt_one_minus_alpha_bar) * model_output)
        
        # Add noise if t > 0
        if t > 0:
            noise = torch.randn_like(sample)
            # Use the posterior variance
            variance = self.posterior_variance[t].view(-1, 1, 1, 1)
            pred_prev_sample = pred_prev_sample + torch.sqrt(variance) * noise
            
        return pred_prev_sample
