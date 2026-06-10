import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)
        
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_mlp = None

    def forward(self, x, time_emb=None):
        h = self.conv1(x)
        h = self.gn1(h)
        h = self.silu(h)
        
        if self.time_mlp is not None and time_emb is not None:
            # Project time embedding and add to spatial dimensions
            time_projected = self.time_mlp(time_emb)
            # Add dimensions for broadcasting to [batch_size, channels, height, width]
            h = h + time_projected[:, :, None, None]
            
        h = self.conv2(h)
        h = self.gn2(h)
        h = self.silu(h)
        return h

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # Encoder (Downsampling path)
        self.inc = DoubleConvBlock(in_channels, 64, time_emb_dim)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(64, 128, time_emb_dim)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(128, 256, time_emb_dim)
        )
        
        # Bottleneck
        self.bottleneck = DoubleConvBlock(256, 256, time_emb_dim)
        
        # Decoder (Upsampling path)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up1 = DoubleConvBlock(256 + 128, 128, time_emb_dim)
        
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up2 = DoubleConvBlock(128 + 64, 64, time_emb_dim)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t):
        # 1. Project timestep to embedding
        t_emb = self.time_mlp(t)
        
        # 2. Encoder
        x1 = self.inc(x, t_emb)          # 64 channels, 28x28 (if input is 28x28)
        x2 = self.down1[1](self.down1[0](x1), t_emb) # 128 channels, 14x14
        x3 = self.down2[1](self.down2[0](x2), t_emb) # 256 channels, 7x7
        
        # 3. Bottleneck
        x3 = self.bottleneck(x3, t_emb)  # 256 channels, 7x7
        
        # 4. Decoder with skip connections
        # For x3 (7x7), upsample -> 14x14
        up_x3 = self.up1(x3)
        # Pad up_x3 if necessary to match x2 size (e.g., if input wasn't multiple of 4)
        if up_x3.shape[-2:] != x2.shape[-2:]:
            diffY = x2.size()[2] - up_x3.size()[2]
            diffX = x2.size()[3] - up_x3.size()[3]
            up_x3 = nn.functional.pad(up_x3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
        x_up = torch.cat([up_x3, x2], dim=1)
        x_up = self.conv_up1(x_up, t_emb) # 128 channels
        
        # Upsample -> 28x28
        up_x_up = self.up2(x_up)
        if up_x_up.shape[-2:] != x1.shape[-2:]:
            diffY = x1.size()[2] - up_x_up.size()[2]
            diffX = x1.size()[3] - up_x_up.size()[3]
            up_x_up = nn.functional.pad(up_x_up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
        x_up2 = torch.cat([up_x_up, x1], dim=1)
        x_up2 = self.conv_up2(x_up2, t_emb) # 64 channels
        
        # 5. Output projection
        logits = self.outc(x_up2)
        return logits
