import torch
import torch.nn as nn

class ImprovedConditionalGenerator(nn.Module):
    def __init__(self, z_dim, num_classes, channels_img, features_g):
        super().__init__()
        self.z_dim = z_dim
        self.label_emb = nn.Embedding(num_classes, z_dim)

        self.net = nn.Sequential(
            # Input: N x (z_dim + num_classes) x 1 x 1
            nn.ConvTranspose2d(z_dim * 2, features_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g, channels_img, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        label_embed = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        label_embed = label_embed.expand(-1, -1, noise.size(2), noise.size(3))
        x = torch.cat([noise, label_embed], dim=1)
        return self.net(x)
    
    