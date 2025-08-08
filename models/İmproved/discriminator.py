import torch
import torch.nn as nn

class ProjectionDiscriminator(nn.Module):
    def __init__(self, channels_img, num_classes, features_d):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, features_d * 8)

        def spectral_conv(in_c, out_c, k, s, p):
            return nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, k, s, p))

        self.conv_blocks = nn.Sequential(
            spectral_conv(channels_img, features_d, 4, 2, 1),  # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),

            spectral_conv(features_d, features_d * 2, 4, 2, 1),  # 32 -> 16
            nn.LeakyReLU(0.2, inplace=True),

            spectral_conv(features_d * 2, features_d * 4, 4, 2, 1),  # 16 -> 8
            nn.LeakyReLU(0.2, inplace=True),

            spectral_conv(features_d * 4, features_d * 8, 4, 2, 1),  # 8 -> 4
            nn.LeakyReLU(0.2, inplace=True),

            spectral_conv(features_d * 8, features_d * 8, 4, 1, 0),  # 4 -> 1
        )

        self.linear = spectral_conv(features_d * 8, 1, 1, 1, 0)

    def forward(self, x, labels):
        features = self.conv_blocks(x)  # N x (fd*8) x 1 x 1
        features = features.view(x.size(0), -1)  # N x (fd*8)
        out = self.linear(features.unsqueeze(-1).unsqueeze(-1)).view(-1)  # N

        # Projection trick
        label_embedding = self.label_emb(labels)  # N x (fd*8)
        proj = torch.sum(features * label_embedding, dim=1)

        return out + proj
