import torch
import torch.nn as nn

class ConditionalDiscriminator(nn.Module):
    def __init__(self, img_channels, num_classes, feature_d):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.net = nn.Sequential(
            nn.Conv2d(img_channels + num_classes, feature_d, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d * 2, 1, 4, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, images, labels):
        label_embedding = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        label_embedding = label_embedding.expand(-1, -1, images.size(2), images.size(3))
        x = torch.cat([images, label_embedding], dim=1)
        return self.net(x)
