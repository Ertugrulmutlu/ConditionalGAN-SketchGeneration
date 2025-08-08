# === Improved Training Script ===
import os
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
from model.generator import ImprovedConditionalGenerator
from model.discriminator import ProjectionDiscriminator

# === Settings ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 512  # (1) Z dimension increased
FEATURES_GEN = 256  # (4) Feature maps increased
FEATURES_DISC = 256
BATCH_SIZE = 128
EPOCHS = 100
RESUME_EPOCH = -1
if __name__ == "__main__":
    DATA_DIR = "augmented_png"
    SAVE_DIR = "improved_outputs"
    SAMPLES_DIR = os.path.join(SAVE_DIR, "samples")
    CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")
    LOSS_DIR = os.path.join(SAVE_DIR, "loss")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOSS_DIR, exist_ok=True)

    # === Data ===
    dataset = datasets.ImageFolder(
        root=DATA_DIR,
        transform = transforms.Compose([
            transforms.Grayscale(),                         # Tek kanal
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),

            # === Augmentations ===
            transforms.RandomHorizontalFlip(p=0.5),         # Yatay çevirme
            transforms.RandomRotation(degrees=15),          # ±15 derece döndürme
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1)),            # Konum kaydırma (10%)

            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),

            transforms.RandomErasing(p=0.2, scale=(0.02, 0.05), ratio=(0.3, 3.3), value=0)
        ])
        
    )

    num_classes = len(dataset.classes)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # === Models ===
    gen = ImprovedConditionalGenerator(Z_DIM, num_classes, CHANNELS_IMG, FEATURES_GEN).to(DEVICE)
    disc = ProjectionDiscriminator(CHANNELS_IMG, num_classes, FEATURES_DISC).to(DEVICE)

    opt_gen = optim.Adam(gen.parameters(), lr=6e-4, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
    scheduler_gen = optim.lr_scheduler.StepLR(opt_gen, step_size=30, gamma=0.5)
    scheduler_disc = optim.lr_scheduler.StepLR(opt_disc, step_size=30, gamma=0.5)

    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 0
    G_losses, D_losses = [], []
    if RESUME_EPOCH >= 0:
        gen.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"epoch_{RESUME_EPOCH}_gen.pth")))
        disc.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"epoch_{RESUME_EPOCH}_disc.pth")))
        start_epoch = RESUME_EPOCH + 1
        print(f"Resumed from epoch {RESUME_EPOCH}")

    nrow = int(np.ceil(np.sqrt(num_classes)))
    total_samples = nrow * nrow
    fixed_noise = torch.randn(total_samples, Z_DIM, 1, 1, device=DEVICE)
    fixed_labels = torch.arange(num_classes, device=DEVICE)
    fixed_labels = torch.cat([fixed_labels, fixed_labels[:total_samples - num_classes]])

    # === Training Loop ===
    for epoch in range(start_epoch, EPOCHS):
        for batch_idx, (real, labels) in enumerate(dataloader):
            real, labels = real.to(DEVICE), labels.to(DEVICE)
            batch_size = real.size(0)

            ### Train Discriminator ###
            noise = torch.randn(batch_size, Z_DIM, 1, 1, device=DEVICE)
            fake_labels = torch.randint(0, num_classes, (batch_size,), device=DEVICE)
            fake = gen(noise, fake_labels)

            # (5) Add noise to real images
            real += 0.05 * torch.randn_like(real)

            real_preds = disc(real, labels)
            fake_preds = disc(fake.detach(), fake_labels)

            # (3) Label smoothing
            real_targets = torch.ones_like(real_preds) * 0.9
            fake_targets = torch.zeros_like(fake_preds)

            loss_d_real = criterion(real_preds, real_targets)
            loss_d_fake = criterion(fake_preds, fake_targets)
            loss_d = (loss_d_real + loss_d_fake) / 2

            disc.zero_grad()
            loss_d.backward()
            opt_disc.step()

            ### Train Generator ###
            output = disc(fake, fake_labels)
            loss_g = criterion(output, torch.ones_like(output))

            gen.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
            opt_gen.step()

            G_losses.append(loss_g.item())
            D_losses.append(loss_d.item())

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(dataloader)} | D Loss: {loss_d:.4f} | G Loss: {loss_g:.4f}")

        # === Save sample images ===
        with torch.no_grad():
            fake_imgs = gen(fixed_noise, fixed_labels)
            grid = utils.make_grid(fake_imgs, nrow=nrow, normalize=True, padding=2)
            plt.figure(figsize=(nrow * 2, nrow * 2))
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(SAMPLES_DIR, f"epoch_{epoch}_samples.png"), dpi=300)
            plt.close()

        # === Save model checkpoints ===
        torch.save(gen.state_dict(), os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}_gen.pth"))
        torch.save(disc.state_dict(), os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}_disc.pth"))

        # Step the scheduler
        scheduler_gen.step()
        scheduler_disc.step()

    # === Save loss plot ===
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_losses, label='Discriminator Loss')
    plt.legend()
    plt.title("Training Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(LOSS_DIR, "loss_curve.png"))
    plt.show()