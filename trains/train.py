import os
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
from model.generator import ConditionalGenerator
from model.discriminator import ConditionalDiscriminator

# === Settings ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
FEATURES_GEN = 64
FEATURES_DISC = 64
BATCH_SIZE = 128
LR = 5e-5
EPOCHS = 50
DATA_DIR = "png"
SAVE_DIR = "cgan_outputs"
SAMPLES_DIR = os.path.join(SAVE_DIR, "samples")
CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")
LOSS_DIR = os.path.join(SAVE_DIR, "loss")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOSS_DIR, exist_ok=True)

LOAD_MODEL = True
MODEL_PATH_GEN = os.path.join(CHECKPOINT_DIR, "latest_generator.pth")
MODEL_PATH_DISC = os.path.join(CHECKPOINT_DIR, "latest_discriminator.pth")

# === Transforms ===
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Dataset and class count detection ===
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
num_classes = len(dataset.classes)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Models ===
gen = ConditionalGenerator(Z_DIM, num_classes, CHANNELS_IMG, FEATURES_GEN).to(DEVICE)
disc = ConditionalDiscriminator(CHANNELS_IMG, num_classes, FEATURES_DISC).to(DEVICE)

# === Load models if requested ===
start_epoch = 0
if LOAD_MODEL and os.path.exists(MODEL_PATH_GEN) and os.path.exists(MODEL_PATH_DISC):
    gen.load_state_dict(torch.load(MODEL_PATH_GEN))
    disc.load_state_dict(torch.load(MODEL_PATH_DISC))
    print(">> Modeller yüklendi, eğitim kaldığı yerden devam ediyor.")
else:
    print(">> Yeni eğitim başlatılıyor.")

# === Optimizers and loss ===
criterion = nn.BCELoss()
opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))

# === Fixed noise for consistent output ===
fixed_noise = torch.randn(num_classes, Z_DIM, 1, 1).to(DEVICE)
fixed_labels = torch.arange(num_classes).to(DEVICE)

# === Loss logs ===
losses_g, losses_d = [], []

# === Training Loop ===
print("Start training...")
for epoch in range(start_epoch, EPOCHS):
    for batch_idx, (real, labels) in enumerate(dataloader):
        real, labels = real.to(DEVICE), labels.to(DEVICE)
        batch_size = real.size(0)

        # === Train Discriminator ===
        noise = torch.randn(batch_size, Z_DIM, 1, 1).to(DEVICE) * 1.2
        fake = gen(noise, labels)

        real_preds = disc(real, labels).view(-1)
        fake_preds = disc(fake.detach(), labels).view(-1)

        loss_d_real = criterion(real_preds, torch.ones_like(real_preds))
        loss_d_fake = criterion(fake_preds, torch.zeros_like(fake_preds))
        loss_d = (loss_d_real + loss_d_fake) / 2

        disc.zero_grad()
        loss_d.backward()
        opt_disc.step()

        # === Train Generator ===
        output = disc(fake, labels).view(-1)
        loss_g = criterion(output, torch.ones_like(output))

        gen.zero_grad()
        loss_g.backward()
        opt_gen.step()

        losses_g.append(loss_g.item())
        losses_d.append(loss_d.item())

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss D: {loss_d:.4f}, Loss G: {loss_g:.4f}")

    # === Save sample grid without class names ===
    with torch.no_grad():
        fake_samples = gen(fixed_noise, fixed_labels).detach().cpu()
        nrow = int(np.ceil(np.sqrt(num_classes)))

        grid = utils.make_grid(fake_samples, nrow=nrow, normalize=True, padding=2)
        np_grid = grid.permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(figsize=(nrow * 2, nrow * 2))
        ax.imshow(np_grid.squeeze(), cmap='gray')
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(SAMPLES_DIR, f"epoch_{epoch}_samples.png"), bbox_inches='tight', dpi=300)
        plt.close()

    # === Save models ===
    torch.save(gen.state_dict(), os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}_generator.pth"))
    torch.save(disc.state_dict(), os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}_discriminator.pth"))
    torch.save(gen.state_dict(), MODEL_PATH_GEN)  # overwrite latest
    torch.save(disc.state_dict(), MODEL_PATH_DISC)

# === Plot Loss ===
plt.figure(figsize=(10, 5))
plt.plot(losses_g, label="Generator Loss")
plt.plot(losses_d, label="Discriminator Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(LOSS_DIR, "loss_curve.png"))
plt.show()