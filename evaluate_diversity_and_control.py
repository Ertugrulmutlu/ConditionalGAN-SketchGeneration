import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from model.generator import ImprovedConditionalGenerator

# === Settings ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Z_DIM = 512
CHANNELS_IMG = 1
FEATURES_GEN = 256
NUM_CLASSES = 5
SAVE_DIR = "improved_outputs/eval_multi_epoch"
EPOCHS = [9,10]
LABEL_CLASS = 3 # Test edilecek sınıf

os.makedirs(SAVE_DIR, exist_ok=True)

for epoch in EPOCHS:
    print(f"\n🔍 Testing epoch {epoch}...")

    MODEL_PATH = f"improved_outputs/checkpoints/epoch_{epoch}_gen.pth"
    gen = ImprovedConditionalGenerator(Z_DIM, NUM_CLASSES, CHANNELS_IMG, FEATURES_GEN).to(DEVICE)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    gen.eval()

    # === Test 1: Same class, different noise ===
    n_variations = 8
    noise = torch.randn(n_variations, Z_DIM, 1, 1).to(DEVICE)
    labels = torch.full((n_variations,), LABEL_CLASS, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        fake_images = gen(noise, labels).cpu()
        grid = vutils.make_grid(fake_images, nrow=n_variations, normalize=True, padding=2)
        np_grid = grid.permute(1, 2, 0).numpy()

        plt.figure(figsize=(n_variations, 2))
        plt.imshow(np_grid.squeeze(), cmap='gray')
        plt.axis("off")
        plt.title(f"Epoch {epoch} - Class {LABEL_CLASS} - Different Noise")
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"epoch_{epoch}_class_{LABEL_CLASS}_noise_variation.png"), dpi=300)
        plt.close()

    # === Test 2: Same noise, different classes ===
    noise = torch.randn(1, Z_DIM, 1, 1).repeat(NUM_CLASSES, 1, 1, 1).to(DEVICE)
    labels = torch.arange(NUM_CLASSES).to(DEVICE)

    with torch.no_grad():
        fake_images = gen(noise, labels).cpu()
        nrow = int(np.ceil(np.sqrt(NUM_CLASSES)))
        grid = vutils.make_grid(fake_images, nrow=nrow, normalize=True, padding=2)
        np_grid = grid.permute(1, 2, 0).numpy()

        plt.figure(figsize=(nrow * 1.5, nrow * 1.5))
        plt.imshow(np_grid.squeeze(), cmap='gray')
        plt.axis("off")
        plt.title(f"Epoch {epoch} - Same Noise - All Classes")
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"epoch_{epoch}_same_noise_all_classes.png"), dpi=300)
        plt.close()

    print(f"✅ Saved visual tests for epoch {epoch}")
