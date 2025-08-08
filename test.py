import torch
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from torchvision.utils import save_image
import os
import numpy as np

from model.generator import ConditionalGenerator  # Adjust path if necessary

# ==== Settings ====
Z_DIM = 100
IMG_SIZE = 64
CHANNELS_IMG = 1
FEATURES_GEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Load classes from folder structure ====
DATA_ROOT = "png"
class_names = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
num_classes = len(class_names)
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

# ==== Load Trained Generator ====
gen = ConditionalGenerator(Z_DIM, num_classes, CHANNELS_IMG, FEATURES_GEN).to(DEVICE)
gen.load_state_dict(torch.load("cgan_outputs\checkpoints\epoch_49_generator.pth", map_location=DEVICE))
gen.eval()

# ==== GUI Setup ====
root = tk.Tk()
root.title("cGAN Sketch Generator")
root.geometry("300x400")

label = tk.Label(root, text="Select a Class:")
label.pack(pady=10)

# Dropdown to choose class
selected_class = tk.StringVar()
selected_class.set(class_names[0])
class_menu = ttk.Combobox(root, textvariable=selected_class, values=class_names, state="readonly")
class_menu.pack(pady=5)

# Image placeholder
image_label = tk.Label(root)
image_label.pack(pady=20)

# ==== Generate and Display Function ====
def generate_image():
    class_name = selected_class.get()
    class_idx = class_to_idx[class_name]

    noise = torch.randn(1, Z_DIM, 1, 1).to(DEVICE)
    label_tensor = torch.tensor([class_idx], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        fake = gen(noise, label_tensor)
    fake = fake.squeeze().cpu()

    save_image(fake, "preview.png", normalize=True)
    img = Image.open("preview.png").resize((128, 128))
    img_tk = ImageTk.PhotoImage(img)
    image_label.configure(image=img_tk)
    image_label.image = img_tk

# Button to trigger generation
generate_btn = tk.Button(root, text="Generate", command=generate_image)
generate_btn.pack(pady=10)

# Start the GUI loop
root.mainloop()
