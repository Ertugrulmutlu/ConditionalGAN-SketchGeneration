import os
from PIL import Image
from torchvision import transforms
import random

# === Ayarlar ===
INPUT_DIR = "png"            # Orijinal veri klasörü
OUTPUT_DIR = "augmented_png" # Yeni augment edilmiş veri klasörü
AUGS_PER_IMAGE = 4           # Her görselden kaç tane üretilecek

# === Augmentasyonlar (random olarak uygulanacak) ===
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
])

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# === Tüm class klasörlerini gez ===
for class_name in os.listdir(INPUT_DIR):
    input_class_dir = os.path.join(INPUT_DIR, class_name)
    output_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    for filename in os.listdir(input_class_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        path = os.path.join(input_class_dir, filename)
        image = Image.open(path).convert("L")  # Grayscale'e çevir

        # Orijinal görseli kopyala
        image.save(os.path.join(output_class_dir, filename))

        # Augmentasyonlu kopyaları üret
        for i in range(AUGS_PER_IMAGE):
            aug_image = augmentation(image)
            aug_filename = filename.replace(".", f"_aug{i+1}.")
            aug_path = os.path.join(output_class_dir, aug_filename)
            aug_image.save(aug_path)

print(f"✅ Data augmentation complete. Output saved to: {OUTPUT_DIR}")