#🧠 ConditionalGAN-SketchGeneration

Welcome to this Conditional GAN (cGAN) project! This repo contains all the code, models, and training scripts for generating black-and-white hand-drawn sketches conditioned on class labels using the Berlin Sketches dataset.

📖 For the full story, insights, and what I learned from training these models, check out the full blog post:
👉 [Read the Blog](https://dev.to/ertugrulmutlu/a-cgan-story-three-attempts-and-an-incomplete-ending-2a1b)

---

##  Project Context

For consistency in the sketch domain, I also referenced the **ClassifySketch** project from TU Berlin by Eitz et al.—a seminal study in sketch recognition using 250 object categories and human-measured accuracy benchmarks.  
This provided inspiring background on sketch classification challenges. Explore the original research here:  
[ClassifySketch – TU Berlin (Eitz et al.)](/cybertron.cg.tu-berlin.de/eitz/projects/classifysketch)

---
## 📁 Project Structure

```
├── models/
│   ├── generator.py                # Phase 1–2 classic generator
│   ├── discriminator.py            # Phase 1–2 classic discriminator
│   └── improved/
│       ├── improved_generator.py   # Phase 3 upgraded generator
│       └── projection_discriminator.py  # Phase 3 upgraded discriminator
├── trains/
│   ├── train.py                # Phase 1–2 classic generator
│   ├── train_2.py            # Phase 1–2 classic discriminator
│   └── improved/
│       ├── train.py    # Phase 3 upgraded generator
│       └── train_2.pyy  # Phase 3 upgraded discriminator
├── data_augment.py                 # Data augmentation routines
├── evaluate_diversity_and_control.py  # Evaluation script
```

---

## 🧱 Models

### 🔹 Phase 1–2 Classic Model

* `models/generator.py`
* `models/discriminator.py`

Simple concatenation of label and noise/image tensors. Embedding size is `num_classes`.

### 🔹 Phase 3 Improved Model

* `models/improved/improved_generator.py`

* `models/improved/projection_discriminator.py`

* Larger `z_dim` (512)

* Projection-based conditioning in discriminator

* Label embedding size = `z_dim * 2`

---

## 🏋️ Training Scripts

### 🔸 `train.py`

* Used in **Phase 1** (classic model) and **Phase 3** (improved model)
* Change model import at the top accordingly

### 🔸 `train_2.py`

* Used in **Phase 2** (resumed classic model training with better loop)
* Also reused for **Phase 3** with improvements (scheduler, smoothing)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Ertugrulmutlu/ConditionalGAN-SketchGeneration.git
cd ConditionalGAN-SketchGeneration
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

Organize the Berlin Sketches dataset using the following structure:

```
data/
├── 0/
│   ├── img1.png
│   └── img2.png
├── 1/
│   ├── img1.png
│   └── img2.png
...
```

### 4. Train the Model

Run a training script:

```bash
python train.py        # Classic or Phase 3 model
python train_2.py      # Improved training loop (Phase 2 or 3)
```

---

## 🧪 Evaluation & Debugging

### Run Controlled Tests

```bash
python evaluate_diversity_and_control.py
```

This checks:

* Same class + different noise → output diversity
* Same noise + different labels → conditional accuracy

---

## 📌 Notes

* If using **Phase 3** model, make sure your GPU has at least 8GB of memory
* Loss plots and sample images are saved automatically

---

Made with ❤️ by [@ErtugrulMutlu](https://github.com/Ertugrulmutlu)
