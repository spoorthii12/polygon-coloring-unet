Here is a concise and well-structured report you can include as a README.md file in your Hugging Face Space. It summarizes all key components of your polygon coloring U-Net project:

---

# 🧠 Polygon Coloring with UNet

This project implements a U-Net architecture conditioned on color to generate realistically colored images from grayscale polygon shapes. It allows interactive colorization through Hugging Face Spaces.

---

## 📌 Overview

* 🔍 Task: Given a grayscale polygon image and a color label, generate a fully colored version of the image.
* 🧠 Model: U-Net with color-conditioning.
* 📁 Frameworks: PyTorch, Gradio, Hugging Face Spaces.

---

## ⚙️ Hyperparameters

| Parameter     | Value         | Rationale                                            |
| ------------- | ------------- | ---------------------------------------------------- |
| IMG\_SIZE     | 128×128       | Small size for quick training/inference              |
| BATCH\_SIZE   | 16            | Fits GPU memory while ensuring good batch statistics |
| NUM\_EPOCHS   | 20            | Balanced convergence without overfitting             |
| LR            | 1e-3          | Adam’s default; stable for MSE loss                  |
| LOSS FUNCTION | MSELoss       | Smooth pixel-wise regression loss                    |
| COLOR\_LIST   | 8 base colors | Supports discrete color conditioning                 |

---

## 🏗️ Architecture

* Based on a classic U-Net design with skip connections.
* Color conditioning is implemented by:

  * One-hot encoding the color label.
  * Expanding and concatenating it with the input image on the channel dimension.
* Layers:

  * Downsampling: 3 levels with DoubleConv → MaxPool.
  * Upsampling: ConvTranspose2d + interpolation if needed + DoubleConv.
  * Output: Final Conv2d with sigmoid to constrain output range \[0,1].

🎯 Total Parameters: 7.7M

---

## 📈 Training Dynamics

Logged using Weights & Biases (wandb):

| Metric | Observations                                  |
| ------ | --------------------------------------------- |
| Loss   | Decreased steadily, plateaued after epoch 15. |
| MAE    | Dropped from \~0.47 to \~0.13                 |
| PSNR   | Improved from 6.4 → 15.5                      |
| SSIM   | Reached up to 0.89                            |

📉 Failure Modes:

* Over-saturation on rare shapes or non-dominant colors.
* Inconsistent edges when shape resolution is poor.

🔧 Fixes Tried:

* Interpolation to match skip sizes (to avoid shape mismatch in upsampling).
* SSIM-based checkpointing (best visually consistent output).
* Normalization using sigmoid output and clipping.

---

## 🎨 Inference Flow

1. Upload grayscale polygon image.
2. Select a target color.
3. Model loads from best\_model.pth.
4. Displays original, processed input, and generated output side-by-side.

---

## 📚 Key Learnings

* Explicit color-conditioning is highly effective when concatenated with the image tensor early in the network.
* SSIM is a better indicator of visual quality than MSE alone.
* Small resolution inputs are sufficient for colorizing geometric patterns.
* Hugging Face Spaces + Gradio enables seamless deployment of vision models.

---

## 🧪 Try It Yourself

Visit the Space:
👉 [https://huggingface.co/spaces/spoorthiii/polygon-coloring-unet](https://huggingface.co/spaces/spoorthiii/polygon-coloring-unet)

---

Would you like me to generate and upload this as a markdown file for download or commit it directly to your Hugging Face space as a README.md?
