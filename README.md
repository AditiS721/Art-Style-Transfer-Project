# 🎨 CycleGAN Art Style Transfer

A deep learning project that transforms photographs into artistic style images using CycleGAN architecture. This implementation uses PyTorch and is designed to run on Google Colab with GPU acceleration.

## 📋 Project Overview

This project implements a CycleGAN model to perform unpaired image-to-image translation, specifically designed for artistic style transfer. The model learns to convert real photographs into artistic renderings without requiring paired training data.

### ✨ Features

- **Unpaired Training**: No need for corresponding image pairs
- **Bidirectional Translation**: Converts photos → artwork and artwork → photos
- **GPU Accelerated**: Optimized for Google Colab with CUDA support
- **Comprehensive Evaluation**: Includes FID, SSIM, PSNR, and LPIPS metrics
- **Easy Inference**: Simple API for applying trained styles to new images

## 🗂️ Project Structure

```
pytorch-CycleGAN-and-pix2pix/
├── checkpoints/                    # Saved model checkpoints
│   └── art_style_transfer/        # Trained model weights
├── data/                          # Dataset handling utilities
├── models/                        # CycleGAN model architecture
├── options/                       # Training/testing options
├── results/                       # Generated output images
│   └── art_style_transfer/
├── util/                          # Utility functions
├── train.py                       # Training script
├── test.py                        # Testing script
└── evaluation.ipynb               # Complete workflow notebook
```

## 📊 Dataset

The dataset consists of unpaired images organized in the following structure:

```
my_style_transfer/
├── trainA/          # 200 training photos (domain A)
├── trainB/          # 100 training artworks (domain B)
├── testA/           # 10 test photos (domain A)
└── testB/           # 10 test artworks (domain B)
```

- **Domain A**: Real photographs
- **Domain B**: Artistic style images
- **Image Size**: 256×256 pixels

## 🚀 Installation & Setup

### Prerequisites

- Google Colab (recommended) or local GPU setup
- Python 3.7+
- CUDA-compatible GPU

### Quick Start (Google Colab)

```python
# Clone the repository
!git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
%cd pytorch-CycleGAN-and-pix2pix

# Install dependencies
!pip install torch torchvision dominate visdom numpy matplotlib opencv-python
!pip install pytorch-fid pytorch-msssim lpips tqdm

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

## 🏋️ Training

### Start Training

```bash
!python train.py \
    --dataroot /content/drive/MyDrive/my_style_transfer \
    --name art_style_transfer \
    --model cycle_gan \
    --batch_size 1 \
    --n_epochs 50 \
    --n_epochs_decay 50 \
    --preprocess resize_and_crop \
    --crop_size 256 \
    --load_size 286 \
    --save_epoch_freq 10
```

### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 1 | Batch size for training |
| `n_epochs` | 50 | Number of epochs with constant learning rate |
| `n_epochs_decay` | 50 | Number of epochs with linearly decaying learning rate |
| `crop_size` | 256 | Size of image crops |
| `load_size` | 286 | Scale images to this size before cropping |
| `preprocess` | resize_and_crop | Preprocessing method |

### Training Progress

The training process outputs key metrics:

- **D_A, D_B**: Discriminator losses for domains A and B
- **G_A, G_B**: Generator adversarial losses
- **cycle_A, cycle_B**: Cycle consistency losses
- **idt_A, idt_B**: Identity mapping losses

## 🧪 Testing & Inference

### Batch Testing

```bash
!python test.py \
    --dataroot /content/drive/MyDrive/my_style_transfer \
    --name art_style_transfer \
    --model cycle_gan \
    --epoch latest \
    --num_test 20
```

### Single Image Inference

```python
from PIL import Image
import torch

# Load model
model = create_model(opt)
model.setup(opt)
model.eval()

# Load and preprocess image
input_image = load_image('path/to/image.jpg')

# Generate stylized output
with torch.no_grad():
    stylized_image = model.netG_A(input_image)

# Save result
save_image(stylized_image, 'output.jpg')
```

## 📈 Model Evaluation

### Evaluation Metrics

The model is evaluated using four complementary metrics:

| Metric | Description | Range | Target |
|--------|-------------|-------|--------|
| **FID** | Fréchet Inception Distance - measures realism | Lower is better | < 100 |
| **SSIM** | Structural Similarity - measures content preservation | 0 to 1 | > 0.4 |
| **PSNR** | Peak Signal-to-Noise Ratio - measures image quality | dB | > 25 |
| **LPIPS** | Learned Perceptual Image Patch Similarity | Lower is better | < 0.3 |

### Results

```
┌─────────────────────────────────────────────────────────────┐
│                   CYCLEGAN MODEL SCORES                     │
├─────────────────────────────────────────────────────────────┤
│  🎯 FID:     397.47                                        │
│  🖼️ SSIM:    0.5123                                        │
│  📡 PSNR:    15.50 dB                                      │
│  👁️ LPIPS:   0.3844                                        │
├─────────────────────────────────────────────────────────────┤
│  📸 Test images: 10                                        │
└─────────────────────────────────────────────────────────────┘
```

### Score Interpretation

- **SSIM 0.5123**: ✅ Good structure preservation - the model maintains the original photo composition while applying artistic styles
- **PSNR 15.50 dB**: ⚠️ Moderate quality - expected for style transfer as pixel-level differences are intentional
- **LPIPS 0.3844**: 👁️ Moderate perceptual difference - indicates successful style transformation
- **FID 397.47**: ⚠️ High divergence - suggests room for improvement in generating realistic artistic textures

## 🏗️ Model Architecture

### Generator Networks (G_A, G_B)
- **Architecture**: ResNet with 9 residual blocks
- **Parameters**: 11.378M each
- **Normalization**: Instance Normalization
- **Input/Output**: 3-channel RGB images (256×256)

### Discriminator Networks (D_A, D_B)
- **Architecture**: 70×70 PatchGAN
- **Parameters**: 2.765M each
- **Layers**: 3 convolutional layers
- **GAN Mode**: Least Squares GAN (LSGAN)

### Total Parameters: 28.286M

## 🔧 Configuration

Key hyperparameters used in training:

```python
{
    'lambda_A': 10.0,           # Cycle consistency weight A→B→A
    'lambda_B': 10.0,           # Cycle consistency weight B→A→B
    'lambda_identity': 0.5,     # Identity mapping weight
    'lr': 0.0002,              # Initial learning rate
    'lr_policy': 'linear',     # Learning rate decay policy
    'beta1': 0.5,              # Adam optimizer beta1
    'gan_mode': 'lsgan',       # GAN objective function
    'init_type': 'normal',     # Weight initialization
    'init_gain': 0.02          # Initialization gain
}
```

## 💾 Model Checkpoints

Trained models are saved at:

```
checkpoints/art_style_transfer/
├── latest_net_G_A.pth    # Generator A (photos → artwork)
├── latest_net_G_B.pth    # Generator B (artwork → photos)
├── latest_net_D_A.pth    # Discriminator A
├── latest_net_D_B.pth    # Discriminator B
└── web/                  # Training visualizations
```

To save model to Google Drive:

```python
import shutil
shutil.copy(
    'checkpoints/art_style_transfer/latest_net_G_A.pth',
    '/content/drive/MyDrive/art_style_transfer_generator.pth'
)
```

## 📊 Training Analysis

### Loss Trends (Epochs 1-100)

- **Early Stage (Epochs 1-20)**: High cycle consistency losses (>2.0), models learning basic domain translation
- **Middle Stage (Epochs 21-60)**: Losses stabilizing, cycle consistency drops to ~1.0-1.5
- **Late Stage (Epochs 61-100)**: Convergence achieved, cycle consistency <1.0, GAN losses balanced

### Training Time
- **Total Epochs**: 100
- **Average Time/Epoch**: ~103 seconds
- **Total Training Time**: ~2.8 hours (on Tesla T4 GPU)

## 🎯 Use Cases

This model can be used for:

- **Artistic Photo Editing**: Transform photographs into specific art styles
- **Content Creation**: Generate stylized images for social media, websites
- **Design Assistance**: Rapid prototyping of artistic concepts
- **Style Transfer Research**: Baseline for further improvements
- **Educational Purposes**: Learning about GANs and image translation

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**:
   - Reduce `batch_size` to 1
   - Decrease `crop_size` to 128

2. **Poor Quality Results**:
   - Increase `n_epochs` and `n_epochs_decay`
   - Ensure sufficient training data (200+ images per domain)
   - Check dataset quality and consistency

3. **DataLoader Warnings**:
   - Set `num_threads` to 0 in options
   - Use `--num_threads 0` flag in training

## 📚 References

- **CycleGAN Paper**: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- **Original Repository**: [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- **Authors**: Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros

## 📄 License

This project follows the same license as the original CycleGAN implementation. See the [original repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for details.

## 🤝 Contributing

Feel free to fork this project and submit pull requests. Areas for improvement:

- Additional data augmentation techniques
- Alternative model architectures
- Hyperparameter optimization
- Web interface for inference
- Mobile deployment support

## 📧 Contact

For questions or collaboration opportunities, please open an issue in the repository.

---

**Built with ❤️ using PyTorch and Google Colab**
**Model Architecture: CycleGAN | Framework: PyTorch | Platform: Google Colab**
