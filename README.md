# Multi-Modal Medical Image Segmentation

A modular framework for multi-organ segmentation using multiple imaging modalities (CT, PET, MRI, Ultrasound) with support for various fusion strategies and explainability methods.

## Features

- **Multi-Modal Support**: CT, PET, MRI, Ultrasound
- **Flexible Fusion Strategies**: Early, Late, Attention, Cross-Attention
- **Multiple Backbones**: SwinUNETR, UNet3D, Dual Encoder
- **Explainability**: GradCAM, Attention Maps, t-SNE, SHAP
- **Analysis Tools**: SUV analysis, TMTV calculation, Histogram generation
- **End-to-End Pipeline**: Preprocessing, Training, Inference, Analysis

## Project Structure

```
multi_organ_segmentations/
├── main.py                     # Unified entry point
├── configs/
│   └── default.yaml            # Default configuration
├── src/
│   ├── data/                   # Dataset and transforms
│   │   ├── dataset.py          # Multi-modal dataset
│   │   ├── transforms.py       # Augmentation & preprocessing
│   │   └── dataloader.py       # DataLoader utilities
│   ├── preprocessing/          # Data preparation
│   │   ├── dicom_converter.py  # DICOM to NIfTI
│   │   ├── suv_calculator.py   # SUV calculation
│   │   ├── registration.py     # Image registration
│   │   └── normalizer.py       # Intensity normalization
│   ├── models/
│   │   ├── backbones/          # Encoder architectures
│   │   │   ├── swin_unetr.py   # Swin UNETR
│   │   │   ├── unet.py         # 3D UNet
│   │   │   └── dual_encoder.py # Dual encoder for multi-modal
│   │   ├── fusion/             # Fusion strategies
│   │   │   ├── early_fusion.py
│   │   │   ├── late_fusion.py
│   │   │   └── attention_fusion.py
│   │   ├── heads/              # Task-specific heads
│   │   │   ├── segmentation.py
│   │   │   └── detection.py
│   │   └── build.py            # Model factory
│   ├── trainer/                # Training pipeline
│   │   ├── trainer.py          # Trainer class
│   │   ├── losses.py           # Loss functions
│   │   └── metrics.py          # Evaluation metrics
│   ├── analysis/               # Post-processing analysis
│   │   ├── suv.py              # SUV analysis
│   │   ├── tmtv.py             # TMTV calculation
│   │   ├── histogram.py        # Histogram generation
│   │   └── report.py           # Report generation
│   ├── explainability/         # Model interpretation
│   │   ├── gradcam.py          # GradCAM, GradCAM++
│   │   ├── attention.py        # Attention visualization
│   │   ├── tsne.py             # t-SNE visualization
│   │   └── shap_analysis.py    # SHAP analysis
│   └── utils/                  # Utilities
│       ├── logger.py           # Logging
│       ├── io.py               # I/O utilities
│       ├── seed.py             # Reproducibility
│       └── visualization.py    # Visualization tools
└── legacy/                     # Old code for reference
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train with default config
python main.py --mode train --config configs/default.yaml

# Train with custom settings
python main.py --mode train \
    --config configs/default.yaml \
    --exp-name my_experiment \
    --epochs 100 \
    --batch-size 2 \
    --model dual_encoder \
    --fusion cross_attention
```

### Evaluation

```bash
python main.py --mode eval \
    --config configs/default.yaml \
    --checkpoint outputs/my_experiment/best.pth
```

### Inference

```bash
python main.py --mode inference \
    --config configs/default.yaml \
    --checkpoint outputs/best.pth \
    --input /path/to/data \
    --output /path/to/predictions
```

### Preprocessing

```bash
python main.py --mode preprocess \
    --config configs/default.yaml \
    --input /path/to/dicom \
    --output /path/to/nifti
```

### Analysis

```bash
python main.py --mode analysis \
    --config configs/default.yaml \
    --input /path/to/predictions \
    --output /path/to/analysis \
    --suv-analysis \
    --tmtv-analysis \
    --histogram \
    --generate-report
```

## Configuration

See `configs/default.yaml` for all available options.

## Supported Architectures

### Backbones
- **SwinUNETR**: Swin Transformer-based UNet
- **UNet3D**: Classic 3D UNet
- **DualEncoder**: Separate encoders for each modality

### Fusion Strategies
- **Early Fusion**: Concatenate inputs
- **Late Fusion**: Combine encoder features
- **Attention Fusion**: Learned attention weights
- **Cross-Attention**: Bidirectional attention between modalities

### Loss Functions
- Dice Loss
- Cross-Entropy Loss
- Dice + CE (combined)
- Focal Loss
- Tversky Loss